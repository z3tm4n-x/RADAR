"""GCR calculation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

from radar.core.log import LogLevel
from radar.core.products import (
    SpectrumProduct,
    select_products_for_assessment_mode,
)
from radar.core.profiles import (
    SourceModelFamily,
    profile_uses_ost_134_1044_2007,
    validate_source_model_metadata_for_profile,
)
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.gcr.output_tables import (
    gcr_on_orbit_product_output_tables,
    gcr_shielding_let_output_tables,
)
from radar.core.spectra import Spectrum1D
from radar.core.spectrum_ops import mark_spectrum_incomplete
from radar.core.source_products import validate_products_match_spectra_and_source
from radar.core.types import (
    AssessmentMode,
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.gcr.model import (
    GCR_SOURCE_DOMAIN_REASON_PREFIX as _GCR_SOURCE_DOMAIN_REASON_PREFIX,
    GcrModelInput,
    GcrModelProtocol,
    GcrModelResult,
    gcr_source_energy_domain_reasons,
    validate_gcr_energy_spectrum,
)
from radar.gcr.penetration import apply_ost_geomagnetic_penetration_to_gcr_products
from radar.geomagnetic.ost_penetration import OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION
from radar.gcr.pipeline_shielding import (
    GCR_SHIELDING_LET_PRODUCT_KIND_ORDER,
    GcrShieldingLetPipelineProducts,
    calculate_gcr_shielding_let_products,
)
from radar.gcr.source_spectra import GcrSourceSpectra
from radar.output_tables import OutputTable
from radar.shielding.resources import load_normative_shielding_tables
from radar.solar_activity.model import build_calculation_solar_activity

GCR_PIPELINE_COMPONENT = "gcr_pipeline"
GCR_MODEL_COMPONENT = "gcr_model"
GCR_GEOMAGNETIC_PENETRATION_COMPONENT = "gcr_geomagnetic_penetration"
GCR_SHIELDING_COMPONENT = "gcr_shielding"
GCR_LET_COMPONENT = "gcr_let"
GCR_OUTPUT_TABLES_COMPONENT = "gcr_output_tables"
GCR_GEOMAGNETIC_PENETRATION_MODEL_VERSION = OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION
GCR_GEOMAGNETIC_PENETRATION_DOCUMENT = "OST 134-1044-2007 Appendix Zh"
GCR_SHIELDING_MODEL = "gcr_al_shielding"
GCR_SHIELDING_MODEL_VERSION = "al_spherical_csda_secondary_v1"
GCR_SHIELDING_DOCUMENT = "RADAR shielding normative tables"
GCR_LET_MODEL = "gcr_si_let"
GCR_LET_MODEL_VERSION = "si_let_histogram_v1"
GCR_LET_DOCUMENT = "RADAR silicon LET normative tables"
GCR_SOURCE_DOMAIN_REASON_PREFIX = _GCR_SOURCE_DOMAIN_REASON_PREFIX


def _incomplete_reasons_from_products(
    products: tuple[SpectrumProduct, ...],
) -> tuple[str, ...]:
    """Collect the explicit release quarantine carried by product spectra."""

    return tuple(
        dict.fromkeys(
            reason
            for product in products
            for reason in product.spectrum.incomplete_reasons
        )
    )


def _incomplete_reasons_from_source_spectra(
    source_spectra: GcrSourceSpectra | None,
) -> tuple[str, ...]:
    if source_spectra is None:
        return ()
    return tuple(
        dict.fromkeys(
            reason
            for ion in source_spectra.ions
            for spectrum in (ion.main, ion.nek, ion.akl, ion.total)
            for reason in spectrum.incomplete_reasons
        )
    )


def _merge_incomplete_reasons(
    *groups: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(reason for group in groups for reason in group))


def _selected_shielding_stage_incomplete_reasons(
    results: tuple[GcrShieldingLetPipelineProducts, ...],
    assessment_mode: AssessmentMode,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            reason
            for result in results
            if select_products_for_assessment_mode(result.products, assessment_mode)
            for reason in result.incomplete_reasons
        )
    )


def _apply_normative_source_domain_contract(
    result: GcrModelResult,
    *,
    model_family: SourceModelFamily,
) -> tuple[GcrModelResult, tuple[str, ...]]:
    """Quarantine a normative GCR result whose production tail is truncated."""

    if model_family not in {
        SourceModelFamily.OST_134_1044_2007,
        SourceModelFamily.GOST_GCR,
    }:
        return result, ()

    reasons_by_index: dict[int, list[str]] = {}
    for index, product in enumerate(result.products):
        reasons = list(gcr_source_energy_domain_reasons(product.spectrum))
        if reasons:
            reasons_by_index[index] = reasons

    if not reasons_by_index:
        return result, ()

    products = tuple(
        replace(
            product,
            spectrum=mark_spectrum_incomplete(
                product.spectrum,
                *reasons_by_index[index],
            ),
        )
        if index in reasons_by_index
        else product
        for index, product in enumerate(result.products)
    )
    updated = replace(
        result,
        spectra=tuple(product.spectrum for product in products),
        products=products,
    )
    return updated, tuple(
        dict.fromkeys(
            reason
            for product_reasons in reasons_by_index.values()
            for reason in product_reasons
        )
    )


def _same_spectrum_domain_and_semantics(
    spectrum: Spectrum1D,
    model_spectrum: Spectrum1D,
) -> bool:
    return (
        spectrum.x == model_spectrum.x
        and spectrum.x_unit is model_spectrum.x_unit
        and spectrum.y_unit is model_spectrum.y_unit
        and spectrum.quantity is model_spectrum.quantity
        and spectrum.particle is model_spectrum.particle
        and spectrum.source is model_spectrum.source
        and spectrum.axis_kind is model_spectrum.axis_kind
        and spectrum.angular_basis is model_spectrum.angular_basis
        and spectrum.temporal_basis is model_spectrum.temporal_basis
        and spectrum.species == model_spectrum.species
        and spectrum.assessment_mode is model_spectrum.assessment_mode
        and spectrum.mission_duration_s == model_spectrum.mission_duration_s
    )


def _structured_chain_derives_from(
    spectrum: Spectrum1D,
    source_spectrum: Spectrum1D,
) -> bool:
    source_chain = source_spectrum.transformation_chain
    if not source_chain:
        return bool(spectrum.transformation_chain)
    return spectrum.transformation_chain[: len(source_chain)] == source_chain


_GCR_LET_QUANTITIES = (
    SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
)


def _validate_gcr_pipeline_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.quantity in _GCR_LET_QUANTITIES:
        if spectrum.source is not RadiationSource.GCR:
            msg = "GCR pipeline LET spectrum source must be GCR."
            raise ValueError(msg)

        if spectrum.particle not in (Particle.PROTON, Particle.HZE):
            msg = "GCR pipeline LET spectrum must describe protons or HZE particles."
            raise ValueError(msg)

        return

    validate_gcr_energy_spectrum(spectrum)


def _calculate_source_spectra_if_supported(
    gcr_model: GcrModelProtocol,
    model_input: GcrModelInput,
) -> GcrSourceSpectra | None:
    method = getattr(gcr_model, "calculate_source_spectra", None)

    if not callable(method):
        return None

    return cast("Callable[[GcrModelInput], GcrSourceSpectra]", method)(model_input)


def _table_id_float_token(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "_")


def _prefixed_output_tables(
    *,
    prefix: str,
    tables: tuple[OutputTable, ...],
) -> tuple[OutputTable, ...]:
    return tuple(replace(table, table_id=f"{prefix}_{table.table_id}") for table in tables)


def _set_output_tables(
    calculation_result: CalculationResult,
    tables: tuple[OutputTable, ...],
) -> CalculationResult:
    result = calculation_result

    for table in tables:
        result = result.set_output_table(table)

    return result


def _shielding_table_prefix(
    *,
    thickness_g_cm2: float,
    product_kind: RadiationProductKind,
) -> str:
    return f"gcr_t{_table_id_float_token(thickness_g_cm2)}_{product_kind.value}"


def _gcr_pipeline_output_tables(
    *,
    source_spectra: GcrSourceSpectra | None,
    on_orbit_products: tuple[SpectrumProduct, ...],
    shielding_let_by_thickness: tuple[GcrShieldingLetPipelineProducts, ...],
    assessment_mode: AssessmentMode,
    include_let_diagnostics: bool,
) -> tuple[OutputTable, ...]:
    if source_spectra is None:
        return ()
    selected_on_orbit = select_products_for_assessment_mode(
        on_orbit_products,
        assessment_mode,
        allow_untyped_legacy_diagnostics=True,
    )
    allowed_kinds = {product.kind for product in selected_on_orbit}
    if not selected_on_orbit:
        return ()
    tables: list[OutputTable] = list(
        gcr_on_orbit_product_output_tables(selected_on_orbit)
    )

    for shielding_result in shielding_let_by_thickness:
        for product_kind in GCR_SHIELDING_LET_PRODUCT_KIND_ORDER:
            if product_kind not in allowed_kinds:
                continue
            proton = shielding_result.proton_by_kind.get(product_kind)
            hze_by_z = shielding_result.hze_by_kind.get(product_kind, {})
            let_result = shielding_result.let_results_by_kind.get(product_kind)

            if proton is None or let_result is None or not hze_by_z:
                continue

            prefix = _shielding_table_prefix(
                thickness_g_cm2=shielding_result.thickness_g_cm2,
                product_kind=product_kind,
            )
            tables.extend(
                _prefixed_output_tables(
                    prefix=prefix,
                    tables=gcr_shielding_let_output_tables(
                        proton=proton,
                        hze_by_z=hze_by_z,
                        let_products=let_result,
                        include_proton_components=False,
                        include_let_diagnostics=include_let_diagnostics,
                    ),
                )
            )

    return tuple(tables)


def _validate_pipeline_spectra_derived_from_model_result(
    *,
    spectra: tuple[Spectrum1D, ...],
    model_spectra: tuple[Spectrum1D, ...],
) -> None:
    if spectra == model_spectra:
        return

    if len(spectra) != len(model_spectra):
        msg = "GCR pipeline spectra must match or derive from GCR model result spectra."
        raise ValueError(msg)

    for spectrum, model_spectrum in zip(spectra, model_spectra):
        if not _same_spectrum_domain_and_semantics(spectrum, model_spectrum):
            msg = (
                "GCR pipeline spectra must preserve spectrum domain and semantics "
                "from GCR model result spectra."
            )
            raise ValueError(msg)

        if not _structured_chain_derives_from(spectrum, model_spectrum):
            msg = (
                "GCR pipeline derived spectra must identify their source model "
                "in the structured transformation chain."
            )
            raise ValueError(msg)


def _validate_pipeline_products_derived_from_model_products(
    *,
    products: tuple[SpectrumProduct, ...],
    model_products: tuple[SpectrumProduct, ...],
) -> None:
    if products == model_products:
        return

    if len(products) != len(model_products):
        msg = "GCR pipeline products must match or derive from GCR model result products."
        raise ValueError(msg)

    for product, model_product in zip(products, model_products):
        if product.kind is not model_product.kind:
            msg = "GCR pipeline products must preserve product kind."
            raise ValueError(msg)

        if not _same_spectrum_domain_and_semantics(
            product.spectrum,
            model_product.spectrum,
        ):
            msg = (
                "GCR pipeline products must preserve spectrum domain and semantics "
                "from GCR model result products."
            )
            raise ValueError(msg)

        if not any(
            candidate.kind is product.kind
            and candidate.spectrum.species == product.spectrum.species
            and _structured_chain_derives_from(product.spectrum, candidate.spectrum)
            for candidate in model_products
        ):
            msg = (
                "GCR pipeline derived product spectra must identify their source model "
                "in the structured transformation chain."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class GcrPipelineResult:
    """Result of GCR pipeline calculation."""

    calculation_result: CalculationResult
    gcr_model_result: GcrModelResult
    spectra: tuple[Spectrum1D, ...]
    source_spectra: GcrSourceSpectra | None = None
    products: tuple[SpectrumProduct, ...] = ()
    on_orbit_products: tuple[SpectrumProduct, ...] = ()
    shielding_let_by_thickness: tuple[GcrShieldingLetPipelineProducts, ...] = ()
    shielded_products: tuple[SpectrumProduct, ...] = ()
    let_products: tuple[SpectrumProduct, ...] = ()
    assessment_products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "GCR pipeline result must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            _validate_gcr_pipeline_spectrum(spectrum)

        products = self.products or self.gcr_model_result.products
        on_orbit_products = self.on_orbit_products or self.gcr_model_result.products

        if not products:
            msg = "GCR pipeline result must contain at least one radiation product."
            raise ValueError(msg)

        if any(product.spectrum.axis_kind is not None for product in on_orbit_products):
            for product in on_orbit_products:
                product.validate_calculation_contract()

        assessment_products = self.assessment_products or select_products_for_assessment_mode(
            products,
            self.calculation_result.config.assessment_mode,
            allow_untyped_legacy_diagnostics=True,
        )
        if not assessment_products:
            msg = (
                "GCR pipeline produced no products for assessment mode "
                f"{self.calculation_result.config.assessment_mode.value}."
            )
            raise ValueError(msg)

        validate_products_match_spectra_and_source(
            products=products,
            spectra=self.spectra,
            source=RadiationSource.GCR,
            mismatch_message="GCR pipeline product spectra must match pipeline spectra.",
        )

        _validate_pipeline_products_derived_from_model_products(
            products=on_orbit_products,
            model_products=self.gcr_model_result.products,
        )

        object.__setattr__(self, "products", products)
        object.__setattr__(self, "on_orbit_products", on_orbit_products)
        object.__setattr__(self, "assessment_products", assessment_products)

    def production_release_products(self) -> tuple[SpectrumProduct, ...]:
        """Return selected products only after strict production validation."""

        assessment_mode = self.calculation_result.config.assessment_mode
        selected_source_products = select_products_for_assessment_mode(
            self.source_products,
            assessment_mode,
        )
        selected_on_orbit_products = select_products_for_assessment_mode(
            self.on_orbit_products,
            assessment_mode,
        )
        selected_shielded_products = select_products_for_assessment_mode(
            self.shielded_products,
            assessment_mode,
        )
        selected_let_products = select_products_for_assessment_mode(
            self.let_products,
            assessment_mode,
        )
        incomplete_reasons = _merge_incomplete_reasons(
            _incomplete_reasons_from_source_spectra(self.source_spectra),
            _incomplete_reasons_from_products(selected_source_products),
            _incomplete_reasons_from_products(selected_on_orbit_products),
            _selected_shielding_stage_incomplete_reasons(
                self.shielding_let_by_thickness,
                assessment_mode,
            ),
            _incomplete_reasons_from_products(selected_shielded_products),
            _incomplete_reasons_from_products(selected_let_products),
        )
        if incomplete_reasons:
            msg = "GCR pipeline is incomplete: " + "; ".join(
                dict.fromkeys(incomplete_reasons)
            )
            raise ValueError(msg)

        return select_products_for_assessment_mode(
            self.products,
            assessment_mode,
            validate_for_release=True,
        )

    @property
    def source_products(self) -> tuple[SpectrumProduct, ...]:
        """Raw GCR products before geomagnetic penetration."""

        return self.gcr_model_result.products


def calculate_gcr_pipeline(
    config: CalculationConfig,
    gcr_model: GcrModelProtocol,
) -> GcrPipelineResult:
    """Calculate GCR spectra and register calculation metadata."""

    calculation_result = CalculationResult(config=config)

    calculation_result = calculation_result.set_component_status(
        component=GCR_PIPELINE_COMPONENT,
        status=ComponentStatus.NOT_STARTED,
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=GCR_PIPELINE_COMPONENT,
        message="GCR pipeline started.",
        details={
            "launch_year": str(config.mission.launch_year),
            "lifetime_years": str(config.mission.lifetime_years),
        },
    )

    validate_source_model_metadata_for_profile(
        metadata=gcr_model.metadata,
        profile=config.methodology.profile,
    )

    solar_activity = build_calculation_solar_activity(config)
    gcr_model_input = GcrModelInput(
        mission=config.mission,
        solar_activity=solar_activity,
    )
    gcr_model_result = gcr_model.calculate(gcr_model_input)
    gcr_model_result, _ = _apply_normative_source_domain_contract(
        gcr_model_result,
        model_family=gcr_model.metadata.model_family,
    )
    source_spectra = _calculate_source_spectra_if_supported(
        gcr_model,
        gcr_model_input,
    )
    products = gcr_model_result.products
    common_source_incomplete_reasons = _incomplete_reasons_from_source_spectra(
        source_spectra,
    )
    selected_source_products = select_products_for_assessment_mode(
        products,
        config.assessment_mode,
    )
    source_incomplete_reasons = _merge_incomplete_reasons(
        common_source_incomplete_reasons,
        _incomplete_reasons_from_products(selected_source_products),
    )

    calculation_result = calculation_result.set_component_status(
        component=GCR_MODEL_COMPONENT,
        status=(
            ComponentStatus.INCOMPLETE
            if source_incomplete_reasons
            else ComponentStatus.COMPLETED
        ),
    )

    calculation_result = calculation_result.set_model_info(
        ModelInfo(
            name=gcr_model_result.model,
            version=gcr_model.metadata.version,
            status="calculated",
            source=gcr_model_result.document,
        )
    )

    model_log_details = {
        "model": gcr_model_result.model,
        "document": gcr_model_result.document,
        "lifetime_years": str(gcr_model_result.lifetime_years),
        "spectrum_count": str(len(gcr_model_result.spectra)),
    }
    model_log_details.update(dict(gcr_model_result.method_metadata))
    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=GCR_MODEL_COMPONENT,
        message="GCR spectra calculated.",
        details=model_log_details,
    )
    if source_incomplete_reasons:
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.WARNING,
            stage=GCR_MODEL_COMPONENT,
            message="GCR source spectra are incomplete.",
            details={
                "reasons": "; ".join(source_incomplete_reasons),
            },
        )

    on_orbit_incomplete_reasons = source_incomplete_reasons
    if profile_uses_ost_134_1044_2007(config.methodology.profile):
        calculation_result = calculation_result.set_component_status(
            component=GCR_GEOMAGNETIC_PENETRATION_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=GCR_GEOMAGNETIC_PENETRATION_COMPONENT,
            message="GCR geomagnetic penetration started.",
            details={
                "kp": str(config.kp),
                "products": str(len(products)),
            },
        )

        penetration_output = apply_ost_geomagnetic_penetration_to_gcr_products(
            config=config,
            products=products,
        )
        products = penetration_output.products
        selected_on_orbit_products = select_products_for_assessment_mode(
            products,
            config.assessment_mode,
        )
        on_orbit_incomplete_reasons = _merge_incomplete_reasons(
            source_incomplete_reasons,
            _incomplete_reasons_from_products(selected_on_orbit_products),
        )

        calculation_result = calculation_result.set_model_info(
            ModelInfo(
                name=penetration_output.penetration.model,
                version=GCR_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
                status="diagnostic" if on_orbit_incomplete_reasons else "calculated",
                source=GCR_GEOMAGNETIC_PENETRATION_DOCUMENT,
            )
        )
        calculation_result = calculation_result.set_component_status(
            component=GCR_GEOMAGNETIC_PENETRATION_COMPONENT,
            status=(
                ComponentStatus.INCOMPLETE
                if on_orbit_incomplete_reasons
                else ComponentStatus.COMPLETED
            ),
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.WARNING if on_orbit_incomplete_reasons else LogLevel.INFO,
            stage=GCR_GEOMAGNETIC_PENETRATION_COMPONENT,
            message=(
                "GCR geomagnetic penetration calculated with unresolved qualification."
                if on_orbit_incomplete_reasons else "GCR geomagnetic penetration completed."
            ),
            details={
                "products": str(len(products)),
                "penetration_model": penetration_output.penetration.model,
                "penetration_version": GCR_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
                "rigidity_grid_points": str(len(penetration_output.rigidity_grid.values_gv)),
                "qualification_reasons": "; ".join(on_orbit_incomplete_reasons),
            },
        )

    on_orbit_products = products
    shielding_let_by_thickness: tuple[GcrShieldingLetPipelineProducts, ...] = ()
    shielded_products: tuple[SpectrumProduct, ...] = ()
    let_products: tuple[SpectrumProduct, ...] = ()
    shielding_incomplete_reasons: tuple[str, ...] = ()
    let_incomplete_reasons: tuple[str, ...] = ()

    if profile_uses_ost_134_1044_2007(config.methodology.profile):
        calculation_result = calculation_result.set_component_status(
            component=GCR_SHIELDING_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.set_component_status(
            component=GCR_LET_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=GCR_SHIELDING_COMPONENT,
            message="GCR shielding and LET calculation started.",
            details={
                "thicknesses_g_cm2": ",".join(
                    f"{value:g}" for value in config.shielding.thicknesses_g_cm2
                ),
                "input_products": str(len(on_orbit_products)),
            },
        )

        tables = load_normative_shielding_tables()
        shielding_input_products = select_products_for_assessment_mode(
            on_orbit_products,
            config.assessment_mode,
        )
        shielding_let_by_thickness = calculate_gcr_shielding_let_products(
            products=shielding_input_products,
            tables=tables,
            thicknesses_g_cm2=config.shielding.thicknesses_g_cm2,
            include_mixed_let_diagnostic=config.methodology.include_mixed_let_diagnostic,
        )
        shielded_products = tuple(
            product for result in shielding_let_by_thickness for product in result.shielded_products
        )
        let_products = tuple(
            product for result in shielding_let_by_thickness for product in result.let_products
        )
        products = (*on_orbit_products, *shielded_products, *let_products)
        selected_shielded_products = select_products_for_assessment_mode(
            shielded_products,
            config.assessment_mode,
        )
        selected_let_products = select_products_for_assessment_mode(
            let_products,
            config.assessment_mode,
        )

        stage_incomplete_reasons = _selected_shielding_stage_incomplete_reasons(
            shielding_let_by_thickness,
            config.assessment_mode,
        )
        shielding_incomplete_reasons = _merge_incomplete_reasons(
            on_orbit_incomplete_reasons,
            stage_incomplete_reasons,
            _incomplete_reasons_from_products(selected_shielded_products),
        )
        let_incomplete_reasons = _merge_incomplete_reasons(
            on_orbit_incomplete_reasons,
            stage_incomplete_reasons,
            _incomplete_reasons_from_products(selected_let_products),
        )
        combined_shielding_let_reasons = _merge_incomplete_reasons(
            shielding_incomplete_reasons,
            let_incomplete_reasons,
        )
        if combined_shielding_let_reasons:
            calculation_result = calculation_result.add_log_entry(
                level=LogLevel.WARNING,
                stage=GCR_SHIELDING_COMPONENT,
                message="GCR shielding/LET products are incomplete.",
                details={
                    "reasons": "; ".join(combined_shielding_let_reasons),
                },
            )

        calculation_result = calculation_result.set_model_info(
            ModelInfo(
                name=GCR_SHIELDING_MODEL,
                version=GCR_SHIELDING_MODEL_VERSION,
                status="calculated",
                source=GCR_SHIELDING_DOCUMENT,
            )
        )
        calculation_result = calculation_result.set_model_info(
            ModelInfo(
                name=GCR_LET_MODEL,
                version=GCR_LET_MODEL_VERSION,
                status="calculated",
                source=GCR_LET_DOCUMENT,
            )
        )
        calculation_result = calculation_result.set_component_status(
            component=GCR_SHIELDING_COMPONENT,
            status=(
                ComponentStatus.INCOMPLETE
                if shielding_incomplete_reasons
                else ComponentStatus.COMPLETED
            ),
        )
        calculation_result = calculation_result.set_component_status(
            component=GCR_LET_COMPONENT,
            status=(
                ComponentStatus.INCOMPLETE
                if let_incomplete_reasons
                else ComponentStatus.COMPLETED
            ),
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=GCR_SHIELDING_COMPONENT,
            message="GCR shielding and LET calculation completed.",
            details={
                "shielded_products": str(len(shielded_products)),
                "let_products": str(len(let_products)),
                "thickness_count": str(len(shielding_let_by_thickness)),
            },
        )

    output_tables = _gcr_pipeline_output_tables(
        source_spectra=source_spectra,
        on_orbit_products=on_orbit_products,
        shielding_let_by_thickness=shielding_let_by_thickness,
        assessment_mode=config.assessment_mode,
        include_let_diagnostics=config.methodology.include_mixed_let_diagnostic,
    )

    if output_tables:
        calculation_result = calculation_result.set_component_status(
            component=GCR_OUTPUT_TABLES_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=GCR_OUTPUT_TABLES_COMPONENT,
            message="GCR output tables started.",
            details={
                "table_count": str(len(output_tables)),
            },
        )
        calculation_result = _set_output_tables(
            calculation_result=calculation_result,
            tables=output_tables,
        )
        calculation_result = calculation_result.set_component_status(
            component=GCR_OUTPUT_TABLES_COMPONENT,
            status=ComponentStatus.COMPLETED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=GCR_OUTPUT_TABLES_COMPONENT,
            message="GCR output tables completed.",
            details={
                "table_count": str(len(output_tables)),
            },
        )

    pipeline_incomplete_reasons = _merge_incomplete_reasons(
        source_incomplete_reasons,
        on_orbit_incomplete_reasons,
        shielding_incomplete_reasons,
        let_incomplete_reasons,
    )
    calculation_result = calculation_result.set_component_status(
        component=GCR_PIPELINE_COMPONENT,
        status=(
            ComponentStatus.INCOMPLETE
            if pipeline_incomplete_reasons
            else ComponentStatus.COMPLETED
        ),
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=GCR_PIPELINE_COMPONENT,
        message="GCR pipeline completed.",
    )

    spectra = tuple(product.spectrum for product in products)

    return GcrPipelineResult(
        calculation_result=calculation_result,
        gcr_model_result=gcr_model_result,
        spectra=spectra,
        source_spectra=source_spectra,
        products=products,
        on_orbit_products=on_orbit_products,
        shielding_let_by_thickness=shielding_let_by_thickness,
        shielded_products=shielded_products,
        let_products=let_products,
    )
