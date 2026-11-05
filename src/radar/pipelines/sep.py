"""SEP calculation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace

from radar.core.log import LogLevel
from radar.core.products import (
    SpectrumProduct,
    product_kinds_for_assessment_mode,
    select_products_for_assessment_mode,
)
from radar.core.profiles import (
    profile_uses_ost_134_1044_2007,
    validate_source_model_metadata_for_profile,
)
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.spectra import Spectrum1D
from radar.core.source_products import validate_products_match_spectra_and_source
from radar.core.types import (
    AssessmentMode,
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.output_tables import OutputTable
from radar.sep.model import (
    SepModelInput,
    SepModelProtocol,
    SepModelResult,
    validate_sep_energy_spectrum,
)
from radar.sep.output_tables import (
    sep_on_orbit_product_output_tables,
    sep_shielding_let_output_tables,
)
from radar.pipelines.sep_source_products import (
    SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
    SEP_GEOMAGNETIC_PENETRATION_DOCUMENT,
    SEP_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
    apply_ost_geomagnetic_penetration_to_sep_products,
)
from radar.sep.pipeline_shielding import (
    SEP_SHIELDING_LET_PRODUCT_KIND_ORDER,
    SepShieldingLetPipelineProducts,
    calculate_sep_shielding_let_products,
)
from radar.shielding.resources import load_normative_shielding_tables

SEP_PIPELINE_COMPONENT = "sep_pipeline"
SEP_MODEL_COMPONENT = "sep_model"
SEP_SHIELDING_COMPONENT = "sep_shielding"
SEP_LET_COMPONENT = "sep_let"
SEP_OUTPUT_TABLES_COMPONENT = "sep_output_tables"

UNVERSIONED_MODEL = "unversioned"
SEP_SHIELDING_MODEL = "sep_al_shielding"
SEP_SHIELDING_MODEL_VERSION = "al_spherical_csda_secondary_v1"
SEP_SHIELDING_DOCUMENT = "RADAR shielding normative tables"
SEP_LET_MODEL = "sep_si_let"
SEP_LET_MODEL_VERSION = "si_let_histogram_v1"
SEP_LET_DOCUMENT = "RADAR silicon LET normative tables"

_SEP_LET_QUANTITIES = (
    SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
)


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


def _merge_incomplete_reasons(
    *groups: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(reason for group in groups for reason in group))


def _selected_shielding_stage_incomplete_reasons(
    results: tuple[SepShieldingLetPipelineProducts, ...],
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


def _validate_sep_pipeline_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.quantity in _SEP_LET_QUANTITIES:
        if spectrum.source is not RadiationSource.SEP:
            msg = "SEP pipeline LET spectrum source must be SEP."
            raise ValueError(msg)

        if spectrum.particle not in (Particle.PROTON, Particle.HZE):
            msg = "SEP pipeline LET spectrum must describe protons or HZE particles."
            raise ValueError(msg)

        return

    validate_sep_energy_spectrum(spectrum)


def _table_id_float_token(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "_")


def _prefixed_output_tables(
    *,
    prefix: str,
    tables: tuple[OutputTable, ...],
) -> tuple[OutputTable, ...]:
    return tuple(
        replace(table, table_id=f"{prefix}_{table.table_id}")
        for table in tables
    )


def _shielding_table_prefix(
    *,
    thickness_g_cm2: float,
    product_kind: RadiationProductKind,
) -> str:
    return (
        f"sep_t{_table_id_float_token(thickness_g_cm2)}_"
        f"{product_kind.value}"
    )


def _set_output_tables(
    calculation_result: CalculationResult,
    tables: tuple[OutputTable, ...],
) -> CalculationResult:
    result = calculation_result

    for table in tables:
        result = result.set_output_table(table)

    return result


def _sep_pipeline_output_tables(
    *,
    on_orbit_products: tuple[SpectrumProduct, ...],
    shielding_let_by_thickness: tuple[SepShieldingLetPipelineProducts, ...],
    assessment_mode: AssessmentMode,
    include_let_diagnostics: bool,
) -> tuple[OutputTable, ...]:
    allowed_kinds = product_kinds_for_assessment_mode(assessment_mode)
    selected_on_orbit = select_products_for_assessment_mode(
        on_orbit_products,
        assessment_mode,
        allow_untyped_legacy_diagnostics=True,
    )
    tables: list[OutputTable] = list(
        sep_on_orbit_product_output_tables(selected_on_orbit),
    )

    for shielding_result in shielding_let_by_thickness:
        for product_kind in SEP_SHIELDING_LET_PRODUCT_KIND_ORDER:
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
                    tables=sep_shielding_let_output_tables(
                        proton=proton,
                        hze_by_z=hze_by_z,
                        let_products=let_result,
                        include_proton_components=False,
                        include_let_diagnostics=include_let_diagnostics,
                    ),
                )
            )

    return tuple(tables)


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


def _validate_pipeline_spectra_derived_from_model_result(
    *,
    spectra: tuple[Spectrum1D, ...],
    model_spectra: tuple[Spectrum1D, ...],
) -> None:
    if len(spectra) < len(model_spectra):
        msg = "SEP pipeline spectra must match or derive from SEP model result spectra."
        raise ValueError(msg)

    for spectrum, model_spectrum in zip(
        spectra[: len(model_spectra)],
        model_spectra,
        strict=True,
    ):
        if spectrum == model_spectrum:
            continue

        if (
            _same_spectrum_domain_and_semantics(spectrum, model_spectrum)
            and _structured_chain_derives_from(spectrum, model_spectrum)
        ):
            continue

        msg = "SEP pipeline spectra must match or derive from SEP model result spectra."
        raise ValueError(msg)

    for spectrum in spectra[len(model_spectra) :]:
        _validate_sep_pipeline_spectrum(spectrum)

        if any(
            _structured_chain_derives_from(spectrum, model_spectrum)
            for model_spectrum in model_spectra
        ):
            continue

        msg = "SEP pipeline spectra must match or derive from SEP model result spectra."
        raise ValueError(msg)


def _validate_pipeline_products_derived_from_model_products(
    *,
    products: tuple[SpectrumProduct, ...],
    model_products: tuple[SpectrumProduct, ...],
) -> None:
    if products == model_products:
        return

    if len(products) != len(model_products):
        msg = "SEP pipeline on-orbit products must match or derive from SEP model products."
        raise ValueError(msg)

    for product, model_product in zip(products, model_products, strict=True):
        if product.kind is not model_product.kind:
            msg = "SEP pipeline on-orbit products must preserve product kind."
            raise ValueError(msg)

        if not _same_spectrum_domain_and_semantics(
            product.spectrum,
            model_product.spectrum,
        ):
            msg = (
                "SEP pipeline on-orbit products must preserve spectrum domain "
                "and semantics from SEP model products."
            )
            raise ValueError(msg)

        if _structured_chain_derives_from(
            product.spectrum,
            model_product.spectrum,
        ):
            continue

        msg = (
            "SEP pipeline on-orbit derived spectra must preserve their structured "
            "source transformation chain."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class SepPipelineResult:
    """Result of SEP pipeline calculation."""

    calculation_result: CalculationResult
    sep_model_result: SepModelResult
    spectra: tuple[Spectrum1D, ...]
    products: tuple[SpectrumProduct, ...] = ()
    on_orbit_products: tuple[SpectrumProduct, ...] = ()
    shielding_let_by_thickness: tuple[SepShieldingLetPipelineProducts, ...] = ()
    shielded_products: tuple[SpectrumProduct, ...] = ()
    let_products: tuple[SpectrumProduct, ...] = ()
    assessment_products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "SEP pipeline result must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            _validate_sep_pipeline_spectrum(spectrum)

        _validate_pipeline_spectra_derived_from_model_result(
            spectra=self.spectra,
            model_spectra=self.sep_model_result.spectra,
        )

        on_orbit_products = self.on_orbit_products or self.sep_model_result.products
        products = self.products or on_orbit_products

        _validate_pipeline_products_derived_from_model_products(
            products=on_orbit_products,
            model_products=self.sep_model_result.products,
        )

        if any(
            product.spectrum.axis_kind is not None
            for product in on_orbit_products
        ):
            for product in on_orbit_products:
                product.validate_calculation_contract()

        if not products:
            msg = "SEP pipeline result must contain at least one radiation product."
            raise ValueError(msg)

        assessment_products = self.assessment_products or select_products_for_assessment_mode(
            products,
            self.calculation_result.config.assessment_mode,
            allow_untyped_legacy_diagnostics=True,
        )
        if not assessment_products:
            msg = (
                "SEP pipeline produced no products for assessment mode "
                f"{self.calculation_result.config.assessment_mode.value}."
            )
            raise ValueError(msg)

        validate_products_match_spectra_and_source(
            products=products,
            spectra=self.spectra,
            source=RadiationSource.SEP,
            mismatch_message="SEP pipeline product spectra must match pipeline spectra.",
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
            msg = "SEP pipeline is incomplete: " + "; ".join(
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
        """Return SEP source products before shielding."""

        return self.sep_model_result.products


def calculate_sep_pipeline(
    config: CalculationConfig,
    sep_model: SepModelProtocol,
) -> SepPipelineResult:
    """Calculate SEP spectra and register calculation metadata."""

    calculation_result = CalculationResult(config=config)

    calculation_result = calculation_result.set_component_status(
        component=SEP_PIPELINE_COMPONENT,
        status=ComponentStatus.NOT_STARTED,
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_PIPELINE_COMPONENT,
        message="SEP pipeline started.",
        details={
            "launch_year": str(config.mission.launch_year),
            "lifetime_years": str(config.mission.lifetime_years),
            "sep_exceedance_probability": str(config.mission.sep_exceedance_probability),
        },
    )

    validate_source_model_metadata_for_profile(
        metadata=sep_model.metadata,
        profile=config.methodology.profile,
    )

    sep_model_input = SepModelInput(mission=config.mission)
    sep_model_result = sep_model.calculate(sep_model_input)
    selected_source_products = select_products_for_assessment_mode(
        sep_model_result.products,
        config.assessment_mode,
    )
    source_incomplete_reasons = _incomplete_reasons_from_products(
        selected_source_products,
    )

    calculation_result = calculation_result.set_component_status(
        component=SEP_MODEL_COMPONENT,
        status=(
            ComponentStatus.INCOMPLETE
            if source_incomplete_reasons
            else ComponentStatus.COMPLETED
        ),
    )

    calculation_result = calculation_result.set_model_info(
        ModelInfo(
            name=sep_model_result.model,
            version=sep_model.metadata.version or UNVERSIONED_MODEL,
            status="calculated",
            source=sep_model_result.document,
        )
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_MODEL_COMPONENT,
        message="SEP spectra calculated.",
        details={
            "model": sep_model_result.model,
            "document": sep_model_result.document,
            "lifetime_years": str(sep_model_result.lifetime_years),
            "exceedance_probability": str(sep_model_result.exceedance_probability),
            "spectrum_count": str(len(sep_model_result.spectra)),
        },
    )
    if source_incomplete_reasons:
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.WARNING,
            stage=SEP_MODEL_COMPONENT,
            message="SEP source spectra are incomplete.",
            details={"reasons": "; ".join(source_incomplete_reasons)},
        )

    source_products = sep_model_result.products
    products = source_products
    on_orbit_products = source_products
    shielding_let_by_thickness: tuple[SepShieldingLetPipelineProducts, ...] = ()
    shielded_products: tuple[SpectrumProduct, ...] = ()
    let_products: tuple[SpectrumProduct, ...] = ()
    on_orbit_incomplete_reasons = source_incomplete_reasons
    shielding_incomplete_reasons: tuple[str, ...] = ()
    let_incomplete_reasons: tuple[str, ...] = ()

    if profile_uses_ost_134_1044_2007(config.methodology.profile):
        calculation_result = calculation_result.set_component_status(
            component=SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
            message="SEP geomagnetic penetration started.",
            details={
                "kp": str(config.kp),
                "products": str(len(products)),
            },
        )

        penetration_output = apply_ost_geomagnetic_penetration_to_sep_products(
            config=config,
            products=products,
        )
        products = penetration_output.products
        on_orbit_products = products
        selected_on_orbit_products = select_products_for_assessment_mode(
            on_orbit_products,
            config.assessment_mode,
        )
        on_orbit_incomplete_reasons = _merge_incomplete_reasons(
            source_incomplete_reasons,
            _incomplete_reasons_from_products(selected_on_orbit_products),
        )

        calculation_result = calculation_result.set_model_info(
            ModelInfo(
                name=penetration_output.penetration.model,
                version=SEP_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
                status="diagnostic" if on_orbit_incomplete_reasons else "calculated",
                source=SEP_GEOMAGNETIC_PENETRATION_DOCUMENT,
            )
        )
        calculation_result = calculation_result.set_component_status(
            component=SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
            status=(
                ComponentStatus.INCOMPLETE
                if on_orbit_incomplete_reasons
                else ComponentStatus.COMPLETED
            ),
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.WARNING if on_orbit_incomplete_reasons else LogLevel.INFO,
            stage=SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
            message=(
                "SEP geomagnetic penetration calculated with unresolved qualification."
                if on_orbit_incomplete_reasons
                else "SEP geomagnetic penetration completed."
            ),
            details={
                "products": str(len(products)),
                "penetration_model": penetration_output.penetration.model,
                "penetration_version": SEP_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
                "rigidity_grid_points": str(
                    len(penetration_output.rigidity_grid.values_gv)
                ),
                "qualification_reasons": "; ".join(on_orbit_incomplete_reasons),
            },
        )

        shielding_input_products = select_products_for_assessment_mode(
            on_orbit_products,
            config.assessment_mode,
        )
        proton_products = tuple(
            product
            for product in shielding_input_products
            if product.spectrum.particle is Particle.PROTON
        )
        hze_products = tuple(
            product
            for product in shielding_input_products
            if product.spectrum.particle is Particle.HZE
        )

        calculation_result = calculation_result.set_component_status(
            component=SEP_SHIELDING_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.set_component_status(
            component=SEP_LET_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=SEP_SHIELDING_COMPONENT,
            message="SEP shielding and LET calculation started.",
            details={
                "thicknesses_g_cm2": ",".join(
                    f"{value:g}" for value in config.shielding.thicknesses_g_cm2
                ),
                "input_proton_products": str(len(proton_products)),
                "input_hze_products": str(len(hze_products)),
            },
        )

        tables = load_normative_shielding_tables()
        shielding_let_by_thickness = calculate_sep_shielding_let_products(
            products=shielding_input_products,
            tables=tables,
            thicknesses_g_cm2=config.shielding.thicknesses_g_cm2,
            include_mixed_let_diagnostic=config.methodology.include_mixed_let_diagnostic,
        )
        shielded_products = tuple(
            product
            for result in shielding_let_by_thickness
            for product in result.shielded_products
        )
        let_products = tuple(
            product
            for result in shielding_let_by_thickness
            for product in result.let_products
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
                stage=SEP_SHIELDING_COMPONENT,
                message="SEP shielding/LET products are incomplete.",
                details={
                    "reasons": "; ".join(combined_shielding_let_reasons),
                },
            )

        calculation_result = calculation_result.set_model_info(
            ModelInfo(
                name=SEP_SHIELDING_MODEL,
                version=SEP_SHIELDING_MODEL_VERSION,
                status="calculated",
                source=SEP_SHIELDING_DOCUMENT,
            )
        )
        calculation_result = calculation_result.set_model_info(
            ModelInfo(
                name=SEP_LET_MODEL,
                version=SEP_LET_MODEL_VERSION,
                status="calculated",
                source=SEP_LET_DOCUMENT,
            )
        )
        calculation_result = calculation_result.set_component_status(
            component=SEP_SHIELDING_COMPONENT,
            status=(
                ComponentStatus.INCOMPLETE
                if shielding_incomplete_reasons
                else ComponentStatus.COMPLETED
            ),
        )
        calculation_result = calculation_result.set_component_status(
            component=SEP_LET_COMPONENT,
            status=(
                ComponentStatus.INCOMPLETE
                if let_incomplete_reasons
                else ComponentStatus.COMPLETED
            ),
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=SEP_SHIELDING_COMPONENT,
            message="SEP shielding and LET calculation completed.",
            details={
                "shielded_products": str(len(shielded_products)),
                "let_products": str(len(let_products)),
                "thickness_count": str(len(shielding_let_by_thickness)),
            },
        )

    output_tables = _sep_pipeline_output_tables(
        on_orbit_products=on_orbit_products,
        shielding_let_by_thickness=shielding_let_by_thickness,
        assessment_mode=config.assessment_mode,
        include_let_diagnostics=config.methodology.include_mixed_let_diagnostic,
    )

    if output_tables:
        calculation_result = calculation_result.set_component_status(
            component=SEP_OUTPUT_TABLES_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=SEP_OUTPUT_TABLES_COMPONENT,
            message="SEP output tables started.",
            details={
                "table_count": str(len(output_tables)),
            },
        )
        calculation_result = _set_output_tables(
            calculation_result=calculation_result,
            tables=output_tables,
        )
        calculation_result = calculation_result.set_component_status(
            component=SEP_OUTPUT_TABLES_COMPONENT,
            status=ComponentStatus.COMPLETED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=SEP_OUTPUT_TABLES_COMPONENT,
            message="SEP output tables completed.",
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
        component=SEP_PIPELINE_COMPONENT,
        status=(
            ComponentStatus.INCOMPLETE
            if pipeline_incomplete_reasons
            else ComponentStatus.COMPLETED
        ),
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_PIPELINE_COMPONENT,
        message="SEP pipeline completed.",
    )

    spectra = tuple(product.spectrum for product in products)

    return SepPipelineResult(
        calculation_result=calculation_result,
        sep_model_result=sep_model_result,
        spectra=spectra,
        products=products,
        on_orbit_products=on_orbit_products,
        shielding_let_by_thickness=shielding_let_by_thickness,
        shielded_products=shielded_products,
        let_products=let_products,
    )


__all__ = [
    "SEP_GEOMAGNETIC_PENETRATION_COMPONENT",
    "SEP_GEOMAGNETIC_PENETRATION_MODEL_VERSION",
    "SEP_LET_COMPONENT",
    "SEP_LET_MODEL",
    "SEP_LET_MODEL_VERSION",
    "SEP_MODEL_COMPONENT",
    "SEP_OUTPUT_TABLES_COMPONENT",
    "SEP_PIPELINE_COMPONENT",
    "SEP_SHIELDING_COMPONENT",
    "SEP_SHIELDING_MODEL",
    "SEP_SHIELDING_MODEL_VERSION",
    "SepPipelineResult",
    "calculate_sep_pipeline",
]
