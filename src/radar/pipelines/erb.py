"""ERB calculation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace

from radar.core.log import LogLevel
from radar.core.products import SpectrumProduct, select_products_for_assessment_mode
from radar.core.profiles import (
    profile_uses_ost_134_1044_2007,
    validate_source_model_metadata_for_profile,
)
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.spectra import Spectrum1D
from radar.core.source_products import validate_products_match_spectra_and_source
from radar.core.types import AssessmentMode, Particle, RadiationProductKind, RadiationSource
from radar.erb.model import (
    ErbModelInput,
    ErbModelProtocol,
    ErbModelResult,
    validate_erb_energy_spectrum,
)
from radar.erb.output_tables import (
    erb_electron_shielding_output_tables,
    erb_product_output_tables,
    erb_proton_shielding_output_tables,
)
from radar.erb.shielding import (
    ErbElectronShieldingResult,
    ErbProtonShieldingResult,
    calculate_erb_electron_shielding_spectrum,
    calculate_erb_proton_shielding_spectrum,
)
from radar.output_tables import OutputTable
from radar.shielding.resources import load_normative_shielding_tables
from radar.solar_activity.model import build_calculation_solar_activity

ERB_PIPELINE_COMPONENT = "erb_pipeline"
ERB_MODEL_COMPONENT = "erb_model"
ERB_SHIELDING_COMPONENT = "erb_shielding"
ERB_OUTPUT_TABLES_COMPONENT = "erb_output_tables"
ERB_SHIELDING_MODEL = "erb_al_shielding"
ERB_SHIELDING_MODEL_VERSION = "al_spherical_csda_proton_secondary_electron_primary_v1"
ERB_SHIELDING_DOCUMENT = "RADAR shielding normative tables"


def _incomplete_reasons_from_products(
    products: tuple[SpectrumProduct, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            reason
            for product in products
            for reason in product.spectrum.incomplete_reasons
        )
    )


def _merge_incomplete_reasons(
    *reason_groups: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(reason for group in reason_groups for reason in group))


def _table_id_float_token(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "_")


def _prefixed_output_tables(
    *,
    prefix: str,
    tables: tuple[OutputTable, ...],
) -> tuple[OutputTable, ...]:
    return tuple(replace(table, table_id=f"{prefix}_{table.table_id}") for table in tables)


def _shielding_table_prefix(
    *,
    thickness_g_cm2: float,
    product_kind: RadiationProductKind,
) -> str:
    return f"erb_t{_table_id_float_token(thickness_g_cm2)}_{product_kind.value}"


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
        msg = "ERB pipeline spectra must match or derive from ERB model result spectra."
        raise ValueError(msg)

    if spectra[: len(model_spectra)] != model_spectra:
        msg = "ERB pipeline spectra must match or derive from ERB model result spectra."
        raise ValueError(msg)

    for spectrum in spectra[len(model_spectra) :]:
        validate_erb_energy_spectrum(spectrum)

        if any(
            _structured_chain_derives_from(spectrum, model_spectrum)
            for model_spectrum in model_spectra
        ):
            continue

        msg = "ERB pipeline spectra must match or derive from ERB model result spectra."
        raise ValueError(msg)


@dataclass(frozen=True)
class ErbShieldingPipelineProduct:
    """ERB shielding result associated with one source product."""

    thickness_g_cm2: float
    source_product: SpectrumProduct
    result: ErbProtonShieldingResult | ErbElectronShieldingResult
    shielded_product: SpectrumProduct


def _calculate_erb_shielding_by_thickness(
    *,
    products: tuple[SpectrumProduct, ...],
    thicknesses_g_cm2: tuple[float, ...],
) -> tuple[ErbShieldingPipelineProduct, ...]:
    shielding_products = tuple(
        product
        for product in products
        if product.spectrum.particle in (Particle.PROTON, Particle.ELECTRON)
    )

    if not shielding_products:
        return ()

    tables = load_normative_shielding_tables()
    shielding_results: list[ErbShieldingPipelineProduct] = []

    for thickness_g_cm2 in thicknesses_g_cm2:
        for product in shielding_products:
            if product.spectrum.particle is Particle.PROTON:
                proton_result = calculate_erb_proton_shielding_spectrum(
                    spectrum=product.spectrum,
                    tables=tables,
                    thickness_g_cm2=thickness_g_cm2,
                )
                result: ErbProtonShieldingResult | ErbElectronShieldingResult = proton_result
                shielded_spectrum = proton_result.total
            elif product.spectrum.particle is Particle.ELECTRON:
                electron_result = calculate_erb_electron_shielding_spectrum(
                    spectrum=product.spectrum,
                    tables=tables,
                    thickness_g_cm2=thickness_g_cm2,
                )
                result = electron_result
                shielded_spectrum = electron_result.primary
            else:
                continue

            source_label = product.label or product.kind.value
            shielded_product = SpectrumProduct(
                kind=product.kind,
                spectrum=shielded_spectrum,
                label=(f"{source_label} behind Al shield {thickness_g_cm2:g} g/cm^2"),
            )
            shielding_results.append(
                ErbShieldingPipelineProduct(
                    thickness_g_cm2=thickness_g_cm2,
                    source_product=product,
                    result=result,
                    shielded_product=shielded_product,
                )
            )

    return tuple(shielding_results)


def _erb_pipeline_output_tables(
    *,
    on_orbit_products: tuple[SpectrumProduct, ...],
    shielding_by_thickness: tuple[ErbShieldingPipelineProduct, ...],
    assessment_mode: AssessmentMode,
) -> tuple[OutputTable, ...]:
    selected_on_orbit = select_products_for_assessment_mode(
        on_orbit_products,
        assessment_mode,
        allow_untyped_legacy_diagnostics=True,
    )
    if not selected_on_orbit:
        return ()
    allowed_kinds = {product.kind for product in selected_on_orbit}
    tables: list[OutputTable] = list(erb_product_output_tables(selected_on_orbit))

    for shielding_result in shielding_by_thickness:
        if shielding_result.source_product.kind not in allowed_kinds:
            continue
        prefix = _shielding_table_prefix(
            thickness_g_cm2=shielding_result.thickness_g_cm2,
            product_kind=shielding_result.source_product.kind,
        )
        result = shielding_result.result

        if isinstance(result, ErbProtonShieldingResult):
            shielding_tables = erb_proton_shielding_output_tables(
                result,
                include_components=False,
            )
        elif isinstance(result, ErbElectronShieldingResult):
            shielding_tables = erb_electron_shielding_output_tables(result)
        else:
            continue

        tables.extend(
            _prefixed_output_tables(
                prefix=prefix,
                tables=shielding_tables,
            )
        )

    return tuple(tables)


def _set_output_tables(
    calculation_result: CalculationResult,
    tables: tuple[OutputTable, ...],
) -> CalculationResult:
    result = calculation_result

    for table in tables:
        result = result.set_output_table(table)

    return result


@dataclass(frozen=True)
class ErbPipelineResult:
    """Result of ERB pipeline calculation."""

    calculation_result: CalculationResult
    erb_model_result: ErbModelResult
    spectra: tuple[Spectrum1D, ...]
    products: tuple[SpectrumProduct, ...] = ()
    on_orbit_products: tuple[SpectrumProduct, ...] = ()
    shielding_by_thickness: tuple[ErbShieldingPipelineProduct, ...] = ()
    shielded_products: tuple[SpectrumProduct, ...] = ()
    assessment_products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "ERB pipeline result must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_erb_energy_spectrum(spectrum)

        _validate_pipeline_spectra_derived_from_model_result(
            spectra=self.spectra,
            model_spectra=self.erb_model_result.spectra,
        )

        on_orbit_products = self.on_orbit_products or self.erb_model_result.products
        products = self.products or on_orbit_products

        if on_orbit_products != self.erb_model_result.products:
            msg = "ERB pipeline on-orbit products must match ERB model result products."
            raise ValueError(msg)

        if not products:
            msg = "ERB pipeline result must contain at least one radiation product."
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
                "ERB pipeline produced no products for assessment mode "
                f"{self.calculation_result.config.assessment_mode.value}."
            )
            raise ValueError(msg)

        validate_products_match_spectra_and_source(
            products=products,
            spectra=self.spectra,
            source=RadiationSource.ERB,
            mismatch_message="ERB pipeline product spectra must match pipeline spectra.",
        )

        object.__setattr__(self, "products", products)
        object.__setattr__(self, "on_orbit_products", on_orbit_products)
        object.__setattr__(self, "assessment_products", assessment_products)

    def production_release_products(self) -> tuple[SpectrumProduct, ...]:
        """Return selected products only after strict production validation."""

        assessment_mode = self.calculation_result.config.assessment_mode
        selected_on_orbit_products = select_products_for_assessment_mode(
            self.on_orbit_products,
            assessment_mode,
        )
        selected_shielded_products = select_products_for_assessment_mode(
            self.shielded_products,
            assessment_mode,
        )
        incomplete_reasons = _merge_incomplete_reasons(
            _incomplete_reasons_from_products(selected_on_orbit_products),
            _incomplete_reasons_from_products(selected_shielded_products),
        )
        if incomplete_reasons:
            msg = "ERB pipeline is incomplete: " + "; ".join(incomplete_reasons)
            raise ValueError(msg)

        return select_products_for_assessment_mode(
            self.products,
            assessment_mode,
            validate_for_release=True,
        )


def calculate_erb_pipeline(
    config: CalculationConfig,
    erb_model: ErbModelProtocol,
) -> ErbPipelineResult:
    """Calculate ERB spectra and register calculation metadata."""

    calculation_result = CalculationResult(config=config)

    calculation_result = calculation_result.set_component_status(
        component=ERB_PIPELINE_COMPONENT,
        status=ComponentStatus.NOT_STARTED,
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=ERB_PIPELINE_COMPONENT,
        message="ERB pipeline started.",
        details={
            "launch_year": str(config.mission.launch_year),
            "lifetime_years": str(config.mission.lifetime_years),
            "kp": str(config.kp),
        },
    )

    validate_source_model_metadata_for_profile(
        metadata=erb_model.metadata,
        profile=config.methodology.profile,
    )

    solar_activity = build_calculation_solar_activity(config)
    erb_model_input = ErbModelInput(
        config=config,
        solar_activity=solar_activity,
    )
    erb_model_result = erb_model.calculate(erb_model_input)
    on_orbit_products = erb_model_result.products
    selected_on_orbit_products = select_products_for_assessment_mode(
        on_orbit_products,
        config.assessment_mode,
    )
    source_incomplete_reasons = _incomplete_reasons_from_products(
        selected_on_orbit_products,
    )

    calculation_result = calculation_result.set_component_status(
        component=ERB_MODEL_COMPONENT,
        status=(
            ComponentStatus.INCOMPLETE
            if source_incomplete_reasons
            else ComponentStatus.COMPLETED
        ),
    )

    calculation_result = calculation_result.set_model_info(
        ModelInfo(
            name=erb_model_result.model,
            version=erb_model.metadata.version,
            status="calculated",
            source=erb_model_result.document,
        )
    )

    model_log_details = {
        "model": erb_model_result.model,
        "document": erb_model_result.document,
        "lifetime_years": str(erb_model_result.lifetime_years),
        "kp": str(erb_model_result.kp),
        "spectrum_count": str(len(erb_model_result.spectra)),
    }
    model_log_details.update(dict(erb_model_result.method_metadata))
    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=ERB_MODEL_COMPONENT,
        message="ERB spectra calculated.",
        details=model_log_details,
    )
    if source_incomplete_reasons:
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.WARNING,
            stage=ERB_MODEL_COMPONENT,
            message="ERB source spectra are incomplete.",
            details={"reasons": "; ".join(source_incomplete_reasons)},
        )

    products = on_orbit_products
    shielding_by_thickness: tuple[ErbShieldingPipelineProduct, ...] = ()
    shielded_products: tuple[SpectrumProduct, ...] = ()
    shielding_incomplete_reasons: tuple[str, ...] = ()

    if profile_uses_ost_134_1044_2007(config.methodology.profile):
        shielding_input_products = select_products_for_assessment_mode(
            on_orbit_products,
            config.assessment_mode,
            allow_untyped_legacy_diagnostics=True,
        )
        proton_products = tuple(
            product
            for product in shielding_input_products
            if product.spectrum.particle is Particle.PROTON
        )
        electron_products = tuple(
            product
            for product in shielding_input_products
            if product.spectrum.particle is Particle.ELECTRON
        )

        calculation_result = calculation_result.set_component_status(
            component=ERB_SHIELDING_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=ERB_SHIELDING_COMPONENT,
            message="ERB shielding started.",
            details={
                "thicknesses_g_cm2": ",".join(
                    f"{value:g}" for value in config.shielding.thicknesses_g_cm2
                ),
                "input_proton_products": str(len(proton_products)),
                "input_electron_products": str(len(electron_products)),
            },
        )

        shielding_by_thickness = _calculate_erb_shielding_by_thickness(
            products=shielding_input_products,
            thicknesses_g_cm2=config.shielding.thicknesses_g_cm2,
        )
        shielded_products = tuple(
            shielding_result.shielded_product for shielding_result in shielding_by_thickness
        )
        products = (*on_orbit_products, *shielded_products)
        selected_shielded_products = select_products_for_assessment_mode(
            shielded_products,
            config.assessment_mode,
        )
        shielding_incomplete_reasons = _merge_incomplete_reasons(
            source_incomplete_reasons,
            _incomplete_reasons_from_products(selected_shielded_products),
        )

        calculation_result = calculation_result.set_model_info(
            ModelInfo(
                name=ERB_SHIELDING_MODEL,
                version=ERB_SHIELDING_MODEL_VERSION,
                status="calculated",
                source=ERB_SHIELDING_DOCUMENT,
            )
        )
        calculation_result = calculation_result.set_component_status(
            component=ERB_SHIELDING_COMPONENT,
            status=(
                ComponentStatus.INCOMPLETE
                if shielding_incomplete_reasons
                else ComponentStatus.COMPLETED
            ),
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=ERB_SHIELDING_COMPONENT,
            message="ERB shielding completed.",
            details={
                "shielded_products": str(len(shielded_products)),
                "thickness_count": str(len(config.shielding.thicknesses_g_cm2)),
            },
        )
        if shielding_incomplete_reasons:
            calculation_result = calculation_result.add_log_entry(
                level=LogLevel.WARNING,
                stage=ERB_SHIELDING_COMPONENT,
                message="ERB shielding products are incomplete.",
                details={"reasons": "; ".join(shielding_incomplete_reasons)},
            )

    output_tables = _erb_pipeline_output_tables(
        on_orbit_products=on_orbit_products,
        shielding_by_thickness=shielding_by_thickness,
        assessment_mode=config.assessment_mode,
    )

    calculation_result = calculation_result.set_component_status(
        component=ERB_OUTPUT_TABLES_COMPONENT,
        status=ComponentStatus.NOT_STARTED,
    )
    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=ERB_OUTPUT_TABLES_COMPONENT,
        message="ERB output tables started.",
        details={
            "table_count": str(len(output_tables)),
        },
    )
    calculation_result = _set_output_tables(
        calculation_result=calculation_result,
        tables=output_tables,
    )
    calculation_result = calculation_result.set_component_status(
        component=ERB_OUTPUT_TABLES_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )
    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=ERB_OUTPUT_TABLES_COMPONENT,
        message="ERB output tables completed.",
        details={
            "table_count": str(len(output_tables)),
        },
    )

    pipeline_incomplete_reasons = _merge_incomplete_reasons(
        source_incomplete_reasons,
        shielding_incomplete_reasons,
    )
    calculation_result = calculation_result.set_component_status(
        component=ERB_PIPELINE_COMPONENT,
        status=(
            ComponentStatus.INCOMPLETE
            if pipeline_incomplete_reasons
            else ComponentStatus.COMPLETED
        ),
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=ERB_PIPELINE_COMPONENT,
        message="ERB pipeline completed.",
        details=(
            {"incomplete_reasons": "; ".join(pipeline_incomplete_reasons)}
            if pipeline_incomplete_reasons
            else None
        ),
    )

    spectra = tuple(product.spectrum for product in products)

    return ErbPipelineResult(
        calculation_result=calculation_result,
        erb_model_result=erb_model_result,
        spectra=spectra,
        products=products,
        on_orbit_products=on_orbit_products,
        shielding_by_thickness=shielding_by_thickness,
        shielded_products=shielded_products,
    )
