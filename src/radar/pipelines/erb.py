"""ERB calculation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace

from radar.core.log import LogLevel
from radar.core.products import SpectrumProduct
from radar.core.profiles import (
    profile_uses_ost_134_1044_2007,
    validate_source_model_metadata_for_profile,
)
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.spectra import Spectrum1D
from radar.core.source_products import validate_products_match_spectra_and_source
from radar.core.types import Particle, RadiationProductKind, RadiationSource
from radar.erb.model import (
    ErbModelInput,
    ErbModelProtocol,
    ErbModelResult,
    validate_erb_energy_spectrum,
)
from radar.erb.output_tables import (
    erb_product_output_tables,
    erb_proton_shielding_output_tables,
)
from radar.erb.shielding import (
    ErbProtonShieldingResult,
    calculate_erb_proton_shielding_spectrum,
)
from radar.output_tables import OutputTable
from radar.shielding.resources import load_normative_shielding_tables

ERB_PIPELINE_COMPONENT = "erb_pipeline"
ERB_MODEL_COMPONENT = "erb_model"
ERB_SHIELDING_COMPONENT = "erb_shielding"
ERB_OUTPUT_TABLES_COMPONENT = "erb_output_tables"
UNVERSIONED_MODEL = "unversioned"
ERB_SHIELDING_MODEL = "erb_proton_al_shielding"
ERB_SHIELDING_MODEL_VERSION = "al_spherical_csda_secondary_v1"
ERB_SHIELDING_DOCUMENT = "RADAR shielding normative tables"


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
        f"erb_t{_table_id_float_token(thickness_g_cm2)}_"
        f"{product_kind.value}"
    )


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

    model_names = tuple(spectrum.model for spectrum in model_spectra)

    for spectrum in spectra[len(model_spectra) :]:
        validate_erb_energy_spectrum(spectrum)

        if any(
            spectrum.model.startswith(f"{model_name}+")
            for model_name in model_names
        ):
            continue

        msg = "ERB pipeline spectra must match or derive from ERB model result spectra."
        raise ValueError(msg)


@dataclass(frozen=True)
class ErbShieldingPipelineProduct:
    """ERB proton shielding result associated with one source product."""

    thickness_g_cm2: float
    source_product: SpectrumProduct
    result: ErbProtonShieldingResult
    shielded_product: SpectrumProduct


def _calculate_erb_proton_shielding_by_thickness(
    *,
    products: tuple[SpectrumProduct, ...],
    thicknesses_g_cm2: tuple[float, ...],
) -> tuple[ErbShieldingPipelineProduct, ...]:
    proton_products = tuple(
        product
        for product in products
        if product.spectrum.particle is Particle.PROTON
    )

    if not proton_products:
        return ()

    tables = load_normative_shielding_tables()
    shielding_results: list[ErbShieldingPipelineProduct] = []

    for thickness_g_cm2 in thicknesses_g_cm2:
        for product in proton_products:
            result = calculate_erb_proton_shielding_spectrum(
                spectrum=product.spectrum,
                tables=tables,
                thickness_g_cm2=thickness_g_cm2,
            )
            source_label = product.label or product.kind.value
            shielded_product = SpectrumProduct(
                kind=product.kind,
                spectrum=result.total,
                label=(
                    f"{source_label} behind Al shield "
                    f"{thickness_g_cm2:g} g/cm^2"
                ),
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
) -> tuple[OutputTable, ...]:
    tables: list[OutputTable] = list(erb_product_output_tables(on_orbit_products))

    for shielding_result in shielding_by_thickness:
        prefix = _shielding_table_prefix(
            thickness_g_cm2=shielding_result.thickness_g_cm2,
            product_kind=shielding_result.source_product.kind,
        )
        tables.extend(
            _prefixed_output_tables(
                prefix=prefix,
                tables=erb_proton_shielding_output_tables(
                    shielding_result.result,
                    include_components=False,
                ),
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

        validate_products_match_spectra_and_source(
            products=products,
            spectra=self.spectra,
            source=RadiationSource.ERB,
            mismatch_message="ERB pipeline product spectra must match pipeline spectra.",
        )

        object.__setattr__(self, "products", products)
        object.__setattr__(self, "on_orbit_products", on_orbit_products)


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

    erb_model_input = ErbModelInput(config=config)
    erb_model_result = erb_model.calculate(erb_model_input)

    calculation_result = calculation_result.set_component_status(
        component=ERB_MODEL_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )

    calculation_result = calculation_result.set_model_info(
        ModelInfo(
            name=erb_model_result.model,
            version=UNVERSIONED_MODEL,
            status="calculated",
            source=erb_model_result.document,
        )
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=ERB_MODEL_COMPONENT,
        message="ERB spectra calculated.",
        details={
            "model": erb_model_result.model,
            "document": erb_model_result.document,
            "lifetime_years": str(erb_model_result.lifetime_years),
            "kp": str(erb_model_result.kp),
            "spectrum_count": str(len(erb_model_result.spectra)),
        },
    )

    on_orbit_products = erb_model_result.products
    products = on_orbit_products
    shielding_by_thickness: tuple[ErbShieldingPipelineProduct, ...] = ()
    shielded_products: tuple[SpectrumProduct, ...] = ()

    if profile_uses_ost_134_1044_2007(config.methodology.profile):
        proton_products = tuple(
            product
            for product in on_orbit_products
            if product.spectrum.particle is Particle.PROTON
        )

        calculation_result = calculation_result.set_component_status(
            component=ERB_SHIELDING_COMPONENT,
            status=ComponentStatus.NOT_STARTED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=ERB_SHIELDING_COMPONENT,
            message="ERB proton shielding started.",
            details={
                "thicknesses_g_cm2": ",".join(
                    f"{value:g}" for value in config.shielding.thicknesses_g_cm2
                ),
                "input_proton_products": str(len(proton_products)),
                "input_electron_products": str(
                    len(on_orbit_products) - len(proton_products)
                ),
            },
        )

        shielding_by_thickness = _calculate_erb_proton_shielding_by_thickness(
            products=on_orbit_products,
            thicknesses_g_cm2=config.shielding.thicknesses_g_cm2,
        )
        shielded_products = tuple(
            shielding_result.shielded_product
            for shielding_result in shielding_by_thickness
        )
        products = (*on_orbit_products, *shielded_products)

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
            status=ComponentStatus.COMPLETED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=ERB_SHIELDING_COMPONENT,
            message="ERB proton shielding completed.",
            details={
                "shielded_products": str(len(shielded_products)),
                "thickness_count": str(len(config.shielding.thicknesses_g_cm2)),
            },
        )

    output_tables = _erb_pipeline_output_tables(
        on_orbit_products=on_orbit_products,
        shielding_by_thickness=shielding_by_thickness,
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

    calculation_result = calculation_result.set_component_status(
        component=ERB_PIPELINE_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=ERB_PIPELINE_COMPONENT,
        message="ERB pipeline completed.",
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
