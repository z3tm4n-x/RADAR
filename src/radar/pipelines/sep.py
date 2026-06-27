"""SEP calculation pipeline."""

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
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
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
) -> tuple[OutputTable, ...]:
    tables: list[OutputTable] = list(
        sep_on_orbit_product_output_tables(on_orbit_products),
    )

    for shielding_result in shielding_let_by_thickness:
        for product_kind in SEP_SHIELDING_LET_PRODUCT_KIND_ORDER:
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
                    ),
                )
            )

    return tuple(tables)


def _validate_pipeline_spectra_derived_from_model_result(
    *,
    spectra: tuple[Spectrum1D, ...],
    model_spectra: tuple[Spectrum1D, ...],
) -> None:
    if len(spectra) < len(model_spectra):
        msg = "SEP pipeline spectra must match or derive from SEP model result spectra."
        raise ValueError(msg)

    if spectra[: len(model_spectra)] != model_spectra:
        msg = "SEP pipeline spectra must match or derive from SEP model result spectra."
        raise ValueError(msg)

    model_names = tuple(spectrum.model for spectrum in model_spectra)

    for spectrum in spectra[len(model_spectra) :]:
        _validate_sep_pipeline_spectrum(spectrum)

        if any(
            spectrum.model.startswith(f"{model_name}+")
            for model_name in model_names
        ):
            continue

        msg = "SEP pipeline spectra must match or derive from SEP model result spectra."
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

        if on_orbit_products != self.sep_model_result.products:
            msg = "SEP pipeline on-orbit products must match SEP model result products."
            raise ValueError(msg)

        if not products:
            msg = "SEP pipeline result must contain at least one radiation product."
            raise ValueError(msg)

        validate_products_match_spectra_and_source(
            products=products,
            spectra=self.spectra,
            source=RadiationSource.SEP,
            mismatch_message="SEP pipeline product spectra must match pipeline spectra.",
        )

        object.__setattr__(self, "products", products)
        object.__setattr__(self, "on_orbit_products", on_orbit_products)

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

    calculation_result = calculation_result.set_component_status(
        component=SEP_MODEL_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )

    calculation_result = calculation_result.set_model_info(
        ModelInfo(
            name=sep_model_result.model,
            version=UNVERSIONED_MODEL,
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

    on_orbit_products = sep_model_result.products
    products = on_orbit_products
    shielding_let_by_thickness: tuple[SepShieldingLetPipelineProducts, ...] = ()
    shielded_products: tuple[SpectrumProduct, ...] = ()
    let_products: tuple[SpectrumProduct, ...] = ()

    if profile_uses_ost_134_1044_2007(config.methodology.profile):
        proton_products = tuple(
            product
            for product in on_orbit_products
            if product.spectrum.particle is Particle.PROTON
        )
        hze_products = tuple(
            product
            for product in on_orbit_products
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
            products=on_orbit_products,
            tables=tables,
            thicknesses_g_cm2=config.shielding.thicknesses_g_cm2,
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
            status=ComponentStatus.COMPLETED,
        )
        calculation_result = calculation_result.set_component_status(
            component=SEP_LET_COMPONENT,
            status=ComponentStatus.COMPLETED,
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

    calculation_result = calculation_result.set_component_status(
        component=SEP_PIPELINE_COMPONENT,
        status=ComponentStatus.COMPLETED,
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
