"""GCR calculation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

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
from radar.core.types import RadiationSource
from radar.gcr.model import (
    GcrModelInput,
    GcrModelProtocol,
    GcrModelResult,
    validate_gcr_energy_spectrum,
)
from radar.gcr.penetration import apply_ost_geomagnetic_penetration_to_gcr_products

GCR_PIPELINE_COMPONENT = "gcr_pipeline"
GCR_MODEL_COMPONENT = "gcr_model"
GCR_GEOMAGNETIC_PENETRATION_COMPONENT = "gcr_geomagnetic_penetration"
UNVERSIONED_MODEL = "unversioned"
GCR_GEOMAGNETIC_PENETRATION_MODEL_VERSION = "appendix_zh_mlt_average_v1"
GCR_GEOMAGNETIC_PENETRATION_DOCUMENT = "OST 134-1044-2007 Appendix Zh"


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
    )


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

        if not spectrum.model.startswith(f"{model_spectrum.model}+"):
            msg = (
                "GCR pipeline derived spectra must identify their source model "
                "in the spectrum model string."
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

        if not product.spectrum.model.startswith(f"{model_product.spectrum.model}+"):
            msg = (
                "GCR pipeline derived product spectra must identify their source model "
                "in the spectrum model string."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class GcrPipelineResult:
    """Result of GCR pipeline calculation."""

    calculation_result: CalculationResult
    gcr_model_result: GcrModelResult
    spectra: tuple[Spectrum1D, ...]
    products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "GCR pipeline result must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_gcr_energy_spectrum(spectrum)

        _validate_pipeline_spectra_derived_from_model_result(
            spectra=self.spectra,
            model_spectra=self.gcr_model_result.spectra,
        )

        products = self.products or self.gcr_model_result.products

        if not products:
            msg = "GCR pipeline result must contain at least one radiation product."
            raise ValueError(msg)

        validate_products_match_spectra_and_source(
            products=products,
            spectra=self.spectra,
            source=RadiationSource.GCR,
            mismatch_message="GCR pipeline product spectra must match pipeline spectra.",
        )

        _validate_pipeline_products_derived_from_model_products(
            products=products,
            model_products=self.gcr_model_result.products,
        )

        object.__setattr__(self, "products", products)

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

    gcr_model_input = GcrModelInput(mission=config.mission)
    gcr_model_result = gcr_model.calculate(gcr_model_input)
    products = gcr_model_result.products

    calculation_result = calculation_result.set_component_status(
        component=GCR_MODEL_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )

    calculation_result = calculation_result.set_model_info(
        ModelInfo(
            name=gcr_model_result.model,
            version=UNVERSIONED_MODEL,
            status="calculated",
            source=gcr_model_result.document,
        )
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=GCR_MODEL_COMPONENT,
        message="GCR spectra calculated.",
        details={
            "model": gcr_model_result.model,
            "document": gcr_model_result.document,
            "lifetime_years": str(gcr_model_result.lifetime_years),
            "spectrum_count": str(len(gcr_model_result.spectra)),
        },
    )

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

        calculation_result = calculation_result.set_model_info(
            ModelInfo(
                name=penetration_output.penetration.model,
                version=GCR_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
                status="calculated",
                source=GCR_GEOMAGNETIC_PENETRATION_DOCUMENT,
            )
        )
        calculation_result = calculation_result.set_component_status(
            component=GCR_GEOMAGNETIC_PENETRATION_COMPONENT,
            status=ComponentStatus.COMPLETED,
        )
        calculation_result = calculation_result.add_log_entry(
            level=LogLevel.INFO,
            stage=GCR_GEOMAGNETIC_PENETRATION_COMPONENT,
            message="GCR geomagnetic penetration completed.",
            details={
                "products": str(len(products)),
                "penetration_model": penetration_output.penetration.model,
                "penetration_version": GCR_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
                "rigidity_grid_points": str(
                    len(penetration_output.rigidity_grid.values_gv)
                ),
            },
        )


    calculation_result = calculation_result.set_component_status(
        component=GCR_PIPELINE_COMPONENT,
        status=ComponentStatus.COMPLETED,
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
        products=products,
    )
