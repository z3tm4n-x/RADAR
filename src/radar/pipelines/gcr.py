"""GCR calculation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.log import LogLevel
from radar.core.products import SpectrumProduct
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.spectra import Spectrum1D
from radar.core.source_products import validate_product_allowed_for_source
from radar.core.types import RadiationSource
from radar.gcr.model import (
    GcrModelInput,
    GcrModelProtocol,
    GcrModelResult,
    validate_gcr_energy_spectrum,
)

GCR_PIPELINE_COMPONENT = "gcr_pipeline"
GCR_MODEL_COMPONENT = "gcr_model"
UNVERSIONED_MODEL = "unversioned"


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

        if self.spectra != self.gcr_model_result.spectra:
            msg = "GCR pipeline spectra must match GCR model result spectra."
            raise ValueError(msg)

        products = self.products or self.gcr_model_result.products

        if not products:
            msg = "GCR pipeline result must contain at least one radiation product."
            raise ValueError(msg)

        for product in products:
            validate_product_allowed_for_source(
                product=product,
                source=RadiationSource.GCR,
            )

        if tuple(product.spectrum for product in products) != self.spectra:
            msg = "GCR pipeline product spectra must match pipeline spectra."
            raise ValueError(msg)

        object.__setattr__(self, "products", products)


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

    gcr_model_input = GcrModelInput(mission=config.mission)
    gcr_model_result = gcr_model.calculate(gcr_model_input)

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

    calculation_result = calculation_result.set_component_status(
        component=GCR_PIPELINE_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=GCR_PIPELINE_COMPONENT,
        message="GCR pipeline completed.",
    )

    return GcrPipelineResult(
        calculation_result=calculation_result,
        gcr_model_result=gcr_model_result,
        spectra=gcr_model_result.spectra,
        products=gcr_model_result.products,
    )
