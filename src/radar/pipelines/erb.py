"""ERB calculation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.log import LogLevel
from radar.core.products import SpectrumProduct
from radar.core.profiles import validate_source_model_metadata_for_profile
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.spectra import Spectrum1D
from radar.core.source_products import (
    validate_products_match_spectra_and_source,
    validate_spectra_match_expected,
)
from radar.core.types import RadiationSource
from radar.erb.model import (
    ErbModelInput,
    ErbModelProtocol,
    ErbModelResult,
    validate_erb_energy_spectrum,
)

ERB_PIPELINE_COMPONENT = "erb_pipeline"
ERB_MODEL_COMPONENT = "erb_model"
UNVERSIONED_MODEL = "unversioned"


@dataclass(frozen=True)
class ErbPipelineResult:
    """Result of ERB pipeline calculation."""

    calculation_result: CalculationResult
    erb_model_result: ErbModelResult
    spectra: tuple[Spectrum1D, ...]
    products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "ERB pipeline result must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_erb_energy_spectrum(spectrum)

        validate_spectra_match_expected(
            spectra=self.spectra,
            expected_spectra=self.erb_model_result.spectra,
            mismatch_message="ERB pipeline spectra must match ERB model result spectra.",
        )

        products = self.products or self.erb_model_result.products

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

    calculation_result = calculation_result.set_component_status(
        component=ERB_PIPELINE_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=ERB_PIPELINE_COMPONENT,
        message="ERB pipeline completed.",
    )

    return ErbPipelineResult(
        calculation_result=calculation_result,
        erb_model_result=erb_model_result,
        spectra=erb_model_result.spectra,
        products=erb_model_result.products,
    )
