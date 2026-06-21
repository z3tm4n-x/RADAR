"""SEP source-products pipeline before geomagnetic penetration."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.log import LogLevel
from radar.core.products import SpectrumProduct
from radar.core.profiles import validate_source_model_metadata_for_profile
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.source_products import validate_products_match_spectra_and_source
from radar.core.types import RadiationSource
from radar.sep.model import (
    SepModelInput,
    SepModelProtocol,
    SepModelResult,
    validate_sep_energy_spectrum,
)

SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT = "sep_source_products_pipeline"
SEP_SOURCE_MODEL_COMPONENT = "sep_source_model"
UNVERSIONED_MODEL = "unversioned"


@dataclass(frozen=True)
class SepSourceProductsPipelineResult:
    """Result of SEP source-products calculation before geomagnetic penetration."""

    calculation_result: CalculationResult
    sep_model_result: SepModelResult
    products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        products = self.products or self.sep_model_result.products

        if not products:
            msg = "SEP source-products pipeline result must contain products."
            raise ValueError(msg)

        if products != self.sep_model_result.products:
            msg = "SEP source pipeline products must match SEP model result products."
            raise ValueError(msg)

        spectra = tuple(product.spectrum for product in products)

        for spectrum in spectra:
            validate_sep_energy_spectrum(spectrum)

        validate_products_match_spectra_and_source(
            products=products,
            spectra=spectra,
            source=RadiationSource.SEP,
            mismatch_message="SEP source product spectra must match source spectra.",
        )

        object.__setattr__(self, "products", products)


def calculate_sep_source_products_pipeline(
    config: CalculationConfig,
    sep_model: SepModelProtocol,
) -> SepSourceProductsPipelineResult:
    """Calculate raw SEP source products before geomagnetic penetration."""

    calculation_result = CalculationResult(config=config)

    calculation_result = calculation_result.set_component_status(
        component=SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT,
        status=ComponentStatus.NOT_STARTED,
    )
    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT,
        message="SEP source-products pipeline started.",
        details={
            "launch_year": str(config.mission.launch_year),
            "lifetime_years": str(config.mission.lifetime_years),
        },
    )

    metadata = sep_model.metadata
    validate_source_model_metadata_for_profile(
        metadata=metadata,
        profile=config.methodology.profile,
    )

    sep_model_result = sep_model.calculate(SepModelInput(mission=config.mission))

    calculation_result = calculation_result.set_component_status(
        component=SEP_SOURCE_MODEL_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )
    calculation_result = calculation_result.set_model_info(
        ModelInfo(
            name=sep_model_result.model,
            version=metadata.version or UNVERSIONED_MODEL,
            status="calculated",
            source=sep_model_result.document,
        )
    )
    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_SOURCE_MODEL_COMPONENT,
        message="SEP source products calculated.",
        details={
            "model": sep_model_result.model,
            "document": sep_model_result.document,
            "version": metadata.version or UNVERSIONED_MODEL,
            "products": str(len(sep_model_result.products)),
        },
    )

    calculation_result = calculation_result.set_component_status(
        component=SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )
    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_SOURCE_PRODUCTS_PIPELINE_COMPONENT,
        message="SEP source-products pipeline completed.",
    )

    return SepSourceProductsPipelineResult(
        calculation_result=calculation_result,
        sep_model_result=sep_model_result,
        products=sep_model_result.products,
    )
