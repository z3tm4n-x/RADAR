"""SEP proton calculation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.log import LogLevel
from radar.core.products import SpectrumProduct
from radar.core.project import CalculationConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.spectra import Spectrum1D
from radar.core.source_products import validate_product_allowed_for_source
from radar.core.types import RadiationProductKind, RadiationSource
from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.spectrum import apply_proton_penetration
from radar.sep.model import (
    SepModelInput,
    SepModelProtocol,
    SepModelResult,
    validate_sep_proton_fluence_spectrum,
)

SEP_PROTON_PIPELINE_COMPONENT = "sep_proton_pipeline"
SEP_MODEL_COMPONENT = "sep_model"
GEOMAGNETIC_PENETRATION_COMPONENT = "geomagnetic_penetration"
UNVERSIONED_MODEL = "unversioned"


def _sep_mission_fluence_product(
    spectrum: Spectrum1D,
    label: str,
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=spectrum,
        label=label,
    )


@dataclass(frozen=True)
class SepProtonPipelineResult:
    """Result of SEP proton pipeline calculation."""

    calculation_result: CalculationResult
    sep_model_result: SepModelResult
    raw_spectrum: Spectrum1D
    penetrated_spectrum: Spectrum1D
    raw_product: SpectrumProduct | None = None
    penetrated_product: SpectrumProduct | None = None

    def __post_init__(self) -> None:
        validate_sep_proton_fluence_spectrum(self.raw_spectrum)
        validate_sep_proton_fluence_spectrum(self.penetrated_spectrum)

        if self.raw_spectrum != self.sep_model_result.spectrum:
            msg = "SEP raw pipeline spectrum must match SEP model spectrum."
            raise ValueError(msg)

        raw_product = self.raw_product or self.sep_model_result.product
        penetrated_product = self.penetrated_product or _sep_mission_fluence_product(
            spectrum=self.penetrated_spectrum,
            label="SEP proton penetrated mission fluence",
        )

        validate_product_allowed_for_source(
            product=raw_product,
            source=RadiationSource.SEP,
        )
        validate_product_allowed_for_source(
            product=penetrated_product,
            source=RadiationSource.SEP,
        )

        if raw_product.kind is not RadiationProductKind.MISSION_FLUENCE:
            msg = "SEP raw pipeline product must be mission fluence."
            raise ValueError(msg)

        if penetrated_product.kind is not RadiationProductKind.MISSION_FLUENCE:
            msg = "SEP penetrated pipeline product must be mission fluence."
            raise ValueError(msg)

        if raw_product.spectrum != self.raw_spectrum:
            msg = "SEP raw product spectrum must match raw spectrum."
            raise ValueError(msg)

        if penetrated_product.spectrum != self.penetrated_spectrum:
            msg = "SEP penetrated product spectrum must match penetrated spectrum."
            raise ValueError(msg)

        object.__setattr__(self, "raw_product", raw_product)
        object.__setattr__(self, "penetrated_product", penetrated_product)


def calculate_sep_proton_pipeline(
    config: CalculationConfig,
    sep_model: SepModelProtocol,
    penetration: PenetrationFunction,
) -> SepProtonPipelineResult:
    """Calculate SEP proton fluence and apply geomagnetic penetration."""

    calculation_result = CalculationResult(config=config)

    calculation_result = calculation_result.set_component_status(
        component=SEP_PROTON_PIPELINE_COMPONENT,
        status=ComponentStatus.NOT_STARTED,
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_PROTON_PIPELINE_COMPONENT,
        message="SEP proton pipeline started.",
        details={
            "launch_year": str(config.mission.launch_year),
            "lifetime_years": str(config.mission.lifetime_years),
        },
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
        message="SEP proton fluence spectrum calculated.",
        details={
            "model": sep_model_result.model,
            "document": sep_model_result.document,
            "lifetime_years": str(sep_model_result.lifetime_years),
        },
    )

    penetrated_spectrum = apply_proton_penetration(
        spectrum=sep_model_result.spectrum,
        penetration=penetration,
    )

    calculation_result = calculation_result.set_component_status(
        component=GEOMAGNETIC_PENETRATION_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )

    calculation_result = calculation_result.set_model_info(
        ModelInfo(
            name=penetration.model,
            version=UNVERSIONED_MODEL,
            status="applied",
            source="geomagnetic penetration",
        )
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=GEOMAGNETIC_PENETRATION_COMPONENT,
        message="Geomagnetic penetration applied to SEP proton spectrum.",
        details={
            "model": penetration.model,
            "kp": str(penetration.kp),
        },
    )

    calculation_result = calculation_result.set_component_status(
        component=SEP_PROTON_PIPELINE_COMPONENT,
        status=ComponentStatus.COMPLETED,
    )

    calculation_result = calculation_result.add_log_entry(
        level=LogLevel.INFO,
        stage=SEP_PROTON_PIPELINE_COMPONENT,
        message="SEP proton pipeline completed.",
    )

    return SepProtonPipelineResult(
        calculation_result=calculation_result,
        sep_model_result=sep_model_result,
        raw_spectrum=sep_model_result.spectrum,
        penetrated_spectrum=penetrated_spectrum,
        raw_product=sep_model_result.product,
        penetrated_product=_sep_mission_fluence_product(
            spectrum=penetrated_spectrum,
            label="SEP proton penetrated mission fluence",
        ),
    )
