"""RADAR — Radiation Analysis and Design Assessment Resource."""

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig, ShieldingConfig
from radar.core.result import CalculationResult
from radar.core.spectra import Spectrum1D

__version__ = "0.1.0"

__all__ = [
    "CalculationConfig",
    "CalculationResult",
    "MissionConfig",
    "OrbitConfig",
    "ShieldingConfig",
    "Spectrum1D",
    "__version__",
]

from radar.model_registry import (
    REGISTERED_SOURCE_MODELS,
    SourceModelRegistration,
    registered_source_model_families,
    source_model_class,
    source_model_class_for_profile,
    source_model_registration,
    source_model_registration_for_profile,
)

__all__ = [
    "REGISTERED_SOURCE_MODELS",
    "SourceModelRegistration",
    "registered_source_model_families",
    "source_model_class",
    "source_model_class_for_profile",
    "source_model_registration",
    "source_model_registration_for_profile",
]
