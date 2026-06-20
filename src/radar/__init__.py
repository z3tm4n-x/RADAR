"""RADAR — Radiation Analysis and Design Assessment Resource."""

from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    ShieldingConfig,
    SourceModelSelectionConfig,
)
from radar.core.result import CalculationResult
from radar.core.spectra import Spectrum1D
from radar.model_registry import (
    REGISTERED_SOURCE_MODELS,
    SourceModelBundle,
    SourceModelRegistration,
    registered_source_model_families,
    source_model_bundle_for_profile,
    source_model_bundle_for_selection,
    source_model_class,
    source_model_class_for_profile,
    source_model_classes_for_profile,
    source_model_classes_for_selection,
    source_model_registration,
    source_model_registration_for_profile,
)

__version__ = "0.1.0"

__all__ = [
    "CalculationConfig",
    "CalculationResult",
    "MethodologyConfig",
    "MissionConfig",
    "OrbitConfig",
    "REGISTERED_SOURCE_MODELS",
    "ShieldingConfig",
    "SourceModelBundle",
    "SourceModelRegistration",
    "SourceModelSelectionConfig",
    "Spectrum1D",
    "__version__",
    "registered_source_model_families",
    "source_model_bundle_for_profile",
    "source_model_bundle_for_selection",
    "source_model_class",
    "source_model_class_for_profile",
    "source_model_classes_for_profile",
    "source_model_classes_for_selection",
    "source_model_registration",
    "source_model_registration_for_profile",
]
