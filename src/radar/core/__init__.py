"""RADAR calculation core."""

from radar.core.dose_units import (
    dose_quantity_for_spectrum_quantity,
    is_accumulated_dose_unit,
    is_dose_rate_unit,
    radiation_quantity_mode_for_spectrum_quantity,
    validate_dose_unit_for_quantity,
)
from radar.core.log import CalculationLog, CalculationLogEntry, LogLevel
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig, ShieldingConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.spectra import Spectrum1D
from radar.core.spectrum_ops import add_spectra, check_spectra_compatible, scale_spectrum
from radar.core.types import DoseQuantity, RadiationQuantityMode

__all__ = [
    "CalculationConfig",
    "CalculationLog",
    "CalculationLogEntry",
    "CalculationResult",
    "ComponentStatus",
    "DoseQuantity",
    "LogLevel",
    "MissionConfig",
    "ModelInfo",
    "OrbitConfig",
    "RadiationQuantityMode",
    "ShieldingConfig",
    "Spectrum1D",
    "add_spectra",
    "check_spectra_compatible",
    "dose_quantity_for_spectrum_quantity",
    "is_accumulated_dose_unit",
    "is_dose_rate_unit",
    "radiation_quantity_mode_for_spectrum_quantity",
    "scale_spectrum",
    "validate_dose_unit_for_quantity",
]
