"""RADAR calculation core."""

from radar.core.log import CalculationLog, CalculationLogEntry, LogLevel
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig, ShieldingConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.spectra import Spectrum1D
from radar.core.spectrum_ops import add_spectra, check_spectra_compatible, scale_spectrum

__all__ = [
    "CalculationConfig",
    "CalculationLog",
    "CalculationLogEntry",
    "CalculationResult",
    "ComponentStatus",
    "LogLevel",
    "MissionConfig",
    "ModelInfo",
    "OrbitConfig",
    "ShieldingConfig",
    "Spectrum1D",
    "add_spectra",
    "check_spectra_compatible",
    "scale_spectrum",
]
