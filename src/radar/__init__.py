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
