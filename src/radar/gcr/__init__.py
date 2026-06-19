"""Galactic cosmic ray models."""

from radar.gcr.model import (
    GcrModelInput,
    GcrModelProtocol,
    GcrModelResult,
    StaticGcrModel,
    validate_gcr_energy_spectrum,
)

__all__ = [
    "GcrModelInput",
    "GcrModelProtocol",
    "GcrModelResult",
    "StaticGcrModel",
    "validate_gcr_energy_spectrum",
]
