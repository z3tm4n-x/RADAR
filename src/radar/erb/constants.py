"""Shared constants for Earth radiation belt models."""

from __future__ import annotations

EARTH_REFERENCE_RADIUS_KM = 6371.2
ERB_DIPOLE_MOMENT_GAUSS = 0.311653
DEFAULT_IGRF_EPOCH = 1985.0
ERB_SECONDS_PER_YEAR = 365.25 * 86_400.0

__all__ = [
    "DEFAULT_IGRF_EPOCH",
    "EARTH_REFERENCE_RADIUS_KM",
    "ERB_DIPOLE_MOMENT_GAUSS",
    "ERB_SECONDS_PER_YEAR",
]
