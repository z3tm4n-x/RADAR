"""Resource loaders for Earth radiation belt models."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from radar.erb.constants import DEFAULT_IGRF_EPOCH
from radar.erb.igrf import IgrfCoefficients, parse_igrf_coefficients

ERB_IGRF14_RESOURCE_NAME = "erb/igrf14coeffs.txt"


@lru_cache(maxsize=8)
def load_igrf14_coefficients(
    *,
    epoch: float = DEFAULT_IGRF_EPOCH,
) -> IgrfCoefficients:
    """Load IGRF-14 coefficients from package resources."""

    resource = resources.files("radar.data.normative").joinpath(
        ERB_IGRF14_RESOURCE_NAME
    )

    return parse_igrf_coefficients(
        resource.read_text(encoding="utf-8"),
        epoch=epoch,
    )


__all__ = [
    "ERB_IGRF14_RESOURCE_NAME",
    "load_igrf14_coefficients",
]
