"""Resource loaders for Earth radiation belt models."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from radar.erb.constants import DEFAULT_IGRF_EPOCH
from radar.erb.igrf import IgrfCoefficients, parse_igrf_coefficients
from radar.erb.radbelt import RadbeltAscMap, parse_radbelt_asc_text

ERB_IGRF14_RESOURCE_NAME = "erb/igrf14coeffs.txt"

ERB_RADBELT_AE8MAX_RESOURCE_NAME = "erb/ae8max.asc"
ERB_RADBELT_AE8MIN_RESOURCE_NAME = "erb/ae8min.asc"
ERB_RADBELT_AP8MAX_RESOURCE_NAME = "erb/ap8max.asc"
ERB_RADBELT_AP8MIN_RESOURCE_NAME = "erb/ap8min.asc"

ERB_RADBELT_RESOURCE_NAMES: dict[str, str] = {
    "AE8MAX": ERB_RADBELT_AE8MAX_RESOURCE_NAME,
    "AE8MIN": ERB_RADBELT_AE8MIN_RESOURCE_NAME,
    "AP8MAX": ERB_RADBELT_AP8MAX_RESOURCE_NAME,
    "AP8MIN": ERB_RADBELT_AP8MIN_RESOURCE_NAME,
}


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


@lru_cache(maxsize=8)
def load_radbelt_asc_map(
    *,
    name: str,
    resource_name: str,
) -> RadbeltAscMap:
    """Load a RADBELT AE8/AP8 ASC map from package resources."""

    resource = resources.files("radar.data.normative").joinpath(resource_name)
    text = resource.read_bytes().decode("ascii", errors="ignore")

    return parse_radbelt_asc_text(
        name=name,
        text=text,
    )


@lru_cache(maxsize=1)
def load_ae8_ap8_radbelt_maps() -> dict[str, RadbeltAscMap]:
    """Load bundled AE8/AP8 RADBELT ASC maps."""

    return {
        name: load_radbelt_asc_map(name=name, resource_name=resource_name)
        for name, resource_name in ERB_RADBELT_RESOURCE_NAMES.items()
    }


__all__ = [
    "ERB_IGRF14_RESOURCE_NAME",
    "ERB_RADBELT_AE8MAX_RESOURCE_NAME",
    "ERB_RADBELT_AE8MIN_RESOURCE_NAME",
    "ERB_RADBELT_AP8MAX_RESOURCE_NAME",
    "ERB_RADBELT_AP8MIN_RESOURCE_NAME",
    "ERB_RADBELT_RESOURCE_NAMES",
    "load_ae8_ap8_radbelt_maps",
    "load_igrf14_coefficients",
    "load_radbelt_asc_map",
]
