"""GCR mission product aggregation policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit

GCR_MISSION_PRODUCTS_MODEL_SUFFIX: Final[str] = "mission_products"


@dataclass(frozen=True)
class GcrFluxTimeBin:
    """One time bin of a GCR differential flux spectrum."""

    spectrum: Spectrum1D
    duration_seconds: float

    def __post_init__(self) -> None:
        _validate_gcr_flux_spectrum(self.spectrum)

        if not isfinite(self.duration_seconds):
            msg = "GCR time-bin duration must be finite."
            raise ValueError(msg)

        if self.duration_seconds <= 0.0:
            msg = "GCR time-bin duration must be positive."
            raise ValueError(msg)


@dataclass(frozen=True)
class GcrMissionProducts:
    """Mission products derived from time-binned GCR flux spectra."""

    mean_flux: SpectrumProduct
    maximum_flux: SpectrumProduct
    mission_fluence: SpectrumProduct

    @property
    def products(self) -> tuple[SpectrumProduct, SpectrumProduct, SpectrumProduct]:
        """Return products in deterministic order."""

        return (
            self.mean_flux,
            self.maximum_flux,
            self.mission_fluence,
        )

    def __post_init__(self) -> None:
        if self.mean_flux.kind is not RadiationProductKind.MEAN_FLUX:
            msg = "GCR mean-flux product kind must be MEAN_FLUX."
            raise ValueError(msg)

        if self.maximum_flux.kind is not RadiationProductKind.MAXIMUM_FLUX:
            msg = "GCR maximum-flux product kind must be MAXIMUM_FLUX."
            raise ValueError(msg)

        if self.mission_fluence.kind is not RadiationProductKind.MISSION_FLUENCE:
            msg = "GCR mission-fluence product kind must be MISSION_FLUENCE."
            raise ValueError(msg)


def _validate_gcr_flux_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.source is not RadiationSource.GCR:
        msg = "GCR mission aggregation input spectrum source must be GCR."
        raise ValueError(msg)

    if spectrum.particle not in (Particle.PROTON, Particle.HZE):
        msg = "GCR mission aggregation supports only proton and HZE spectra."
        raise ValueError(msg)

    if spectrum.quantity is not SpectrumQuantity.DIFFERENTIAL_FLUX:
        msg = "GCR mission aggregation input spectrum quantity must be differential flux."
        raise ValueError(msg)

    if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUX:
        msg = "GCR mission aggregation input spectrum unit must be differential flux."
        raise ValueError(msg)

    if spectrum.x_unit not in (Unit.MEV, Unit.GEV_PER_NUCLEON):
        msg = "GCR mission aggregation input energy grid must be in MeV or GeV/nucleon."
        raise ValueError(msg)


def _validate_matching_bins(time_bins: tuple[GcrFluxTimeBin, ...]) -> None:
    if not time_bins:
        msg = "GCR mission aggregation requires at least one time bin."
        raise ValueError(msg)

    reference = time_bins[0].spectrum

    for time_bin in time_bins[1:]:
        spectrum = time_bin.spectrum

        if spectrum.x != reference.x:
            msg = "GCR time-binned spectra must share the same energy grid."
            raise ValueError(msg)

        if spectrum.x_unit is not reference.x_unit:
            msg = "GCR time-binned spectra must share the same energy unit."
            raise ValueError(msg)

        if spectrum.particle is not reference.particle:
            msg = "GCR time-binned spectra must share the same particle group."
            raise ValueError(msg)

        if spectrum.source is not reference.source:
            msg = "GCR time-binned spectra must share the same source."
            raise ValueError(msg)


def _total_duration_seconds(time_bins: tuple[GcrFluxTimeBin, ...]) -> float:
    return sum(time_bin.duration_seconds for time_bin in time_bins)


def _time_integral_values(time_bins: tuple[GcrFluxTimeBin, ...]) -> tuple[float, ...]:
    point_count = len(time_bins[0].spectrum.y)
    values: list[float] = []

    for index in range(point_count):
        values.append(
            sum(
                time_bin.spectrum.y[index] * time_bin.duration_seconds
                for time_bin in time_bins
            )
        )

    return tuple(values)


def _time_weighted_mean_values(time_bins: tuple[GcrFluxTimeBin, ...]) -> tuple[float, ...]:
    total_duration = _total_duration_seconds(time_bins)
    fluence_values = _time_integral_values(time_bins)

    return tuple(value / total_duration for value in fluence_values)


def _pointwise_maximum_values(time_bins: tuple[GcrFluxTimeBin, ...]) -> tuple[float, ...]:
    point_count = len(time_bins[0].spectrum.y)
    values: list[float] = []

    for index in range(point_count):
        values.append(max(time_bin.spectrum.y[index] for time_bin in time_bins))

    return tuple(values)


def _product_spectrum(
    *,
    reference: Spectrum1D,
    values: tuple[float, ...],
    y_unit: Unit,
    quantity: SpectrumQuantity,
    model_component: str,
) -> Spectrum1D:
    return Spectrum1D(
        x=reference.x,
        y=values,
        x_unit=reference.x_unit,
        y_unit=y_unit,
        quantity=quantity,
        particle=reference.particle,
        source=RadiationSource.GCR,
        model=f"{reference.model}:{GCR_MISSION_PRODUCTS_MODEL_SUFFIX}:{model_component}",
    )


def calculate_gcr_mission_products(
    time_bins: tuple[GcrFluxTimeBin, ...],
) -> GcrMissionProducts:
    """Aggregate time-binned GCR flux spectra into mission products.

    Policy:
        MEAN_FLUX = time-weighted mean flux.
        MAXIMUM_FLUX = pointwise maximum flux.
        MISSION_FLUENCE = time integral of flux in seconds.
    """

    _validate_matching_bins(time_bins)

    reference = time_bins[0].spectrum

    mean_flux_spectrum = _product_spectrum(
        reference=reference,
        values=_time_weighted_mean_values(time_bins),
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        model_component="mean_flux",
    )
    maximum_flux_spectrum = _product_spectrum(
        reference=reference,
        values=_pointwise_maximum_values(time_bins),
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        model_component="maximum_flux",
    )
    mission_fluence_spectrum = _product_spectrum(
        reference=reference,
        values=_time_integral_values(time_bins),
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        model_component="mission_fluence",
    )

    return GcrMissionProducts(
        mean_flux=SpectrumProduct(
            kind=RadiationProductKind.MEAN_FLUX,
            spectrum=mean_flux_spectrum,
            label=f"GCR {reference.particle.value} mean flux",
        ),
        maximum_flux=SpectrumProduct(
            kind=RadiationProductKind.MAXIMUM_FLUX,
            spectrum=maximum_flux_spectrum,
            label=f"GCR {reference.particle.value} maximum flux",
        ),
        mission_fluence=SpectrumProduct(
            kind=RadiationProductKind.MISSION_FLUENCE,
            spectrum=mission_fluence_spectrum,
            label=f"GCR {reference.particle.value} mission fluence",
        ),
    )


__all__ = [
    "GCR_MISSION_PRODUCTS_MODEL_SUFFIX",
    "GcrFluxTimeBin",
    "GcrMissionProducts",
    "calculate_gcr_mission_products",
]
