"""Combination utilities for LET spectra."""

from __future__ import annotations

from math import exp, log
from typing import Iterable

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, SpectrumQuantity
from radar.core.units import Unit

COMBINED_LET_MODEL = "combined_proton_hze_let"

_LET_QUANTITIES = (
    SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
)

_LET_UNITS = (
    Unit.DIFFERENTIAL_LET_FLUENCE,
    Unit.DIFFERENTIAL_LET_FLUX,
)


def _log_edges_from_centers(centers: tuple[float, ...]) -> tuple[float, ...]:
    if len(centers) < 2:
        msg = "At least two LET centers are required to build logarithmic edges."
        raise ValueError(msg)

    if any(center <= 0.0 for center in centers):
        msg = "LET centers must be positive."
        raise ValueError(msg)

    if any(right <= left for left, right in zip(centers, centers[1:])):
        msg = "LET centers must be strictly increasing."
        raise ValueError(msg)

    log_centers = tuple(log(center) for center in centers)
    edges: list[float] = [0.0] * (len(centers) + 1)

    for index in range(1, len(centers)):
        edges[index] = exp(0.5 * (log_centers[index - 1] + log_centers[index]))

    edges[0] = exp(log_centers[0] - 0.5 * (log_centers[1] - log_centers[0]))
    edges[-1] = exp(log_centers[-1] + 0.5 * (log_centers[-1] - log_centers[-2]))

    return tuple(edges)


def _validate_let_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.x_unit is not Unit.LET:
        msg = "LET spectrum x unit must be Unit.LET."
        raise ValueError(msg)

    if spectrum.y_unit not in _LET_UNITS:
        msg = "LET spectrum y unit must be a differential LET unit."
        raise ValueError(msg)

    if spectrum.quantity not in _LET_QUANTITIES:
        msg = "LET spectrum quantity must be a LET differential quantity."
        raise ValueError(msg)

    if len(spectrum.x) < 2:
        msg = "LET spectrum must contain at least two LET points."
        raise ValueError(msg)


def _integral_piecewise_constant(spectrum: Spectrum1D) -> float:
    edges = _log_edges_from_centers(spectrum.x)

    return sum(
        value * (edges[index + 1] - edges[index])
        for index, value in enumerate(spectrum.y)
    )


def rebin_let_spectrum_to_grid(
    *,
    spectrum: Spectrum1D,
    let_centers: tuple[float, ...],
) -> Spectrum1D:
    """Conservatively rebin a LET spectrum to a target LET grid."""

    _validate_let_spectrum(spectrum)

    source_edges = _log_edges_from_centers(spectrum.x)
    target_edges = _log_edges_from_centers(let_centers)
    target_counts = [0.0 for _ in let_centers]

    for source_index, source_value in enumerate(spectrum.y):
        source_left = source_edges[source_index]
        source_right = source_edges[source_index + 1]
        source_width = source_right - source_left

        if source_width <= 0.0 or source_value == 0.0:
            continue

        for target_index in range(len(let_centers)):
            target_left = target_edges[target_index]
            target_right = target_edges[target_index + 1]
            overlap_left = max(source_left, target_left)
            overlap_right = min(source_right, target_right)
            overlap = overlap_right - overlap_left

            if overlap <= 0.0:
                continue

            target_counts[target_index] += source_value * overlap

    target_values = tuple(
        count / (target_edges[index + 1] - target_edges[index])
        for index, count in enumerate(target_counts)
    )

    return Spectrum1D(
        x=let_centers,
        y=target_values,
        x_unit=Unit.LET,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=spectrum.source,
        model=f"{spectrum.model}+rebin_let",
    )


def _as_tuple(spectra: Iterable[Spectrum1D]) -> tuple[Spectrum1D, ...]:
    result = tuple(spectra)

    if not result:
        msg = "At least one LET spectrum is required."
        raise ValueError(msg)

    return result


def combine_let_spectra(
    *,
    spectra: Iterable[Spectrum1D],
    let_centers: tuple[float, ...] | None = None,
    particle: Particle = Particle.HZE,
    model: str | None = None,
) -> Spectrum1D:
    """Return a combined LET spectrum on a common LET grid.

    Particle.HZE is used as the existing group particle marker because RADAR does
    not yet define a dedicated mixed-particle enum value.
    """

    spectra_tuple = _as_tuple(spectra)
    first = spectra_tuple[0]
    _validate_let_spectrum(first)

    for spectrum in spectra_tuple[1:]:
        _validate_let_spectrum(spectrum)

        if spectrum.source is not first.source:
            msg = "Combined LET spectra must have the same radiation source."
            raise ValueError(msg)

        if spectrum.quantity is not first.quantity:
            msg = "Combined LET spectra must have the same quantity."
            raise ValueError(msg)

        if spectrum.y_unit is not first.y_unit:
            msg = "Combined LET spectra must have the same y unit."
            raise ValueError(msg)

    output_centers = let_centers or first.x
    _ = _log_edges_from_centers(output_centers)

    rebinned = tuple(
        rebin_let_spectrum_to_grid(spectrum=spectrum, let_centers=output_centers)
        for spectrum in spectra_tuple
    )
    total_values = tuple(
        sum(spectrum.y[index] for spectrum in rebinned)
        for index in range(len(output_centers))
    )

    return Spectrum1D(
        x=output_centers,
        y=total_values,
        x_unit=Unit.LET,
        y_unit=first.y_unit,
        quantity=first.quantity,
        particle=particle,
        source=first.source,
        model=model or COMBINED_LET_MODEL,
    )


def combine_proton_and_hze_let_spectra(
    *,
    proton_let: Spectrum1D,
    hze_let_spectra: Iterable[Spectrum1D],
    let_centers: tuple[float, ...] | None = None,
    model: str | None = None,
) -> Spectrum1D:
    """Return combined proton + HZE LET spectrum."""

    if proton_let.particle is not Particle.PROTON:
        msg = "Combined proton/HZE LET requires a proton LET spectrum."
        raise ValueError(msg)

    hze_tuple = tuple(hze_let_spectra)

    if not hze_tuple:
        msg = "Combined proton/HZE LET requires at least one HZE LET spectrum."
        raise ValueError(msg)

    for spectrum in hze_tuple:
        if spectrum.particle is not Particle.HZE:
            msg = "Combined proton/HZE LET requires HZE LET spectra."
            raise ValueError(msg)

    return combine_let_spectra(
        spectra=(proton_let, *hze_tuple),
        let_centers=let_centers,
        particle=Particle.HZE,
        model=model or COMBINED_LET_MODEL,
    )


__all__ = [
    "COMBINED_LET_MODEL",
    "combine_let_spectra",
    "combine_proton_and_hze_let_spectra",
    "rebin_let_spectrum_to_grid",
]
