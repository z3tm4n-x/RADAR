"""OST 134-1044-2007 Appendix E ERB spectrum helpers."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, log


def _validate_energy_flux_grid(
    *,
    energies_mev: Sequence[float],
    differential_flux: Sequence[float],
) -> None:
    if not energies_mev:
        msg = "Appendix E energy grid must not be empty."
        raise ValueError(msg)

    if len(energies_mev) != len(differential_flux):
        msg = "Appendix E energy and flux grids must have the same length."
        raise ValueError(msg)

    if any(not isfinite(energy) or energy <= 0.0 for energy in energies_mev):
        msg = "Appendix E energy grid values must be finite and positive."
        raise ValueError(msg)

    if tuple(sorted(energies_mev)) != tuple(energies_mev):
        msg = "Appendix E energy grid must be sorted."
        raise ValueError(msg)

    if len(set(energies_mev)) != len(energies_mev):
        msg = "Appendix E energy grid values must be unique."
        raise ValueError(msg)

    if any(not isfinite(value) or value < 0.0 for value in differential_flux):
        msg = "Appendix E differential flux values must be finite and non-negative."
        raise ValueError(msg)


def _power_law_interval_integral(
    *,
    lower_energy_mev: float,
    upper_energy_mev: float,
    lower_flux: float,
    upper_flux: float,
) -> float:
    if lower_flux <= 0.0 and upper_flux <= 0.0:
        return 0.0

    if lower_flux > 0.0 and upper_flux > 0.0:
        exponent = -log(upper_flux / lower_flux) / log(upper_energy_mev / lower_energy_mev)

        if abs(exponent - 1.0) < 1.0e-8:
            return lower_flux * lower_energy_mev * log(upper_energy_mev / lower_energy_mev)

        return float(
            lower_flux
            * lower_energy_mev**exponent
            * (
                upper_energy_mev ** (1.0 - exponent)
                - lower_energy_mev ** (1.0 - exponent)
            )
            / (1.0 - exponent)
        )

    return 0.5 * (lower_flux + upper_flux) * (upper_energy_mev - lower_energy_mev)


def _power_law_tail_integral(
    *,
    energies_mev: Sequence[float],
    differential_flux: Sequence[float],
) -> float:
    positive_indices = [
        index
        for index, value in enumerate(differential_flux)
        if value > 0.0
    ]

    if len(positive_indices) < 2:
        return 0.0

    upper_index = positive_indices[-1]
    lower_index = positive_indices[-2]
    lower_flux = differential_flux[lower_index]
    upper_flux = differential_flux[upper_index]
    lower_energy = energies_mev[lower_index]
    upper_energy = energies_mev[upper_index]

    exponent = -log(upper_flux / lower_flux) / log(upper_energy / lower_energy)

    if exponent <= 1.05:
        return 0.0

    return upper_flux * upper_energy / (exponent - 1.0)


def integrate_differential_flux_tail_power_law(
    *,
    energies_mev: Sequence[float],
    differential_flux: Sequence[float],
) -> tuple[float, ...]:
    """Return integral flux Φ(>E) from a tabulated differential flux φ(E).

    This implements the Appendix E.4 integral spectrum numerically for a finite
    tabulated energy grid. Positive intervals use a power-law interpolation in
    energy; mixed zero/non-zero intervals use trapezoidal integration. The
    high-energy tail is extrapolated from the last two positive points only when
    the inferred power-law exponent gives a finite integral.
    """

    _validate_energy_flux_grid(
        energies_mev=energies_mev,
        differential_flux=differential_flux,
    )

    integral_flux = [0.0 for _ in differential_flux]

    accumulated = _power_law_tail_integral(
        energies_mev=energies_mev,
        differential_flux=differential_flux,
    )
    integral_flux[-1] = accumulated

    for index in range(len(energies_mev) - 2, -1, -1):
        accumulated += _power_law_interval_integral(
            lower_energy_mev=energies_mev[index],
            upper_energy_mev=energies_mev[index + 1],
            lower_flux=differential_flux[index],
            upper_flux=differential_flux[index + 1],
        )
        integral_flux[index] = accumulated

    return tuple(integral_flux)


__all__ = [
    "integrate_differential_flux_tail_power_law",
]
