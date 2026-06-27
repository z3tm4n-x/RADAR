"""Safe operations on typed spectra."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, log

from radar.core.spectra import Spectrum1D


def _validate_differential_integration_grid(
    *,
    energies: Sequence[float],
    differential_values: Sequence[float],
    context: str,
) -> None:
    if not energies:
        msg = f"{context} energy grid must not be empty."
        raise ValueError(msg)

    if len(energies) != len(differential_values):
        msg = f"{context} energy and differential-value grids must have the same length."
        raise ValueError(msg)

    if any(not isfinite(energy) or energy <= 0.0 for energy in energies):
        msg = f"{context} energy grid values must be finite and positive."
        raise ValueError(msg)

    if tuple(sorted(energies)) != tuple(energies):
        msg = f"{context} energy grid must be sorted."
        raise ValueError(msg)

    if len(set(energies)) != len(energies):
        msg = f"{context} energy grid values must be unique."
        raise ValueError(msg)

    if any(not isfinite(value) or value < 0.0 for value in differential_values):
        msg = f"{context} differential values must be finite and non-negative."
        raise ValueError(msg)


def _power_law_interval_integral(
    *,
    lower_energy: float,
    upper_energy: float,
    lower_value: float,
    upper_value: float,
) -> float:
    if lower_value <= 0.0 and upper_value <= 0.0:
        return 0.0

    if lower_value > 0.0 and upper_value > 0.0:
        exponent = -log(upper_value / lower_value) / log(upper_energy / lower_energy)

        if abs(exponent - 1.0) < 1.0e-8:
            return lower_value * lower_energy * log(upper_energy / lower_energy)

        return float(
            lower_value
            * lower_energy**exponent
            * (
                upper_energy ** (1.0 - exponent)
                - lower_energy ** (1.0 - exponent)
            )
            / (1.0 - exponent)
        )

    return 0.5 * (lower_value + upper_value) * (upper_energy - lower_energy)


def _power_law_tail_integral(
    *,
    energies: Sequence[float],
    differential_values: Sequence[float],
) -> float:
    positive_indices = [
        index
        for index, value in enumerate(differential_values)
        if value > 0.0
    ]

    if len(positive_indices) < 2:
        return 0.0

    upper_index = positive_indices[-1]
    lower_index = positive_indices[-2]
    lower_value = differential_values[lower_index]
    upper_value = differential_values[upper_index]
    lower_energy = energies[lower_index]
    upper_energy = energies[upper_index]

    exponent = -log(upper_value / lower_value) / log(upper_energy / lower_energy)

    if exponent <= 1.05:
        return 0.0

    return upper_value * upper_energy / (exponent - 1.0)


def integrate_differential_spectrum_tail_power_law(
    *,
    energies: Sequence[float],
    differential_values: Sequence[float],
    context: str = "Spectrum",
) -> tuple[float, ...]:
    """Return integral values from tabulated differential values.

    Positive intervals use power-law interpolation in energy. Mixed zero/non-zero
    intervals use trapezoidal integration. The high-energy tail is extrapolated
    from the last two positive points only when the inferred power-law exponent
    gives a finite integral.
    """

    _validate_differential_integration_grid(
        energies=energies,
        differential_values=differential_values,
        context=context,
    )

    integral_values = [0.0 for _ in differential_values]

    accumulated = _power_law_tail_integral(
        energies=energies,
        differential_values=differential_values,
    )
    integral_values[-1] = accumulated

    for index in range(len(energies) - 2, -1, -1):
        accumulated += _power_law_interval_integral(
            lower_energy=energies[index],
            upper_energy=energies[index + 1],
            lower_value=differential_values[index],
            upper_value=differential_values[index + 1],
        )
        integral_values[index] = accumulated

    return tuple(integral_values)


def _check_same_grid(left: Spectrum1D, right: Spectrum1D) -> None:
    if left.x != right.x:
        msg = "Spectra must have the same grid."
        raise ValueError(msg)


def _check_compatible_metadata(left: Spectrum1D, right: Spectrum1D) -> None:
    if left.x_unit is not right.x_unit:
        msg = "Spectra must have the same x unit."
        raise ValueError(msg)

    if left.y_unit is not right.y_unit:
        msg = "Spectra must have the same y unit."
        raise ValueError(msg)

    if left.quantity is not right.quantity:
        msg = "Spectra must have the same quantity."
        raise ValueError(msg)

    if left.particle is not right.particle:
        msg = "Spectra must have the same particle."
        raise ValueError(msg)

    if left.source is not right.source:
        msg = "Spectra must have the same radiation source."
        raise ValueError(msg)


def check_spectra_compatible(left: Spectrum1D, right: Spectrum1D) -> None:
    """Validate that two spectra may be combined pointwise."""

    _check_same_grid(left=left, right=right)
    _check_compatible_metadata(left=left, right=right)


def scale_spectrum(
    spectrum: Spectrum1D,
    factor: float,
    model: str | None = None,
) -> Spectrum1D:
    """Return a spectrum multiplied by a non-negative finite factor."""

    if not isfinite(factor):
        msg = "Spectrum scale factor must be finite."
        raise ValueError(msg)

    if factor < 0.0:
        msg = "Spectrum scale factor must be non-negative."
        raise ValueError(msg)

    output_model = model

    if output_model is None:
        output_model = f"{spectrum.model}*{factor:g}"

    return Spectrum1D(
        x=spectrum.x,
        y=tuple(value * factor for value in spectrum.y),
        x_unit=spectrum.x_unit,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=spectrum.source,
        model=output_model,
    )


def add_spectra(
    left: Spectrum1D,
    right: Spectrum1D,
    model: str | None = None,
) -> Spectrum1D:
    """Return pointwise sum of two compatible spectra."""

    check_spectra_compatible(left=left, right=right)

    output_model = model

    if output_model is None:
        output_model = f"{left.model}+{right.model}"

    return Spectrum1D(
        x=left.x,
        y=tuple(left_value + right_value for left_value, right_value in zip(left.y, right.y)),
        x_unit=left.x_unit,
        y_unit=left.y_unit,
        quantity=left.quantity,
        particle=left.particle,
        source=left.source,
        model=output_model,
    )
