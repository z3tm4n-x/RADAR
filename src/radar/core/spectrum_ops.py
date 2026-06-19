"""Safe operations on typed spectra."""

from __future__ import annotations

from math import isfinite

from radar.core.spectra import Spectrum1D


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
