"""Typed one-dimensional spectra."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit


@dataclass(frozen=True)
class Spectrum1D:
    """One-dimensional spectrum with explicit physical meaning."""

    x: tuple[float, ...]
    y: tuple[float, ...]
    x_unit: Unit
    y_unit: Unit
    quantity: SpectrumQuantity
    particle: Particle
    source: RadiationSource
    model: str

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            msg = "Spectrum grid and values must have the same length."
            raise ValueError(msg)

        if not self.x:
            msg = "Spectrum must contain at least one point."
            raise ValueError(msg)

        if not self.model:
            msg = "Spectrum model name must not be empty."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.x):
            msg = "Spectrum grid must contain only finite values."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.y):
            msg = "Spectrum values must contain only finite values."
            raise ValueError(msg)

        if any(value < 0.0 for value in self.y):
            msg = "Spectrum values must be non-negative."
            raise ValueError(msg)

        if any(right <= left for left, right in zip(self.x, self.x[1:])):
            msg = "Spectrum grid must be strictly increasing."
            raise ValueError(msg)

    def integral_trapezoid(self) -> float:
        """Return trapezoidal integral over the spectrum grid."""

        if len(self.x) == 1:
            return 0.0

        total = 0.0
        for left_x, right_x, left_y, right_y in zip(
            self.x,
            self.x[1:],
            self.y,
            self.y[1:],
        ):
            total += 0.5 * (left_y + right_y) * (right_x - left_x)

        return total
