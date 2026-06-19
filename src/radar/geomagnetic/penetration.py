"""Geomagnetic penetration function containers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from radar.core.constants import DEFAULT_KP, MAX_KP, MIN_KP
from radar.geomagnetic.rigidity import RigidityGrid


@dataclass(frozen=True)
class PenetrationFunction:
    """Geomagnetic penetration function psi(R).

    The values describe the fraction of time or trajectory access
    for particles with a given magnetic rigidity.

    This class stores and interpolates the function only.
    It does not calculate cutoff rigidity.
    """

    rigidity_grid: RigidityGrid
    values: tuple[float, ...]
    model: str
    kp: int = DEFAULT_KP

    def __post_init__(self) -> None:
        if len(self.values) != len(self.rigidity_grid.values_gv):
            msg = "Penetration values must match rigidity grid length."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.values):
            msg = "Penetration values must be finite."
            raise ValueError(msg)

        if any(value < 0.0 or value > 1.0 for value in self.values):
            msg = "Penetration values must be in the range 0..1."
            raise ValueError(msg)

        if not isinstance(self.kp, int):
            msg = "Kp must be an integer."
            raise ValueError(msg)

        if not MIN_KP <= self.kp <= MAX_KP:
            msg = "Kp must be in the range 0..9."
            raise ValueError(msg)

        if not self.model:
            msg = "Penetration model name must not be empty."
            raise ValueError(msg)

    def value_at(self, rigidity_gv: float) -> float:
        """Return linearly interpolated penetration value."""

        if not isfinite(rigidity_gv):
            msg = "Rigidity must be finite."
            raise ValueError(msg)

        if rigidity_gv <= 0.0:
            msg = "Rigidity must be positive."
            raise ValueError(msg)

        grid = self.rigidity_grid.values_gv

        if rigidity_gv <= grid[0]:
            return self.values[0]

        if rigidity_gv >= grid[-1]:
            return self.values[-1]

        for index in range(1, len(grid)):
            right_rigidity = grid[index]

            if rigidity_gv <= right_rigidity:
                left_rigidity = grid[index - 1]
                left_value = self.values[index - 1]
                right_value = self.values[index]

                fraction = (rigidity_gv - left_rigidity) / (
                    right_rigidity - left_rigidity
                )

                return left_value + fraction * (right_value - left_value)

        msg = "Failed to interpolate penetration function."
        raise RuntimeError(msg)
