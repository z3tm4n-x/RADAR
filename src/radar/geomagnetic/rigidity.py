"""Magnetic rigidity grid utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RigidityGrid:
    """Strictly increasing magnetic rigidity grid in GV."""

    values_gv: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values_gv:
            msg = "Rigidity grid must contain at least one point."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.values_gv):
            msg = "Rigidity grid values must be finite."
            raise ValueError(msg)

        if any(value <= 0.0 for value in self.values_gv):
            msg = "Rigidity grid values must be positive."
            raise ValueError(msg)

        if any(
            right <= left
            for left, right in zip(self.values_gv, self.values_gv[1:])
        ):
            msg = "Rigidity grid must be strictly increasing."
            raise ValueError(msg)


def linear_rigidity_grid(
    min_rigidity_gv: float,
    max_rigidity_gv: float,
    point_count: int,
) -> RigidityGrid:
    """Create a linear magnetic rigidity grid in GV."""

    if not isfinite(min_rigidity_gv) or not isfinite(max_rigidity_gv):
        msg = "Rigidity grid bounds must be finite."
        raise ValueError(msg)

    if min_rigidity_gv <= 0.0:
        msg = "Minimum rigidity must be positive."
        raise ValueError(msg)

    if max_rigidity_gv <= min_rigidity_gv:
        msg = "Maximum rigidity must be greater than minimum rigidity."
        raise ValueError(msg)

    if not isinstance(point_count, int):
        msg = "Rigidity point count must be an integer."
        raise ValueError(msg)

    if point_count < 2:
        msg = "Rigidity point count must be at least 2."
        raise ValueError(msg)

    step = (max_rigidity_gv - min_rigidity_gv) / (point_count - 1)

    return RigidityGrid(
        values_gv=tuple(
            min_rigidity_gv + index * step
            for index in range(point_count)
        )
    )
