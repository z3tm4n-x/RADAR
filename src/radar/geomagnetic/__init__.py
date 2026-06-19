"""Geomagnetic calculation utilities."""

from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.rigidity import RigidityGrid, linear_rigidity_grid
from radar.geomagnetic.spectrum import apply_proton_penetration, proton_penetration_factors

__all__ = [
    "PenetrationFunction",
    "RigidityGrid",
    "apply_proton_penetration",
    "linear_rigidity_grid",
    "proton_penetration_factors",
]
