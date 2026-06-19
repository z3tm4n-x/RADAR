"""Magnetic rigidity conversions."""

from __future__ import annotations

from math import isfinite, sqrt

from radar.physics.particles import PROTON_CHARGE_NUMBER, PROTON_REST_ENERGY_MEV


def proton_kinetic_energy_to_rigidity_gv(kinetic_energy_mev: float) -> float:
    """Convert proton kinetic energy in MeV to magnetic rigidity in GV."""

    if not isfinite(kinetic_energy_mev):
        msg = "Proton kinetic energy must be finite."
        raise ValueError(msg)

    if kinetic_energy_mev < 0.0:
        msg = "Proton kinetic energy must be non-negative."
        raise ValueError(msg)

    momentum_mev_c = sqrt(
        kinetic_energy_mev**2
        + 2.0 * kinetic_energy_mev * PROTON_REST_ENERGY_MEV
    )

    return momentum_mev_c / (1000.0 * PROTON_CHARGE_NUMBER)


def proton_rigidity_to_kinetic_energy_mev(rigidity_gv: float) -> float:
    """Convert proton magnetic rigidity in GV to kinetic energy in MeV."""

    if not isfinite(rigidity_gv):
        msg = "Proton magnetic rigidity must be finite."
        raise ValueError(msg)

    if rigidity_gv < 0.0:
        msg = "Proton magnetic rigidity must be non-negative."
        raise ValueError(msg)

    momentum_mev_c = 1000.0 * rigidity_gv * PROTON_CHARGE_NUMBER

    return sqrt(momentum_mev_c**2 + PROTON_REST_ENERGY_MEV**2) - PROTON_REST_ENERGY_MEV
