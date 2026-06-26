"""GCR kinematics and unit-conversion helpers."""

from __future__ import annotations

from math import isfinite, pi, sqrt
from typing import Final

from radar.physics.particles import PROTON_REST_ENERGY_MEV
from radar.physics.rigidity import ion_kinetic_energy_per_nucleon_to_rigidity_gv

MEV_PER_GEV: Final[float] = 1000.0
OMNIDIRECTIONAL_4PI_FACTOR: Final[float] = 4.0 * pi

MAIN_GCR_DIRECTIONAL_M2_GEV_TO_OMNI_CM2_MEV_FACTOR: Final[float] = (
    OMNIDIRECTIONAL_4PI_FACTOR * 1.0e-7
)
LOW_ENERGY_DIRECTIONAL_CM2_MEV_TO_OMNI_CM2_MEV_FACTOR: Final[float] = (
    OMNIDIRECTIONAL_4PI_FACTOR
)


def _validate_non_negative_finite(
    *,
    value: float,
    name: str,
) -> None:
    if not isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)

    if value < 0.0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)


def gcr_rigidity_gv_from_mev_per_nucleon(
    kinetic_energy_mev_per_nucleon: float,
    mass_to_charge: float,
) -> float:
    """Return magnetic rigidity in GV for GCR energy in MeV/nucleon.

    The mass_to_charge argument is A/Z. For protons A/Z = 1.
    """

    return ion_kinetic_energy_per_nucleon_to_rigidity_gv(
        kinetic_energy_mev_per_nucleon=kinetic_energy_mev_per_nucleon,
        mass_to_charge=mass_to_charge,
    )


def gcr_beta_from_mev_per_nucleon(
    kinetic_energy_mev_per_nucleon: float,
) -> float:
    """Return v/c for kinetic energy in MeV/nucleon."""

    _validate_non_negative_finite(
        value=kinetic_energy_mev_per_nucleon,
        name="GCR kinetic energy per nucleon",
    )

    if kinetic_energy_mev_per_nucleon == 0.0:
        return 0.0

    numerator = sqrt(
        kinetic_energy_mev_per_nucleon
        * (kinetic_energy_mev_per_nucleon + 2.0 * PROTON_REST_ENERGY_MEV)
    )
    denominator = kinetic_energy_mev_per_nucleon + PROTON_REST_ENERGY_MEV

    return numerator / denominator


def mev_per_nucleon_to_gev_per_nucleon(
    kinetic_energy_mev_per_nucleon: float,
) -> float:
    """Convert MeV/nucleon to GeV/nucleon."""

    _validate_non_negative_finite(
        value=kinetic_energy_mev_per_nucleon,
        name="GCR kinetic energy per nucleon",
    )

    return kinetic_energy_mev_per_nucleon / MEV_PER_GEV


def gev_per_nucleon_to_mev_per_nucleon(
    kinetic_energy_gev_per_nucleon: float,
) -> float:
    """Convert GeV/nucleon to MeV/nucleon."""

    _validate_non_negative_finite(
        value=kinetic_energy_gev_per_nucleon,
        name="GCR kinetic energy per nucleon",
    )

    return kinetic_energy_gev_per_nucleon * MEV_PER_GEV


def main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev(
    differential_flux_m2_sr_s_gev: float,
) -> float:
    """Convert main GCR directional flux to RADAR omnidirectional units.

    Input:
        m^-2 sr^-1 s^-1 (GeV/nucleon)^-1

    Output:
        cm^-2 s^-1 (MeV/nucleon)^-1

    The factor is 4*pi * 1e-7:
        4*pi for omnidirectional isotropic flux,
        1e-4 for m^-2 -> cm^-2,
        1e-3 for GeV^-1 -> MeV^-1.
    """

    _validate_non_negative_finite(
        value=differential_flux_m2_sr_s_gev,
        name="Main GCR differential flux",
    )

    return (
        differential_flux_m2_sr_s_gev
        * MAIN_GCR_DIRECTIONAL_M2_GEV_TO_OMNI_CM2_MEV_FACTOR
    )


def low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(
    differential_flux_cm2_sr_s_mev: float,
) -> float:
    """Convert NEK/AKL directional flux to RADAR omnidirectional units.

    Input:
        cm^-2 sr^-1 s^-1 (MeV/nucleon)^-1

    Output:
        cm^-2 s^-1 (MeV/nucleon)^-1

    The factor is 4*pi for omnidirectional isotropic flux.
    """

    _validate_non_negative_finite(
        value=differential_flux_cm2_sr_s_mev,
        name="Low-energy GCR differential flux",
    )

    return (
        differential_flux_cm2_sr_s_mev
        * LOW_ENERGY_DIRECTIONAL_CM2_MEV_TO_OMNI_CM2_MEV_FACTOR
    )


__all__ = [
    "LOW_ENERGY_DIRECTIONAL_CM2_MEV_TO_OMNI_CM2_MEV_FACTOR",
    "MAIN_GCR_DIRECTIONAL_M2_GEV_TO_OMNI_CM2_MEV_FACTOR",
    "MEV_PER_GEV",
    "OMNIDIRECTIONAL_4PI_FACTOR",
    "gcr_beta_from_mev_per_nucleon",
    "gcr_rigidity_gv_from_mev_per_nucleon",
    "gev_per_nucleon_to_mev_per_nucleon",
    "low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev",
    "main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev",
    "mev_per_nucleon_to_gev_per_nucleon",
]
