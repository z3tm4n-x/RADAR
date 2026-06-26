from math import inf, nan, pi, sqrt

import pytest

from radar.gcr.kinematics import (
    LOW_ENERGY_DIRECTIONAL_CM2_MEV_TO_OMNI_CM2_MEV_FACTOR,
    MAIN_GCR_DIRECTIONAL_M2_GEV_TO_OMNI_CM2_MEV_FACTOR,
    MEV_PER_GEV,
    OMNIDIRECTIONAL_4PI_FACTOR,
    gcr_beta_from_mev_per_nucleon,
    gcr_rigidity_gv_from_mev_per_nucleon,
    gev_per_nucleon_to_mev_per_nucleon,
    low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev,
    main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev,
    mev_per_nucleon_to_gev_per_nucleon,
)
from radar.physics.particles import PROTON_REST_ENERGY_MEV
from radar.physics.rigidity import proton_kinetic_energy_to_rigidity_gv


def test_gcr_rigidity_matches_existing_proton_conversion_for_a_over_z_one() -> None:
    assert gcr_rigidity_gv_from_mev_per_nucleon(
        kinetic_energy_mev_per_nucleon=1000.0,
        mass_to_charge=1.0,
    ) == pytest.approx(proton_kinetic_energy_to_rigidity_gv(1000.0))


def test_gcr_rigidity_scales_with_mass_to_charge() -> None:
    proton_like = gcr_rigidity_gv_from_mev_per_nucleon(
        kinetic_energy_mev_per_nucleon=1000.0,
        mass_to_charge=1.0,
    )
    helium_like = gcr_rigidity_gv_from_mev_per_nucleon(
        kinetic_energy_mev_per_nucleon=1000.0,
        mass_to_charge=2.0,
    )

    assert helium_like == pytest.approx(2.0 * proton_like)


def test_gcr_rigidity_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        gcr_rigidity_gv_from_mev_per_nucleon(
            kinetic_energy_mev_per_nucleon=-1.0,
            mass_to_charge=1.0,
        )

    with pytest.raises(ValueError, match="mass-to-charge"):
        gcr_rigidity_gv_from_mev_per_nucleon(
            kinetic_energy_mev_per_nucleon=1.0,
            mass_to_charge=0.0,
        )


def test_gcr_beta_matches_relativistic_formula() -> None:
    energy = 1000.0
    expected = sqrt(energy * (energy + 2.0 * PROTON_REST_ENERGY_MEV)) / (
        energy + PROTON_REST_ENERGY_MEV
    )

    assert gcr_beta_from_mev_per_nucleon(energy) == pytest.approx(expected)


def test_gcr_beta_is_zero_at_zero_energy_and_below_one_for_positive_energy() -> None:
    assert gcr_beta_from_mev_per_nucleon(0.0) == pytest.approx(0.0)
    assert 0.0 < gcr_beta_from_mev_per_nucleon(1.0) < 1.0
    assert 0.0 < gcr_beta_from_mev_per_nucleon(100000.0) < 1.0


@pytest.mark.parametrize("bad_energy", [-1.0, inf, nan])
def test_gcr_beta_rejects_bad_energy(bad_energy: float) -> None:
    with pytest.raises(ValueError, match="finite|non-negative"):
        gcr_beta_from_mev_per_nucleon(bad_energy)


def test_gcr_energy_unit_conversions_are_inverse() -> None:
    assert MEV_PER_GEV == pytest.approx(1000.0)
    assert mev_per_nucleon_to_gev_per_nucleon(2500.0) == pytest.approx(2.5)
    assert gev_per_nucleon_to_mev_per_nucleon(2.5) == pytest.approx(2500.0)

    energy_mev = 1234.5
    recovered = gev_per_nucleon_to_mev_per_nucleon(
        mev_per_nucleon_to_gev_per_nucleon(energy_mev),
    )

    assert recovered == pytest.approx(energy_mev)


@pytest.mark.parametrize("bad_energy", [-1.0, inf, nan])
def test_energy_unit_conversions_reject_bad_energy(bad_energy: float) -> None:
    with pytest.raises(ValueError, match="finite|non-negative"):
        mev_per_nucleon_to_gev_per_nucleon(bad_energy)

    with pytest.raises(ValueError, match="finite|non-negative"):
        gev_per_nucleon_to_mev_per_nucleon(bad_energy)


def test_main_gcr_conversion_factor_is_four_pi_times_one_e_minus_seven() -> None:
    assert OMNIDIRECTIONAL_4PI_FACTOR == pytest.approx(4.0 * pi)
    assert MAIN_GCR_DIRECTIONAL_M2_GEV_TO_OMNI_CM2_MEV_FACTOR == pytest.approx(
        4.0 * pi * 1.0e-7,
    )
    assert main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev(
        1.0,
    ) == pytest.approx(4.0 * pi * 1.0e-7)


def test_low_energy_conversion_factor_is_four_pi() -> None:
    assert LOW_ENERGY_DIRECTIONAL_CM2_MEV_TO_OMNI_CM2_MEV_FACTOR == pytest.approx(
        4.0 * pi,
    )
    assert low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(
        1.0,
    ) == pytest.approx(4.0 * pi)


@pytest.mark.parametrize("bad_flux", [-1.0, inf, nan])
def test_flux_conversions_reject_bad_flux(bad_flux: float) -> None:
    with pytest.raises(ValueError, match="finite|non-negative"):
        main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev(bad_flux)

    with pytest.raises(ValueError, match="finite|non-negative"):
        low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(bad_flux)
