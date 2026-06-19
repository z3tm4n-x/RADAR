import math

import pytest

from radar.physics.particles import PROTON_CHARGE_NUMBER, PROTON_REST_ENERGY_MEV
from radar.physics.rigidity import (
    proton_kinetic_energy_to_rigidity_gv,
    proton_rigidity_to_kinetic_energy_mev,
)


def test_proton_constants_are_positive() -> None:
    assert PROTON_REST_ENERGY_MEV > 0.0
    assert PROTON_CHARGE_NUMBER == 1


def test_zero_energy_has_zero_rigidity() -> None:
    assert proton_kinetic_energy_to_rigidity_gv(0.0) == pytest.approx(0.0)
    assert proton_rigidity_to_kinetic_energy_mev(0.0) == pytest.approx(0.0)


def test_proton_energy_to_rigidity_known_value() -> None:
    rigidity_gv = proton_kinetic_energy_to_rigidity_gv(1000.0)

    assert rigidity_gv == pytest.approx(1.6960377, rel=1e-7)


def test_proton_rigidity_to_energy_known_value() -> None:
    kinetic_energy_mev = proton_rigidity_to_kinetic_energy_mev(1.0)

    assert kinetic_energy_mev == pytest.approx(432.98810500566026, rel=1e-6)


@pytest.mark.parametrize(
    "kinetic_energy_mev",
    [
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    ],
)
def test_proton_energy_rigidity_roundtrip(kinetic_energy_mev: float) -> None:
    rigidity_gv = proton_kinetic_energy_to_rigidity_gv(kinetic_energy_mev)
    recovered_energy_mev = proton_rigidity_to_kinetic_energy_mev(rigidity_gv)

    assert recovered_energy_mev == pytest.approx(kinetic_energy_mev, rel=1e-12)


@pytest.mark.parametrize(
    "rigidity_gv",
    [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ],
)
def test_proton_rigidity_energy_roundtrip(rigidity_gv: float) -> None:
    kinetic_energy_mev = proton_rigidity_to_kinetic_energy_mev(rigidity_gv)
    recovered_rigidity_gv = proton_kinetic_energy_to_rigidity_gv(kinetic_energy_mev)

    assert recovered_rigidity_gv == pytest.approx(rigidity_gv, rel=1e-12)


def test_proton_energy_to_rigidity_rejects_negative_energy() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        proton_kinetic_energy_to_rigidity_gv(-1.0)


def test_proton_rigidity_to_energy_rejects_negative_rigidity() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        proton_rigidity_to_kinetic_energy_mev(-1.0)


def test_proton_energy_to_rigidity_rejects_non_finite_energy() -> None:
    with pytest.raises(ValueError, match="finite"):
        proton_kinetic_energy_to_rigidity_gv(math.inf)


def test_proton_rigidity_to_energy_rejects_non_finite_rigidity() -> None:
    with pytest.raises(ValueError, match="finite"):
        proton_rigidity_to_kinetic_energy_mev(math.nan)
