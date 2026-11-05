"""Analytic checks for fractional projectile mass diagnostics, not isotope claims."""

from collections.abc import Callable
from dataclasses import asdict, replace
from math import inf, log, nextafter
from typing import Any

import pytest

from radar.validation.parameterized_stopping import (
    ParameterizedStoppingRow,
    ParameterizedStoppingValidation,
    validate_parameterized_rows,
)


ENERGIES = (1.0, 2.0, 4.0, 10.0)
PROJECTILES = ((10, 20.2), (26, 55.8))


def _rows(power: float = 2.0) -> tuple[ParameterizedStoppingRow, ...]:
    result = []
    for z, mass in PROJECTILES:
        for material in (13, 14):
            for energy in ENERGIES:
                stopping = (
                    mass / (3 * power * energy ** (power - 1)) if power else mass * energy / 3
                )
                result.append(ParameterizedStoppingRow(
                    z, mass, material, energy, 0.7 * stopping, 0.2 * stopping,
                    3 * energy**power if power else 3 * log(energy) + 7,
                    0.1 * stopping,
                ))
    return tuple(result)


def _validate(
    rows: tuple[ParameterizedStoppingRow, ...], **changes: Any,
) -> ParameterizedStoppingValidation:
    arguments: dict[str, Any] = dict(
        required_projectiles=PROJECTILES, material_atomic_numbers=(13, 14),
        energy_nodes_mev_per_nucleon=ENERGIES, relative_tolerance=1e-12,
    )
    arguments.update(changes)
    return validate_parameterized_rows(rows, **arguments)


@pytest.mark.parametrize("power", [0.0, 0.25, 1.0, 2.0, 4.0, 100.0])
def test_fractional_mass_analytic_power_laws(power: float) -> None:
    report = _validate(_rows(power))
    assert report.row_count == 16
    assert report.group_count == 4
    assert len(report.groups) == 4
    assert all(group.interval_count == 3 for group in report.groups)
    assert report.numerical_consistency_passed
    assert report.max_relative_discrepancy < 1e-12
    assert not report.normative_export_allowed
    assert not report.absolute_range_origin_verified
    assert not report.isotope_identification_established
    assert "intervals" not in asdict(report)
    assert report.interpolation == "piecewise_log_log_total_stopping"
    for group in report.groups:
        assert group.integrated_delta_range_g_cm2 == pytest.approx(group.tabulated_delta_range_g_cm2)
        assert group.worst_interval.within_tolerance


def test_stopping_is_whole_projectile_but_let_excludes_nuclear_and_radiative() -> None:
    row = ParameterizedStoppingRow(26, 55.8, 14, 10, 1200, 300, 1, 500)
    assert row.stopping_total_mev_cm2_g == 2000
    assert row.let_electronic_mev_cm2_mg == 1.2


def test_mass_factor_is_not_rounded_or_dropped() -> None:
    original = _rows()
    rounded = tuple(replace(row, mass_parameter_u=round(row.mass_parameter_u)) for row in original)
    report = _validate(rounded, required_projectiles=((10, 20.0), (26, 56.0)))
    assert not report.numerical_consistency_passed
    neon = report.groups[0]
    assert neon.max_relative_discrepancy == pytest.approx(abs(20.2 / 20.0 - 1))
    wrong_whole_energy = tuple(replace(
        row,
        stopping_electronic_mev_cm2_g=row.stopping_electronic_mev_cm2_g / row.mass_parameter_u,
        stopping_nuclear_mev_cm2_g=row.stopping_nuclear_mev_cm2_g / row.mass_parameter_u,
        stopping_radiative_mev_cm2_g=row.stopping_radiative_mev_cm2_g / row.mass_parameter_u,
    ) for row in original)
    assert not _validate(wrong_whole_energy).numerical_consistency_passed


def test_radiative_stopping_belongs_in_range_not_electronic_let() -> None:
    original = _rows()
    changed = tuple(replace(row, stopping_radiative_mev_cm2_g=0.0) for row in original)
    assert changed[0].let_electronic_mev_cm2_mg == original[0].let_electronic_mev_cm2_mg
    assert not _validate(changed).numerical_consistency_passed


@pytest.mark.parametrize("field", [
    "mass_parameter_u", "energy_mev_per_nucleon", "stopping_electronic_mev_cm2_g", "range_g_cm2",
])
@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"), "1"])
def test_reject_invalid_positive_fields(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        replace(_rows()[0], **{field: value})


@pytest.mark.parametrize("field", ["stopping_nuclear_mev_cm2_g", "stopping_radiative_mev_cm2_g"])
@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "0"])
def test_reject_invalid_nonnegative_fields(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        replace(_rows()[0], **{field: value})


@pytest.mark.parametrize("changes", [
    {"z": True}, {"z": 1}, {"z": 93}, {"z": 10.0}, {"mass_parameter_u": 9.9},
    {"material_z": 12}, {"material_z": 13.0}, {"material_z": True},
])
def test_reject_invalid_projectile_or_material(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(_rows()[0], **changes)


def test_allow_zero_other_stopping_but_reject_overflowed_total() -> None:
    row = replace(_rows()[0], stopping_nuclear_mev_cm2_g=0.0, stopping_radiative_mev_cm2_g=0.0)
    assert row.stopping_total_mev_cm2_g == row.stopping_electronic_mev_cm2_g
    with pytest.raises(ValueError, match="total stopping"):
        replace(row, stopping_electronic_mev_cm2_g=1e308, stopping_nuclear_mev_cm2_g=1e308)


@pytest.mark.parametrize("transform", [
    lambda rows: rows[1:],
    lambda rows: (rows[0],) + rows,
    lambda rows: (rows[1], rows[0]) + rows[2:],
    lambda rows: tuple(replace(row, mass_parameter_u=56.0) if row.z == 26 else row for row in rows),
    lambda rows: tuple(row for row in rows if row.material_z == 13),
])
def test_reject_missing_duplicate_unsorted_or_wrong_mass_rows(
    transform: Callable[[tuple[ParameterizedStoppingRow, ...]], tuple[ParameterizedStoppingRow, ...]],
) -> None:
    with pytest.raises(ValueError, match="Missing"):
        _validate(transform(_rows()))


@pytest.mark.parametrize("second_range", [3.0, 2.0])
def test_reject_nonincreasing_range(second_range: float) -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="range must increase"):
        _validate((rows[0], replace(rows[1], range_g_cm2=second_range)) + rows[2:])


def test_bad_consistency_is_reported_not_repaired() -> None:
    rows = _rows()
    changed = tuple(replace(row, range_g_cm2=row.range_g_cm2 * 1.2) for row in rows)
    report = _validate(changed)
    assert not report.numerical_consistency_passed
    assert report.max_relative_discrepancy == pytest.approx(0.2)
    assert changed[0].range_g_cm2 == rows[0].range_g_cm2 * 1.2
    assert all(not group.worst_interval.within_tolerance for group in report.groups)


def test_range_offset_cannot_establish_absolute_origin() -> None:
    shifted = tuple(replace(row, range_g_cm2=row.range_g_cm2 + 37) for row in _rows())
    report = _validate(shifted)
    assert report.numerical_consistency_passed
    assert not report.absolute_range_origin_verified


def test_nonmonotone_stopping_with_increasing_range_is_valid() -> None:
    mass = 20.2
    rows = (
        ParameterizedStoppingRow(10, mass, 14, 1, 1, 0, 1),
        ParameterizedStoppingRow(10, mass, 14, 2, 2, 0, 1 + mass * log(2)),
        ParameterizedStoppingRow(10, mass, 14, 4, 1, 0, 1 + mass * log(2) + 1.5 * mass),
    )
    report = _validate(rows, required_projectiles=((10, mass),), material_atomic_numbers=(14,),
                       energy_nodes_mev_per_nucleon=(1, 2, 4))
    assert report.numerical_consistency_passed


def test_adjacent_float_energies_are_not_collapsed() -> None:
    energies = (1.0, nextafter(1.0, inf))
    rows = tuple(ParameterizedStoppingRow(10, 20.2, 13, e, 20.2, 0, e) for e in energies)
    report = _validate(rows, required_projectiles=((10, 20.2),), material_atomic_numbers=(13,),
                       energy_nodes_mev_per_nucleon=energies)
    assert report.numerical_consistency_passed


def test_negative_integral_exponent_is_supported() -> None:
    rows = tuple(
        ParameterizedStoppingRow(10, 20.2, 13, energy, energy**2, 0, 30 - 20.2 / energy)
        for energy in ENERGIES
    )
    assert _validate(
        rows, required_projectiles=((10, 20.2),), material_atomic_numbers=(13,),
    ).numerical_consistency_passed


@pytest.mark.parametrize("changes", [
    {"required_projectiles": ()}, {"required_projectiles": ((10, 20.2), (10, 20.2))},
    {"required_projectiles": ((10,),)}, {"required_projectiles": ((10, 20.2, 1),)},
    {"required_projectiles": ((True, 20.2),)}, {"required_projectiles": ((10, True),)},
    {"material_atomic_numbers": ()}, {"material_atomic_numbers": (13, 13)},
    {"energy_nodes_mev_per_nucleon": (1,)}, {"energy_nodes_mev_per_nucleon": (1, 1)},
    {"energy_nodes_mev_per_nucleon": (2, 1)}, {"energy_nodes_mev_per_nucleon": (0, 1)},
    {"relative_tolerance": True}, {"relative_tolerance": 0},
    {"relative_tolerance": float("nan")}, {"relative_tolerance": 1},
])
def test_invalid_matrix_contract(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _validate(_rows(), **changes)


def test_nonrow_object_is_rejected() -> None:
    with pytest.raises(ValueError, match="All rows"):
        _validate(({},))  # type: ignore[arg-type]


@pytest.mark.parametrize("name", [
    "normative_export_allowed", "absolute_range_origin_verified", "isotope_identification_established",
])
def test_acceptance_flags_cannot_be_changed_through_constructor(name: str) -> None:
    changes: dict[str, Any] = {name: True}
    with pytest.raises((ValueError, TypeError)):
        replace(_validate(_rows()), **changes)
