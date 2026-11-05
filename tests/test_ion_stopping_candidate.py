"""Analytic checks independent of the external stopping-library candidate."""

from dataclasses import asdict, replace
from collections.abc import Callable
from math import inf, log, nextafter
from typing import Any

import pytest

from radar.validation.ion_stopping_candidate import (
    IonStoppingRow, IonStoppingValidation, validate_candidate_rows,
)


ENERGIES = (1.0, 2.0, 4.0, 10.0)


def power_rows(power: float = 2.0) -> tuple[IonStoppingRow, ...]:
    # R(e)=3*e**power (or 3*log(e)+7), and S_total=A/(dR/de).
    return tuple(
        IonStoppingRow(
            z, a, material, energy,
            (a / (3 * power * energy ** (power - 1)) if power else a * energy / 3) * 0.9,
            (a / (3 * power * energy ** (power - 1)) if power else a * energy / 3) * 0.1,
            3 * energy ** power if power else 3 * log(energy) + 7,
        )
        for z, a in ((2, 4), (26, 56))
        for material in (13, 14)
        for energy in ENERGIES
    )


def validate(rows: tuple[IonStoppingRow, ...], **kwargs: Any) -> IonStoppingValidation:
    options: dict[str, Any] = dict(
        required_ions=((2, 4), (26, 56)), material_atomic_numbers=(13, 14),
        energy_nodes_mev_per_nucleon=ENERGIES, relative_tolerance=1e-12,
    )
    options.update(kwargs)
    return validate_candidate_rows(rows, **options)


@pytest.mark.parametrize("power", [0.0, 0.25, 1.0, 2.0, 4.0])
def test_analytic_range_integrals(power: float) -> None:
    report = validate(power_rows(power))
    assert report.row_count == 16
    assert report.group_count == 4
    assert len(report.intervals) == 12
    assert report.max_relative_discrepancy < 4e-15
    assert report.numerical_consistency_passed
    assert not report.normative_export_allowed
    assert not report.absolute_range_origin_verified
    assert asdict(report)["normative_export_allowed"] is False
    assert report.interpolation == "piecewise_log_log_total_stopping"


def test_whole_ion_stopping_and_electronic_let_are_distinct() -> None:
    row = IonStoppingRow(26, 56, 14, 10, 1200, 300, 1, 500)
    assert row.stopping_total_mev_cm2_g == 2000
    assert row.let_electronic_mev_cm2_mg == 1.2


def test_range_increment_includes_radiative_stopping_but_electronic_let_does_not() -> None:
    rows = tuple(
        replace(row,
                stopping_electronic_mev_cm2_g=row.stopping_electronic_mev_cm2_g / 2,
                stopping_radiative_mev_cm2_g=row.stopping_electronic_mev_cm2_g / 2)
        for row in power_rows()
    )
    assert validate(rows).numerical_consistency_passed
    assert rows[0].let_electronic_mev_cm2_mg == power_rows()[0].let_electronic_mev_cm2_mg / 2
    assert not validate(tuple(replace(row, stopping_radiative_mev_cm2_g=0)
                              for row in rows)).numerical_consistency_passed


@pytest.mark.parametrize("field", ["energy_mev_per_nucleon", "range_g_cm2",
                                    "stopping_electronic_mev_cm2_g"])
@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"), "1"])
def test_invalid_positive_fields(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        replace(power_rows()[0], **{field: value})


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "0"])
def test_invalid_nuclear_stopping(value: Any) -> None:
    with pytest.raises(ValueError):
        replace(power_rows()[0], stopping_nuclear_mev_cm2_g=value)


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "0"])
def test_invalid_radiative_stopping(value: Any) -> None:
    with pytest.raises(ValueError):
        replace(power_rows()[0], stopping_radiative_mev_cm2_g=value)


def test_zero_nuclear_stopping_is_allowed_but_overflowed_total_is_not() -> None:
    assert replace(power_rows()[0], stopping_nuclear_mev_cm2_g=0).stopping_nuclear_mev_cm2_g == 0
    with pytest.raises(ValueError, match="total stopping"):
        IonStoppingRow(2, 4, 13, 1, 1e308, 1e308, 1)


@pytest.mark.parametrize("changes", [
    {"z": True}, {"z": 0}, {"z": 2.0}, {"mass_number": 1}, {"mass_number": 4.1},
    {"mass_number": True}, {"material_z": 12}, {"material_z": 13.0}, {"material_z": True},
])
def test_invalid_ion_or_material(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(power_rows()[0], **changes)


@pytest.mark.parametrize("transform", [
    lambda rows: rows[1:],
    lambda rows: (rows[0],) + rows,
    lambda rows: (rows[1], rows[0]) + rows[2:],
    lambda rows: tuple(replace(row, mass_number=58) if row.z == 26 else row for row in rows),
    lambda rows: tuple(row for row in rows if row.material_z == 13),
])
def test_missing_duplicate_unsorted_or_wrong_isotope_is_error(
    transform: Callable[[tuple[IonStoppingRow, ...]], tuple[IonStoppingRow, ...]],
) -> None:
    with pytest.raises(ValueError, match="Missing"):
        validate(transform(power_rows()))


@pytest.mark.parametrize("second_range", [3.0, 2.0])
def test_nonmonotone_range_is_error(second_range: float) -> None:
    rows = power_rows()
    with pytest.raises(ValueError, match="range must increase"):
        validate((rows[0], replace(rows[1], range_g_cm2=second_range)) + rows[2:])


@pytest.mark.parametrize("scale", [1 / 1000, 1000, 1 / 56])
def test_unit_or_mass_normalization_error_fails_consistency(scale: float) -> None:
    rows = tuple(replace(row, stopping_electronic_mev_cm2_g=row.stopping_electronic_mev_cm2_g*scale,
                         stopping_nuclear_mev_cm2_g=row.stopping_nuclear_mev_cm2_g*scale)
                 for row in power_rows())
    report = validate(rows)
    assert not report.numerical_consistency_passed
    assert report.max_relative_discrepancy == pytest.approx(abs(scale - 1))
    assert not report.normative_export_allowed


def test_constant_range_offset_cannot_be_verified_by_increment_check() -> None:
    report = validate(tuple(replace(row, range_g_cm2=row.range_g_cm2 + 20) for row in power_rows()))
    assert report.numerical_consistency_passed
    assert not report.absolute_range_origin_verified


def test_nonmonotone_stopping_is_not_rejected_as_nonphysical() -> None:
    # Piecewise S=1,2,1 has a peak. Integrate A/S analytically on each interval.
    rows = (
        IonStoppingRow(2, 4, 14, 1, 1, 0, 1),
        IonStoppingRow(2, 4, 14, 2, 2, 0, 1 + 4 * log(2)),
        IonStoppingRow(2, 4, 14, 4, 1, 0, 7 + 4 * log(2)),
    )
    report = validate(rows, required_ions=((2, 4),), material_atomic_numbers=(14,),
                      energy_nodes_mev_per_nucleon=(1, 2, 4))
    assert report.numerical_consistency_passed


def test_steep_power_law_uses_stable_large_exponent_integral() -> None:
    assert validate(power_rows(100)).numerical_consistency_passed


def test_reciprocal_range_law_checks_negative_integral_exponent() -> None:
    rows = tuple(IonStoppingRow(2, 4, 13, e, e*e, 0, 10 - 4/e) for e in ENERGIES)
    assert validate(rows, required_ions=((2, 4),), material_atomic_numbers=(13,)).numerical_consistency_passed


def test_adjacent_float_energy_nodes_are_not_collapsed_by_log_subtraction() -> None:
    energies = (1.0, nextafter(1.0, inf))
    rows = tuple(IonStoppingRow(2, 4, 13, e, 4, 0, e) for e in energies)
    report = validate(rows, required_ions=((2, 4),), material_atomic_numbers=(13,),
                      energy_nodes_mev_per_nucleon=energies)
    assert report.numerical_consistency_passed


@pytest.mark.parametrize("changes", [
    {"required_ions": ()}, {"required_ions": ((2, 4), (2, 4))},
    {"material_atomic_numbers": ()}, {"material_atomic_numbers": (13, 13)},
    {"energy_nodes_mev_per_nucleon": (1,)}, {"energy_nodes_mev_per_nucleon": (1, 1)},
    {"energy_nodes_mev_per_nucleon": (2, 1)}, {"energy_nodes_mev_per_nucleon": (0, 1)},
    {"relative_tolerance": True}, {"relative_tolerance": 0},
    {"relative_tolerance": float("nan")}, {"relative_tolerance": 1},
])
def test_invalid_matrix_contract(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate(power_rows(), **changes)


def test_report_flags_cannot_be_changed_through_constructor() -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(validate(power_rows()), normative_export_allowed=True)
