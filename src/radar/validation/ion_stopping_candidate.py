"""Independent consistency checks for candidate ion stopping/range data.

These checks compare range *increments* with the integral of A/S_total over
the supplied energy-per-nucleon intervals. They neither establish the range
origin below the first node nor qualify a source for normative release.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import exp, expm1, isfinite, log, log1p
from numbers import Integral, Real


def _positive_number(value: float, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric value, not bool.")
    try:
        valid = isfinite(value) and (value >= 0 if allow_zero else value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")


def _ion(z: int, mass_number: int) -> None:
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (z, mass_number)):
        raise ValueError("Ion Z and isotope mass number A must be integers, not bool.")
    if z < 1 or mass_number < z:
        raise ValueError("Ion Z must be positive and isotope mass number A must be >= Z.")
    _positive_number(mass_number, "mass_number")


def _material(material_z: int) -> None:
    if (
        isinstance(material_z, bool)
        or not isinstance(material_z, Integral)
        or material_z not in (13, 14)
    ):
        raise ValueError("material_z must be integer 13 (Al) or 14 (Si).")


@dataclass(frozen=True)
class IonStoppingRow:
    """One isotope/material node; all stopping columns refer to the whole ion."""

    z: int
    mass_number: int
    material_z: int
    energy_mev_per_nucleon: float
    stopping_electronic_mev_cm2_g: float
    stopping_nuclear_mev_cm2_g: float
    range_g_cm2: float
    stopping_radiative_mev_cm2_g: float = 0.0

    def __post_init__(self) -> None:
        _ion(self.z, self.mass_number)
        _material(self.material_z)
        for name in (
            "energy_mev_per_nucleon", "stopping_electronic_mev_cm2_g", "range_g_cm2"
        ):
            _positive_number(getattr(self, name), name)
        _positive_number(
            self.stopping_nuclear_mev_cm2_g, "stopping_nuclear_mev_cm2_g", allow_zero=True
        )
        _positive_number(
            self.stopping_radiative_mev_cm2_g, "stopping_radiative_mev_cm2_g", allow_zero=True
        )
        _positive_number(self.stopping_total_mev_cm2_g, "total stopping")

    @property
    def stopping_total_mev_cm2_g(self) -> float:
        return (
            self.stopping_electronic_mev_cm2_g + self.stopping_nuclear_mev_cm2_g
            + self.stopping_radiative_mev_cm2_g
        )

    @property
    def let_electronic_mev_cm2_mg(self) -> float:
        return self.stopping_electronic_mev_cm2_g / 1000.0


@dataclass(frozen=True)
class IonStoppingInterval:
    z: int
    mass_number: int
    material_z: int
    lower_energy_mev_per_nucleon: float
    upper_energy_mev_per_nucleon: float
    tabulated_delta_range_g_cm2: float
    integrated_delta_range_g_cm2: float
    relative_discrepancy: float
    within_tolerance: bool


@dataclass(frozen=True)
class IonStoppingValidation:
    row_count: int
    group_count: int
    relative_tolerance: float
    intervals: tuple[IonStoppingInterval, ...]
    max_relative_discrepancy: float
    numerical_consistency_passed: bool
    normative_export_allowed: bool = field(default=False, init=False)
    absolute_range_origin_verified: bool = field(default=False, init=False)
    interpolation: str = field(default="piecewise_log_log_total_stopping", init=False)


def _integrated_increment(left: IonStoppingRow, right: IonStoppingRow) -> float:
    # S(e)=S0*(e/e0)**b, hence deltaR=A*e0/S0*expm1((1-b)*log(e1/e0))/(1-b).
    # Work in logs to avoid overflow in intermediate powers and products.
    log_ratio = _log_ratio(right.energy_mev_per_nucleon, left.energy_mev_per_nucleon)
    x = log_ratio - _log_ratio(
        right.stopping_total_mev_cm2_g, left.stopping_total_mev_cm2_g
    )
    if x == 0:
        log_exprel = 0.0
    elif x > 50:
        log_exprel = x + log(-expm1(-x)) - log(x)
    else:
        log_exprel = log(expm1(x) / x)
    try:
        result = exp(
            log(left.mass_number) + log(left.energy_mev_per_nucleon)
            - log(left.stopping_total_mev_cm2_g) + log(log_ratio) + log_exprel
        )
    except (OverflowError, ValueError) as exc:
        raise ValueError("Range increment cannot be represented as a positive finite float.") from exc
    _positive_number(result, "integrated range increment")
    return result


def _log_ratio(numerator: float, denominator: float) -> float:
    relative_difference = (numerator - denominator) / denominator
    if isfinite(relative_difference) and relative_difference > -1:
        return log1p(relative_difference)
    return log(numerator) - log(denominator)


def validate_candidate_rows(
    rows: Sequence[IonStoppingRow],
    *,
    required_ions: Sequence[tuple[int, int]],
    material_atomic_numbers: Sequence[int],
    energy_nodes_mev_per_nucleon: Sequence[float],
    relative_tolerance: float,
) -> IonStoppingValidation:
    """Require exact rectangular coverage and report a bounded numerical check.

    Missing/extra/duplicate nodes or nonmonotone ranges are errors. A discrepancy
    above tolerance is returned as a failed check, not repaired or suppressed.
    The interval relative discrepancy uses the integrated increment as denominator.
    A constant offset in every range is invisible to this check by design.
    """
    _positive_number(relative_tolerance, "relative_tolerance")
    if relative_tolerance >= 1:
        raise ValueError("relative_tolerance must be less than one.")
    ions = tuple(required_ions)
    materials = tuple(material_atomic_numbers)
    energies = tuple(energy_nodes_mev_per_nucleon)
    for z, mass_number in ions:
        _ion(z, mass_number)
    for material in materials:
        _material(material)
    for energy in energies:
        _positive_number(energy, "energy node")
    if not ions or len(set(ions)) != len(ions):
        raise ValueError("Required ions must be nonempty and unique.")
    if not materials or len(set(materials)) != len(materials):
        raise ValueError("Required materials must be nonempty and unique.")
    if len(energies) < 2 or any(a >= b for a, b in zip(energies, energies[1:])):
        raise ValueError("At least two strictly increasing energy nodes are required.")
    expected_groups = {(z, a, m) for z, a in ions for m in materials}
    grouped: dict[tuple[int, int, int], list[IonStoppingRow]] = {}
    for row in rows:
        if not isinstance(row, IonStoppingRow):
            raise ValueError("All rows must be IonStoppingRow instances.")
        key = (row.z, row.mass_number, row.material_z)
        grouped.setdefault(key, []).append(row)
    if set(grouped) != expected_groups:
        raise ValueError("Missing or unexpected isotope/material groups in candidate rows.")
    intervals: list[IonStoppingInterval] = []
    for key in sorted(grouped):
        group = grouped[key]
        if tuple(row.energy_mev_per_nucleon for row in group) != energies:
            raise ValueError(f"Missing, extra, duplicate or unsorted energy nodes for {key}.")
        for left, right in zip(group, group[1:]):
            observed = right.range_g_cm2 - left.range_g_cm2
            _positive_number(observed, "tabulated range increment (range must increase)")
            expected = _integrated_increment(left, right)
            discrepancy = abs(observed / expected - 1.0)
            _positive_number(discrepancy, "relative discrepancy", allow_zero=True)
            intervals.append(IonStoppingInterval(
                *key, left.energy_mev_per_nucleon, right.energy_mev_per_nucleon,
                observed, expected, discrepancy, discrepancy <= relative_tolerance,
            ))
    maximum = max(interval.relative_discrepancy for interval in intervals)
    return IonStoppingValidation(
        len(rows), len(grouped), relative_tolerance, tuple(intervals), maximum,
        all(interval.within_tolerance for interval in intervals),
    )
