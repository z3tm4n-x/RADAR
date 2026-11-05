"""Bounded consistency checks for explicitly parameterized projectile masses.

A possibly fractional mass parameter is not an isotope identification. The
dataset must state its mass policy separately. This diagnostic checks only
range increments against the integral of mass_parameter/S_total over energy
per nucleon; no absolute range origin or normative qualification follows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import exp, expm1, fsum, isfinite, log, log1p
from numbers import Integral, Real


def _number(value: float, name: str, *, zero_allowed: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not bool.")
    try:
        valid = isfinite(value) and (value >= 0 if zero_allowed else value > 0)
    except OverflowError:
        valid = False
    if not valid:
        domain = "nonnegative" if zero_allowed else "positive"
        raise ValueError(f"{name} must be finite and {domain}.")


def _projectile(z: int, mass_parameter_u: float) -> None:
    if isinstance(z, bool) or not isinstance(z, Integral) or not 2 <= z <= 92:
        raise ValueError("Projectile Z must be an integer from 2 through 92, not bool.")
    _number(mass_parameter_u, "mass_parameter_u")
    if mass_parameter_u < z:
        raise ValueError("mass_parameter_u must be >= Z.")


def _material(material_z: int) -> None:
    if (
        isinstance(material_z, bool) or not isinstance(material_z, Integral)
        or material_z not in (13, 14)
    ):
        raise ValueError("material_z must be integer 13 (Al) or 14 (Si).")


@dataclass(frozen=True)
class ParameterizedStoppingRow:
    """One declared mass/material node; stopping refers to the whole projectile."""

    z: int
    mass_parameter_u: float
    material_z: int
    energy_mev_per_nucleon: float
    stopping_electronic_mev_cm2_g: float
    stopping_nuclear_mev_cm2_g: float
    range_g_cm2: float
    stopping_radiative_mev_cm2_g: float = 0.0

    def __post_init__(self) -> None:
        _projectile(self.z, self.mass_parameter_u)
        _material(self.material_z)
        for name in (
            "energy_mev_per_nucleon", "stopping_electronic_mev_cm2_g", "range_g_cm2",
        ):
            _number(getattr(self, name), name)
        for name in ("stopping_nuclear_mev_cm2_g", "stopping_radiative_mev_cm2_g"):
            _number(getattr(self, name), name, zero_allowed=True)
        _number(self.stopping_total_mev_cm2_g, "total stopping")

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
class ParameterizedStoppingInterval:
    lower_energy_mev_per_nucleon: float
    upper_energy_mev_per_nucleon: float
    tabulated_delta_range_g_cm2: float
    integrated_delta_range_g_cm2: float
    relative_discrepancy: float
    within_tolerance: bool


@dataclass(frozen=True)
class ParameterizedStoppingGroupValidation:
    z: int
    mass_parameter_u: float
    material_z: int
    interval_count: int
    tabulated_delta_range_g_cm2: float
    integrated_delta_range_g_cm2: float
    worst_interval: ParameterizedStoppingInterval
    max_relative_discrepancy: float
    numerical_consistency_passed: bool


@dataclass(frozen=True)
class ParameterizedStoppingValidation:
    row_count: int
    group_count: int
    relative_tolerance: float
    groups: tuple[ParameterizedStoppingGroupValidation, ...]
    max_relative_discrepancy: float
    numerical_consistency_passed: bool
    normative_export_allowed: bool = field(default=False, init=False)
    absolute_range_origin_verified: bool = field(default=False, init=False)
    isotope_identification_established: bool = field(default=False, init=False)
    interpolation: str = field(default="piecewise_log_log_total_stopping", init=False)


def _log_ratio(numerator: float, denominator: float) -> float:
    difference = (numerator - denominator) / denominator
    if isfinite(difference) and difference > -1:
        return log1p(difference)
    return log(numerator) - log(denominator)


def _integrated_increment(
    left: ParameterizedStoppingRow, right: ParameterizedStoppingRow,
) -> float:
    # With S(e)=S0*(e/e0)^b, integrate m/S_total, not 1/S_total:
    # deltaR = m*e0/S0 * log(e1/e0) * exprel((1-b)*log(e1/e0)).
    log_energy_ratio = _log_ratio(
        right.energy_mev_per_nucleon, left.energy_mev_per_nucleon,
    )
    x = log_energy_ratio - _log_ratio(
        right.stopping_total_mev_cm2_g, left.stopping_total_mev_cm2_g,
    )
    if x == 0:
        log_exprel = 0.0
    elif x > 50:
        log_exprel = x + log(-expm1(-x)) - log(x)
    else:
        log_exprel = log(expm1(x) / x)
    try:
        value = exp(
            log(left.mass_parameter_u) + log(left.energy_mev_per_nucleon)
            - log(left.stopping_total_mev_cm2_g) + log(log_energy_ratio) + log_exprel
        )
    except (OverflowError, ValueError) as exc:
        raise ValueError("Range increment is not representable as a positive finite float.") from exc
    _number(value, "integrated range increment")
    return value


def validate_parameterized_rows(
    rows: Sequence[ParameterizedStoppingRow],
    *,
    required_projectiles: Sequence[tuple[int, float]],
    material_atomic_numbers: Sequence[int],
    energy_nodes_mev_per_nucleon: Sequence[float],
    relative_tolerance: float = 0.01,
) -> ParameterizedStoppingValidation:
    """Require exact declared coverage and report, without repairing, discrepancies.

    The relative discrepancy denominator is the integrated range increment.
    Only the worst interval per group is retained to bound report size. An
    additive range offset is invisible by design, and fractional mass equality
    is never promoted to isotope identification or physical model acceptance.
    """
    _number(relative_tolerance, "relative_tolerance")
    if relative_tolerance >= 1:
        raise ValueError("relative_tolerance must be less than one.")
    projectiles = tuple(required_projectiles)
    materials = tuple(material_atomic_numbers)
    energies = tuple(energy_nodes_mev_per_nucleon)
    for item in projectiles:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Each required projectile must declare exactly Z and mass parameter.")
        _projectile(*item)
    # Normalize accepted pair sequences without modifying any mass parameter.
    projectiles = tuple((z, mass) for z, mass in projectiles)
    if not projectiles or len(set(projectiles)) != len(projectiles):
        raise ValueError("Required projectiles must be nonempty and unique.")
    for material in materials:
        _material(material)
    if not materials or len(set(materials)) != len(materials):
        raise ValueError("Required materials must be nonempty and unique.")
    for energy in energies:
        _number(energy, "energy node")
    if len(energies) < 2 or any(a >= b for a, b in zip(energies, energies[1:])):
        raise ValueError("At least two strictly increasing energy nodes are required.")
    expected_groups = {(z, mass, material) for z, mass in projectiles for material in materials}
    grouped: dict[tuple[int, float, int], list[ParameterizedStoppingRow]] = {}
    for row in rows:
        if not isinstance(row, ParameterizedStoppingRow):
            raise ValueError("All rows must be ParameterizedStoppingRow instances.")
        grouped.setdefault((row.z, row.mass_parameter_u, row.material_z), []).append(row)
    if set(grouped) != expected_groups:
        raise ValueError("Missing or unexpected projectile/material groups.")
    summaries = []
    for key in sorted(grouped):
        group = grouped[key]
        if tuple(row.energy_mev_per_nucleon for row in group) != energies:
            raise ValueError(f"Missing, extra, duplicate or unsorted energy nodes for {key}.")
        intervals = []
        for left, right in zip(group, group[1:]):
            observed = right.range_g_cm2 - left.range_g_cm2
            _number(observed, "tabulated range increment (range must increase)")
            expected = _integrated_increment(left, right)
            discrepancy = abs(observed / expected - 1.0)
            _number(discrepancy, "relative discrepancy", zero_allowed=True)
            intervals.append(ParameterizedStoppingInterval(
                left.energy_mev_per_nucleon, right.energy_mev_per_nucleon,
                observed, expected, discrepancy, discrepancy <= relative_tolerance,
            ))
        worst = max(intervals, key=lambda interval: interval.relative_discrepancy)
        try:
            integrated_total = fsum(interval.integrated_delta_range_g_cm2 for interval in intervals)
        except OverflowError as exc:
            raise ValueError("Integrated group increment exceeds finite representation.") from exc
        _number(integrated_total, "integrated group increment")
        summaries.append(ParameterizedStoppingGroupValidation(
            *key, len(intervals), group[-1].range_g_cm2 - group[0].range_g_cm2,
            integrated_total, worst, worst.relative_discrepancy,
            all(interval.within_tolerance for interval in intervals),
        ))
    return ParameterizedStoppingValidation(
        len(rows), len(summaries), relative_tolerance, tuple(summaries),
        max(group.max_relative_discrepancy for group in summaries),
        all(group.numerical_consistency_passed for group in summaries),
    )
