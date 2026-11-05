"""Anchored range/stopping diagnostics, never a replacement range generator.

CSDA requires dR/dE = 1/S_total for TOTAL kinetic energy and mass stopping.
The two explicit inter-node models below are numerical hypotheses, not physical
uncertainty bounds. A projected range is retained as projected, even if its
increments happen to agree. No absolute range or stopping origin is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Context, Decimal, DecimalException, ROUND_HALF_EVEN, localcontext
from typing import Literal


RangeKind = Literal["projected", "unverified", "csda_declared"]
_KINDS = ("projected", "unverified", "csda_declared")
_WORK_CONTEXT = Context(prec=110, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
_RESULT_CONTEXT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
_ONE = Decimal(1)
_HUNDRED = Decimal(100)


@dataclass(frozen=True)
class RangeIntegralComparison:
    increment_g_cm2: Decimal
    anchored_integral_g_cm2: Decimal
    interval_range_minus_integral_g_cm2: Decimal
    interval_relative_difference_percent_of_integral: Decimal
    anchored_range_minus_integral_g_cm2: Decimal
    anchored_relative_difference_percent_of_integral: Decimal


@dataclass(frozen=True)
class RangeStoppingInterval:
    lower_energy_total_mev: Decimal
    upper_energy_total_mev: Decimal
    range_increment_g_cm2: Decimal
    anchored_range_increment_g_cm2: Decimal
    linear_stopping: RangeIntegralComparison
    loglog_stopping: RangeIntegralComparison
    interpolation_loglog_minus_linear_increment_g_cm2: Decimal
    interpolation_interval_percent_of_linear: Decimal
    interpolation_loglog_minus_linear_anchored_g_cm2: Decimal
    interpolation_anchored_percent_of_linear: Decimal


@dataclass(frozen=True)
class RangeStoppingConsistencyAudit:
    energy_total_mev: tuple[Decimal, ...]
    total_stopping_mev_cm2_g: tuple[Decimal, ...]
    range_g_cm2: tuple[Decimal, ...]
    range_kind: RangeKind
    intervals: tuple[RangeStoppingInterval, ...]
    diagnostic_status: str = field(default="INCOMPLETE", init=False)
    normative_export_allowed: bool = field(default=False, init=False)
    physical_accuracy_established: bool = field(default=False, init=False)
    range_kind_verified: bool = field(default=False, init=False)
    absolute_range_origin_established: bool = field(default=False, init=False)
    production_tables_replaced: bool = field(default=False, init=False)
    interpolation_spread_is_error_bound: bool = field(default=False, init=False)
    precision_decimal_digits: int = field(default=80, init=False)
    energy_coordinate: str = field(default="total_kinetic_energy_MeV", init=False)
    stopping_definition: str = field(default="total_mass_stopping_MeV_cm2_g", init=False)
    range_definition: str = field(default="mass_range_g_cm2_as_supplied", init=False)
    anchoring: str = field(default="differences_from_first_supplied_node_only", init=False)
    linear_interpolation: str = field(default="S_linear_in_total_energy", init=False)
    loglog_interpolation: str = field(default="log_S_linear_in_log_total_energy", init=False)
    references: tuple[str, ...] = field(default=(
        "https://physics.nist.gov/PhysRefData/Star/Text/appendix.html",
        "https://physics.nist.gov/PhysRefData/Star/Text/programs.html",
    ), init=False)
    limitations: tuple[str, ...] = field(default=(
        "No extrapolation below or above the supplied nodes.",
        "No absolute range is reconstructed and no range origin is assumed.",
        "Input rounding and unknown inter-node physics are not uncertainty estimates.",
        "CSDA declaration is supplied metadata, not independent verification.",
        "For an energy-per-nucleon coordinate epsilon, dE_total=A*d_epsilon is required.",
        "Nuclear stopping means elastic recoil loss, not nuclear-reaction attenuation.",
    ), init=False)


def _validate(values: tuple[Decimal, ...], name: str, *, positive: bool) -> None:
    if type(values) is not tuple or len(values) < 2:
        raise ValueError(f"{name} must be an immutable tuple with at least two nodes.")
    for value in values:
        if (type(value) is not Decimal or not value.is_finite() or value < 0
                or (positive and value == 0)):
            raise ValueError(f"{name} requires finite nonnegative Decimal values.")
        # An explicit bounded arithmetic contract, not a physical energy cutoff.
        if len(value.as_tuple().digits) > 80 or (value != 0 and abs(value.adjusted()) > 1000):
            raise ValueError(f"{name} exceeds the supported 80-digit/exponent-1000 input domain.")


def _rounded(value: Decimal) -> Decimal:
    if not value.is_finite():
        raise ValueError("Nonfinite range/stopping diagnostic result.")
    with localcontext(_RESULT_CONTEXT):
        return +value


def _log1p_over_x(x: Decimal) -> Decimal:
    """Cancellation-safe log(1+x)/x for the near-constant stopping limit."""
    if x == 0:
        return _ONE
    if abs(x) > Decimal("0.01"):
        return (_ONE + x).ln() / x
    total, power, denominator = _ONE, _ONE, 1
    while True:
        power *= -x
        denominator += 1
        updated = total + power / denominator
        if updated == total:
            return total
        total = updated


def _log_ratio(left: Decimal, right: Decimal) -> Decimal:
    relative = (right - left) / left
    if abs(relative) <= Decimal("0.01"):
        return relative * _log1p_over_x(relative)
    # Taking the ratio avoids subtracting two almost equal logarithms. The
    # input exponent limit makes every supported ratio finite in our context.
    return (right / left).ln()


def _expm1_over_x(x: Decimal) -> Decimal:
    """Cancellation-safe (exp(x)-1)/x, including power-law exponent one."""
    if x == 0:
        return _ONE
    if abs(x) > Decimal("0.1"):
        return (x.exp() - _ONE) / x
    total, term, denominator = _ONE, _ONE, 1
    while True:
        denominator += 1
        term *= x / denominator
        updated = total + term
        if updated == total:
            return total
        total = updated


def _integrals(e0: Decimal, e1: Decimal, s0: Decimal, s1: Decimal) -> tuple[Decimal, Decimal]:
    relative = (s1 - s0) / s0
    if abs(relative) <= Decimal("0.01"):
        linear = (e1 - e0) / s0 * _log1p_over_x(relative)
    else:
        linear = (e1 - e0) * _log_ratio(s0, s1) / (s1 - s0)
    log_energy = _log_ratio(e0, e1)
    # q = (1-p) log(E1/E0); this expression avoids division by p-1.
    q = log_energy - _log_ratio(s0, s1)
    loglog = e0 / s0 * log_energy * _expm1_over_x(q)
    if linear <= 0 or loglog <= 0:
        raise ValueError("Integrated positive stopping must give positive range increments.")
    return linear, loglog


def _comparison(delta_range: Decimal, anchored_range: Decimal,
                increment: Decimal, anchored_integral: Decimal) -> RangeIntegralComparison:
    difference = delta_range - increment
    anchored_difference = anchored_range - anchored_integral
    return RangeIntegralComparison(*(_rounded(value) for value in (
        increment, anchored_integral, difference, _HUNDRED * difference / increment,
        anchored_difference, _HUNDRED * anchored_difference / anchored_integral,
    )))


def audit_range_stopping_consistency(
    *, energy_total_mev: tuple[Decimal, ...],
    total_stopping_mev_cm2_g: tuple[Decimal, ...],
    range_g_cm2: tuple[Decimal, ...], range_kind: RangeKind,
) -> RangeStoppingConsistencyAudit:
    """Compare anchored mass-range differences with two integrals of 1/S_total.

    All nodes and range semantics are retained. Differences are signed as
    (supplied range increment - integral), relative percentages use the integral
    as denominator. No tolerance or numeric result establishes physical fitness.
    Inputs support at most 80 significant digits and adjusted exponents +/-1000;
    private 110-digit working arithmetic is rounded to 80 digits for the ledger.
    """
    _validate(energy_total_mev, "energy", positive=True)
    _validate(total_stopping_mev_cm2_g, "stopping", positive=True)
    _validate(range_g_cm2, "range", positive=False)
    if len({len(energy_total_mev), len(total_stopping_mev_cm2_g), len(range_g_cm2)}) != 1:
        raise ValueError("Energy, total stopping and range must have equal node counts.")
    if type(range_kind) is not str or range_kind not in _KINDS:
        raise ValueError("Explicit supported range_kind is required.")
    if any(b <= a for values in (energy_total_mev, range_g_cm2)
           for a, b in zip(values, values[1:])):
        raise ValueError("Energy and range nodes must be strictly increasing.")
    intervals = []
    try:
        with localcontext(_WORK_CONTEXT):
            linear_total, loglog_total = Decimal(0), Decimal(0)
            for e0, e1, s0, s1, r0, r1 in zip(
                energy_total_mev, energy_total_mev[1:], total_stopping_mev_cm2_g,
                total_stopping_mev_cm2_g[1:], range_g_cm2, range_g_cm2[1:],
            ):
                linear, loglog = _integrals(e0, e1, s0, s1)
                linear_total += linear
                loglog_total += loglog
                delta_range = r1 - r0
                anchored_range = r1 - range_g_cm2[0]
                intervals.append(RangeStoppingInterval(
                    e0, e1, _rounded(delta_range), _rounded(anchored_range),
                    _comparison(delta_range, anchored_range, linear, linear_total),
                    _comparison(delta_range, anchored_range, loglog, loglog_total),
                    _rounded(loglog - linear), _rounded(_HUNDRED * (loglog - linear) / linear),
                    _rounded(loglog_total - linear_total),
                    _rounded(_HUNDRED * (loglog_total - linear_total) / linear_total),
                ))
    except DecimalException as exc:
        raise ValueError("Unsupported arithmetic in range/stopping diagnostic.") from exc
    return RangeStoppingConsistencyAudit(
        energy_total_mev, total_stopping_mev_cm2_g, range_g_cm2, range_kind, tuple(intervals),
    )
