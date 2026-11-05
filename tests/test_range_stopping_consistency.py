"""Analytical controls for anchored diagnostics; no native program execution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from decimal import Decimal, ROUND_DOWN, localcontext
from typing import Any

import pytest

from radar.validation.range_stopping_consistency import audit_range_stopping_consistency


D = Decimal


def audit(**changes: Any) -> Any:
    kwargs: dict[str, Any] = {
        "energy_total_mev": (D(1), D(2), D(4)),
        "total_stopping_mev_cm2_g": (D(2), D(2), D(2)),
        "range_g_cm2": (D("0.5"), D(1), D(2)),
        "range_kind": "projected",
    }
    kwargs.update(changes)
    return audit_range_stopping_consistency(**kwargs)


def close(actual: Decimal, expected: Decimal, tolerance: str = "1e-77") -> None:
    with localcontext() as context:
        context.prec = 110
        assert abs(actual - expected) <= D(tolerance) * max(D(1), abs(expected))


def test_constant_stopping_and_anchored_origin() -> None:
    result = audit()
    for row, expected in zip(result.intervals, (D("0.5"), D(1)), strict=True):
        assert row.linear_stopping.increment_g_cm2 == expected
        close(row.loglog_stopping.increment_g_cm2, expected)
        assert row.linear_stopping.interval_range_minus_integral_g_cm2 == 0
        close(row.loglog_stopping.anchored_relative_difference_percent_of_integral, D(0))
    assert result.intervals[-1].linear_stopping.anchored_integral_g_cm2 == D("1.5")
    assert not result.absolute_range_origin_established


def test_linear_stopping_analytic_integral() -> None:
    result = audit(total_stopping_mev_cm2_g=(D(3), D(5), D(9)))
    with localcontext() as context:
        context.prec = 100
        close(result.intervals[0].linear_stopping.increment_g_cm2, (D(5)/3).ln()/2)
        close(result.intervals[-1].linear_stopping.anchored_integral_g_cm2, D(3).ln()/2)
    assert result.intervals[0].interpolation_loglog_minus_linear_increment_g_cm2 != 0


@pytest.mark.parametrize("exponent", [-3, -1, 0, 1, 2, 4])
def test_loglog_power_law_analytic_integrals(exponent: int) -> None:
    energy = (D(1), D(2), D(4))
    stopping = tuple(D(3) * e**exponent for e in energy)
    result = audit(total_stopping_mev_cm2_g=stopping)
    with localcontext() as context:
        context.prec = 100
        expected = (D(4).ln()/3 if exponent == 1 else
                    (D(4)**(1-exponent)-1)/(3*(1-exponent)))
    close(result.intervals[-1].loglog_stopping.anchored_integral_g_cm2, expected)


def test_range_offset_does_not_change_any_interval() -> None:
    first = audit()
    shifted = audit(range_g_cm2=(D("10.5"), D(11), D(12)))
    assert first.intervals == shifted.intervals
    assert first.range_g_cm2 != shifted.range_g_cm2


def test_total_energy_requires_mass_factor_not_stopping_division() -> None:
    baseline = audit()
    scaled = audit(energy_total_mev=(D(12), D(24), D(48)),
                   range_g_cm2=(D(6), D(12), D(24)))
    for first, second in zip(baseline.intervals, scaled.intervals, strict=True):
        close(second.linear_stopping.increment_g_cm2, 12 * first.linear_stopping.increment_g_cm2)
        close(second.loglog_stopping.increment_g_cm2, 12 * first.loglog_stopping.increment_g_cm2)
        assert second.linear_stopping.interval_relative_difference_percent_of_integral == 0


def test_simultaneous_energy_stopping_scaling_preserves_range_integral() -> None:
    baseline = audit()
    scaled = audit(energy_total_mev=(D(1000), D(2000), D(4000)),
                   total_stopping_mev_cm2_g=(D(2000), D(2000), D(2000)))
    for first, second in zip(baseline.intervals, scaled.intervals, strict=True):
        assert first.linear_stopping == second.linear_stopping
        assert first.loglog_stopping == second.loglog_stopping


def test_nearly_constant_stopping_retains_small_increment() -> None:
    s1 = D("1." + "0" * 70 + "1")
    result = audit(energy_total_mev=(D(1), D(2)),
                   total_stopping_mev_cm2_g=(D(1), s1), range_g_cm2=(D(0), D(1)))
    with localcontext() as context:
        context.prec = 160
        expected = s1.ln() / (s1 - 1)
    close(result.intervals[0].linear_stopping.increment_g_cm2, expected, "1e-79")
    assert result.intervals[0].linear_stopping.increment_g_cm2 < 1


def test_very_close_energy_nodes_not_rounded_into_one_node() -> None:
    e1 = D("1." + "0" * 70 + "1")
    r1 = D("1e-71")
    result = audit(energy_total_mev=(D(1), e1),
                   total_stopping_mev_cm2_g=(D(1), D(1)), range_g_cm2=(D(0), r1))
    assert result.intervals[0].linear_stopping.increment_g_cm2 == r1
    close(result.intervals[0].loglog_stopping.increment_g_cm2 / r1, D(1))


def test_nearly_unit_power_law_exponent_is_stable() -> None:
    with localcontext() as context:
        context.prec = 100
        epsilon = D("1e-65")
        stopping: tuple[Decimal, ...] = (D(1), D(2) ** (1 + epsilon))
        # Keep within the explicit <=80 significant digit input contract.
        context.prec = 78
        stopping = tuple(+s for s in stopping)
    result = audit(energy_total_mev=(D(1), D(2)), total_stopping_mev_cm2_g=stopping,
                   range_g_cm2=(D(0), D(1)))
    with localcontext() as context:
        context.prec = 160
        q = D(2).ln() - stopping[1].ln()
        expected = D(2).ln() * (q.exp() - 1) / q
    close(result.intervals[0].loglog_stopping.increment_g_cm2, expected)


def test_private_context_ignores_caller_precision_rounding_and_traps() -> None:
    expected = audit(total_stopping_mev_cm2_g=(D(3), D(5), D(9)))
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_DOWN
        context.Emax = 5
        context.Emin = -5
        for signal in context.traps:
            context.traps[signal] = True
        assert audit(total_stopping_mev_cm2_g=(D(3), D(5), D(9))) == expected


@pytest.mark.parametrize("kind", ["projected", "unverified", "csda_declared"])
def test_semantics_and_nonqualification_are_retained(kind: str) -> None:
    result = audit(range_kind=kind)
    assert result.range_kind == kind
    assert result.diagnostic_status == "INCOMPLETE"
    for key, value in asdict(result).items():
        if type(value) is bool:
            assert value is False, key
    with pytest.raises(FrozenInstanceError):
        result.normative_export_allowed = True


def test_relative_denominators_and_cumulative_quantities() -> None:
    result = audit(range_g_cm2=(D(0), D(1), D(2)))
    assert result.intervals[0].linear_stopping.interval_relative_difference_percent_of_integral == 100
    assert result.intervals[1].linear_stopping.interval_relative_difference_percent_of_integral == 0
    close(result.intervals[1].linear_stopping.anchored_relative_difference_percent_of_integral,
          D(100)/3, "1e-26")
    assert result.intervals[1].linear_stopping.anchored_range_minus_integral_g_cm2 == D("0.5")


@pytest.mark.parametrize("field", ["energy_total_mev", "total_stopping_mev_cm2_g", "range_g_cm2"])
@pytest.mark.parametrize("bad", [(), (D(1),), [D(1), D(2)], (1, 2), (True, D(2)),
                                (1.0, D(2)), (D("NaN"), D(2)), (D("sNaN"), D(2)),
                                (D("Infinity"), D(2)), (D(-1), D(2)),
                                (D("1e1001"), D("2e1001")),
                                (D("1." + "1" * 80), D(2))])
def test_invalid_input(field: str, bad: Any) -> None:
    with pytest.raises(ValueError):
        audit(**{field: bad})


@pytest.mark.parametrize("field", ["energy_total_mev", "total_stopping_mev_cm2_g"])
def test_energy_and_stopping_cannot_be_zero(field: str) -> None:
    with pytest.raises(ValueError):
        audit(**{field: (D(0), D(1), D(2))})


@pytest.mark.parametrize("field", ["energy_total_mev", "range_g_cm2"])
@pytest.mark.parametrize("values", [(D(1), D(1), D(2)), (D(1), D(3), D(2))])
def test_energy_and_range_must_increase(field: str, values: tuple[Decimal, ...]) -> None:
    with pytest.raises(ValueError):
        audit(**{field: values})


@pytest.mark.parametrize("kind", [None, True, 1, "csda", "", "PROJECTED"])
def test_unknown_range_kind_rejected(kind: Any) -> None:
    with pytest.raises(ValueError):
        audit(range_kind=kind)


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError):
        audit(total_stopping_mev_cm2_g=(D(1), D(2)))


@pytest.mark.parametrize("stopping", [(D("1e-1000"), D("1e1000")),
                                     (D("1e1000"), D("1e-1000"))])
def test_extreme_supported_ratios_produce_finite_positive_integrals(
    stopping: tuple[Decimal, ...],
) -> None:
    result = audit(energy_total_mev=(D(1), D(2)), total_stopping_mev_cm2_g=stopping,
                   range_g_cm2=(D(0), D(1)))
    for comparison in (result.intervals[0].linear_stopping, result.intervals[0].loglog_stopping):
        for value in asdict(comparison).values():
            assert value.is_finite()
        assert comparison.increment_g_cm2 > 0
