import math

import pytest

from radar.geomagnetic.ost_disturbance import (
    ost_cutoff_disturbance_factor,
    ost_cutoff_disturbance_factor_mlt_average,
    ost_disturbance_coefficients,
    ost_disturbed_cutoff_rigidity_gv,
    ost_disturbed_cutoff_rigidity_mlt_average_gv,
)


def _reference_coefficients(kp: int, t_mlt: float) -> tuple[float, float, float]:
    argument_factor = math.pi / 12.0

    a_c = 0.0768 * math.sin(argument_factor * (t_mlt + 6.082)) + 0.0769
    b_c = 2.3564 * math.sin(argument_factor * (t_mlt + 5.785)) + 3.5876

    a_a = -0.037 * math.sin(argument_factor * (t_mlt - 5.844)) + 0.357
    b_a = -0.267 * math.sin(argument_factor * (t_mlt - 5.198)) + 6.073

    a_b = 0.0022 * math.sin(argument_factor * (t_mlt - 6.448)) + 0.00177
    b_b = 0.0091 * math.sin(argument_factor * (t_mlt - 6.390)) - 0.30538

    return a_c * kp**2 + b_c, a_a * kp + b_a, a_b * kp + b_b


def _reference_delta(rc: float, kp: int, t_mlt: float) -> float:
    c_limit, a_parameter, b_parameter = _reference_coefficients(kp, t_mlt)
    raw_delta = 1.0 + 0.001 * math.exp(a_parameter * max(rc, 1.0e-4) ** b_parameter - 1.0)
    return min(raw_delta, c_limit)


def test_ost_disturbance_coefficients_match_appendix_zh_formulas() -> None:
    assert ost_disturbance_coefficients(
        kp=5,
        magnetic_local_time_hours=7.5,
    ) == pytest.approx(_reference_coefficients(5, 7.5))


def test_ost_cutoff_disturbance_factor_matches_appendix_zh_formula() -> None:
    assert ost_cutoff_disturbance_factor(
        cutoff_rigidity_gv=2.5,
        kp=4,
        magnetic_local_time_hours=13.0,
    ) == pytest.approx(_reference_delta(2.5, 4, 13.0))


def test_ost_disturbed_cutoff_is_quiet_cutoff_divided_by_delta() -> None:
    rc = 3.0
    kp = 6
    t_mlt = 9.0
    delta = _reference_delta(rc, kp, t_mlt)

    assert ost_disturbed_cutoff_rigidity_gv(
        cutoff_rigidity_gv=rc,
        kp=kp,
        magnetic_local_time_hours=t_mlt,
    ) == pytest.approx(rc / delta)


def test_ost_cutoff_disturbance_factor_mlt_average_matches_discrete_mean() -> None:
    rc = 1.2
    kp = 5
    sample_count = 24
    expected = sum(
        _reference_delta(rc, kp, 24.0 * index / sample_count)
        for index in range(sample_count)
    ) / sample_count

    assert ost_cutoff_disturbance_factor_mlt_average(
        cutoff_rigidity_gv=rc,
        kp=kp,
        mlt_sample_count=sample_count,
    ) == pytest.approx(expected)


def test_ost_disturbed_cutoff_mlt_average_is_not_larger_than_quiet_cutoff() -> None:
    rc = 4.0

    assert ost_disturbed_cutoff_rigidity_mlt_average_gv(
        cutoff_rigidity_gv=rc,
        kp=5,
    ) <= rc


def test_ost_disturbance_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="range 0..9"):
        ost_cutoff_disturbance_factor(
            cutoff_rigidity_gv=1.0,
            kp=10,
            magnetic_local_time_hours=0.0,
        )

    with pytest.raises(ValueError, match="non-negative"):
        ost_cutoff_disturbance_factor(
            cutoff_rigidity_gv=-1.0,
            kp=1,
            magnetic_local_time_hours=0.0,
        )

    with pytest.raises(ValueError, match="MLT sample count"):
        ost_cutoff_disturbance_factor_mlt_average(
            cutoff_rigidity_gv=1.0,
            kp=1,
            mlt_sample_count=0,
        )
