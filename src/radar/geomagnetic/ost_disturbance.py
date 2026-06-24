"""OST 134-1044-2007 geomagnetic disturbance correction.

Implements Appendix Zh formulas Zh.4--Zh.6 for converting quiet cutoff
rigidity Rc into disturbed cutoff rigidity Rc* using Kp and magnetic local time.
"""

from __future__ import annotations

import math

MIN_DISTURBANCE_CUTOFF_GV = 1.0e-4


def _validate_kp(kp: int) -> None:
    if not isinstance(kp, int):
        msg = "Kp must be an integer."
        raise ValueError(msg)

    if not 0 <= kp <= 9:
        msg = "Kp must be in the range 0..9."
        raise ValueError(msg)


def _validate_cutoff_rigidity(cutoff_rigidity_gv: float) -> None:
    if not math.isfinite(cutoff_rigidity_gv):
        msg = "Cutoff rigidity must be finite."
        raise ValueError(msg)

    if cutoff_rigidity_gv < 0.0:
        msg = "Cutoff rigidity must be non-negative."
        raise ValueError(msg)


def _normalized_mlt_hours(magnetic_local_time_hours: float) -> float:
    if not math.isfinite(magnetic_local_time_hours):
        msg = "Magnetic local time must be finite."
        raise ValueError(msg)

    return magnetic_local_time_hours % 24.0


def _validate_mlt_sample_count(mlt_sample_count: int) -> None:
    if not isinstance(mlt_sample_count, int):
        msg = "MLT sample count must be an integer."
        raise ValueError(msg)

    if mlt_sample_count < 1:
        msg = "MLT sample count must be positive."
        raise ValueError(msg)


def ost_disturbance_coefficients(
    *,
    kp: int,
    magnetic_local_time_hours: float,
) -> tuple[float, float, float]:
    """Return Appendix Zh disturbance coefficients C, A and B."""

    _validate_kp(kp)
    t_mlt = _normalized_mlt_hours(magnetic_local_time_hours)
    argument_factor = math.pi / 12.0

    a_c = 0.0768 * math.sin(argument_factor * (t_mlt + 6.082)) + 0.0769
    b_c = 2.3564 * math.sin(argument_factor * (t_mlt + 5.785)) + 3.5876

    a_a = -0.037 * math.sin(argument_factor * (t_mlt - 5.844)) + 0.357
    b_a = -0.267 * math.sin(argument_factor * (t_mlt - 5.198)) + 6.073

    a_b = 0.0022 * math.sin(argument_factor * (t_mlt - 6.448)) + 0.00177
    b_b = 0.0091 * math.sin(argument_factor * (t_mlt - 6.390)) - 0.30538

    c_limit = a_c * kp**2 + b_c
    a_parameter = a_a * kp + b_a
    b_parameter = a_b * kp + b_b

    return c_limit, a_parameter, b_parameter


def ost_cutoff_disturbance_factor(
    *,
    cutoff_rigidity_gv: float,
    kp: int,
    magnetic_local_time_hours: float,
) -> float:
    """Return Appendix Zh disturbance factor delta(Rc, Kp, tMLT)."""

    _validate_cutoff_rigidity(cutoff_rigidity_gv)
    c_limit, a_parameter, b_parameter = ost_disturbance_coefficients(
        kp=kp,
        magnetic_local_time_hours=magnetic_local_time_hours,
    )

    effective_cutoff_gv = max(cutoff_rigidity_gv, MIN_DISTURBANCE_CUTOFF_GV)
    raw_delta = 1.0 + 0.001 * math.exp(
        a_parameter * effective_cutoff_gv**b_parameter - 1.0,
    )

    return float(min(raw_delta, c_limit))


def ost_cutoff_disturbance_factor_mlt_average(
    *,
    cutoff_rigidity_gv: float,
    kp: int,
    mlt_sample_count: int = 24,
) -> float:
    """Return MLT-averaged disturbance factor for the current RADAR builder."""

    _validate_cutoff_rigidity(cutoff_rigidity_gv)
    _validate_kp(kp)
    _validate_mlt_sample_count(mlt_sample_count)

    return sum(
        ost_cutoff_disturbance_factor(
            cutoff_rigidity_gv=cutoff_rigidity_gv,
            kp=kp,
            magnetic_local_time_hours=24.0 * index / mlt_sample_count,
        )
        for index in range(mlt_sample_count)
    ) / mlt_sample_count


def ost_disturbed_cutoff_rigidity_gv(
    *,
    cutoff_rigidity_gv: float,
    kp: int,
    magnetic_local_time_hours: float,
) -> float:
    """Return disturbed cutoff rigidity Rc* = Rc / delta."""

    delta = ost_cutoff_disturbance_factor(
        cutoff_rigidity_gv=cutoff_rigidity_gv,
        kp=kp,
        magnetic_local_time_hours=magnetic_local_time_hours,
    )
    return float(cutoff_rigidity_gv / delta)


def ost_disturbed_cutoff_rigidity_mlt_average_gv(
    *,
    cutoff_rigidity_gv: float,
    kp: int,
    mlt_sample_count: int = 24,
) -> float:
    """Return MLT-averaged disturbed cutoff rigidity Rc* = Rc / <delta>."""

    delta = ost_cutoff_disturbance_factor_mlt_average(
        cutoff_rigidity_gv=cutoff_rigidity_gv,
        kp=kp,
        mlt_sample_count=mlt_sample_count,
    )
    return float(cutoff_rigidity_gv / delta)
