"""Build a diagnostic, pointwise-access interpretation of OST Appendix Zh.

The strict cutoff condition follows Zh.7. Geometric Earth shadowing and an
optional uniform MLT ensemble remain explicit model-qualification limitations.
This is a weighted mean-access operator, not an instantaneous peak operator.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
import hashlib
import math

from radar.core.constants import DEFAULT_KP, MAX_KP, MIN_KP
from radar.core.project import CalculationConfig, OrbitConfig
from radar.geomagnetic.ost import ost_cutoff_rigidity_gv, ost_earth_shadow_factor
from radar.geomagnetic.ost_orbit import (
    OST_ORBIT_SAMPLING_MODEL_VERSION,
    OST_ORBIT_TIME_ORIGIN_MODEL_VERSION,
    OstOrbitSample,
    sample_ost_orbit,
)
from radar.geomagnetic.ost_disturbance import ost_disturbed_cutoff_rigidity_gv
from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.local_time import LocalTimeHistory
from radar.geomagnetic.rigidity import RigidityGrid

OST_GEOMAGNETIC_PENETRATION_MODEL = "ost_134_1044_2007_geomagnetic_penetration"
OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION = "appendix_zh_pointwise_access_v2"

_SHADOW_QUALIFICATION_REASON = (
    "OST Zh.9 shadow qualification unresolved: geometric sphere differs "
    "from printed empirical expression."
)
_MLT_QUALIFICATION_REASON = (
    "Uniform MLT ensemble is not a reconstructed mission MLT history."
)
_ORBIT_WINDOW_QUALIFICATION_REASON = (
    "Finite orbit averaging window has not been qualified as representative of the mission."
)
_EXPLICIT_LOCAL_TIME_QUALIFICATION_REASON = (
    "Explicit local-time values have no qualified derivation or trajectory provenance."
)
_LOCAL_TIME_CONVENTION_REASON = (
    "OST Zh.6 local-time convention and its calibration have not been confirmed."
)


def _resolved_local_times(
    samples: tuple[OstOrbitSample, ...],
    raw_values: tuple[float, ...] | None,
    history: LocalTimeHistory | None,
) -> tuple[float, ...] | None:
    if history is None:
        return raw_values
    if raw_values is not None:
        raise ValueError("Use either raw local-time values or a trace-bound history, not both.")
    if not isinstance(history, LocalTimeHistory):
        raise ValueError("Local-time history must use the explicit LocalTimeHistory contract.")
    history.validate_samples(samples)
    return history.values_hours


def _local_time_values_sha256(values: tuple[float, ...]) -> str:
    """Canonical periodic numerical input, not a certificate of its physical origin."""
    return hashlib.sha256(
        "\n".join(float(value % 24.0).hex() for value in values).encode("ascii"),
    ).hexdigest()


def _normalized_samples(
    samples: tuple[OstOrbitSample, ...],
) -> tuple[OstOrbitSample, ...]:
    if not samples:
        msg = "OST penetration samples must not be empty."
        raise ValueError(msg)

    return tuple(samples)


def _normalized_sample_weights(
    sample_count: int,
    sample_weights: tuple[float, ...] | None,
) -> tuple[float, ...]:
    """Normalize dimensionless weights without overflowing their sum."""
    if sample_weights is None:
        return (1.0 / sample_count,) * sample_count
    if len(sample_weights) != sample_count:
        raise ValueError("Sample weights must match the orbit sample count.")
    if any(
        isinstance(weight, bool) or not math.isfinite(weight) or weight < 0.0
        for weight in sample_weights
    ):
        raise ValueError("Sample weights must be finite non-negative numbers.")
    scale = max(sample_weights)
    if scale == 0.0:
        raise ValueError("At least one sample weight must be positive.")
    scaled = tuple(weight / scale for weight in sample_weights)
    denominator = math.fsum(scaled)
    normalized = tuple(weight / denominator for weight in scaled)
    if any(raw > 0.0 and weight == 0.0 for raw, weight in zip(sample_weights, normalized)):
        raise ValueError("Sample-weight normalization underflow would discard a positive weight.")
    return normalized


def _validate_access_options(
    *,
    sample_count: int,
    kp: int,
    apply_disturbance: bool,
    mlt_sample_count: int,
    magnetic_local_times_hours: tuple[float, ...] | None,
) -> None:
    if isinstance(kp, bool) or not isinstance(kp, int) or not MIN_KP <= kp <= MAX_KP:
        raise ValueError("Kp must be an integer in the range 0..9.")
    if not isinstance(apply_disturbance, bool):
        raise ValueError("apply_disturbance must be boolean.")
    if (
        isinstance(mlt_sample_count, bool)
        or not isinstance(mlt_sample_count, int)
        or mlt_sample_count < 1
    ):
        raise ValueError("MLT sample count must be a positive integer.")
    if magnetic_local_times_hours is not None:
        if not apply_disturbance:
            raise ValueError("Explicit magnetic local times require disturbance correction.")
        if len(magnetic_local_times_hours) != sample_count:
            raise ValueError("Magnetic local times must match the orbit sample count.")
        if any(
            isinstance(value, bool) or not math.isfinite(value)
            for value in magnetic_local_times_hours
        ):
            raise ValueError("Magnetic local times must be finite numbers.")


def _add_compensated(
    sums: list[float], corrections: list[float], index: int, value: float,
) -> None:
    """Neumaier summation, retaining small non-negative contributions."""
    previous = sums[index]
    updated = previous + value
    if abs(previous) >= abs(value):
        corrections[index] += (previous - updated) + value
    else:
        corrections[index] += (value - updated) + previous
    sums[index] = updated


def ost_sample_cutoff_rigidity_gv(sample: OstOrbitSample) -> float:
    """Return OST cutoff rigidity for one sampled orbit point."""

    return ost_cutoff_rigidity_gv(
        latitude_deg=sample.latitude_deg,
        longitude_deg=sample.longitude_deg,
        altitude_km=sample.altitude_km,
    )


def ost_sample_shadow_weight(sample: OstOrbitSample) -> float:
    """Return OST Earth-shadow weighting factor for one sampled orbit point."""

    return ost_earth_shadow_factor(sample.altitude_km)


def ost_penetration_values_from_samples(
    *,
    samples: tuple[OstOrbitSample, ...],
    rigidity_grid: RigidityGrid,
    kp: int = DEFAULT_KP,
    apply_disturbance: bool = False,
    mlt_sample_count: int = 24,
    sample_weights: tuple[float, ...] | None = None,
    magnetic_local_times_hours: tuple[float, ...] | None = None,
    local_time_history: LocalTimeHistory | None = None,
) -> tuple[float, ...]:
    """Return the shadow-weighted mean of strict pointwise access indicators.

    With normalized sample weights w_i and explicit per-sample MLT values:

        psi(R) = sum(w_i * mu_i * I(R > Rc_i / delta_i)).

    If disturbance is disabled, delta_i is one. With disturbance enabled but
    no supplied MLT history, each orbit point is averaged over a uniform MLT
    ensemble of ``mlt_sample_count`` nodes. The indicators are averaged, never
    delta before applying the threshold. That ensemble is an explicit
    diagnostic assumption, not a reconstruction of the mission's MLT history.

    Weights are relative quadrature weights (equal by default), not an extra
    mission-duration multiplier. This routine does not infer a peak statistic.
    Bin deposition uses O(number of rigidity nodes) accumulation memory rather
    than retaining all orbit-by-MLT pairs.
    """

    normalized_samples = _normalized_samples(samples)
    magnetic_local_times_hours = _resolved_local_times(
        normalized_samples, magnetic_local_times_hours, local_time_history,
    )
    _validate_access_options(
        sample_count=len(normalized_samples), kp=kp, apply_disturbance=apply_disturbance,
        mlt_sample_count=mlt_sample_count, magnetic_local_times_hours=magnetic_local_times_hours,
    )
    weights = _normalized_sample_weights(len(normalized_samples), sample_weights)
    grid = rigidity_grid.values_gv
    bin_sums = [0.0] * len(grid)
    bin_corrections = [0.0] * len(grid)
    ensemble = (
        tuple(24.0 * index / mlt_sample_count for index in range(mlt_sample_count))
        if apply_disturbance and magnetic_local_times_hours is None else ()
    )

    for index, (sample, weight) in enumerate(zip(normalized_samples, weights)):
        if weight == 0.0:
            continue
        cutoff_rigidity_gv = ost_sample_cutoff_rigidity_gv(sample)
        shadow_weight = ost_sample_shadow_weight(sample)
        if not math.isfinite(shadow_weight) or not 0.0 <= shadow_weight <= 1.0:
            raise ValueError("Earth-shadow weights must be finite and within 0..1.")
        if not math.isfinite(cutoff_rigidity_gv) or cutoff_rigidity_gv < 0.0:
            raise ValueError("Sample cutoff rigidity must be finite and non-negative.")

        if apply_disturbance:
            local_times = (
                (magnetic_local_times_hours[index],)
                if magnetic_local_times_hours is not None else ensemble
            )
            # Aggregate node counts per sample before depositing to avoid a
            # large state list and unnecessary rounding of identical bins.
            counts: dict[int, int] = {}
            for local_time in local_times:
                cutoff = ost_disturbed_cutoff_rigidity_gv(
                    cutoff_rigidity_gv=cutoff_rigidity_gv,
                    kp=kp, magnetic_local_time_hours=local_time,
                )
                if not math.isfinite(cutoff) or cutoff < 0.0:
                    raise ValueError("Disturbed cutoff rigidity must be finite and non-negative.")
                first_passing_node = bisect_right(grid, cutoff)
                if first_passing_node < len(grid):
                    counts[first_passing_node] = counts.get(first_passing_node, 0) + 1
            for first_passing_node, count in counts.items():
                contribution = weight * shadow_weight * (count / len(local_times))
                if contribution == 0.0 and shadow_weight > 0.0:
                    raise ValueError("Positive weighted access underflows to zero.")
                _add_compensated(bin_sums, bin_corrections, first_passing_node, contribution)
        else:
            first_passing_node = bisect_right(grid, cutoff_rigidity_gv)
            if first_passing_node < len(grid):
                contribution = weight * shadow_weight
                if contribution == 0.0 and shadow_weight > 0.0:
                    raise ValueError("Positive weighted access underflows to zero.")
                _add_compensated(
                    bin_sums, bin_corrections, first_passing_node, contribution,
                )

    prefix_sum = [0.0]
    prefix_correction = [0.0]
    values: list[float] = []
    for total, correction in zip(bin_sums, bin_corrections):
        _add_compensated(prefix_sum, prefix_correction, 0, math.fsum((total, correction)))
        value = math.fsum((prefix_sum[0], prefix_correction[0]))
        if not -8.0 * math.ulp(1.0) <= value <= 1.0 + 8.0 * math.ulp(1.0):
            raise ArithmeticError("Weighted penetration escaped its physical bounds.")
        # Only roundoff-sized excursions are permitted; physical input errors
        # are rejected above and never repaired by this final bound guard.
        values.append(min(1.0, max(0.0, value)))

    return tuple(values)


def build_ost_penetration_function_from_samples(
    *,
    samples: tuple[OstOrbitSample, ...],
    rigidity_grid: RigidityGrid,
    model: str = OST_GEOMAGNETIC_PENETRATION_MODEL,
    kp: int = DEFAULT_KP,
    apply_disturbance: bool = False,
    mlt_sample_count: int = 24,
    sample_weights: tuple[float, ...] | None = None,
    magnetic_local_times_hours: tuple[float, ...] | None = None,
    local_time_history: LocalTimeHistory | None = None,
) -> PenetrationFunction:
    """Build diagnostic mean access with its explicit qualification limits."""

    resolved_times = _resolved_local_times(samples, magnetic_local_times_hours, local_time_history)
    mlt_policy = (
        "not_used_quiet" if not apply_disturbance else
        "trace_bound_history" if local_time_history is not None else
        "explicit_per_sample" if resolved_times is not None else
        "uniform_ensemble"
    )
    reasons: tuple[str, ...] = (_SHADOW_QUALIFICATION_REASON,)
    if apply_disturbance:
        reasons += (_LOCAL_TIME_CONVENTION_REASON,)
        if local_time_history is not None:
            reasons += local_time_history.incomplete_reasons
        elif resolved_times is not None:
            reasons += (_EXPLICIT_LOCAL_TIME_QUALIFICATION_REASON,)
        else:
            reasons += (_MLT_QUALIFICATION_REASON,)

    # Validate numerics before hashing; a history binds data but does not certify
    # the convention used to calibrate Zh.6, the ephemeris or the mission epoch.
    values = ost_penetration_values_from_samples(
        samples=samples, rigidity_grid=rigidity_grid, kp=kp,
        apply_disturbance=apply_disturbance, mlt_sample_count=mlt_sample_count,
        sample_weights=sample_weights, magnetic_local_times_hours=resolved_times,
    )
    history_parameters: tuple[tuple[str, str], ...] = ()
    if resolved_times is not None:
        history_parameters += (("local_time_values_sha256", _local_time_values_sha256(resolved_times)),)
    if local_time_history is not None:
        history_parameters += (
            ("local_time_model", local_time_history.model),
            ("local_time_samples_sha256", local_time_history.sample_fingerprint),
            *((f"local_time_parameter_{key}", value)
              for key, value in local_time_history.parameters),
        )

    return PenetrationFunction(
        rigidity_grid=rigidity_grid,
        values=values,
        model=model,
        kp=kp,
        model_version=OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
        parameters=(
            ("averaging", "weighted_mean_of_pointwise_access"),
            ("cutoff_condition", "R>Rc_star"),
            ("shadow_model", "opaque_sphere_geometric"),
            ("mlt_policy", mlt_policy),
            ("local_time_qualification_version", "trace_bound_local_time_v1"),
            ("local_time_normative_qualified", "false"),
            ("sample_weight_policy", "equal_samples" if sample_weights is None
             else "explicit_normalized"),
            ("mlt_sample_count", str(
                0 if not apply_disturbance else
                1 if resolved_times is not None else mlt_sample_count
            )),
            *history_parameters,
        ),
        incomplete_reasons=tuple(dict.fromkeys(reasons)),
    )


def build_ost_penetration_function_for_orbit(
    config: OrbitConfig,
    *,
    rigidity_grid: RigidityGrid,
    raan_deg: float = 0.0,
    start_sample_index: int = 0,
    min_days: float = 16.0,
    min_orbits: int = 10,
    steps_per_orbit: int = 60,
    max_samples: int = 400000,
    model: str = OST_GEOMAGNETIC_PENETRATION_MODEL,
    kp: int = DEFAULT_KP,
    apply_disturbance: bool = False,
    mlt_sample_count: int = 24,
    sample_weights: tuple[float, ...] | None = None,
    magnetic_local_times_hours: tuple[float, ...] | None = None,
    local_time_history: LocalTimeHistory | None = None,
) -> PenetrationFunction:
    """Build finite-window mean access; refuse a budget-induced loss of resolution.

    The window's mission representativeness is not established merely by a
    successful finite-window quadrature or by using the same operator for both
    source statistics. Explicit sample weights are relative quadrature weights.
    A nonzero start index advances every orbital and Earth-rotation phase on
    the same global grid; it does not reset the reference perigee or node.
    """

    trace = sample_ost_orbit(
        config,
        raan_deg=raan_deg,
        start_sample_index=start_sample_index,
        min_days=min_days,
        min_orbits=min_orbits,
        steps_per_orbit=steps_per_orbit,
        max_samples=max_samples,
    )

    penetration = build_ost_penetration_function_from_samples(
        samples=trace.samples,
        rigidity_grid=rigidity_grid,
        model=model,
        kp=kp,
        apply_disturbance=apply_disturbance,
        mlt_sample_count=mlt_sample_count,
        sample_weights=sample_weights,
        magnetic_local_times_hours=magnetic_local_times_hours,
        local_time_history=local_time_history,
    )
    requested_step_s = trace.orbital_period_s / steps_per_orbit
    sampling_capped = trace.step_s > requested_step_s and not math.isclose(
        trace.step_s, requested_step_s, rel_tol=1e-12, abs_tol=0.0,
    )
    reasons = penetration.incomplete_reasons
    if trace.sampling_policy == OST_ORBIT_SAMPLING_MODEL_VERSION:
        reasons += (_ORBIT_WINDOW_QUALIFICATION_REASON,)
    if sampling_capped:
        reasons += (
            "Orbit sample cap prevents requested density: "
            f"requested step {requested_step_s:.17g} s, actual step {trace.step_s:.17g} s; "
            "orbit-sampling convergence is not established.",
        )
    return replace(
        penetration,
        parameters=(*penetration.parameters,
                    ("orbit_sample_count", str(len(trace.samples))),
                    ("requested_steps_per_orbit", str(steps_per_orbit)),
                    ("actual_step_s", f"{trace.step_s:.17g}"),
                    ("orbit_window_s", f"{trace.window_s:.17g}"),
                    ("orbit_sample_limit", str(max_samples)),
                    ("orbit_sampling_capped", str(sampling_capped).lower()),
                    ("orbit_sampling_policy", trace.sampling_policy),
                    ("orbit_sampling_version", OST_ORBIT_SAMPLING_MODEL_VERSION),
                    ("orbit_time_origin_version", OST_ORBIT_TIME_ORIGIN_MODEL_VERSION),
                    ("orbit_start_sample_index", str(trace.start_sample_index)),
                    ("orbit_start_time_s", f"{trace.start_time_s:.17g}"),
                    ("orbit_end_time_s", f"{trace.end_time_s:.17g}"),
                    ("orbit_time_reference", "elapsed_since_reference_perigee_not_UTC"),
                    ("requested_window_s", "unspecified" if trace.requested_window_s is None
                     else f"{trace.requested_window_s:.17g}"),
                    ("minimum_orbits", str(min_orbits)),
                    ("orbit_quadrature", "equal_time_left_cells" if sample_weights is None
                     else "left_time_nodes_with_explicit_relative_weights"),
                    ("orbit_window_scope", "finite_window_not_qualified_mission_mean"),
                    ("secular_drift_model", "ost_literal_D9_D10_with_1_minus_e"),
                    ("initial_raan_deg", f"{raan_deg:.17g}"),
                    ("initial_node_reference", "relative_to_Greenwich_at_reference_perigee"),
                    ("initial_Greenwich_angle", "implicit_zero_not_reconstructed_from_launch_year"),
                    ("minimum_sampling_days", f"{min_days:.17g}")),
        incomplete_reasons=tuple(dict.fromkeys(reasons)),
    )



def build_ost_penetration_function_for_config(
    config: CalculationConfig,
    *,
    rigidity_grid: RigidityGrid,
    raan_deg: float = 0.0,
    start_sample_index: int = 0,
    min_days: float = 16.0,
    min_orbits: int = 10,
    steps_per_orbit: int = 60,
    max_samples: int = 400000,
    model: str = OST_GEOMAGNETIC_PENETRATION_MODEL,
    apply_disturbance: bool = True,
    mlt_sample_count: int = 24,
    sample_weights: tuple[float, ...] | None = None,
    magnetic_local_times_hours: tuple[float, ...] | None = None,
    local_time_history: LocalTimeHistory | None = None,
) -> PenetrationFunction:
    """Build OST geomagnetic penetration function from calculation config.

    The production config-level builder uses the mission orbit and Kp index
    from ``CalculationConfig``. Kp disturbance correction is enabled by default
    because Appendix Zh defines the disturbed cutoff rigidity Rc* through Kp.
    This builder remains a diagnostic mean-access operator. Its applicability
    to a source statistic must be qualified by the calling pipeline; the
    assessment mode does not silently turn it into a peak/max operator.
    """

    return build_ost_penetration_function_for_orbit(
        config.orbit,
        rigidity_grid=rigidity_grid,
        raan_deg=raan_deg,
        start_sample_index=start_sample_index,
        min_days=min_days,
        min_orbits=min_orbits,
        steps_per_orbit=steps_per_orbit,
        max_samples=max_samples,
        model=model,
        kp=config.kp,
        apply_disturbance=apply_disturbance,
        mlt_sample_count=mlt_sample_count,
        sample_weights=sample_weights,
        magnetic_local_times_hours=magnetic_local_times_hours,
        local_time_history=local_time_history,
    )
