"""Independent weighted-access examples for Appendix Zh's strict threshold.

These checks establish the averaging algebra, not a mission's MLT distribution
or physical/normative qualification of the adopted shadow model.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.geomagnetic import ost_penetration as kernel
from radar.geomagnetic.ost import EARTH_RADIUS_KM
from radar.geomagnetic.ost_orbit import OstOrbitSample, sample_ost_orbit
from radar.geomagnetic.rigidity import RigidityGrid


def _samples(count: int = 2) -> tuple[OstOrbitSample, ...]:
    return tuple(
        OstOrbitSample(
            index=index, time_s=float(index), true_anomaly_deg=0.0,
            radius_km=EARTH_RADIUS_KM + 400.0, altitude_km=400.0,
            latitude_deg=0.0, longitude_deg=0.0,
        )
        for index in range(count)
    )


def _constant_environment(monkeypatch: pytest.MonkeyPatch, cutoff: float = 2.5) -> None:
    monkeypatch.setattr(kernel, "ost_sample_cutoff_rigidity_gv", lambda sample: cutoff)
    monkeypatch.setattr(kernel, "ost_sample_shadow_weight", lambda sample: 1.0)


def _two_state_cutoff(
    *, cutoff_rigidity_gv: float, kp: int, magnetic_local_time_hours: float,
) -> float:
    del kp
    delta = 2.0 if magnetic_local_time_hours % 24.0 < 12.0 else 4.0
    return cutoff_rigidity_gv / delta


def test_quiet_threshold_is_strict_including_adjacent_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutoff = 0.3
    _constant_environment(monkeypatch, cutoff)
    grid = RigidityGrid((math.nextafter(cutoff, 0.0), cutoff, math.nextafter(cutoff, math.inf)))
    values = kernel.ost_penetration_values_from_samples(samples=_samples(), rigidity_grid=grid)
    assert values == (0.0, 0.0, 1.0)


def test_two_state_example_averages_access_not_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    _constant_environment(monkeypatch)
    monkeypatch.setattr(kernel, "ost_disturbed_cutoff_rigidity_gv", _two_state_cutoff)
    values = kernel.ost_penetration_values_from_samples(
        samples=_samples(), rigidity_grid=RigidityGrid((0.5, 0.625, 1.0, 1.25, 2.0)),
        apply_disturbance=True, magnetic_local_times_hours=(0.0, 12.0),
    )
    assert values == (0.0, 0.0, 0.5, 0.5, 1.0)
    # Averaging delta first gives full access at R=1 instead of the correct 1/2.
    assert int(1.0 > 2.5 / ((2.0 + 4.0) / 2.0)) == 1


@pytest.mark.parametrize("weights", [(2.0, 8.0), (0.2, 0.8), (2e307, 8e307)])
def test_unequal_weights_are_normalized_not_extra_mission_time(
    monkeypatch: pytest.MonkeyPatch, weights: tuple[float, ...],
) -> None:
    _constant_environment(monkeypatch)
    monkeypatch.setattr(kernel, "ost_disturbed_cutoff_rigidity_gv", _two_state_cutoff)
    assert kernel.ost_penetration_values_from_samples(
        samples=_samples(), rigidity_grid=RigidityGrid((1.0, 2.0)),
        apply_disturbance=True, sample_weights=weights,
        magnetic_local_times_hours=(0.0, 12.0),
    ) == pytest.approx((0.8, 1.0))


def test_zero_weight_is_explicit_not_dropped_from_implicit_equal_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _constant_environment(monkeypatch)
    monkeypatch.setattr(kernel, "ost_disturbed_cutoff_rigidity_gv", _two_state_cutoff)
    assert kernel.ost_penetration_values_from_samples(
        samples=_samples(), rigidity_grid=RigidityGrid((1.0, 2.0)),
        apply_disturbance=True, sample_weights=(1.0, 0.0),
        magnetic_local_times_hours=(0.0, 12.0),
    ) == (0.0, 1.0)


@pytest.mark.parametrize("disturbed", [False, True])
def test_positive_access_underflow_is_not_silently_replaced_with_zero(
    monkeypatch: pytest.MonkeyPatch, disturbed: bool,
) -> None:
    _constant_environment(monkeypatch, cutoff=0.3)
    monkeypatch.setattr(kernel, "ost_sample_shadow_weight", lambda sample: 0.5)
    with pytest.raises(ValueError, match="access underflows"):
        kernel.ost_penetration_values_from_samples(
            samples=_samples(), rigidity_grid=RigidityGrid((20.0,)),
            sample_weights=(5e-324, 1.0), apply_disturbance=disturbed,
        )


def test_default_uniform_ensemble_matches_explicit_states_and_not_old_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _constant_environment(monkeypatch, cutoff=0.3)
    grid = RigidityGrid((0.05, 0.1, 0.2, 0.3))
    ensemble = kernel.ost_penetration_values_from_samples(
        samples=_samples(1), rigidity_grid=grid, kp=3, apply_disturbance=True,
        mlt_sample_count=24,
    )
    explicit = kernel.ost_penetration_values_from_samples(
        samples=_samples(24), rigidity_grid=grid, kp=3, apply_disturbance=True,
        magnetic_local_times_hours=tuple(float(t) for t in range(24)),
    )
    assert ensemble == pytest.approx((7 / 24, 15 / 24, 21 / 24, 1.0))
    assert explicit == pytest.approx(ensemble)
    assert tuple(int(r > 0.3 / 4.2797) for r in grid.values_gv) == (0, 1, 1, 1)


@pytest.mark.parametrize("mlt_count", [1, 2, 24, 48])
def test_constant_state_is_invariant_to_ensemble_density(
    monkeypatch: pytest.MonkeyPatch, mlt_count: int,
) -> None:
    _constant_environment(monkeypatch)
    monkeypatch.setattr(kernel, "ost_disturbed_cutoff_rigidity_gv", lambda **kwargs: 1.25)
    values = kernel.ost_penetration_values_from_samples(
        samples=_samples(3), rigidity_grid=RigidityGrid((1.0, 1.25, 2.0)),
        apply_disturbance=True, mlt_sample_count=mlt_count,
    )
    assert values == (0.0, 0.0, 1.0)


def test_many_equal_weights_are_bounded_and_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    _constant_environment(monkeypatch, cutoff=0.3)
    values = kernel.ost_penetration_values_from_samples(
        samples=_samples(10_001), rigidity_grid=RigidityGrid((0.01, 0.05, 0.1, 1.0)),
        apply_disturbance=True, mlt_sample_count=24,
    )
    assert values == tuple(sorted(values))
    assert all(0 <= value <= 1 for value in values)
    assert values[-1] == pytest.approx(1.0, abs=2e-15)


@pytest.mark.parametrize(
    "weights",
    [(), (1.0,), (1.0, 1.0, 1.0), (0.0, 0.0), (-1.0, 2.0),
     (math.nan, 1.0), (math.inf, 1.0), (True, 1.0), (5e-324, 1e308)],
)
def test_invalid_weights_fail_closed(weights: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="[Ww]eight"):
        kernel.ost_penetration_values_from_samples(
            samples=_samples(), rigidity_grid=RigidityGrid((1.0,)), sample_weights=weights,
        )


@pytest.mark.parametrize(
    "times", [(), (0.0,), (0.0, 1.0, 2.0), (0.0, math.nan), (0.0, math.inf), (0.0, True)],
)
def test_invalid_explicit_mlt_fails_closed(times: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="[Mm]agnetic local times"):
        kernel.ost_penetration_values_from_samples(
            samples=_samples(), rigidity_grid=RigidityGrid((1.0,)),
            apply_disturbance=True, magnetic_local_times_hours=times,
        )


def test_explicit_mlt_is_not_silently_ignored_in_quiet_mode() -> None:
    with pytest.raises(ValueError, match="require disturbance"):
        kernel.ost_penetration_values_from_samples(
            samples=_samples(), rigidity_grid=RigidityGrid((1.0,)),
            magnetic_local_times_hours=(0.0, 12.0),
        )


@pytest.mark.parametrize("kp", [-1, 10, 3.5, True])
def test_invalid_kp_is_rejected_even_for_quiet_mode(kp: Any) -> None:
    with pytest.raises(ValueError, match="Kp"):
        kernel.ost_penetration_values_from_samples(
            samples=_samples(), rigidity_grid=RigidityGrid((1.0,)), kp=kp,
        )


@pytest.mark.parametrize("mlt_count", [0, -1, 2.5, True])
def test_invalid_ensemble_count_is_rejected(mlt_count: Any) -> None:
    with pytest.raises(ValueError, match="MLT sample count"):
        kernel.ost_penetration_values_from_samples(
            samples=_samples(), rigidity_grid=RigidityGrid((1.0,)), mlt_sample_count=mlt_count,
        )


def test_nonboolean_disturbance_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="boolean"):
        kernel.ost_penetration_values_from_samples(
            samples=_samples(), rigidity_grid=RigidityGrid((1.0,)),
            apply_disturbance=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("shadow", [-0.1, 1.01, math.nan, math.inf])
def test_invalid_shadow_is_rejected_not_clipped(
    monkeypatch: pytest.MonkeyPatch, shadow: float,
) -> None:
    monkeypatch.setattr(kernel, "ost_sample_shadow_weight", lambda sample: shadow)
    with pytest.raises(ValueError, match="Earth-shadow"):
        kernel.ost_penetration_values_from_samples(
            samples=_samples(), rigidity_grid=RigidityGrid((1.0,)),
        )


def test_explicit_mlt_uses_periodic_local_time_values() -> None:
    samples = _samples()
    grid = RigidityGrid((0.01, 0.1, 1.0, 20.0))
    first = kernel.ost_penetration_values_from_samples(
        samples=samples, rigidity_grid=grid,
        apply_disturbance=True, magnetic_local_times_hours=(-12.0, 24.0),
    )
    second = kernel.ost_penetration_values_from_samples(
        samples=samples, rigidity_grid=grid,
        apply_disturbance=True, magnetic_local_times_hours=(12.0, 0.0),
    )
    assert first == second


def test_from_samples_builder_declares_uniform_ensemble_and_unresolved_shadow() -> None:
    result = kernel.build_ost_penetration_function_from_samples(
        samples=_samples(), rigidity_grid=RigidityGrid((0.1, 1.0, 20.0)),
        apply_disturbance=True,
    )
    parameters = dict(result.parameters)
    assert result.model_version == "appendix_zh_pointwise_access_v2"
    assert parameters["mlt_policy"] == "uniform_ensemble"
    assert parameters["mlt_sample_count"] == "24"
    assert parameters["sample_weight_policy"] == "equal_samples"
    assert parameters["cutoff_condition"] == "R>Rc_star"
    assert parameters["shadow_model"] == "opaque_sphere_geometric"
    assert any("Zh.9" in reason for reason in result.incomplete_reasons)
    assert any("MLT ensemble" in reason for reason in result.incomplete_reasons)


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_or_quiet_policies_do_not_claim_uniform_mlt_history(explicit: bool) -> None:
    result = kernel.build_ost_penetration_function_from_samples(
        samples=_samples(), rigidity_grid=RigidityGrid((0.1, 1.0, 20.0)),
        sample_weights=(1.0, 3.0), apply_disturbance=explicit,
        magnetic_local_times_hours=(0.0, 12.0) if explicit else None,
    )
    parameters = dict(result.parameters)
    assert parameters["mlt_policy"] == ("explicit_per_sample" if explicit else "not_used_quiet")
    assert parameters["mlt_sample_count"] == ("1" if explicit else "0")
    assert parameters["sample_weight_policy"] == "explicit_normalized"
    assert len(result.incomplete_reasons) == (3 if explicit else 1)
    assert "Zh.9" in result.incomplete_reasons[0]
    if explicit:
        assert any("no qualified derivation" in r for r in result.incomplete_reasons)
        assert any("local-time convention" in r for r in result.incomplete_reasons)


def test_orbit_and_config_builders_forward_explicit_weights_and_mlt() -> None:
    orbit = OrbitConfig.circular(altitude_km=400.0, inclination_deg=0.0)
    trace = sample_ost_orbit(orbit, min_days=0.05, steps_per_orbit=12, max_samples=200)
    weights = tuple(float(i + 1) for i in range(len(trace.samples)))
    times = tuple(float(i % 24) for i in range(len(trace.samples)))
    grid = RigidityGrid((0.01, 0.1, 1.0, 20.0))
    expected = kernel.ost_penetration_values_from_samples(
        samples=trace.samples, rigidity_grid=grid, kp=4, apply_disturbance=True,
        sample_weights=weights, magnetic_local_times_hours=times,
    )
    from_orbit = kernel.build_ost_penetration_function_for_orbit(
        orbit, rigidity_grid=grid, min_days=0.05, steps_per_orbit=12, max_samples=200,
        kp=4, apply_disturbance=True, sample_weights=weights, magnetic_local_times_hours=times,
    )
    config = CalculationConfig(mission=MissionConfig(2026, 1), orbit=orbit, kp=4)
    from_config = kernel.build_ost_penetration_function_for_config(
        config, rigidity_grid=grid, min_days=0.05, steps_per_orbit=12, max_samples=200,
        sample_weights=weights, magnetic_local_times_hours=times,
    )
    assert from_orbit.values == from_config.values == expected


def test_orbit_builder_checks_mlt_shape_after_sampling() -> None:
    with pytest.raises(ValueError, match="must match the orbit sample count"):
        kernel.build_ost_penetration_function_for_orbit(
            OrbitConfig.circular(altitude_km=400.0, inclination_deg=0.0),
            rigidity_grid=RigidityGrid((1.0,)), min_days=0.05,
            steps_per_orbit=12, max_samples=200, apply_disturbance=True,
            magnetic_local_times_hours=(0.0,),
        )
