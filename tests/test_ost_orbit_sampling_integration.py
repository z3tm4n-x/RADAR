"""Bounded orbit windows stay explicit through access and spectrum provenance."""

import pytest

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.core.types import OrbitType
from radar.geomagnetic.ost_orbit import OST_ORBIT_SAMPLING_MODEL_VERSION, sample_ost_orbit
from radar.geomagnetic.ost_penetration import (
    build_ost_penetration_function_for_config,
    build_ost_penetration_function_for_orbit,
    build_ost_penetration_function_from_samples,
)
from radar.geomagnetic.rigidity import RigidityGrid


def _heo() -> OrbitConfig:
    return OrbitConfig(
        orbit_type=OrbitType.ELLIPTIC, perigee_altitude_km=500.0,
        apogee_altitude_km=39700.0, inclination_deg=63.4,
    )


@pytest.mark.parametrize("minimum_orbits", [10, 20])
def test_orbit_and_config_builders_preserve_bounded_window_policy(minimum_orbits: int) -> None:
    orbit = _heo()
    grid = RigidityGrid((0.001, 0.01, 0.1, 1.0, 100.0))
    trace = sample_ost_orbit(
        orbit, min_days=1e-9, min_orbits=minimum_orbits, steps_per_orbit=12, max_samples=300,
    )
    reference = build_ost_penetration_function_from_samples(
        samples=trace.samples, rigidity_grid=grid, kp=3, apply_disturbance=True,
    )
    actual = build_ost_penetration_function_for_orbit(
        orbit, rigidity_grid=grid, min_days=1e-9, min_orbits=minimum_orbits,
        steps_per_orbit=12, max_samples=300, apply_disturbance=True,
    )
    configured = build_ost_penetration_function_for_config(
        CalculationConfig(mission=MissionConfig(2026, 5), orbit=orbit),
        rigidity_grid=grid, min_days=1e-9, min_orbits=minimum_orbits,
        steps_per_orbit=12, max_samples=300,
    )
    assert actual.values == configured.values == reference.values
    parameters = dict(actual.parameters)
    assert parameters["orbit_sampling_policy"] == OST_ORBIT_SAMPLING_MODEL_VERSION
    assert parameters["orbit_sampling_version"] == OST_ORBIT_SAMPLING_MODEL_VERSION
    assert parameters["minimum_orbits"] == str(minimum_orbits)
    assert parameters["orbit_sample_count"] == str(minimum_orbits * 12)
    assert parameters["orbit_sampling_capped"] == "false"
    assert parameters["orbit_window_scope"] == "finite_window_not_qualified_mission_mean"
    assert float(parameters["requested_window_s"]) == pytest.approx(
        minimum_orbits * trace.orbital_period_s,
    )
    assert float(parameters["orbit_window_s"]) == trace.window_s
    assert any("representative of the mission" in reason for reason in actual.incomplete_reasons)
    assert all(reason in actual.incomplete_reasons for reason in reference.incomplete_reasons)
    assert configured.incomplete_reasons == actual.incomplete_reasons


def test_orbit_budget_failure_is_not_a_partial_spectrum() -> None:
    grid = RigidityGrid((0.001, 1.0))
    with pytest.raises(ValueError, match="sample|budget"):
        build_ost_penetration_function_for_orbit(
            _heo(), rigidity_grid=grid, min_days=1e-9, min_orbits=10,
            steps_per_orbit=60, max_samples=599,
        )
    with pytest.raises(ValueError, match="sample|budget"):
        build_ost_penetration_function_for_config(
            CalculationConfig(mission=MissionConfig(2026, 5), orbit=_heo()),
            rigidity_grid=grid, min_days=1e-9, min_orbits=10,
            steps_per_orbit=60, max_samples=599,
        )
