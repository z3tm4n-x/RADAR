"""Block time provenance must not reset phases or confer mission qualification."""

import pytest

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.core.types import OrbitType
from radar.geomagnetic.ost_orbit import (
    OST_ORBIT_TIME_ORIGIN_MODEL_VERSION,
    sample_ost_orbit,
)
from radar.geomagnetic.ost_penetration import (
    build_ost_penetration_function_for_config,
    build_ost_penetration_function_for_orbit,
    build_ost_penetration_function_from_samples,
)
from radar.geomagnetic.rigidity import RigidityGrid


@pytest.mark.parametrize("start_index", [0, 137, 960])
@pytest.mark.parametrize("explicit_weights", [False, True])
def test_builders_preserve_global_time_and_relative_weight_semantics(
    start_index: int, explicit_weights: bool,
) -> None:
    orbit = OrbitConfig(OrbitType.ELLIPTIC, 500.0, 39700.0, 63.4)
    grid = RigidityGrid((0.001, 0.01, 0.1, 1.0, 10.0, 100.0))
    trace = sample_ost_orbit(
        orbit, raan_deg=90.0, start_sample_index=start_index,
        min_days=1e-9, min_orbits=1, steps_per_orbit=24,
    )
    weights = tuple(float(i + 1) for i in range(24)) if explicit_weights else None
    reference = build_ost_penetration_function_from_samples(
        samples=trace.samples, rigidity_grid=grid, apply_disturbance=True,
        kp=3, sample_weights=weights,
    )
    actual = build_ost_penetration_function_for_orbit(
        orbit, rigidity_grid=grid, raan_deg=90.0, start_sample_index=start_index,
        min_days=1e-9, min_orbits=1, steps_per_orbit=24,
        apply_disturbance=True, sample_weights=weights,
    )
    configured = build_ost_penetration_function_for_config(
        CalculationConfig(MissionConfig(2026, 5), orbit), rigidity_grid=grid,
        raan_deg=90.0, start_sample_index=start_index,
        min_days=1e-9, min_orbits=1, steps_per_orbit=24,
        sample_weights=weights,
    )
    assert actual == configured
    assert actual.values == reference.values
    parameters = dict(actual.parameters)
    assert parameters["orbit_time_origin_version"] == OST_ORBIT_TIME_ORIGIN_MODEL_VERSION
    assert parameters["orbit_start_sample_index"] == str(start_index)
    assert float(parameters["orbit_start_time_s"]) == trace.start_time_s
    assert float(parameters["orbit_end_time_s"]) == trace.end_time_s
    assert parameters["orbit_time_reference"] == "elapsed_since_reference_perigee_not_UTC"
    assert parameters["initial_node_reference"] == "relative_to_Greenwich_at_reference_perigee"
    assert parameters["initial_Greenwich_angle"].startswith("implicit_zero")
    assert parameters["orbit_window_scope"] == "finite_window_not_qualified_mission_mean"
    assert set(reference.incomplete_reasons).issubset(actual.incomplete_reasons)
    assert any("representative of the mission" in r for r in actual.incomplete_reasons)


@pytest.mark.parametrize("start_index", [-1, True, 0.5, float("inf")])
def test_config_builder_refuses_invalid_start_index(start_index: object) -> None:
    with pytest.raises(ValueError):
        build_ost_penetration_function_for_config(
            CalculationConfig(MissionConfig(2026, 5), OrbitConfig.circular(400.0, 51.6)),
            rigidity_grid=RigidityGrid((0.001, 100.0)),
            start_sample_index=start_index,  # type: ignore[arg-type]
        )
