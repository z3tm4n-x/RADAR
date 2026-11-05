from dataclasses import replace

import pytest

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.geomagnetic.ost import (
    EARTH_RADIUS_KM,
    OST_CUTOFF_REFERENCE_ALTITUDE_KM,
    ost_earth_shadow_factor,
)
from radar.geomagnetic import ost_penetration as penetration_module
from radar.geomagnetic.ost_orbit import OstOrbitSample, OstOrbitTrace
from radar.geomagnetic.ost_penetration import (
    OST_GEOMAGNETIC_PENETRATION_MODEL,
    build_ost_penetration_function_for_config,
    build_ost_penetration_function_for_orbit,
    build_ost_penetration_function_from_samples,
    ost_penetration_values_from_samples,
    ost_sample_cutoff_rigidity_gv,
)
from radar.geomagnetic.rigidity import RigidityGrid


def _sample(
    *,
    index: int,
    latitude_deg: float,
    longitude_deg: float,
    altitude_km: float = OST_CUTOFF_REFERENCE_ALTITUDE_KM,
) -> OstOrbitSample:
    return OstOrbitSample(
        index=index,
        time_s=float(index),
        true_anomaly_deg=0.0,
        radius_km=EARTH_RADIUS_KM + altitude_km,
        altitude_km=altitude_km,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
    )


def test_ost_sample_cutoff_rigidity_uses_cutoff_table_and_altitude() -> None:
    sample = _sample(index=0, latitude_deg=85.0, longitude_deg=0.0)

    assert ost_sample_cutoff_rigidity_gv(sample) == pytest.approx(0.004)


def test_ost_penetration_values_are_cumulative_shadow_weighted_fraction() -> None:
    samples = (
        _sample(index=0, latitude_deg=85.0, longitude_deg=0.0),
        _sample(index=1, latitude_deg=0.0, longitude_deg=0.0),
    )
    rigidity_grid = RigidityGrid(values_gv=(0.001, 0.004, 1.0, 20.0))

    values = ost_penetration_values_from_samples(
        samples=samples,
        rigidity_grid=rigidity_grid,
    )
    weight = ost_earth_shadow_factor(OST_CUTOFF_REFERENCE_ALTITUDE_KM)

    assert values == pytest.approx(
        (
            0.0,
            0.0,
            weight / 2.0,
            weight,
        )
    )


def test_ost_penetration_values_are_monotonic_and_bounded() -> None:
    samples = (
        _sample(index=0, latitude_deg=85.0, longitude_deg=0.0),
        _sample(index=1, latitude_deg=45.0, longitude_deg=0.0),
        _sample(index=2, latitude_deg=0.0, longitude_deg=0.0),
    )
    values = ost_penetration_values_from_samples(
        samples=samples,
        rigidity_grid=RigidityGrid(values_gv=(0.001, 0.1, 1.0, 5.0, 20.0)),
    )

    assert values == tuple(sorted(values))
    assert all(0.0 <= value <= 1.0 for value in values)


def test_build_ost_penetration_function_from_samples_returns_container() -> None:
    samples = (
        _sample(index=0, latitude_deg=85.0, longitude_deg=0.0),
        _sample(index=1, latitude_deg=0.0, longitude_deg=0.0),
    )
    rigidity_grid = RigidityGrid(values_gv=(0.001, 0.004, 20.0))

    penetration = build_ost_penetration_function_from_samples(
        samples=samples,
        rigidity_grid=rigidity_grid,
        kp=5,
    )

    assert penetration.model == OST_GEOMAGNETIC_PENETRATION_MODEL
    assert penetration.kp == 5
    assert penetration.rigidity_grid == rigidity_grid
    assert penetration.values == pytest.approx(
        ost_penetration_values_from_samples(
            samples=samples,
            rigidity_grid=rigidity_grid,
        )
    )


def test_build_ost_penetration_function_from_samples_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="samples must not be empty"):
        build_ost_penetration_function_from_samples(
            samples=(),
            rigidity_grid=RigidityGrid(values_gv=(0.1, 1.0)),
        )


def test_build_ost_penetration_function_for_orbit_returns_valid_penetration() -> None:
    penetration = build_ost_penetration_function_for_orbit(
        OrbitConfig.circular(altitude_km=500.0, inclination_deg=0.0),
        rigidity_grid=RigidityGrid(values_gv=(0.001, 1.0, 20.0)),
        min_days=0.05,
        steps_per_orbit=12,
        max_samples=200,
    )

    assert penetration.model == OST_GEOMAGNETIC_PENETRATION_MODEL
    assert len(penetration.values) == 3
    assert penetration.values == tuple(sorted(penetration.values))
    assert all(0.0 <= value <= 1.0 for value in penetration.values)



def test_ost_penetration_disturbance_increases_access_at_fixed_grid() -> None:
    samples = (
        _sample(index=0, latitude_deg=85.0, longitude_deg=0.0),
        _sample(index=1, latitude_deg=45.0, longitude_deg=0.0),
        _sample(index=2, latitude_deg=0.0, longitude_deg=0.0),
    )
    rigidity_grid = RigidityGrid(values_gv=(0.001, 0.1, 1.0, 5.0, 20.0))

    quiet_values = ost_penetration_values_from_samples(
        samples=samples,
        rigidity_grid=rigidity_grid,
    )
    disturbed_values = ost_penetration_values_from_samples(
        samples=samples,
        rigidity_grid=rigidity_grid,
        kp=5,
        apply_disturbance=True,
    )

    assert all(
        disturbed_value >= quiet_value
        for quiet_value, disturbed_value in zip(quiet_values, disturbed_values)
    )


def test_ost_penetration_builder_passes_disturbance_options() -> None:
    samples = (
        _sample(index=0, latitude_deg=85.0, longitude_deg=0.0),
        _sample(index=1, latitude_deg=0.0, longitude_deg=0.0),
    )
    rigidity_grid = RigidityGrid(values_gv=(0.001, 0.004, 20.0))

    penetration = build_ost_penetration_function_from_samples(
        samples=samples,
        rigidity_grid=rigidity_grid,
        kp=6,
        apply_disturbance=True,
        mlt_sample_count=12,
    )

    assert penetration.kp == 6
    assert penetration.values == pytest.approx(
        ost_penetration_values_from_samples(
            samples=samples,
            rigidity_grid=rigidity_grid,
            kp=6,
            apply_disturbance=True,
            mlt_sample_count=12,
        )
    )



def test_build_ost_penetration_function_for_config_uses_orbit_and_kp() -> None:
    config = CalculationConfig(
        mission=MissionConfig(launch_year=2025, lifetime_years=1),
        orbit=OrbitConfig.circular(altitude_km=500.0, inclination_deg=0.0),
        kp=6,
    )
    rigidity_grid = RigidityGrid(values_gv=(0.001, 1.0, 20.0))

    from_config = build_ost_penetration_function_for_config(
        config,
        rigidity_grid=rigidity_grid,
        min_days=0.05,
        steps_per_orbit=12,
        max_samples=200,
        mlt_sample_count=12,
    )
    from_orbit = build_ost_penetration_function_for_orbit(
        config.orbit,
        rigidity_grid=rigidity_grid,
        min_days=0.05,
        steps_per_orbit=12,
        max_samples=200,
        kp=config.kp,
        apply_disturbance=True,
        mlt_sample_count=12,
    )

    assert from_config.kp == config.kp
    assert from_config.rigidity_grid == rigidity_grid
    assert from_config.values == pytest.approx(from_orbit.values)


def test_build_ost_penetration_function_for_config_can_disable_disturbance() -> None:
    config = CalculationConfig(
        mission=MissionConfig(launch_year=2025, lifetime_years=1),
        orbit=OrbitConfig.circular(altitude_km=500.0, inclination_deg=0.0),
        kp=8,
    )
    rigidity_grid = RigidityGrid(values_gv=(0.001, 1.0, 20.0))

    quiet = build_ost_penetration_function_for_config(
        config,
        rigidity_grid=rigidity_grid,
        min_days=0.05,
        steps_per_orbit=12,
        max_samples=200,
        apply_disturbance=False,
    )
    explicit_quiet = build_ost_penetration_function_for_orbit(
        config.orbit,
        rigidity_grid=rigidity_grid,
        min_days=0.05,
        steps_per_orbit=12,
        max_samples=200,
        kp=config.kp,
        apply_disturbance=False,
    )

    assert quiet.kp == config.kp
    assert quiet.values == pytest.approx(explicit_quiet.values)


@pytest.mark.parametrize(
    ("step_s", "capped"), [(1.0, False), (1.0 + 5e-13, False), (1.0 + 2e-12, True), (2.0, True)],
)
def test_orbit_sampling_cap_is_explicit_qualification_not_silent_density_loss(
    monkeypatch: pytest.MonkeyPatch, step_s: float, capped: bool,
) -> None:
    samples = (
        _sample(index=0, latitude_deg=85.0, longitude_deg=0.0),
        replace(_sample(index=1, latitude_deg=0.0, longitude_deg=0.0), time_s=step_s),
    )
    trace = OstOrbitTrace(
        samples=samples, semi_major_axis_km=EARTH_RADIUS_KM + 450.0, eccentricity=0.0,
        orbital_period_s=12.0, window_s=2 * step_s, step_s=step_s,
        node_precession_rad_s=0.0, perigee_precession_rad_s=0.0,
    )
    monkeypatch.setattr(penetration_module, "sample_ost_orbit", lambda *args, **kwargs: trace)
    grid = RigidityGrid((0.001, 0.1, 1.0, 20.0))
    weights = (1.0, 3.0)
    local_times = (0.0, 12.0)
    baseline = build_ost_penetration_function_from_samples(
        samples=samples, rigidity_grid=grid, apply_disturbance=True,
        sample_weights=weights, magnetic_local_times_hours=local_times,
    )
    result = build_ost_penetration_function_for_orbit(
        OrbitConfig.circular(450.0, 0.0), rigidity_grid=grid,
        raan_deg=37.0, min_days=16.0, steps_per_orbit=12, max_samples=4,
        apply_disturbance=True, sample_weights=weights, magnetic_local_times_hours=local_times,
    )
    assert result.values == baseline.values
    assert result.model_version == baseline.model_version
    assert all(reason in result.incomplete_reasons for reason in baseline.incomplete_reasons)
    cap_reasons = tuple(reason for reason in result.incomplete_reasons if "sample cap" in reason)
    assert bool(cap_reasons) is capped
    if capped:
        assert "convergence is not established" in cap_reasons[0]
    else:
        assert result.incomplete_reasons == baseline.incomplete_reasons
    parameters = dict(result.parameters)
    assert parameters["orbit_sample_count"] == "2"
    assert parameters["requested_steps_per_orbit"] == "12"
    assert float(parameters["actual_step_s"]) == step_s
    assert float(parameters["orbit_window_s"]) == 2 * step_s
    assert parameters["orbit_sample_limit"] == "4"
    assert parameters["orbit_sampling_capped"] == str(capped).lower()
    assert parameters["initial_raan_deg"] == "37"
    assert parameters["minimum_sampling_days"] == "16"
    assert parameters["sample_weight_policy"] == "explicit_normalized"
    assert parameters["orbit_quadrature"] == "left_time_nodes_with_explicit_relative_weights"
    assert parameters["mlt_policy"] == "explicit_per_sample"
