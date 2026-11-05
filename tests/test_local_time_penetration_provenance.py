"""Explicit time conventions remain diagnostic and are bound to their trajectory."""

from dataclasses import replace

import pytest

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.geomagnetic.local_time import build_centered_dipole_mlt_history
from radar.geomagnetic.ost_orbit import sample_ost_orbit
from radar.geomagnetic.ost_penetration import (
    build_ost_penetration_function_for_config,
    build_ost_penetration_function_for_orbit,
    build_ost_penetration_function_from_samples,
    ost_penetration_values_from_samples,
)
from radar.geomagnetic.rigidity import RigidityGrid


def test_trace_bound_history_matches_raw_numbers_but_does_not_confer_qualification() -> None:
    orbit = OrbitConfig.circular(400.0, 51.6)
    trace = sample_ost_orbit(orbit, min_days=1e-9, min_orbits=1, steps_per_orbit=12)
    grid = RigidityGrid((0.001, 0.1, 1.0, 10.0, 100.0))
    history = build_centered_dipole_mlt_history(
        samples=trace.samples, sun_directions_ecef=((1.0, 0.0, 0.0),) * len(trace.samples),
        north_axis_ecef=(0.0, 0.0, 1.0), sun_model="synthetic_constant_Earth_fixed",
        time_reference="test_no_UTC_epoch", incomplete_reasons=(
            "Test ephemeris is synthetic.",
            "Finite orbit averaging window has not been qualified as representative of the mission.",
        ),
    )
    raw = build_ost_penetration_function_from_samples(
        samples=trace.samples, rigidity_grid=grid, apply_disturbance=True,
        magnetic_local_times_hours=history.values_hours,
    )
    bound = build_ost_penetration_function_from_samples(
        samples=trace.samples, rigidity_grid=grid, apply_disturbance=True, local_time_history=history,
    )
    assert raw.values == bound.values == ost_penetration_values_from_samples(
        samples=trace.samples, rigidity_grid=grid, apply_disturbance=True, local_time_history=history,
    )
    assert "no qualified derivation" in " ".join(raw.incomplete_reasons)
    assert "local-time convention" in " ".join(bound.incomplete_reasons)
    assert set(history.incomplete_reasons).issubset(bound.incomplete_reasons)
    assert "Test ephemeris is synthetic." in bound.incomplete_reasons
    params = dict(bound.parameters)
    assert params["mlt_policy"] == "trace_bound_history"
    assert params["local_time_normative_qualified"] == "false"
    assert params["local_time_model"] == history.model
    assert params["local_time_samples_sha256"] == history.sample_fingerprint
    assert params["local_time_values_sha256"] == dict(raw.parameters)["local_time_values_sha256"]
    results = (
        build_ost_penetration_function_for_orbit(
            orbit, rigidity_grid=grid, apply_disturbance=True, local_time_history=history,
            min_days=1e-9, min_orbits=1, steps_per_orbit=12,
        ),
        build_ost_penetration_function_for_config(
            CalculationConfig(MissionConfig(2026, 5), orbit),
            rigidity_grid=grid, apply_disturbance=True, local_time_history=history,
            min_days=1e-9, min_orbits=1, steps_per_orbit=12,
        ),
    )
    for result in results:
        assert result.values == bound.values
        assert set(bound.incomplete_reasons).issubset(result.incomplete_reasons)
        assert any("representative of the mission" in reason for reason in result.incomplete_reasons)


@pytest.mark.parametrize("change", ["order", "time", "longitude"])
def test_same_length_history_for_different_trace_is_rejected(change: str) -> None:
    samples = sample_ost_orbit(
        OrbitConfig.circular(400.0, 51.6), min_days=1e-9, min_orbits=1, steps_per_orbit=12,
    ).samples
    history = build_centered_dipole_mlt_history(
        samples=samples, sun_directions_ecef=((1.0, 0.0, 0.0),) * len(samples),
        north_axis_ecef=(0.0, 0.0, 1.0), sun_model="synthetic", time_reference="test",
    )
    changed = (samples[1], samples[0], *samples[2:]) if change == "order" else (
        replace(samples[0], time_s=1.0) if change == "time"
        else replace(samples[0], longitude_deg=1.0),
        *samples[1:],
    )
    with pytest.raises(ValueError):
        build_ost_penetration_function_from_samples(
            samples=changed, rigidity_grid=RigidityGrid((0.01, 1.0)),
            apply_disturbance=True, local_time_history=history,
        )


def test_conflicting_or_quiet_history_input_is_rejected() -> None:
    samples = sample_ost_orbit(
        OrbitConfig.circular(400.0, 51.6), min_days=1e-9, min_orbits=1, steps_per_orbit=12,
    ).samples
    history = build_centered_dipole_mlt_history(
        samples=samples, sun_directions_ecef=((1.0, 0.0, 0.0),) * len(samples),
        north_axis_ecef=(0.0, 0.0, 1.0), sun_model="synthetic", time_reference="test",
    )
    with pytest.raises(ValueError, match="either"):
        build_ost_penetration_function_from_samples(
            samples=samples, rigidity_grid=RigidityGrid((0.01, 1.0)), apply_disturbance=True,
            local_time_history=history, magnetic_local_times_hours=history.values_hours,
        )
    with pytest.raises(ValueError, match="require disturbance"):
        build_ost_penetration_function_from_samples(
            samples=samples, rigidity_grid=RigidityGrid((0.01, 1.0)), local_time_history=history,
        )
