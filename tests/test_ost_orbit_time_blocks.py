"""Global-phase block identity without qualifying a mission time model."""

from dataclasses import replace
import math

import pytest

from radar.core.project import OrbitConfig
from radar.core.types import OrbitType
from radar.geomagnetic import ost_orbit as orbit_module
from radar.geomagnetic.ost_orbit import (
    OST_ORBIT_SAMPLING_MODEL_VERSION,
    OST_ORBIT_TIME_ORIGIN_MODEL_VERSION,
    sample_ost_orbit,
)


def _heo(inclination: float = 63.4) -> OrbitConfig:
    return OrbitConfig(
        orbit_type=OrbitType.ELLIPTIC, perigee_altitude_km=500.0,
        apogee_altitude_km=39700.0, inclination_deg=inclination,
        argument_of_perigee_deg=43.0,
    )


@pytest.mark.parametrize("orbit", [
    OrbitConfig.circular(400.0, 51.6), OrbitConfig.circular(35786.0, 0.0),
    _heo(), _heo(math.degrees(math.acos(math.sqrt(0.2)))),
    _heo(180.0 - math.degrees(math.acos(math.sqrt(0.2)))),
])
@pytest.mark.parametrize("start_index", [0, 137, 2_000_003])
def test_nonoverlapping_blocks_are_bit_exact_slices_of_one_global_trace(
    orbit: OrbitConfig, start_index: int,
) -> None:
    whole = sample_ost_orbit(
        orbit, min_orbits=8, start_sample_index=start_index,
        steps_per_orbit=24, min_days=1e-8, raan_deg=37.0,
    )
    left = sample_ost_orbit(
        orbit, min_orbits=3, start_sample_index=start_index,
        steps_per_orbit=24, min_days=1e-8, raan_deg=37.0,
    )
    right = sample_ost_orbit(
        orbit, min_orbits=5, start_sample_index=start_index + len(left.samples),
        steps_per_orbit=24, min_days=1e-8, raan_deg=37.0,
    )
    combined = (*left.samples, *right.samples)
    assert len(combined) == len(whole.samples)
    assert left.end_time_s == right.start_time_s
    assert whole.start_time_s == left.start_time_s
    assert whole.end_time_s == right.end_time_s
    for index, (actual, expected) in enumerate(zip(combined, whole.samples, strict=True)):
        assert replace(actual, index=index) == expected
        assert actual.time_s == (start_index + index) * whole.step_s
    assert tuple(sample.index for sample in right.samples) == tuple(range(len(right.samples)))
    assert left.window_s + right.window_s == pytest.approx(whole.window_s)
    assert whole.sampling_policy == OST_ORBIT_SAMPLING_MODEL_VERSION
    assert OST_ORBIT_TIME_ORIGIN_MODEL_VERSION == "global_sample_index_v1"


def test_zero_offset_preserves_default_trace_exactly() -> None:
    assert sample_ost_orbit(_heo(), min_days=1e-8, min_orbits=2, steps_per_orbit=60) == sample_ost_orbit(
        _heo(), start_sample_index=0, min_days=1e-8, min_orbits=2, steps_per_orbit=60,
    )


def test_shifted_block_does_not_restart_mean_anomaly_or_secular_or_earth_phases() -> None:
    orbit = replace(_heo(), argument_of_perigee_deg=0.0)
    first = sample_ost_orbit(orbit, min_days=1e-8, min_orbits=1, steps_per_orbit=60)
    # About a year later, with a non-integral orbital phase offset.
    offset = 60_017
    later = sample_ost_orbit(
        orbit, min_days=1e-8, min_orbits=1, steps_per_orbit=60, start_sample_index=offset,
    )
    assert later.start_time_s == offset * first.step_s
    assert later.end_time_s == (offset + len(later.samples)) * first.step_s
    assert later.window_s == first.window_s
    assert later.samples[0].radius_km != pytest.approx(first.samples[0].radius_km)
    assert later.samples[0].latitude_deg != pytest.approx(first.samples[0].latitude_deg)
    assert later.samples[0].longitude_deg != pytest.approx(first.samples[0].longitude_deg)
    assert later.samples[0].true_anomaly_deg != pytest.approx(first.samples[0].true_anomaly_deg)


@pytest.mark.parametrize("raan", [45.0, 90.0, 273.0])
def test_raan_is_pure_geographic_longitude_rotation_not_a_latitude_or_radius_bug(
    raan: float,
) -> None:
    initial = sample_ost_orbit(
        _heo(), raan_deg=0.0, min_days=1e-8, min_orbits=2, start_sample_index=60_017,
    )
    rotated = sample_ost_orbit(
        _heo(), raan_deg=raan, min_days=1e-8, min_orbits=2, start_sample_index=60_017,
    )
    for a, b in zip(initial.samples, rotated.samples, strict=True):
        assert a.time_s == b.time_s
        assert a.radius_km == b.radius_km
        assert a.latitude_deg == b.latitude_deg
        wrapped_error = (b.longitude_deg - a.longitude_deg - raan + 180.0) % 360.0 - 180.0
        assert abs(wrapped_error) < 1e-10


@pytest.mark.parametrize("index", [-1, True, False, 1.5, float("nan"), float("inf")])
def test_bad_offset_is_rejected_before_position_evaluation(
    monkeypatch: pytest.MonkeyPatch, index: object,
) -> None:
    monkeypatch.setattr(orbit_module, "_solve_kepler_elliptic", lambda *args: pytest.fail("sampled"))
    with pytest.raises(ValueError, match="start sample index"):
        sample_ost_orbit(_heo(), start_sample_index=index)  # type: ignore[arg-type]


@pytest.mark.parametrize(("index", "message"), [
    (10**400, "cannot be represented"), (10**308, "block times must be finite"),
    (2**60, "global time resolution"),
])
def test_overflow_and_time_resolution_loss_fail_before_position_evaluation(
    monkeypatch: pytest.MonkeyPatch, index: int, message: str,
) -> None:
    monkeypatch.setattr(orbit_module, "_solve_kepler_elliptic", lambda *args: pytest.fail("sampled"))
    with pytest.raises(ValueError, match=message):
        sample_ost_orbit(_heo(), start_sample_index=index)


def test_offset_does_not_bypass_the_per_block_sample_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orbit_module, "_solve_kepler_elliptic", lambda *args: pytest.fail("sampled"))
    with pytest.raises(ValueError, match="sample budget exceeded"):
        sample_ost_orbit(_heo(), max_samples=599, start_sample_index=1_000_000)


@pytest.mark.parametrize("index", [-1, True, 1.5, float("nan"), 10**400, 2**60])
def test_trace_metadata_rejects_invalid_global_time_origin(index: object) -> None:
    trace = sample_ost_orbit(_heo(), min_days=1e-8, min_orbits=1, steps_per_orbit=4)
    with pytest.raises(ValueError):
        replace(trace, start_sample_index=index)  # type: ignore[arg-type]


def test_declared_sampler_trace_cannot_relabel_unchanged_samples_with_a_new_offset() -> None:
    trace = sample_ost_orbit(_heo(), min_days=1e-8, min_orbits=1, steps_per_orbit=4)
    with pytest.raises(ValueError, match="global cell indices"):
        replace(trace, start_sample_index=1)
