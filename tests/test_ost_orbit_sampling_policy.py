"""Finite orbit-cell quadrature, not mission or full-spectrum qualification."""

from dataclasses import replace
import math

import pytest

from radar.core.project import OrbitConfig
from radar.core.types import OrbitType
from radar.geomagnetic import ost_orbit as orbit_module
from radar.geomagnetic.ost import EARTH_RADIUS_KM
from radar.geomagnetic.ost_orbit import (
    OST_EARTH_ROTATION_RATE_RAD_S,
    OST_GRAVITATIONAL_PARAMETER_KM3_S2,
    OST_ORBIT_SAMPLING_MODEL_VERSION,
    OST_SECULAR_PRECESSION_PARAMETER,
    sample_ost_orbit,
)


def _heo(inclination: float = 63.4) -> OrbitConfig:
    return OrbitConfig(
        orbit_type=OrbitType.ELLIPTIC, perigee_altitude_km=500.0,
        apogee_altitude_km=39700.0, inclination_deg=inclination,
        argument_of_perigee_deg=0.0,
    )


@pytest.mark.parametrize(("altitude", "inclination", "expected_count"), [
    (400.0, 51.6, 14959), (35786.0, 0.0, 963),
])
def test_circular_defaults_preserve_legacy_left_endpoint_coordinates(
    altitude: float, inclination: float, expected_count: int,
) -> None:
    trace = sample_ost_orbit(OrbitConfig.circular(altitude, inclination))
    radius = EARTH_RADIUS_KM + altitude
    angular_rate = math.sqrt(OST_GRAVITATIONAL_PARAMETER_KM3_S2 / radius**3)
    period = 2.0 * math.pi / angular_rate
    legacy_step = period / 60
    legacy_count = math.ceil(16.0 * 86400.0 / legacy_step)
    cos_i, sin_i = math.cos(math.radians(inclination)), math.sin(math.radians(inclination))
    node_rate = -OST_SECULAR_PRECESSION_PARAMETER * cos_i / radius**2 / period
    perigee_rate = (
        OST_SECULAR_PRECESSION_PARAMETER * (5.0 * cos_i**2 - 1.0)
        / (2.0 * radius**2) / period
    )
    assert len(trace.samples) == legacy_count == expected_count
    assert trace.step_s == legacy_step
    assert trace.requested_window_s == 16.0 * 86400.0
    for index, sample in enumerate(trace.samples):
        time = index * legacy_step
        # Independent circular geometry, using the same retained D.9-D.10
        # rates. No call to the sampler's Kepler solver or position helpers.
        u = angular_rate * time % (2.0 * math.pi) + perigee_rate * time
        node = node_rate * time
        x = math.cos(u) * math.cos(node) - math.sin(u) * math.sin(node) * cos_i
        y = math.cos(u) * math.sin(node) + math.sin(u) * math.cos(node) * cos_i
        latitude = math.degrees(math.asin(math.sin(u) * sin_i))
        longitude = math.degrees(math.atan2(y, x) - OST_EARTH_ROTATION_RATE_RAD_S * time) % 360
        assert sample.time_s == time
        assert sample.altitude_km == pytest.approx(altitude, abs=1e-10)
        assert sample.latitude_deg == pytest.approx(latitude, abs=2e-11)
        wrapped_error = (sample.longitude_deg - longitude + 180.0) % 360.0 - 180.0
        assert abs(wrapped_error) < 2e-11


@pytest.mark.parametrize(("steps", "expected_count"), [(60, 1936), (120, 3871), (240, 7741)])
def test_heo_density_refinement_really_refines_with_no_sample_cap(
    steps: int, expected_count: int,
) -> None:
    trace = sample_ost_orbit(_heo(), steps_per_orbit=steps, max_samples=30000)
    assert len(trace.samples) == expected_count
    assert trace.step_s == trace.orbital_period_s / steps
    assert trace.requested_window_s == 16.0 * 86400.0
    assert trace.requested_window_s <= trace.window_s < trace.requested_window_s + trace.step_s
    assert trace.window_s == expected_count * trace.step_s
    assert trace.sampling_policy == OST_ORBIT_SAMPLING_MODEL_VERSION


@pytest.mark.parametrize("inclination", [
    0.0, 63.4, math.degrees(math.acos(math.sqrt(0.2))),
    63.5, 90.0, 116.5, 180.0 - math.degrees(math.acos(math.sqrt(0.2))), 116.6,
])
def test_near_critical_and_retrograde_orbits_do_not_expand_window(inclination: float) -> None:
    trace = sample_ost_orbit(_heo(inclination))
    assert len(trace.samples) == 1936
    assert trace.window_s < 16.01 * 86400.0
    assert trace.step_s < trace.orbital_period_s
    assert all(math.isfinite(sample.longitude_deg) for sample in trace.samples)


@pytest.mark.parametrize("min_orbits", [1, 10, 17])
def test_minimum_orbit_count_uses_exact_integer_number_of_cells(min_orbits: int) -> None:
    trace = sample_ost_orbit(_heo(), min_days=1e-10, min_orbits=min_orbits, steps_per_orbit=60)
    assert len(trace.samples) == 60 * min_orbits
    assert trace.minimum_orbits == min_orbits
    assert trace.requested_window_s == min_orbits * trace.orbital_period_s
    assert trace.window_s == len(trace.samples) * trace.step_s
    assert trace.samples[0].time_s == 0.0
    assert trace.samples[-1].time_s < trace.window_s
    assert math.fsum(trace.step_s for _ in trace.samples) == pytest.approx(trace.window_s)


def test_budget_is_exact_and_refusal_occurs_before_any_position_is_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = sample_ost_orbit(_heo(), min_days=1e-6, steps_per_orbit=60, max_samples=600)
    assert len(trace.samples) == 600

    def forbidden_position(*args: object) -> float:
        pytest.fail("Over-budget trace must not evaluate an orbital position.")

    monkeypatch.setattr(orbit_module, "_solve_kepler_elliptic", forbidden_position)
    with pytest.raises(ValueError, match="budget exceeded.*600 samples.*599"):
        sample_ost_orbit(_heo(), min_days=1e-6, steps_per_orbit=60, max_samples=599)
    with pytest.raises(ValueError, match="budget exceeded.*1936 samples.*1935"):
        sample_ost_orbit(_heo(), steps_per_orbit=60, max_samples=1935)


@pytest.mark.parametrize(("name", "value"), [
    ("min_orbits", 0), ("min_orbits", -1), ("min_orbits", True),
    ("min_orbits", 1.5), ("min_orbits", float("nan")),
    ("min_days", True), ("min_days", 0.0), ("min_days", -1.0),
    ("min_days", float("nan")), ("min_days", float("inf")), ("min_days", 1e308),
    ("raan_deg", True), ("raan_deg", float("nan")), ("raan_deg", float("inf")),
    ("steps_per_orbit", True), ("steps_per_orbit", 4.5), ("steps_per_orbit", 3),
    ("max_samples", True), ("max_samples", 3.5), ("max_samples", 3),
])
def test_invalid_sampling_controls_fail_before_positions(
    monkeypatch: pytest.MonkeyPatch, name: str, value: object,
) -> None:
    monkeypatch.setattr(orbit_module, "_solve_kepler_elliptic", lambda *args: pytest.fail("sampled"))
    options = {name: value}
    with pytest.raises(ValueError):
        sample_ost_orbit(_heo(), **options)  # type: ignore[arg-type]


def test_unrepresentable_period_is_a_descriptive_error() -> None:
    with pytest.raises(ValueError, match="period cannot be represented"):
        sample_ost_orbit(OrbitConfig.circular(1e200, 0.0))


@pytest.mark.parametrize("option", ["steps_per_orbit", "min_orbits"])
def test_unrepresentable_integer_controls_fail_before_positions(
    monkeypatch: pytest.MonkeyPatch, option: str,
) -> None:
    monkeypatch.setattr(orbit_module, "_solve_kepler_elliptic", lambda *args: pytest.fail("sampled"))
    options = {option: 10**400}
    with pytest.raises(ValueError, match="cannot be represented"):
        sample_ost_orbit(_heo(), max_samples=10**500, **options)


def test_printed_precession_arithmetic_is_unchanged() -> None:
    trace = sample_ost_orbit(_heo(), min_days=1e-6, min_orbits=1)
    a, e, period = trace.semi_major_axis_km, trace.eccentricity, trace.orbital_period_s
    cos_i = math.cos(math.radians(63.4))
    denominator = a**2 * (1 - e**2)**2
    assert trace.node_precession_rad_s == (
        -OST_SECULAR_PRECESSION_PARAMETER * (1 - e) * cos_i / denominator / period
    )
    assert trace.perigee_precession_rad_s == (
        OST_SECULAR_PRECESSION_PARAMETER * (1 - e) * (5 * cos_i**2 - 1)
        / (2 * denominator) / period
    )


@pytest.mark.parametrize("metadata", [
    {"requested_window_s": 0.0}, {"requested_window_s": float("nan")},
    {"requested_window_s": float("inf")}, {"requested_window_s": True},
    {"requested_window_s": 1e20}, {"minimum_orbits": 0}, {"minimum_orbits": True},
    {"sampling_policy": " "},
])
def test_trace_rejects_ambiguous_sampling_metadata(metadata: dict[str, object]) -> None:
    trace = sample_ost_orbit(_heo(), min_days=1e-6, min_orbits=1, steps_per_orbit=4)
    with pytest.raises(ValueError):
        replace(trace, **metadata)  # type: ignore[arg-type]


def test_legacy_synthetic_trace_can_omit_new_metadata() -> None:
    trace = sample_ost_orbit(_heo(), min_days=1e-6, min_orbits=1, steps_per_orbit=4)
    synthetic = replace(trace, requested_window_s=None, sampling_policy="unspecified")
    assert synthetic.requested_window_s is None
    assert synthetic.sampling_policy == "unspecified"
