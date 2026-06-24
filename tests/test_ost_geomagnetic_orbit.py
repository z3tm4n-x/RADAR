import pytest

from radar.core.project import OrbitConfig
from radar.core.types import OrbitType
from radar.geomagnetic.ost import EARTH_RADIUS_KM
from radar.geomagnetic.ost_orbit import sample_ost_orbit


def test_sample_ost_orbit_equatorial_circular_trace() -> None:
    trace = sample_ost_orbit(
        OrbitConfig.circular(altitude_km=500.0, inclination_deg=0.0),
        min_days=0.3,
        steps_per_orbit=12,
    )

    assert len(trace.samples) >= 4
    assert trace.eccentricity == pytest.approx(0.0)
    assert all(altitude == pytest.approx(500.0) for altitude in trace.altitudes_km())
    assert all(latitude == pytest.approx(0.0) for latitude in trace.latitudes_deg())
    assert all(0.0 <= longitude < 360.0 for longitude in trace.longitudes_deg())


def test_sample_ost_orbit_uses_raan_for_initial_longitude() -> None:
    trace = sample_ost_orbit(
        OrbitConfig.circular(altitude_km=500.0, inclination_deg=0.0),
        raan_deg=30.0,
        min_days=0.1,
        steps_per_orbit=12,
    )

    assert trace.samples[0].longitude_deg == pytest.approx(30.0)


def test_sample_ost_orbit_polar_circular_latitude_extrema() -> None:
    trace = sample_ost_orbit(
        OrbitConfig.circular(altitude_km=500.0, inclination_deg=90.0),
        min_days=0.1,
        steps_per_orbit=4,
    )

    latitudes = trace.latitudes_deg()

    assert max(latitudes) == pytest.approx(90.0, abs=0.2)
    assert min(latitudes) == pytest.approx(-90.0, abs=0.2)


def test_sample_ost_orbit_elliptic_altitude_range() -> None:
    trace = sample_ost_orbit(
        OrbitConfig(
            orbit_type=OrbitType.ELLIPTIC,
            perigee_altitude_km=500.0,
            apogee_altitude_km=1500.0,
            inclination_deg=63.4,
        ),
        min_days=0.2,
        steps_per_orbit=60,
    )

    assert min(trace.altitudes_km()) == pytest.approx(500.0, abs=1.0)
    assert max(trace.altitudes_km()) == pytest.approx(1500.0, abs=1.0)


def test_sample_ost_orbit_expands_window_for_elliptic_perigee_drift() -> None:
    trace = sample_ost_orbit(
        OrbitConfig(
            orbit_type=OrbitType.ELLIPTIC,
            perigee_altitude_km=500.0,
            apogee_altitude_km=1500.0,
            inclination_deg=63.4,
        ),
        min_days=0.1,
        steps_per_orbit=60,
    )

    assert trace.window_s >= 0.1 * 86400.0
    assert trace.perigee_precession_rad_s != pytest.approx(0.0)


def test_sample_ost_orbit_rejects_bad_sampling_inputs() -> None:
    orbit = OrbitConfig.circular(altitude_km=500.0, inclination_deg=0.0)

    with pytest.raises(ValueError, match="steps per orbit"):
        sample_ost_orbit(orbit, steps_per_orbit=3)

    with pytest.raises(ValueError, match="maximum sample count"):
        sample_ost_orbit(orbit, max_samples=3)

    with pytest.raises(ValueError, match="minimum window"):
        sample_ost_orbit(orbit, min_days=0.0)


def test_sample_ost_orbit_uses_ost_earth_radius() -> None:
    trace = sample_ost_orbit(
        OrbitConfig.circular(altitude_km=500.0, inclination_deg=0.0),
        min_days=0.1,
        steps_per_orbit=12,
    )

    assert trace.samples[0].radius_km == pytest.approx(EARTH_RADIUS_KM + 500.0)
