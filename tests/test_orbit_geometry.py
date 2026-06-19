import pytest

from radar.core.project import OrbitConfig
from radar.core.types import OrbitType
from radar.orbit.geometry import OrbitGeometry
from radar.orbit.trace import sample_orbit

EARTH_RADIUS_KM = 6371.0


def test_circular_orbit_geometry() -> None:
    config = OrbitConfig.circular(altitude_km=500.0, inclination_deg=82.0)

    geometry = OrbitGeometry.from_config(
        config=config,
        earth_radius_km=EARTH_RADIUS_KM,
    )

    assert geometry.perigee_radius_km == pytest.approx(6871.0)
    assert geometry.apogee_radius_km == pytest.approx(6871.0)
    assert geometry.semi_major_axis_km == pytest.approx(6871.0)
    assert geometry.eccentricity == pytest.approx(0.0)

    assert geometry.radius_at_true_anomaly(0.0) == pytest.approx(6871.0)
    assert geometry.radius_at_true_anomaly(90.0) == pytest.approx(6871.0)
    assert geometry.radius_at_true_anomaly(180.0) == pytest.approx(6871.0)


def test_elliptic_orbit_radius_at_perigee_and_apogee() -> None:
    config = OrbitConfig(
        orbit_type=OrbitType.ELLIPTIC,
        perigee_altitude_km=500.0,
        apogee_altitude_km=1500.0,
        inclination_deg=63.4,
        argument_of_perigee_deg=0.0,
    )

    geometry = OrbitGeometry.from_config(
        config=config,
        earth_radius_km=EARTH_RADIUS_KM,
    )

    assert geometry.perigee_radius_km == pytest.approx(6871.0)
    assert geometry.apogee_radius_km == pytest.approx(7871.0)

    assert geometry.radius_at_true_anomaly(0.0) == pytest.approx(6871.0)
    assert geometry.radius_at_true_anomaly(180.0) == pytest.approx(7871.0)


def test_geocentric_latitude_for_polar_orbit() -> None:
    config = OrbitConfig.circular(altitude_km=500.0, inclination_deg=90.0)

    geometry = OrbitGeometry.from_config(
        config=config,
        earth_radius_km=EARTH_RADIUS_KM,
    )

    assert geometry.geocentric_latitude_at_true_anomaly(0.0) == pytest.approx(0.0)
    assert geometry.geocentric_latitude_at_true_anomaly(90.0) == pytest.approx(90.0)
    assert geometry.geocentric_latitude_at_true_anomaly(180.0) == pytest.approx(0.0)
    assert geometry.geocentric_latitude_at_true_anomaly(270.0) == pytest.approx(-90.0)


def test_sample_orbit_returns_expected_points() -> None:
    config = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    trace = sample_orbit(
        config=config,
        earth_radius_km=EARTH_RADIUS_KM,
        point_count=4,
    )

    assert len(trace.points) == 4
    assert tuple(point.true_anomaly_deg for point in trace.points) == (
        0.0,
        90.0,
        180.0,
        270.0,
    )
    assert trace.altitudes_km() == pytest.approx((35786.0, 35786.0, 35786.0, 35786.0))
    assert trace.geocentric_latitudes_deg() == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_sample_orbit_requires_at_least_four_points() -> None:
    config = OrbitConfig.circular(altitude_km=500.0, inclination_deg=0.0)

    with pytest.raises(ValueError, match="at least 4"):
        sample_orbit(
            config=config,
            earth_radius_km=EARTH_RADIUS_KM,
            point_count=3,
        )


def test_orbit_geometry_rejects_bad_earth_radius() -> None:
    config = OrbitConfig.circular(altitude_km=500.0, inclination_deg=0.0)

    with pytest.raises(ValueError, match="Earth"):
        OrbitGeometry.from_config(
            config=config,
            earth_radius_km=0.0,
        )
