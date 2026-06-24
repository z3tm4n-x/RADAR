import pytest

from radar.core.project import OrbitConfig
from radar.geomagnetic.ost import (
    EARTH_RADIUS_KM,
    OST_CUTOFF_REFERENCE_ALTITUDE_KM,
    ost_earth_shadow_factor,
)
from radar.geomagnetic.ost_orbit import OstOrbitSample
from radar.geomagnetic.ost_penetration import (
    OST_GEOMAGNETIC_PENETRATION_MODEL,
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
            weight / 2.0,
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
