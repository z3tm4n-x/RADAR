"""Orbit trace sampling utilities for RADAR."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from radar.core.project import OrbitConfig
from radar.orbit.geometry import OrbitGeometry


@dataclass(frozen=True)
class OrbitPoint:
    """One sampled point of an orbit."""

    index: int
    true_anomaly_deg: float
    radius_km: float
    altitude_km: float
    geocentric_latitude_deg: float

    def __post_init__(self) -> None:
        if self.index < 0:
            msg = "Orbit point index must be non-negative."
            raise ValueError(msg)

        values = (
            self.true_anomaly_deg,
            self.radius_km,
            self.altitude_km,
            self.geocentric_latitude_deg,
        )

        if any(not isfinite(value) for value in values):
            msg = "Orbit point values must be finite."
            raise ValueError(msg)

        if self.radius_km <= 0.0:
            msg = "Orbit point radius must be positive."
            raise ValueError(msg)


@dataclass(frozen=True)
class OrbitTrace:
    """Sampled orbit trace."""

    geometry: OrbitGeometry
    earth_radius_km: float
    points: tuple[OrbitPoint, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.earth_radius_km) or self.earth_radius_km <= 0.0:
            msg = "Earth reference radius must be positive and finite."
            raise ValueError(msg)

        if not self.points:
            msg = "Orbit trace must contain at least one point."
            raise ValueError(msg)

    def radii_km(self) -> tuple[float, ...]:
        """Return sampled geocentric radii."""

        return tuple(point.radius_km for point in self.points)

    def altitudes_km(self) -> tuple[float, ...]:
        """Return sampled altitudes."""

        return tuple(point.altitude_km for point in self.points)

    def geocentric_latitudes_deg(self) -> tuple[float, ...]:
        """Return sampled geocentric latitudes."""

        return tuple(point.geocentric_latitude_deg for point in self.points)


def sample_orbit(
    config: OrbitConfig,
    earth_radius_km: float,
    point_count: int,
) -> OrbitTrace:
    """Sample orbit points uniformly by true anomaly."""

    if not isinstance(point_count, int):
        msg = "Orbit point count must be an integer."
        raise ValueError(msg)

    if point_count < 4:
        msg = "Orbit point count must be at least 4."
        raise ValueError(msg)

    geometry = OrbitGeometry.from_config(
        config=config,
        earth_radius_km=earth_radius_km,
    )

    points: list[OrbitPoint] = []

    for index in range(point_count):
        true_anomaly_deg = 360.0 * index / point_count
        radius_km = geometry.radius_at_true_anomaly(true_anomaly_deg)
        altitude_km = radius_km - earth_radius_km
        latitude_deg = geometry.geocentric_latitude_at_true_anomaly(true_anomaly_deg)

        points.append(
            OrbitPoint(
                index=index,
                true_anomaly_deg=true_anomaly_deg,
                radius_km=radius_km,
                altitude_km=altitude_km,
                geocentric_latitude_deg=latitude_deg,
            )
        )

    return OrbitTrace(
        geometry=geometry,
        earth_radius_km=earth_radius_km,
        points=tuple(points),
    )
