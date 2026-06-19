"""Orbit geometry utilities for RADAR.

This module contains only two-body geometric orbit helpers.
Geomagnetic coordinates and cutoff rigidity are calculated in separate modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, degrees, isfinite, radians, sin

from radar.core.project import OrbitConfig


@dataclass(frozen=True)
class OrbitGeometry:
    """Derived geometric parameters of an orbit."""

    perigee_radius_km: float
    apogee_radius_km: float
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    argument_of_perigee_deg: float

    @classmethod
    def from_config(cls, config: OrbitConfig, earth_radius_km: float) -> OrbitGeometry:
        """Build orbit geometry from user orbit configuration."""

        if not isfinite(earth_radius_km) or earth_radius_km <= 0.0:
            msg = "Earth reference radius must be positive and finite."
            raise ValueError(msg)

        perigee_radius_km = earth_radius_km + config.perigee_altitude_km
        apogee_radius_km = earth_radius_km + config.apogee_altitude_km

        semi_major_axis_km = 0.5 * (perigee_radius_km + apogee_radius_km)
        eccentricity = (apogee_radius_km - perigee_radius_km) / (
            apogee_radius_km + perigee_radius_km
        )

        return cls(
            perigee_radius_km=perigee_radius_km,
            apogee_radius_km=apogee_radius_km,
            semi_major_axis_km=semi_major_axis_km,
            eccentricity=eccentricity,
            inclination_deg=config.inclination_deg,
            argument_of_perigee_deg=config.argument_of_perigee_deg,
        )

    def __post_init__(self) -> None:
        values = (
            self.perigee_radius_km,
            self.apogee_radius_km,
            self.semi_major_axis_km,
            self.eccentricity,
            self.inclination_deg,
            self.argument_of_perigee_deg,
        )

        if any(not isfinite(value) for value in values):
            msg = "Orbit geometry values must be finite."
            raise ValueError(msg)

        if self.perigee_radius_km <= 0.0:
            msg = "Perigee radius must be positive."
            raise ValueError(msg)

        if self.apogee_radius_km < self.perigee_radius_km:
            msg = "Apogee radius must be greater than or equal to perigee radius."
            raise ValueError(msg)

        if self.semi_major_axis_km <= 0.0:
            msg = "Semi-major axis must be positive."
            raise ValueError(msg)

        if not 0.0 <= self.eccentricity < 1.0:
            msg = "Eccentricity must be in the range 0..1."
            raise ValueError(msg)

        if not 0.0 <= self.inclination_deg <= 180.0:
            msg = "Inclination must be in the range 0..180 degrees."
            raise ValueError(msg)

        if not 0.0 <= self.argument_of_perigee_deg < 360.0:
            msg = "Argument of perigee must be in the range 0..360 degrees."
            raise ValueError(msg)

    def radius_at_true_anomaly(self, true_anomaly_deg: float) -> float:
        """Return geocentric radius at true anomaly."""

        if not isfinite(true_anomaly_deg):
            msg = "True anomaly must be finite."
            raise ValueError(msg)

        true_anomaly_rad = radians(true_anomaly_deg)
        semi_latus_rectum_km = self.semi_major_axis_km * (1.0 - self.eccentricity**2)
        denominator = 1.0 + self.eccentricity * cos(true_anomaly_rad)

        if denominator <= 0.0:
            msg = "Invalid orbit radius denominator."
            raise ValueError(msg)

        return semi_latus_rectum_km / denominator

    def geocentric_latitude_at_true_anomaly(self, true_anomaly_deg: float) -> float:
        """Return geocentric latitude in a simple inclined-orbit approximation."""

        if not isfinite(true_anomaly_deg):
            msg = "True anomaly must be finite."
            raise ValueError(msg)

        inclination_rad = radians(self.inclination_deg)
        argument_of_latitude_rad = radians(self.argument_of_perigee_deg + true_anomaly_deg)

        latitude_rad = asin(sin(inclination_rad) * sin(argument_of_latitude_rad))

        return degrees(latitude_rad)
