"""Orbit sampling utilities for Earth radiation belt models."""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    acos,
    atan2,
    cos,
    isfinite,
    pi,
    radians,
    sin,
    sqrt,
)

from radar.erb.constants import EARTH_REFERENCE_RADIUS_KM


@dataclass(frozen=True)
class ErbOrbitSample:
    """One spherical geocentric orbit sample."""

    radius_km: float
    theta_rad: float
    longitude_rad: float

    def __post_init__(self) -> None:
        if not isfinite(self.radius_km):
            msg = "ERB orbit sample radius must be finite."
            raise ValueError(msg)

        if self.radius_km <= 0.0:
            msg = "ERB orbit sample radius must be positive."
            raise ValueError(msg)

        if not isfinite(self.theta_rad):
            msg = "ERB orbit sample colatitude must be finite."
            raise ValueError(msg)

        if not 0.0 <= self.theta_rad <= pi:
            msg = "ERB orbit sample colatitude must be in the range 0..pi."
            raise ValueError(msg)

        if not isfinite(self.longitude_rad):
            msg = "ERB orbit sample longitude must be finite."
            raise ValueError(msg)


@dataclass(frozen=True)
class ErbOrbitGrid:
    """Deterministic grid of orbit samples."""

    samples: tuple[ErbOrbitSample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            msg = "ERB orbit grid must contain at least one sample."
            raise ValueError(msg)


def _validate_orbit_inputs(
    *,
    perigee_km: float,
    apogee_km: float,
    inclination_deg: float,
    anomaly_samples: int,
    node_samples: int,
) -> None:
    if not isfinite(perigee_km) or not isfinite(apogee_km):
        msg = "ERB orbit perigee and apogee must be finite."
        raise ValueError(msg)

    if perigee_km < 0.0 or apogee_km < 0.0:
        msg = "ERB orbit perigee and apogee altitudes must be non-negative."
        raise ValueError(msg)

    if apogee_km < perigee_km:
        msg = "ERB orbit apogee must not be below perigee."
        raise ValueError(msg)

    if not isfinite(inclination_deg):
        msg = "ERB orbit inclination must be finite."
        raise ValueError(msg)

    if not 0.0 <= inclination_deg <= 180.0:
        msg = "ERB orbit inclination must be in the range 0..180 degrees."
        raise ValueError(msg)

    if anomaly_samples < 1:
        msg = "ERB orbit anomaly sample count must be positive."
        raise ValueError(msg)

    if node_samples < 1:
        msg = "ERB orbit node sample count must be positive."
        raise ValueError(msg)


def _solve_kepler(mean_anomaly_rad: float, eccentricity: float) -> float:
    eccentric_anomaly = mean_anomaly_rad

    for _ in range(12):
        denominator = 1.0 - eccentricity * cos(eccentric_anomaly)
        eccentric_anomaly -= (
            eccentric_anomaly
            - eccentricity * sin(eccentric_anomaly)
            - mean_anomaly_rad
        ) / denominator

    return eccentric_anomaly


def sample_erb_orbit_grid(
    *,
    perigee_km: float,
    apogee_km: float,
    inclination_deg: float,
    argument_of_perigee_deg: float = 0.0,
    anomaly_samples: int = 48,
    node_samples: int = 36,
) -> ErbOrbitGrid:
    """Sample an orbit over mean anomaly and RAAN.

    Mean-anomaly sampling is used so elliptical orbits are sampled uniformly in
    time to first order, matching the ERB reference model workflow.
    """

    _validate_orbit_inputs(
        perigee_km=perigee_km,
        apogee_km=apogee_km,
        inclination_deg=inclination_deg,
        anomaly_samples=anomaly_samples,
        node_samples=node_samples,
    )

    if not isfinite(argument_of_perigee_deg):
        msg = "ERB orbit argument of perigee must be finite."
        raise ValueError(msg)

    inclination = radians(inclination_deg)
    argument_of_perigee = radians(argument_of_perigee_deg)

    perigee_radius = EARTH_REFERENCE_RADIUS_KM + perigee_km
    apogee_radius = EARTH_REFERENCE_RADIUS_KM + apogee_km
    semimajor_axis = 0.5 * (perigee_radius + apogee_radius)
    eccentricity = (apogee_radius - perigee_radius) / (apogee_radius + perigee_radius)

    samples: list[ErbOrbitSample] = []

    for node_index in range(node_samples):
        raan = 2.0 * pi * node_index / node_samples

        for anomaly_index in range(anomaly_samples):
            mean_anomaly = 2.0 * pi * anomaly_index / anomaly_samples
            eccentric_anomaly = _solve_kepler(mean_anomaly, eccentricity)

            true_anomaly = atan2(
                sqrt(1.0 - eccentricity * eccentricity) * sin(eccentric_anomaly),
                cos(eccentric_anomaly) - eccentricity,
            )
            radius = semimajor_axis * (1.0 - eccentricity * cos(eccentric_anomaly))
            argument_of_latitude = argument_of_perigee + true_anomaly

            x = radius * (
                cos(raan) * cos(argument_of_latitude)
                - sin(raan) * sin(argument_of_latitude) * cos(inclination)
            )
            y = radius * (
                sin(raan) * cos(argument_of_latitude)
                + cos(raan) * sin(argument_of_latitude) * cos(inclination)
            )
            z = radius * sin(argument_of_latitude) * sin(inclination)

            theta = acos(max(-1.0, min(1.0, z / radius)))
            longitude = atan2(y, x)

            samples.append(
                ErbOrbitSample(
                    radius_km=radius,
                    theta_rad=theta,
                    longitude_rad=longitude,
                )
            )

    return ErbOrbitGrid(samples=tuple(samples))


__all__ = [
    "ErbOrbitGrid",
    "ErbOrbitSample",
    "sample_erb_orbit_grid",
]
