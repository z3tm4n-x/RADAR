"""OST 134-1044-2007 orbit sampling for geomagnetic penetration.

The sampler follows the reference calculation structure used for Appendix D:
Keplerian motion, Earth rotation, and secular node/perigee drift. It returns a
geographic trace used later to build the geomagnetic penetration function psi(R).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from radar.core.project import OrbitConfig
from radar.geomagnetic.ost import EARTH_RADIUS_KM

OST_GRAVITATIONAL_PARAMETER_KM3_S2 = 398600.4418
OST_EARTH_ROTATION_RATE_RAD_S = 7.2921159e-5
OST_SECULAR_PRECESSION_PARAMETER = 4.1378e5
SECONDS_PER_DAY = 86400.0
TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class OstOrbitSample:
    """One sampled geographic orbit point for OST geomagnetic calculations."""

    index: int
    time_s: float
    true_anomaly_deg: float
    radius_km: float
    altitude_km: float
    latitude_deg: float
    longitude_deg: float

    def __post_init__(self) -> None:
        if self.index < 0:
            msg = "OST orbit sample index must be non-negative."
            raise ValueError(msg)

        values = (
            self.time_s,
            self.true_anomaly_deg,
            self.radius_km,
            self.altitude_km,
            self.latitude_deg,
            self.longitude_deg,
        )

        if any(not math.isfinite(value) for value in values):
            msg = "OST orbit sample values must be finite."
            raise ValueError(msg)

        if self.time_s < 0.0:
            msg = "OST orbit sample time must be non-negative."
            raise ValueError(msg)

        if self.radius_km <= 0.0:
            msg = "OST orbit sample radius must be positive."
            raise ValueError(msg)

        if self.altitude_km < 0.0:
            msg = "OST orbit sample altitude must be non-negative."
            raise ValueError(msg)

        if not -90.0 <= self.latitude_deg <= 90.0:
            msg = "OST orbit sample latitude must be in the range -90..90 degrees."
            raise ValueError(msg)

        if not 0.0 <= self.longitude_deg < 360.0:
            msg = "OST orbit sample longitude must be in the range 0..360 degrees."
            raise ValueError(msg)


@dataclass(frozen=True)
class OstOrbitTrace:
    """Sampled OST geographic orbit trace."""

    samples: tuple[OstOrbitSample, ...]
    semi_major_axis_km: float
    eccentricity: float
    orbital_period_s: float
    window_s: float
    step_s: float
    node_precession_rad_s: float
    perigee_precession_rad_s: float

    def __post_init__(self) -> None:
        if not self.samples:
            msg = "OST orbit trace must contain at least one sample."
            raise ValueError(msg)

        values = (
            self.semi_major_axis_km,
            self.eccentricity,
            self.orbital_period_s,
            self.window_s,
            self.step_s,
            self.node_precession_rad_s,
            self.perigee_precession_rad_s,
        )

        if any(not math.isfinite(value) for value in values):
            msg = "OST orbit trace metadata values must be finite."
            raise ValueError(msg)

        if self.semi_major_axis_km <= 0.0:
            msg = "OST orbit semi-major axis must be positive."
            raise ValueError(msg)

        if not 0.0 <= self.eccentricity < 1.0:
            msg = "OST orbit eccentricity must be in the range 0..1."
            raise ValueError(msg)

        if self.orbital_period_s <= 0.0:
            msg = "OST orbit period must be positive."
            raise ValueError(msg)

        if self.window_s <= 0.0:
            msg = "OST orbit sampling window must be positive."
            raise ValueError(msg)

        if self.step_s <= 0.0:
            msg = "OST orbit sampling step must be positive."
            raise ValueError(msg)

    def altitudes_km(self) -> tuple[float, ...]:
        """Return sampled altitudes."""

        return tuple(sample.altitude_km for sample in self.samples)

    def latitudes_deg(self) -> tuple[float, ...]:
        """Return sampled geographic latitudes."""

        return tuple(sample.latitude_deg for sample in self.samples)

    def longitudes_deg(self) -> tuple[float, ...]:
        """Return sampled geographic longitudes."""

        return tuple(sample.longitude_deg for sample in self.samples)


def _validate_sampling_inputs(
    *,
    raan_deg: float,
    min_days: float,
    steps_per_orbit: int,
    max_samples: int,
) -> None:
    if not math.isfinite(raan_deg):
        msg = "RAAN must be finite."
        raise ValueError(msg)

    if not math.isfinite(min_days) or min_days <= 0.0:
        msg = "OST orbit sampling minimum window must be positive and finite."
        raise ValueError(msg)

    if not isinstance(steps_per_orbit, int):
        msg = "OST orbit steps per orbit must be an integer."
        raise ValueError(msg)

    if steps_per_orbit < 4:
        msg = "OST orbit steps per orbit must be at least 4."
        raise ValueError(msg)

    if not isinstance(max_samples, int):
        msg = "OST orbit maximum sample count must be an integer."
        raise ValueError(msg)

    if max_samples < 4:
        msg = "OST orbit maximum sample count must be at least 4."
        raise ValueError(msg)


def _solve_kepler_elliptic(
    mean_anomaly_rad: float,
    eccentricity: float,
) -> float:
    """Solve Kepler equation M = E - e sin(E) for elliptic orbit."""

    reduced_mean_anomaly = mean_anomaly_rad % TWO_PI
    eccentric_anomaly = reduced_mean_anomaly

    for _ in range(60):
        denominator = 1.0 - eccentricity * math.cos(eccentric_anomaly)

        if denominator <= 0.0:
            msg = "Invalid Kepler equation denominator."
            raise ValueError(msg)

        correction = (
            eccentric_anomaly
            - eccentricity * math.sin(eccentric_anomaly)
            - reduced_mean_anomaly
        ) / denominator
        eccentric_anomaly -= correction

        if abs(correction) < 1e-13:
            break

    return eccentric_anomaly


def _true_anomaly_from_eccentric_anomaly(
    eccentric_anomaly_rad: float,
    eccentricity: float,
) -> float:
    numerator = math.sqrt(1.0 + eccentricity) * math.sin(
        eccentric_anomaly_rad / 2.0,
    )
    denominator = math.sqrt(1.0 - eccentricity) * math.cos(
        eccentric_anomaly_rad / 2.0,
    )

    return 2.0 * math.atan2(numerator, denominator)


def sample_ost_orbit(
    config: OrbitConfig,
    *,
    raan_deg: float = 0.0,
    min_days: float = 16.0,
    steps_per_orbit: int = 60,
    max_samples: int = 400000,
) -> OstOrbitTrace:
    """Sample an OST geographic orbit trace.

    The window is at least ``min_days``. For elliptic orbits with non-zero
    secular perigee drift, the window is expanded to cover 2.5 perigee-drift
    periods, matching the reference implementation strategy.
    """

    _validate_sampling_inputs(
        raan_deg=raan_deg,
        min_days=min_days,
        steps_per_orbit=steps_per_orbit,
        max_samples=max_samples,
    )

    apogee_radius_km = EARTH_RADIUS_KM + config.apogee_altitude_km
    perigee_radius_km = EARTH_RADIUS_KM + config.perigee_altitude_km
    semi_major_axis_km = 0.5 * (apogee_radius_km + perigee_radius_km)
    eccentricity = (apogee_radius_km - perigee_radius_km) / (
        apogee_radius_km + perigee_radius_km
    )
    inclination_rad = math.radians(config.inclination_deg)

    mean_motion_rad_s = math.sqrt(
        OST_GRAVITATIONAL_PARAMETER_KM3_S2 / semi_major_axis_km**3,
    )
    orbital_period_s = TWO_PI / mean_motion_rad_s

    drift_denominator = semi_major_axis_km**2 * (1.0 - eccentricity**2) ** 2
    node_drift_per_orbit = (
        -OST_SECULAR_PRECESSION_PARAMETER
        * (1.0 - eccentricity)
        * math.cos(inclination_rad)
        / drift_denominator
    )
    perigee_drift_per_orbit = (
        OST_SECULAR_PRECESSION_PARAMETER
        * (1.0 - eccentricity)
        * (5.0 * math.cos(inclination_rad) ** 2 - 1.0)
        / (2.0 * drift_denominator)
    )

    node_precession_rad_s = node_drift_per_orbit / orbital_period_s
    perigee_precession_rad_s = perigee_drift_per_orbit / orbital_period_s

    window_s = min_days * SECONDS_PER_DAY

    if eccentricity > 1e-4 and abs(perigee_precession_rad_s) > 0.0:
        window_s = max(
            window_s,
            2.5 * TWO_PI / abs(perigee_precession_rad_s),
        )

    step_s = orbital_period_s / steps_per_orbit

    if window_s / step_s > max_samples:
        step_s = window_s / max_samples

    sample_count = max(1, int(math.ceil(window_s / step_s)))

    samples: list[OstOrbitSample] = []
    raan_rad = math.radians(raan_deg)
    argument_of_perigee_rad = math.radians(config.argument_of_perigee_deg)

    for index in range(sample_count):
        time_s = index * step_s
        mean_anomaly_rad = mean_motion_rad_s * time_s
        eccentric_anomaly_rad = _solve_kepler_elliptic(
            mean_anomaly_rad,
            eccentricity,
        )
        true_anomaly_rad = _true_anomaly_from_eccentric_anomaly(
            eccentric_anomaly_rad,
            eccentricity,
        )
        radius_km = semi_major_axis_km * (
            1.0 - eccentricity * math.cos(eccentric_anomaly_rad)
        )
        altitude_km = radius_km - EARTH_RADIUS_KM

        argument_of_latitude_rad = (
            argument_of_perigee_rad
            + perigee_precession_rad_s * time_s
            + true_anomaly_rad
        )
        node_rad = raan_rad + node_precession_rad_s * time_s

        cos_u = math.cos(argument_of_latitude_rad)
        sin_u = math.sin(argument_of_latitude_rad)
        cos_node = math.cos(node_rad)
        sin_node = math.sin(node_rad)
        cos_i = math.cos(inclination_rad)
        sin_i = math.sin(inclination_rad)

        x_unit = cos_u * cos_node - sin_u * sin_node * cos_i
        y_unit = cos_u * sin_node + sin_u * cos_node * cos_i
        z_unit = sin_u * sin_i

        latitude_deg = math.degrees(
            math.asin(max(-1.0, min(1.0, z_unit))),
        )
        longitude_deg = (
            math.degrees(math.atan2(y_unit, x_unit) - OST_EARTH_ROTATION_RATE_RAD_S * time_s)
            % 360.0
        )

        samples.append(
            OstOrbitSample(
                index=index,
                time_s=time_s,
                true_anomaly_deg=math.degrees(true_anomaly_rad) % 360.0,
                radius_km=radius_km,
                altitude_km=altitude_km,
                latitude_deg=latitude_deg,
                longitude_deg=longitude_deg,
            )
        )

    return OstOrbitTrace(
        samples=tuple(samples),
        semi_major_axis_km=semi_major_axis_km,
        eccentricity=eccentricity,
        orbital_period_s=orbital_period_s,
        window_s=window_s,
        step_s=step_s,
        node_precession_rad_s=node_precession_rad_s,
        perigee_precession_rad_s=perigee_precession_rad_s,
    )
