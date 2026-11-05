"""Independent frame algebra for the declared, simplified OST orbit model.

Source: OST Appendix D.1--D.7 and D.18--D.20 (PDF pages 116--119).
The east-positive terrestrial rotation below follows ESA Navipedia equations
(4)--(5), not a declaration that the printed OST D.20 is an official erratum:
https://gssc.esa.int/navipedia/index.php/Transformation_between_Celestial_and_Terrestrial_Frames

Kepler's equation is solved by bounded bisection and positions are constructed
in the orbital plane, independently of production anomaly/position helpers.
The printed D.9--D.10 drift constants and factors are retained model inputs,
not independently verified physical rates. These tests do not implement EOP,
nutation, a calendar epoch, MLT, or a representative mission average.
"""

import math

import pytest

from radar.core.project import OrbitConfig
from radar.core.types import OrbitType
from radar.geomagnetic import ost_orbit as orbit_module
from radar.geomagnetic.ost_orbit import OstOrbitSample, sample_ost_orbit

_RADIUS_KM = 6371.2
_MU_KM3_S2 = 398600.4418
_PRINTED_DRIFT_PARAMETER = 4.1378e5
_EARTH_RATE_RAD_S = 7.2921159e-5
_TWO_PI = 2.0 * math.pi
_STEPS = 32
_TINY_DAYS = 1e-10
_FIVE_YEARS_DAYS = 5.0 * 365.25

type_vector = tuple[float, float, float]
type_matrix = tuple[type_vector, type_vector, type_vector]


def _r1(angle: float) -> type_matrix:
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, cosine, sine), (0.0, -sine, cosine))


def _r3(angle: float) -> type_matrix:
    """Passive right-handed frame rotation, ESA Navipedia equation (4)."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((cosine, sine, 0.0), (-sine, cosine, 0.0), (0.0, 0.0, 1.0))


def _multiply(matrix: type_matrix, vector: type_vector) -> type_vector:
    first, second, third = (
        math.fsum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix
    )
    return first, second, third


def _elements(config: OrbitConfig) -> tuple[float, float, float]:
    perigee = _RADIUS_KM + config.perigee_altitude_km
    apogee = _RADIUS_KM + config.apogee_altitude_km
    semi_major = (perigee + apogee) / 2.0
    eccentricity = (apogee - perigee) / (apogee + perigee)
    mean_motion = math.sqrt(_MU_KM3_S2 / semi_major**3)
    return semi_major, eccentricity, mean_motion


def _bisect_kepler(mean_anomaly: float, eccentricity: float) -> float:
    """The elliptic equation is strictly increasing and bracketed on [-pi, pi]."""
    reduced = math.remainder(mean_anomaly, _TWO_PI)
    lower, upper = -math.pi, math.pi
    for _ in range(80):
        middle = lower + (upper - lower) / 2.0
        residual = middle - eccentricity * math.sin(middle) - reduced
        if residual == 0.0:
            return middle
        if residual < 0.0:
            lower = middle
        else:
            upper = middle
    result = lower + (upper - lower) / 2.0
    assert abs(result - eccentricity * math.sin(result) - reduced) <= 2e-14
    return result


def _reference_ecef(
    config: OrbitConfig, *, time_s: float, absolute_node_deg: float, greenwich_deg: float,
) -> type_vector:
    semi_major, eccentricity, mean_motion = _elements(config)
    eccentric_anomaly = _bisect_kepler(mean_motion * time_s, eccentricity)
    orbital_plane = (
        semi_major * (math.cos(eccentric_anomaly) - eccentricity),
        semi_major * math.sqrt(1.0 - eccentricity**2) * math.sin(eccentric_anomaly),
        0.0,
    )
    inclination = math.radians(config.inclination_deg)
    period = _TWO_PI / mean_motion
    # Retain the declared printed model; this is not a J2-rate accuracy test.
    drift_scale = (
        _PRINTED_DRIFT_PARAMETER * (1.0 - eccentricity)
        / (semi_major**2 * (1.0 - eccentricity**2) ** 2) / period
    )
    node = math.radians(absolute_node_deg) - drift_scale * math.cos(inclination) * time_s
    perigee = (
        math.radians(config.argument_of_perigee_deg)
        + drift_scale * (5.0 * math.cos(inclination) ** 2 - 1.0) * time_s / 2.0
    )
    inertial = _multiply(
        _r3(-node), _multiply(_r1(-inclination), _multiply(_r3(-perigee), orbital_plane)),
    )
    terrestrial_angle = math.radians(greenwich_deg) + _EARTH_RATE_RAD_S * time_s
    return _multiply(_r3(terrestrial_angle), inertial)


def _sample_ecef(sample: OstOrbitSample) -> type_vector:
    latitude = math.radians(sample.latitude_deg)
    longitude = math.radians(sample.longitude_deg)
    equatorial_radius = sample.radius_km * math.cos(latitude)
    return (
        equatorial_radius * math.cos(longitude),
        equatorial_radius * math.sin(longitude),
        sample.radius_km * math.sin(latitude),
    )


def _orbit(inclination: float, perigee_argument: float, apogee: float = 39700.0) -> OrbitConfig:
    return OrbitConfig(
        orbit_type=OrbitType.ELLIPTIC, perigee_altitude_km=500.0,
        apogee_altitude_km=apogee, inclination_deg=inclination,
        argument_of_perigee_deg=perigee_argument,
    )


def _start_index(config: OrbitConfig, elapsed_days: float) -> int:
    _, _, mean_motion = _elements(config)
    step = _TWO_PI / mean_motion / _STEPS
    return round(elapsed_days * 86400.0 / step)


def _wrapped_difference(left: float, right: float) -> float:
    return (left - right + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("apogee", [1500.0, 39700.0])
@pytest.mark.parametrize(("inclination", "argument", "relative_node"), [
    (0.0, 0.0, 0.0), (51.6, 47.0, 123.0), (63.4, 270.0, -195.0),
    (90.0, 13.0, 60.0), (116.6, 42.0, 180.0), (180.0, 320.0, 370.0),
])
@pytest.mark.parametrize("elapsed_days", [0.0, 16.0, 365.25, _FIVE_YEARS_DAYS])
def test_eccentric_trace_matches_bisection_and_matrix_reference(
    apogee: float, inclination: float, argument: float, relative_node: float,
    elapsed_days: float,
) -> None:
    config = _orbit(inclination, argument, apogee)
    start = _start_index(config, elapsed_days)
    trace = sample_ost_orbit(
        config, raan_deg=relative_node, min_days=_TINY_DAYS, min_orbits=1,
        steps_per_orbit=_STEPS, max_samples=_STEPS, start_sample_index=start,
    )
    assert len(trace.samples) == _STEPS
    assert abs(trace.samples[0].time_s - elapsed_days * 86400.0) <= trace.step_s / 2.0
    for index, sample in enumerate(trace.samples):
        assert sample.time_s == (start + index) * trace.step_s
        reference = _reference_ecef(
            config, time_s=sample.time_s, absolute_node_deg=relative_node, greenwich_deg=0.0,
        )
        # Cartesian comparison avoids undefined longitude at a pole. The 2 cm
        # absolute tolerance includes independent angle reduction after 5 years;
        # it is a numerical cross-check, not an orbit-prediction accuracy claim.
        assert _sample_ecef(sample) == pytest.approx(reference, rel=0.0, abs=2e-5)
        assert sample.radius_km == pytest.approx(math.hypot(*reference), rel=0.0, abs=2e-5)


def test_default_zero_start_preserves_explicit_zero_start() -> None:
    config = _orbit(63.4, 39.0)
    default = sample_ost_orbit(
        config, raan_deg=127.0, min_days=_TINY_DAYS, min_orbits=1,
        steps_per_orbit=_STEPS, max_samples=_STEPS,
    )
    explicit = sample_ost_orbit(
        config, raan_deg=127.0, min_days=_TINY_DAYS, min_orbits=1,
        steps_per_orbit=_STEPS, max_samples=_STEPS, start_sample_index=0,
    )
    assert default == explicit
    assert default.samples[0].time_s == 0.0


@pytest.mark.parametrize("elapsed_days", [0.0, 16.0, _FIVE_YEARS_DAYS])
@pytest.mark.parametrize("common_shift", [-720.0, -37.0, 90.0, 450.0])
def test_common_node_and_greenwich_shift_preserves_relative_frame(
    elapsed_days: float, common_shift: float,
) -> None:
    config = _orbit(63.4, 37.0)
    node, greenwich = 23.0, 91.0
    relative = node - greenwich
    shifted_relative = (node + common_shift) - (greenwich + common_shift)
    assert shifted_relative == relative
    trace = sample_ost_orbit(
        config, raan_deg=shifted_relative, min_days=_TINY_DAYS, min_orbits=1,
        steps_per_orbit=_STEPS, max_samples=_STEPS,
        start_sample_index=_start_index(config, elapsed_days),
    )
    for sample in trace.samples:
        reference = _reference_ecef(
            config, time_s=sample.time_s, absolute_node_deg=node, greenwich_deg=greenwich,
        )
        shifted = _reference_ecef(
            config, time_s=sample.time_s, absolute_node_deg=node + common_shift,
            greenwich_deg=greenwich + common_shift,
        )
        assert shifted == pytest.approx(reference, rel=0.0, abs=2e-5)
        # The production API exposes only the relative phase, not two absolute
        # angles or a calendar epoch. This test does not claim otherwise.
        assert _sample_ecef(sample) == pytest.approx(reference, rel=0.0, abs=2e-5)


@pytest.mark.parametrize("inclination", [0.0, 63.4, 90.0, 116.6])
@pytest.mark.parametrize("elapsed_days", [0.0, _FIVE_YEARS_DAYS])
@pytest.mark.parametrize("node_shift", [90.0, -117.0, 360.0])
def test_node_only_shift_changes_longitude_not_latitude_or_radius(
    inclination: float, elapsed_days: float, node_shift: float,
) -> None:
    config = _orbit(inclination, 33.0)
    start = _start_index(config, elapsed_days)
    traces = tuple(sample_ost_orbit(
        config, raan_deg=node, min_days=_TINY_DAYS, min_orbits=1,
        steps_per_orbit=_STEPS, max_samples=_STEPS, start_sample_index=start,
    ) for node in (17.0, 17.0 + node_shift))
    for initial, shifted in zip(traces[0].samples, traces[1].samples, strict=True):
        assert shifted.time_s == initial.time_s
        assert shifted.radius_km == initial.radius_km
        assert shifted.latitude_deg == initial.latitude_deg
        assert abs(_wrapped_difference(
            shifted.longitude_deg, initial.longitude_deg + node_shift,
        )) < 2e-9


def test_positive_earth_rotation_moves_a_fixed_celestial_vector_westward() -> None:
    terrestrial = _multiply(_r3(math.pi / 2.0), (1.0, 0.0, 0.0))
    assert terrestrial == pytest.approx((0.0, -1.0, 0.0), rel=0.0, abs=1e-15)


def test_ideal_corotating_circular_orbit_is_stationary_in_terrestrial_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Synthetic zero-drift Kepler limit isolates the Earth-rotation sign. It
    # does not replace the printed rates in any production calculation.
    monkeypatch.setattr(orbit_module, "OST_SECULAR_PRECESSION_PARAMETER", 0.0)
    radius = (_MU_KM3_S2 / _EARTH_RATE_RAD_S**2) ** (1.0 / 3.0)
    config = OrbitConfig.circular(radius - _RADIUS_KM, 0.0)
    trace = sample_ost_orbit(
        config, raan_deg=31.0, min_days=_TINY_DAYS, min_orbits=1,
        steps_per_orbit=_STEPS, max_samples=_STEPS, start_sample_index=3 * _STEPS,
    )
    for sample in trace.samples:
        assert sample.latitude_deg == 0.0
        assert sample.radius_km == pytest.approx(radius, rel=0.0, abs=1e-9)
        assert abs(_wrapped_difference(sample.longitude_deg, 31.0)) < 2e-10
