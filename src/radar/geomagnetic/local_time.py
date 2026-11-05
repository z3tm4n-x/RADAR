"""Explicit, diagnostic magnetic and solar-direction local time.

All vectors use the same geocentric, Earth-fixed Cartesian frame. No civil
epoch, Earth rotation angle, solar ephemeris or magnetic pole is inferred from
the mission launch year. The caller supplies the Sun direction and north axis.
This geometry is not a calibration of the local-time convention used by the
OST disturbance correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real

from radar.geomagnetic.ost_orbit import OstOrbitSample

Vector3 = tuple[float, float, float]
CENTERED_DIPOLE_MLT_MODEL = "centered_dipole_ecef_mlt_v1"
SOLAR_LOCAL_TIME_MODEL = "geographic_solar_direction_local_time_v1"
ORBIT_SAMPLE_FINGERPRINT_VERSION = "ost_orbit_samples_float_hex_v1"
# Dimensionless sine of the angle to the magnetic axis. This numerical guard
# rejects ill-conditioned magnetic longitude; it is not a physical cutoff.
MLT_MIN_EQUATORIAL_PROJECTION = 1e-12
LOCAL_TIME_HISTORY_QUALIFICATION_REASON = (
    "Explicit local-time history provenance is recorded but its physical time/orientation "
    "and OST disturbance-model qualification are not established."
)
CENTERED_DIPOLE_MLT_QUALIFICATION_REASON = (
    "Centered-dipole MLT geometry has not been qualified as the MLT convention "
    "of the OST disturbance correction."
)
SOLAR_LOCAL_TIME_QUALIFICATION_REASON = (
    "Solar-direction local-time geometry and supplied Sun history have not been "
    "qualified for the OST disturbance correction."
)


def _finite_float(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be a finite real number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number.")
    return 0.0 if result == 0.0 else result


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string.")
    return value


def _parameters(values: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError("Local-time parameters must be a sequence of key/value pairs.")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("MLT parameters must be key/value pairs.")
        key, value = item
        _text(key, "MLT parameter key")
        _text(value, "MLT parameter value")
        if key in seen:
            raise ValueError("MLT parameter keys must be unique.")
        seen.add(key)
        result.append((key, value))
    return tuple(result)


def _reasons(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError("Local-time incomplete reasons must be a sequence of strings.")
    return tuple(dict.fromkeys(_text(value, "MLT incomplete reason") for value in values))


def _unit_vector(vector: Vector3, label: str) -> Vector3:
    if not isinstance(vector, (tuple, list)) or len(vector) != 3:
        raise ValueError(f"{label} must contain exactly three finite components.")
    x, y, z = (_finite_float(value, label) for value in vector)
    scale = max(abs(x), abs(y), abs(z))
    if scale == 0.0:
        raise ValueError(f"{label} must be nonzero.")
    # Scale before norm: both subnormal and near-maximum finite vectors work.
    x, y, z = x / scale, y / scale, z / scale
    norm = math.hypot(x, y, z)
    return x / norm, y / norm, z / norm


def _cross(left: Vector3, right: Vector3) -> Vector3:
    x, y, z = left
    u, v, w = right
    return y * w - z * v, z * u - x * w, x * v - y * u


def _dot(left: Vector3, right: Vector3) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _equatorial_direction(vector: Vector3, axis: Vector3, label: str) -> Vector3:
    # axis x vector measures sin(angle); the second cross projects onto the
    # magnetic equator without subtracting almost equal parallel components.
    perpendicular = _cross(axis, vector)
    if math.hypot(*perpendicular) <= MLT_MIN_EQUATORIAL_PROJECTION:
        raise ValueError(f"{label} magnetic longitude is undefined or ill-conditioned near a pole.")
    return _unit_vector(_cross(perpendicular, axis), f"{label} equatorial projection")


def centered_dipole_mlt_hours(
    *,
    position_ecef: Vector3,
    sun_direction_ecef: Vector3,
    north_axis_ecef: Vector3,
) -> float:
    """Return MLT in [0,24): subsolar=12, east=18, antisolar=0, west=6.

    MLT = (12 + signed_magnetic_longitude_difference * 12/pi) modulo 24.
    The signed angle is from the projected Sun direction to the projected
    position, positive about the supplied magnetic north axis. Vector scales
    have no effect. A magnetic-pole degeneracy raises instead of inventing MLT.
    """
    axis = _unit_vector(north_axis_ecef, "Magnetic north axis")
    position = _equatorial_direction(
        _unit_vector(position_ecef, "Position"), axis, "Position",
    )
    sun = _equatorial_direction(
        _unit_vector(sun_direction_ecef, "Sun direction"), axis, "Sun direction",
    )
    angle = math.atan2(_dot(axis, _cross(sun, position)), _dot(sun, position))
    return (12.0 + angle * 12.0 / math.pi) % 24.0


def local_solar_time_hours(*, position_ecef: Vector3, sun_direction_ecef: Vector3) -> float:
    """Geographic solar-direction local time in [0,24), not civil/mean solar time.

    The hour angle is measured about geographic north, with noon on the supplied
    Sun meridian. Accuracy/apparent versus geometric interpretation of that Sun
    direction is the caller's responsibility; no ephemeris or UTC is inferred.
    """
    return centered_dipole_mlt_hours(
        position_ecef=position_ecef, sun_direction_ecef=sun_direction_ecef,
        north_axis_ecef=(0.0, 0.0, 1.0),
    )


def _hex(value: float) -> str:
    return (0.0 if value == 0.0 else float(value)).hex()


def orbit_samples_fingerprint(samples: tuple[OstOrbitSample, ...]) -> str:
    """SHA256 of every ordered sample field, not a provenance certificate.

    Numerically identical int/float and signed-zero representations canonicalize
    identically. Local index, time, anomaly and all geographic fields are bound.
    """
    if not samples:
        raise ValueError("MLT orbit samples must not be empty.")
    digest = hashlib.sha256()
    digest.update(f"{ORBIT_SAMPLE_FINGERPRINT_VERSION}\n{len(samples)}\n".encode("ascii"))
    for sample in samples:
        if not isinstance(sample, OstOrbitSample):
            raise ValueError("MLT binding requires OstOrbitSample records.")
        if isinstance(sample.index, bool) or not isinstance(sample.index, int) or sample.index < 0:
            raise ValueError("MLT sample index must be a non-negative integer.")
        sample_fields = (
            sample.time_s, sample.true_anomaly_deg, sample.radius_km, sample.altitude_km,
            sample.latitude_deg, sample.longitude_deg,
        )
        values = tuple(_finite_float(value, "MLT sample field") for value in sample_fields)
        if any(
            isinstance(raw, Integral) and int(normalized) != raw
            for raw, normalized in zip(sample_fields, values, strict=True)
        ):
            raise ValueError("Integer MLT sample fields must be exactly representable as floats.")
        digest.update((str(sample.index) + "," + ",".join(map(_hex, values)) + "\n").encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LocalTimeHistory:
    """Sample-bound local-time history, never an authenticity/qualification certificate.

    Every instance retains a mandatory qualification reason. Direct construction
    is useful for imported histories but cannot declare them normatively valid.
    """

    values_hours: tuple[float, ...]
    sample_fingerprint: str
    model: str
    parameters: tuple[tuple[str, str], ...] = ()
    incomplete_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = tuple(_finite_float(value, "MLT value") for value in self.values_hours)
        if not values or any(not 0.0 <= value < 24.0 for value in values):
            raise ValueError("MLT values must be nonempty and in [0,24).")
        fingerprint = self.sample_fingerprint
        if (
            not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("MLT sample fingerprint must be a lowercase SHA256 hex digest.")
        _text(self.model, "MLT model")
        object.__setattr__(self, "values_hours", values)
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        object.__setattr__(self, "incomplete_reasons", _reasons((
            *_reasons(self.incomplete_reasons), LOCAL_TIME_HISTORY_QUALIFICATION_REASON,
        )))

    def validate_samples(self, samples: tuple[OstOrbitSample, ...]) -> None:
        """Reject reuse on reordered, rephased or geographically changed samples."""
        if len(self.values_hours) != len(samples):
            raise ValueError("MLT history length does not match the orbit sample count.")
        if orbit_samples_fingerprint(samples) != self.sample_fingerprint:
            raise ValueError("MLT history fingerprint does not match the supplied orbit samples.")


def build_centered_dipole_mlt_history(
    *,
    samples: tuple[OstOrbitSample, ...],
    sun_directions_ecef: tuple[Vector3, ...],
    north_axis_ecef: Vector3,
    sun_model: str,
    time_reference: str,
    parameters: tuple[tuple[str, str], ...] = (),
    incomplete_reasons: tuple[str, ...] = (),
) -> LocalTimeHistory:
    """Build diagnostic MLT from geocentric samples and caller-supplied ECEF Sun vectors.

    ``sun_model`` and ``time_reference`` are mandatory provenance descriptions,
    not validated ephemerides. The Sun hash covers normalized effective vectors
    in sample order. The magnetic north axis is explicit and fixed for this
    history. Geographic latitude is geocentric, as in ``OstOrbitSample``.
    """
    return _build_local_time_history(
        samples=samples, sun_directions_ecef=sun_directions_ecef,
        north_axis_ecef=north_axis_ecef, sun_model=sun_model, time_reference=time_reference,
        parameters=parameters, incomplete_reasons=incomplete_reasons,
        model=CENTERED_DIPOLE_MLT_MODEL,
        qualification_reason=CENTERED_DIPOLE_MLT_QUALIFICATION_REASON,
        axis_parameter="north_axis_ecef_unit",
    )


def build_solar_local_time_history(
    *,
    samples: tuple[OstOrbitSample, ...],
    sun_directions_ecef: tuple[Vector3, ...],
    sun_model: str,
    time_reference: str,
    parameters: tuple[tuple[str, str], ...] = (),
    incomplete_reasons: tuple[str, ...] = (),
) -> LocalTimeHistory:
    """Build geographic solar-direction hours; never infer UTC from launch year."""
    return _build_local_time_history(
        samples=samples, sun_directions_ecef=sun_directions_ecef,
        north_axis_ecef=(0.0, 0.0, 1.0), sun_model=sun_model, time_reference=time_reference,
        parameters=parameters, incomplete_reasons=incomplete_reasons,
        model=SOLAR_LOCAL_TIME_MODEL, qualification_reason=SOLAR_LOCAL_TIME_QUALIFICATION_REASON,
        axis_parameter="geographic_north_axis_ecef_unit",
    )


def _build_local_time_history(
    *, samples: tuple[OstOrbitSample, ...], sun_directions_ecef: tuple[Vector3, ...],
    north_axis_ecef: Vector3, sun_model: str, time_reference: str,
    parameters: tuple[tuple[str, str], ...], incomplete_reasons: tuple[str, ...],
    model: str, qualification_reason: str, axis_parameter: str,
) -> LocalTimeHistory:
    parameters = _parameters(parameters)
    incomplete_reasons = _reasons(incomplete_reasons)
    fingerprint = orbit_samples_fingerprint(samples)
    if len(sun_directions_ecef) != len(samples):
        raise ValueError("Sun directions must match the orbit sample count.")
    _text(sun_model, "Sun model")
    _text(time_reference, "MLT time reference")
    axis = _unit_vector(north_axis_ecef, "Magnetic north axis")
    suns = tuple(_unit_vector(vector, "Sun direction") for vector in sun_directions_ecef)
    sun_hash = hashlib.sha256(b"normalized_ecef_sun_vectors_float_hex_v1\n")
    values: list[float] = []
    for sample, sun in zip(samples, suns, strict=True):
        sun_hash.update((",".join(map(_hex, sun)) + "\n").encode("ascii"))
        latitude, longitude = math.radians(sample.latitude_deg), math.radians(sample.longitude_deg)
        position = (
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude), math.sin(latitude),
        )
        values.append(centered_dipole_mlt_hours(
            position_ecef=position, sun_direction_ecef=sun, north_axis_ecef=axis,
        ))
    recorded_parameters = (
        ("coordinate_frame", "geocentric_earth_fixed_cartesian"),
        ("mlt_convention", "subsolar_12_east_18_antisolar_0_west_6"),
        (axis_parameter, ",".join(format(value, ".17g") for value in axis)),
        ("sun_model", sun_model), ("time_reference", time_reference),
        ("sun_directions_ecef_sha256", sun_hash.hexdigest()),
        ("sun_hash_policy", "normalized_ecef_sun_vectors_float_hex_v1"),
        ("sample_fingerprint_policy", ORBIT_SAMPLE_FINGERPRINT_VERSION),
        ("minimum_equatorial_projection", format(MLT_MIN_EQUATORIAL_PROJECTION, ".17g")),
        *parameters,
    )
    return LocalTimeHistory(
        values_hours=tuple(values), sample_fingerprint=fingerprint,
        model=model, parameters=recorded_parameters,
        incomplete_reasons=(*incomplete_reasons, qualification_reason),
    )


__all__ = [
    "CENTERED_DIPOLE_MLT_MODEL", "CENTERED_DIPOLE_MLT_QUALIFICATION_REASON",
    "LOCAL_TIME_HISTORY_QUALIFICATION_REASON", "MLT_MIN_EQUATORIAL_PROJECTION",
    "LocalTimeHistory", "ORBIT_SAMPLE_FINGERPRINT_VERSION", "SOLAR_LOCAL_TIME_MODEL",
    "SOLAR_LOCAL_TIME_QUALIFICATION_REASON", "build_centered_dipole_mlt_history",
    "build_solar_local_time_history", "centered_dipole_mlt_hours", "local_solar_time_hours",
    "orbit_samples_fingerprint",
]
