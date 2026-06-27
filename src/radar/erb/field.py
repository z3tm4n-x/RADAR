"""Magnetic-field and L/B/B0 utilities for ERB models."""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    atan2,
    cos,
    isfinite,
    nan,
    sin,
    sqrt,
)

from radar.erb.constants import EARTH_REFERENCE_RADIUS_KM, ERB_DIPOLE_MOMENT_GAUSS
from radar.erb.igrf import IgrfCoefficients
from radar.erb.orbit import ErbOrbitSample


@dataclass(frozen=True)
class MagneticFieldVector:
    """Magnetic-field vector in nT."""

    bx_nt: float
    by_nt: float
    bz_nt: float
    magnitude_nt: float

    def __post_init__(self) -> None:
        if not all(
            isfinite(value)
            for value in (self.bx_nt, self.by_nt, self.bz_nt, self.magnitude_nt)
        ):
            msg = "Magnetic field vector values must be finite."
            raise ValueError(msg)

        if self.magnitude_nt < 0.0:
            msg = "Magnetic field magnitude must be non-negative."
            raise ValueError(msg)


@dataclass(frozen=True)
class ErbFieldLineSample:
    """Derived ERB magnetic coordinates for one orbit sample."""

    l_shell: float
    b_over_b0: float
    local_b_nt: float
    mirror_b0_nt: float
    valid: bool

    def __post_init__(self) -> None:
        if self.valid:
            if not all(
                isfinite(value)
                for value in (
                    self.l_shell,
                    self.b_over_b0,
                    self.local_b_nt,
                    self.mirror_b0_nt,
                )
            ):
                msg = "Valid ERB field-line sample values must be finite."
                raise ValueError(msg)

            if self.l_shell <= 0.0:
                msg = "Valid ERB L shell must be positive."
                raise ValueError(msg)

            if self.b_over_b0 < 1.0:
                msg = "Valid ERB B/B0 must be at least 1."
                raise ValueError(msg)

            if self.local_b_nt <= 0.0 or self.mirror_b0_nt <= 0.0:
                msg = "Valid ERB magnetic field magnitudes must be positive."
                raise ValueError(msg)


def _empty_field_line_sample() -> ErbFieldLineSample:
    return ErbFieldLineSample(
        l_shell=nan,
        b_over_b0=nan,
        local_b_nt=nan,
        mirror_b0_nt=nan,
        valid=False,
    )


def _cartesian_from_spherical(
    *,
    radius_km: float,
    theta_rad: float,
    longitude_rad: float,
) -> tuple[float, float, float]:
    sin_theta = sin(theta_rad)

    return (
        radius_km * sin_theta * cos(longitude_rad),
        radius_km * sin_theta * sin(longitude_rad),
        radius_km * cos(theta_rad),
    )


def _legendre_schmidt(
    *,
    theta_rad: float,
    nmax: int,
) -> tuple[list[list[float]], list[list[float]]]:
    cos_theta = cos(theta_rad)
    sin_theta = sin(theta_rad)

    p = [[0.0 for _ in range(nmax + 1)] for _ in range(nmax + 1)]
    dp = [[0.0 for _ in range(nmax + 1)] for _ in range(nmax + 1)]
    p[0][0] = 1.0

    for n in range(1, nmax + 1):
        diagonal_factor = sqrt((2.0 if n == 1 else 1.0) * (2 * n - 1) / (2.0 * n))
        p[n][n] = diagonal_factor * sin_theta * p[n - 1][n - 1]
        dp[n][n] = diagonal_factor * (
            sin_theta * dp[n - 1][n - 1] + cos_theta * p[n - 1][n - 1]
        )

        for m in range(n):
            denominator = sqrt(float(n * n - m * m))
            coefficient = (2 * n - 1) / denominator
            previous_coefficient = sqrt(max(float((n - 1) * (n - 1) - m * m), 0.0)) / denominator

            previous_p = p[n - 2][m] if n >= 2 else 0.0
            previous_dp = dp[n - 2][m] if n >= 2 else 0.0

            p[n][m] = coefficient * cos_theta * p[n - 1][m] - previous_coefficient * previous_p
            dp[n][m] = (
                coefficient * (cos_theta * dp[n - 1][m] - sin_theta * p[n - 1][m])
                - previous_coefficient * previous_dp
            )

    return p, dp


def magnetic_field_vector_km(
    *,
    x_km: float,
    y_km: float,
    z_km: float,
    coefficients: IgrfCoefficients,
) -> MagneticFieldVector:
    """Evaluate the IGRF magnetic field in Cartesian geocentric coordinates."""

    if not all(isfinite(value) for value in (x_km, y_km, z_km)):
        msg = "Magnetic field input coordinates must be finite."
        raise ValueError(msg)

    radius = sqrt(x_km * x_km + y_km * y_km + z_km * z_km)

    if radius <= 0.0:
        msg = "Magnetic field input radius must be positive."
        raise ValueError(msg)

    cos_theta = max(-1.0, min(1.0, z_km / radius))
    theta = atan2(sqrt(max(1.0 - cos_theta * cos_theta, 0.0)), cos_theta)
    sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    longitude = atan2(y_km, x_km)

    p, dp = _legendre_schmidt(theta_rad=theta, nmax=coefficients.nmax)

    cos_phi = [cos(m * longitude) for m in range(coefficients.nmax + 1)]
    sin_phi = [sin(m * longitude) for m in range(coefficients.nmax + 1)]

    br_nt = 0.0
    bt_nt = 0.0
    bl_nt = 0.0

    radius_ratio_power = (EARTH_REFERENCE_RADIUS_KM / radius) ** 2

    for n in range(1, coefficients.nmax + 1):
        radius_ratio_power *= EARTH_REFERENCE_RADIUS_KM / radius

        for m in range(n + 1):
            g_nm = coefficients.g_nm(n, m)
            h_nm = coefficients.h_nm(n, m)
            gh = g_nm * cos_phi[m] + h_nm * sin_phi[m]
            dgh = m * (-g_nm * sin_phi[m] + h_nm * cos_phi[m])

            br_nt += (n + 1) * radius_ratio_power * gh * p[n][m]
            bt_nt += -radius_ratio_power * gh * dp[n][m]
            bl_nt += -radius_ratio_power * dgh * p[n][m]

    if sin_theta > 1.0e-12:
        bl_nt /= sin_theta
    else:
        bl_nt = 0.0

    cylindrical_radius = sqrt(x_km * x_km + y_km * y_km)

    if cylindrical_radius > 0.0:
        sin_lon = y_km / cylindrical_radius
        cos_lon = x_km / cylindrical_radius
    else:
        sin_lon = 0.0
        cos_lon = 1.0

    bx_nt = br_nt * (sin_theta * cos_lon) + bt_nt * (cos_theta * cos_lon) + bl_nt * (-sin_lon)
    by_nt = br_nt * (sin_theta * sin_lon) + bt_nt * (cos_theta * sin_lon) + bl_nt * cos_lon
    bz_nt = br_nt * cos_theta - bt_nt * sin_theta
    magnitude_nt = sqrt(bx_nt * bx_nt + by_nt * by_nt + bz_nt * bz_nt)

    return MagneticFieldVector(
        bx_nt=bx_nt,
        by_nt=by_nt,
        bz_nt=bz_nt,
        magnitude_nt=magnitude_nt,
    )


def magnetic_field_at_orbit_sample(
    sample: ErbOrbitSample,
    coefficients: IgrfCoefficients,
) -> MagneticFieldVector:
    """Evaluate the magnetic field at one orbit sample."""

    x_km, y_km, z_km = _cartesian_from_spherical(
        radius_km=sample.radius_km,
        theta_rad=sample.theta_rad,
        longitude_rad=sample.longitude_rad,
    )

    return magnetic_field_vector_km(
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        coefficients=coefficients,
    )


def trace_l_shell_b_over_b0(
    sample: ErbOrbitSample,
    coefficients: IgrfCoefficients,
    *,
    step_km: float = 0.06 * EARTH_REFERENCE_RADIUS_KM,
    max_steps: int = 1500,
) -> ErbFieldLineSample:
    """Trace a field line toward its minimum-B point and return L and B/B0."""

    if step_km <= 0.0 or not isfinite(step_km):
        msg = "ERB field-line tracing step must be positive and finite."
        raise ValueError(msg)

    if max_steps < 1:
        msg = "ERB field-line tracing max_steps must be positive."
        raise ValueError(msg)

    x_km, y_km, z_km = _cartesian_from_spherical(
        radius_km=sample.radius_km,
        theta_rad=sample.theta_rad,
        longitude_rad=sample.longitude_rad,
    )
    local_field = magnetic_field_vector_km(
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        coefficients=coefficients,
    )

    if local_field.magnitude_nt <= 0.0:
        return _empty_field_line_sample()

    ux = local_field.bx_nt / local_field.magnitude_nt
    uy = local_field.by_nt / local_field.magnitude_nt
    uz = local_field.bz_nt / local_field.magnitude_nt

    forward = magnetic_field_vector_km(
        x_km=x_km + ux * step_km,
        y_km=y_km + uy * step_km,
        z_km=z_km + uz * step_km,
        coefficients=coefficients,
    )
    backward = magnetic_field_vector_km(
        x_km=x_km - ux * step_km,
        y_km=y_km - uy * step_km,
        z_km=z_km - uz * step_km,
        coefficients=coefficients,
    )
    sign = 1.0 if forward.magnitude_nt < backward.magnitude_nt else -1.0
    mirror_b0_nt = local_field.magnitude_nt

    for _ in range(max_steps):
        field = magnetic_field_vector_km(
            x_km=x_km,
            y_km=y_km,
            z_km=z_km,
            coefficients=coefficients,
        )

        if field.magnitude_nt <= 0.0:
            return _empty_field_line_sample()

        half_step = sign * step_km / field.magnitude_nt
        x_mid = x_km + 0.5 * half_step * field.bx_nt
        y_mid = y_km + 0.5 * half_step * field.by_nt
        z_mid = z_km + 0.5 * half_step * field.bz_nt

        mid_field = magnetic_field_vector_km(
            x_km=x_mid,
            y_km=y_mid,
            z_km=z_mid,
            coefficients=coefficients,
        )

        if mid_field.magnitude_nt <= 0.0:
            return _empty_field_line_sample()

        full_step = sign * step_km / mid_field.magnitude_nt
        x_next = x_km + full_step * mid_field.bx_nt
        y_next = y_km + full_step * mid_field.by_nt
        z_next = z_km + full_step * mid_field.bz_nt

        next_radius = sqrt(x_next * x_next + y_next * y_next + z_next * z_next)

        if next_radius < 0.98 * EARTH_REFERENCE_RADIUS_KM:
            return _empty_field_line_sample()

        next_field = magnetic_field_vector_km(
            x_km=x_next,
            y_km=y_next,
            z_km=z_next,
            coefficients=coefficients,
        )

        if next_field.magnitude_nt > mirror_b0_nt:
            break

        x_km = x_next
        y_km = y_next
        z_km = z_next
        mirror_b0_nt = next_field.magnitude_nt

    if mirror_b0_nt <= 0.0:
        return _empty_field_line_sample()

    mirror_b0_gauss = mirror_b0_nt * 1.0e-5
    l_shell = (ERB_DIPOLE_MOMENT_GAUSS / mirror_b0_gauss) ** (1.0 / 3.0)
    b_over_b0 = local_field.magnitude_nt / mirror_b0_nt

    return ErbFieldLineSample(
        l_shell=l_shell,
        b_over_b0=b_over_b0,
        local_b_nt=local_field.magnitude_nt,
        mirror_b0_nt=mirror_b0_nt,
        valid=True,
    )


def trace_l_shell_b_over_b0_for_orbit(
    samples: tuple[ErbOrbitSample, ...],
    coefficients: IgrfCoefficients,
) -> tuple[ErbFieldLineSample, ...]:
    """Trace all orbit samples to ERB L and B/B0 coordinates."""

    return tuple(
        trace_l_shell_b_over_b0(sample, coefficients)
        for sample in samples
    )


__all__ = [
    "ErbFieldLineSample",
    "MagneticFieldVector",
    "magnetic_field_at_orbit_sample",
    "magnetic_field_vector_km",
    "trace_l_shell_b_over_b0",
    "trace_l_shell_b_over_b0_for_orbit",
]
