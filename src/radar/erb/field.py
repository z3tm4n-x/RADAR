"""Magnetic-field and L/B/B0 utilities for ERB models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import (
    atan2,
    cos,
    isfinite,
    nan,
    radians,
    sin,
    sqrt,
)

from radar.erb.constants import (
    EARTH_REFERENCE_RADIUS_KM,
    ERB_DIPOLE_MOMENT_GAUSS,
    OST_ERB_DIPOLE_AXIS_LATITUDE_DEG,
    OST_ERB_DIPOLE_AXIS_LONGITUDE_DEG,
    OST_ERB_DIPOLE_OFFSET_KM,
    OST_ERB_DIPOLE_OFFSET_LATITUDE_DEG,
    OST_ERB_DIPOLE_OFFSET_LONGITUDE_DEG,
    OST_ERB_OUTER_L_BOUNDARY,
    OST_ERB_SHIFTED_DIPOLE_MOMENT_GAUSS_CM3,
)
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
            isfinite(value) for value in (self.bx_nt, self.by_nt, self.bz_nt, self.magnitude_nt)
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
    method: str = "unspecified"
    incomplete_reasons: tuple[str, ...] = ()
    classifier_l_shell: float | None = None
    alternative_l_shell: float | None = None

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


def _empty_field_line_sample(reason: str) -> ErbFieldLineSample:
    return ErbFieldLineSample(
        l_shell=nan,
        b_over_b0=nan,
        local_b_nt=nan,
        mirror_b0_nt=nan,
        valid=False,
        method="igrf_rk4_minimum_b",
        incomplete_reasons=(reason,),
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
    cos_theta: float,
    sin_theta: float,
    nmax: int,
) -> tuple[list[list[float]], list[list[float]]]:
    p = [[0.0 for _ in range(nmax + 1)] for _ in range(nmax + 1)]
    dp = [[0.0 for _ in range(nmax + 1)] for _ in range(nmax + 1)]
    p[0][0] = 1.0

    for n in range(1, nmax + 1):
        diagonal_factor = sqrt((2.0 if n == 1 else 1.0) * (2 * n - 1) / (2.0 * n))
        p[n][n] = diagonal_factor * sin_theta * p[n - 1][n - 1]
        dp[n][n] = diagonal_factor * (sin_theta * dp[n - 1][n - 1] + cos_theta * p[n - 1][n - 1])

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

    cylindrical_radius = sqrt(x_km * x_km + y_km * y_km)
    cos_theta = z_km / radius
    sin_theta = cylindrical_radius / radius
    longitude = atan2(y_km, x_km)

    p, dp = _legendre_schmidt(cos_theta=cos_theta, sin_theta=sin_theta, nmax=coefficients.nmax)

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
            # P(n,1)/sin(theta) has a nonzero polar limit. Setting B_phi
            # to zero loses the transverse dipole/quadrupole components.
            p_over_sin = (
                p[n][m] / sin_theta
                if sin_theta > 0.0
                else (dp[n][m] * cos_theta if m == 1 else 0.0)
            )
            bl_nt += -radius_ratio_power * dgh * p_over_sin

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
    """Locate the IGRF minimum B, then apply OST D.32 (not an L invariant).

    RK4 follows dr/ds = +/- B/|B|, the Cartesian form of D.29-D.31.
    A bracketed minimum is refined along that line, not just at step nodes.
    Exhausting the step budget is a failure, never a located equator.
    """

    from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

    if step_km <= 0.0 or not isfinite(step_km):
        msg = "ERB field-line tracing step must be positive and finite."
        raise ValueError(msg)

    if max_steps < 1:
        msg = "ERB field-line tracing max_steps must be positive."
        raise ValueError(msg)

    if sample.radius_km < 0.98 * EARTH_REFERENCE_RADIUS_KM:
        return _empty_field_line_sample("Orbit sample is below the magnetic tracing domain")

    initial = _cartesian_from_spherical(
        radius_km=sample.radius_km,
        theta_rad=sample.theta_rad,
        longitude_rad=sample.longitude_rad,
    )

    def field_at(point: tuple[float, float, float]) -> MagneticFieldVector:
        return magnetic_field_vector_km(
            x_km=point[0], y_km=point[1], z_km=point[2], coefficients=coefficients
        )

    def direction(point: tuple[float, float, float]) -> tuple[float, float, float]:
        field = field_at(point)
        if field.magnitude_nt <= 0.0:
            raise ArithmeticError("Zero magnetic field encountered during tracing")
        return (
            field.bx_nt / field.magnitude_nt,
            field.by_nt / field.magnitude_nt,
            field.bz_nt / field.magnitude_nt,
        )

    def rk4(point: tuple[float, float, float], distance: float) -> tuple[float, float, float]:
        def shifted(vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
            return (
                point[0] + scale * vector[0],
                point[1] + scale * vector[1],
                point[2] + scale * vector[2],
            )

        k1 = direction(point)
        k2 = direction(shifted(k1, distance * 0.5))
        k3 = direction(shifted(k2, distance * 0.5))
        k4 = direction(shifted(k3, distance))
        return (
            point[0] + distance / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            point[1] + distance / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            point[2] + distance / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        )

    local_field = field_at(initial)
    if local_field.magnitude_nt <= 0.0:
        return _empty_field_line_sample("Zero magnetic field at the orbit sample")

    try:
        probe = min(0.01 * step_km, 0.001 * EARTH_REFERENCE_RADIUS_KM)
        sign = (
            1.0
            if field_at(rk4(initial, probe)).magnitude_nt
            < field_at(rk4(initial, -probe)).magnitude_nt
            else -1.0
        )
        current = initial
        current_b = local_field.magnitude_nt
        previous = initial
        for index in range(max_steps):
            next_point = rk4(current, sign * step_km)
            next_b = field_at(next_point).magnitude_nt
            if next_b > current_b:
                # On the first step the initial point may itself be the
                # minimum; retain both signs in the search interval.
                anchor = previous if index else initial
                lower, upper = (0.0, 2.0 * step_km) if index else (-step_km, step_km)

                def point_on_line(distance: float) -> tuple[float, float, float]:
                    if distance > step_km:
                        return rk4(rk4(anchor, sign * step_km), sign * (distance - step_km))
                    return rk4(anchor, sign * distance)

                def objective(distance: float) -> float:
                    return field_at(point_on_line(distance)).magnitude_nt

                minimum = minimize_scalar(
                    objective,
                    bounds=(lower, upper),
                    method="bounded",
                    options={"xatol": 1.0e-6, "maxiter": 64},
                )
                if not minimum.success:
                    return _empty_field_line_sample("Magnetic-equator minimum refinement failed")
                minimum_point = point_on_line(float(minimum.x))
                if (
                    sqrt(sum(value * value for value in minimum_point))
                    < 0.98 * EARTH_REFERENCE_RADIUS_KM
                ):
                    return _empty_field_line_sample(
                        "Magnetic equator lies below the tracing domain"
                    )
                mirror_b0_nt = min(float(minimum.fun), current_b, local_field.magnitude_nt)
                break
            if sqrt(sum(value * value for value in next_point)) < 0.98 * EARTH_REFERENCE_RADIUS_KM:
                return _empty_field_line_sample(
                    "Field line entered Earth before locating the equator"
                )
            previous, current, current_b = current, next_point, next_b
        else:
            return _empty_field_line_sample("Magnetic equator not located within max_steps")
    except ArithmeticError as exc:
        return _empty_field_line_sample(str(exc))

    mirror_b0_gauss = mirror_b0_nt * 1.0e-5
    l_shell = (ERB_DIPOLE_MOMENT_GAUSS / mirror_b0_gauss) ** (1.0 / 3.0)
    b_over_b0 = local_field.magnitude_nt / mirror_b0_nt

    return ErbFieldLineSample(
        l_shell=l_shell,
        b_over_b0=b_over_b0,
        local_b_nt=local_field.magnitude_nt,
        mirror_b0_nt=mirror_b0_nt,
        valid=True,
        method="igrf_rk4_minimum_b",
    )


def trace_l_shell_b_over_b0_for_orbit(
    samples: tuple[ErbOrbitSample, ...],
    coefficients: IgrfCoefficients,
) -> tuple[ErbFieldLineSample, ...]:
    """Trace all orbit samples to ERB L and B/B0 coordinates."""

    return tuple(trace_l_shell_b_over_b0(sample, coefficients) for sample in samples)


def shifted_dipole_l_shell_b_over_b0(
    sample: ErbOrbitSample, *, moment_scale: float = 1.0
) -> ErbFieldLineSample:
    """OST D.21-D.24 shifted dipole; g=1 is a reference convention.

    g(t) is not specified by Appendix D. L and B/B0 do not depend on g;
    only the two absolute fields use this explicit positive moment scale.
    """
    if not isfinite(moment_scale) or moment_scale <= 0.0:
        raise ValueError("Shifted-dipole moment scale must be positive and finite.")

    def geographic_vector(latitude_deg: float, longitude_deg: float) -> tuple[float, float, float]:
        latitude, longitude = radians(latitude_deg), radians(longitude_deg)
        return cos(latitude) * cos(longitude), cos(latitude) * sin(longitude), sin(latitude)

    point = _cartesian_from_spherical(
        radius_km=sample.radius_km,
        theta_rad=sample.theta_rad,
        longitude_rad=sample.longitude_rad,
    )
    offset_direction = geographic_vector(
        OST_ERB_DIPOLE_OFFSET_LATITUDE_DEG, OST_ERB_DIPOLE_OFFSET_LONGITUDE_DEG
    )
    axis = geographic_vector(OST_ERB_DIPOLE_AXIS_LATITUDE_DEG, OST_ERB_DIPOLE_AXIS_LONGITUDE_DEG)
    shifted = tuple(
        coordinate - OST_ERB_DIPOLE_OFFSET_KM * offset
        for coordinate, offset in zip(point, offset_direction, strict=True)
    )
    radius = sqrt(sum(value * value for value in shifted))
    if radius <= 0.0:
        return replace(
            _empty_field_line_sample("Orbit sample is at the shifted-dipole centre"),
            method="ost_shifted_dipole_d21_d24",
        )
    sin_latitude = max(
        -1.0,
        min(
            1.0,
            sum(value * direction for value, direction in zip(shifted, axis, strict=True)) / radius,
        ),
    )
    # Squared transverse distance avoids subtracting two almost equal units.
    cos_latitude_squared = sum(
        (coordinate / radius - sin_latitude * direction) ** 2
        for coordinate, direction in zip(shifted, axis, strict=True)
    )
    cos_latitude_squared = min(1.0, cos_latitude_squared)
    if cos_latitude_squared < 1.0e-24:
        return replace(
            _empty_field_line_sample("Shifted-dipole magnetic axis has no finite L shell"),
            method="ost_shifted_dipole_d21_d24",
        )
    l_shell = radius / (EARTH_REFERENCE_RADIUS_KM * cos_latitude_squared)
    angular_factor = sqrt(1.0 + 3.0 * sin_latitude**2)
    local_b_nt = moment_scale * OST_ERB_SHIFTED_DIPOLE_MOMENT_GAUSS_CM3 / (radius * 1.0e5) ** 3
    local_b_nt *= angular_factor * 1.0e5  # gauss -> nT
    mirror_b0_nt = moment_scale * OST_ERB_SHIFTED_DIPOLE_MOMENT_GAUSS_CM3
    mirror_b0_nt *= 1.0e5 / (l_shell * EARTH_REFERENCE_RADIUS_KM * 1.0e5) ** 3
    return ErbFieldLineSample(
        l_shell=l_shell,
        b_over_b0=angular_factor / cos_latitude_squared**3,
        local_b_nt=local_b_nt,
        mirror_b0_nt=mirror_b0_nt,
        valid=True,
        method="ost_shifted_dipole_d21_d24",
    )


def ost_l_shell_b_over_b0(
    sample: ErbOrbitSample,
    coefficients: IgrfCoefficients,
    *,
    step_km: float = 0.06 * EARTH_REFERENCE_RADIUS_KM,
    max_steps: int = 1500,
    compare_outer_igrf: bool = False,
) -> ErbFieldLineSample:
    """OST D.2.6: inner L<=5.5 (IGRF), outer L>5.5 (shifted dipole).

    Appendix D does not define a pilot L for choosing between its two models.
    RADAR uses the analytical shifted-dipole L and records this convention.
    Both candidates are checked for 5<=L_dipole<=6 (or when requested).
    No artificial smoothing, blending or forced agreement is applied.
    """
    dipole = shifted_dipole_l_shell_b_over_b0(sample)
    if not dipole.valid:
        return dipole
    outer = dipole.l_shell > OST_ERB_OUTER_L_BOUNDARY
    inner: ErbFieldLineSample | None = None
    if not outer or compare_outer_igrf or 5.0 <= dipole.l_shell <= 6.0:
        inner = trace_l_shell_b_over_b0(sample, coefficients, step_km=step_km, max_steps=max_steps)
    selected = dipole if outer else inner
    assert selected is not None
    reasons = list(selected.incomplete_reasons)
    if inner is not None:
        if not inner.valid and outer:
            reasons.append("OST D.2.6 diagnostic IGRF boundary comparison failed")
        elif inner.valid and (inner.l_shell > OST_ERB_OUTER_L_BOUNDARY) != outer:
            reasons.append(
                "OST D.2.6 IGRF and shifted-dipole L candidates select different regions"
            )
    return replace(
        selected,
        classifier_l_shell=dipole.l_shell,
        alternative_l_shell=(inner.l_shell if inner is not None and inner.valid else None)
        if outer
        else dipole.l_shell,
        incomplete_reasons=tuple(reasons),
    )


def ost_l_shell_b_over_b0_for_orbit(
    samples: tuple[ErbOrbitSample, ...], coefficients: IgrfCoefficients
) -> tuple[ErbFieldLineSample, ...]:
    """Apply the OST-only hybrid mapping without changing AE8/AP8 dispatch."""
    return tuple(ost_l_shell_b_over_b0(sample, coefficients) for sample in samples)


__all__ = [
    "ErbFieldLineSample",
    "MagneticFieldVector",
    "magnetic_field_at_orbit_sample",
    "magnetic_field_vector_km",
    "ost_l_shell_b_over_b0",
    "ost_l_shell_b_over_b0_for_orbit",
    "shifted_dipole_l_shell_b_over_b0",
    "trace_l_shell_b_over_b0",
    "trace_l_shell_b_over_b0_for_orbit",
]
