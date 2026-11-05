"""Independent Cartesian-potential checks of the IGRF synthesis and tracing.

IGRF defines B = -grad(V), with reference radius a = 6371.2 km and
Schmidt semi-normalised associated Legendre functions:
https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field

The low-degree expectations below are Cartesian polynomial derivatives, not
another copy of the spherical synthesis or its Legendre recurrence. The
higher-degree check uses SciPy's independent unnormalised Legendre function
and a five-point Cartesian gradient of the scalar potential.
"""

from math import atan2, cos, factorial, pi, radians, sin, sqrt

import pytest
from scipy.special import lpmv

from radar.erb.constants import EARTH_REFERENCE_RADIUS_KM, ERB_DIPOLE_MOMENT_GAUSS
from radar.erb.field import magnetic_field_vector_km, trace_l_shell_b_over_b0
from radar.erb.igrf import IgrfCoefficients
from radar.erb.orbit import ErbOrbitSample


def _field_at_dimensionless_point(
    point: tuple[float, float, float], coefficients: IgrfCoefficients
) -> tuple[float, float, float]:
    field = magnetic_field_vector_km(
        x_km=point[0] * EARTH_REFERENCE_RADIUS_KM,
        y_km=point[1] * EARTH_REFERENCE_RADIUS_KM,
        z_km=point[2] * EARTH_REFERENCE_RADIUS_KM,
        coefficients=coefficients,
    )
    components = (field.bx_nt, field.by_nt, field.bz_nt)
    assert field.magnitude_nt == pytest.approx(sqrt(sum(value**2 for value in components)))
    return components


def _dipole_cartesian_field(
    point: tuple[float, float, float], dipole: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Differentiate V/a = (d dot u)/|u|^3, u = position/a."""
    radius = sqrt(sum(coordinate**2 for coordinate in point))
    dot = sum(coordinate * moment for coordinate, moment in zip(point, dipole, strict=True))
    return tuple(
        3.0 * dot * coordinate / radius**5 - moment / radius**3
        for coordinate, moment in zip(point, dipole, strict=True)
    )


@pytest.mark.parametrize(
    "point",
    [(1.2, 0.7, -0.4), (-0.6, 1.3, 0.8), (2.0, 0.0, 0.0), (0.0, 0.0, 1.3), (0.0, 0.0, -1.3)],
)
@pytest.mark.parametrize(
    "dipole",
    [(0.0, 0.0, -30000.0), (-1600.0, 0.0, 0.0), (0.0, 4800.0, 0.0), (-1600.0, 4800.0, -30000.0)],
)
def test_degree_one_field_equals_exact_cartesian_dipole(
    point: tuple[float, float, float], dipole: tuple[float, float, float]
) -> None:
    coefficients = IgrfCoefficients(
        epoch=1985.0,
        g={(1, 0): dipole[2], (1, 1): dipole[0]},
        h={(1, 1): dipole[1]},
        nmax=1,
    )
    assert _field_at_dimensionless_point(point, coefficients) == pytest.approx(
        _dipole_cartesian_field(point, dipole), rel=2.0e-12, abs=1.0e-8
    )


@pytest.mark.parametrize("pole_sign", [-1.0, 1.0])
@pytest.mark.parametrize("longitude", [0.0, pi / 4.0, -pi / 3.0, pi])
@pytest.mark.parametrize("transverse_radius", [1.0e-4, 1.0e-7, 1.0e-10])
def test_tilted_dipole_has_continuous_cartesian_polar_limit(
    pole_sign: float, longitude: float, transverse_radius: float
) -> None:
    dipole = (-1600.0, 4800.0, -30000.0)
    point = (
        transverse_radius * cos(longitude),
        transverse_radius * sin(longitude),
        pole_sign * 1.3,
    )
    coefficients = IgrfCoefficients(
        epoch=1985.0,
        g={(1, 0): dipole[2], (1, 1): dipole[0]},
        h={(1, 1): dipole[1]},
        nmax=1,
    )
    assert _field_at_dimensionless_point(point, coefficients) == pytest.approx(
        _dipole_cartesian_field(point, dipole), rel=2.0e-12, abs=1.0e-8
    )


def _quadrupole_cartesian_field(
    point: tuple[float, float, float], terms: tuple[float, float, float, float, float]
) -> tuple[float, float, float]:
    """Differentiate V/a = Q(u)/|u|^5; Q uses the degree-two Schmidt factors."""
    x, y, z = point
    g20, g21, h21, g22, h22 = terms
    q = (
        0.5 * g20 * (2.0 * z * z - x * x - y * y)
        + sqrt(3.0) * (g21 * x * z + h21 * y * z + h22 * x * y)
        + 0.5 * sqrt(3.0) * g22 * (x * x - y * y)
    )
    gradient_q = (
        -g20 * x + sqrt(3.0) * (g21 * z + g22 * x + h22 * y),
        -g20 * y + sqrt(3.0) * (h21 * z - g22 * y + h22 * x),
        2.0 * g20 * z + sqrt(3.0) * (g21 * x + h21 * y),
    )
    radius = sqrt(x * x + y * y + z * z)
    return tuple(
        5.0 * q * coordinate / radius**7 - derivative / radius**5
        for coordinate, derivative in zip(point, gradient_q, strict=True)
    )


@pytest.mark.parametrize("term_index", range(5))
@pytest.mark.parametrize("point", [(1.2, 0.7, -0.4), (0.0, 0.0, 1.3), (0.0, 0.0, -1.3)])
def test_isolated_degree_two_terms_equal_cartesian_quadrupole(
    point: tuple[float, float, float], term_index: int
) -> None:
    terms = tuple(1700.0 if index == term_index else 0.0 for index in range(5))
    g20, g21, h21, g22, h22 = terms
    coefficients = IgrfCoefficients(
        epoch=1985.0,
        g={(2, 0): g20, (2, 1): g21, (2, 2): g22},
        h={(2, 1): h21, (2, 2): h22},
        nmax=2,
    )
    assert _field_at_dimensionless_point(point, coefficients) == pytest.approx(
        _quadrupole_cartesian_field(point, terms), rel=2.0e-12, abs=1.0e-8
    )


def _single_harmonic_potential_over_a(
    point: list[float], *, degree: int, order: int, sine_term: bool
) -> float:
    """V/a in nT, using SciPy's Condon-Shortley convention explicitly undone."""
    x, y, z = point
    radius = sqrt(x * x + y * y + z * z)
    schmidt = sqrt(
        (1.0 if order == 0 else 2.0) * factorial(degree - order) / factorial(degree + order)
    )
    legendre = (-1.0) ** order * schmidt * float(lpmv(order, degree, z / radius))
    longitude_term = sin(order * atan2(y, x)) if sine_term else cos(order * atan2(y, x))
    return 1700.0 * legendre * longitude_term / radius ** (degree + 1)


@pytest.mark.parametrize("degree", [3, 7, 13])
@pytest.mark.parametrize("order_kind", ["zonal", "first", "sectoral"])
@pytest.mark.parametrize("sine_term", [False, True])
def test_higher_degree_synthesis_equals_independent_cartesian_potential_gradient(
    degree: int, order_kind: str, sine_term: bool
) -> None:
    order = {"zonal": 0, "first": 1, "sectoral": degree}[order_kind]
    # h(n,0) has identically zero potential; include it to check that invariant.
    coefficients = IgrfCoefficients(
        epoch=1985.0,
        g={(degree, order): 0.0 if sine_term else 1700.0},
        h={(degree, order): 1700.0 if sine_term else 0.0},
        nmax=degree,
    )
    point = (1.2, 0.7, -0.4)
    step = 1.0e-4
    expected: list[float] = []
    for axis in range(3):
        evaluations = []
        for offset in [-2.0, -1.0, 1.0, 2.0]:
            shifted = list(point)
            shifted[axis] += offset * step
            evaluations.append(
                _single_harmonic_potential_over_a(
                    shifted, degree=degree, order=order, sine_term=sine_term
                )
            )
        # The minus sign converts the fourth-order derivative into B = -grad V.
        expected.append(
            -(evaluations[0] - 8.0 * evaluations[1] + 8.0 * evaluations[2] - evaluations[3])
            / (12.0 * step)
        )
    assert _field_at_dimensionless_point(point, coefficients) == pytest.approx(
        expected, rel=2.0e-8, abs=2.0e-7
    )


@pytest.mark.parametrize("degree", [1, 2, 7, 13])
@pytest.mark.parametrize("pole_sign", [-1.0, 1.0])
def test_all_longitudinal_components_have_the_analytic_exact_pole_limit(
    degree: int, pole_sign: float
) -> None:
    # P_n'(1)=n(n+1)/2 and Schmidt's m=1 normalisation imply
    # lim(theta->0) Pbar_n^1(cos(theta))/sin(theta)=sqrt(n(n+1)/2).
    # P_n(-x)=(-1)^n P_n(x) fixes the southern signs independently.
    radius_re = 1.3
    g0, g1, h1 = 330.0, 1200.0, -1700.0
    coefficients = IgrfCoefficients(
        epoch=1985.0,
        g={(degree, 0): g0, (degree, 1): g1},
        h={(degree, 1): h1},
        nmax=degree,
    )
    radial_factor = radius_re ** (-degree - 2)
    parity = pole_sign ** (degree + 1)
    transverse_factor = -parity * sqrt(degree * (degree + 1) / 2.0) * radial_factor
    expected = (
        transverse_factor * g1,
        transverse_factor * h1,
        parity * (degree + 1) * radial_factor * g0,
    )
    assert _field_at_dimensionless_point(
        (0.0, 0.0, pole_sign * radius_re), coefficients
    ) == pytest.approx(expected, rel=2.0e-12, abs=1.0e-8)


@pytest.mark.parametrize("latitude_deg", [-40.0, -20.0, 20.0, 40.0])
def test_traced_axial_dipole_matches_analytic_field_line(latitude_deg: float) -> None:
    latitude = radians(latitude_deg)
    radius_re = 2.0
    dipole_nt = ERB_DIPOLE_MOMENT_GAUSS * 1.0e5
    coefficients = IgrfCoefficients(epoch=1985.0, g={(1, 0): -dipole_nt}, h={}, nmax=1)
    traced = trace_l_shell_b_over_b0(
        ErbOrbitSample(
            radius_km=radius_re * EARTH_REFERENCE_RADIUS_KM,
            theta_rad=pi / 2.0 - latitude,
            longitude_rad=0.7,
        ),
        coefficients,
        step_km=0.002 * EARTH_REFERENCE_RADIUS_KM,
        max_steps=4000,
    )
    expected_l = radius_re / cos(latitude) ** 2
    expected_b_over_b0 = sqrt(1.0 + 3.0 * sin(latitude) ** 2) / cos(latitude) ** 6
    assert traced.valid
    assert traced.l_shell == pytest.approx(expected_l, rel=2.0e-5)
    assert traced.mirror_b0_nt == pytest.approx(dipole_nt / expected_l**3, rel=2.0e-5)
    assert traced.b_over_b0 == pytest.approx(expected_b_over_b0, rel=2.0e-5)


def test_exhausted_trace_does_not_claim_a_located_minimum() -> None:
    coefficients = IgrfCoefficients(
        epoch=1985.0,
        g={(1, 0): -ERB_DIPOLE_MOMENT_GAUSS * 1.0e5},
        h={},
        nmax=1,
    )
    traced = trace_l_shell_b_over_b0(
        ErbOrbitSample(
            radius_km=2.0 * EARTH_REFERENCE_RADIUS_KM,
            theta_rad=radians(50.0),
            longitude_rad=0.0,
        ),
        coefficients,
        step_km=0.001 * EARTH_REFERENCE_RADIUS_KM,
        max_steps=1,
    )
    assert not traced.valid
