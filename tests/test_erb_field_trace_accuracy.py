"""Analytical dipole error and failure-path checks for magnetic-line tracing."""

from math import cos, pi, radians, sin, sqrt

import pytest

from radar.erb.constants import EARTH_REFERENCE_RADIUS_KM, ERB_DIPOLE_MOMENT_GAUSS
from radar.erb.field import trace_l_shell_b_over_b0
from radar.erb.igrf import IgrfCoefficients
from radar.erb.orbit import ErbOrbitSample


def _axial_dipole() -> IgrfCoefficients:
    return IgrfCoefficients(
        epoch=1985.0,
        g={(1, 0): -ERB_DIPOLE_MOMENT_GAUSS * 1.0e5},
        h={},
        nmax=1,
    )


def _sample(radius_re: float, latitude_deg: float) -> ErbOrbitSample:
    return ErbOrbitSample(
        radius_km=radius_re * EARTH_REFERENCE_RADIUS_KM,
        theta_rad=pi / 2.0 - radians(latitude_deg),
        longitude_rad=0.7,
    )


@pytest.mark.parametrize("radius_re", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("latitude_deg", [-60.0, -40.0, -20.0, 20.0, 40.0, 60.0])
def test_default_and_half_trace_steps_converge_to_exact_dipole(
    radius_re: float, latitude_deg: float
) -> None:
    latitude = radians(latitude_deg)
    expected_l = radius_re / cos(latitude) ** 2
    expected_b0 = ERB_DIPOLE_MOMENT_GAUSS * 1.0e5 / expected_l**3
    expected_ratio = sqrt(1.0 + 3.0 * sin(latitude) ** 2) / cos(latitude) ** 6
    coefficients = _axial_dipole()
    sample = _sample(radius_re, latitude_deg)

    default = trace_l_shell_b_over_b0(sample, coefficients)
    half = trace_l_shell_b_over_b0(sample, coefficients, step_km=0.03 * EARTH_REFERENCE_RADIUS_KM)
    for result in (default, half):
        assert result.valid
        assert result.incomplete_reasons == ()
        assert result.l_shell == pytest.approx(expected_l, rel=2.0e-6)
        assert result.mirror_b0_nt == pytest.approx(expected_b0, rel=2.0e-6)
        assert result.b_over_b0 == pytest.approx(expected_ratio, rel=2.0e-6)
        assert result.local_b_nt / result.mirror_b0_nt == pytest.approx(result.b_over_b0)

    default_error = abs(default.mirror_b0_nt / expected_b0 - 1.0)
    half_error = abs(half.mirror_b0_nt / expected_b0 - 1.0)
    # RK4's fourth-order convergence predicts roughly 1/16. Permit 1/10
    # because the final minimum is found inside a different step interval.
    assert half_error < 0.1 * default_error + 2.0e-12


@pytest.mark.parametrize("radius_re", [1.0, 2.0])
@pytest.mark.parametrize("latitude_deg", [-0.1, -0.0001, 0.0, 0.0001, 0.1])
def test_minimum_inside_first_step_is_refined_instead_of_using_initial_point(
    radius_re: float, latitude_deg: float
) -> None:
    expected_l = radius_re / cos(radians(latitude_deg)) ** 2
    result = trace_l_shell_b_over_b0(_sample(radius_re, latitude_deg), _axial_dipole(), max_steps=1)
    assert result.valid
    assert result.incomplete_reasons == ()
    assert result.l_shell == pytest.approx(expected_l, rel=2.0e-10)
    assert result.mirror_b0_nt == pytest.approx(
        ERB_DIPOLE_MOMENT_GAUSS * 1.0e5 / expected_l**3, rel=2.0e-10
    )
    if abs(latitude_deg) == 0.1:
        assert result.b_over_b0 > 1.0 + 1.0e-5
        assert result.mirror_b0_nt < result.local_b_nt


@pytest.mark.parametrize("latitude_deg", [-60.0, -40.0, 40.0, 60.0])
def test_unreached_dipole_minimum_is_explicit_failure(latitude_deg: float) -> None:
    result = trace_l_shell_b_over_b0(_sample(2.0, latitude_deg), _axial_dipole(), max_steps=1)
    assert not result.valid
    assert any("max_steps" in reason for reason in result.incomplete_reasons)


@pytest.mark.parametrize("latitude_deg", [-90.0, 90.0])
def test_dipole_axis_has_no_finite_equator(latitude_deg: float) -> None:
    result = trace_l_shell_b_over_b0(_sample(2.0, latitude_deg), _axial_dipole(), max_steps=32)
    assert not result.valid
    assert any("max_steps" in reason for reason in result.incomplete_reasons)


def test_trace_start_below_stated_domain_cannot_escape_and_become_valid() -> None:
    # At this latitude the first outward step crosses 0.98 R. Checking only
    # later positions would accept a trace that started outside its domain.
    result = trace_l_shell_b_over_b0(_sample(0.97, 40.0), _axial_dipole())
    assert not result.valid
    assert result.incomplete_reasons
