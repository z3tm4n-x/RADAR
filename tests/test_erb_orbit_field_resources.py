from math import isfinite, pi

import pytest

from radar.erb.constants import EARTH_REFERENCE_RADIUS_KM, ERB_DIPOLE_MOMENT_GAUSS
from radar.erb.field import (
    magnetic_field_at_orbit_sample,
    magnetic_field_vector_km,
    trace_l_shell_b_over_b0,
)
from radar.erb.igrf import IgrfCoefficients, parse_igrf_coefficients
from radar.erb.orbit import ErbOrbitSample, sample_erb_orbit_grid
from radar.erb.radbelt import parse_radbelt_asc_text


def _dipole_coefficients() -> IgrfCoefficients:
    return IgrfCoefficients(
        epoch=1985.0,
        g={(1, 0): -ERB_DIPOLE_MOMENT_GAUSS * 1.0e5},
        h={},
        nmax=1,
    )


def _asc_line(values: tuple[int, ...]) -> str:
    return " " + "".join(f"{value:6d}" for value in values)


def test_erb_orbit_grid_samples_circular_equatorial_orbit() -> None:
    grid = sample_erb_orbit_grid(
        perigee_km=500.0,
        apogee_km=500.0,
        inclination_deg=0.0,
        anomaly_samples=4,
        node_samples=3,
    )

    assert len(grid.samples) == 12
    assert all(sample.radius_km == pytest.approx(EARTH_REFERENCE_RADIUS_KM + 500.0) for sample in grid.samples)
    assert all(sample.theta_rad == pytest.approx(pi / 2.0) for sample in grid.samples)


def test_erb_orbit_grid_rejects_apogee_below_perigee() -> None:
    with pytest.raises(ValueError, match="apogee"):
        sample_erb_orbit_grid(
            perigee_km=1000.0,
            apogee_km=500.0,
            inclination_deg=10.0,
        )


def test_parse_igrf_coefficients_interpolates_epoch() -> None:
    text = """\
g/h n m 1980.0 1985.0 1990.0 2025-30
g  1  0 -30000 -29900 -29800 10
g  1  1  -2000  -1900  -1800 20
h  1  1   5000   5100   5200 -5
"""

    coefficients = parse_igrf_coefficients(text, epoch=1987.5)

    assert coefficients.nmax == 1
    assert coefficients.g_nm(1, 0) == pytest.approx(-29850.0)
    assert coefficients.g_nm(1, 1) == pytest.approx(-1850.0)
    assert coefficients.h_nm(1, 1) == pytest.approx(5150.0)
    assert coefficients.h_nm(2, 0) == 0.0


def test_parse_igrf_coefficients_extrapolates_with_sv() -> None:
    text = """\
g/h n m 2020.0 2025.0 2025-30
g  1  0 -29403.41 -29350.0 12.6
"""

    coefficients = parse_igrf_coefficients(text, epoch=2027.0)

    assert coefficients.g_nm(1, 0) == pytest.approx(-29324.8)


def test_magnetic_field_vector_matches_equatorial_dipole_magnitude() -> None:
    field = magnetic_field_vector_km(
        x_km=EARTH_REFERENCE_RADIUS_KM,
        y_km=0.0,
        z_km=0.0,
        coefficients=_dipole_coefficients(),
    )

    assert field.magnitude_nt == pytest.approx(ERB_DIPOLE_MOMENT_GAUSS * 1.0e5)
    assert field.bx_nt == pytest.approx(0.0, abs=1.0e-10)
    assert field.by_nt == pytest.approx(0.0)


def test_magnetic_field_at_orbit_sample_and_trace_equatorial_dipole() -> None:
    sample = ErbOrbitSample(
        radius_km=EARTH_REFERENCE_RADIUS_KM,
        theta_rad=pi / 2.0,
        longitude_rad=0.0,
    )
    coefficients = _dipole_coefficients()

    field = magnetic_field_at_orbit_sample(sample, coefficients)
    traced = trace_l_shell_b_over_b0(sample, coefficients)

    assert field.magnitude_nt == pytest.approx(ERB_DIPOLE_MOMENT_GAUSS * 1.0e5)
    assert traced.valid is True
    assert traced.l_shell == pytest.approx(1.0)
    assert traced.b_over_b0 == pytest.approx(1.0)
    assert traced.local_b_nt == pytest.approx(traced.mirror_b0_nt)


def test_trace_field_line_rejects_invalid_max_steps() -> None:
    sample = ErbOrbitSample(
        radius_km=EARTH_REFERENCE_RADIUS_KM,
        theta_rad=pi / 2.0,
        longitude_rad=0.0,
    )

    with pytest.raises(ValueError, match="max_steps"):
        trace_l_shell_b_over_b0(
            sample,
            _dipole_coefficients(),
            max_steps=0,
        )


def test_parse_radbelt_asc_text_reads_descriptor_and_map() -> None:
    text = _asc_line((2, 4, 1970, 100, 2048, 2048, 1024, 4, 11, 22, 33, 44))

    parsed = parse_radbelt_asc_text(name="AP8TEST", text=text)

    assert parsed.name == "AP8TEST"
    assert parsed.descriptor == (2, 4, 1970, 100, 2048, 2048, 1024, 4)
    assert parsed.map_values == (11, 22, 33, 44)
    assert parsed.fistep == pytest.approx(256.0)
    assert parsed.descriptor_1based(8) == 4


def test_parse_radbelt_asc_text_rejects_map_length_mismatch() -> None:
    text = _asc_line((2, 4, 1970, 100, 2048, 2048, 1024, 4, 11, 22, 33))

    with pytest.raises(ValueError, match="mismatch"):
        parse_radbelt_asc_text(name="BAD", text=text)


def test_trace_result_values_are_finite_for_valid_sample() -> None:
    sample = ErbOrbitSample(
        radius_km=EARTH_REFERENCE_RADIUS_KM + 500.0,
        theta_rad=pi / 2.0,
        longitude_rad=0.0,
    )

    traced = trace_l_shell_b_over_b0(sample, _dipole_coefficients())

    assert traced.valid is True
    assert isfinite(traced.l_shell)
    assert isfinite(traced.b_over_b0)
