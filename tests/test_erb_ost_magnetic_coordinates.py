"""OST Appendix D geometry, strict branch boundary and model wiring."""

from dataclasses import replace
from math import acos, atan2, cos, pi, radians, sin, sqrt

import pytest

import radar.erb.field as field_module
import radar.erb.model as model_module
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.core.types import SpectrumCompleteness
from radar.erb.constants import EARTH_REFERENCE_RADIUS_KM
from radar.erb.field import (
    ErbFieldLineSample,
    ost_l_shell_b_over_b0,
    shifted_dipole_l_shell_b_over_b0,
)
from radar.erb.igrf import IgrfCoefficients
from radar.erb.model import Ae8Ap8ErbModel, ErbModelInput, OstErbModel
from radar.erb.orbit import ErbOrbitSample
from radar.erb.resources import load_igrf14_coefficients


def _coefficients() -> IgrfCoefficients:
    return IgrfCoefficients(epoch=1985.0, g={(1, 0): -31165.3}, h={}, nmax=1)


def _shifted_dipole_point(l_shell: float, latitude_deg: float, azimuth: float) -> ErbOrbitSample:
    """Construct a known dipole field line by an independent rigid rotation.

    The constants here are transcribed from the supplied PDF, not imported
    from production. This catches swapped latitude/longitude and cm/km errors.
    """
    tilt, longitude = radians(11.0), radians(291.0)
    axis = (sin(tilt) * cos(longitude), sin(tilt) * sin(longitude), cos(tilt))
    first = (cos(tilt) * cos(longitude), cos(tilt) * sin(longitude), -sin(tilt))
    second = (-sin(longitude), cos(longitude), 0.0)
    latitude = radians(latitude_deg)
    radius = l_shell * 6371.2 * cos(latitude) ** 2
    offset_lat, offset_lon = radians(15.5), radians(150.5)
    offset = (
        430.0 * cos(offset_lat) * cos(offset_lon),
        430.0 * cos(offset_lat) * sin(offset_lon),
        430.0 * sin(offset_lat),
    )
    point = tuple(
        offset[i]
        + radius
        * (
            cos(latitude) * (cos(azimuth) * first[i] + sin(azimuth) * second[i])
            + sin(latitude) * axis[i]
        )
        for i in range(3)
    )
    geocentric_radius = sqrt(sum(value * value for value in point))
    return ErbOrbitSample(
        radius_km=geocentric_radius,
        theta_rad=acos(point[2] / geocentric_radius),
        longitude_rad=atan2(point[1], point[0]),
    )


@pytest.mark.parametrize("l_shell", [2.0, 5.5, 6.6])
@pytest.mark.parametrize("latitude_deg", [-40.0, 0.0, 35.0])
@pytest.mark.parametrize("azimuth", [0.0, 0.9, pi])
def test_shifted_dipole_d21_d24_with_independent_inverse_geometry(
    l_shell: float, latitude_deg: float, azimuth: float
) -> None:
    result = shifted_dipole_l_shell_b_over_b0(_shifted_dipole_point(l_shell, latitude_deg, azimuth))
    latitude = radians(latitude_deg)
    expected_ratio = sqrt(1.0 + 3.0 * sin(latitude) ** 2) / cos(latitude) ** 6
    expected_b0 = 8.06e25 / (l_shell * 6371.2e5) ** 3 * 1.0e5
    assert result.valid
    assert result.l_shell == pytest.approx(l_shell, rel=2.0e-14)
    assert result.b_over_b0 == pytest.approx(expected_ratio, rel=2.0e-14)
    assert result.mirror_b0_nt == pytest.approx(expected_b0, rel=2.0e-14)
    assert result.local_b_nt == pytest.approx(expected_b0 * expected_ratio, rel=2.0e-14)
    assert result.local_b_nt / result.mirror_b0_nt == pytest.approx(result.b_over_b0)


def test_outer_moment_factor_cancels_from_radiation_coordinates() -> None:
    point = _shifted_dipole_point(6.0, 30.0, 0.4)
    reference = shifted_dipole_l_shell_b_over_b0(point)
    scaled = shifted_dipole_l_shell_b_over_b0(point, moment_scale=0.9)
    assert scaled.l_shell == reference.l_shell
    assert scaled.b_over_b0 == reference.b_over_b0
    assert scaled.local_b_nt == pytest.approx(0.9 * reference.local_b_nt)
    assert scaled.mirror_b0_nt == pytest.approx(0.9 * reference.mirror_b0_nt)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_outer_moment_scale_rejected(scale: float) -> None:
    with pytest.raises(ValueError, match="moment scale"):
        shifted_dipole_l_shell_b_over_b0(_shifted_dipole_point(6.0, 0.0, 0.0), moment_scale=scale)


@pytest.mark.parametrize(
    "pilot_l,outer", [(5.5 - 1.0e-8, False), (5.5, False), (5.5 + 1.0e-8, True)]
)
def test_d26_boundary_is_strictly_outer_above_55_only(
    monkeypatch: pytest.MonkeyPatch, pilot_l: float, outer: bool
) -> None:
    # The exact comparator is tested independently of roundoff in rotation.
    dipole = ErbFieldLineSample(pilot_l, 2.0, 200.0, 100.0, True, method="dipole")
    inner = ErbFieldLineSample(5.4, 1.2, 240.0, 200.0, True, method="igrf")
    monkeypatch.setattr(field_module, "shifted_dipole_l_shell_b_over_b0", lambda sample: dipole)
    monkeypatch.setattr(field_module, "trace_l_shell_b_over_b0", lambda *args, **kwargs: inner)
    result = ost_l_shell_b_over_b0(_shifted_dipole_point(5.5, 0.0, 0.0), _coefficients())
    expected = dipole if outer else inner
    assert result.l_shell == expected.l_shell
    assert result.local_b_nt == expected.local_b_nt
    assert result.mirror_b0_nt == expected.mirror_b0_nt
    assert result.method == expected.method
    assert result.classifier_l_shell == pilot_l
    assert bool(result.incomplete_reasons) is outer


@pytest.mark.parametrize("pilot_l,inner_l", [(5.4, 5.6), (5.6, 5.4)])
def test_candidate_region_disagreement_is_not_silenced(
    monkeypatch: pytest.MonkeyPatch, pilot_l: float, inner_l: float
) -> None:
    dipole = ErbFieldLineSample(pilot_l, 2.0, 200.0, 100.0, True)
    inner = replace(dipole, l_shell=inner_l)
    monkeypatch.setattr(field_module, "shifted_dipole_l_shell_b_over_b0", lambda sample: dipole)
    monkeypatch.setattr(field_module, "trace_l_shell_b_over_b0", lambda *args, **kwargs: inner)
    result = ost_l_shell_b_over_b0(_shifted_dipole_point(5.5, 0.0, 0.0), _coefficients())
    assert result.valid
    assert any("different regions" in reason for reason in result.incomplete_reasons)


def test_inner_trace_failure_does_not_fall_back_to_valid_dipole() -> None:
    sample = ErbOrbitSample(2.0 * EARTH_REFERENCE_RADIUS_KM, radians(60.0), 0.0)
    result = ost_l_shell_b_over_b0(sample, _coefficients(), max_steps=1)
    assert not result.valid
    assert result.classifier_l_shell is not None
    assert any("max_steps" in reason for reason in result.incomplete_reasons)


def test_outer_is_not_replaced_by_failed_diagnostic_igrf() -> None:
    point = _shifted_dipole_point(5.6, 30.0, 0.0)
    result = ost_l_shell_b_over_b0(point, _coefficients(), max_steps=1)
    assert result.valid
    assert result.method == "ost_shifted_dipole_d21_d24"
    assert any("comparison failed" in reason for reason in result.incomplete_reasons)


def _input(altitude: float = 35786.0) -> ErbModelInput:
    return ErbModelInput(
        config=CalculationConfig(
            orbit=OrbitConfig.circular(altitude_km=altitude, inclination_deg=0.0),
            mission=MissionConfig(launch_year=2026, lifetime_years=5),
        )
    )


def test_ost_model_uses_outer_branch_on_geo_and_igrf_on_low_equatorial_orbit() -> None:
    model = OstErbModel(anomaly_samples=4, node_samples=2, igrf_coefficients=_coefficients())
    assert all(
        sample.method == "ost_shifted_dipole_d21_d24"
        for sample in model._field_line_samples(_input())
    )
    assert all(
        sample.method == "igrf_rk4_minimum_b" for sample in model._field_line_samples(_input(400.0))
    )


def test_experimental_ae8_ap8_dispatch_not_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = (ErbFieldLineSample(6.0, 1.0, 100.0, 100.0, True),)
    monkeypatch.setattr(model_module, "trace_l_shell_b_over_b0_for_orbit", lambda *args: expected)
    assert (
        Ae8Ap8ErbModel(anomaly_samples=1, node_samples=1)._field_line_samples(_input()) == expected
    )


def test_magnetic_diagnostic_reasons_reach_every_ost_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = "OST D.2.6 IGRF and shifted-dipole L candidates select different regions"
    sample = ErbFieldLineSample(2.0, 1.0, 3000.0, 3000.0, True, incomplete_reasons=(reason,))
    monkeypatch.setattr(OstErbModel, "_field_line_samples", lambda *args: (sample,))
    result = OstErbModel().calculate(_input(1000.0))
    assert len(result.products) == 6
    for product in result.products:
        assert product.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
        assert reason in product.spectrum.incomplete_reasons
    metadata = dict(result.method_metadata)
    assert metadata["appendix_d_region_selection"] == "L_shifted_dipole_gt_5.5_outer_else_inner"
    assert metadata["appendix_d_boundary_smoothing"] == "none"
    assert metadata["magnetic_valid_samples"] == "1"
    assert metadata["appendix_d_diagnostic_samples"] == "1"


def test_real_igrf_boundary_disagreement_is_diagnostic_not_artificially_smoothed() -> None:
    """Actual coefficients, not a claim of COSRAD agreement or a gold spectrum."""
    coefficients = load_igrf14_coefficients(epoch=1985.0)
    below = ost_l_shell_b_over_b0(_shifted_dipole_point(5.5 - 1.0e-8, 30.0, 0.9), coefficients)
    above = ost_l_shell_b_over_b0(_shifted_dipole_point(5.5 + 1.0e-8, 30.0, 0.9), coefficients)
    assert below.valid and above.valid
    assert below.method == "igrf_rk4_minimum_b"
    assert above.method == "ost_shifted_dipole_d21_d24"
    assert below.l_shell < 5.5 < above.l_shell
    assert any("different regions" in reason for reason in above.incomplete_reasons)
    assert above.alternative_l_shell == pytest.approx(below.l_shell, rel=1.0e-7)
    expected_ratio = sqrt(1.0 + 3.0 * sin(radians(30.0)) ** 2) / cos(radians(30.0)) ** 6
    assert above.b_over_b0 == pytest.approx(expected_ratio)
    # This broad guard detects concealed model jumps; it is not an acceptance tolerance.
    assert abs(above.b_over_b0 / below.b_over_b0 - 1.0) > 0.05


@pytest.mark.parametrize("model_type", [OstErbModel, Ae8Ap8ErbModel])
def test_method_metadata_records_epoch_of_the_actual_coefficients(model_type: type) -> None:
    model = model_type(
        anomaly_samples=1,
        node_samples=1,
        igrf_coefficients=replace(_coefficients(), epoch=2026.0),
    )
    metadata = dict(model.calculate(_input(1000.0)).method_metadata)
    assert metadata["igrf_epoch"] == "2026"
    assert metadata["igrf_configured_epoch"] == "1985"
