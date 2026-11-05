"""Fast synthetic and mocked contracts for the COSRAD preparation report."""

from dataclasses import replace
import json
from math import nan, nextafter, pi
from pathlib import Path

import pytest

import radar.erb.field as field_module
from radar.erb.field import ErbFieldLineSample, shifted_dipole_l_shell_b_over_b0
from radar.erb.igrf import IgrfCoefficients
from radar.erb.orbit import ErbOrbitSample
import radar.validation.erb_magnetic_boundary as validation


@pytest.fixture
def fast_traces(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float, int]]:
    calls: list[tuple[float, float, int]] = []

    def load(*, epoch: float) -> IgrfCoefficients:
        return IgrfCoefficients(epoch, {(1, 0): -31165.3}, {}, 1)

    def trace(
        sample: ErbOrbitSample,
        coefficients: IgrfCoefficients,
        *,
        step_km: float,
        max_steps: int,
    ) -> ErbFieldLineSample:
        calls.append((coefficients.epoch, step_km, max_steps))
        return ErbFieldLineSample(
            l_shell=5.4 + step_km / 6371.2,
            b_over_b0=2.0,
            local_b_nt=400.0,
            mirror_b0_nt=200.0,
            valid=True,
            method="synthetic_igrf_for_test_only",
        )

    monkeypatch.setattr(validation, "load_igrf14_coefficients", load)
    monkeypatch.setattr(validation, "trace_l_shell_b_over_b0", trace)
    # Keep the real production selector, but bound its internal trace as well.
    monkeypatch.setattr(field_module, "trace_l_shell_b_over_b0", trace)
    return calls


def test_full_matrix_explicit_epochs_and_three_evaluations_per_point(fast_traces) -> None:
    report = validation.run_erb_magnetic_boundary_validation(epochs=[2026, 1985, 2026])
    assert report["matrix"]["epochs"] == [1985.0, 2026.0]
    assert report["matrix"]["case_count"] == 36
    assert len({case["case_id"] for case in report["cases"]}) == 36
    assert len(fast_traces) == 108
    assert {item[1] for item in fast_traces} == {0.06 * 6371.2, 0.03 * 6371.2}
    assert {item[2] for item in fast_traces} == {1500}
    assert report["source"]["epochs"][0]["loaded_epoch"] == 1985.0
    assert len(report["source"]["igrf_resource_sha256"]) == 64
    assert len(report["source"]["epochs"][0]["loaded_coefficients_sha256"]) == 64


def test_success_cannot_be_claimed_as_cosrad_or_normative_validation(fast_traces) -> None:
    report = validation.run_erb_magnetic_boundary_validation(epochs=[1985])
    assert report["summary"]["invalid_selection_count"] == 0
    assert report["intended_use"] == {
        "diagnostic_only": True,
        "numerical_gold": False,
        "normative_export_allowed": False,
    }
    assert report["cosrad_reference"] == {
        "status": "not_supplied",
        "comparison_performed": False,
        "values": None,
    }
    assert report["release_gate"]["report_eligible_for_normative_export"] is False
    assert report["release_gate"]["blockers"]


def test_known_fixture_coordinates_and_geographic_fields(fast_traces) -> None:
    report = validation.run_erb_magnetic_boundary_validation(epochs=[1985])
    assert report["matrix"]["magnetic_azimuths_rad"] == [0.0, 0.9, pi]
    for case in report["cases"]:
        expected_l = case["target_magnetic_coordinates"]["shifted_dipole_l_shell"]
        geographic = case["geographic_input"]
        point = ErbOrbitSample(
            radius_km=geographic["radius_km"],
            theta_rad=geographic["colatitude_rad"],
            longitude_rad=geographic["longitude_rad"],
        )
        assert geographic["coordinate_system"] == "geocentric_not_geodetic"
        assert shifted_dipole_l_shell_b_over_b0(point).l_shell == pytest.approx(
            expected_l, rel=2.0e-14
        )
        expected_ratio = (
            1.0 if case["target_magnetic_coordinates"]["latitude_deg"] == 0 else (3.135705257558032)
        )
        assert case["candidates"]["shifted_dipole"]["b_over_b0"] == pytest.approx(expected_ratio)


def test_relative_differences_use_named_reference_and_signed_dimensionless_units(
    fast_traces,
) -> None:
    report = validation.run_erb_magnetic_boundary_validation(epochs=[1985])
    case = report["cases"][0]
    differences = case["relative_differences"]
    assert differences["igrf_0.06_re_vs_0.03_re"]["l_shell"] == pytest.approx(5.46 / 5.43 - 1)
    assert differences["igrf_0.06_re_vs_0.03_re"]["b_over_b0"] == 0.0
    assert differences["shifted_dipole_vs_igrf_0.03_re"]["b_over_b0"] == -0.5


def test_roundoff_on_exact_boundary_is_recorded_without_snapping(
    fast_traces, monkeypatch: pytest.MonkeyPatch
) -> None:
    dipole_l = nextafter(5.5, float("inf"))
    result = ErbFieldLineSample(
        dipole_l, 1.0, 200.0, 200.0, True, method="synthetic_dipole_for_test_only"
    )
    monkeypatch.setattr(validation, "shifted_dipole_l_shell_b_over_b0", lambda sample: result)
    monkeypatch.setattr(field_module, "shifted_dipole_l_shell_b_over_b0", lambda sample: result)
    report = validation.run_erb_magnetic_boundary_validation(epochs=[1985])
    cases = [
        case for case in report["cases"] if case["boundary_roundoff"]["nominal_l_equals_boundary"]
    ]
    assert len(cases) == 6
    for case in cases:
        rounding = case["boundary_roundoff"]
        assert rounding["nominal_region"] == "inner"
        assert rounding["recomputed_classifier_region"] == "outer"
        assert rounding["recomputed_l_minus_nominal_in_ulps"] == 1.0
        assert rounding["snapping_applied"] is False
        assert case["production_selection"]["l_shell"] == dipole_l
        assert case["production_selection"]["method"] == result.method


def test_failed_candidates_are_preserved_with_reasons_and_json_null(
    fast_traces, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = validation.trace_l_shell_b_over_b0

    def trace(sample, coefficients, *, step_km, max_steps):
        if step_km > 0.04 * 6371.2:
            return ErbFieldLineSample(
                nan,
                nan,
                nan,
                nan,
                False,
                method="synthetic_failure",
                incomplete_reasons=("Magnetic equator not located within max_steps",),
            )
        return original(sample, coefficients, step_km=step_km, max_steps=max_steps)

    monkeypatch.setattr(validation, "trace_l_shell_b_over_b0", trace)
    monkeypatch.setattr(field_module, "trace_l_shell_b_over_b0", trace)
    report = validation.run_erb_magnetic_boundary_validation(epochs=[1985], max_steps=1)
    assert report["summary"]["invalid_candidate_count"] == 18
    assert report["summary"]["invalid_selection_count"] > 0
    for case in report["cases"]:
        coarse = case["candidates"]["igrf_step_0.06_re"]
        assert coarse["valid"] is False
        assert coarse["l_shell"] is None
        assert coarse["incomplete_reasons"]
        assert set(case["relative_differences"]["igrf_0.06_re_vs_0.03_re"].values()) == {None}
    rendered = validation.erb_magnetic_boundary_validation_json(report)
    assert "NaN" not in rendered
    assert json.loads(rendered)["summary"]["invalid_candidate_count"] == 18


@pytest.mark.parametrize("epochs", [[], [1900], [1985, 2000], [nan], [float("inf")]])
def test_unsupported_or_empty_epochs_rejected(epochs) -> None:
    with pytest.raises(ValueError, match="epoch"):
        validation.run_erb_magnetic_boundary_validation(epochs=epochs)


@pytest.mark.parametrize("budget", [0, -1, 1501, True, 1.5])
def test_unbounded_or_noninteger_step_budgets_rejected(budget) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        validation.run_erb_magnetic_boundary_validation(epochs=[1985], max_steps=budget)


def test_report_is_deterministic_for_equivalent_epoch_requests(fast_traces) -> None:
    first = validation.run_erb_magnetic_boundary_validation(epochs=[1985, 2026])
    second = validation.run_erb_magnetic_boundary_validation(epochs=[2026, 1985, 1985])
    assert validation.erb_magnetic_boundary_validation_json(first) == (
        validation.erb_magnetic_boundary_validation_json(second)
    )


def test_cli_requires_explicit_supported_epoch() -> None:
    for arguments in ([], ["--epoch", "2000"], ["--epoch", "1985", "--max-steps", "0"]):
        with pytest.raises(SystemExit) as error:
            validation.main(arguments)
        assert error.value.code == 2


def test_cli_writes_same_report_to_stdout_or_utf8_file(fast_traces, capsys, tmp_path: Path) -> None:
    assert validation.main(["--epoch", "1985"]) == 0
    stdout = capsys.readouterr().out
    destination = tmp_path / "boundary.json"
    assert validation.main(["--epoch", "1985", "--output", str(destination)]) == 0
    assert destination.read_text(encoding="utf-8") == stdout
    assert json.loads(stdout)["matrix"]["case_count"] == 18


def test_epoch_fingerprint_identifies_actual_loaded_coefficient_values(
    fast_traces, monkeypatch
) -> None:
    first = validation.run_erb_magnetic_boundary_validation(epochs=[1985])
    monkeypatch.setattr(
        validation,
        "load_igrf14_coefficients",
        lambda *, epoch: replace(
            IgrfCoefficients(epoch, {(1, 0): -31165.3}, {}, 1), g={(1, 0): -30000.0}
        ),
    )
    second = validation.run_erb_magnetic_boundary_validation(epochs=[1985])
    assert (
        first["source"]["epochs"][0]["loaded_coefficients_sha256"]
        != (second["source"]["epochs"][0]["loaded_coefficients_sha256"])
    )
