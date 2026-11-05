"""Pinned-source numerical checks must not confer physical qualification."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/validate_nymmik_source.py"
    spec = importlib.util.spec_from_file_location("nymmik_source_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source_text(runner: ModuleType) -> str:
    texts, _ = runner._read_verified_sources()
    return str(texts[(runner._EVIDENCE / "cornymmik").as_posix()])


def test_parser_preserves_all_eighteen_numbers_with_signed_phases(
    runner: ModuleType, source_text: str,
) -> None:
    actual = runner.parse_coefficient_strings(source_text)
    expected = {
        "Aa": (-0.037, -5.844, 0.357), "Ba": (-0.267, -5.198, 6.073),
        "Ab": (0.0022, -6.448, 0.00177), "Bb": (0.0091, -6.390, -0.30538),
        "Ac": (0.0768, 6.082, 0.0769), "Bc": (2.3564, 5.785, 3.5876),
    }
    assert set(actual) == set(expected)
    for name, (amplitude, phase, offset) in expected.items():
        assert actual[name].amplitude == amplitude
        assert actual[name].phase_hours == phase
        assert actual[name].offset == offset


@pytest.mark.parametrize(("old", "new"), [
    ('$hr = "15.0"', '$hr = "24.0"'),
    ('(#pi/180.0)', '(#pi/12.0)'),
    ('$a = "($Aa*$kp + $Ba)"', '$a = "($Aa*$kp*$kp + $Ba)"'),
    ('0.001*exp', '0.002*exp'),
    ('($cor / ($delta))', '($cor * ($delta))'),
    ('-0.037*sin', '-0.037*cos'),
    ('5.844*$hr', '5.844*system("bad")'),
])
def test_parser_rejects_unrecognized_units_or_expression_grammar(
    runner: ModuleType, source_text: str, old: str, new: str,
) -> None:
    assert old in source_text
    with pytest.raises(ValueError):
        runner.parse_coefficient_strings(source_text.replace(old, new))


def test_duplicate_or_missing_assignment_is_rejected(runner: ModuleType, source_text: str) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        runner.parse_coefficient_strings(source_text + '\n$Aa = "duplicate";\n')
    with pytest.raises(ValueError, match="exactly one"):
        runner.parse_coefficient_strings(source_text.replace('$Aa = "', '$Other = "'))


def test_reader_rejects_mutated_pinned_source_without_execution(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dict(runner._SOURCES)
    changed[next(iter(changed))] = "0" * 64
    monkeypatch.setattr(runner, "_SOURCES", changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        runner._read_verified_sources()


def test_runner_has_no_dynamic_execution_or_process_launch_calls(runner: ModuleType) -> None:
    assert runner.__file__ is not None
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    builtin_forbidden = {"eval", "exec", "compile", "__import__"}
    process_forbidden = {"system", "popen", "Popen", "run", "check_call", "check_output"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in builtin_forbidden | process_forbidden
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in process_forbidden


def test_degree_input_equals_hour_input_after_explicit_factor_fifteen(
    runner: ModuleType, source_text: str,
) -> None:
    coefficients = runner.parse_coefficient_strings(source_text)
    for hours in (0.0, 6.0, 12.0, 18.0, 23.75):
        assert runner.source_coefficients(coefficients, 3, hours * 15.0) == pytest.approx(
            runner.ost_disturbance_coefficients(kp=3, magnetic_local_time_hours=hours),
            rel=2e-14,
        )


@pytest.mark.parametrize("cutoff", [0.0, -1.0, 1e-5, math.inf, math.nan])
def test_source_reference_does_not_silently_apply_radar_floor(
    runner: ModuleType, cutoff: float,
) -> None:
    with pytest.raises(ValueError, match="restricted"):
        runner.source_delta(cutoff, (2.0, 6.0, -0.3))


def test_cardinal_clock_reproduction_exposes_twelve_hour_source_conflict(runner: ModuleType) -> None:
    comparison = runner.clock_comparison()
    cardinal = comparison["cardinal_rows"]
    assert [row["documented_solar_hours"] for row in cardinal] == [12.0, 18.0, 0.0, 6.0]
    assert [row["radar_solar_hours"] for row in cardinal] == [12.0, 18.0, 0.0, 6.0]
    assert [row["translated_prefilter_hours"] for row in cardinal] == [0.0, 6.0, 12.0, 18.0]
    assert comparison["source_code_documentation_conflict_detected"] is True
    assert comparison["case_count"] == 364
    assert len(comparison["synthetic_rows"]) == 360
    assert comparison["maximum_deviation_from_12_hour_shift"] < 1e-12
    assert comparison["full_prefilter_executed"] is False
    assert comparison["ephemeris_replayed"] is False
    assert comparison["normative_clock_selection_performed"] is False


@pytest.mark.parametrize("position", [(0.0, 0.0, 1.0), (math.nan, 1.0, 0.0), (True, 0.0, 0.0)])
def test_clock_translation_rejects_undefined_or_invalid_longitude(
    runner: ModuleType, position: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError):
        runner.translated_prefilter_local_time_degrees(position, (1.0, 0.0, 0.0))


@pytest.fixture(scope="module")
def report(runner: ModuleType) -> dict[str, Any]:
    result: dict[str, Any] = runner.build_report()
    return result


def test_bounded_grid_records_formula_evidence_but_no_normative_acceptance(
    runner: ModuleType, report: dict[str, Any],
) -> None:
    formula = report["formula_comparison"]
    assert formula["coefficient_case_count"] == len(formula["coefficient_rows"]) == 960
    assert formula["cutoff_case_count"] == len(formula["cutoff_rows"]) == 6720
    assert formula["hours"] == tuple(index / 4 for index in range(96))
    assert formula["kp_values"] == tuple(range(10))
    assert formula["numerical_comparison_passed"] is True
    assert formula["clock_input_semantics_qualified_by_numeric_agreement"] is False
    for key in ("max_coefficient_relative_error", "max_delta_relative_error",
                "max_cutoff_relative_error"):
        assert formula[key] < 2e-13
    for key in ("normative_release_eligible", "mission_radiation_history_qualified",
                "stage4_spectral_convergence_established", "production_policy_changed",
                "NASA_external_code_executed", "complete_HEASoft_replay_performed",
                "zero_cutoff_comparison_performed", "below_floor_comparison_performed"):
        assert report[key] is False
    assert report["source_sha256_before"] == report["source_sha256_after"] == runner._SOURCES
    assert report["model_sha256_before"] == report["model_sha256_after"]
    assert report["source_cutoff_column_unit_label"] == "GeV/c"
    assert report["radar_rigidity_unit"] == "GV"
    assert report["rigidity_numeric_scale_assumption"] == (
        "same_numeric_cutoff_scalar_as_in_matching_formula_"
        "not_general_particle_momentum_conversion"
    )
    assert report["ion_momentum_unit_conversion_verified"] is False
    assert any("no FITS replay or general ion-momentum unit conversion verified" in reason
               for reason in report["limitations"])
    assert len(report["clock_shift_cutoff_examples"]) == 12
    for row in report["clock_shift_cutoff_examples"]:
        assert row["translated_producer_hours"] == (row["documented_solar_hours"] + 12) % 24
        expected = (row["shifted_clock_disturbed_cutoff_gv"]
                    - row["documented_clock_disturbed_cutoff_gv"]) / row[
                        "documented_clock_disturbed_cutoff_gv"]
        assert row["signed_relative_change_vs_documented_clock"] == expected
        assert "not_physical_error" in row["scope"]
    json.dumps(report, allow_nan=False)


def test_implementation_change_during_run_invalidates_report(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashes = iter(({"implementation": "before"}, {"implementation": "after"}))
    monkeypatch.setattr(runner, "_model_hashes", lambda: next(hashes))
    with pytest.raises(ValueError, match="changed during"):
        runner.build_report()


def test_source_change_after_validation_invalidates_report(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"changed_source": "bad"})
    with pytest.raises(ValueError, match="changed during"):
        runner.build_report()


def test_report_output_is_exclusive_and_rejects_nonfinite_before_creation(
    runner: ModuleType, tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    runner.write_report_exclusively({"keep": True}, output)
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively({"keep": False}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"keep": True}
    invalid = tmp_path / "nan.json"
    with pytest.raises(ValueError):
        runner.write_report_exclusively({"bad": math.nan}, invalid)
    assert not invalid.exists()


def test_cli_rejects_existing_output_before_work(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "exists.json"
    runner.write_report_exclusively({"keep": True}, output)

    def forbidden() -> Any:
        raise AssertionError("Existing output must be checked before source work")

    monkeypatch.setattr(runner, "build_report", forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main(["--output", str(output)])
    assert error.value.code == 2
