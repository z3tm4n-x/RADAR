"""Cross-run comparisons reject changed physics and retain bounded claims."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/compare_heo_mission_windows.py"
    spec = importlib.util.spec_from_file_location("mission_density_comparison_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(steps: int) -> dict[str, Any]:
    return {
        "schema": "radar_heo_finite_mission_window_diagnostic_v1",
        "normative_release_eligible": False, "mission_radiation_history_qualified": False,
        "source_products_sha256": "same", "source_products": ["fixed"],
        "source_scenario": {"lifetime_years": 5}, "source_mission_duration_s": 100.0,
        "model_sha256_before": {"model": "same"}, "model_sha256_after": {"model": "same"},
        "input_sha256_before": {"data": "same"}, "input_sha256_after": {"data": "same"},
        "rigidity_gv": [0.01, 1.0], "conventions": {"time_multiplier": "none"},
        "production_operator_version": "same",
        "variants": [{
            "name": f"n{steps}_phase{phase}",
            "requested": {"orbit": {"inclination": 63.4}, "duration_s": 100.0,
                          "raan_deg": phase, "mlt_sample_count": 24, "steps_per_orbit": steps},
            "exact_configured_mission_temporal_coverage": True,
            "contiguous_without_overlap": True, "global_phases_preserved": True,
            "coverage_start_time_s": 0.0, "coverage_end_time_s": 100.0,
            "represented_duration_s": 100.0, "step_s": 120.0 / steps,
            "penetration": [0.1, 0.8],
            "source_weighted_integrals": {str(i): {"value": float(i), "unit": "cm^-2",
                                                   "time_multiplier": "none"} for i in range(12)},
        } for phase in (0.0, 90.0)],
    }


def test_comparison_keeps_zero_reference_unknown_and_never_claims_acceptance(runner: ModuleType) -> None:
    left, right = _report(60), _report(120)
    right["variants"][0]["source_weighted_integrals"]["2"]["value"] = 2.01
    result = runner.compare_reports(left, right)
    metrics = result["comparisons"][0]["source_integral_changes"]
    assert metrics["0"]["relative_change"] is None
    assert metrics["0"]["relative_change_status"] == "zero_reference"
    assert metrics["2"]["relative_change"] == pytest.approx(0.005)
    assert not result["normative_release_eligible"]
    assert not result["stage4_spectral_convergence_established"]


@pytest.mark.parametrize("key", ["source_products_sha256", "source_products", "source_scenario",
                                  "model_sha256_before", "input_sha256_before", "rigidity_gv",
                                  "conventions", "production_operator_version"])
def test_changed_run_contract_is_refused(runner: ModuleType, key: str) -> None:
    right = _report(120)
    right[key] = "changed"
    with pytest.raises(ValueError):
        runner.compare_reports(_report(60), right)


@pytest.mark.parametrize("key", ["contiguous_without_overlap", "global_phases_preserved",
                                  "exact_configured_mission_temporal_coverage"])
def test_incomplete_temporal_coverage_is_refused(runner: ModuleType, key: str) -> None:
    right = _report(120)
    right["variants"][0][key] = False
    with pytest.raises(ValueError, match="full common interval"):
        runner.compare_reports(_report(60), right)


def test_changed_units_or_nonfinite_values_are_refused(runner: ModuleType) -> None:
    right = _report(120)
    right["variants"][0]["source_weighted_integrals"]["1"]["unit"] = "cm^-2 s^-1"
    with pytest.raises(ValueError, match="units"):
        runner.compare_reports(_report(60), right)
    right = _report(120)
    right["variants"][0]["source_weighted_integrals"]["1"]["value"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner.compare_reports(_report(60), right)


def test_wrong_density_or_changed_orientation_is_refused(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="double"):
        runner.compare_reports(_report(60), _report(240))
    right = _report(120)
    right["variants"][1] = deepcopy(right["variants"][0])
    with pytest.raises(ValueError, match="unique"):
        runner.compare_reports(_report(60), right)


@pytest.mark.parametrize("field,value", [
    ("represented_duration_s", float("nan")), ("step_s", float("inf")),
    ("step_s", True), ("represented_duration_s", 0.0),
])
def test_nonfinite_or_ambiguous_temporal_metadata_is_refused(
    runner: ModuleType, field: str, value: object,
) -> None:
    left, right = _report(60), _report(120)
    left["variants"][0][field] = value
    right["variants"][0][field] = value
    with pytest.raises(ValueError, match="finite and positive"):
        runner.compare_reports(left, right)


@pytest.mark.parametrize("steps", [True, 60.0, float("inf"), 0])
def test_requested_density_must_be_an_unambiguous_integer(
    runner: ModuleType, steps: object,
) -> None:
    left = _report(60)
    left["variants"][0]["requested"]["steps_per_orbit"] = steps
    with pytest.raises(ValueError, match="finite and positive"):
        runner.compare_reports(left, _report(120))


def test_cli_pins_inputs_and_refuses_overwrite(runner: ModuleType, tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("left.json", "right.json", "comparison.json")]
    for path, value in zip(paths[:2], (_report(60), _report(120)), strict=True):
        path.write_text(json.dumps(value), encoding="utf-8")
    arguments = ["--left", str(paths[0]), "--right", str(paths[1]), "--output", str(paths[2])]
    assert runner.main(arguments) == 0
    result = json.loads(paths[2].read_text(encoding="utf-8"))
    assert len(result["inputs"]) == 2
    assert len(result["comparison_implementation_sha256"]) == 64
    original = paths[2].read_bytes()
    with pytest.raises(SystemExit):
        runner.main(arguments)
    assert paths[2].read_bytes() == original
