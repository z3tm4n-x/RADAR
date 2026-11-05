"""Bounded runner checks; no full normative-resource benchmark is executed."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from test_proton_secondary_depth_crossings import _analytic_narrow_count, _kernel, _source, _stopping


@pytest.fixture
def benchmark() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_secondary_bin_quadrature.py"
    spec = spec_from_file_location("independent_screening_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_case(**kwargs: Any) -> dict[str, Any]:
    screened = kwargs["screening"]
    strategy = kwargs["strategy"]
    reuse = kwargs.get("reuse_birth_partition", False)
    return {
        "strategy": strategy,
        "prequadrature_screening": screened,
        "reuse_birth_survival_partition": reuse,
        "algorithm_label": (
            "v9_reused_birth_survival_partition" if reuse
            else "v8_screened_active_cells" if screened
            else "v7_unscreened_active_cells" if strategy == "per_output_bin_active_cells"
            else strategy
        ),
        "incident_options": {
            "relative_tolerance": 1e-3,
            "prequadrature_screening": screened,
            "reuse_birth_survival_partition": reuse,
        },
        "thickness_g_cm2": kwargs["thickness"],
        "calculation_seconds_including_internal_preparation": 1.0 if screened else 2.0,
        "output_bin_edges_mev": (1.0, 2.0, 3.0),
        "output_counts_by_bin": (0.0, 5.0),
        "quantity": "differential_fluence",
        "y_unit": "synthetic_unit",
        "completeness": "incomplete",
        "incomplete_reasons": ("synthetic missing physical contribution",),
        "parameters": {
            "incident_quadrature_status": "incomplete",
            "incident_quadrature_supported_status": "converged",
            "incident_quadrature_estimated_absolute_error": "unavailable",
            "incident_quadrature_supported_estimated_absolute_error": "0",
        },
    }


def test_existing_defaults_remain_unscreened_stdout_with_original_strategies(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    defaults = benchmark._parser().parse_args([])
    assert defaults.strategy == "both"
    assert defaults.prequadrature_screening is False
    assert defaults.compare_screening is False
    assert defaults.reuse_birth_survival_partition is False
    assert defaults.compare_birth_partition is False
    assert defaults.output is None
    calls = []

    def fake_case(**kwargs: Any) -> dict[str, Any]:
        calls.append((kwargs["strategy"], kwargs["screening"], kwargs["reuse_birth_partition"]))
        return _fake_case(**kwargs)

    monkeypatch.setattr(benchmark, "_case", fake_case)
    monkeypatch.setattr(benchmark, "_input_hashes", lambda: {"input": "same"})
    benchmark.main(["--thickness", "0.01"])
    report = json.loads(capsys.readouterr().out)
    assert calls == [("global_union", False, False), ("per_output_bin", False, False)]
    assert report["input_hashes_consistent"] is True
    assert report["comparisons"][0]["global_over_per_bin_wall_time_ratio"] == 1.0
    assert report["comparisons"][0]["both_outer_integrals_converged"] is False
    assert report["normative_certification"] is False


@pytest.mark.parametrize("flag", ["--prequadrature-screening", "--compare-screening"])
@pytest.mark.parametrize("strategy", ["both", "global_union", "per_output_bin"])
def test_screening_flags_reject_other_strategies_before_any_work(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, flag: str, strategy: str,
) -> None:
    def forbidden_hash() -> dict[str, str]:
        pytest.fail("Invalid options must be rejected before reading benchmark inputs.")

    monkeypatch.setattr(benchmark, "_input_hashes", forbidden_hash)
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--strategy", strategy, flag])
    assert exc.value.code == 2


@pytest.mark.parametrize("screened", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_case_passes_screening_to_preflight_and_real_synthetic_calculation(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, screened: bool, reuse: bool,
) -> None:
    source = _source()
    monkeypatch.setattr(benchmark, "_SOURCE_X", source.x)
    monkeypatch.setattr(benchmark, "_SOURCE_Y", source.y)
    monkeypatch.setattr(benchmark, "load_normative_proton_al_range_table", _stopping)
    monkeypatch.setattr(benchmark, "load_normative_secondary_proton_kernel", _kernel)
    original_partition = benchmark.secondary._prepare_incident_output_partition
    forwarded_sources = []

    def counted_partition(**kwargs: Any) -> Any:
        forwarded_sources.append(kwargs["screening_source"])
        return original_partition(**kwargs)

    monkeypatch.setattr(benchmark.secondary, "_prepare_incident_output_partition", counted_partition)
    args = benchmark._parser().parse_args(
        ["--strategy", "per_output_bin_active_cells"]
        + (["--prequadrature-screening"] if screened else [])
        + (["--reuse-birth-survival-partition"] if reuse else [])
    )
    case = benchmark._case(thickness=3.0, strategy=args.strategy, args=args)
    assert len(forwarded_sources) == 2 * len(source.x)  # preflight and internal preparation
    assert all((item is not None) == screened for item in forwarded_sources)
    assert case["incident_options"]["prequadrature_screening"] is screened
    assert case["incident_options"]["reuse_birth_survival_partition"] is reuse
    assert case["prequadrature_screening"] is screened
    assert case["reuse_birth_survival_partition"] is reuse
    assert case["algorithm_label"] == (
        "v9_reused_birth_survival_partition" if reuse
        else "v8_screened_active_cells" if screened else "v7_unscreened_active_cells"
    )
    assert case["transformation_version"].startswith("9_" if reuse else "8_" if screened else "7_")
    assert case["parameters"]["prequadrature_screening"] == ("enabled" if screened else "disabled")
    assert case["total_count"] == pytest.approx(_analytic_narrow_count(), rel=1e-7)
    assert case["all_crossing_partitions_included"] is True
    assert case["crossing_partition_scope"] == (
        "evaluable_screened_intervals_only" if screened else "all_positive_source_intervals"
    )


def test_pair_comparison_writes_complete_json_with_hashes_and_retained_statuses(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events = []

    def fake_hashes() -> dict[str, str]:
        events.append("hash")
        return {"synthetic-input": "unchanged"}

    def fake_case(**kwargs: Any) -> dict[str, Any]:
        events.append((kwargs["thickness"], kwargs["screening"]))
        return _fake_case(**kwargs)

    monkeypatch.setattr(benchmark, "_input_hashes", fake_hashes)
    monkeypatch.setattr(benchmark, "_case", fake_case)
    output = tmp_path / "nested" / "comparison.json"
    benchmark.main([
        "--strategy", "per_output_bin_active_cells", "--compare-screening",
        "--thickness", "0.01", "0.1", "--output", str(output),
    ])
    assert capsys.readouterr().out == ""
    report = json.loads(output.read_text(encoding="utf-8"))
    assert events == ["hash", (0.01, False), (0.01, True), (0.1, False), (0.1, True), "hash"]
    assert report["file_sha256_at_start"] == report["file_sha256_at_end"]
    assert report["file_sha256"] == report["file_sha256_at_start"]
    assert report["input_hashes_consistent"] is True
    assert [item["comparison_role"] for item in report["cases"]] == [
        "reference", "candidate", "reference", "candidate",
    ]
    for comparison in report["comparisons"]:
        assert comparison["reference_label"] == "v7_unscreened_active_cells"
        assert comparison["candidate_label"] == "v8_screened_active_cells"
        assert comparison["absolute_count_difference_by_bin"] == [0.0, 0.0]
        assert comparison["relative_count_difference_by_nonzero_reference_bin"] == [None, 0.0]
        assert comparison["both_outer_integrals_converged"] is False
        assert comparison["both_retained_outer_integrals_converged"] is True
        assert comparison["normative_certification"] is False
        assert "global_over_per_bin_wall_time_ratio" not in comparison


def test_input_hash_drift_does_not_publish_a_report(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hashes = iter(({"production.py": "before"}, {"production.py": "after"}))
    monkeypatch.setattr(benchmark, "_input_hashes", lambda: next(hashes))
    monkeypatch.setattr(benchmark, "_case", _fake_case)
    output = tmp_path / "changed-inputs.json"
    with pytest.raises(RuntimeError, match="production.py"):
        benchmark.main(["--thickness", "0.01", "--output", str(output)])
    assert not output.exists()
    assert capsys.readouterr().out == ""


def test_existing_report_is_not_overwritten(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve this report", encoding="utf-8")

    def forbidden_hashes() -> dict[str, str]:
        pytest.fail("Existing output must be rejected before running the benchmark.")

    monkeypatch.setattr(benchmark, "_input_hashes", forbidden_hashes)
    with pytest.raises(SystemExit):
        benchmark.main(["--output", str(output)])
    assert output.read_text(encoding="utf-8") == "preserve this report"


@pytest.mark.parametrize("flag", ["--reuse-birth-survival-partition", "--compare-birth-partition"])
def test_birth_partition_options_reject_combined_screening_comparison_before_work(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, flag: str,
) -> None:
    def forbidden_hash() -> dict[str, str]:
        pytest.fail("Mixed comparison axes must be rejected before reading inputs.")

    monkeypatch.setattr(benchmark, "_input_hashes", forbidden_hash)
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--strategy", "per_output_bin_active_cells", "--compare-screening", flag])
    assert exc.value.code == 2


@pytest.mark.parametrize("screened", [False, True])
def test_birth_partition_pair_changes_only_reuse_and_preserves_screening(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], screened: bool,
) -> None:
    calls = []

    def fake_case(**kwargs: Any) -> dict[str, Any]:
        calls.append((kwargs["strategy"], kwargs["screening"], kwargs["reuse_birth_partition"]))
        return _fake_case(**kwargs)

    monkeypatch.setattr(benchmark, "_case", fake_case)
    monkeypatch.setattr(benchmark, "_input_hashes", lambda: {"input": "same"})
    benchmark.main([
        "--strategy", "per_output_bin_active_cells", "--compare-birth-partition",
        "--thickness", "0.0001",
    ] + (["--prequadrature-screening"] if screened else []))
    report = json.loads(capsys.readouterr().out)
    assert calls == [
        ("per_output_bin_active_cells", screened, False),
        ("per_output_bin_active_cells", screened, True),
    ]
    assert [case["comparison_role"] for case in report["cases"]] == ["reference", "candidate"]
    comparison = report["comparisons"][0]
    assert comparison["comparison_kind"] == "birth_survival_partition_reuse"
    assert comparison["reference_label"] == (
        "v8_screened_active_cells" if screened else "v7_unscreened_active_cells"
    )
    assert comparison["candidate_label"] == "v9_reused_birth_survival_partition"
    assert comparison["count_vectors_exactly_equal"] is True
    assert comparison["normative_certification"] is False
    assert "global_over_per_bin_wall_time_ratio" not in comparison


def test_birth_partition_pair_keeps_each_original_strategy_separate(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(benchmark, "_case", _fake_case)
    monkeypatch.setattr(benchmark, "_input_hashes", lambda: {"input": "same"})
    benchmark.main(["--compare-birth-partition", "--thickness", "0.0001"])
    report = json.loads(capsys.readouterr().out)
    assert [(case["strategy"], case["reuse_birth_survival_partition"]) for case in report["cases"]] == [
        ("global_union", False), ("global_union", True),
        ("per_output_bin", False), ("per_output_bin", True),
    ]
    assert len(report["comparisons"]) == 2
    assert all(item["comparison_kind"] == "birth_survival_partition_reuse"
               for item in report["comparisons"])


@pytest.mark.parametrize("field,value", [
    ("prequadrature_screening", True), ("strategy", "global_union"),
    ("depth_steps", 2), ("reuse_birth_survival_partition", False),
])
def test_birth_partition_comparison_rejects_other_changed_configuration(
    benchmark: ModuleType, field: str, value: Any,
) -> None:
    reference = _fake_case(
        strategy="per_output_bin_active_cells", screening=False,
        reuse_birth_partition=False, thickness=0.0001,
    )
    candidate = _fake_case(
        strategy="per_output_bin_active_cells", screening=False,
        reuse_birth_partition=True, thickness=0.0001,
    )
    candidate[field] = value
    with pytest.raises(ValueError):
        benchmark._compare_cases(
            reference, candidate, screening_comparison=False, birth_partition_comparison=True,
        )


def test_birth_partition_comparison_rejects_changed_incident_tolerance(benchmark: ModuleType) -> None:
    reference = _fake_case(
        strategy="per_output_bin_active_cells", screening=True,
        reuse_birth_partition=False, thickness=0.0001,
    )
    candidate = _fake_case(
        strategy="per_output_bin_active_cells", screening=True,
        reuse_birth_partition=True, thickness=0.0001,
    )
    candidate["incident_options"]["relative_tolerance"] = 0.1
    with pytest.raises(ValueError, match="incident options"):
        benchmark._compare_cases(
            reference, candidate, screening_comparison=False, birth_partition_comparison=True,
        )


def test_input_hash_registry_includes_shared_transport_and_loading_dependencies(
    benchmark: ModuleType,
) -> None:
    assert len(benchmark._INPUT_FILES) == len(set(benchmark._INPUT_FILES))
    assert {
        "src/radar/shielding/proton_al.py",
        "src/radar/shielding/proton_al_survival.py",
        "src/radar/shielding/resources.py",
        "src/radar/nuclear/tendl.py",
        "src/radar/core/spectrum_ops.py",
        "src/radar/data/integrity.py",
        "src/radar/data/normative/integrity_manifest.json",
    }.issubset(benchmark._INPUT_FILES)
    assert all((benchmark._PROJECT_ROOT / name).is_file() for name in benchmark._INPUT_FILES)


def test_nonfinite_result_does_not_publish_json(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def nonfinite_case(**kwargs: Any) -> dict[str, Any]:
        return {**_fake_case(**kwargs), "diagnostic": float("nan")}

    monkeypatch.setattr(benchmark, "_input_hashes", lambda: {"input": "same"})
    monkeypatch.setattr(benchmark, "_case", nonfinite_case)
    output = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        benchmark.main([
            "--strategy", "per_output_bin_active_cells", "--reuse-birth-survival-partition",
            "--thickness", "0.0001", "--output", str(output),
        ])
    assert not output.exists()
