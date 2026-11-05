"""Candidate replay rejects incomplete schemas, mutated files and unsafe paths."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from radar.validation.ion_stopping_candidate import IonStoppingRow


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/validate_ion_stopping_candidate.py"
    spec = importlib.util.spec_from_file_location("ion_stopping_candidate_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    resource = tmp_path / "resource.txt"
    resource.write_bytes(b"unchanged production resource")
    source_root = tmp_path / "research/ion_stopping_20261031"
    source_root.mkdir(parents=True)
    source = source_root / "source.txt"
    source.write_bytes(b"immutable source fixture")
    generator = tmp_path / "scripts/generate_catima_candidate.py"
    generator.parent.mkdir()
    generator.write_bytes(b"# inert test generator")
    raw = {
        "schema_version": "radar.ion_stopping_candidate.raw.v1",
        "energies_mev_per_nucleon": [1.0, 2.0, 4.0],
        "ions": [{"z": 2, "mass_number": 4, "symbol": "He"}],
        "material_atomic_numbers": [13, 14],
        "rows": [asdict(IonStoppingRow(2, 4, m, e, 1, 1, e, 2))
                 for m in (13, 14) for e in (1, 2, 4)],
        "source_provenance": {
            "source": "synthetic analytic fixture",
            "source_root": "research/ion_stopping_20261031",
            "hashes": {"source.txt": hashlib.sha256(source.read_bytes()).hexdigest()},
            "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        },
        "generation_checks": {"fixture": True},
        "production_resources_sha256": {
            "resource.txt": hashlib.sha256(resource.read_bytes()).hexdigest(),
        },
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, raw


def _write(path: Path, raw: dict[str, Any]) -> None:
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_analytic_replay_retains_evidence_but_not_normative_qualification(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
) -> None:
    path, raw = candidate
    report = runner.run_validator(path, repository_root=path.parent)
    assert report["validation"]["numerical_consistency_passed"]
    assert len(report["validation"]["intervals"]) == 4
    assert report["source_provenance"] == raw["source_provenance"]
    assert report["generation_checks"] == raw["generation_checks"]
    assert report["input_sha256_before"] == report["input_sha256_after"]
    assert report["production_resources_sha256_before"] == raw["production_resources_sha256"]
    assert report["source_files_sha256_before"] == report["source_files_sha256_after"]
    assert report["declared_source_file_hashes_verified"] is True
    for field in ("source_provenance_claims_independently_verified",
                  "generation_checks_independently_replayed", "normative_release_eligible",
                  "production_tables_replaced", "full_required_ion_coverage_established"):
        assert report[field] is False


@pytest.mark.parametrize("field", ["ions", "rows", "source_provenance",
                                  "production_resources_sha256", "generation_checks"])
def test_missing_required_fields(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], field: str,
) -> None:
    path, raw = candidate
    del raw[field]
    _write(path, raw)
    with pytest.raises(ValueError):
        runner.run_validator(path, repository_root=path.parent)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "future-v2"), ("ions", []), ("rows", []),
    ("source_provenance", {}), ("generation_checks", {}),
    ("production_resources_sha256", {}), ("material_atomic_numbers", [13]),
])
def test_invalid_schema_values(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], field: str, value: Any,
) -> None:
    path, raw = candidate
    raw[field] = value
    _write(path, raw)
    with pytest.raises(ValueError):
        runner.run_validator(path, repository_root=path.parent)


@pytest.mark.parametrize("bad_hash", [None, "abc", "0" * 64, "A" * 64, True])
def test_missing_invalid_or_mismatched_resource_hash(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], bad_hash: Any,
) -> None:
    path, raw = candidate
    raw["production_resources_sha256"]["resource.txt"] = bad_hash
    _write(path, raw)
    with pytest.raises(ValueError):
        runner.run_validator(path, repository_root=path.parent)


@pytest.mark.parametrize("resource_path", ["../resource.txt", "/resource.txt", "C:/file",
                                         "\\\\server\\file", "resource.txt:stream", "."])
def test_escaping_or_nonrelative_resource_paths(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], resource_path: str,
) -> None:
    path, raw = candidate
    raw["production_resources_sha256"] = {resource_path: "0" * 64}
    _write(path, raw)
    with pytest.raises(ValueError):
        runner.run_validator(path, repository_root=path.parent)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metadata_is_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], value: float,
) -> None:
    path, raw = candidate
    raw["source_provenance"]["hidden"] = {"value": value}
    _write(path, raw)
    with pytest.raises(ValueError):
        runner.run_validator(path, repository_root=path.parent)


def test_overflow_literal_and_duplicate_json_key_are_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
) -> None:
    path, _ = candidate
    for text in ('{"x":1e999}', '{"x":1,"x":2}'):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            runner.run_validator(path, repository_root=path.parent)


@pytest.mark.parametrize("field", ["input", "resource", "source", "generator"])
def test_concurrent_mutation_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    path, _ = candidate
    original = runner.validate_candidate_rows

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        targets = {"input": path, "resource": path.parent / "resource.txt",
                   "source": path.parent / "research/ion_stopping_20261031/source.txt",
                   "generator": path.parent / "scripts/generate_catima_candidate.py"}
        target = targets[field]
        target.write_bytes(target.read_bytes() + b" ")
        return result

    monkeypatch.setattr(runner, "validate_candidate_rows", mutate)
    with pytest.raises(ValueError, match="changed|mismatch"):
        runner.run_validator(path, repository_root=path.parent)


@pytest.mark.parametrize("field,value", [
    ("source_root", "../source"), ("source_root", "."), ("hashes", {}),
    ("hashes", {"../source.txt": "0" * 64}), ("generator_sha256", "0" * 64),
])
def test_invalid_source_provenance_paths_and_hashes(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], field: str, value: Any,
) -> None:
    path, raw = candidate
    raw["source_provenance"][field] = value
    _write(path, raw)
    with pytest.raises(ValueError):
        runner.run_validator(path, repository_root=path.parent)


def test_required_radiative_column_is_not_defaulted_for_raw_data(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
) -> None:
    path, raw = candidate
    del raw["rows"][0]["stopping_radiative_mev_cm2_g"]
    _write(path, raw)
    with pytest.raises(ValueError, match="schema"):
        runner.run_validator(path, repository_root=path.parent)


def test_unknown_row_fields_and_missing_isotope_are_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
) -> None:
    path, raw = candidate
    raw["rows"][0]["invented"] = 1
    _write(path, raw)
    with pytest.raises(ValueError, match="schema"):
        runner.run_validator(path, repository_root=path.parent)
    del raw["rows"][0]["invented"]
    raw["rows"][0]["mass_number"] = 5
    _write(path, raw)
    with pytest.raises(ValueError, match="isotope"):
        runner.run_validator(path, repository_root=path.parent)


def test_report_write_is_exclusive_and_finite(runner: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    runner.write_report_exclusively({"result": 1}, output)
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively({"result": 2}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"result": 1}
    with pytest.raises(ValueError):
        runner.write_report_exclusively({"result": float("nan")}, tmp_path / "nan.json")
    assert not (tmp_path / "nan.json").exists()


def test_cli_runs_and_refuses_existing_output_before_calculation(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, _ = candidate
    output = path.parent / "validation.json"
    monkeypatch.setattr(runner, "_ROOT", path.parent)
    assert runner.main([str(path), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["normative_release_eligible"] is False
    monkeypatch.setattr(runner, "run_validator", lambda *args, **kwargs: pytest.fail("ran twice"))
    with pytest.raises(SystemExit):
        runner.main([str(path), "--output", str(output)])
