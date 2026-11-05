"""The isolated GSI comparison preserves units, provenance and release limits."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from radar.validation.ion_stopping_candidate import IonStoppingRow


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/compare_ion_stopping_candidate.py"
    spec = importlib.util.spec_from_file_location("ion_stopping_comparison_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate(tmp_path: Path, runner: ModuleType) -> tuple[Path, dict[str, Any]]:
    rows = []
    for name, case in runner._CASES.items():
        text = (runner._ROOT / runner._EVIDENCE / name).read_text(encoding="utf-8")
        target = tmp_path / runner._EVIDENCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        ref = runner.parse_gsi_response(text, case)
        z, a, energy, material = case
        rows.append(asdict(IonStoppingRow(
            z, a, material, energy, ref["stopping_total_mev_cm2_g"] - 2,
            1, ref["range_g_cm2"], 1,
        )))
    raw = {
        "schema_version": "radar.ion_stopping_candidate.raw.v1", "rows": rows,
        "materials": [{"z": 13, "atomic_mass_u": 26.9815},
                      {"z": 14, "atomic_mass_u": 28.0855}],
        "backend_configuration": {"z_effective": 1, "corrections": 0},
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, raw


def test_exact_fixture_units_hashes_and_no_qualification(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
) -> None:
    path, _ = candidate
    report = runner.run_comparison(path, repository_root=path.parent)
    assert len(report["comparisons"]) == 3
    assert report["input_sha256_before"] == report["input_sha256_after"]
    assert len(report["input_sha256_before"]) == 4
    first = report["comparisons"][0]
    assert first["reference"]["stopping_total_mev_cm2_g"] == pytest.approx(3819.626)
    assert first["reference"]["range_g_cm2"] == pytest.approx(0.872005265)
    assert first["candidate_material"]["atomic_mass_u"] == 26.9815
    assert first["reference"]["target_atomic_mass_u"] == 26.981541
    assert first["reference"]["version_literal"] == "1.4"
    assert first["reference"]["form_version_label"] == "1.41"
    for comparison in report["comparisons"]:
        assert comparison["signed_relative_stopping_difference"] == 0
        assert comparison["signed_relative_range_difference"] == 0
    for flag in ("normative_release_eligible", "production_tables_replaced", "physics_validated"):
        assert report[flag] is False


@pytest.mark.parametrize("old,new", [
    ("Mass_p=56", "Mass_p=55"), ("Z=26 A=56.000000", "Z=26 A=55.000000"),
    ("target            : Z=13", "target            : Z=14"),
    ("MeV/(mg/cm^2)", "MeV/(g/cm^2)"), ("872.005265", "NaN"),
    ("3.819626", "0.000000"), ("Atima 1.4 <br>", "Atima 1.41 <br>"),
])
def test_rejects_mismatched_or_nonfinite_gsi_readback(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], old: str, new: str,
) -> None:
    path, _ = candidate
    html = path.parent / runner._EVIDENCE / "gsi_fe56_al_100.html"
    text = html.read_text(encoding="utf-8")
    assert old in text
    with pytest.raises(ValueError):
        runner.parse_gsi_response(text.replace(old, new), (26, 56, 100, 13))


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "overflow", "bool", "components"])
def test_rejects_invalid_candidate(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], defect: str,
) -> None:
    path, raw = candidate
    if defect == "missing":
        raw["rows"].pop()
    elif defect == "duplicate":
        raw["rows"].append(raw["rows"][0])
    elif defect == "nan":
        raw["backend_configuration"]["unknown"] = float("nan")
    elif defect == "overflow":
        raw["materials"][0]["atomic_mass_u"] = float("inf")
    elif defect == "bool":
        raw["rows"][0]["z"] = True
    else:
        del raw["rows"][0]["stopping_radiative_mev_cm2_g"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.run_comparison(path, repository_root=path.parent)


def test_duplicate_json_field_is_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]],
) -> None:
    path, _ = candidate
    text = path.read_text(encoding="utf-8")
    path.write_text('{"rows": [],' + text[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        runner.run_comparison(path, repository_root=path.parent)


def test_mutated_input_is_rejected(
    runner: ModuleType, candidate: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = candidate
    monkeypatch.setattr(runner, "_sha", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="changed"):
        runner.run_comparison(path, repository_root=path.parent)


def test_exclusive_write_and_nonfinite_output(runner: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    runner.write_report({"fixture": True}, output)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        runner.write_report({"fixture": False}, output)
    assert output.read_bytes() == before
    new_output = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        runner.write_report({"value": float("inf")}, new_output)
    assert not new_output.exists()


def test_cli_refuses_existing_output_before_reading_input(
    runner: ModuleType, tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        runner.main(["--candidate", str(tmp_path / "missing.json"), "--output", str(output)])
    assert output.read_bytes() == b"preserve"
