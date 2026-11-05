"""The experiment roster retains literal parameters without isotope inference."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from radar.validation.ion_mass_contract import CandidateIsotope, audit_ost_ion_mass_contract


PILOT = tuple(CandidateIsotope(z, a, symbol) for z, a, symbol in (
    (2, 4, "He"), (6, 12, "C"), (7, 14, "N"), (8, 16, "O"), (10, 20, "Ne"),
    (12, 24, "Mg"), (14, 28, "Si"), (16, 32, "S"), (18, 40, "Ar"), (20, 40, "Ca"),
    (24, 52, "Cr"), (25, 55, "Mn"), (26, 56, "Fe"), (27, 59, "Co"), (28, 58, "Ni"),
))


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/build_stopping_mass_manifest.py"
    spec = importlib.util.spec_from_file_location("stopping_mass_manifest_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({"schema_version": "radar.ion_stopping_candidate.raw.v1",
                                "ions": [asdict(ion) for ion in PILOT]}), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_positive_source_inventory_and_explicit_pilot(
    runner: ModuleType, candidate: Path,
) -> None:
    manifest = runner.build_manifest(candidate, expected_candidate_sha256=_sha(candidate))
    assert manifest["schema_version"] == "radar.stopping_mass_manifest.v1"
    assert len(manifest["projectiles"]) == 84
    assert len(manifest["source_rows"]) == 118
    assert len(manifest["declared_zero_sources"]) == 10
    assert manifest["source_counts"] == {
        "GCR": {"hze_table_rows": 91, "positive_hze": 84, "declared_zero": 7},
        "SEP": {"hze_table_rows": 27, "positive_hze": 24, "declared_zero": 3},
    }
    assert all(p["source_membership"][0] == "GCR" for p in manifest["projectiles"])
    assert sum("SEP" in p["source_membership"] for p in manifest["projectiles"]) == 24
    assert manifest["pilot_isotopes"] == [
        {"z": p.z, "symbol": p.symbol, "mass_parameter_u": p.mass_number} for p in PILOT
    ]
    for projectile in manifest["projectiles"]:
        assert set(projectile) == {"z", "symbol", "mass_parameter_u", "source_membership"}
    assert manifest["source_roster_inventory_complete"]
    for flag in ("normative_export_allowed", "physical_mass_policy_qualified",
                 "transport_tables_generated", "production_tables_replaced"):
        assert manifest[flag] is False


def test_fractional_parameters_unusual_printed_values_and_charge_ratio_survive(
    runner: ModuleType, candidate: Path,
) -> None:
    manifest = runner.build_manifest(candidate, expected_candidate_sha256=_sha(candidate))
    projectiles = {p["symbol"]: p for p in manifest["projectiles"]}
    for symbol, mass in (("Ne", 20.2), ("Fe", 55.8), ("Ni", 58.7),
                         ("Sm", 145.0), ("Pb", 200.4), ("Bi", 207.2)):
        assert projectiles[symbol]["mass_parameter_u"] == mass
    pilots = {p["symbol"]: p for p in manifest["pilot_isotopes"]}
    assert pilots["Ni"]["mass_parameter_u"] == 58
    assert pilots["Fe"]["mass_parameter_u"] == 56
    sep_fe = next(r for r in manifest["source_rows"] if r["source"] == "SEP" and r["z"] == 26)
    assert sep_fe["source_tabulated_mass_parameter"] == 55.8
    assert sep_fe["source_mass_to_charge"] == 4.12
    assert projectiles["Li"]["source_membership"] == ["GCR"]
    assert not {84, 85, 86, 87, 88, 89, 91} & {p["z"] for p in manifest["projectiles"]}


def test_candidate_and_all_declared_source_hashes_are_pinned(
    runner: ModuleType, candidate: Path,
) -> None:
    manifest = runner.build_manifest(candidate, expected_candidate_sha256=_sha(candidate))
    assert manifest["candidate_sha256_before"] == manifest["candidate_sha256_after"] == _sha(candidate)
    assert manifest["candidate_expected_sha256"] == _sha(candidate)
    assert manifest["candidate_path"] == str(candidate.resolve())
    assert manifest["source_files_sha256_before"] == manifest["source_files_sha256_after"]
    assert len(manifest["source_files_sha256_before"]) == 6
    assert "stopping_rows_not_validated" in manifest["candidate_payload_scope"]


@pytest.mark.parametrize("pin", ["0" * 64, "bad", "A" * 64, None, True])
def test_bad_or_wrong_pin_is_rejected(runner: ModuleType, candidate: Path, pin: Any) -> None:
    with pytest.raises(ValueError):
        runner.build_manifest(candidate, expected_candidate_sha256=pin)


@pytest.mark.parametrize("payload", [
    '{"schema_version":"radar.ion_stopping_candidate.raw.v1","ions":[],"ions":[]}',
    '{"x":NaN}', '{"x":1e999}', '{}', '[]',
    '{"schema_version":"radar.ion_stopping_candidate.raw.v2","ions":[]}',
    '{"schema_version":"radar.ion_stopping_candidate.raw.v1","ions":[]}',
    '{"schema_version":"radar.ion_stopping_candidate.raw.v1","ions":[{"z":2}]}',
])
def test_malformed_json_inventory_rejected(runner: ModuleType, candidate: Path, payload: str) -> None:
    candidate.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        runner.build_manifest(candidate, expected_candidate_sha256=_sha(candidate))


@pytest.mark.parametrize("ions", [
    [{"z": 2, "mass_number": 4.0, "symbol": "He"}],
    [{"z": True, "mass_number": 4, "symbol": "He"}],
    [{"z": 2, "mass_number": 4, "symbol": "he"}],
    [{"z": 2, "mass_number": 4, "symbol": "He", "extra": "no"}],
    [{"z": 2, "mass_number": 4, "symbol": "He"}] * 2,
    [{"z": 1, "mass_number": 1, "symbol": "H"}],
    [{"z": 84, "mass_number": 209, "symbol": "Po"}],
])
def test_implicit_duplicate_or_outside_positive_pilot_rejected(
    runner: ModuleType, candidate: Path, ions: Any,
) -> None:
    candidate.write_text(json.dumps({"schema_version": "radar.ion_stopping_candidate.raw.v1",
                                    "ions": ions}), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.build_manifest(candidate, expected_candidate_sha256=_sha(candidate))


def test_literal_mass_conflict_does_not_get_averaged_or_rounded(runner: ModuleType) -> None:
    rows = audit_ost_ion_mass_contract(PILOT).rows
    changed = tuple(replace(row, source_tabulated_mass_parameter=56.0)
                    if row.source == "SEP" and row.z == 26 else row for row in rows)
    with pytest.raises(ValueError, match="Conflicting"):
        runner._source_inventory(changed)


@pytest.mark.parametrize("change", ["drop", "duplicate", "missing_positive", "nonfinite"])
def test_incomplete_or_invalid_source_inventory_is_rejected(runner: ModuleType, change: str) -> None:
    rows = audit_ost_ion_mass_contract(PILOT).rows
    if change == "drop":
        rows = rows[:-1]
    elif change == "duplicate":
        rows = (*rows, rows[0])
    elif change == "missing_positive":
        rows = tuple(replace(row, source_weight_parameter=0, nonzero_source_components=())
                     if row.source == "GCR" and row.z == 29 else row for row in rows)
    else:
        rows = (replace(rows[0], source_tabulated_mass_parameter=float("nan")), *rows[1:])
    with pytest.raises(ValueError):
        runner._source_inventory(rows)


def test_input_and_source_mutation_detected(
    runner: ModuleType, candidate: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runner.audit_ost_ion_mass_contract

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        candidate.write_bytes(candidate.read_bytes() + b" ")
        return result

    monkeypatch.setattr(runner, "audit_ost_ion_mass_contract", mutate)
    with pytest.raises(ValueError, match="changed"):
        runner.build_manifest(candidate, expected_candidate_sha256=_sha(candidate))
    monkeypatch.setattr(runner, "audit_ost_ion_mass_contract", original)
    hashes = iter(({"source": "a"}, {"source": "b"}))
    monkeypatch.setattr(runner, "_source_hashes", lambda: next(hashes))
    with pytest.raises(ValueError, match="changed"):
        runner.build_manifest(candidate, expected_candidate_sha256=_sha(candidate))


def test_cli_writes_once_and_refuses_overwrite(runner: ModuleType, candidate: Path) -> None:
    output = candidate.parent / "manifest.json"
    args = ["--candidate", str(candidate), "--candidate-sha256", _sha(candidate),
            "--output", str(output)]
    assert runner.main(args) == 0
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        runner.main(args)
    assert output.read_bytes() == before
    with pytest.raises(FileExistsError):
        runner.write_manifest_exclusively({}, output)
    assert output.read_bytes() == before
