"""Synthetic wrapper tests for the read-only source-energy ledger; no native launch."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/audit_srim_energy_support.py"
    spec = importlib.util.spec_from_file_location("srim_energy_runner_test_subject", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def native_replay(runner: ModuleType) -> ModuleType:
    # Importing the pinned reader never imports or executes the native launcher.
    return runner._load_replay()  # type: ignore[no-any-return]


@dataclass(frozen=True)
class _Control:
    name: str


@dataclass(frozen=True)
class _Ledger:
    observed: tuple[str, ...]
    lower: Decimal = Decimal("0.25")
    normative_export_allowed: bool = False


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def saved(runner: ModuleType, native_replay: ModuleType, tmp_path: Path,
          monkeypatch: pytest.MonkeyPatch) -> Any:
    root = tmp_path / "synthetic_repo"
    root.mkdir()
    for relative in (*runner.IMPLEMENTATION, runner.REPLAY_PATH):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"SYNTHETIC NONEXECUTABLE PIN {relative}".encode())
    replay_sha = _sha((root / runner.REPLAY_PATH).read_bytes())
    campaign_path = root / "campaign.json"
    campaign_path.write_bytes(b'{"synthetic": true}\n')
    campaign_sha = _sha(campaign_path.read_bytes())
    observations = ("observed_table_1", "observed_table_2")
    catalog = (_Control("synthetic1"), _Control("synthetic2"))
    calls: dict[str, Any] = {"catalog": [], "audit": [], "replay": []}
    campaign = {
        "mode": "full",
        "verified_native_control_count": 999,
        "energy_support": {"not_a_real_ledger": True},
        "native_control_status_counts": {"CONTROL_COMPLETE": 1, "NATIVE_FAILED": 1},
        "cases": [
            {"case": {"name": "complete"}, "status": "CONTROL_COMPLETE",
             "checkpoint": {"path": "retained complete"}},
            {"case": {"name": "failed"}, "status": "NATIVE_FAILED",
             "checkpoint": {"path": "retained failure"}},
        ],
    }

    def replay(path: Path, digest: str, controls: Any, evidence: Any) -> Any:
        calls["replay"].append((path, digest, controls))
        evidence.read(path, digest)
        return campaign, observations

    def load_catalog(selected_root: Path) -> Any:
        calls["catalog"].append(selected_root)
        return catalog

    def audit(selected: Any) -> _Ledger:
        calls["audit"].append(selected)
        return _Ledger(selected)

    fake_reader = SimpleNamespace(
        _Evidence=native_replay._Evidence, _same=native_replay._same,
        _json=native_replay._json, _replay=replay,
    )
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "REPLAY_SHA256", replay_sha)
    monkeypatch.setattr(runner, "_load_replay", lambda: fake_reader)
    monkeypatch.setattr(runner, "build_source_catalog", load_catalog)
    monkeypatch.setattr(runner, "audit_srim_energy_support", audit)
    return SimpleNamespace(root=root, path=campaign_path, sha=campaign_sha, catalog=catalog,
                           observations=observations, calls=calls, campaign=campaign,
                           reader=fake_reader, audit=audit)


def test_real_replay_is_the_frozen_read_only_module(runner: ModuleType,
                                                  native_replay: ModuleType) -> None:
    assert native_replay.__file__ is not None
    assert Path(native_replay.__file__).name == "summarize_srim_source_campaign.py"
    assert _sha(Path(native_replay.__file__).read_bytes()) == runner.REPLAY_SHA256
    assert callable(native_replay._replay)
    assert not hasattr(native_replay, "_execute")


def test_changed_replay_is_rejected_before_import(runner: ModuleType, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / runner.REPLAY_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b"raise AssertionError('THIS MUST NEVER RUN')")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Frozen native replay"):
        runner._load_replay()


@pytest.mark.parametrize("missing_loader", [False, True])
def test_missing_import_spec_or_loader_rejected(runner: ModuleType,
                                              monkeypatch: pytest.MonkeyPatch,
                                              missing_loader: bool) -> None:
    monkeypatch.setattr(runner.importlib.util, "spec_from_file_location",
                        lambda *_: SimpleNamespace(loader=None) if missing_loader else None)
    with pytest.raises(ValueError, match="Cannot load"):
        runner._load_replay()


def test_reconstructs_ledger_from_observations_and_preserves_failures(runner: ModuleType,
                                                                   saved: Any) -> None:
    result = runner.build_report(saved.path, expected_report_sha256=saved.sha)
    assert result["schema_version"] == "radar.srim_source_energy_support.v1"
    assert result["campaign_report"] == {"path": str(saved.path.resolve()), "sha256": saved.sha}
    assert result["campaign_mode"] == "full"
    assert result["verified_native_control_count"] == 2
    assert result["energy_support"] == {"observed": saved.observations, "lower": Decimal("0.25"),
                                        "normative_export_allowed": False}
    assert result["unavailable_controls"] == [saved.campaign["cases"][1]]
    assert result["native_control_status_counts"] == saved.campaign["native_control_status_counts"]
    assert result["pinned_replay_sha256"] == runner.REPLAY_SHA256
    assert saved.calls["audit"] == [saved.observations]
    assert saved.calls["replay"] == [(saved.path, saved.sha, saved.catalog)]
    assert saved.calls["catalog"] == [saved.root, saved.root]
    assert result["evidence_sha256_before"] == result["evidence_sha256_after"]
    assert len(result["implementation_sha256"]) == len(runner.IMPLEMENTATION)
    assert str(saved.path.resolve()) in result["evidence_sha256_before"]


@pytest.mark.parametrize("flag", [
    "normative_export_allowed", "production_tables_replaced", "source_grids_changed",
    "source_spectrum_convolution_performed", "native_execution_performed",
    "physical_accuracy_verified",
])
def test_output_cannot_claim_execution_convolution_or_qualification(runner: ModuleType,
                                                                 saved: Any, flag: str) -> None:
    saved.campaign[flag] = True
    result = runner.build_report(saved.path, expected_report_sha256=saved.sha)
    assert result[flag] is False
    assert result["diagnostic_status"] == "INCOMPLETE"


@pytest.mark.parametrize("status", ["NATIVE_FAILED", "INVALID_OUTPUT", "REPEAT_MISMATCH"])
def test_unavailable_status_is_retained_without_synthetic_table(runner: ModuleType,
                                                              saved: Any, status: str) -> None:
    saved.campaign["cases"][1]["status"] = status
    result = runner.build_report(saved.path, expected_report_sha256=saved.sha)
    assert result["unavailable_controls"][0]["status"] == status
    assert saved.calls["audit"] == [saved.observations]


@pytest.mark.parametrize("pin", ["", "NOT_A_HASH", "0" * 64])
def test_bad_campaign_pin_propagates_before_audit(runner: ModuleType, saved: Any,
                                                pin: str) -> None:
    with pytest.raises(ValueError, match="Pinned file changed|SHA256"):
        runner.build_report(saved.path, expected_report_sha256=pin)
    assert saved.calls["audit"] == []


@pytest.mark.parametrize("relative", [
    "scripts/audit_srim_energy_support.py",
    "src/radar/validation/srim_energy_support.py",
    "src/radar/validation/srim_native_coverage.py",
    "src/radar/validation/srim_source_catalog.py",
    "src/radar/validation/srim_output.py",
    "src/radar/core/constants.py", "src/radar/core/types.py", "src/radar/core/units.py",
    "scripts/summarize_srim_source_campaign.py", "campaign.json",
])
def test_file_mutation_during_audit_is_detected(runner: ModuleType, saved: Any,
                                              monkeypatch: pytest.MonkeyPatch,
                                              relative: str) -> None:
    def audit(observations: Any) -> _Ledger:
        result = saved.audit(observations)
        with (saved.root / relative).open("ab") as stream:
            stream.write(b"CHANGED AFTER REPLAY")
        return result  # type: ignore[no-any-return]

    monkeypatch.setattr(runner, "audit_srim_energy_support", audit)
    with pytest.raises(ValueError, match="Pinned file changed|File changed"):
        runner.build_report(saved.path, expected_report_sha256=saved.sha)


def test_missing_implementation_dependency_is_rejected(runner: ModuleType, saved: Any) -> None:
    (saved.root / runner.IMPLEMENTATION[0]).unlink()
    with pytest.raises(FileNotFoundError):
        runner.build_report(saved.path, expected_report_sha256=saved.sha)
    assert saved.calls["audit"] == []


def test_catalog_reconstruction_must_agree_after_audit(runner: ModuleType, saved: Any,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter((saved.catalog, (_Control("changed source"),)))
    monkeypatch.setattr(runner, "build_source_catalog", lambda _: next(values))
    with pytest.raises(ValueError, match="source catalog after energy audit"):
        runner.build_report(saved.path, expected_report_sha256=saved.sha)


def test_audit_error_is_not_converted_to_empty_supported_ledger(runner: ModuleType, saved: Any,
                                                              monkeypatch: pytest.MonkeyPatch
                                                              ) -> None:
    def fail(_: Any) -> None:
        raise ValueError("synthetic mass disagreement")

    monkeypatch.setattr(runner, "audit_srim_energy_support", fail)
    with pytest.raises(ValueError, match="mass disagreement"):
        runner.build_report(saved.path, expected_report_sha256=saved.sha)


def _argv(runner: ModuleType, saved: Any, output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["audit_srim_energy_support.py", "--campaign-report",
                                     str(saved.path), "--report-sha256", saved.sha,
                                     "--output", str(output)])


def test_cli_writes_decimal_strings_to_new_file(runner: ModuleType, saved: Any,
                                              tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "result.json"
    _argv(runner, saved, output, monkeypatch)
    assert runner.main() == 0
    report = json.loads(output.read_bytes())
    assert report["energy_support"]["lower"] == "0.25"
    assert report["normative_export_allowed"] is False


def test_existing_cli_output_is_preserved_before_replay(runner: ModuleType, saved: Any,
                                                      tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "existing.json"
    output.write_bytes(b"PRESERVE USER OUTPUT")
    _argv(runner, saved, output, monkeypatch)
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == b"PRESERVE USER OUTPUT"
    assert saved.calls["replay"] == []


def test_concurrent_cli_writer_cannot_be_overwritten(runner: ModuleType, saved: Any,
                                                   tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "concurrent.json"
    _argv(runner, saved, output, monkeypatch)

    def build(*_: Any, **__: Any) -> dict[str, Any]:
        output.write_bytes(b"OTHER WRITER")
        return {"normative_export_allowed": False}

    monkeypatch.setattr(runner, "build_report", build)
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == b"OTHER WRITER"


def test_nonfinite_report_cannot_create_cli_output(runner: ModuleType, saved: Any,
                                                 tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "invalid.json"
    _argv(runner, saved, output, monkeypatch)
    monkeypatch.setattr(runner, "build_report", lambda *_args, **_kwargs: {"bad": float("nan")})
    with pytest.raises(ValueError, match="JSON compliant"):
        runner.main()
    assert not output.exists()
