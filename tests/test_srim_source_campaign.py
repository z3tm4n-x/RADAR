"""Synthetic orchestration checks only: no executable or physical model is run."""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, localcontext
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from radar.validation.srim_source_catalog import (
    SourceControl, build_source_catalog, expected_control_grid, pilot_controls,
)


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/run_srim_source_campaign.py"
    spec = importlib.util.spec_from_file_location("srim_source_campaign_test_subject", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def catalog() -> tuple[SourceControl, ...]:
    return build_source_catalog(Path(__file__).resolve().parents[1])


def _text(case: SourceControl) -> bytes:
    fixture = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    lines = [line.strip() for line in fixture.read_bytes().decode("cp1251").splitlines()
             if line.strip()]
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC NOT PHYSICAL DATA"
    header[5] = "Disk File Name = srim_output.txt"
    header[6] = header[17] = (
        f"Ion = Synthetic [{case.ion_z}] , Mass = {case.ion_mass_u} amu")
    header[7] = f"Density = {case.density_g_cm3} g/cm3 = 6.0000E+22 atoms/cm3"
    header[12] = f"{'Al' if case.target_z == 13 else 'Si'} {case.target_z} 100.00 100.00"
    rows = [f"{e} MeV 0.2 0.001 {i} um 1 A 1 A"
            for i, e in enumerate(expected_control_grid(case), 1)]
    footer = lines[lines.index("-" * 59):]
    text = re.sub(r"(?<=\d),(?=\d)", ".", "\r\n".join(header + rows + footer[:-1]))
    return (text + "\r\n" + footer[-1] + "\r\n").encode("ascii")


@pytest.fixture
def isolated(runner: ModuleType, catalog: tuple[SourceControl, ...], tmp_path: Path,
             monkeypatch: pytest.MonkeyPatch) -> Any:
    root, source = tmp_path / "repo", tmp_path / "source"
    root.mkdir()
    source.mkdir()
    (root / "outputs").mkdir()
    for name in runner.SOURCE_HASHES:
        (source / name).write_bytes(f"SYNTHETIC NONEXECUTABLE {name}".encode())
    (source / "SR.IN").write_bytes(b"ORIGINAL INPUT MUST NOT CHANGE")
    runtime = tmp_path / "private.dll"
    runtime.write_bytes(b"SYNTHETIC NONEXECUTABLE DLL")
    protected = root / "protected.txt"
    protected.write_bytes(b"FROZEN PRODUCTION PLACEHOLDER")
    for name in runner.IMPLEMENTATION:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"SYNTHETIC IMPLEMENTATION PIN {name}".encode())
    parent = root / runner.PARENT_CANDIDATE
    parent.write_text(json.dumps({"protected_sha256_before": {
        "protected.txt": runner._sha(protected.read_bytes())},
        "protected_sha256_after": {
            "protected.txt": runner._sha(protected.read_bytes())}}), encoding="utf-8")
    baseline = root / runner.LEGACY_BASELINE
    baseline.parent.mkdir()
    baseline.write_text(json.dumps({"protected_sha256_before": {
        str(protected): runner._sha(protected.read_bytes())},
        "protected_sha256_after": {
            str(protected): runner._sha(protected.read_bytes())}}), encoding="utf-8")
    pilots = pilot_controls(catalog)
    # Ten cases retain the exact eight-case pilot plus two unexecuted cases.
    # The real 168-case catalog is checked independently below.
    selected = tuple(c for c in catalog if c.ion_z in (2, 9, 10, 26, 92))
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "PARENT_SHA256", runner._sha(parent.read_bytes()))
    monkeypatch.setattr(runner, "LEGACY_BASELINE_SHA256", runner._sha(baseline.read_bytes()))
    monkeypatch.setattr(runner, "SOURCE_HASHES", {
        name: runner._sha((source / name).read_bytes()) for name in runner.SOURCE_HASHES})
    monkeypatch.setattr(runner, "PRIVATE_RUNTIME_SHA256", runner._sha(runtime.read_bytes()))
    monkeypatch.setattr(runner, "build_source_catalog", lambda _: selected)
    monkeypatch.setattr(runner, "pilot_controls", lambda _: pilots)
    calls: list[Path] = []

    def execute(work: Path) -> dict[str, Any]:
        calls.append(work)
        name = work.name.rsplit("_r", 1)[0]
        case = next(c for c in selected if c.name == name)
        (work / "srim_output.txt").write_bytes(_text(case))
        command = {"argv": [str(work / "SRModule.exe")], "cwd": str(work),
                   "pid": 12345 + len(calls), "returncode": 0, "timed_out": False,
                   "elapsed_seconds": 0.001, "stdout": "", "stderr": ""}
        (work / "command.json").write_text(json.dumps(command), encoding="utf-8")
        return command

    def run(mode: str = "pilot", name: str | None = None,
            pilot: Path | None = None, sha: str | None = None) -> dict[str, Any]:
        return cast(dict[str, Any], runner.run_campaign(
            mode=mode, source=source, private_runtime=runtime,
            output=root / "outputs" / (name or mode), pilot_report=pilot, pilot_sha256=sha))

    monkeypatch.setattr(runner, "_execute", execute)
    return SimpleNamespace(root=root, source=source, runtime=runtime, protected=protected,
                           parent=parent, baseline=baseline, selected=selected, pilots=pilots,
                           execute=execute, calls=calls, run=run)


def test_fixed_catalog_and_pilot_contract(catalog: tuple[SourceControl, ...]) -> None:
    assert len(catalog) == 168
    assert len({c.ion_z for c in catalog}) == 84
    assert {c.target_z for c in catalog} == {13, 14}
    pilots = pilot_controls(catalog)
    assert [(c.ion_z, c.ion_mass_u, c.target_z) for c in pilots] == [
        (z, Decimal(mass), target)
        for z, mass in ((9, "19"), (10, "20.2"), (26, "55.8"), (92, "238"))
        for target in (13, 14)]
    assert all(len(expected_control_grid(c)) == 107 for c in catalog)
    assert all(not c.normative_export_allowed and not c.physical_mass_policy_qualified
               and not c.source_energy_domain_covered for c in catalog)


def test_pilot_then_full_reuses_eight_without_reexecution(isolated: Any, runner: ModuleType) -> None:
    pilot = isolated.run()
    assert len(isolated.calls) == pilot["executed_here_attempts"] == 16
    assert pilot["case_count"] == 8 and pilot["all_native_controls_repeated"] is True
    path = isolated.root / "outputs/pilot/report.json"
    full = isolated.run("full", pilot=path, sha=runner._sha(path.read_bytes()))
    assert len(isolated.calls) == 20
    assert full["case_count"] == 10
    assert full["executed_here_attempts"] == 4
    assert full["reused_case_count"] == 8
    assert all(w.name.startswith("he4_") for w in isolated.calls[16:])
    assert full["expected_catalog"] == [asdict(c) for c in isolated.selected]
    assert full["protected_sha256_before"] == full["protected_sha256_after"]
    assert full["diagnostic_status"] == "INCOMPLETE"
    for key in ("normative_export_allowed", "production_tables_replaced", "isotope_policy_qualified",
                "source_energy_domain_covered", "physical_accuracy_verified"):
        assert full[key] is False
    for record in full["cases"]:
        assert len(record["repeats"]) == 2
        assert record["complete_native_control_grid_verified"] is True
        assert record["semantic_outputs_identical"] is True
        assert len(record["expected_control_grid_mev"]) == 107
        assert record["physical_accuracy_verified"] is False


@pytest.mark.parametrize("mode,pilot,sha", [
    ("unknown", None, None), ("full", None, None), ("full", Path("missing"), None),
    ("full", None, "digest"), ("pilot", Path("missing"), None), ("pilot", None, "digest"),
])
def test_mode_and_pilot_argument_guards(isolated: Any, mode: str, pilot: Path | None,
                                       sha: str | None) -> None:
    with pytest.raises(ValueError):
        isolated.run(mode, pilot=pilot, sha=sha)
    assert isolated.calls == []


@pytest.mark.parametrize("kind", ["root", "outside", "existing", "traversal"])
def test_output_must_be_new_strict_outputs_child(isolated: Any, runner: ModuleType,
                                                kind: str) -> None:
    base = isolated.root / "outputs"
    output = {"root": base, "outside": isolated.root / "outside",
              "existing": isolated.parent, "traversal": base / ".." / "escape"}[kind]
    with pytest.raises((ValueError, FileExistsError)):
        runner.run_campaign(mode="pilot", source=isolated.source,
                            private_runtime=isolated.runtime, output=output)
    assert isolated.calls == []


@pytest.mark.parametrize("filename", ["SRModule.exe", "SCOEF03.dat", "SNUC03.dat"])
def test_changed_native_source_rejected_before_launch(isolated: Any, filename: str) -> None:
    (isolated.source / filename).write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="Pinned file changed"):
        isolated.run()
    assert isolated.calls == []


def test_changed_private_runtime_rejected_before_launch(isolated: Any) -> None:
    isolated.runtime.write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="Pinned file changed"):
        isolated.run()
    assert isolated.calls == []


@pytest.mark.parametrize("kind", ["before_after", "production"])
def test_parent_protection_is_checked(isolated: Any, runner: ModuleType,
                                    monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    if kind == "production":
        isolated.protected.write_bytes(b"CHANGED")
    else:
        report = json.loads(isolated.parent.read_bytes())
        report["protected_sha256_after"] = {}
        isolated.parent.write_text(json.dumps(report), encoding="utf-8")
        monkeypatch.setattr(runner, "PARENT_SHA256", runner._sha(isolated.parent.read_bytes()))
    with pytest.raises(ValueError):
        isolated.run()
    assert isolated.calls == []


@pytest.mark.parametrize("old,new", [
    (b"Mass = 20.2", b"Mass = 20"), (b"[10]", b"[11]"),
    (b"Al 13", b"Si 14"), (b"Density = 2.70", b"Density = 2.71"),
    (b"Correction = 0.00%", b"Correction = 1.00%"),
    (b"SRIM-2006", b"SRIM-2007"), (b"SRIM-2012.01", b"SRIM-2013.01"),
    (b"srim_output.txt", b"other.txt"),
    (b"12000.0 MeV 0.2 0.001 107 um 1 A 1 A\r\n", b""),
])
def test_native_metadata_and_full_grid_are_exact(runner: ModuleType,
                                               catalog: tuple[SourceControl, ...],
                                               old: bytes, new: bytes) -> None:
    case = next(c for c in catalog if c.name == "ne20p2_al")
    raw = _text(case)
    assert old in raw
    parsed = runner.parse_srim_range_output(raw.replace(old, new))
    with pytest.raises(ValueError, match="metadata mismatch|energy grid mismatch"):
        runner._validate_native(parsed, case)


@pytest.mark.parametrize("failure", ["missing", "malformed", "wrong_mass", "grid"])
def test_invalid_outputs_are_explicit_and_block_full(isolated: Any, runner: ModuleType,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     failure: str) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        path = work / "srim_output.txt"
        if failure == "missing":
            path.unlink()
        elif failure == "malformed":
            path.write_bytes(b"NOT A NATIVE OUTPUT")
        elif failure == "wrong_mass":
            path.write_bytes(path.read_bytes().replace(b"Mass = ", b"Mass = 9"))
        else:
            path.write_bytes(path.read_bytes().replace(
                b"12000.0 MeV 0.2 0.001 107 um 1 A 1 A\r\n", b""))
        return command

    monkeypatch.setattr(runner, "_execute", execute)
    report = isolated.run()
    assert len(isolated.calls) == 8
    assert report["native_control_status_counts"]["INVALID_OUTPUT"] == 8
    assert report["all_native_controls_repeated"] is False
    for record in report["cases"]:
        assert len(record["repeats"]) == 1
        assert "parsed" not in record["repeats"][0]
        assert "semantic_sha256" not in record["repeats"][0]
    path = isolated.root / "outputs/pilot/report.json"
    with pytest.raises(ValueError, match="complete matching pilot"):
        isolated.run("full", pilot=path, sha=runner._sha(path.read_bytes()))
    assert len(isolated.calls) == 8
    assert not (isolated.root / "outputs/full").exists()


@pytest.mark.parametrize("timed_out", [False, True])
def test_retained_native_failure_is_checkpointed_without_numbers(
        isolated: Any, runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
        timed_out: bool) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        command.update(returncode=-1 if timed_out else 7, timed_out=timed_out)
        (work / "command.json").write_text(json.dumps(command), encoding="utf-8")
        raise RuntimeError("Synthetic timeout or native nonzero exit")

    monkeypatch.setattr(runner, "_execute", execute)
    report = isolated.run()
    assert report["native_control_status_counts"]["NATIVE_FAILED"] == 8
    assert len(isolated.calls) == 8
    for record in report["cases"]:
        assert record["checkpoint"]["bytes"] > 0
        assert len(record["repeats"]) == 1
        assert "parsed" not in record["repeats"][0]
        assert record["repeats"][0]["output"] is not None  # raw failure evidence retained


@pytest.mark.parametrize("failure", ["unexpected_runtime", "oserror", "command_difference",
                                     "argv", "pid", "elapsed", "fake_success"])
def test_launcher_or_ledger_errors_abort_without_final_report(
        isolated: Any, runner: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        if failure == "unexpected_runtime":
            raise RuntimeError("Launcher raised despite a successful ledger")
        if failure == "oserror":
            raise OSError("Synthetic launcher failure")
        if failure == "command_difference":
            return {**command, "pid": 54321}
        changes: dict[str, dict[str, Any]] = {
            "argv": {"argv": ["other.exe"]}, "pid": {"pid": 0},
            "elapsed": {"elapsed_seconds": float("nan")}, "fake_success": {"returncode": 4}}
        command.update(changes[failure])
        (work / "command.json").write_text(json.dumps(command), encoding="utf-8")
        return command

    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        isolated.run()
    assert len(isolated.calls) == 1
    assert not (isolated.root / "outputs/pilot/report.json").exists()


@pytest.mark.parametrize("mutation", ["source", "original_input", "private_runtime", "protected",
                                      "copied_exe", "copied_runtime", "copied_input"])
def test_mutations_are_detected_even_when_native_execution_fails(
        isolated: Any, runner: ModuleType, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        command["returncode"] = 7
        (work / "command.json").write_text(json.dumps(command), encoding="utf-8")
        path = {"source": isolated.source / "SCOEF03.dat",
                "original_input": isolated.source / "SR.IN", "private_runtime": isolated.runtime,
                "protected": isolated.protected, "copied_exe": work / "SRModule.exe",
                "copied_runtime": work / "MSVBVM50.dll", "copied_input": work / "SR.IN"}[mutation]
        path.write_bytes(b"CHANGED DURING SYNTHETIC FAILURE")
        raise RuntimeError("Synthetic native failure with a mutation")

    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="changed"):
        isolated.run()
    assert len(isolated.calls) == 1
    assert not (isolated.root / "outputs/pilot/report.json").exists()


def test_repeat_semantic_disagreement_stays_incomplete(isolated: Any, runner: ModuleType,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        if work.name.endswith("_r2"):
            path = work / "srim_output.txt"
            path.write_bytes(path.read_bytes().replace(b"0.2 0.001", b"0.3 0.001"))
        return command

    monkeypatch.setattr(runner, "_execute", execute)
    report = isolated.run()
    assert report["native_control_status_counts"]["REPEAT_MISMATCH"] == 8
    assert report["all_native_controls_repeated"] is False
    assert len(isolated.calls) == 16


@pytest.mark.parametrize("target", ["output", "command.json", "checkpoint", "copied_runtime"])
def test_final_validation_catches_mutation_of_an_earlier_case(
        isolated: Any, runner: ModuleType, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        if len(isolated.calls) == 16:
            first = isolated.calls[0]
            path = {"output": first / "srim_output.txt", "command.json": first / "command.json",
                    "checkpoint": first.parent / f"{isolated.pilots[0].name}.json",
                    "copied_runtime": first / "MSVBVM50.dll"}[target]
            path.write_bytes(b"MUTATED AFTER THE FIRST CASE COMPLETED")
        return command

    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="Pinned file changed"):
        isolated.run()
    assert len(isolated.calls) == 16
    assert not (isolated.root / "outputs/pilot/report.json").exists()


@pytest.mark.parametrize("tamper", ["checkpoint_content", "duplicate_repeat_path", "catalog",
                                    "record_order", "protected_pins", "digest"])
def test_full_rejects_repacked_or_changed_pilot_before_new_runs(
        isolated: Any, runner: ModuleType, tamper: str) -> None:
    isolated.run()
    path = isolated.root / "outputs/pilot/report.json"
    report = json.loads(path.read_bytes())
    record = report["cases"][0]
    checkpoint = Path(record["checkpoint"]["path"])
    if tamper == "checkpoint_content":
        stored = json.loads(checkpoint.read_bytes())
        stored["scope"]["physical_accuracy_verified"] = True
        checkpoint.write_text(json.dumps(stored), encoding="utf-8")
        record["checkpoint"] = runner._identity(checkpoint)
    elif tamper == "duplicate_repeat_path":
        for key in ("input", "output", "command", "command_file"):
            record["repeats"][1][key] = record["repeats"][0][key]
        checkpoint.write_text(runner._json({k: v for k, v in record.items()
                                            if k != "checkpoint"}), encoding="utf-8")
        record["checkpoint"] = runner._identity(checkpoint)
    elif tamper == "catalog":
        report["expected_catalog"][0]["ion_mass_u"] = "999"
    elif tamper == "record_order":
        report["cases"].reverse()
    elif tamper == "protected_pins":
        report["protected_sha256_before"].pop(str(isolated.protected))
        report["protected_sha256_after"].pop(str(isolated.protected))
    path.write_text(json.dumps(report), encoding="utf-8")
    digest = "0" * 64 if tamper == "digest" else runner._sha(path.read_bytes())
    with pytest.raises(ValueError):
        isolated.run("full", pilot=path, sha=digest)
    assert len(isolated.calls) == 16
    assert not (isolated.root / "outputs/full").exists()


def test_scope_does_not_depend_on_decimal_context(runner: ModuleType,
                                                 catalog: tuple[SourceControl, ...]) -> None:
    cases = tuple(c for c in catalog if c.ion_z in (2, 10, 92) and c.target_z == 13)
    expected = [runner._scope(case) for case in cases]
    with localcontext() as context:
        context.prec = 2
        context.Emax = 3
        context.Emin = -3
        observed = [runner._scope(case) for case in cases]
    assert observed == expected
    assert [item["package_trim_help_high_energy_caution"] for item in expected] == [True, False, False]
    assert all(item["within_package_table_claim"] for item in expected)
    assert all(item["physical_accuracy_verified"] is False for item in expected)
