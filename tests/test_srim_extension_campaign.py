"""Synthetic bounded extension campaign checks; never execute SRIM."""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from radar.validation.srim_output import parse_srim_range_output
from radar.validation.srim_source_catalog import SourceControl, build_source_catalog, expected_control_grid
from radar.validation.srim_source_extension import (
    ExtensionControl, build_extension_catalog, expected_extension_grid, pilot_extension_controls,
)


def _load(name: str, relative: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return _load("srim_extension_campaign_test_subject", "scripts/run_srim_source_extension.py")


@pytest.fixture(scope="module")
def reader() -> ModuleType:
    return _load("srim_extension_frozen_reader", "scripts/summarize_srim_source_campaign.py")


@pytest.fixture(scope="module")
def catalog() -> tuple[ExtensionControl, ...]:
    return build_extension_catalog(Path(__file__).resolve().parents[1])


def _text(case: SourceControl | ExtensionControl) -> bytes:
    fixture = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    lines = [line.strip() for line in fixture.read_bytes().decode("cp1251").splitlines()
             if line.strip()]
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC NOT PHYSICAL DATA"
    header[5] = "Disk File Name = srim_output.txt"
    header[6] = header[17] = f"Ion = Synthetic [{case.ion_z}] , Mass = {case.ion_mass_u} amu"
    header[7] = f"Density = {case.density_g_cm3} g/cm3 = 6.0000E+22 atoms/cm3"
    header[12] = f"{'Al' if case.target_z == 13 else 'Si'} {case.target_z} 100.00 100.00"
    grid = (expected_control_grid(case) if isinstance(case, SourceControl)
            else expected_extension_grid(case))
    rows = [f"{e} MeV 0.2 0.001 {i} um 1 A 1 A" for i, e in enumerate(grid, 1)]
    footer = lines[lines.index("-" * 59):]
    text = re.sub(r"(?<=\d),(?=\d)", ".", "\r\n".join(header + rows + footer[:-1]))
    return (text + "\r\n" + footer[-1] + "\r\n").encode("ascii")


def test_fixed_catalog_nonshrinking_ceiling(catalog: tuple[ExtensionControl, ...]) -> None:
    assert len(catalog) == 158
    assert len({c.ion_z for c in catalog}) == 79
    assert {c.target_z for c in catalog} == {13, 14}
    assert all(c.energy_max_mev > 12000 and c.energy_max_mev <= 1000*c.ion_mass_u
               for c in catalog)
    assert not {2, 3, 4, 5, 6} & {c.ion_z for c in catalog}
    pilot = pilot_extension_controls(catalog)
    assert [(c.ion_z, c.target_z, c.energy_max_mev) for c in pilot] == [
        (z, target, Decimal(maximum))
        for z, maximum in ((9, 18000), (10, 20000), (26, 55000), (92, 225000))
        for target in (13, 14)]
    assert all(not c.normative_export_allowed and not c.physical_accuracy_verified
               for c in catalog)


@pytest.fixture
def isolated(runner: ModuleType, reader: ModuleType, catalog: tuple[ExtensionControl, ...],
             tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
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
    for name in (*runner.IMPLEMENTATION, runner.REPLAY):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"SYNTHETIC IMPLEMENTATION PIN {name}".encode())
    pins = {str(path.resolve()): runner._sha(path.read_bytes())
            for path in (protected, runtime, source / "SR.IN",
                         *(source / name for name in runner.SOURCE_HASHES))}
    parent = {"mode": "full", "all_native_controls_repeated": True,
              "source_directory": str(source), "private_runtime": runner._identity(runtime),
              "protected_sha256_before": pins.copy(), "protected_sha256_after": pins.copy()}
    baseline_path = root / runner.BASELINE
    baseline_path.parent.mkdir()
    baseline_path.write_text(json.dumps(parent), encoding="utf-8")
    baseline_sha = runner._sha(baseline_path.read_bytes())
    selected = tuple(c for c in catalog if c.ion_z in (7, 9, 10, 26, 92))
    pilots = pilot_extension_controls(catalog)
    bases = {c.name: c for c in build_source_catalog(Path(__file__).resolve().parents[1])
             if c.name in {e.base_source_name for e in selected}}
    native_baselines = {name: parse_srim_range_output(_text(case)) for name, case in bases.items()}
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "BASELINE_SHA256", baseline_sha)
    monkeypatch.setattr(runner, "REPLAY_SHA256", runner._sha((root / runner.REPLAY).read_bytes()))
    monkeypatch.setattr(runner, "SOURCE_HASHES", {
        name: runner._sha((source / name).read_bytes()) for name in runner.SOURCE_HASHES})
    monkeypatch.setattr(runner, "PRIVATE_RUNTIME_SHA256", runner._sha(runtime.read_bytes()))
    monkeypatch.setattr(runner, "build_extension_catalog", lambda _: selected)
    monkeypatch.setattr(runner, "pilot_extension_controls", lambda _: pilots)

    def context() -> Any:
        evidence = reader._Evidence()
        evidence.read(baseline_path, baseline_sha)
        for filename, sha in pins.items():
            evidence.read(Path(filename), sha)
        for name in (*runner.IMPLEMENTATION, runner.REPLAY):
            evidence.read(root / name)
        return reader, evidence, parent, native_baselines, selected

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
            mode=mode, output=root / "outputs" / (name or mode),
            pilot_report=pilot, pilot_sha256=sha))

    monkeypatch.setattr(runner, "_context", context)
    monkeypatch.setattr(runner, "_execute", execute)
    return SimpleNamespace(root=root, source=source, runtime=runtime, protected=protected,
                           parent=parent, baseline_path=baseline_path, selected=selected,
                           pilots=pilots, baselines=native_baselines, bases=bases,
                           context=context, execute=execute, calls=calls, run=run)


def test_pilot_then_full_reuses_exactly_eight(isolated: Any, runner: ModuleType) -> None:
    pilot = isolated.run()
    assert len(isolated.calls) == pilot["executed_here_attempts"] == 16
    assert pilot["case_count"] == 8 and pilot["all_native_controls_repeated"] is True
    path = isolated.root / "outputs/pilot/report.json"
    full = isolated.run("full", pilot=path, sha=runner._sha(path.read_bytes()))
    assert len(isolated.calls) == 20
    assert full["case_count"] == 10 and full["executed_here_attempts"] == 4
    assert full["reused_case_count"] == 8
    assert all(w.name.startswith("n14_") for w in isolated.calls[16:])
    assert full["expected_catalog"] == [asdict(c) for c in isolated.selected]
    assert full["protected_sha256_before"] == full["protected_sha256_after"]
    assert full["diagnostic_status"] == "INCOMPLETE"
    for key in ("normative_export_allowed", "production_tables_replaced", "isotope_policy_qualified",
                "source_energy_domain_covered", "physical_accuracy_verified"):
        assert full[key] is False
    for record in full["cases"]:
        assert len(record["repeats"]) == 2
        assert record["baseline_prefix_identical"] is True
        assert record["baseline_prefix_rows"] == 107
        assert record["semantic_outputs_identical"] is True
        assert len(record["expected_control_grid_mev"]) > 107


@pytest.mark.parametrize("mode,pilot,sha", [
    ("unknown", None, None), ("full", None, None), ("full", Path("missing"), None),
    ("full", None, "digest"), ("pilot", Path("missing"), None), ("pilot", None, "digest"),
])
def test_mode_and_pilot_arguments(isolated: Any, mode: str, pilot: Path | None,
                                  sha: str | None) -> None:
    with pytest.raises(ValueError):
        isolated.run(mode, pilot=pilot, sha=sha)
    assert isolated.calls == []


@pytest.mark.parametrize("kind", ["root", "outside", "existing", "traversal"])
def test_output_is_new_strict_outputs_child(isolated: Any, runner: ModuleType,
                                           kind: str) -> None:
    base = isolated.root / "outputs"
    output = {"root": base, "outside": isolated.root / "outside",
              "existing": isolated.baseline_path, "traversal": base / ".." / "escape"}[kind]
    with pytest.raises((ValueError, FileExistsError)):
        runner.run_campaign(mode="pilot", output=output)
    assert isolated.calls == []


@pytest.mark.parametrize("target", ["source", "private_runtime", "original_input", "protected"])
def test_changed_parent_inputs_fail_before_launch(isolated: Any, target: str) -> None:
    path = {"source": isolated.source / "SCOEF03.dat", "private_runtime": isolated.runtime,
            "original_input": isolated.source / "SR.IN", "protected": isolated.protected}[target]
    path.write_bytes(b"CHANGED")
    with pytest.raises(ValueError):
        isolated.run()
    assert isolated.calls == []


@pytest.mark.parametrize("column,replacement", [
    ("energy", b"1.01 MeV 0.2 0.001 1 um 1 A 1 A"),
    ("electronic", b"1 MeV 0.3 0.001 1 um 1 A 1 A"),
    ("nuclear", b"1 MeV 0.2 0.002 1 um 1 A 1 A"),
    ("range", b"1 MeV 0.2 0.001 1.1 um 1 A 1 A"),
    ("longitudinal", b"1 MeV 0.2 0.001 1 um 2 A 1 A"),
    ("lateral", b"1 MeV 0.2 0.001 1 um 1 A 2 A"),
])
def test_complete_six_column_prefix_is_required(isolated: Any, runner: ModuleType,
                                               column: str, replacement: bytes) -> None:
    case = isolated.pilots[0]
    raw = _text(case)
    assert b"1 MeV 0.2 0.001 1 um 1 A 1 A" in raw
    changed = parse_srim_range_output(raw.replace(b"1 MeV 0.2 0.001 1 um 1 A 1 A", replacement))
    with pytest.raises(ValueError, match="grid|baseline row"):
        runner._validate_native(changed, case, isolated.baselines[case.base_source_name])


@pytest.mark.parametrize("old,new", [
    (b"Mass = 20.2", b"Mass = 20"), (b"[10]", b"[11]"),
    (b"Al 13", b"Si 14"), (b"Density = 2.70", b"Density = 2.71"),
    (b"SRIM-2006", b"SRIM-2007"), (b"SRIM-2012.01", b"SRIM-2013.01"),
    (b"srim_output.txt", b"other.txt"),
])
def test_exact_native_metadata(isolated: Any, runner: ModuleType, old: bytes, new: bytes) -> None:
    case = next(c for c in isolated.pilots if c.ion_z == 10 and c.target_z == 13)
    raw = _text(case)
    assert old in raw
    parsed = parse_srim_range_output(raw.replace(old, new))
    with pytest.raises(ValueError, match="metadata"):
        runner._validate_native(parsed, case, isolated.baselines[case.base_source_name])


@pytest.mark.parametrize("failure", ["missing", "malformed", "prefix", "grid"])
def test_invalid_native_outputs_remain_explicit_and_block_full(
        isolated: Any, runner: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        path = work / "srim_output.txt"
        raw = path.read_bytes()
        if failure == "missing":
            path.unlink()
        elif failure == "malformed":
            path.write_bytes(b"NOT NATIVE OUTPUT")
        elif failure == "prefix":
            path.write_bytes(raw.replace(b"1 MeV 0.2 0.001 1 um", b"1 MeV 0.3 0.001 1 um"))
        else:
            lines = raw.splitlines(keepends=True)
            index = max(i for i, line in enumerate(lines) if b" MeV " in line)
            path.write_bytes(b"".join(lines[:index] + lines[index+1:]))
        return command

    monkeypatch.setattr(runner, "_execute", execute)
    report = isolated.run()
    assert len(isolated.calls) == 8
    assert report["native_control_status_counts"]["INVALID_OUTPUT"] == 8
    for record in report["cases"]:
        assert len(record["repeats"]) == 1
        assert "parsed" not in record["repeats"][0]
        assert "semantic_sha256" not in record["repeats"][0]
    path = isolated.root / "outputs/pilot/report.json"
    with pytest.raises(ValueError, match="Complete extension pilot"):
        isolated.run("full", pilot=path, sha=runner._sha(path.read_bytes()))
    assert len(isolated.calls) == 8
    assert not (isolated.root / "outputs/full").exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_native_failure_retains_raw_evidence_without_numbers(
        isolated: Any, runner: ModuleType, monkeypatch: pytest.MonkeyPatch, timeout: bool) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        command.update(returncode=-1 if timeout else 7, timed_out=timeout)
        (work / "command.json").write_text(json.dumps(command), encoding="utf-8")
        raise RuntimeError("Synthetic native failure")

    monkeypatch.setattr(runner, "_execute", execute)
    report = isolated.run()
    assert report["native_control_status_counts"]["NATIVE_FAILED"] == 8
    assert len(isolated.calls) == 8
    for record in report["cases"]:
        assert len(record["repeats"]) == 1
        assert record["repeats"][0]["output"] is not None
        assert "parsed" not in record["repeats"][0]


@pytest.mark.parametrize("mutation", ["source", "original_input", "private_runtime", "protected",
                                      "copied_exe", "copied_runtime", "copied_input"])
def test_mutation_detected_even_on_failure(isolated: Any, runner: ModuleType,
                                         monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        command["returncode"] = 7
        (work / "command.json").write_text(json.dumps(command), encoding="utf-8")
        path = {"source": isolated.source / "SCOEF03.dat", "original_input": isolated.source / "SR.IN",
                "private_runtime": isolated.runtime, "protected": isolated.protected,
                "copied_exe": work / "SRModule.exe", "copied_runtime": work / "MSVBVM50.dll",
                "copied_input": work / "SR.IN"}[mutation]
        path.write_bytes(b"MUTATION DURING FAILURE")
        raise RuntimeError("Synthetic failure")

    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="changed"):
        isolated.run()
    assert len(isolated.calls) == 1


def test_repeat_disagreement_does_not_select_extension(isolated: Any, runner: ModuleType,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        if work.name.endswith("_r2"):
            path = work / "srim_output.txt"
            lines = path.read_bytes().splitlines(keepends=True)
            index = max(i for i, line in enumerate(lines) if b" MeV " in line)
            lines[index] = lines[index].replace(b"0.2 0.001", b"0.3 0.001")
            path.write_bytes(b"".join(lines))
        return command

    monkeypatch.setattr(runner, "_execute", execute)
    report = isolated.run()
    assert report["native_control_status_counts"]["REPEAT_MISMATCH"] == 8
    path = isolated.root / "outputs/pilot/report.json"
    summary = runner.summarize_extension(path, runner._sha(path.read_bytes()))
    assert summary["extended_table_count"] == 0
    assert summary["retained_baseline_table_count"] == 10
    assert len(summary["unavailable_extensions"]) == 8


def test_full_failure_preserves_baseline_and_complete_pilot_extensions(
        isolated: Any, runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    isolated.run()
    pilot = isolated.root / "outputs/pilot/report.json"

    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        (work / "srim_output.txt").write_bytes(b"SYNTHETIC INVALID EXTENSION")
        return command

    monkeypatch.setattr(runner, "_execute", execute)
    report = isolated.run("full", pilot=pilot, sha=runner._sha(pilot.read_bytes()))
    assert report["native_control_status_counts"]["INVALID_OUTPUT"] == 2
    path = isolated.root / "outputs/full/report.json"
    count = len(isolated.calls)
    summary = runner.summarize_extension(path, runner._sha(path.read_bytes()))
    assert len(isolated.calls) == count
    assert summary["extended_table_count"] == 8
    assert summary["retained_baseline_table_count"] == 2
    assert summary["selected_native_table_count"] == 10
    assert len(summary["unavailable_extensions"]) == 2
    assert summary["native_execution_performed_by_summary"] is False
    assert summary["evidence_sha256_before"] == summary["evidence_sha256_after"]
    assert all(row["baseline_prefix_identical"] for row in summary["table_selection"])


@pytest.mark.parametrize("failure", ["unexpected_runtime", "oserror", "command_difference",
                                     "argv", "pid", "elapsed", "fake_success"])
def test_launcher_ledger_errors_abort(isolated: Any, runner: ModuleType,
                                     monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        if failure == "unexpected_runtime":
            raise RuntimeError("Synthetic exception with successful ledger")
        if failure == "oserror":
            raise OSError("Synthetic launcher error")
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


@pytest.mark.parametrize("target", ["output", "command", "checkpoint", "copied_runtime"])
def test_final_validation_rechecks_prior_case_artifacts(
        isolated: Any, runner: ModuleType, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    def execute(work: Path) -> dict[str, Any]:
        command: dict[str, Any] = isolated.execute(work)
        if len(isolated.calls) == 16:
            first = isolated.calls[0]
            path = {"output": first / "srim_output.txt", "command": first / "command.json",
                    "checkpoint": first.parent / f"{isolated.pilots[0].name}.json",
                    "copied_runtime": first / "MSVBVM50.dll"}[target]
            path.write_bytes(b"MUTATION AFTER CASE COMPLETION")
        return command

    monkeypatch.setattr(runner, "_execute", execute)
    with pytest.raises(ValueError, match="changed"):
        isolated.run()
    assert len(isolated.calls) == 16
    assert not (isolated.root / "outputs/pilot/report.json").exists()


@pytest.mark.parametrize("tamper", ["checkpoint", "repeat_path", "catalog", "order", "pins",
                                    "counter", "baseline_reference", "qualification", "digest"])
def test_repacked_pilot_rejected_before_more_native_attempts(
        isolated: Any, runner: ModuleType, tamper: str) -> None:
    isolated.run()
    path = isolated.root / "outputs/pilot/report.json"
    report = json.loads(path.read_bytes())
    record = report["cases"][0]
    checkpoint = Path(record["checkpoint"]["path"])
    if tamper == "checkpoint":
        stored = json.loads(checkpoint.read_bytes())
        stored["baseline_prefix_rows"] = 106
        checkpoint.write_text(json.dumps(stored), encoding="utf-8")
        record["checkpoint"] = runner._identity(checkpoint)
    elif tamper == "repeat_path":
        for key in ("input", "output", "command", "command_file"):
            record["repeats"][1][key] = record["repeats"][0][key]
        checkpoint.write_text(runner._json({k: v for k, v in record.items()
                                            if k != "checkpoint"}), encoding="utf-8")
        record["checkpoint"] = runner._identity(checkpoint)
    elif tamper == "catalog":
        report["expected_catalog"][0]["ion_mass_u"] = "999"
    elif tamper == "order":
        report["cases"].reverse()
    elif tamper == "pins":
        report["protected_sha256_before"].pop(str(isolated.protected))
        report["protected_sha256_after"].pop(str(isolated.protected))
    elif tamper == "counter":
        report["executed_here_attempts"] = 1
    elif tamper == "baseline_reference":
        report["baseline_reference"]["sha256"] = "0"*64
    elif tamper == "qualification":
        report["normative_export_allowed"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    digest = "0"*64 if tamper == "digest" else runner._sha(path.read_bytes())
    with pytest.raises(ValueError):
        isolated.run("full", pilot=path, sha=digest)
    assert len(isolated.calls) == 16
    assert not (isolated.root / "outputs/full").exists()


def test_context_rejects_changed_reader_before_import(runner: ModuleType, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / runner.REPLAY
    path.parent.mkdir(parents=True)
    path.write_bytes(b"raise AssertionError('MUST NOT EXECUTE')")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Frozen native replay"):
        runner._context()


@pytest.mark.parametrize("kind", ["missing", "loader"])
def test_context_requires_import_spec(runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                      kind: str) -> None:
    monkeypatch.setattr(runner.importlib.util, "spec_from_file_location",
                        lambda *_: None if kind == "missing" else SimpleNamespace(loader=None))
    with pytest.raises(ValueError, match="Cannot load"):
        runner._context()


@pytest.mark.parametrize("argv", [
    ["--mode", "full", "--report", "wrong"],
    ["--mode", "pilot", "--report-sha256", "wrong"],
    ["--mode", "replay"],
    ["--mode", "replay", "--report", "missing"],
    ["--mode", "replay", "--report-sha256", "wrong"],
    ["--mode", "replay", "--report", "missing", "--report-sha256", "wrong",
     "--pilot-report", "wrong"],
])
def test_cli_rejects_mixed_modes_without_execution(runner: ModuleType, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch,
                                                  argv: list[str]) -> None:
    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("No execution or summary allowed for invalid CLI arguments")
    monkeypatch.setattr(runner, "run_campaign", forbidden)
    monkeypatch.setattr(runner, "summarize_extension", forbidden)
    monkeypatch.setattr(sys, "argv", ["runner", *argv, "--output", str(tmp_path / "new")])
    with pytest.raises(ValueError):
        runner.main()


def test_replay_cli_is_exclusive_and_never_launches_native(
        runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def summarize(path: Path, sha: str) -> dict[str, Any]:
        calls.append((path, sha))
        return {"native_control_status_counts": {"CONTROL_COMPLETE": 8},
                "normative_export_allowed": False, "source": Decimal("20.2")}

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("Replay must not execute native campaign")
    output = tmp_path / "summary.json"
    monkeypatch.setattr(runner, "run_campaign", forbidden)
    monkeypatch.setattr(runner, "summarize_extension", summarize)
    monkeypatch.setattr(sys, "argv", ["runner", "--mode", "replay", "--report", "retained",
                                      "--report-sha256", "a"*64, "--output", str(output)])
    assert runner.main() == 0
    original = output.read_bytes()
    assert json.loads(original)["source"] == "20.2"
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        runner.main()
    assert len(calls) == 1 and output.read_bytes() == original


def test_replay_cli_does_not_overwrite_a_concurrent_output(
        runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "summary.json"

    def summarize(*_: Any) -> dict[str, Any]:
        output.write_bytes(b"CONCURRENT OWNER")
        return {"native_control_status_counts": {"CONTROL_COMPLETE": 8}}

    monkeypatch.setattr(runner, "summarize_extension", summarize)
    monkeypatch.setattr(sys, "argv", ["runner", "--mode", "replay", "--report", "retained",
                                      "--report-sha256", "a"*64, "--output", str(output)])
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == b"CONCURRENT OWNER"


@pytest.mark.parametrize("tamper", ["missing_parent_pin", "relative_extra_pin"])
def test_read_only_replay_requires_complete_absolute_parent_pins(
        isolated: Any, runner: ModuleType, tamper: str) -> None:
    isolated.run()
    path = isolated.root / "outputs/pilot/report.json"
    report = json.loads(path.read_bytes())
    for key in ("protected_sha256_before", "protected_sha256_after"):
        if tamper == "missing_parent_pin":
            report[key].pop(str(isolated.protected))
        else:
            report[key]["relative_dependency.txt"] = "0"*64
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="protected baseline|must be absolute"):
        runner.summarize_extension(path, runner._sha(path.read_bytes()))
    assert len(isolated.calls) == 16


def test_baseline_date_may_differ_without_changing_numerical_prefix(
        isolated: Any, runner: ModuleType) -> None:
    case = isolated.pilots[0]
    raw = _text(case).replace(b"SYNTHETIC NOT PHYSICAL DATA", b"DIFFERENT SYNTHETIC DATE")
    parsed = parse_srim_range_output(raw)
    runner._validate_native(parsed, case, isolated.baselines[case.base_source_name])


@pytest.mark.parametrize("complete,expected_exit", [(True, 0), (False, 1)])
def test_native_cli_exit_reflects_control_completion(runner: ModuleType, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    complete: bool, expected_exit: int) -> None:
    monkeypatch.setattr(runner, "run_campaign", lambda **_: {
        "all_native_controls_repeated": complete,
        "native_control_status_counts": {"CONTROL_COMPLETE": 8 if complete else 0}})
    monkeypatch.setattr(sys, "argv", ["runner", "--mode", "pilot", "--output", str(tmp_path / "new")])
    assert runner.main() == expected_exit

