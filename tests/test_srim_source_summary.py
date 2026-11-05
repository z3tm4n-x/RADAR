"""Synthetic read-only campaign replay; never launches SRIM or edits production."""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, localcontext
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from radar.validation.srim_output import parse_srim_range_output
from radar.validation.srim_source_catalog import SourceControl, build_source_catalog


D = Decimal
GRID = (D(1), D(2), D(4), D(8))


@pytest.fixture(scope="module")
def reporter() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/summarize_srim_source_campaign.py"
    spec = importlib.util.spec_from_file_location("srim_source_summary_test_subject", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def controls() -> tuple[SourceControl, ...]:
    catalog = build_source_catalog(Path(__file__).resolve().parents[1])
    return tuple(case for case in catalog if case.ion_z in (2, 6))


def native(case: SourceControl, *, altered: bool = False) -> bytes:
    fixture = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    lines = [line.strip() for line in fixture.read_bytes().decode("cp1251").splitlines()
             if line.strip()]
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC NOT PHYSICAL DATA"
    header[5] = "Disk File Name = srim_output.txt"
    header[6] = header[17] = f"Ion = Synthetic [{case.ion_z}] , Mass = {case.ion_mass_u} amu"
    header[7] = f"Density = {case.density_g_cm3} g/cm3 = 6.0000E+22 atoms/cm3"
    header[12] = f"{'Al' if case.target_z == 13 else 'Si'} {case.target_z} 100.00 100.00"
    rows = [f"{e} MeV {'0.2' if altered else '0.1'} 0.001 {i} um 1 A 1 A"
            for i, e in enumerate(GRID, 1)]
    footer = lines[lines.index("-" * 59):]
    text = re.sub(r"(?<=\d),(?=\d)", ".", "\r\n".join(header + rows + footer[:-1]))
    return (text + "\r\n" + footer[-1] + "\r\n").encode("ascii")


@pytest.fixture
def saved(reporter: ModuleType, controls: tuple[SourceControl, ...], tmp_path: Path,
          monkeypatch: pytest.MonkeyPatch) -> Any:
    root = tmp_path / "repo"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    def identity(path: Path) -> dict[str, Any]:
        return {"path": str(path.resolve()), "bytes": path.stat().st_size,
                "sha256": reporter._sha(path.read_bytes())}

    for relative in (*reporter.IMPLEMENTATION, *reporter.REQUIRED_CAMPAIGN_PINS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"SYNTHETIC PIN {relative}".encode())
    for name in reporter.NATIVE_SOURCE_HASHES:
        (source / name).write_bytes(f"SYNTHETIC NOT EXECUTABLE {name}".encode())
    (source / "SR.IN").write_bytes(b"ORIGINAL INPUT NEVER CHANGES")
    runtime = tmp_path / "MSVBVM50.dll"
    runtime.write_bytes(b"SYNTHETIC NOT EXECUTABLE RUNTIME")
    monkeypatch.setattr(reporter, "ROOT", root)
    monkeypatch.setattr(reporter, "NATIVE_SOURCE_HASHES", {
        name: identity(source / name)["sha256"] for name in reporter.NATIVE_SOURCE_HASHES})
    monkeypatch.setattr(reporter, "PRIVATE_RUNTIME_SHA256", identity(runtime)["sha256"])
    monkeypatch.setattr(reporter, "build_source_catalog", lambda _: controls)
    monkeypatch.setattr(reporter, "pilot_controls", lambda _: controls[:2])
    monkeypatch.setattr(reporter, "expected_control_grid", lambda _: GRID)
    pins = {str((root / relative).resolve()): identity(root / relative)["sha256"]
            for relative in reporter.REQUIRED_CAMPAIGN_PINS}
    pins.update({str((source / name).resolve()): sha
                 for name, sha in reporter.NATIVE_SOURCE_HASHES.items()})
    pins[str(source / "SR.IN")] = identity(source / "SR.IN")["sha256"]
    pins[str(runtime)] = identity(runtime)["sha256"]

    def finish_case(record: dict[str, Any]) -> None:
        repeats = record["repeats"]
        valid = len(repeats) == 2 and all(r["status"] == "VALID_NATIVE_OUTPUT" for r in repeats)
        same = valid and repeats[0]["semantic_sha256"] == repeats[1]["semantic_sha256"]
        record.update(status="CONTROL_COMPLETE" if same else "REPEAT_MISMATCH" if valid
                      else repeats[-1]["status"], complete_native_control_grid_verified=valid,
                      semantic_outputs_identical=same, raw_output_bytes_identical=valid and
                      repeats[0]["output"]["sha256"] == repeats[1]["output"]["sha256"])

    def make_case(case: SourceControl, directory: Path) -> dict[str, Any]:
        repeats = []
        for number in (1, 2):
            work = directory / f"{case.name}_r{number}"
            work.mkdir()
            for name in reporter.NATIVE_SOURCE_HASHES:
                (work / name).write_bytes((source / name).read_bytes())
            (work / "MSVBVM50.dll").write_bytes(runtime.read_bytes())
            (work / "SR.IN").write_bytes(reporter.input_bytes(case))
            (work / "srim_output.txt").write_bytes(native(case))
            command = {"argv": [str(work / "SRModule.exe")], "cwd": str(work), "pid": 1234,
                       "returncode": 0, "timed_out": False, "elapsed_seconds": .01,
                       "stdout": "", "stderr": ""}
            (work / "command.json").write_text(reporter._json(command), encoding="utf-8")
            parsed = parse_srim_range_output(native(case))
            repeats.append({"repeat": number, "input": identity(work / "SR.IN"),
                            "output": identity(work / "srim_output.txt"),
                            "command_file": identity(work / "command.json"), "command": command,
                            "status": "VALID_NATIVE_OUTPUT", "parsed": asdict(parsed),
                            "semantic_sha256": parsed.semantic_sha256,
                            "physical_accuracy_verified": False, "normative_export_allowed": False})
        result = {"case": asdict(case), "scope": reporter._scope(case), "repeats": repeats,
                  "origin": "executed_here", "expected_control_grid_mev": GRID,
                  "physical_accuracy_verified": False, "normative_export_allowed": False}
        finish_case(result)
        return result

    def finish_report(report: dict[str, Any]) -> None:
        cases = report["cases"]
        report.update(case_count=len(cases), requested_runs_per_case=2,
                      executed_here_attempts=sum(len(c["repeats"]) for c in cases
                                                 if c["origin"] == "executed_here"),
                      reused_case_count=sum(c["origin"] == "pilot_reused" for c in cases),
                      all_native_controls_repeated=all(c["status"] == "CONTROL_COMPLETE"
                                                       for c in cases),
                      native_control_status_counts={s: sum(c["status"] == s for c in cases)
                                                     for s in reporter._STATUSES})

    def save(report: dict[str, Any], path: Path) -> None:
        for record in report["cases"]:
            if record["origin"] != "executed_here":
                continue
            for repeat in record["repeats"]:
                work = Path(repeat["input"]["path"]).parent
                (work / "result.json").write_text(reporter._json(repeat), encoding="utf-8")
            checkpoint = path.parent / f"{record['case']['name']}.json"
            checkpoint.write_text(reporter._json({k: v for k, v in record.items()
                                                 if k != "checkpoint"}), encoding="utf-8")
            record["checkpoint"] = identity(checkpoint)
        path.write_text(reporter._json(report), encoding="utf-8")

    def create(mode: str = "pilot", pilot: dict[str, Any] | None = None,
               pilot_path: Path | None = None) -> tuple[dict[str, Any], Path]:
        directory = root / "outputs" / mode
        directory.mkdir(parents=True)
        selected = controls[:2] if mode == "pilot" else controls
        reused = {} if pilot is None else {c["case"]["name"]: c for c in pilot["cases"]}
        cases = [{**reused[c.name], "origin": "pilot_reused"} if c.name in reused
                 else make_case(c, directory) for c in selected]
        campaign_pins = pins.copy()
        if pilot_path is not None:
            campaign_pins[str(pilot_path)] = identity(pilot_path)["sha256"]
        report = {"schema_version": "radar.srim_ost_source_campaign.v1", "mode": mode,
                  "expected_catalog": [asdict(c) for c in selected], "cases": cases,
                  "source_directory": str(source), "native_source_files": reporter.NATIVE_SOURCE_HASHES,
                  "original_input_sha256": identity(source / "SR.IN")["sha256"],
                  "private_runtime": identity(runtime),
                  "protected_sha256_before": campaign_pins.copy(),
                  "protected_sha256_after": campaign_pins.copy(),
                  "pilot_reference": None if pilot_path is None else identity(pilot_path),
                  "diagnostic_status": "INCOMPLETE", "normative_export_allowed": False,
                  "production_tables_replaced": False, "isotope_policy_qualified": False,
                  "source_energy_domain_covered": False, "physical_accuracy_verified": False}
        finish_report(report)
        path = directory / "report.json"
        save(report, path)
        return report, path

    report, path = create()

    def summarize(at: Path | None = None) -> dict[str, Any]:
        selected = path if at is None else at
        result: dict[str, Any] = reporter.summarize_campaign(
            selected, expected_report_sha256=identity(selected)["sha256"])
        return result

    return SimpleNamespace(root=root, source=source, runtime=runtime, identity=identity,
                           report=report, path=path, save=save, summarize=summarize,
                           create=create, finish_case=finish_case, finish_report=finish_report)


def test_replay_is_deterministic_and_not_physical_qualification(saved: Any) -> None:
    report = saved.summarize()
    assert saved.summarize() == report
    assert report["verified_complete_native_control_count"] == 2
    assert report["al_candidate_count"] == 2
    assert report["silicon_complete_native_control_count"] == 1
    assert report["all_requested_native_controls_repeated"] is True
    assert report["observed_positive_roster_complete"] is False
    assert report["evidence_sha256_before"] == report["evidence_sha256_after"]
    for name in ("normative_export_allowed", "source_energy_domain_covered", "physical_accuracy_verified",
                 "production_tables_replaced", "source_spectrum_convolution_performed",
                 "isotope_policy_qualified", "silicon_production_let_adapter_created",
                 "native_execution_performed_by_summary"):
        assert report[name] is False
    assert report["diagnostic_status"] == "INCOMPLETE"
    summaries = {row["source"]: row for row in report["source_native_coverage"]["summaries"]}
    assert summaries["GCR"]["missing_positive_species"] == 83
    assert summaries["SEP"]["missing_positive_species"] == 23
    assert summaries["GCR"]["declared_zero_species"] == 7
    assert summaries["SEP"]["declared_zero_species"] == 3
    assert summaries["GCR"]["positive_species_numeric_mass_equal_both_materials"] == 1
    for table in report["al_diagnostic_candidates"]:
        assert table["range_kind"] == "derived_anchored_csda_path"
        assert table["native_range_kind"] == "projected"
        assert table["anchored_range_g_cm2"][0] == 0
        assert table["native_projected_range_g_cm2"][0] > 0
        assert table["energy_total_mev"] == GRID
        assert table["normative_export_allowed"] is False


def test_full_reuses_verified_pilot_without_duplicate_observations(saved: Any) -> None:
    full, path = saved.create("full", saved.report, saved.path)
    assert full["reused_case_count"] == 2
    summary = saved.summarize(path)
    assert summary["verified_complete_native_control_count"] == 4
    assert summary["al_candidate_count"] == 4
    assert summary["silicon_complete_native_control_count"] == 2
    assert summary["source_native_coverage"]["summaries"][0]["missing_positive_species"] == 82


@pytest.mark.parametrize("status", ["NATIVE_FAILED", "INVALID_OUTPUT", "REPEAT_MISMATCH"])
def test_incomplete_controls_are_missing_not_zero(saved: Any, reporter: ModuleType,
                                                 controls: tuple[SourceControl, ...], status: str) -> None:
    record = saved.report["cases"][0]
    if status == "REPEAT_MISMATCH":
        repeat = record["repeats"][1]
        output = Path(repeat["output"]["path"])
        raw = native(controls[0], altered=True)
        output.write_bytes(raw)
        parsed = parse_srim_range_output(raw)
        repeat.update(output=saved.identity(output), parsed=asdict(parsed),
                      semantic_sha256=parsed.semantic_sha256)
    else:
        record["repeats"] = record["repeats"][:1]
        repeat = record["repeats"][0]
        output = Path(repeat["output"]["path"])
        output.write_bytes(b"SYNTHETIC FAILED OUTPUT NOT A SPECTRUM")
        repeat.update(status=status, output=saved.identity(output), reason="Synthetic failure")
        del repeat["parsed"], repeat["semantic_sha256"]
        if status == "NATIVE_FAILED":
            repeat["command"]["returncode"] = 7
            command_path = Path(repeat["command_file"]["path"])
            command_path.write_text(reporter._json(repeat["command"]), encoding="utf-8")
            repeat["command_file"] = saved.identity(command_path)
    saved.finish_case(record)
    saved.finish_report(saved.report)
    saved.save(saved.report, saved.path)
    summary = saved.summarize()
    assert summary["verified_complete_native_control_count"] == 1
    assert summary["al_candidate_count"] == 0
    assert summary["unavailable_controls"][0]["status"] == status
    assert summary["source_native_coverage"]["summaries"][0]["positive_species_missing_one_material"] == 1
    assert not summary["observed_positive_roster_complete"]


@pytest.mark.parametrize("kind", ["source", "runtime", "production", "raw", "copy", "input"])
def test_tampered_bytes_rejected(saved: Any, kind: str) -> None:
    repeat = saved.report["cases"][0]["repeats"][0]
    work = Path(repeat["input"]["path"]).parent
    path = {"source": saved.source / "SCOEF03.dat", "runtime": saved.runtime,
            "production": saved.root / "src/radar/data/normative/stopping/hze_si_let.csv",
            "raw": Path(repeat["output"]["path"]), "copy": work / "SRModule.exe",
            "input": Path(repeat["input"]["path"])}[kind]
    path.write_bytes(b"TAMPERED")
    with pytest.raises(ValueError, match="Pinned file changed"):
        saved.summarize()


@pytest.mark.parametrize("kind", ["missing", "duplicate", "catalog", "forged_count", "protected",
                                 "before_after", "qualified", "input", "command", "parsed",
                                 "semantic", "grid", "scope", "repeat_number", "one_repeat",
                                 "origin", "artifact_directory", "native_failure_exit"])
def test_forged_report_rejected(saved: Any, reporter: ModuleType, kind: str) -> None:
    report = saved.report
    case = report["cases"][0]
    repeat = case["repeats"][0]
    if kind == "missing":
        report["cases"].pop()
    elif kind == "duplicate":
        report["cases"][1] = case
    elif kind == "catalog":
        report["expected_catalog"] = []
    elif kind == "forged_count":
        report["case_count"] = 999
    elif kind == "protected":
        for name in ("protected_sha256_before", "protected_sha256_after"):
            report[name].pop(str(saved.root / "src/radar/data/normative/integrity_manifest.json"))
    elif kind == "before_after":
        report["protected_sha256_after"] = {}
    elif kind == "qualified":
        report["normative_export_allowed"] = True
    elif kind == "input":
        path = Path(repeat["input"]["path"])
        path.write_bytes(b"DIFFERENT INPUT WITH REFRESHED HASH")
        repeat["input"] = saved.identity(path)
    elif kind == "command":
        repeat["command"]["argv"] = ["something_else.exe"]
    elif kind == "parsed":
        repeat["parsed"]["ion_mass_u"] = "999"
    elif kind == "semantic":
        repeat["semantic_sha256"] = "a" * 64
    elif kind == "grid":
        case["expected_control_grid_mev"] = (D(1), D(8))
    elif kind == "scope":
        case["scope"]["physical_accuracy_verified"] = True
    elif kind == "repeat_number":
        case["repeats"][1]["repeat"] = 1
    elif kind == "one_repeat":
        case["repeats"].pop()
    elif kind == "origin":
        case["origin"] = "pilot_reused"
    elif kind == "artifact_directory":
        repeat["output"] = case["repeats"][1]["output"].copy()
    elif kind == "native_failure_exit":
        repeat.update(status="NATIVE_FAILED", reason="Claimed failure but exit zero")
        del repeat["parsed"], repeat["semantic_sha256"]
    saved.save(report, saved.path)
    with pytest.raises(ValueError):
        saved.summarize()


@pytest.mark.parametrize("change", ["mass", "grid", "label"])
def test_reparsed_metadata_and_complete_grid_checked(saved: Any, reporter: ModuleType,
                                                    change: str) -> None:
    repeat = saved.report["cases"][0]["repeats"][0]
    path = Path(repeat["output"]["path"])
    raw = path.read_bytes()
    if change == "mass":
        raw = raw.replace(b"Mass = 4.0", b"Mass = 4.1")
    elif change == "grid":
        raw = raw.replace(b"2 MeV 0.1 0.001 2 um 1 A 1 A\r\n", b"")
    else:
        raw = raw.replace(b"SRIM-2012.01", b"SRIM-2013.01")
    path.write_bytes(raw)
    parsed = parse_srim_range_output(raw)
    repeat.update(output=saved.identity(path), parsed=asdict(parsed),
                  semantic_sha256=parsed.semantic_sha256)
    saved.save(saved.report, saved.path)
    with pytest.raises(ValueError, match="metadata mismatch|energy grid mismatch"):
        saved.summarize()


def test_stale_checkpoint_and_repeat_ledger_rejected(saved: Any) -> None:
    checkpoint = Path(saved.report["cases"][0]["checkpoint"]["path"])
    checkpoint.write_bytes(b"STALE CHECKPOINT")
    with pytest.raises(ValueError, match="Pinned file changed"):
        saved.summarize()


def test_checkpoint_content_not_just_hash_checked(saved: Any, reporter: ModuleType) -> None:
    case = saved.report["cases"][0]
    path = Path(case["checkpoint"]["path"])
    checkpoint = json.loads(path.read_bytes())
    checkpoint["semantic_outputs_identical"] = False
    path.write_text(reporter._json(checkpoint), encoding="utf-8")
    case["checkpoint"] = saved.identity(path)
    saved.path.write_text(reporter._json(saved.report), encoding="utf-8")
    with pytest.raises(ValueError, match="case checkpoint"):
        saved.summarize()


def test_repeat_result_content_checked(saved: Any, reporter: ModuleType) -> None:
    repeat = saved.report["cases"][0]["repeats"][0]
    path = Path(repeat["input"]["path"]).parent / "result.json"
    path.write_text(reporter._json({**repeat, "status": "INVALID_OUTPUT"}), encoding="utf-8")
    with pytest.raises(ValueError, match="repeat checkpoint"):
        saved.summarize()


def test_full_pilot_reuse_cannot_be_relabelled(saved: Any) -> None:
    full, path = saved.create("full", saved.report, saved.path)
    full["cases"][0]["origin"] = "executed_here"
    saved.save(full, path)
    with pytest.raises(ValueError, match="reused pilot case"):
        saved.summarize(path)


@pytest.mark.parametrize("value", [None, 1, "", "a" * 63, "a" * 65, "A" * 64, "b" * 64])
def test_report_requires_explicit_exact_sha(saved: Any, reporter: ModuleType, value: Any) -> None:
    with pytest.raises(ValueError, match="Pinned file changed"):
        reporter.summarize_campaign(saved.path, expected_report_sha256=value)


@pytest.mark.parametrize("kind", ["artifact", "protected", "original"])
def test_nullable_hash_cannot_disable_pin_validation(saved: Any, kind: str) -> None:
    if kind == "artifact":
        saved.report["cases"][0]["repeats"][0]["output"]["sha256"] = None
    elif kind == "protected":
        for key in ("protected_sha256_before", "protected_sha256_after"):
            saved.report[key][str(saved.root / "src/radar/validation/srim_output.py")] = None
    else:
        saved.report["original_input_sha256"] = None
    saved.save(saved.report, saved.path)
    with pytest.raises(ValueError, match="SHA256"):
        saved.summarize()


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'[]'])
def test_invalid_json_rejected(reporter: ModuleType, raw: bytes) -> None:
    with pytest.raises(ValueError):
        reporter._load(raw)


def test_ambient_decimal_context_does_not_change_summary(saved: Any) -> None:
    expected = saved.summarize()
    with localcontext() as context:
        context.prec = 6
        actual = saved.summarize()
    assert actual == expected


def test_final_recheck_catches_mutation_during_derivation(saved: Any, reporter: ModuleType,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    original = reporter.audit_srim_native_coverage

    def mutate(observations: Any) -> Any:
        (saved.source / "VERSION").write_bytes(b"CHANGED DURING REPLAY")
        return original(observations)

    monkeypatch.setattr(reporter, "audit_srim_native_coverage", mutate)
    with pytest.raises(ValueError, match="Pinned file changed"):
        saved.summarize()


def test_cli_writes_exclusively_and_keeps_diagnostic_status(saved: Any, reporter: ModuleType,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    output = saved.root / "summary.json"
    monkeypatch.setattr(sys, "argv", ["summary", "--campaign-report", str(saved.path),
                                     "--report-sha256", saved.identity(saved.path)["sha256"],
                                     "--output", str(output)])
    assert reporter.main() == 0
    assert json.loads(output.read_bytes())["normative_export_allowed"] is False
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        reporter.main()
    assert output.read_bytes() == before
