"""Bounded normal-mode SRIM extension and read-only replay; never a release tool.

The fixed extension remains at or below a diagnostic 1000 MeV/source-A ceiling.
Native execution first requires an eight-case pilot, then reuses its evidence.
Every extended table must retain the complete preceding 107-row native prefix.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_srim_controls as launcher  # noqa: E402
from scripts.run_srim_coverage import _retained_command  # noqa: E402
from radar.validation.srim_energy_support import audit_srim_energy_support  # noqa: E402
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output  # noqa: E402
from radar.validation.srim_source_catalog import build_source_catalog  # noqa: E402
from radar.validation.srim_source_extension import (  # noqa: E402
    ExtensionControl, build_extension_catalog, expected_extension_grid,
    extension_input_bytes, pilot_extension_controls,
)

BASELINE = "outputs/srim_ost_source_full_20261101/report.json"
BASELINE_SHA256 = "0302ffd3f4de1acd6f5e687d85128e1641e93c5570f6ebb0aec2af2f1506279b"
REPLAY = "scripts/summarize_srim_source_campaign.py"
REPLAY_SHA256 = "7ad98463fea65bcfb4f3919a28928c923ff1c14e204e472f85eac2ba2a43cf9f"
IMPLEMENTATION = (
    "scripts/run_srim_source_extension.py", "src/radar/validation/srim_source_extension.py",
    "src/radar/validation/srim_energy_support.py", "src/radar/core/constants.py",
    "src/radar/core/types.py", "src/radar/core/units.py",
)
_json = launcher._json
_identity = launcher._identity
_sha = launcher._sha
_write_new = launcher._write_new
_execute = launcher._execute
SOURCE_HASHES = launcher.SOURCE_HASHES
PRIVATE_RUNTIME_SHA256 = launcher.PRIVATE_RUNTIME_SHA256
_STATUSES = ("CONTROL_COMPLETE", "REPEAT_MISMATCH", "NATIVE_FAILED", "INVALID_OUTPUT")
_FLAGS = {"diagnostic_status": "INCOMPLETE", "normative_export_allowed": False,
          "production_tables_replaced": False, "source_energy_domain_covered": False,
          "physical_accuracy_verified": False, "isotope_policy_qualified": False}


def _context() -> tuple[Any, Any, dict[str, Any], dict[str, SrimRangeOutput],
                        tuple[ExtensionControl, ...]]:
    path = ROOT / REPLAY
    if _sha(path.read_bytes()) != REPLAY_SHA256:
        raise ValueError("Frozen native replay implementation changed.")
    spec = importlib.util.spec_from_file_location("srim_extension_parent_replay", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load pinned native replay.")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    evidence = helper._Evidence()
    evidence.read(path, REPLAY_SHA256)
    for name in IMPLEMENTATION:
        evidence.read(ROOT / name)
    source_catalog = build_source_catalog(ROOT)
    report, observations = helper._replay(ROOT / BASELINE, BASELINE_SHA256,
                                          source_catalog, evidence)
    if report["mode"] != "full" or not report["all_native_controls_repeated"]:
        raise ValueError("Complete fixed source-native baseline is required.")
    baselines = {case.name: native for case, native in zip(source_catalog, observations, strict=True)}
    return helper, evidence, report, baselines, build_extension_catalog(ROOT)


def _same(actual: Any, expected: Any, label: str) -> None:
    if _json(actual) != _json(expected):
        raise ValueError(f"Retained {label} differs from reconstructed evidence.")


def _validate_native(parsed: SrimRangeOutput, case: ExtensionControl,
                     baseline: SrimRangeOutput) -> None:
    if (parsed.ion_z != case.ion_z or parsed.ion_mass_u != case.ion_mass_u
            or parsed.target_z != case.target_z or parsed.target_symbol != baseline.target_symbol
            or parsed.density_g_cm3 != case.density_g_cm3 or parsed.bragg_correction_percent != 0
            or parsed.engine_label != "SRIM-2006" or parsed.version_label != "SRIM-2012.01"
            or parsed.output_name != "srim_output.txt"):
        raise ValueError("Extension metadata differs from fixed input.")
    for key in ("ion_z", "ion_name", "ion_mass_u", "target_z", "target_symbol", "density_g_cm3",
                "atomic_density_cm3", "bragg_correction_percent", "engine_label", "version_label",
                "conversion_factors"):
        if getattr(parsed, key) != getattr(baseline, key):
            raise ValueError(f"Extension changed baseline metadata: {key}")
    if tuple(row.energy_total_mev for row in parsed.rows) != expected_extension_grid(case):
        raise ValueError("Extension native grid differs from the complete fixed grid.")
    if (len(parsed.rows) <= len(baseline.rows)
            or parsed.rows[:len(baseline.rows)] != baseline.rows):
        raise ValueError("Extension must preserve every baseline row and original token exactly.")


def _pins_unchanged(pins: dict[str, str]) -> None:
    for name, sha in pins.items():
        if _sha(Path(name).read_bytes()) != sha:
            raise ValueError(f"Protected extension input changed: {name}")


def _run_case(case: ExtensionControl, *, output: Path, originals: dict[str, bytes],
              runtime: bytes, pins: dict[str, str], baseline: SrimRangeOutput) -> dict[str, Any]:
    repeats = []
    for number in (1, 2):
        work = output / f"{case.name}_r{number}"
        work.mkdir()
        copied = {work / name: raw for name, raw in originals.items()}
        copied.update({work / "MSVBVM50.dll": runtime, work / "SR.IN": extension_input_bytes(case)})
        for path, raw in copied.items():
            _write_new(path, raw)
        record: dict[str, Any] = {"repeat": number, "input": _identity(work / "SR.IN"),
                                  "physical_accuracy_verified": False, "normative_export_allowed": False}
        try:
            try:
                returned = _execute(work)
            except RuntimeError as error:
                command = _retained_command(work)
                if not command["timed_out"] and command["returncode"] == 0:
                    raise
                record.update(status="NATIVE_FAILED", reason=str(error), command=command)
            else:
                command = _retained_command(work)
                if returned != command or command["timed_out"] or command["returncode"] != 0:
                    raise ValueError("Native success disagrees with retained command.")
                record["command"] = command
                try:
                    parsed = parse_srim_range_output((work / "srim_output.txt").read_bytes())
                    _validate_native(parsed, case, baseline)
                except (ValueError, FileNotFoundError) as error:
                    record.update(status="INVALID_OUTPUT", reason=str(error))
                else:
                    record.update(status="VALID_NATIVE_OUTPUT", parsed=asdict(parsed),
                                  semantic_sha256=parsed.semantic_sha256)
            record["command_file"] = _identity(work / "command.json")
            path = work / "srim_output.txt"
            record["output"] = _identity(path) if path.exists() else None
        finally:
            _pins_unchanged(pins)
            if any(path.read_bytes() != raw for path, raw in copied.items()):
                raise ValueError("Native execution changed its copied input or runtime.")
        _write_new(work / "result.json", _json(record).encode("utf-8"))
        repeats.append(record)
        print(f"{case.name} repeat {number}: {record['status']}", flush=True)
        if record["status"] != "VALID_NATIVE_OUTPUT":
            break
    valid = len(repeats) == 2 and all(r["status"] == "VALID_NATIVE_OUTPUT" for r in repeats)
    identical = valid and repeats[0]["semantic_sha256"] == repeats[1]["semantic_sha256"]
    result = {"case": asdict(case), "repeats": repeats, "origin": "executed_here",
              "status": "CONTROL_COMPLETE" if identical else "REPEAT_MISMATCH" if valid
              else repeats[-1]["status"],
              "baseline_raw_sha256": baseline.raw_sha256,
              "baseline_prefix_rows": len(baseline.rows),
              "baseline_prefix_identical": valid,
              "expected_control_grid_mev": expected_extension_grid(case),
              "semantic_outputs_identical": identical,
              "raw_output_bytes_identical": valid and repeats[0]["output"]["sha256"] ==
              repeats[1]["output"]["sha256"], **_FLAGS}
    checkpoint = output / f"{case.name}.json"
    _write_new(checkpoint, _json(result).encode("utf-8"))
    return {**result, "checkpoint": _identity(checkpoint)}


def _verify_record(record: dict[str, Any], case: ExtensionControl, baseline: SrimRangeOutput,
                   evidence: Any, report_dir: Path) -> SrimRangeOutput | None:
    _same(record["case"], asdict(case), "fixed extension input")
    _same({key: record[key] for key in _FLAGS}, _FLAGS, "case qualification")
    checkpoint, raw = evidence.identity(record["checkpoint"])
    if checkpoint.name != f"{case.name}.json":
        raise ValueError("Invalid extension checkpoint name.")
    import json
    stored = json.loads(raw)
    _same({k: v for k, v in stored.items() if k != "origin"},
          {k: v for k, v in record.items() if k not in ("origin", "checkpoint")}, "checkpoint")
    if (stored["origin"] != "executed_here" or record["origin"] not in ("executed_here", "pilot_reused")
            or (record["origin"] == "executed_here" and checkpoint.parent != report_dir)):
        raise ValueError("Extension origin or directory mismatch.")
    repeats = record["repeats"]
    if not isinstance(repeats, list) or not 1 <= len(repeats) <= 2:
        raise ValueError("Expected one or two retained attempts.")
    observations: list[SrimRangeOutput | None] = []
    for number, repeat in enumerate(repeats, 1):
        if (type(repeat["repeat"]) is not int or repeat["repeat"] != number
                or (observations and observations[-1] is None)):
            raise ValueError("Invalid repeat sequence after native failure.")
        work = checkpoint.parent / f"{case.name}_r{number}"
        data: dict[str, bytes | None] = {}
        for key, filename in (("input", "SR.IN"), ("output", "srim_output.txt"),
                              ("command_file", "command.json")):
            if repeat[key] is None:
                if key != "output" or (work / filename).exists():
                    raise ValueError("Unrecorded extension artifact.")
                data[key] = None
            else:
                path, raw = evidence.identity(repeat[key])
                if path != work / filename:
                    raise ValueError("Artifact outside the distinct canonical repeat directory.")
                data[key] = raw
        if data["input"] != extension_input_bytes(case):
            raise ValueError("Noncanonical extension input.")
        for name, sha in {**SOURCE_HASHES, "MSVBVM50.dll": PRIVATE_RUNTIME_SHA256}.items():
            evidence.read(work / name, sha)
        _same(json.loads(evidence.read(work / "result.json")), repeat, "repeat checkpoint")
        command = _retained_command(work)
        _same(command, repeat["command"], "command")
        _same({key: repeat[key] for key in ("normative_export_allowed", "physical_accuracy_verified")},
              {"normative_export_allowed": False, "physical_accuracy_verified": False}, "repeat flags")
        successful_exit = not command["timed_out"] and command["returncode"] == 0
        status = repeat["status"]
        if status not in ("VALID_NATIVE_OUTPUT", "NATIVE_FAILED", "INVALID_OUTPUT"):
            raise ValueError("Unknown extension native status.")
        if (status == "NATIVE_FAILED") == successful_exit:
            raise ValueError("Native status contradicts exit/timeout.")
        if status != "VALID_NATIVE_OUTPUT":
            if ("parsed" in repeat or "semantic_sha256" in repeat
                    or not isinstance(repeat.get("reason"), str) or not repeat["reason"]):
                raise ValueError("Failed control must have a reason and no numerical observation.")
        parsed = None
        if status != "NATIVE_FAILED":
            try:
                if data["output"] is None:
                    raise ValueError("No native output.")
                parsed = parse_srim_range_output(data["output"])
                _validate_native(parsed, case, baseline)
            except ValueError:
                if status != "INVALID_OUTPUT":
                    raise
                parsed = None
            else:
                if status == "INVALID_OUTPUT":
                    raise ValueError("Invalid-output status contradicts valid native evidence.")
                _same(repeat["parsed"], asdict(parsed), "native parsed result")
                _same(repeat["semantic_sha256"], parsed.semantic_sha256, "semantic digest")
        observations.append(parsed)
    if len(observations) == 1 and observations[0] is not None:
        raise ValueError("Successful extension requires its second attempt.")
    valid = len(observations) == 2 and all(n is not None for n in observations)
    identical = valid and repeats[0]["semantic_sha256"] == repeats[1]["semantic_sha256"]
    expected = {"status": "CONTROL_COMPLETE" if identical else "REPEAT_MISMATCH" if valid
                else repeats[-1]["status"], "semantic_outputs_identical": identical,
                "baseline_prefix_identical": valid, "baseline_raw_sha256": baseline.raw_sha256,
                "baseline_prefix_rows": len(baseline.rows),
                "expected_control_grid_mev": expected_extension_grid(case),
                "raw_output_bytes_identical": valid and repeats[0]["output"]["sha256"] ==
                repeats[1]["output"]["sha256"]}
    _same({key: record[key] for key in expected}, expected, "completion state")
    return observations[0] if identical else None


def replay_extension(report_path: Path, expected_sha256: str, *, helper: Any, evidence: Any,
                     baselines: dict[str, SrimRangeOutput], catalog: tuple[ExtensionControl, ...]
                     ) -> tuple[dict[str, Any], dict[str, SrimRangeOutput]]:
    report_path = report_path.resolve()
    report = helper._load(evidence.read(report_path, helper._digest(expected_sha256)))
    if (report["schema_version"] != "radar.srim_source_extension_campaign.v1"
            or report["mode"] not in ("pilot", "full")):
        raise ValueError("Unexpected extension campaign schema/mode.")
    _same(report["baseline_reference"], _identity(ROOT / BASELINE), "baseline reference")
    _same({key: report[key] for key in _FLAGS}, _FLAGS, "campaign qualification")
    pins = report["protected_sha256_before"]
    _same(pins, report["protected_sha256_after"], "protected extension inputs")
    parent = helper._load(evidence.read(ROOT / BASELINE, BASELINE_SHA256))
    _same(parent["protected_sha256_before"], parent["protected_sha256_after"], "baseline pins")
    required = dict(parent["protected_sha256_after"])
    required.update({str((ROOT / name).resolve()): _sha(evidence.read(ROOT / name))
                     for name in (*IMPLEMENTATION, REPLAY, BASELINE)})
    if any(pins.get(name) != sha for name, sha in required.items()):
        raise ValueError("Missing or changed protected baseline/extension dependency.")
    for name, sha in pins.items():
        if not Path(name).is_absolute():
            raise ValueError("Protected extension paths must be absolute.")
        evidence.read(Path(name), sha)
    selected = pilot_extension_controls(catalog) if report["mode"] == "pilot" else catalog
    _same(report["expected_catalog"], [asdict(c) for c in selected], "catalog")
    reused = {}
    if report["mode"] == "full":
        pilot_path, _ = evidence.identity(report["pilot_reference"])
        # A full report may only reference a pilot, never itself or another full report.
        candidate = helper._load(pilot_path.read_bytes())
        if candidate.get("mode") != "pilot":
            raise ValueError("Full extension requires a pilot reference.")
        pilot, _ = replay_extension(pilot_path, report["pilot_reference"]["sha256"],
                                   helper=helper, evidence=evidence, baselines=baselines, catalog=catalog)
        if not pilot["all_native_controls_repeated"]:
            raise ValueError("Incomplete extension pilot.")
        for name, sha in pilot["protected_sha256_before"].items():
            if pins.get(name) != sha:
                raise ValueError("Extension pilot and full campaign pins differ.")
        reused = {r["case"]["name"]: r for r in pilot["cases"]}
    elif report["pilot_reference"] is not None:
        raise ValueError("Pilot must not reference another campaign.")
    records = report["cases"]
    _same([r["case"]["name"] for r in records], [c.name for c in selected], "case ordering")
    observations = {}
    for record, case in zip(records, selected, strict=True):
        if case.name in reused:
            _same(record, {**reused[case.name], "origin": "pilot_reused"}, "pilot reuse")
        elif record["origin"] != "executed_here":
            raise ValueError("Unexpected reused extension case.")
        parsed = _verify_record(record, case, baselines[case.base_source_name], evidence,
                                report_path.parent)
        if parsed is not None:
            observations[case.name] = parsed
    _same(_counts(records, len(reused)), {key: report[key] for key in _counts(records, len(reused))},
          "extension counters")
    return report, observations


def _counts(records: list[dict[str, Any]], reused: int) -> dict[str, Any]:
    return {"case_count": len(records), "requested_runs_per_case": 2,
            "reused_case_count": reused,
            "executed_here_attempts": sum(len(r["repeats"]) for r in records
                                           if r["origin"] == "executed_here"),
            "all_native_controls_repeated": all(r["status"] == "CONTROL_COMPLETE" for r in records),
            "native_control_status_counts": {s: sum(r["status"] == s for r in records) for s in _STATUSES}}


def run_campaign(*, mode: str, output: Path, pilot_report: Path | None = None,
                 pilot_sha256: str | None = None) -> dict[str, Any]:
    if mode not in ("pilot", "full"):
        raise ValueError("Only fixed pilot/full execution modes are allowed.")
    output = output.resolve()
    root = (ROOT / "outputs").resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("Native extension output must be a new child of repository outputs.")
    if output.exists():
        raise FileExistsError(output)
    if ((mode == "pilot" and (pilot_report is not None or pilot_sha256 is not None))
            or (mode == "full" and (pilot_report is None or pilot_sha256 is None))):
        raise ValueError("Full extension requires pilot path and SHA; pilot accepts neither.")
    helper, evidence, parent, baselines, catalog = _context()
    selected = pilot_extension_controls(catalog) if mode == "pilot" else catalog
    pins = dict(parent["protected_sha256_after"])
    pins.update({str((ROOT / name).resolve()): _sha(evidence.read(ROOT / name))
                 for name in (*IMPLEMENTATION, REPLAY, BASELINE)})
    reused = {}
    pilot_reference = None
    if pilot_report is not None:
        assert pilot_sha256 is not None
        pilot, _ = replay_extension(pilot_report, pilot_sha256, helper=helper, evidence=evidence,
                                    baselines=baselines, catalog=catalog)
        if pilot["mode"] != "pilot" or not pilot["all_native_controls_repeated"]:
            raise ValueError("Complete extension pilot is required before full native execution.")
        for name, sha in pins.items():
            if pilot["protected_sha256_after"].get(name) != sha:
                raise ValueError("Current source/implementation differs from pilot.")
        reused = {r["case"]["name"]: {**r, "origin": "pilot_reused"} for r in pilot["cases"]}
        pilot_reference = _identity(pilot_report)
        pins[str(pilot_report.resolve())] = pilot_sha256
    source = Path(parent["source_directory"])
    originals = {name: evidence.read(source / name, sha) for name, sha in SOURCE_HASHES.items()}
    _, runtime = evidence.identity(parent["private_runtime"])
    if _sha(runtime) != PRIVATE_RUNTIME_SHA256:
        raise ValueError("Unexpected private runtime.")
    _pins_unchanged(pins)
    immediate = {name: sha for name, sha in pins.items() if Path(name).stat().st_size < 1_000_000
                 or name == parent["private_runtime"]["path"]}
    output.mkdir(parents=False)
    records = []
    for case in selected:
        record = reused.get(case.name)
        if record is None:
            record = _run_case(case, output=output, originals=originals, runtime=runtime,
                               pins=immediate, baseline=baselines[case.base_source_name])
        records.append(record)
    for record, case in zip(records, selected, strict=True):
        _verify_record(record, case, baselines[case.base_source_name], evidence, output)
    _pins_unchanged(pins)
    evidence.verify_unchanged()
    report = {"schema_version": "radar.srim_source_extension_campaign.v1", "mode": mode,
              "baseline_reference": _identity(ROOT / BASELINE), "pilot_reference": pilot_reference,
              "expected_catalog": [asdict(c) for c in selected], "cases": records,
              **_counts(records, len(reused)), **_FLAGS,
              "protected_sha256_before": pins,
              "protected_sha256_after": {name: _sha(Path(name).read_bytes()) for name in pins}}
    _write_new(output / "report.json", _json(report).encode("utf-8"))
    return report


def summarize_extension(report_path: Path, expected_sha256: str) -> dict[str, Any]:
    helper, evidence, _, baselines, catalog = _context()
    report, extended = replay_extension(report_path, expected_sha256, helper=helper, evidence=evidence,
                                        baselines=baselines, catalog=catalog)
    selected = dict(baselines)
    selection = []
    for case in catalog:
        if case.name in extended:
            selected[case.base_source_name] = extended[case.name]
            selection.append({"base_source_name": case.base_source_name, "extension_name": case.name,
                              "baseline_raw_sha256": baselines[case.base_source_name].raw_sha256,
                              "selected_raw_sha256": extended[case.name].raw_sha256,
                              "baseline_prefix_identical": True})
    result = {"schema_version": "radar.srim_source_extension_summary.v1",
              "extension_report": _identity(report_path), "mode": report["mode"],
              "native_control_status_counts": report["native_control_status_counts"],
              "extended_table_count": len(selection), "retained_baseline_table_count":
              len(baselines) - len(selection), "selected_native_table_count": len(selected),
              "table_selection": selection,
              "unavailable_extensions": [{"case": r["case"], "status": r["status"]}
                                          for r in report["cases"] if r["status"] != "CONTROL_COMPLETE"],
              "energy_support": asdict(audit_srim_energy_support(tuple(selected.values()))),
              "baseline_energy_support": asdict(audit_srim_energy_support(tuple(baselines.values()))),
              "native_execution_performed_by_summary": False, "source_grids_changed": False,
              "source_spectrum_convolution_performed": False, **_FLAGS,
              "limitations": [
                  "Successful wider whole tables replace narrower diagnostic observations only.",
                  "No table splice, extrapolation, production connection or source truncation.",
                  "Failed or unavailable extensions retain baseline support only, not a zero spectrum.",
                  "The 1000 MeV/source-A ceiling is a bounded campaign choice, not an accuracy guarantee.",
                  "He and other unextended light ions retain their entire prior observed support.",
                  "Grid-count improvement is not a fluence, dose or SEE contribution estimate.",
              ]}
    _same([asdict(c) for c in build_extension_catalog(ROOT)], [asdict(c) for c in catalog],
          "catalog after replay")
    before = evidence.pins.copy()
    evidence.verify_unchanged()
    result.update(evidence_sha256_before=before, evidence_sha256_after=evidence.pins.copy())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full", "replay"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--pilot-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-sha256")
    args = parser.parse_args()
    if args.mode == "replay":
        if args.report is None or args.report_sha256 is None or args.pilot_report or args.pilot_sha256:
            raise ValueError("Replay requires report/SHA only; it never executes native cases.")
        if args.output.exists():
            raise FileExistsError(args.output)
        report = summarize_extension(args.report, args.report_sha256)
        _write_new(args.output, _json(report).encode("utf-8"))
    else:
        if args.report is not None or args.report_sha256 is not None:
            raise ValueError("Native modes do not accept replay report arguments.")
        report = run_campaign(mode=args.mode, output=args.output,
                              pilot_report=args.pilot_report, pilot_sha256=args.pilot_sha256)
    print(_json({"output": str(args.output), "mode": args.mode,
                 "native_control_status_counts": report["native_control_status_counts"],
                 "normative_export_allowed": False}))
    return 0 if args.mode == "replay" or report["all_native_controls_repeated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
