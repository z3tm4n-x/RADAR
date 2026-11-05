"""Isolated normal-mode SRIM campaign for literal OST source mass parameters.

Pilot gate precedes the full fixed roster. No installers, special-mode files,
production replacements, isotope selection, extrapolation or normative export.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Context, ROUND_HALF_EVEN, localcontext
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_srim_controls as native_runner  # noqa: E402
from scripts import run_srim_coverage as legacy_runner  # noqa: E402
from radar.validation.srim_output import parse_srim_range_output, SrimRangeOutput  # noqa: E402
from radar.validation.srim_source_catalog import (  # noqa: E402
    SourceControl, build_source_catalog, expected_control_grid, input_bytes, pilot_controls,
)

PARENT_CANDIDATE = "outputs/srim_range_candidate_final_20261101.json"
PARENT_SHA256 = "f423c5da9eb45e8489787257a84479a0489f406c866ed12802a97313fb065b53"
LEGACY_BASELINE = "outputs/srim_legacy_baseline_final_20261101/report.json"
LEGACY_BASELINE_SHA256 = "8ed401964181371ebb508f16bec41aa5b3600b136f47f763be4c0596f448c9cc"
IMPLEMENTATION = (
    "scripts/run_srim_source_campaign.py", "src/radar/validation/srim_source_catalog.py",
)
SOURCE_HASHES = native_runner.SOURCE_HASHES
PRIVATE_RUNTIME_SHA256 = native_runner.PRIVATE_RUNTIME_SHA256
_execute = native_runner._execute
_identity = native_runner._identity
_sha = native_runner._sha
_json = native_runner._json
_write_new = native_runner._write_new
_CONTEXT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)


def _read_pinned(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if _sha(raw) != expected:
        raise ValueError(f"Pinned file changed: {path}")
    return raw


def _identity_valid(item: dict[str, Any]) -> Path:
    path = Path(item["path"])
    if len(_read_pinned(path, item["sha256"])) != item["bytes"]:
        raise ValueError("Retained file size mismatch.")
    return path


def _initial_pins() -> dict[Path, str]:
    parent = json.loads(_read_pinned(ROOT / PARENT_CANDIDATE, PARENT_SHA256))
    baseline = json.loads(_read_pinned(ROOT / LEGACY_BASELINE, LEGACY_BASELINE_SHA256))
    if (parent["protected_sha256_before"] != parent["protected_sha256_after"]
            or baseline["protected_sha256_before"] != baseline["protected_sha256_after"]):
        raise ValueError("Parent reports recorded a protected-file mutation.")
    pins = {ROOT / name: sha for name, sha in parent["protected_sha256_after"].items()}
    pins.update({Path(name): sha for name, sha in baseline["protected_sha256_after"].items()})
    pins.update({ROOT / PARENT_CANDIDATE: PARENT_SHA256,
                 ROOT / LEGACY_BASELINE: LEGACY_BASELINE_SHA256})
    pins.update({ROOT / name: _sha((ROOT / name).read_bytes()) for name in IMPLEMENTATION})
    _verify_pins(pins)
    return pins


def _verify_pins(pins: dict[Path, str]) -> None:
    for path, expected in pins.items():
        _read_pinned(path, expected)


def _validate_native(parsed: SrimRangeOutput, case: SourceControl) -> None:
    if (parsed.ion_z != case.ion_z or parsed.ion_mass_u != case.ion_mass_u
            or parsed.target_z != case.target_z or parsed.density_g_cm3 != case.density_g_cm3
            or parsed.bragg_correction_percent != 0 or parsed.engine_label != "SRIM-2006"
            or parsed.version_label != "SRIM-2012.01" or parsed.output_name != "srim_output.txt"):
        raise ValueError("Native source-control metadata mismatch; no mass rounding permitted.")
    if tuple(row.energy_total_mev for row in parsed.rows) != expected_control_grid(case):
        raise ValueError("Native source-control energy grid mismatch.")


def _verify_case_artifacts(record: dict[str, Any], case: SourceControl) -> None:
    """Tie checkpoints and distinct repeat paths to canonical isolated directories."""
    checkpoint = _identity_valid(record["checkpoint"])
    if checkpoint.name != f"{case.name}.json":
        raise ValueError("Case checkpoint has an unexpected name.")
    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
    excluded = {"checkpoint", "origin"}
    expected = json.loads(_json({k: v for k, v in record.items() if k not in excluded}))
    if {k: v for k, v in stored.items() if k not in excluded} != expected:
        raise ValueError("Case checkpoint content differs from the embedded record.")
    if stored["origin"] != "executed_here" or record["origin"] not in ("executed_here", "pilot_reused"):
        raise ValueError("Unexpected case execution origin.")
    if not 1 <= len(record["repeats"]) <= 2:
        raise ValueError("Every attempted case must retain one or two repeat records.")
    for number, repeat in enumerate(record["repeats"], 1):
        if repeat["repeat"] != number:
            raise ValueError("Repeat numbering differs from distinct canonical work directories.")
        work = (checkpoint.parent / f"{case.name}_r{number}").resolve()
        for key, filename in (("input", "SR.IN"), ("output", "srim_output.txt"),
                              ("command_file", "command.json")):
            item = repeat[key]
            if item is None:
                if key != "output" or (work / filename).exists():
                    raise ValueError("Missing retained artifact identity.")
                continue
            if Path(item["path"]).resolve() != work / filename:
                raise ValueError("Repeat artifact is not in its canonical isolated directory.")
            _identity_valid(item)
        if (work / "SR.IN").read_bytes() != input_bytes(case):
            raise ValueError("Retained input differs from the fixed source control.")
        for name, sha in {**SOURCE_HASHES, "MSVBVM50.dll": PRIVATE_RUNTIME_SHA256}.items():
            _read_pinned(work / name, sha)
        command = legacy_runner._retained_command(work)
        if command != repeat["command"]:
            raise ValueError("Command ledger differs from the retained command file.")
        if repeat["status"] != "VALID_NATIVE_OUTPUT" and (
                "parsed" in repeat or "semantic_sha256" in repeat):
            raise ValueError("Failed native output cannot contain numerical observations.")


def _verify_reusable_case(record: dict[str, Any], case: SourceControl) -> None:
    if (record["status"] != "CONTROL_COMPLETE"
            or json.loads(_json(record["case"])) != json.loads(_json(asdict(case)))
            or len(record["repeats"]) != 2):
        raise ValueError("Pilot control does not match the complete fixed case.")
    _verify_case_artifacts(record, case)
    semantics = []
    for number, repeat in enumerate(record["repeats"], 1):
        if repeat["repeat"] != number or repeat["status"] != "VALID_NATIVE_OUTPUT":
            raise ValueError("Pilot requires two valid repeated native outputs.")
        path = _identity_valid(repeat["input"])
        if path.read_bytes() != input_bytes(case):
            raise ValueError("Pilot input differs from canonical source control.")
        output = _identity_valid(repeat["output"])
        command_path = _identity_valid(repeat["command_file"])
        command = legacy_runner._retained_command(command_path.parent)
        if command != repeat["command"] or command["timed_out"] or command["returncode"] != 0:
            raise ValueError("Pilot command did not complete successfully.")
        parsed = parse_srim_range_output(output.read_bytes())
        _validate_native(parsed, case)
        if (parsed.semantic_sha256 != repeat["semantic_sha256"]
                or json.loads(_json(asdict(parsed))) != json.loads(_json(repeat["parsed"]))):
            raise ValueError("Pilot parsed output differs from retained bytes.")
        semantics.append(parsed.semantic_sha256)
    if len(set(semantics)) != 1:
        raise ValueError("Pilot semantic repeats differ.")


def _load_pilot(path: Path, expected: str, pilots: tuple[SourceControl, ...],
                pins: dict[Path, str]) -> dict[str, dict[str, Any]]:
    report = json.loads(_read_pinned(path, expected))
    wanted = json.loads(_json([asdict(c) for c in pilots]))
    if (report["schema_version"] != "radar.srim_ost_source_campaign.v1"
            or report["mode"] != "pilot" or report["expected_catalog"] != wanted
            or not report["all_native_controls_repeated"]
            or report["protected_sha256_before"] != report["protected_sha256_after"]):
        raise ValueError("Full campaign requires the complete matching pilot report.")
    for path_text, sha in report["protected_sha256_after"].items():
        _read_pinned(Path(path_text), sha)
    for protected, sha in pins.items():
        if report["protected_sha256_after"].get(str(protected.resolve())) != sha:
            raise ValueError("Pilot implementation/source pins differ from this campaign.")
    if [r["case"]["name"] for r in report["cases"]] != [c.name for c in pilots]:
        raise ValueError("Pilot contains missing, reordered or duplicate cases.")
    result = {}
    for record, case in zip(report["cases"], pilots, strict=True):
        _verify_reusable_case(record, case)
        result[case.name] = {**record, "origin": "pilot_reused"}
    return result


def _scope(case: SourceControl) -> dict[str, Any]:
    with localcontext(_CONTEXT):
        maximum = case.energy_max_mev / case.ion_mass_u
        caution = case.energy_max_mev > 1000 * case.ion_mass_u
        within_claim = case.energy_max_mev <= 10000 * case.ion_mass_u
    return {"energy_total_mev_bounds": (case.energy_min_mev, case.energy_max_mev),
            "max_energy_divided_by_declared_mass": maximum,
            "package_trim_help_high_energy_caution": caution,
            "within_package_table_claim": within_claim,
            "physical_accuracy_verified": False, "normative_export_allowed": False}


def _run_case(case: SourceControl, *, output: Path, originals: dict[str, bytes],
              runtime: bytes, immediate_pins: dict[Path, str]) -> dict[str, Any]:
    repeats: list[dict[str, Any]] = []
    for repeat in (1, 2):
        work = output / f"{case.name}_r{repeat}"
        work.mkdir()
        copied = {work / name: raw for name, raw in originals.items()}
        copied.update({work / "MSVBVM50.dll": runtime, work / "SR.IN": input_bytes(case)})
        for path, raw in copied.items():
            _write_new(path, raw)
        record: dict[str, Any] = {"repeat": repeat, "input": _identity(work / "SR.IN"),
                                  "physical_accuracy_verified": False,
                                  "normative_export_allowed": False}
        try:
            try:
                returned = _execute(work)
            except RuntimeError as error:
                command = legacy_runner._retained_command(work)
                if not command["timed_out"] and command["returncode"] == 0:
                    raise
                record.update(status="NATIVE_FAILED", reason=str(error), command=command)
            else:
                command = legacy_runner._retained_command(work)
                if returned != command or command["timed_out"] or command["returncode"] != 0:
                    raise ValueError("Returned command disagrees with retained native success.")
                record["command"] = command
                try:
                    parsed = parse_srim_range_output((work / "srim_output.txt").read_bytes())
                    _validate_native(parsed, case)
                except (ValueError, FileNotFoundError) as error:
                    record.update(status="INVALID_OUTPUT", reason=str(error))
                else:
                    record.update(status="VALID_NATIVE_OUTPUT", parsed=asdict(parsed),
                                  semantic_sha256=parsed.semantic_sha256)
            record["command_file"] = _identity(work / "command.json")
            path = work / "srim_output.txt"
            record["output"] = _identity(path) if path.is_file() else None
        finally:
            _verify_pins(immediate_pins)
            if any(path.read_bytes() != raw for path, raw in copied.items()):
                raise ValueError("Native execution changed a copied runtime/input file.")
        _write_new(work / "result.json", _json(record).encode("utf-8"))
        repeats.append(record)
        print(f"{case.name} repeat {repeat}: {record['status']}", flush=True)
        if record["status"] != "VALID_NATIVE_OUTPUT":
            break
    valid = len(repeats) == 2 and all(r["status"] == "VALID_NATIVE_OUTPUT" for r in repeats)
    identical = valid and repeats[0]["semantic_sha256"] == repeats[1]["semantic_sha256"]
    record = {
        "case": asdict(case), "repeats": repeats, "scope": _scope(case),
        "status": "CONTROL_COMPLETE" if identical else "REPEAT_MISMATCH" if valid
        else repeats[-1]["status"],
        "origin": "executed_here", "expected_control_grid_mev": expected_control_grid(case),
        "complete_native_control_grid_verified": valid, "semantic_outputs_identical": identical,
        "raw_output_bytes_identical": valid and repeats[0]["output"]["sha256"] ==
        repeats[1]["output"]["sha256"],
        "physical_accuracy_verified": False, "normative_export_allowed": False,
    }
    checkpoint = output / f"{case.name}.json"
    _write_new(checkpoint, _json(record).encode("utf-8"))
    return {**record, "checkpoint": _identity(checkpoint)}


def run_campaign(*, mode: str, source: Path, private_runtime: Path, output: Path,
                 pilot_report: Path | None = None, pilot_sha256: str | None = None) -> dict[str, Any]:
    if mode not in ("pilot", "full") or type(mode) is not str:
        raise ValueError("Only fixed pilot/full modes are supported.")
    output = output.resolve()
    base = (ROOT / "outputs").resolve()
    if output == base or not output.is_relative_to(base):
        raise ValueError("Native campaign output must be a new child of repository outputs.")
    if output.exists():
        raise FileExistsError(output)
    if (mode == "pilot" and (pilot_report is not None or pilot_sha256 is not None)) or (
            mode == "full" and (pilot_report is None or pilot_sha256 is None)):
        raise ValueError("Full campaign requires both pilot path and SHA; pilot mode accepts neither.")
    pins = _initial_pins()
    catalog = build_source_catalog(ROOT)
    pilots = pilot_controls(catalog)
    selected = pilots if mode == "pilot" else catalog
    originals = {name: _read_pinned(source / name, sha) for name, sha in SOURCE_HASHES.items()}
    runtime = _read_pinned(private_runtime, PRIVATE_RUNTIME_SHA256)
    original_input = (source / "SR.IN").read_bytes()
    pins.update({(source / name).resolve(): sha for name, sha in SOURCE_HASHES.items()})
    pins.update({(source / "SR.IN").resolve(): _sha(original_input),
                 private_runtime.resolve(): PRIVATE_RUNTIME_SHA256})
    reused = {}
    pilot_reference = None
    if pilot_report is not None:
        assert pilot_sha256 is not None
        reused = _load_pilot(pilot_report, pilot_sha256, pilots, pins)
        pilot_reference = _identity(pilot_report)
        pins[pilot_report.resolve()] = pilot_sha256
    _verify_pins(pins)
    # Large frozen report files are checked before/after the campaign. Small
    # source, implementation, runtime and input files are checked every repeat.
    immediate = {path: sha for path, sha in pins.items() if path.stat().st_size < 1_000_000
                 or path.resolve() == private_runtime.resolve()}
    output.mkdir(parents=False)
    cases = []
    for case in selected:
        record = reused.get(case.name)
        if record is None:
            record = _run_case(case, output=output, originals=originals, runtime=runtime,
                               immediate_pins=immediate)
        cases.append(record)
    _verify_pins(pins)
    for case, record in zip(selected, cases, strict=True):
        if record["status"] == "CONTROL_COMPLETE":
            _verify_reusable_case(record, case)
        else:
            _verify_case_artifacts(record, case)
    report = {
        "schema_version": "radar.srim_ost_source_campaign.v1", "mode": mode,
        "expected_catalog": [asdict(c) for c in selected], "cases": cases,
        "case_count": len(cases), "requested_runs_per_case": 2,
        "executed_here_attempts": sum(len(c["repeats"]) for c in cases if c["origin"] == "executed_here"),
        "reused_case_count": len(reused), "pilot_reference": pilot_reference,
        "all_native_controls_repeated": all(c["status"] == "CONTROL_COMPLETE" for c in cases),
        "native_control_status_counts": {status: sum(c["status"] == status for c in cases)
            for status in ("CONTROL_COMPLETE", "REPEAT_MISMATCH", "NATIVE_FAILED", "INVALID_OUTPUT")},
        "source_directory": str(source.resolve()), "native_source_files": SOURCE_HASHES,
        "original_input_sha256": _sha(original_input), "private_runtime": _identity(private_runtime),
        "protected_sha256_before": {str(p.resolve()): sha for p, sha in pins.items()},
        "protected_sha256_after": {str(p.resolve()): _sha(p.read_bytes()) for p in pins},
        "diagnostic_status": "INCOMPLETE", "normative_export_allowed": False,
        "production_tables_replaced": False, "isotope_policy_qualified": False,
        "source_energy_domain_covered": False, "physical_accuracy_verified": False,
        "limitations": [
            "Literal OST source A is a numeric diagnostic parameter, not a selected isotope distribution.",
            "Complete roster and native execution do not establish complete source energy coverage.",
            "Fixed 1..12000 MeV total-energy controls, not a replacement production grid.",
            "Projected native range is not imported as CSDA; stopping components remain separate.",
            "Package TRIM high-energy caution is not a verified SRModule physical uncertainty bound.",
            "Only normal coefficients and private runtime copies; no installation or special list mode.",
            "Native software copies remain private working evidence, not production distribution assets.",
            "Failed outputs remain explicit and never become zero-valued particle contributions.",
        ],
    }
    _write_new(output / "report.json", _json(report).encode("utf-8"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--source", type=Path,
                        default=Path(r"C:\Users\Иван\Desktop\ПО\SRIM-2013\SR Module"))
    parser.add_argument("--private-runtime", type=Path,
                        default=ROOT / "outputs/srim_runtime_dependencies_20261101/MSVBVM50.dll")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--pilot-sha256")
    args = parser.parse_args()
    report = run_campaign(mode=args.mode, source=args.source, private_runtime=args.private_runtime,
                          output=args.output, pilot_report=args.pilot_report,
                          pilot_sha256=args.pilot_sha256)
    print(_json({"report": str(args.output / "report.json"),
                 "native_control_status_counts": report["native_control_status_counts"],
                 "normative_export_allowed": False}))
    return 0 if report["all_native_controls_repeated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
