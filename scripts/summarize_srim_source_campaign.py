"""Read-only replay of the literal OST source-mass SRIM campaign.

No executable is imported or launched. Complete native controls contribute to a
source-roster audit and two explicitly diagnostic Al stopping integrals. Native
Si stopping is retained as evidence, never promoted to a production LET adapter.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.validation.srim_anchored_range import (  # noqa: E402
    AnchoredSrimRangeTable, StoppingInterpolation,
)
from radar.validation.srim_native_coverage import audit_srim_native_coverage  # noqa: E402
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output  # noqa: E402
from radar.validation.srim_source_catalog import (  # noqa: E402
    SourceControl, build_source_catalog, expected_control_grid, input_bytes, pilot_controls,
)

NATIVE_SOURCE_HASHES = {
    "SRModule.exe": "a679e55defdb12ebe39c02ca0562016b79ad73105dcaeddc9e8ba40f57c16288",
    "SCOEF03.dat": "a328eaaa04e48efed1d15cfbcdf810849115ff352feaac1ee323c3bf4229ecea",
    "SNUC03.dat": "4c7ff553b7be34a1b28a31a6ac89055b800392cbbf0401784da6afc5a693a030",
    "VERSION": "f0265c292d60ec141b8518ed688b52b977f2ec81761b66195545024579ef4c40",
    "HELP- SR Module.rtf": "5df89c445c295c4df6b38a0af03c695764db723876603164ba32d390bb5636de",
}
PRIVATE_RUNTIME_SHA256 = "8611dc1b60ae5c383bba6cb3ffd8a51aeebff23b95844f0ab3d6e5ecd0fadc84"
IMPLEMENTATION = (
    "scripts/summarize_srim_source_campaign.py",
    "src/radar/validation/srim_source_catalog.py",
    "src/radar/validation/srim_output.py",
    "src/radar/validation/srim_native_coverage.py",
    "src/radar/validation/srim_anchored_range.py",
    "src/radar/validation/range_stopping_consistency.py",
)
REQUIRED_CAMPAIGN_PINS = (
    "scripts/run_srim_source_campaign.py", "src/radar/validation/srim_source_catalog.py",
    "src/radar/validation/srim_output.py", "outputs/stopping_mass_manifest_20261031.json",
    "src/radar/data/normative/integrity_manifest.json",
    "src/radar/data/normative/stopping/proton_al_range.csv",
    "src/radar/data/normative/stopping/proton_si_let.csv",
    "src/radar/data/normative/stopping/hze_al_range.csv",
    "src/radar/data/normative/stopping/hze_si_let.csv",
    "src/radar/shielding/stopping_tables.py",
)
INTERPOLATIONS: tuple[StoppingInterpolation, ...] = ("linear_stopping", "loglog_stopping")
_STATUSES = ("CONTROL_COMPLETE", "REPEAT_MISMATCH", "NATIVE_FAILED", "INVALID_OUTPUT")
_CONTEXT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Pinned file changed or invalid SHA256 identity supplied.")
    return value


def _json(value: Any) -> str:
    def encode(item: Any) -> str:
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, default=encode) + "\n"


def _load(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant: {value}")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("A JSON object is required.")
    return value


def _same(actual: Any, expected: Any, label: str) -> None:
    # JSON representations distinguish bool/int and preserve retained Decimal text.
    if _json(actual) != _json(expected):
        raise ValueError(f"Retained {label} differs from independently reconstructed evidence.")


class _Evidence:
    def __init__(self) -> None:
        self.pins: dict[str, str] = {}

    def read(self, path: Path, expected: str | None = None) -> bytes:
        raw = path.read_bytes()
        digest = _sha(raw)
        if expected is not None and digest != _digest(expected):
            raise ValueError(f"Pinned file changed: {path}")
        key = str(path.resolve())
        if key in self.pins and self.pins[key] != digest:
            raise ValueError(f"File changed during replay: {path}")
        self.pins[key] = digest
        return raw

    def identity(self, item: Any) -> tuple[Path, bytes]:
        if (not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}
                or not isinstance(item["path"], str) or not Path(item["path"]).is_absolute()
                or type(item["bytes"]) is not int or item["bytes"] < 0):
            raise ValueError("Invalid retained file identity.")
        path = Path(item["path"]).resolve()
        raw = self.read(path, _digest(item["sha256"]))
        if len(raw) != item["bytes"]:
            raise ValueError("Retained file size mismatch.")
        return path, raw

    def verify_unchanged(self) -> None:
        for name, sha in tuple(self.pins.items()):
            self.read(Path(name), sha)


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


def _validate_native(parsed: SrimRangeOutput, case: SourceControl) -> None:
    if (parsed.ion_z != case.ion_z or parsed.ion_mass_u != case.ion_mass_u
            or parsed.target_z != case.target_z or parsed.density_g_cm3 != case.density_g_cm3
            or parsed.target_symbol != {13: "Al", 14: "Si"}[case.target_z]
            or parsed.bragg_correction_percent != 0 or parsed.engine_label != "SRIM-2006"
            or parsed.version_label != "SRIM-2012.01" or parsed.output_name != "srim_output.txt"):
        raise ValueError("Native source-control metadata mismatch; no mass rounding permitted.")
    if tuple(row.energy_total_mev for row in parsed.rows) != expected_control_grid(case):
        raise ValueError("Native source-control energy grid mismatch.")


def _command(command: dict[str, Any], work: Path) -> None:
    if (command.get("argv") != [str(work / "SRModule.exe")]
            or command.get("cwd") != str(work)
            or type(command.get("returncode")) is not int
            or type(command.get("timed_out")) is not bool
            or type(command.get("pid")) is not int or command["pid"] <= 0
            or type(command.get("elapsed_seconds")) not in (int, float)
            or not isfinite(command["elapsed_seconds"]) or command["elapsed_seconds"] < 0
            or not isinstance(command.get("stdout"), str)
            or not isinstance(command.get("stderr"), str)):
        raise ValueError("Invalid retained native command record.")


def _repeat(record: dict[str, Any], case: SourceControl, work: Path,
            evidence: _Evidence) -> SrimRangeOutput | None:
    for flag in ("physical_accuracy_verified", "normative_export_allowed"):
        if record[flag] is not False:
            raise ValueError("Native replay cannot establish physical qualification.")
    data: dict[str, bytes | None] = {}
    for key, filename in (("input", "SR.IN"), ("output", "srim_output.txt"),
                          ("command_file", "command.json")):
        if record[key] is None:
            if key != "output" or (work / filename).exists():
                raise ValueError("Missing retained artifact identity.")
            data[key] = None
        else:
            path, raw = evidence.identity(record[key])
            if path != work / filename:
                raise ValueError("Repeat artifact is outside its canonical isolated directory.")
            data[key] = raw
    if data["input"] != input_bytes(case):
        raise ValueError("Retained input differs from the fixed source control.")
    for name, sha in {**NATIVE_SOURCE_HASHES, "MSVBVM50.dll": PRIVATE_RUNTIME_SHA256}.items():
        evidence.read(work / name, sha)
    raw_command = data["command_file"]
    assert raw_command is not None
    command = _load(raw_command)
    _command(command, work)
    _same(record["command"], command, "command ledger")
    # This ledger was retained by the runner before aggregating the case.
    _same(_load(evidence.read(work / "result.json")), record, "repeat checkpoint")
    success = not command["timed_out"] and command["returncode"] == 0
    status = record["status"]
    if status not in ("VALID_NATIVE_OUTPUT", "NATIVE_FAILED", "INVALID_OUTPUT"):
        raise ValueError("Unknown repeat status.")
    if (status == "NATIVE_FAILED") == success:
        raise ValueError("Native command status contradicts the retained exit/timeout.")
    if status != "VALID_NATIVE_OUTPUT":
        if "parsed" in record or "semantic_sha256" in record:
            raise ValueError("Failed output cannot supply numerical observations.")
        if not isinstance(record.get("reason"), str) or not record["reason"]:
            raise ValueError("Failed output requires an explicit diagnostic reason.")
        if status == "NATIVE_FAILED":
            return None
    parsed = None
    try:
        if data["output"] is None:
            raise ValueError("No native output.")
        parsed = parse_srim_range_output(data["output"])
        _validate_native(parsed, case)
    except ValueError:
        if status == "INVALID_OUTPUT":
            return None
        raise
    if status == "INVALID_OUTPUT":
        raise ValueError("Retained INVALID_OUTPUT is actually a valid complete native output.")
    assert parsed is not None
    _same(record["parsed"], asdict(parsed), "parsed native output")
    _same(record["semantic_sha256"], parsed.semantic_sha256, "semantic hash")
    return parsed


def _case(record: dict[str, Any], case: SourceControl, report_path: Path,
          evidence: _Evidence) -> SrimRangeOutput | None:
    _same(record["case"], asdict(case), "fixed source control")
    _same(record["scope"], _scope(case), "scope")
    _same(record["expected_control_grid_mev"], expected_control_grid(case), "expected grid")
    checkpoint, raw = evidence.identity(record["checkpoint"])
    if checkpoint.name != f"{case.name}.json":
        raise ValueError("Case checkpoint has an unexpected name.")
    stored = _load(raw)
    excluded = {"checkpoint", "origin"}
    _same({k: v for k, v in stored.items() if k not in excluded},
          {k: v for k, v in record.items() if k not in excluded}, "case checkpoint")
    if (stored["origin"] != "executed_here"
            or record["origin"] not in ("executed_here", "pilot_reused")
            or (record["origin"] == "executed_here" and checkpoint.parent != report_path.parent)):
        raise ValueError("Unexpected case execution origin or checkpoint directory.")
    repeats = record["repeats"]
    if not isinstance(repeats, list) or not 1 <= len(repeats) <= 2:
        raise ValueError("Every attempted case requires one or two repeat records.")
    observations: list[SrimRangeOutput | None] = []
    for number, repeat in enumerate(repeats, 1):
        if type(repeat["repeat"]) is not int or repeat["repeat"] != number:
            raise ValueError("Repeat numbering must identify distinct canonical directories.")
        if observations and observations[-1] is None:
            raise ValueError("The fixed campaign stops this control after a failed repeat.")
        work = (checkpoint.parent / f"{case.name}_r{number}").resolve()
        observations.append(_repeat(repeat, case, work, evidence))
    if len(observations) == 1 and observations[0] is not None:
        raise ValueError("A successful first repeat cannot be the final case record.")
    valid = len(observations) == 2 and all(item is not None for item in observations)
    identical = valid and repeats[0]["semantic_sha256"] == repeats[1]["semantic_sha256"]
    status = ("CONTROL_COMPLETE" if identical else "REPEAT_MISMATCH" if valid
              else repeats[-1]["status"])
    expected = {"status": status, "complete_native_control_grid_verified": valid,
                "semantic_outputs_identical": identical,
                "raw_output_bytes_identical": valid and
                repeats[0]["output"]["sha256"] == repeats[1]["output"]["sha256"],
                "physical_accuracy_verified": False, "normative_export_allowed": False}
    _same({key: record[key] for key in expected}, expected, "control completion")
    return observations[0] if identical else None


def _replay(path: Path, expected_sha256: str, catalog: tuple[SourceControl, ...],
            evidence: _Evidence, *, must_be_pilot: bool = False
            ) -> tuple[dict[str, Any], tuple[SrimRangeOutput, ...]]:
    path = path.resolve()
    report = _load(evidence.read(path, _digest(expected_sha256)))
    if report["schema_version"] != "radar.srim_ost_source_campaign.v1":
        raise ValueError("Unexpected native campaign schema.")
    mode = report["mode"]
    if mode not in ("pilot", "full") or (must_be_pilot and mode != "pilot"):
        raise ValueError("Expected a fixed pilot/full campaign (no nested full reference).")
    selected = pilot_controls(catalog) if mode == "pilot" else catalog
    _same(report["expected_catalog"], [asdict(case) for case in selected], "complete catalog")
    pins = report["protected_sha256_before"]
    _same(pins, report["protected_sha256_after"], "protected before/after hashes")
    for name in REQUIRED_CAMPAIGN_PINS:
        if str((ROOT / name).resolve()) not in pins:
            raise ValueError(f"Required protected campaign dependency absent: {name}")
    for name, sha in pins.items():
        if not Path(name).is_absolute():
            raise ValueError("Campaign protection paths must be absolute.")
        evidence.read(Path(name), _digest(sha))
    _same(report["native_source_files"], NATIVE_SOURCE_HASHES, "normal native dependencies")
    source = Path(report["source_directory"])
    if not source.is_absolute():
        raise ValueError("Native source directory must be absolute.")
    for name, sha in NATIVE_SOURCE_HASHES.items():
        evidence.read(source / name, sha)
    evidence.read(source / "SR.IN", _digest(report["original_input_sha256"]))
    runtime, _ = evidence.identity(report["private_runtime"])
    evidence.read(runtime, PRIVATE_RUNTIME_SHA256)
    flags = {"diagnostic_status": "INCOMPLETE", "normative_export_allowed": False,
             "production_tables_replaced": False, "isotope_policy_qualified": False,
             "source_energy_domain_covered": False, "physical_accuracy_verified": False}
    _same({key: report[key] for key in flags}, flags, "diagnostic boundaries")
    pilot = None
    if mode == "full":
        pilot_path, _ = evidence.identity(report["pilot_reference"])
        pilot, _ = _replay(pilot_path, report["pilot_reference"]["sha256"], catalog,
                           evidence, must_be_pilot=True)
        if not pilot["all_native_controls_repeated"]:
            raise ValueError("Full campaign requires a completely replayed pilot.")
        for name, sha in pilot["protected_sha256_before"].items():
            if pins.get(name) != sha:
                raise ValueError("Pilot/campaign protection pins differ.")
    elif report["pilot_reference"] is not None:
        raise ValueError("Pilot cannot carry a prior pilot reference.")
    cases = report["cases"]
    if (not isinstance(cases, list)
            or [case["case"]["name"] for case in cases] != [case.name for case in selected]):
        raise ValueError("Campaign contains missing, reordered or duplicate controls.")
    reused = {} if pilot is None else {case["case"]["name"]: case for case in pilot["cases"]}
    observations = []
    for record, control in zip(cases, selected, strict=True):
        if control.name in reused:
            _same(record, {**reused[control.name], "origin": "pilot_reused"}, "reused pilot case")
        elif record["origin"] != "executed_here":
            raise ValueError("Only matching completed pilot cases can be reused.")
        parsed = _case(record, control, path, evidence)
        if parsed is not None:
            observations.append(parsed)
    counts = {status: sum(case["status"] == status for case in cases) for status in _STATUSES}
    totals = {"case_count": len(cases), "requested_runs_per_case": 2,
              "executed_here_attempts": sum(len(case["repeats"]) for case in cases
                                            if case["origin"] == "executed_here"),
              "reused_case_count": len(reused),
              "all_native_controls_repeated": counts["CONTROL_COMPLETE"] == len(cases),
              "native_control_status_counts": counts}
    _same({key: report[key] for key in totals}, totals, "campaign counters")
    return report, tuple(observations)


def summarize_campaign(report_path: Path, *, expected_report_sha256: str) -> dict[str, Any]:
    """Authenticate saved evidence and derive diagnostics, never execute SRIM."""
    evidence = _Evidence()
    implementation = {name: _sha(evidence.read(ROOT / name)) for name in IMPLEMENTATION}
    catalog = build_source_catalog(ROOT)
    report, observations = _replay(report_path, expected_report_sha256, catalog, evidence)
    audit = audit_srim_native_coverage(observations)
    coverage = asdict(audit)
    candidates = []
    for native in observations:
        if native.target_z == 13:
            for interpolation in INTERPOLATIONS:
                table = AnchoredSrimRangeTable.from_native(native, interpolation=interpolation)
                candidates.append(asdict(table))
    complete_roster = all(summary.missing_positive_species == 0
                          and summary.positive_species_missing_one_material == 0
                          and summary.positive_species_with_any_numeric_mass_difference == 0
                          for summary in audit.summaries)
    result = {
        "schema_version": "radar.srim_ost_source_summary.v1",
        "campaign_report": {"path": str(report_path.resolve()),
                            "sha256": expected_report_sha256},
        "mode": report["mode"], "native_control_status_counts": report["native_control_status_counts"],
        "verified_complete_native_control_count": len(observations),
        "all_requested_native_controls_repeated": report["all_native_controls_repeated"],
        "observed_positive_roster_complete": complete_roster,
        "source_native_coverage": coverage,
        "unavailable_controls": [{"case": case["case"], "status": case["status"],
                                  "checkpoint": case["checkpoint"]}
                                 for case in report["cases"] if case["status"] != "CONTROL_COMPLETE"],
        "al_candidate_count": len(candidates), "al_interpolations": INTERPOLATIONS,
        "al_diagnostic_candidates": candidates,
        "silicon_complete_native_control_count": sum(n.target_z == 14 for n in observations),
        "silicon_production_let_adapter_created": False,
        "diagnostic_status": "INCOMPLETE", "normative_export_allowed": False,
        "production_tables_replaced": False, "source_energy_domain_covered": False,
        "physical_accuracy_verified": False, "isotope_policy_qualified": False,
        "source_spectrum_convolution_performed": False,
        "native_execution_performed_by_summary": False,
        "implementation_sha256": implementation,
        "limitations": [
            "Literal source A equality is numerical observation, not a qualified isotope policy.",
            "Roster completeness is distinct from energy-domain coverage and physical validation.",
            "Only twice-validated matching controls contribute observations; failures are not zeros.",
            "Declared zero source components are retained separately by the source-coverage audit.",
            "Al candidates integrate electronic plus elastic nuclear stopping, not projected range.",
            "The first energy is a coordinate anchor, not a zero-energy or physical stopping origin.",
            "Both stopping interpolants are diagnostic; neither is selected for production.",
            "Si electronic/nuclear stopping remains native evidence; no production LET adapter.",
        ],
    }
    # Reconstruct the pinned roster again and re-read every artifact after derivation.
    _same([asdict(c) for c in build_source_catalog(ROOT)], [asdict(c) for c in catalog],
          "source catalog after replay")
    before = evidence.pins.copy()
    evidence.verify_unchanged()
    result["evidence_sha256_before"] = before
    result["evidence_sha256_after"] = evidence.pins.copy()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = summarize_campaign(args.campaign_report, expected_report_sha256=args.report_sha256)
    with args.output.open("xb") as stream:
        stream.write(_json(report).encode("utf-8"))
    print(_json({"output": str(args.output), "al_candidate_count": report["al_candidate_count"],
                 "observed_positive_roster_complete": report["observed_positive_roster_complete"],
                 "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
