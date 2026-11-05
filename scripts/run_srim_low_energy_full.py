"""Full, isolated lower-energy SRIM campaign from the reviewed six-case pilot.

84 fixed OST ions in Al/Si. Reuse six authenticated whole pilot tables; run
the other 162 cases twice. No interpolation/extrapolation, installers, system
changes or production-table replacement. A failed control stops the campaign.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_srim_controls as launcher  # noqa: E402
from scripts import run_srim_low_energy as pilot  # noqa: E402
from scripts.run_srim_coverage import _retained_command  # noqa: E402
from radar.shielding.srim_dataset import (  # noqa: E402
    POLICIES, SCHEMA, SrimHzeDataset, load_srim_hze_dataset,
)
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output  # noqa: E402
from radar.validation.srim_source_catalog import build_source_catalog  # noqa: E402

PILOT_REPORT_SHA256 = "b280096db02201fee59bc0dc9e7ba68daaa30cbd823158790fd80b45fddc9f2c"
PILOT_BUNDLE_SHA256 = "8ec49fef7e5c36fb7b2ea2acfad70fa6acab100588d6d1bc3a5ebb216347b078"
PILOT_CODE_SHA256 = "3bde501f5f239be67285e597080c170840688c958bf9736858972521c8b2403f"
_sha, _identity, _write_new, _json = launcher._sha, launcher._identity, launcher._write_new, launcher._json
_execute = launcher._execute
FLAGS = pilot._FLAGS


def controls(dataset: SrimHzeDataset) -> tuple[tuple[pilot.LowControl, SrimRangeOutput], ...]:
    result = []
    catalog = build_source_catalog(ROOT)
    for base in catalog:
        table = dataset.tables_by_z[base.ion_z]
        native = table.native_al if base.target_z == 13 else table.native_si
        if (native.ion_mass_u != base.ion_mass_u or native.density_g_cm3 != base.density_g_cm3
                or native.target_z != base.target_z or native.rows[0].energy_total_mev != 1):
            raise ValueError("Full control differs from literal source/material/energy baseline.")
        maximum = native.rows[-1].energy_total_mev
        pilot.expected_grid(maximum)
        result.append((pilot.LowControl(f"{base.name}_low10ev", base.ion_z, base.ion_mass_u,
                                      base.target_z, base.target_mass_u, base.density_g_cm3,
                                      maximum), native))
    identities = {(c.ion_z, c.target_z) for c, _ in result}
    if (len(result) != 168 or len(identities) != 168
            or set(dataset.tables_by_z) != {c.ion_z for c, _ in result}
            or len(dataset.tables_by_z) != 84):
        raise ValueError("Full fixed roster requires exactly 84 Al/Si pairs.")
    return tuple(result)


def input_bytes(case: pilot.LowControl, allowed: tuple[pilot.LowControl, ...]) -> bytes:
    if type(case) is not pilot.LowControl or case not in allowed:
        raise ValueError("Input must belong to the validated full fixed catalog.")
    pilot.expected_grid(case.energy_max_mev)
    target = "Aluminum" if case.target_z == 13 else "Silicon"
    lines = (
        "---Stopping/Range Input Data (Number-format: Period = Decimal Point)",
        "---Output File Name", '"srim_output.txt"', "---Ion(Z), Ion Mass(u)",
        f"{case.ion_z}   {case.ion_mass_u}",
        "---Target Data: (Solid=0,Gas=1), Density(g/cm3), Compound Corr.",
        f"0    {case.density_g_cm3}    1", "---Number of Target Elements", "1",
        "---Target Elements: (Z), Target name, Stoich, Target Mass(u)",
        f'{case.target_z}   "{target}"   1   {case.target_mass_u}',
        "---Output Stopping Units (1-8)", "5",
        "---Ion Energy : E-Min(keV), E-Max(keV)",
        f"0.01    {case.energy_max_mev * 1000}",
    )
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def _read_identity(record: dict[str, Any], expected: Path, pins: dict[str, str]) -> bytes:
    path = Path(record["path"]).resolve()
    if path != expected.resolve():
        raise ValueError("Retained pilot artifact path differs from its fixed case directory.")
    raw = path.read_bytes()
    if _sha(raw) != record["sha256"] or len(raw) != record["bytes"]:
        raise ValueError("Retained pilot artifact fingerprint mismatch.")
    pins[str(path)] = _sha(raw)
    return raw


def authenticate_pilot(report_path: Path, bundle_path: Path,
                       selected: tuple[tuple[pilot.LowControl, SrimRangeOutput], ...],
                       pins: dict[str, str]) -> dict[str, dict[str, Any]]:
    if _sha(Path(pilot.__file__).read_bytes()) != PILOT_CODE_SHA256:
        raise ValueError("Reviewed pilot implementation changed.")
    raw = report_path.read_bytes()
    if _sha(raw) != PILOT_REPORT_SHA256:
        raise ValueError("Reviewed pilot report fingerprint mismatch.")
    report = json.loads(raw)
    dataset = load_srim_hze_dataset(bundle_path, expected_sha256=PILOT_BUNDLE_SHA256)
    pins.update(pilot._pins([report_path, bundle_path, Path(pilot.__file__)]))
    if (report["schema_version"] != "radar.srim_low_energy_pilot.v1"
            or report["scope"] != "pilot" or report["all_native_controls_repeated"] is not True
            or report["not_attempted_cases"] or len(report["cases"]) != 6
            or report["baseline_bundle"]["sha256"] != pilot.BASELINE_SHA256
            or report["protected_sha256_before"] != report["protected_sha256_after"]
            or any(report[key] != value for key, value in FLAGS.items())):
        raise ValueError("Pilot report does not establish the fixed completed pilot.")
    wanted = {c.name: (c, n) for c, n in selected if c.ion_z in pilot.PILOT_Z}
    if len(wanted) != 6 or tuple(dataset.tables_by_z) != pilot.PILOT_Z:
        raise ValueError("Pilot table inventory mismatch.")
    reused = {}
    for record in report["cases"]:
        name = record["case"]["name"]
        if name not in wanted or name in reused:
            raise ValueError("Duplicate or unrelated pilot case.")
        case, baseline = wanted[name]
        if (_json(record["case"]) != _json(asdict(case))
                or record["baseline_raw_sha256"] != baseline.raw_sha256
                or record["status"] != "CONTROL_COMPLETE"
                or len(record["repeats"]) != 2
                or any(record[key] != value for key, value in FLAGS.items())):
            raise ValueError("Pilot input/qualification differs from the full catalog.")
        identities = []
        for number, repeat in enumerate(record["repeats"], 1):
            work = report_path.parent.resolve() / f"{name}_r{number}"
            if repeat["repeat"] != number or repeat["status"] != "VALID_NATIVE_OUTPUT":
                raise ValueError("Pilot repetition sequence is incomplete.")
            if any(repeat[key] != value for key, value in FLAGS.items()):
                raise ValueError("Pilot repetition release flags differ.")
            original_input = _read_identity(repeat["input"], work / "SR.IN", pins)
            if original_input != input_bytes(case, tuple(c for c, _ in selected)):
                raise ValueError("Pilot input bytes changed.")
            _read_identity(repeat["command_file"], work / "command.json", pins)
            command = _retained_command(work)
            if command != repeat["command"] or command["timed_out"] or command["returncode"] != 0:
                raise ValueError("Pilot retained command is not successful.")
            raw_native = _read_identity(repeat["output"], work / "srim_output.txt", pins)
            native = parse_srim_range_output(raw_native)
            comparison = pilot.validate_native(native, baseline, case)
            if (native.semantic_sha256 != repeat["semantic_sha256"]
                    or _json(comparison) != _json(repeat["comparison"])):
                raise ValueError("Pilot native interpretation changed.")
            for filename, sha in {**launcher.SOURCE_HASHES,
                                  "MSVBVM50.dll": launcher.PRIVATE_RUNTIME_SHA256}.items():
                path = work / filename
                if _sha(path.read_bytes()) != sha:
                    raise ValueError("Pilot copied native source/runtime changed.")
                pins[str(path)] = sha
            pair = dataset.tables_by_z[case.ion_z]
            packaged = pair.native_al if case.target_z == 13 else pair.native_si
            if packaged.raw_sha256 != native.raw_sha256:
                raise ValueError("Pilot bundle differs from retained repeated output.")
            identities.append((native.raw_sha256, native.semantic_sha256))
        if identities[0] != identities[1]:
            raise ValueError("Pilot repeated output identities differ.")
        reused[name] = {**record, "case": asdict(case), "origin": "reviewed_pilot_reused"}
    if set(reused) != set(wanted):
        raise ValueError("Pilot is incomplete.")
    return reused


def run_case(case: pilot.LowControl, baseline: SrimRangeOutput, *, output: Path,
             input_raw: bytes, originals: dict[str, bytes], runtime: bytes,
             pins: dict[str, str]) -> dict[str, Any]:
    repeats = []
    for number in (1, 2):
        work = output / f"{case.name}_r{number}"
        work.mkdir()
        copied = {**originals, "MSVBVM50.dll": runtime, "SR.IN": input_raw}
        for name, raw in copied.items():
            _write_new(work / name, raw)
        result: dict[str, Any] = {"repeat": number, "input": _identity(work / "SR.IN"), **FLAGS}
        try:
            try:
                returned = _execute(work)
            except RuntimeError as error:
                command = _retained_command(work)
                if not command["timed_out"] and command["returncode"] == 0:
                    raise
                result.update(status="NATIVE_FAILED", reason=str(error), command=command)
            else:
                command = _retained_command(work)
                if command != returned or command["timed_out"] or command["returncode"] != 0:
                    raise ValueError("Native returned command disagrees with retained success.")
                result["command"] = command
                try:
                    native = parse_srim_range_output((work / "srim_output.txt").read_bytes())
                    comparison = pilot.validate_native(native, baseline, case)
                except (ValueError, FileNotFoundError) as error:
                    result.update(status="INVALID_OUTPUT", reason=str(error))
                else:
                    result.update(status="VALID_NATIVE_OUTPUT", comparison=comparison,
                                  semantic_sha256=native.semantic_sha256)
            result["command_file"] = _identity(work / "command.json")
            result["output"] = (_identity(work / "srim_output.txt")
                                if (work / "srim_output.txt").exists() else None)
        finally:
            pilot._verify(pins)
            if any((work / name).read_bytes() != raw for name, raw in copied.items()):
                raise ValueError("Native execution changed its copied runtime or input.")
        _write_new(work / "result.json", _json(result).encode("utf-8"))
        repeats.append(result)
        print(f"{case.name} repeat {number}: {result['status']}", flush=True)
        if result["status"] != "VALID_NATIVE_OUTPUT":
            break
    valid = len(repeats) == 2 and all(r["status"] == "VALID_NATIVE_OUTPUT" for r in repeats)
    same = valid and repeats[0]["semantic_sha256"] == repeats[1]["semantic_sha256"]
    record = {"case": asdict(case), "baseline_raw_sha256": baseline.raw_sha256,
              "origin": "executed_here", "repeats": repeats,
              "status": "CONTROL_COMPLETE" if same else "REPEAT_MISMATCH" if valid else repeats[-1]["status"],
              "semantic_outputs_identical": same, "raw_output_bytes_identical": valid and
              repeats[0]["output"]["sha256"] == repeats[1]["output"]["sha256"], **FLAGS}
    _write_new(output / f"{case.name}.json", _json(record).encode("utf-8"))
    return record


def package(report_path: Path, report: dict[str, Any], output: Path) -> dict[str, Any]:
    if report["all_native_controls_repeated"] is not True or len(report["cases"]) != 168:
        raise ValueError("Full 168-case campaign required before packaging.")
    members: dict[str, bytes] = {}
    records = []
    observed = {}
    seen = set()
    for result in report["cases"]:
        if result["status"] != "CONTROL_COMPLETE":
            raise ValueError("Cannot package failed controls.")
        case = result["case"]
        key = (case["ion_z"], case["target_z"])
        if key in seen:
            raise ValueError("Duplicate packaged ion/material.")
        seen.add(key)
        evidence = result["repeats"][0]["output"]
        raw = Path(evidence["path"]).read_bytes()
        if _sha(raw) != evidence["sha256"]:
            raise ValueError("Native output changed before packaging.")
        native = parse_srim_range_output(raw)
        member = f"native/{case['name']}.txt"
        members[member] = raw
        observed[case["name"]] = str(native.rows[0].energy_total_mev)
        records.append({"case_name": case["name"], "z": case["ion_z"],
                        "mass_parameter_u_decimal": format(case["ion_mass_u"], "f"),
                        "material_z": case["target_z"], "member": member, "sha256": _sha(raw)})
    manifest = {"schema_version": SCHEMA, **POLICIES, "tables": records,
                "baseline_bundle_sha256": pilot.BASELINE_SHA256,
                "native_report": _identity(report_path), "reviewed_pilot_sha256": PILOT_REPORT_SHA256,
                "generator_sha256": _sha(Path(__file__).read_bytes()),
                "requested_minimum_total_ion_energy_mev": str(pilot.MINIMUM_MEV),
                "observed_minimum_total_ion_energy_mev_by_table": observed,
                "absolute_range_origin_established": False,
                "software_labels": [["SRIM-2006", "SRIM-2012.01"]],
                "limitations": ["Finite observed support only; no continuation to rest.",
                                "Projected ranges are not CSDA path ranges.",
                                "No normative qualification or source-tail extrapolation."]}
    members["manifest.json"] = _json(manifest).encode("utf-8")
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(members.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw, compresslevel=9)
    raw = stream.getvalue()
    _write_new(output, raw)
    loaded = load_srim_hze_dataset(output, expected_sha256=_sha(raw))
    if len(loaded.tables_by_z) != 84:
        raise ValueError("Full packaged ion inventory mismatch.")
    return _identity(output)


def run(*, source: Path, private_runtime: Path, output: Path, bundle: Path,
        pilot_report: Path, pilot_bundle: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Full output must stay in a new repository outputs subdirectory.")
    originals = {name: (source / name).read_bytes() for name in launcher.SOURCE_HASHES}
    if any(_sha(raw) != launcher.SOURCE_HASHES[name] for name, raw in originals.items()):
        raise ValueError("Pinned normal-mode SRIM source changed.")
    runtime = private_runtime.read_bytes()
    if _sha(runtime) != launcher.PRIVATE_RUNTIME_SHA256:
        raise ValueError("Private VB runtime identity mismatch.")
    dataset = load_srim_hze_dataset(bundle, expected_sha256=pilot.BASELINE_SHA256)
    selected = controls(dataset)
    allowed = tuple(c for c, _ in selected)
    inputs = {c.name: input_bytes(c, allowed) for c, _ in selected}
    protected = [bundle, private_runtime, source / "SR.IN", *[source / n for n in originals],
                 Path(__file__), Path(pilot.__file__), ROOT / "scripts/run_srim_controls.py",
                 ROOT / "scripts/run_srim_coverage.py", *sorted((ROOT / "src/radar").rglob("*.py")),
                 *sorted(p for p in (ROOT / "src/radar/data/normative").rglob("*") if p.is_file())]
    pins = pilot._pins(protected)
    reused = authenticate_pilot(pilot_report, pilot_bundle, selected, pins)
    output.mkdir()
    cases = []
    for index, (case, baseline) in enumerate(selected, 1):
        result = reused.get(case.name)
        if result is None:
            result = run_case(case, baseline, output=output, input_raw=inputs[case.name],
                              originals=originals, runtime=runtime, pins=pins)
        else:
            _write_new(output / f"{case.name}.json", _json(result).encode("utf-8"))
        cases.append(result)
        print(f"Full campaign: {index}/168 ({result['status']})", flush=True)
        if result["status"] != "CONTROL_COMPLETE":
            break
    pilot._verify(pins)
    complete = len(cases) == 168 and all(c["status"] == "CONTROL_COMPLETE" for c in cases)
    report = {"schema_version": "radar.srim_low_energy_full.v1", "scope": "full",
              "baseline_bundle": _identity(bundle), "reviewed_pilot": _identity(pilot_report),
              "reviewed_pilot_bundle": _identity(pilot_bundle),
              "source_directory": str(source.resolve()), "private_runtime": _identity(private_runtime),
              "requested_minimum_total_ion_energy_mev": pilot.MINIMUM_MEV,
              "planned_cases": [asdict(c) for c, _ in selected], "cases": cases,
              "not_attempted_cases": [c.name for c, _ in selected[len(cases):]],
              "runs_per_case": 2, "all_native_controls_repeated": complete,
              "reused_pilot_cases": sum(c["origin"] == "reviewed_pilot_reused" for c in cases),
              "new_native_cases": sum(c["origin"] == "executed_here" for c in cases),
              "protected_sha256_before": pins,
              "protected_sha256_after": {name: _sha(Path(name).read_bytes()) for name in pins},
              "limitations": ["Whole native tables only, no stitched/extrapolated rows.",
                              "Positive lower endpoint does not establish stopping at rest.",
                              "Output integrity is not physical accuracy or release qualification."], **FLAGS}
    report_path = output / "report.json"
    _write_new(report_path, _json(report).encode("utf-8"))
    if complete:
        identity = package(report_path, report, output / "observed_full.zip")
        _write_new(output / "bundle_identity.json", _json(identity).encode("utf-8"))
    pilot._verify(pins)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(r"C:\Users\Иван\Desktop\ПО\SRIM-2013\SR Module"))
    parser.add_argument("--private-runtime", type=Path, default=ROOT /
                        "outputs/srim_source_extension_pilot_20261101/u238_si_upper225000_r2/MSVBVM50.dll")
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs/srim_hze_observed_20261105.zip")
    parser.add_argument("--pilot-report", type=Path, default=ROOT /
                        "outputs/srim_lower_pilot_verified_20261105/report.json")
    parser.add_argument("--pilot-bundle", type=Path, default=ROOT /
                        "outputs/srim_lower_pilot_verified_20261105/observed_pilot.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(source=args.source, private_runtime=args.private_runtime, output=args.output,
                 bundle=args.bundle, pilot_report=args.pilot_report, pilot_bundle=args.pilot_bundle)
    print(json.dumps({"output": str(args.output), "all_native_controls_repeated":
                      report["all_native_controls_repeated"], "normative_export_allowed": False}))
    return 0 if report["all_native_controls_repeated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
