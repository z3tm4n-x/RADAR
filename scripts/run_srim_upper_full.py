"""Bounded 84-ion whole-table upper extension; never physical qualification.

Reuse eight authenticated pilot controls and execute the other 160 twice.
No boundary search, model splice, extrapolation or production replacement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
import io
import json
from pathlib import Path
import sys
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts import run_srim_upper_pilot as pilot  # noqa: E402
from scripts import run_srim_low_energy as low  # noqa: E402
from scripts import run_srim_controls as launcher  # noqa: E402
from scripts.run_srim_coverage import _retained_command  # noqa: E402
from radar.shielding.srim_dataset import POLICIES, SCHEMA, SrimHzeDataset, load_srim_hze_dataset  # noqa: E402
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output  # noqa: E402
from radar.validation.srim_source_catalog import build_source_catalog, MANIFEST_PATH  # noqa: E402

PILOT_REPORT_SHA256 = "5955844fd96615eddbd251ddf1748d12e2a7614bd777c923e8307d46e9e1ce3c"
PILOT_BUNDLE_SHA256 = "aef824e81874d71592b6f630aa30fe62779dae1fd7c0773f76fa0727702b90c6"
PILOT_CODE_SHA256 = "4d6780d1be3ca0d4e3872228dabadfcb9be146e3806b1c29a7f0848a1c67dc51"
HELPER_SHA256 = {
    "scripts/run_srim_low_energy.py": "3bde501f5f239be67285e597080c170840688c958bf9736858972521c8b2403f",
    "scripts/run_srim_controls.py": "fb23dc4d16788498f6bb1c6ff655524bcba0b3972c57a522bc5bfae20e7b97e5",
    "scripts/run_srim_coverage.py": "7544ac2a0611f114e87561e984588ba34043f347cdd4fdc534600770994523c4",
}
FLAGS = pilot.FLAGS
_sha, _identity, _write_new, _json = launcher._sha, launcher._identity, launcher._write_new, launcher._json
_execute = launcher._execute


@dataclass(frozen=True)
class FullUpperControl:
    name: str
    ion_z: int
    ion_mass_u: Decimal
    target_z: int
    target_mass_u: Decimal
    density_g_cm3: Decimal
    energy_max_mev: Decimal


def endpoint(mass: Decimal) -> Decimal:
    if type(mass) is not Decimal or not mass.is_finite() or not 4 <= mass <= 238:
        raise ValueError("Source mass is outside the fixed heavy-ion domain.")
    with localcontext() as context:
        context.prec = 80
        return Decimal(int(pilot.normal_grid(mass * pilot.CEILING_MEV_PER_SOURCE_A)[-1]))


def controls(dataset: SrimHzeDataset) -> tuple[tuple[FullUpperControl, SrimRangeOutput], ...]:
    catalog = build_source_catalog(ROOT)
    expected = {(base.ion_z, base.target_z) for base in catalog}
    if (len(catalog) != 168 or len(expected) != 168 or len(dataset.tables_by_z) != 84
            or set(dataset.tables_by_z) != {base.ion_z for base in catalog}):
        raise ValueError("Full fixed roster requires exactly 84 Al/Si pairs.")
    result = []
    for base in catalog:
        pair = dataset.tables_by_z[base.ion_z]
        native = pair.native_al if base.target_z == 13 else pair.native_si
        maximum = endpoint(base.ion_mass_u)
        if (native.ion_z != base.ion_z or native.ion_mass_u != base.ion_mass_u
                or native.target_z != base.target_z or native.density_g_cm3 != base.density_g_cm3
                or native.rows[0].energy_total_mev != Decimal("0.00000999999")
                or native.rows[-1].energy_total_mev >= maximum):
            raise ValueError("Parent differs from the fixed literal source/material/whole table.")
        case = FullUpperControl(f"{base.name}_upper{maximum:f}", base.ion_z, base.ion_mass_u,
                                base.target_z, base.target_mass_u, base.density_g_cm3, maximum)
        result.append((case, native))
    return tuple(result)


def input_bytes(case: FullUpperControl, allowed: tuple[FullUpperControl, ...]) -> bytes:
    if type(case) is not FullUpperControl or case not in allowed:
        raise ValueError("Input must belong to the authenticated fixed source catalog.")
    if case.energy_max_mev != endpoint(case.ion_mass_u):
        raise ValueError("Upper endpoint differs from the fixed engineering ceiling.")
    target = "Aluminum" if case.target_z == 13 else "Silicon"
    lines = (
        "---Stopping/Range Input Data (Number-format: Period = Decimal Point)",
        "---Output File Name", '"srim_output.txt"', "---Ion(Z), Ion Mass(u)",
        f"{case.ion_z}   {case.ion_mass_u}",
        "---Target Data: (Solid=0,Gas=1), Density(g/cm3), Compound Corr.",
        f"0    {case.density_g_cm3}    1", "---Number of Target Elements", "1",
        "---Target Elements: (Z), Target name, Stoich, Target Mass(u)",
        f'{case.target_z}   "{target}"   1   {case.target_mass_u}',
        "---Output Stopping Units (1-8)", "5", "---Ion Energy : E-Min(keV), E-Max(keV)",
        f"0.01    {int(case.energy_max_mev) * 1000}",
    )
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def validate_native(native: SrimRangeOutput, baseline: SrimRangeOutput,
                    case: FullUpperControl) -> dict[str, Any]:
    for key in ("ion_z", "ion_name", "ion_mass_u", "target_z", "target_symbol", "density_g_cm3",
                "atomic_density_cm3", "bragg_correction_percent", "engine_label", "version_label",
                "conversion_factors", "output_name"):
        if getattr(native, key) != getattr(baseline, key):
            raise ValueError(f"Native identity differs from parent: {key}")
    if (native.ion_z != case.ion_z or native.ion_mass_u != case.ion_mass_u
            or native.target_z != case.target_z or native.density_g_cm3 != case.density_g_cm3
            or native.engine_label != "SRIM-2006" or native.version_label != "SRIM-2012.01"
            or native.bragg_correction_percent != 0 or native.output_name != "srim_output.txt"
            or case.energy_max_mev != endpoint(case.ion_mass_u)):
        raise ValueError("Native output differs from fixed full input.")
    count = len(baseline.rows)
    if len(native.rows) <= count or native.rows[:count] != baseline.rows:
        raise ValueError("Every parent row and original token must be preserved exactly.")
    expected = tuple(r.energy_total_mev for r in baseline.rows) + tuple(
        e for e in pilot.normal_grid(case.energy_max_mev) if e > baseline.rows[-1].energy_total_mev)
    if tuple(r.energy_total_mev for r in native.rows) != expected:
        raise ValueError("Upper extension lacks the complete fixed normal-energy grid.")
    with localcontext() as context:
        context.prec = 28  # Match the authenticated pilot's decimal report representation.
        ratio = native.rows[-1].energy_total_mev/native.ion_mass_u
    return {"native_rows": len(native.rows), "parent_rows": count,
            "new_upper_rows": len(native.rows)-count, "parent_prefix_identical": True,
            "observed_minimum_total_ion_energy_mev": native.rows[0].energy_total_mev,
            "observed_maximum_total_ion_energy_mev": native.rows[-1].energy_total_mev,
            "observed_maximum_mev_per_source_a": ratio}


def _read_identity(record: dict[str, Any], expected: Path, pins: dict[str, str]) -> bytes:
    path = Path(record["path"]).resolve()
    if path != expected.resolve():
        raise ValueError("Pilot artifact escaped its fixed retained case directory.")
    raw = path.read_bytes()
    if _sha(raw) != record["sha256"] or len(raw) != record["bytes"]:
        raise ValueError("Retained pilot artifact fingerprint mismatch.")
    pins[str(path)] = _sha(raw)
    return raw


def authenticate_pilot(report_path: Path, bundle_path: Path,
                       selected: tuple[tuple[FullUpperControl, SrimRangeOutput], ...],
                       pins: dict[str, str]) -> dict[str, dict[str, Any]]:
    if _sha(Path(pilot.__file__).read_bytes()) != PILOT_CODE_SHA256:
        raise ValueError("Reviewed pilot implementation changed.")
    raw = report_path.read_bytes()
    if _sha(raw) != PILOT_REPORT_SHA256:
        raise ValueError("Reviewed pilot report fingerprint mismatch.")
    report = json.loads(raw)
    dataset = load_srim_hze_dataset(bundle_path, expected_sha256=PILOT_BUNDLE_SHA256)
    pins.update(low._pins([report_path, bundle_path, Path(pilot.__file__)]))
    if (report["schema_version"] != "radar.srim_upper_pilot.v1" or report["scope"] != "pilot"
            or report["all_native_controls_repeated"] is not True or report["not_attempted_cases"]
            or len(report["cases"]) != 8 or report["runs_per_case"] != 2
            or report["baseline_bundle"]["sha256"] != pilot.BASELINE_SHA256
            or report["protected_sha256_before"] != report["protected_sha256_after"]
            or any(report[key] != value for key, value in FLAGS.items())):
        raise ValueError("Pilot report does not establish the fixed completed pilot.")
    # A completed pilot must use the same current calculation code/source inputs.
    low._verify(report["protected_sha256_before"])
    pins.update(report["protected_sha256_before"])
    wanted = {c.name: (c, n) for c, n in selected if c.ion_z in pilot.PILOT_Z}
    if len(wanted) != 8 or tuple(dataset.tables_by_z) != pilot.PILOT_Z:
        raise ValueError("Pilot table inventory mismatch.")
    reused = {}
    for record in report["cases"]:
        name = record["case"]["name"]
        if name not in wanted or name in reused:
            raise ValueError("Duplicate or unrelated pilot case.")
        case, baseline = wanted[name]
        if (_json(record["case"]) != _json(asdict(case))
                or record["baseline_raw_sha256"] != baseline.raw_sha256
                or record["status"] != "CONTROL_COMPLETE" or len(record["repeats"]) != 2
                or any(record[key] != value for key, value in FLAGS.items())):
            raise ValueError("Pilot input/qualification differs from the full catalog.")
        identities = []
        for number, repeat in enumerate(record["repeats"], 1):
            work = report_path.parent.resolve() / f"{name}_r{number}"
            if (repeat["repeat"] != number or repeat["status"] != "VALID_NATIVE_OUTPUT"
                    or any(repeat[key] != value for key, value in FLAGS.items())):
                raise ValueError("Pilot repetition sequence or release flags differ.")
            if _read_identity(repeat["input"], work / "SR.IN", pins) != input_bytes(case, (case,)):
                raise ValueError("Pilot input bytes changed.")
            _read_identity(repeat["command_file"], work / "command.json", pins)
            command = _retained_command(work)
            if command != repeat["command"] or command["timed_out"] or command["returncode"] != 0:
                raise ValueError("Pilot retained command is not successful.")
            native = parse_srim_range_output(_read_identity(repeat["output"], work / "srim_output.txt", pins))
            comparison = validate_native(native, baseline, case)
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
                raise ValueError("Pilot bundle differs from repeated output.")
            identities.append((native.raw_sha256, native.semantic_sha256))
        if identities[0] != identities[1]:
            raise ValueError("Pilot repeated outputs differ.")
        reused[name] = {**record, "case": asdict(case), "origin": "reviewed_pilot_reused"}
    if set(reused) != set(wanted):
        raise ValueError("Pilot is incomplete.")
    return reused


def run_case(case: FullUpperControl, baseline: SrimRangeOutput, *, output: Path,
             input_raw: bytes, originals: dict[str, bytes], runtime: bytes,
             pins: dict[str, str]) -> dict[str, Any]:
    repeats = []
    for number in (1, 2):
        work = output / f"{case.name}_r{number}"
        work.mkdir()
        copied = {**originals, "MSVBVM50.dll": runtime, "SR.IN": input_raw}
        for name, raw in copied.items():
            _write_new(work / name, raw)
        record: dict[str, Any] = {"repeat": number, "input": _identity(work / "SR.IN"), **FLAGS}
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
                if command != returned or command["timed_out"] or command["returncode"] != 0:
                    raise ValueError("Returned native command disagrees with retained success.")
                record["command"] = command
                try:
                    native = parse_srim_range_output((work / "srim_output.txt").read_bytes())
                    comparison = validate_native(native, baseline, case)
                except (ValueError, FileNotFoundError) as error:
                    record.update(status="INVALID_OUTPUT", reason=str(error))
                else:
                    record.update(status="VALID_NATIVE_OUTPUT", comparison=comparison,
                                  semantic_sha256=native.semantic_sha256)
            record["command_file"] = _identity(work / "command.json")
            record["output"] = (_identity(work / "srim_output.txt")
                                if (work / "srim_output.txt").exists() else None)
        finally:
            low._verify(pins)
            if any((work / name).read_bytes() != raw for name, raw in copied.items()):
                raise ValueError("Native execution changed its copied input or runtime.")
        _write_new(work / "result.json", _json(record).encode("utf-8"))
        repeats.append(record)
        print(f"{case.name} repeat {number}: {record['status']}", flush=True)
        if record["status"] != "VALID_NATIVE_OUTPUT":
            break
    valid = len(repeats) == 2 and all(r["status"] == "VALID_NATIVE_OUTPUT" for r in repeats)
    same = valid and repeats[0]["semantic_sha256"] == repeats[1]["semantic_sha256"]
    raw_same = valid and repeats[0]["output"]["sha256"] == repeats[1]["output"]["sha256"]
    result = {"case": asdict(case), "baseline_raw_sha256": baseline.raw_sha256,
              "repeats": repeats, "origin": "executed_here", "status": "CONTROL_COMPLETE" if same and raw_same else
              "REPEAT_MISMATCH" if valid else repeats[-1]["status"],
              "semantic_outputs_identical": same, "raw_output_bytes_identical": raw_same, **FLAGS}
    _write_new(output / f"{case.name}.json", _json(result).encode("utf-8"))
    return result


def package(report_path: Path, report: dict[str, Any], output: Path) -> dict[str, Any]:
    if report["all_native_controls_repeated"] is not True or len(report["cases"]) != 168:
        raise ValueError("Full 168-case campaign required before packaging.")
    catalog = build_source_catalog(ROOT)
    expected = {(c.ion_z, c.target_z) for c in catalog}
    if {(r["case"]["ion_z"], r["case"]["target_z"]) for r in report["cases"]} != expected:
        raise ValueError("Packaged table roster differs from the authenticated source catalog.")
    members: dict[str, bytes] = {}
    records = []
    for result in report["cases"]:
        if (result["status"] != "CONTROL_COMPLETE" or len(result["repeats"]) != 2
                or not result["semantic_outputs_identical"] or not result["raw_output_bytes_identical"]):
            raise ValueError("Failed or unrepeated controls cannot supply a complete bundle.")
        case = result["case"]
        outputs = [r["output"] for r in result["repeats"]]
        raws = [Path(o["path"]).read_bytes() for o in outputs]
        if raws[0] != raws[1] or any(_sha(raw) != o["sha256"] for raw, o in zip(raws, outputs, strict=True)):
            raise ValueError("Native repeated output changed before packaging.")
        member = f"native/{case['name']}.txt"
        if member in members:
            raise ValueError("Duplicate native table.")
        members[member] = raws[0]
        records.append({"case_name": case["name"], "z": case["ion_z"],
                        "mass_parameter_u_decimal": format(case["ion_mass_u"], "f"),
                        "material_z": case["target_z"], "member": member, "sha256": _sha(raws[0])})
    manifest = {"schema_version": SCHEMA, **POLICIES, "tables": records,
                "baseline_bundle_sha256": pilot.BASELINE_SHA256, "native_report": _identity(report_path),
                "reviewed_pilot_sha256": PILOT_REPORT_SHA256,
                "reviewed_pilot_bundle_sha256": PILOT_BUNDLE_SHA256,
                "generator_sha256": _sha(Path(__file__).read_bytes()),
                "ceiling_mev_per_literal_source_a": str(pilot.CEILING_MEV_PER_SOURCE_A),
                "ceiling_policy": "bounded_native_execution_experiment_not_physical_validity",
                "software_labels": [["SRIM-2006", "SRIM-2012.01"]],
                "limitations": ["Whole tables only; no source-tail extrapolation or model splice.",
                                "No absolute range origin or normative qualification.",
                                "An engineering execution ceiling is not an accuracy bound."]}
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
    if set(loaded.tables_by_z) != {c.ion_z for c in catalog}:
        raise ValueError("Full packaged ion inventory mismatch.")
    return _identity(output)


def run(*, source: Path, private_runtime: Path, output: Path, bundle: Path,
        pilot_report: Path, pilot_bundle: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Full output must stay in a new repository outputs subdirectory.")
    for relative, sha in HELPER_SHA256.items():
        if _sha((ROOT / relative).read_bytes()) != sha:
            raise ValueError("Reviewed helper implementation changed.")
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
                 Path(__file__), Path(pilot.__file__), ROOT / MANIFEST_PATH,
                 *[ROOT / p for p in HELPER_SHA256], *sorted((ROOT / "src/radar").rglob("*.py")),
                 *sorted(p for p in (ROOT / "src/radar/data/normative").rglob("*") if p.is_file())]
    pins = low._pins(protected)
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
        print(f"Full upper campaign: {index}/168 ({result['status']})", flush=True)
        if result["status"] != "CONTROL_COMPLETE":
            break
    low._verify(pins)
    complete = len(cases) == 168 and all(c["status"] == "CONTROL_COMPLETE" for c in cases)
    report = {"schema_version": "radar.srim_upper_full.v1", "scope": "full",
              "baseline_bundle": _identity(bundle), "reviewed_pilot": _identity(pilot_report),
              "reviewed_pilot_bundle": _identity(pilot_bundle),
              "source_directory": str(source.resolve()), "private_runtime": _identity(private_runtime),
              "planned_cases": [asdict(c) for c, _ in selected], "cases": cases,
              "not_attempted_cases": [c.name for c, _ in selected[len(cases):]],
              "runs_per_case": 2, "all_native_controls_repeated": complete,
              "reused_pilot_cases": sum(c["origin"] == "reviewed_pilot_reused" for c in cases),
              "new_native_cases": sum(c["origin"] == "executed_here" for c in cases),
              "ceiling_mev_per_literal_source_a": pilot.CEILING_MEV_PER_SOURCE_A,
              "ceiling_policy": "bounded_native_execution_experiment_not_physical_validity",
              "protected_sha256_before": pins, "protected_sha256_after": low._pins([Path(p) for p in pins]),
              "limitations": ["Failure stops the campaign; no automatic boundary search.",
                              "Observed high-energy outputs do not establish physical accuracy.",
                              "Upper source tails and low-energy range origin remain unresolved."], **FLAGS}
    report_path = output / "report.json"
    _write_new(report_path, _json(report).encode("utf-8"))
    if complete:
        identity = package(report_path, report, output / "observed_upper_full.zip")
        _write_new(output / "bundle_identity.json", _json(identity).encode("utf-8"))
    low._verify(pins)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"C:\Users\Иван\Desktop\ПО\SRIM-2013\SR Module"))
    parser.add_argument("--private-runtime", type=Path, default=ROOT /
                        "outputs/srim_source_extension_pilot_20261101/u238_si_upper225000_r2/MSVBVM50.dll")
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs/srim_lower_full_20261105/observed_full.zip")
    parser.add_argument("--pilot-report", type=Path, default=ROOT / "outputs/srim_upper_pilot_20261105/report.json")
    parser.add_argument("--pilot-bundle", type=Path, default=ROOT /
                        "outputs/srim_upper_pilot_20261105/observed_upper_pilot.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(source=args.source, private_runtime=args.private_runtime, output=args.output,
                 bundle=args.bundle, pilot_report=args.pilot_report, pilot_bundle=args.pilot_bundle)
    print(json.dumps({"output": str(args.output), "all_native_controls_repeated":
                      report["all_native_controls_repeated"], "normative_export_allowed": False}))
    return 0 if report["all_native_controls_repeated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
