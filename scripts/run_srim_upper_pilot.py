"""Bounded whole-table upper-energy SRIM pilot; no model splice or release claim.

He/Ne/Fe/U in Al/Si, retained 10 eV requested lower endpoint and largest normal
node <=5000 MeV per literal source A. This is an execution experiment, not a
validated physical interval. A failure stops the eight-case campaign.
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

from scripts import run_srim_controls as launcher  # noqa: E402
from scripts import run_srim_low_energy as low  # noqa: E402
from scripts.run_srim_coverage import _retained_command  # noqa: E402
from radar.shielding.srim_dataset import POLICIES, SCHEMA, SrimHzeDataset, load_srim_hze_dataset  # noqa: E402
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output  # noqa: E402
from radar.validation.srim_source_catalog import build_source_catalog  # noqa: E402

BASELINE_SHA256 = "20c7e5686e61d14200dff807627009e0859f560c2adfdc34e37bc0f0d3314607"
PILOT_Z = (2, 10, 26, 92)
CEILING_MEV_PER_SOURCE_A = Decimal(5000)
MAXIMUM_INPUT_KEV = Decimal(2147483647)
_MASSES = {2: Decimal(4), 10: Decimal("20.2"), 26: Decimal("55.8"), 92: Decimal(238)}
_NAMES = {2: "he4", 10: "ne20p2", 26: "fe55p8", 92: "u238"}
_TARGETS = {13: ("al", Decimal("26.981539"), Decimal("2.70")),
            14: ("si", Decimal("28.0855"), Decimal("2.3290"))}
FLAGS = {"diagnostic_status": "INCOMPLETE", "normative_export_allowed": False,
         "physical_accuracy_verified": False, "production_tables_replaced": False,
         "source_energy_domain_covered": False, "absolute_range_origin_established": False}
_sha, _identity, _json, _write_new = launcher._sha, launcher._identity, launcher._json, launcher._write_new
_execute = launcher._execute


def normal_grid(maximum: Decimal) -> tuple[Decimal, ...]:
    if (type(maximum) is not Decimal or not maximum.is_finite() or maximum <= 0
            or maximum * 1000 > MAXIMUM_INPUT_KEV):
        raise ValueError("Upper energy exceeds the bounded native input domain.")
    mantissas = tuple(v for v in low.expected_grid(Decimal(10)) if 1 <= v < 10)
    with localcontext() as context:
        context.prec = 80
        return tuple(e for exponent in range(-5, 7) for m in mantissas
                     if (e := m * Decimal(10) ** exponent) <= maximum)


def endpoint(mass: Decimal) -> Decimal:
    if type(mass) is not Decimal or mass not in _MASSES.values():
        raise ValueError("Only the four fixed literal pilot masses are allowed.")
    ceiling = mass * CEILING_MEV_PER_SOURCE_A
    return Decimal(int(normal_grid(ceiling)[-1]))


@dataclass(frozen=True)
class UpperControl:
    name: str
    ion_z: int
    ion_mass_u: Decimal
    target_z: int
    target_mass_u: Decimal
    density_g_cm3: Decimal
    energy_max_mev: Decimal

    def __post_init__(self) -> None:
        if type(self.ion_z) is not int or self.ion_z not in PILOT_Z or type(self.target_z) is not int:
            raise ValueError("Only fixed pilot ion/material identities are allowed.")
        if self.target_z not in _TARGETS:
            raise ValueError("Only Al and Si targets are allowed.")
        suffix, mass, density = _TARGETS[self.target_z]
        if (any(type(v) is not Decimal or not v.is_finite() for v in (
                self.ion_mass_u, self.target_mass_u, self.density_g_cm3, self.energy_max_mev))
                or self.ion_mass_u != _MASSES[self.ion_z]
                or self.target_mass_u != mass or self.density_g_cm3 != density
                or self.energy_max_mev != endpoint(self.ion_mass_u)
                or self.name != f"{_NAMES[self.ion_z]}_{suffix}_upper{self.energy_max_mev:f}"):
            raise ValueError("Pilot input differs from the fixed literal-mass normal-grid control.")


def controls(dataset: SrimHzeDataset) -> tuple[tuple[UpperControl, SrimRangeOutput], ...]:
    result = []
    for base in build_source_catalog(ROOT):
        if base.ion_z not in PILOT_Z:
            continue
        table = dataset.tables_by_z[base.ion_z]
        native = table.native_al if base.target_z == 13 else table.native_si
        maximum = endpoint(base.ion_mass_u)
        if (native.ion_mass_u != base.ion_mass_u or native.target_z != base.target_z
                or native.density_g_cm3 != base.density_g_cm3
                or native.rows[0].energy_total_mev != Decimal("0.00000999999")
                or native.rows[-1].energy_total_mev >= maximum):
            raise ValueError("Parent table does not match the fixed whole-table extension.")
        case = UpperControl(f"{base.name}_upper{maximum:f}", base.ion_z, base.ion_mass_u,
                            base.target_z, base.target_mass_u, base.density_g_cm3, maximum)
        result.append((case, native))
    if (len(result) != 8 or {(c.ion_z, c.target_z) for c, _ in result}
            != {(z, t) for z in PILOT_Z for t in (13, 14)}):
        raise ValueError("Expected exactly eight pilot cases.")
    return tuple(result)


def input_bytes(case: UpperControl) -> bytes:
    case.__post_init__()
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
        f"0.01    {case.energy_max_mev * 1000}",
    )
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def validate_native(native: SrimRangeOutput, baseline: SrimRangeOutput,
                    case: UpperControl) -> dict[str, Any]:
    case.__post_init__()
    for key in ("ion_z", "ion_name", "ion_mass_u", "target_z", "target_symbol", "density_g_cm3",
                "atomic_density_cm3", "bragg_correction_percent", "engine_label", "version_label",
                "conversion_factors", "output_name"):
        if getattr(native, key) != getattr(baseline, key):
            raise ValueError(f"Native identity differs from baseline: {key}")
    if (native.ion_z != case.ion_z or native.ion_mass_u != case.ion_mass_u
            or native.target_z != case.target_z or native.density_g_cm3 != case.density_g_cm3
            or native.engine_label != "SRIM-2006" or native.version_label != "SRIM-2012.01"
            or native.bragg_correction_percent != 0 or native.output_name != "srim_output.txt"):
        raise ValueError("Native output differs from fixed pilot input.")
    count = len(baseline.rows)
    if len(native.rows) <= count or native.rows[:count] != baseline.rows:
        raise ValueError("Every parent row and original token must be preserved exactly.")
    expected = tuple(row.energy_total_mev for row in baseline.rows) + tuple(
        e for e in normal_grid(case.energy_max_mev) if e > baseline.rows[-1].energy_total_mev)
    if tuple(row.energy_total_mev for row in native.rows) != expected:
        raise ValueError("Upper extension lacks the complete fixed normal-energy grid.")
    return {"native_rows": len(native.rows), "parent_rows": count,
            "new_upper_rows": len(native.rows) - count, "parent_prefix_identical": True,
            "observed_minimum_total_ion_energy_mev": native.rows[0].energy_total_mev,
            "observed_maximum_total_ion_energy_mev": native.rows[-1].energy_total_mev,
            "observed_maximum_mev_per_source_a": native.rows[-1].energy_total_mev / native.ion_mass_u}


def run_case(case: UpperControl, baseline: SrimRangeOutput, *, output: Path,
             originals: dict[str, bytes], runtime: bytes, pins: dict[str, str]) -> dict[str, Any]:
    repeats = []
    for number in (1, 2):
        work = output / f"{case.name}_r{number}"
        work.mkdir()
        copied = {**originals, "MSVBVM50.dll": runtime, "SR.IN": input_bytes(case)}
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
    result = {"case": asdict(case), "baseline_raw_sha256": baseline.raw_sha256,
              "repeats": repeats, "status": "CONTROL_COMPLETE" if same else
              "REPEAT_MISMATCH" if valid else repeats[-1]["status"],
              "semantic_outputs_identical": same, "raw_output_bytes_identical": valid and
              repeats[0]["output"]["sha256"] == repeats[1]["output"]["sha256"], **FLAGS}
    _write_new(output / f"{case.name}.json", _json(result).encode("utf-8"))
    return result


def package(report_path: Path, report: dict[str, Any], output: Path) -> dict[str, Any]:
    if report["all_native_controls_repeated"] is not True or len(report["cases"]) != 8:
        raise ValueError("All eight successful cases are required before packaging.")
    members: dict[str, bytes] = {}
    records = []
    for result in report["cases"]:
        if result["status"] != "CONTROL_COMPLETE":
            raise ValueError("Failed case cannot supply a complete bundle.")
        case, evidence = result["case"], result["repeats"][0]["output"]
        raw = Path(evidence["path"]).read_bytes()
        if _sha(raw) != evidence["sha256"]:
            raise ValueError("Native output changed before packaging.")
        name = case["name"]
        member = f"native/{name}.txt"
        if member in members:
            raise ValueError("Duplicate native table.")
        members[member] = raw
        records.append({"case_name": name, "z": case["ion_z"],
                        "mass_parameter_u_decimal": format(case["ion_mass_u"], "f"),
                        "material_z": case["target_z"], "member": member, "sha256": _sha(raw)})
    manifest = {"schema_version": SCHEMA, **POLICIES, "tables": records,
                "baseline_bundle_sha256": BASELINE_SHA256, "native_report": _identity(report_path),
                "generator_sha256": _sha(Path(__file__).read_bytes()),
                "ceiling_mev_per_literal_source_a": str(CEILING_MEV_PER_SOURCE_A),
                "ceiling_policy": "bounded_native_execution_experiment_not_physical_validity",
                "software_labels": [["SRIM-2006", "SRIM-2012.01"]],
                "limitations": ["Four-ion pilot only, no source-tail extrapolation.",
                                "No absolute range origin or normative qualification.",
                                "Whole tables only; no stitched model or renormalization."]}
    members["manifest.json"] = _json(manifest).encode("utf-8")
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(members.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw, compresslevel=9)
    raw = buffer.getvalue()
    _write_new(output, raw)
    dataset = load_srim_hze_dataset(output, expected_sha256=_sha(raw))
    if tuple(dataset.tables_by_z) != PILOT_Z:
        raise ValueError("Packaged pilot inventory differs from the four fixed ions.")
    return _identity(output)


def run(*, source: Path, private_runtime: Path, output: Path, bundle: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Upper pilot output must be a new repository outputs subdirectory.")
    originals = {name: (source / name).read_bytes() for name in launcher.SOURCE_HASHES}
    if any(_sha(raw) != launcher.SOURCE_HASHES[name] for name, raw in originals.items()):
        raise ValueError("Pinned normal-mode source changed.")
    runtime = private_runtime.read_bytes()
    if _sha(runtime) != launcher.PRIVATE_RUNTIME_SHA256:
        raise ValueError("Pinned private runtime changed.")
    dataset = load_srim_hze_dataset(bundle, expected_sha256=BASELINE_SHA256)
    selected = controls(dataset)
    protected = [bundle, private_runtime, source / "SR.IN", *[source / n for n in originals],
                 Path(__file__), Path(low.__file__), Path(launcher.__file__),
                 ROOT / "scripts/run_srim_coverage.py", *sorted((ROOT / "src/radar").rglob("*.py")),
                 *sorted(p for p in (ROOT / "src/radar/data/normative").rglob("*") if p.is_file())]
    pins = low._pins(protected)
    output.mkdir()
    cases = []
    for case, baseline in selected:
        result = run_case(case, baseline, output=output, originals=originals, runtime=runtime, pins=pins)
        cases.append(result)
        if result["status"] != "CONTROL_COMPLETE":
            break
    low._verify(pins)
    complete = len(cases) == 8 and all(c["status"] == "CONTROL_COMPLETE" for c in cases)
    report = {"schema_version": "radar.srim_upper_pilot.v1", "scope": "pilot",
              "baseline_bundle": _identity(bundle), "source_directory": str(source.resolve()),
              "private_runtime": _identity(private_runtime), "planned_cases": [asdict(c) for c, _ in selected],
              "cases": cases, "not_attempted_cases": [c.name for c, _ in selected[len(cases):]],
              "all_native_controls_repeated": complete, "runs_per_case": 2,
              "ceiling_mev_per_literal_source_a": CEILING_MEV_PER_SOURCE_A,
              "ceiling_policy": "bounded_native_execution_experiment_not_physical_validity",
              "protected_sha256_before": pins, "protected_sha256_after": low._pins(protected),
              "limitations": ["Failure stops the pilot; no automatic boundary search.",
                              "Observed high-energy outputs do not establish physical accuracy.",
                              "Upper source tails and low-energy range origin remain unresolved."], **FLAGS}
    report_path = output / "report.json"
    _write_new(report_path, _json(report).encode("utf-8"))
    if complete:
        result = package(report_path, report, output / "observed_upper_pilot.zip")
        _write_new(output / "bundle_identity.json", _json(result).encode("utf-8"))
    low._verify(pins)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"C:\Users\Иван\Desktop\ПО\SRIM-2013\SR Module"))
    parser.add_argument("--private-runtime", type=Path, default=ROOT /
                        "outputs/srim_source_extension_pilot_20261101/u238_si_upper225000_r2/MSVBVM50.dll")
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs/srim_lower_full_20261105/observed_full.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(source=args.source, private_runtime=args.private_runtime, output=args.output, bundle=args.bundle)
    print(json.dumps({"output": str(args.output), "all_native_controls_repeated":
                      report["all_native_controls_repeated"], "normative_export_allowed": False}))
    return 0 if report["all_native_controls_repeated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
