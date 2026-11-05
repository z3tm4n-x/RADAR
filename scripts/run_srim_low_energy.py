"""Six-case isolated SRIM lower-energy pilot; no extrapolation or release claim.

He, Fe and U in Al/Si, 10 eV TOTAL to the unchanged authenticated upper node.
Two repeats, whole native tables, exact shared stopping and separate range audit.
Only pilot execution is enabled; a full campaign needs a reviewed pilot first.
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
from scripts.run_srim_coverage import _retained_command  # noqa: E402
from radar.shielding.srim_dataset import (  # noqa: E402
    POLICIES, SCHEMA, SrimHzeDataset, load_srim_hze_dataset,
)
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output  # noqa: E402
from radar.validation.srim_source_catalog import build_source_catalog  # noqa: E402

BASELINE_SHA256 = "f2b523dfafbbe8bfbcf70b27a186f05ed009a19918b14042f97b39e37e7f0bb2"
MINIMUM_MEV = Decimal("0.00001")
PILOT_Z = (2, 26, 92)
_sha, _identity, _write_new, _json = launcher._sha, launcher._identity, launcher._write_new, launcher._json
_execute = launcher._execute
_FLAGS = {"diagnostic_status": "INCOMPLETE", "normative_export_allowed": False,
          "physical_accuracy_verified": False, "production_tables_replaced": False,
          "absolute_range_origin_established": False}


@dataclass(frozen=True)
class LowControl:
    name: str
    ion_z: int
    ion_mass_u: Decimal
    target_z: int
    target_mass_u: Decimal
    density_g_cm3: Decimal
    energy_max_mev: Decimal


def controls(dataset: SrimHzeDataset) -> tuple[tuple[LowControl, SrimRangeOutput], ...]:
    catalog = build_source_catalog(ROOT)
    result = []
    for base in catalog:
        if base.ion_z not in PILOT_Z:
            continue
        table = dataset.tables_by_z[base.ion_z]
        native = table.native_al if base.target_z == 13 else table.native_si
        if (native.ion_mass_u != base.ion_mass_u or native.density_g_cm3 != base.density_g_cm3
                or native.rows[0].energy_total_mev != 1):
            raise ValueError("Pilot differs from literal source mass/material/energy baseline.")
        result.append((LowControl(f"{base.name}_low10ev", base.ion_z, base.ion_mass_u,
                                  base.target_z, base.target_mass_u, base.density_g_cm3,
                                  native.rows[-1].energy_total_mev), native))
    if len(result) != 6 or {(c.ion_z, c.target_z) for c, _ in result} != {
            (z, target) for z in PILOT_Z for target in (13, 14)}:
        raise ValueError("Pilot must contain exactly He, Fe, U Al/Si pairs.")
    return tuple(result)


def expected_grid(maximum: Decimal) -> tuple[Decimal, ...]:
    mantissas = tuple(Decimal(v) for v in (
        "1", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2",
        "2.25", "2.5", "2.75", "3", "3.25", "3.5", "3.75", "4", "4.5", "5",
        "5.5", "6", "6.5", "7", "8", "9",
    ))
    if (type(maximum) is not Decimal or not maximum.is_finite()
            or not Decimal(1) <= maximum <= 225000):
        raise ValueError("Unsupported retained upper energy.")
    with localcontext() as context:
        context.prec = 80
        grid = tuple(e for exponent in range(-5, 6) for m in mantissas
                     if (e := m * Decimal(10) ** exponent) <= maximum)
    if grid[-1] != maximum:
        raise ValueError("Upper energy must be an existing native-grid node.")
    return grid


def input_bytes(case: LowControl) -> bytes:
    expected_grid(case.energy_max_mev)
    if (case.ion_z not in PILOT_Z or case.target_z not in (13, 14)
            or case.ion_mass_u != {2: Decimal(4), 26: Decimal("55.8"), 92: Decimal(238)}[case.ion_z]
            or case.target_mass_u != {13: Decimal("26.981539"), 14: Decimal("28.0855")}[case.target_z]
            or case.density_g_cm3 != {13: Decimal("2.70"), 14: Decimal("2.3290")}[case.target_z]):
        raise ValueError("Only the fixed lower-energy pilot identities are permitted.")
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


def validate_native(native: SrimRangeOutput, baseline: SrimRangeOutput,
                    case: LowControl) -> dict[str, Any]:
    for key in ("ion_z", "ion_name", "ion_mass_u", "target_z", "target_symbol",
                "density_g_cm3", "atomic_density_cm3", "bragg_correction_percent",
                "engine_label", "version_label", "conversion_factors", "output_name"):
        if getattr(native, key) != getattr(baseline, key):
            raise ValueError(f"Lower-energy metadata differs from baseline: {key}")
    if (native.ion_z != case.ion_z or native.ion_mass_u != case.ion_mass_u
            or native.target_z != case.target_z or native.density_g_cm3 != case.density_g_cm3
            or native.engine_label != "SRIM-2006" or native.version_label != "SRIM-2012.01"
            or native.bragg_correction_percent != 0 or native.output_name != "srim_output.txt"):
        raise ValueError("Native result differs from fixed pilot identity.")
    nominal = expected_grid(case.energy_max_mev)
    if len(native.rows) != len(nominal):
        raise ValueError("Native output lacks the complete fixed 10eV-to-upper grid.")
    shifts = []
    units = {"eV": Decimal("0.000001"), "keV": Decimal("0.001"),
             "MeV": Decimal(1), "GeV": Decimal(1000)}
    for row, expected in zip(native.rows, nominal, strict=True):
        token = Decimal(row.printed_tokens[0].replace(",", "."))
        exponent = token.as_tuple().exponent
        if not isinstance(exponent, int):
            raise ValueError("Nonfinite printed energy.")
        quantum = Decimal(10) ** exponent * units[row.energy_unit]
        difference = abs(row.energy_total_mev - expected)
        relative = difference / expected
        # Native low-energy rendering can truncate the internal value by one
        # printed digit. Keep the observed energy; never snap it to nominal.
        # The extra relative cap rejects arbitrarily coarse-token departures.
        if difference > quantum or relative > Decimal("0.00001") or (expected >= 1 and difference):
            raise ValueError("Native energy differs from its one-to-one printed-grid allowance.")
        shifts.append((difference, relative))
    suffix = tuple(row for row in native.rows if row.energy_total_mev >= 1)
    if len(suffix) != len(baseline.rows):
        raise ValueError("Shared baseline suffix is incomplete.")
    for new, old in zip(suffix, baseline.rows, strict=True):
        if any(getattr(new, name) != getattr(old, name) for name in (
                "energy_total_mev", "electronic_mev_cm2_mg", "nuclear_mev_cm2_mg")):
            raise ValueError("Shared stopping values changed: no automatic table replacement.")
    changes = []
    for name in ("projected_range_cm", "longitudinal_straggling_cm", "lateral_straggling_cm"):
        differences = [(new.energy_total_mev, getattr(new, name), getattr(old, name))
                       for new, old in zip(suffix, baseline.rows, strict=True)
                       if getattr(new, name) != getattr(old, name)]
        changes.append({"field": name, "changed_rows": len(differences),
                        "first_change_energy_new_old": differences[0] if differences else None})
    return {"native_rows": len(native.rows), "new_lower_rows": len(native.rows) - len(suffix),
            "shared_rows": len(suffix), "shared_stopping_identical": True,
            "shared_complete_rows_identical": suffix == baseline.rows,
            "requested_minimum_total_ion_energy_mev": MINIMUM_MEV,
            "observed_minimum_total_ion_energy_mev": native.rows[0].energy_total_mev,
            "printed_energy_shift_count": sum(delta > 0 for delta, _ in shifts),
            "max_printed_energy_shift_mev": max(delta for delta, _ in shifts),
            "max_printed_energy_shift_relative": max(relative for _, relative in shifts),
            "grid_policy": "one_to_one_nominal_nodes_within_one_printed_quantum_and_1e-5_relative;"
                           "shared_suffix_exact;observed_energies_unchanged",
            "projected_range_and_straggling_changes": changes}


def _pins(files: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): _sha(path.read_bytes()) for path in files}


def _verify(pins: dict[str, str]) -> None:
    if any(_sha(Path(name).read_bytes()) != sha for name, sha in pins.items()):
        raise ValueError("Protected native source, input, data or implementation changed.")


def _case(case: LowControl, baseline: SrimRangeOutput, *, output: Path,
          copied_source: dict[str, bytes], runtime: bytes, pins: dict[str, str]) -> dict[str, Any]:
    repeats: list[dict[str, Any]] = []
    for number in (1, 2):
        work = output / f"{case.name}_r{number}"
        work.mkdir()
        copied = {**copied_source, "MSVBVM50.dll": runtime, "SR.IN": input_bytes(case)}
        for name, raw in copied.items():
            _write_new(work / name, raw)
        result: dict[str, Any] = {"repeat": number, "input": _identity(work / "SR.IN"), **_FLAGS}
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
                    comparison = validate_native(native, baseline, case)
                except (ValueError, FileNotFoundError) as error:
                    result.update(status="INVALID_OUTPUT", reason=str(error))
                else:
                    result.update(status="VALID_NATIVE_OUTPUT", comparison=comparison,
                                  semantic_sha256=native.semantic_sha256)
            result["command_file"] = _identity(work / "command.json")
            result["output"] = (_identity(work / "srim_output.txt")
                                if (work / "srim_output.txt").exists() else None)
        finally:
            _verify(pins)
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
              "repeats": repeats, "status": "CONTROL_COMPLETE" if same else
              "REPEAT_MISMATCH" if valid else repeats[-1]["status"],
              "semantic_outputs_identical": same,
              "raw_output_bytes_identical": valid and
              repeats[0]["output"]["sha256"] == repeats[1]["output"]["sha256"], **_FLAGS}
    _write_new(output / f"{case.name}.json", _json(record).encode("utf-8"))
    return record


def package(report_path: Path, report: dict[str, Any], output: Path) -> dict[str, Any]:
    """Package only whole successful native pairs; never splice table rows."""
    if report["all_native_controls_repeated"] is not True or len(report["cases"]) != 6:
        raise ValueError("A complete six-case pilot is required for packaging.")
    members: dict[str, bytes] = {}
    records = []
    observed_minima = {}
    for result in report["cases"]:
        if result["status"] != "CONTROL_COMPLETE":
            raise ValueError("Cannot package a failed native control.")
        case = result["case"]
        evidence = result["repeats"][0]["output"]
        path = Path(evidence["path"])
        raw = path.read_bytes()
        if _sha(raw) != evidence["sha256"]:
            raise ValueError("Native output changed before packaging.")
        name = case["name"]
        observed_minima[name] = str(parse_srim_range_output(raw).rows[0].energy_total_mev)
        member = f"native/{name}.txt"
        members[member] = raw
        records.append({"case_name": name, "z": case["ion_z"],
                        "mass_parameter_u_decimal": format(case["ion_mass_u"], "f"),
                        "material_z": case["target_z"], "member": member, "sha256": _sha(raw)})
    manifest = {"schema_version": SCHEMA, **POLICIES, "tables": records,
                "baseline_bundle_sha256": report["baseline_bundle"]["sha256"],
                "native_report": _identity(report_path),
                "generator_sha256": _sha(Path(__file__).read_bytes()),
                "requested_minimum_total_ion_energy_mev": str(MINIMUM_MEV),
                "observed_minimum_total_ion_energy_mev_by_table": observed_minima,
                "absolute_range_origin_established": False,
                "software_labels": [["SRIM-2006", "SRIM-2012.01"]],
                "limitations": ["Observed support only; no continuation from 10eV to rest.",
                                "Projected ranges are not CSDA path ranges.",
                                "Pilot covers only He, Fe and U; absent ions remain incomplete."]}
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
    if tuple(loaded.tables_by_z) != PILOT_Z:
        raise ValueError("Packaged pilot ion inventory mismatch.")
    return _identity(output)


def run(*, source: Path, private_runtime: Path, output: Path, bundle: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Native pilot output must stay in a new repository outputs subdirectory.")
    originals = {name: (source / name).read_bytes() for name in launcher.SOURCE_HASHES}
    if any(_sha(raw) != launcher.SOURCE_HASHES[name] for name, raw in originals.items()):
        raise ValueError("Pinned normal-mode SRIM source changed.")
    runtime = private_runtime.read_bytes()
    if _sha(runtime) != launcher.PRIVATE_RUNTIME_SHA256:
        raise ValueError("Private VB runtime identity mismatch.")
    dataset = load_srim_hze_dataset(bundle, expected_sha256=BASELINE_SHA256)
    selected = controls(dataset)
    protected = [bundle, private_runtime, source / "SR.IN", *[source / n for n in originals],
                 Path(__file__), ROOT / "scripts/run_srim_controls.py",
                 ROOT / "scripts/run_srim_coverage.py",
                 *sorted((ROOT / "src/radar").rglob("*.py")),
                 *sorted(p for p in (ROOT / "src/radar/data/normative").rglob("*") if p.is_file())]
    pins = _pins(protected)
    output.mkdir()
    cases = []
    for case, baseline in selected:
        result = _case(case, baseline, output=output, copied_source=originals,
                       runtime=runtime, pins=pins)
        cases.append(result)
        if result["status"] != "CONTROL_COMPLETE":
            break
    _verify(pins)
    complete = len(cases) == len(selected) and all(c["status"] == "CONTROL_COMPLETE" for c in cases)
    report = {"schema_version": "radar.srim_low_energy_pilot.v1", "scope": "pilot",
              "baseline_bundle": _identity(bundle), "source_directory": str(source.resolve()),
              "private_runtime": _identity(private_runtime),
              "requested_minimum_total_ion_energy_mev": MINIMUM_MEV,
              "planned_cases": [asdict(c) for c, _ in selected], "cases": cases,
              "not_attempted_cases": [c.name for c, _ in selected[len(cases):]],
              "runs_per_case": 2, "all_native_controls_repeated": complete,
              "protected_sha256_before": pins, "protected_sha256_after": _pins(protected),
              "limitations": ["Whole new native tables, no stitched or extrapolated rows.",
                              "Lower energy still positive: stopped classification is not established.",
                              "Successful execution validates output integrity, not physical accuracy."],
              **_FLAGS}
    report_path = output / "report.json"
    _write_new(report_path, _json(report).encode("utf-8"))
    if complete:
        result = package(report_path, report, output / "observed_pilot.zip")
        _write_new(output / "bundle_identity.json", _json(result).encode("utf-8"))
    _verify(pins)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(r"C:\Users\Иван\Desktop\ПО\SRIM-2013\SR Module"))
    parser.add_argument("--private-runtime", type=Path, default=ROOT /
                        "outputs/srim_source_extension_pilot_20261101/u238_si_upper225000_r2/MSVBVM50.dll")
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs/srim_hze_observed_20261105.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(source=args.source, private_runtime=args.private_runtime,
                 bundle=args.bundle, output=args.output)
    print(json.dumps({"report": str(args.output / "report.json"),
                      "all_native_controls_repeated": report["all_native_controls_repeated"],
                      "normative_export_allowed": False}))
    return 0 if report["all_native_controls_repeated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
