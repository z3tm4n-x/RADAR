"""Fixed legacy-ion SRIM diagnostics, never a production-table generator.

The pilot and its dependencies are pinned. Native failures and incomplete output
remain explicit; only two successful, identical runs complete a native control.
Neither a complete control nor a full legacy grid establishes physical validity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal, localcontext
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_srim_controls as pilot  # noqa: E402
from radar.validation.srim_output import parse_srim_range_output  # noqa: E402
from radar.validation.srim_comparison import compare_legacy_table  # noqa: E402

SOURCE_HASHES = pilot.SOURCE_HASHES
PRIVATE_RUNTIME_SHA256 = pilot.PRIVATE_RUNTIME_SHA256
PRODUCTION = pilot.PRODUCTION
IMPLEMENTATION = (*pilot.IMPLEMENTATION, "scripts/run_srim_coverage.py")
PILOT_REPORT = "outputs/srim_controls_final_20261101/report.json"
PILOT_REPORT_SHA256 = "edf9fe709600d7c1dc4a042e2706418b95dac4bca9fa341cc8f3e2a9ef633565"
_execute = pilot._execute
_sha = pilot._sha
_identity = pilot._identity
_json = pilot._json
_write_new = pilot._write_new
SrimControl = pilot.SrimControl

LEGACY_IONS = (
    (2, "He", 4), (6, "C", 12), (7, "N", 14), (8, "O", 16), (10, "Ne", 20),
    (12, "Mg", 24), (14, "Si", 28), (16, "S", 32), (18, "Ar", 40), (20, "Ca", 40),
    (24, "Cr", 52), (25, "Mn", 55), (26, "Fe", 56), (27, "Co", 59), (28, "Ni", 58),
)


def _catalog(upper: bool) -> tuple[SrimControl, ...]:
    return tuple(SrimControl(
        f"{symbol.lower()}{mass}_{target}", z, Decimal(mass), tz, Decimal(tmass),
        Decimal(density), Decimal(1), Decimal(45000 if upper else 12000),
        f"stopping/hze_{target}_{'range' if target == 'al' else 'let'}.csv",
    ) for z, symbol, mass in LEGACY_IONS if not upper or z != 2
        for target, tz, tmass, density in (("al", 13, "26.981539", "2.70"),
                                         ("si", 14, "28.0855", "2.3290")))


BOUNDARY_CASES = tuple(SrimControl(
    f"h_{target}_{maximum}", 1, Decimal("1.008"), tz, Decimal(tmass), Decimal(density),
    Decimal("0.1"), Decimal(maximum),
    f"stopping/proton_{target}_{'range' if target == 'al' else 'let'}.csv",
) for target, tz, tmass, density, maximum in (
    ("al", 13, "26.981539", "2.70", 2000),
    ("al", 13, "26.981539", "2.70", 4000),
    ("al", 13, "26.981539", "2.70", 6000),
    ("si", 14, "28.0855", "2.3290", 4000),
))
CATALOGS = {"baseline": _catalog(False), "upper": _catalog(True), "boundary": BOUNDARY_CASES}


def _check_case(case: SrimControl) -> None:
    if (type(case) is not SrimControl or type(case.ion_z) is not int
            or type(case.target_z) is not int or any(type(value) is not Decimal for value in (
                case.ion_mass_u, case.target_mass_u, case.density_g_cm3,
                case.energy_min_mev, case.energy_max_mev,
            )) or not any(case in cases for cases in CATALOGS.values())):
        raise ValueError("Only fixed legacy-ion coverage controls are allowed.")


def expected_control_grid(case: SrimControl) -> tuple[Decimal, ...]:
    _check_case(case)
    mantissas = tuple(Decimal(value) for value in (
        "1", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2",
        "2.25", "2.5", "2.75", "3", "3.25", "3.5", "3.75", "4", "4.5", "5",
        "5.5", "6", "6.5", "7", "8", "9",
    ))
    return tuple(energy for exponent in range(-1, 5) for value in mantissas
                 if case.energy_min_mev <= (energy := value * Decimal(10) ** exponent)
                 <= case.energy_max_mev)


def input_bytes(case: SrimControl) -> bytes:
    _check_case(case)
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
        f"{case.energy_min_mev * 1000}    {case.energy_max_mev * 1000}",
    )
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def _retained_command(work: Path) -> dict[str, Any]:
    command = json.loads((work / "command.json").read_text(encoding="utf-8"))
    if (not isinstance(command, dict)
            or command.get("argv") != [str(work / "SRModule.exe")]
            or command.get("cwd") != str(work)
            or type(command.get("returncode")) is not int
            or type(command.get("timed_out")) is not bool
            or type(command.get("pid")) is not int or command["pid"] <= 0
            or type(command.get("elapsed_seconds")) not in (int, float)
            or not isfinite(command["elapsed_seconds"]) or command["elapsed_seconds"] < 0
            or not isinstance(command.get("stdout"), str)
            or not isinstance(command.get("stderr"), str)):
        raise ValueError("Invalid retained native command record.")
    return command


def _verify_unchanged(files: dict[Path, bytes]) -> None:
    if any(path.read_bytes() != raw for path, raw in files.items()):
        raise ValueError("Protected source, runtime, input or implementation changed.")


def _scope(case: SrimControl) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = 80
        maximum = case.energy_max_mev / case.ion_mass_u
    return {
        "max_energy_mev_per_declared_u": maximum,
        "ratio_decimal_precision": 80,
        "package_trim_help_high_energy_caution": case.energy_max_mev > 1000 * case.ion_mass_u,
        "within_package_table_claim_10000_mev_per_u":
            case.energy_max_mev <= 10000 * case.ion_mass_u,
        "physical_accuracy_verified": False,
        "normative_export_allowed": False,
    }


def run_coverage(*, catalog: str, source: Path, private_runtime: Path,
                 output: Path) -> dict[str, Any]:
    if not isinstance(catalog, str) or catalog not in CATALOGS:
        raise ValueError("Unknown fixed coverage catalog.")
    if output.exists():
        raise FileExistsError(output)
    pilot_path = ROOT / PILOT_REPORT
    pilot_bytes = pilot_path.read_bytes()
    if _sha(pilot_bytes) != PILOT_REPORT_SHA256:
        raise ValueError("Pinned pilot report differs.")
    baseline = json.loads(pilot_bytes)
    baseline_pins = baseline["selected_files_sha256_after"]
    if set(baseline_pins) != set((*pilot.IMPLEMENTATION, *PRODUCTION)):
        raise ValueError("Pilot protected-file inventory differs.")
    protected = {ROOT / name: (ROOT / name).read_bytes() for name in IMPLEMENTATION + PRODUCTION}
    for name, expected in baseline_pins.items():
        if _sha(protected[ROOT / name]) != expected:
            raise ValueError(f"Pinned pilot implementation/production file differs: {name}.")
    originals = {name: (source / name).read_bytes() for name in SOURCE_HASHES}
    if any(_sha(raw) != SOURCE_HASHES[name] for name, raw in originals.items()):
        raise ValueError("Pinned normal-mode SRIM source differs.")
    runtime = private_runtime.read_bytes()
    if _sha(runtime) != PRIVATE_RUNTIME_SHA256:
        raise ValueError("Private bundled VB runtime identity mismatch.")
    original_input = (source / "SR.IN").read_bytes()
    protected.update({source / name: raw for name, raw in originals.items()})
    protected.update({source / "SR.IN": original_input, private_runtime: runtime,
                      pilot_path: pilot_bytes})
    output.mkdir(parents=False)
    output = output.resolve()
    cases: list[dict[str, Any]] = []
    for case in CATALOGS[catalog]:
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
                    # Only a retained, explicit native exit/timeout is recoverable.
                    # Unexpected launcher/runtime errors propagate, rather than
                    # being relabelled a successful or diagnosed native run.
                    command = _retained_command(work)
                    if not command["timed_out"] and command["returncode"] == 0:
                        raise
                    record.update(status="NATIVE_FAILED", reason=str(error), command=command)
                else:
                    command = _retained_command(work)
                    if returned != command or command["timed_out"] or command["returncode"] != 0:
                        raise ValueError("Native success disagrees with retained command record.")
                    record["command"] = command
                    path = work / "srim_output.txt"
                    try:
                        parsed = parse_srim_range_output(path.read_bytes())
                        if (parsed.ion_z != case.ion_z or parsed.ion_mass_u != case.ion_mass_u
                                or parsed.target_z != case.target_z
                                or parsed.density_g_cm3 != case.density_g_cm3
                                or parsed.bragg_correction_percent != 0
                                or parsed.engine_label != "SRIM-2006"
                                or parsed.version_label != "SRIM-2012.01"
                                or parsed.output_name != "srim_output.txt"):
                            raise ValueError("Native case metadata mismatch.")
                        if tuple(row.energy_total_mev for row in parsed.rows) != \
                                expected_control_grid(case):
                            raise ValueError("Native full control energy grid mismatch.")
                    except (ValueError, FileNotFoundError) as error:
                        record.update(status="INVALID_OUTPUT", reason=str(error))
                    else:
                        comparison = compare_legacy_table(
                            parsed, protected[ROOT / "src/radar/data/normative" / case.resource_path],
                            case.resource_path,
                        )
                        record.update(status="VALID_NATIVE_OUTPUT", parsed=asdict(parsed),
                                      semantic_sha256=parsed.semantic_sha256, comparison=comparison)
                record["command_file"] = _identity(work / "command.json")
                path = work / "srim_output.txt"
                record["output"] = _identity(path) if path.is_file() else None
            finally:
                _verify_unchanged({**protected, **copied})
            _write_new(work / "result.json", _json(record).encode("utf-8"))
            repeats.append(record)
            print(f"{catalog} {case.name} repeat {repeat}: {record['status']}", flush=True)
            if record["status"] != "VALID_NATIVE_OUTPUT":
                break
        valid = len(repeats) == 2 and all(r["status"] == "VALID_NATIVE_OUTPUT" for r in repeats)
        identical = valid and repeats[0]["semantic_sha256"] == repeats[1]["semantic_sha256"]
        status = ("CONTROL_COMPLETE" if identical else "REPEAT_MISMATCH" if valid
                  else repeats[-1]["status"])
        case_record = {
            "case": asdict(case), "scope": _scope(case), "repeats": repeats, "status": status,
            "expected_control_grid_mev": expected_control_grid(case),
            "complete_native_control_grid_verified": valid,
            "semantic_outputs_identical": identical,
            "raw_output_bytes_identical": valid and repeats[0]["output"]["sha256"] ==
                repeats[1]["output"]["sha256"],
            "physical_accuracy_verified": False, "normative_export_allowed": False,
        }
        _write_new(output / f"{case.name}.json", _json(case_record).encode("utf-8"))
        cases.append(case_record)
    _verify_unchanged(protected)
    report = {
        "schema_version": "radar.srim_fixed_legacy_coverage.v1", "catalog": catalog,
        "pilot_report": _identity(pilot_path), "source_directory": str(source.resolve()),
        "source_sha256": SOURCE_HASHES, "original_input_sha256": _sha(original_input),
        "private_runtime": _identity(private_runtime),
        "protected_sha256_before": {str(path): _sha(raw) for path, raw in protected.items()},
        "protected_sha256_after": {str(path): _sha(path.read_bytes()) for path in protected},
        "cases": cases, "case_count": len(cases), "requested_runs_per_case": 2,
        "all_native_controls_repeated": all(c["status"] == "CONTROL_COMPLETE" for c in cases),
        "native_control_status_counts": {status: sum(c["status"] == status for c in cases)
            for status in ("CONTROL_COMPLETE", "NATIVE_FAILED", "INVALID_OUTPUT", "REPEAT_MISMATCH")},
        "legacy_hze_ion_count": 15,
        "catalog_kind": "proton_boundary" if catalog == "boundary" else "legacy_hze_coverage",
        "catalog_ion_count": len({c.ion_z for c in CATALOGS[catalog]}),
        "catalog_species": [
            {"z": z, "mass_u": mass} for z, mass in sorted(
                {(case.ion_z, case.ion_mass_u) for case in CATALOGS[catalog]})
        ],
        "positive_gcr_species_count": 84, "positive_sep_species_count": 24,
        "required_runtime_species_and_masses_fully_validated": False,
        "diagnostic_status": "INCOMPLETE", "physical_accuracy_verified": False,
        "normative_export_allowed": False, "production_data_modified": False,
        "release_status_changed": False, "source_files_unchanged": True,
        "limitations": [
            "The fixed legacy-ion catalog does not cover the full positive GCR/SEP roster.",
            "No interpolation, extrapolation, zero-filled failed cases or automatic source promotion.",
            "Normal SRModule projected range is not CSDA range; historical range meaning is open.",
            "Package TRIM help warns about sparse high-energy data above 1 GeV/u; applicability "
            "to SRModule accuracy is not independently established. Neither side is qualified here.",
            "Package table claim of 10 GeV/u is not a physical accuracy or execution guarantee.",
            "No high-energy nuclear-reaction transport is supplied by this stopping-table comparison.",
            "User-pinned binary, private VB runtime and normal coefficient files only; "
            "no installation, underscore substitution or source-model fitting.",
        ],
    }
    _write_new(output / "report.json", _json(report).encode("utf-8"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", choices=tuple(CATALOGS), required=True)
    parser.add_argument("--source", type=Path,
                        default=Path(r"C:\Users\Иван\Desktop\ПО\SRIM-2013\SR Module"))
    parser.add_argument("--private-runtime", type=Path,
                        default=ROOT / "outputs/srim_runtime_dependencies_20261101/MSVBVM50.dll")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_coverage(catalog=args.catalog, source=args.source,
                          private_runtime=args.private_runtime, output=args.output)
    print(_json({"report": str(args.output / "report.json"),
                 "native_control_status_counts": report["native_control_status_counts"],
                 "normative_export_allowed": False}))
    return 0 if report["all_native_controls_repeated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
