"""Pinned, isolated normal-mode SRIM controls; never a production-table generator.

Runs the user-provided module, not the installer or GUI. Requires an existing
private VB5 runtime. No system changes, network access or alternative underscore
data files. Output and every case directory must be new. Native source files,
input/output bytes and selected production resources are retained/fingerprinted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.validation.srim_output import parse_srim_range_output  # noqa: E402
from radar.validation.srim_comparison import compare_legacy_table  # noqa: E402


SOURCE_HASHES = {
    "SRModule.exe": "a679e55defdb12ebe39c02ca0562016b79ad73105dcaeddc9e8ba40f57c16288",
    "SCOEF03.dat": "a328eaaa04e48efed1d15cfbcdf810849115ff352feaac1ee323c3bf4229ecea",
    "SNUC03.dat": "4c7ff553b7be34a1b28a31a6ac89055b800392cbbf0401784da6afc5a693a030",
    "VERSION": "f0265c292d60ec141b8518ed688b52b977f2ec81761b66195545024579ef4c40",
    "HELP- SR Module.rtf": "5df89c445c295c4df6b38a0af03c695764db723876603164ba32d390bb5636de",
}
PRIVATE_RUNTIME_SHA256 = "8611dc1b60ae5c383bba6cb3ffd8a51aeebff23b95844f0ab3d6e5ecd0fadc84"
IMPLEMENTATION = (
    "scripts/run_srim_controls.py", "src/radar/validation/srim_output.py",
    "src/radar/validation/srim_comparison.py",
)
PRODUCTION = (
    "src/radar/data/normative/integrity_manifest.json",
    "src/radar/data/normative/stopping/proton_al_range.csv",
    "src/radar/data/normative/stopping/proton_si_let.csv",
    "src/radar/data/normative/stopping/hze_al_range.csv",
    "src/radar/data/normative/stopping/hze_si_let.csv",
    "src/radar/shielding/stopping_tables.py",
)


@dataclass(frozen=True)
class SrimControl:
    name: str
    ion_z: int
    ion_mass_u: Decimal
    target_z: int
    target_mass_u: Decimal
    density_g_cm3: Decimal
    energy_min_mev: Decimal
    energy_max_mev: Decimal
    resource_path: str


def _cases() -> tuple[SrimControl, ...]:
    result = []
    for ion, z, mass in (("h", 1, "1.008"), ("he4", 2, "4"), ("fe56", 26, "56")):
        for target, tz, tmass, density in (("al", 13, "26.981539", "2.70"),
                                          ("si", 14, "28.0855", "2.3290")):
            species = "proton" if z == 1 else "hze"
            suffix = "range" if target == "al" else "let"
            result.append(SrimControl(
                f"{ion}_{target}", z, Decimal(mass), tz, Decimal(tmass), Decimal(density),
                Decimal("0.1" if z == 1 else "1"),
                Decimal("1000" if z == 1 else "12000"),
                f"stopping/{species}_{target}_{suffix}.csv",
            ))
    return tuple(result)


CONTROL_CASES = _cases()


def expected_control_grid(case: SrimControl) -> tuple[Decimal, ...]:
    """Observed normal-mode node contract for these pinned controls, not a model.

    Do not infer completeness merely from matching endpoints. The same grid was
    retained in all six development outputs; a changed module needs review.
    """
    if case not in CONTROL_CASES:
        raise ValueError("Unknown fixed SRIM control.")
    mantissas = tuple(Decimal(value) for value in (
        "1", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2",
        "2.25", "2.5", "2.75", "3", "3.25", "3.5", "3.75", "4", "4.5", "5",
        "5.5", "6", "6.5", "7", "8", "9",
    ))
    return tuple(energy for exponent in range(-1, 5) for mantissa in mantissas
                 if case.energy_min_mev <= (energy := mantissa * Decimal(10) ** exponent)
                 <= case.energy_max_mev)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": _sha(raw)}


def _json(value: Any) -> str:
    def encode(item: Any) -> str:
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, default=encode) + "\n"


def _write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)


def input_bytes(case: SrimControl) -> bytes:
    if (type(case) is not SrimControl or type(case.ion_z) is not int
            or type(case.target_z) is not int
            or any(type(value) is not Decimal for value in (
                case.ion_mass_u, case.target_mass_u, case.density_g_cm3,
                case.energy_min_mev, case.energy_max_mev,
            )) or case not in CONTROL_CASES):
        raise ValueError("Only the six declared normal-mode control cases are allowed.")
    target_name = "Aluminum" if case.target_z == 13 else "Silicon"
    text = f'''---Stopping/Range Input Data (Number-format: Period = Decimal Point)
---Output File Name
"srim_output.txt"
---Ion(Z), Ion Mass(u)
{case.ion_z}   {case.ion_mass_u}
---Target Data: (Solid=0,Gas=1), Density(g/cm3), Compound Corr.
0    {case.density_g_cm3}    1
---Number of Target Elements
1
---Target Elements: (Z), Target name, Stoich, Target Mass(u)
{case.target_z}   "{target_name}"   1   {case.target_mass_u}
---Output Stopping Units (1-8)
5
---Ion Energy : E-Min(keV), E-Max(keV)
{case.energy_min_mev * 1000}    {case.energy_max_mev * 1000}
'''
    return text.replace("\n", "\r\n").encode("ascii")


def _execute(work: Path) -> dict[str, Any]:
    if sys.platform != "win32":
        raise RuntimeError("Native SRIM controls require the approved Windows environment.")
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    argv = [str(work / "SRModule.exe")]
    begin = time.monotonic()
    process = subprocess.Popen(argv, cwd=work, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
    timed_out = False
    try:
        out, err = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        # This is only our just-created isolated child, never a pre-existing app.
        timed_out = True
        process.kill()
        out, err = process.communicate(timeout=5)
    record = {
        "argv": argv, "cwd": str(work), "pid": process.pid,
        "returncode": process.returncode, "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - begin,
        "stdout": out.decode("cp1251", errors="backslashreplace"),
        "stderr": err.decode("cp1251", errors="backslashreplace"),
    }
    _write_new(work / "command.json", _json(record).encode("utf-8"))
    if timed_out or process.returncode != 0:
        raise RuntimeError(f"SRIM control failed; retained command.json in {work}.")
    return record


def run_controls(*, source: Path, private_runtime: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    originals = {name: (source / name).read_bytes() for name in SOURCE_HASHES}
    for name, raw in originals.items():
        if _sha(raw) != SOURCE_HASHES[name]:
            raise ValueError(f"Pinned normal-mode SRIM source changed: {name}.")
    runtime_bytes = private_runtime.read_bytes()
    if _sha(runtime_bytes) != PRIVATE_RUNTIME_SHA256:
        raise ValueError("Private bundled VB runtime identity mismatch.")
    original_input = (source / "SR.IN").read_bytes()
    protected = {path: (ROOT / path).read_bytes() for path in (*IMPLEMENTATION, *PRODUCTION)}
    output.mkdir(parents=False)
    output = output.resolve()
    case_records: list[dict[str, Any]] = []
    for case in CONTROL_CASES:
        repeats: list[dict[str, Any]] = []
        for repeat in (1, 2):
            work = output / f"{case.name}_r{repeat}"
            work.mkdir()
            for name, raw in originals.items():
                _write_new(work / name, raw)
            _write_new(work / "MSVBVM50.dll", runtime_bytes)
            expected_input = input_bytes(case)
            _write_new(work / "SR.IN", expected_input)
            command = _execute(work)
            raw_result = (work / "srim_output.txt").read_bytes()
            parsed = parse_srim_range_output(raw_result)
            if (parsed.ion_z != case.ion_z or parsed.ion_mass_u != case.ion_mass_u
                    or parsed.target_z != case.target_z
                    or parsed.density_g_cm3 != case.density_g_cm3
                    or parsed.bragg_correction_percent != 0
                    or parsed.engine_label != "SRIM-2006"
                    or parsed.version_label != "SRIM-2012.01"
                    or parsed.output_name != "srim_output.txt"
                    or parsed.rows[0].energy_total_mev != case.energy_min_mev
                    or parsed.rows[-1].energy_total_mev != case.energy_max_mev):
                raise ValueError(f"SRIM input/output metadata or coverage mismatch: {case.name}.")
            if tuple(row.energy_total_mev for row in parsed.rows) != expected_control_grid(case):
                raise ValueError(f"SRIM full control energy grid mismatch: {case.name}.")
            comparison = compare_legacy_table(
                parsed, protected["src/radar/data/normative/" + case.resource_path],
                case.resource_path,
            )
            for name, raw in {**originals, "MSVBVM50.dll": runtime_bytes,
                              "SR.IN": expected_input}.items():
                if (work / name).read_bytes() != raw:
                    raise ValueError("Native execution changed a copied runtime/input file.")
            repeats.append({
                "repeat": repeat, "command": command,
                "command_file": _identity(work / "command.json"),
                "input": _identity(work / "SR.IN"), "output": _identity(work / "srim_output.txt"),
                "parsed": asdict(parsed), "semantic_sha256": parsed.semantic_sha256,
                "comparison": comparison,
            })
            print(f"{case.name} repeat {repeat}: {len(parsed.rows)} rows", flush=True)
        if repeats[0]["semantic_sha256"] != repeats[1]["semantic_sha256"]:
            raise ValueError(f"SRIM repeated numeric/metadata result changed: {case.name}.")
        case_records.append({
            "case": asdict(case), "repeats": repeats,
            "expected_control_grid_mev": expected_control_grid(case),
            "complete_native_control_grid_verified": True,
            "planned_control_energy_max_mev": Decimal("10000") if case.name == "h_al"
                else case.energy_max_mev,
            "planned_control_range_covered": case.name != "h_al",
            "native_upper_range_limitation": (
                "H/Al 0.1..10000 MeV probe displayed native Run-time error '6': Overflow. "
                "This fixed control uses 0.1..1000 MeV only; the maximum supported energy "
                "is not established. Unmatched higher legacy nodes remain unvalidated."
                if case.name == "h_al" else None
            ),
            "raw_output_bytes_identical": repeats[0]["output"]["sha256"] == repeats[1]["output"]["sha256"],
            "semantic_outputs_identical": True,
        })
    if any((source / name).read_bytes() != raw for name, raw in originals.items()):
        raise ValueError("Original SRIM source changed during controls.")
    if (source / "SR.IN").read_bytes() != original_input:
        raise ValueError("Original user SRIM input changed during controls.")
    if private_runtime.read_bytes() != runtime_bytes:
        raise ValueError("Original private runtime changed during controls.")
    if any((ROOT / path).read_bytes() != raw for path, raw in protected.items()):
        raise ValueError("Implementation or selected production resource changed during controls.")
    report = {
        "schema_version": "radar.srim_normal_mode_controls.v1",
        "source_directory": str(source.resolve()),
        "source_sha256": SOURCE_HASHES, "original_input_sha256": _sha(original_input),
        "private_runtime": _identity(private_runtime),
        "selected_files_sha256_before": {path: _sha(raw) for path, raw in protected.items()},
        "selected_files_sha256_after": {path: _sha((ROOT / path).read_bytes()) for path in protected},
        "cases": case_records, "case_count": len(case_records), "runs_per_case": 2,
        "source_files_unchanged": True, "selected_production_files_unchanged": True,
        "production_data_modified": False, "release_status_changed": False,
        "physical_accuracy_verified": False, "normative_export_allowed": False,
        "diagnostic_status": "INCOMPLETE",
        "all_planned_control_ranges_covered": False,
        "limitations": [
            "User-supplied pinned binary and normal data, not independently rebuilt source code.",
            "No underscore-file substitution or special-energy-list mode.",
            "Raw numeric tokens and decimal locale retained; no Windows locale changes.",
            "Normal module reports projected range, not CSDA path range.",
            "Comparison is matched-node diagnostic evidence, not historical source authentication.",
            "Proton/Al native range was restricted after Overflow at the intended upper bound; "
            "no successful result is claimed for the failed interval or the entire legacy domain.",
            "Engine/version labels differ from parent-package label and are not rewritten.",
            "Target masses/densities are explicit controls, not recovered historical SRIM settings.",
            "The module help warns about nuclear-stopping differences from full GUI tables.",
            "Private VB runtime only; OS dependencies are not a hermetic environment.",
            "No copying of the runtime into production, no system installation or registry changes.",
        ],
    }
    _write_new(output / "report.json", _json(report).encode("utf-8"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(r"C:\Users\Иван\Desktop\ПО\SRIM-2013\SR Module"))
    parser.add_argument("--private-runtime", type=Path,
                        default=ROOT / "outputs/srim_runtime_dependencies_20261101/MSVBVM50.dll")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_controls(source=args.source, private_runtime=args.private_runtime, output=args.output)
    print(_json({"report": str(args.output / "report.json"), "case_count": report["case_count"],
                 "normative_export_allowed": report["normative_export_allowed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
