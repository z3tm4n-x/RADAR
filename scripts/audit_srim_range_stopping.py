"""Replay pinned SRIM Al controls against anchored integrals of their stopping.

No native program is run and no production table is generated. Projected ranges
remain projected; legacy ranges remain undefined. Cross-dataset comparisons
cannot establish the ancestry, meaning or accuracy of the old tables.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_srim_coverage as coverage  # noqa: E402
from radar.validation.range_stopping_consistency import (  # noqa: E402
    audit_range_stopping_consistency,
)
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output  # noqa: E402

REPORT_PINS = {
    "baseline": "8ed401964181371ebb508f16bec41aa5b3600b136f47f763be4c0596f448c9cc",
    "upper": "3bbb3992b432074860b049820b065d8f47dd19d5cfc9cd074d8541f41754c029",
    "boundary": "4bd0adfcf7dbe1320ace41f34113cf50ffa846c5c438375b63a591acbb1ddd24",
}
SELECTION = (("baseline", "he4_al"),) + tuple(
    ("upper", f"{symbol.lower()}{mass}_al")
    for z, symbol, mass in coverage.LEGACY_IONS if z != 2
) + (("boundary", "h_al_4000"),)
IMPLEMENTATION = (
    "scripts/audit_srim_range_stopping.py",
    "src/radar/validation/range_stopping_consistency.py",
)
_CONVERSION_CONTEXT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_identity(item: dict[str, Any]) -> Path:
    path = Path(item["path"])
    if _sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
        raise ValueError(f"Retained file identity mismatch: {path}")
    return path


def _verify_report(catalog: str) -> tuple[dict[str, Any], dict[str, SrimRangeOutput]]:
    path = ROOT / f"outputs/srim_legacy_{catalog}_final_20261101/report.json"
    if _sha(path) != REPORT_PINS[catalog]:
        raise ValueError(f"Pinned report mismatch: {catalog}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["protected_sha256_before"] != report["protected_sha256_after"]:
        raise ValueError("Original native run recorded a protected-file mutation.")
    for raw, expected in report["protected_sha256_after"].items():
        if _sha(Path(raw)) != expected:
            raise ValueError(f"Protected file changed since native control: {raw}")
    parsed_by_case = {}
    for case in report["cases"]:
        repeated = []
        for repeat in case["repeats"]:
            for key in ("input", "output", "command_file"):
                if repeat[key] is not None:
                    _check_identity(repeat[key])
            if repeat["status"] != "VALID_NATIVE_OUTPUT":
                if "parsed" in repeat or "comparison" in repeat:
                    raise ValueError("Failed output has numerical results.")
                continue
            native = parse_srim_range_output(Path(repeat["output"]["path"]).read_bytes())
            if (native.semantic_sha256 != repeat["semantic_sha256"]
                    or json.loads(coverage._json(asdict(native))) != repeat["parsed"]):
                raise ValueError("Parsed native output differs from retained report.")
            control = next(c for c in coverage.CATALOGS[catalog]
                           if c.name == case["case"]["name"])
            if tuple(row.energy_total_mev for row in native.rows) != (
                    coverage.expected_control_grid(control)):
                raise ValueError("Native energy grid is incomplete.")
            repeated.append(native)
        if case["status"] == "CONTROL_COMPLETE":
            if len(repeated) != 2 or repeated[0].semantic_sha256 != repeated[1].semantic_sha256:
                raise ValueError("Complete control lacks identical valid repeats.")
            parsed_by_case[case["case"]["name"]] = repeated[0]
    return report, parsed_by_case


def _compare_case(native: SrimRangeOutput, comparison: dict[str, Any]) -> dict[str, Any]:
    """Use shared native nodes only; legacy roundtrip policy stays in its ledger."""
    if native.target_z != 13 or native.target_symbol != "Al":
        raise ValueError("Only Al range/stopping controls are in scope.")
    matched = [row for row in comparison["rows"] if "comparisons" in row]
    by_energy = {row.energy_total_mev: row for row in native.rows}
    energies = tuple(Decimal(row["native_energy_total_mev"]) for row in matched)
    if (len(set(energies)) != len(energies) or len(energies) < 2
            or any(b <= a for a, b in zip(energies, energies[1:]))
            or any(e not in by_energy for e in energies)):
        raise ValueError("Strictly increasing unique shared energy nodes are required.")
    with localcontext(_CONVERSION_CONTEXT):
        stops = tuple(by_energy[e].stopping_total_mev_cm2_mg * 1000 for e in energies)
        projected = tuple(by_energy[e].projected_range_cm * native.density_g_cm3 for e in energies)
    legacy = tuple(Decimal(row["comparisons"][
        "range_g_cm2_vs_projected_at_native_density"]["legacy"]) for row in matched)
    bands: dict[str, Any] = {}
    for label, lower, upper in (
        ("whole_shared_support", energies[0], energies[-1]),
        ("through_10000_mev", energies[0], Decimal(10000)),
        ("from_10000_mev", Decimal(10000), energies[-1]),
    ):
        indices = tuple(i for i, e in enumerate(energies) if lower <= e <= upper)
        if len(indices) < 2:
            bands[label] = {"status": "NOT_COVERED", "node_count": len(indices)}
            continue
        args = {
            "energy_total_mev": tuple(energies[i] for i in indices),
            "total_stopping_mev_cm2_g": tuple(stops[i] for i in indices),
        }
        bands[label] = {
            "status": "DIAGNOSTIC_ONLY", "node_count": len(indices),
            "native_projected_vs_native_total_stopping": asdict(audit_range_stopping_consistency(
                **args, range_g_cm2=tuple(projected[i] for i in indices), range_kind="projected",
            )),
            "legacy_unknown_range_vs_native_total_stopping": asdict(audit_range_stopping_consistency(
                **args, range_g_cm2=tuple(legacy[i] for i in indices), range_kind="unverified",
            )),
        }
    return {
        "ion_z": native.ion_z, "ion_mass_u": native.ion_mass_u,
        "native_output_sha256": native.raw_sha256, "density_g_cm3": native.density_g_cm3,
        "legacy_csv_sha256": comparison["legacy_csv_sha256"],
        "native_range_kind": native.range_kind,
        "energy_coordinate": "native_total_ion_kinetic_energy_MeV_no_division_by_A",
        "stopping_conversion": "1000*(electronic+nuclear)_MeV_cm2_mg",
        "range_conversion": "printed_projected_range_cm*printed_density_g_cm3",
        "legacy_energy_match_policy": comparison["energy_match_policy"],
        "bands": bands, "normative_export_allowed": False,
    }


def run_audit() -> dict[str, Any]:
    implementations = {relative: _sha(ROOT / relative) for relative in IMPLEMENTATION}
    verified = {name: _verify_report(name) for name in REPORT_PINS}
    results = []
    for catalog, name in SELECTION:
        report, parsed = verified[catalog]
        case = next(c for c in report["cases"] if c["case"]["name"] == name)
        if case["status"] != "CONTROL_COMPLETE" or name not in parsed:
            raise ValueError(f"Required control is incomplete: {catalog}/{name}")
        results.append({"catalog": catalog, "case": name,
                        **_compare_case(parsed[name], case["repeats"][0]["comparison"])})
    for catalog, (report, _) in verified.items():
        path = ROOT / f"outputs/srim_legacy_{catalog}_final_20261101/report.json"
        if _sha(path) != REPORT_PINS[catalog]:
            raise ValueError("Report changed during audit.")
        for raw, expected in report["protected_sha256_after"].items():
            if _sha(Path(raw)) != expected:
                raise ValueError("Protected file changed during audit.")
        for case in report["cases"]:
            for repeat in case["repeats"]:
                for key in ("input", "output", "command_file"):
                    if repeat[key] is not None:
                        _check_identity(repeat[key])
    if implementations != {relative: _sha(ROOT / relative) for relative in IMPLEMENTATION}:
        raise ValueError("Diagnostic implementation changed during audit.")
    return {
        "schema_version": "radar.srim_anchored_range_stopping.v1",
        "report_sha256": REPORT_PINS, "implementation_sha256": implementations,
        "selected_controls": results,
        "unavailable_native_controls": [
            {"catalog": name, "case": c["case"], "status": c["status"]}
            for name, (report, _) in verified.items() for c in report["cases"]
            if c["status"] != "CONTROL_COMPLETE"
        ],
        "diagnostic_status": "INCOMPLETE", "normative_export_allowed": False,
        "production_data_modified": False, "physical_accuracy_verified": False,
        "limitations": [
            "Only differences between tabulated nodes; no absolute range or low-energy origin.",
            "Two declared interpolants are numerical alternatives, not SRIM's internal algorithm.",
            "Printed precision and interpolation differences are not physical uncertainty bounds.",
            "Projected range is not reclassified CSDA by numerical agreement.",
            "Legacy-vs-native is cross-dataset: old stopping, origin and range meaning are unknown.",
            "Electronic plus elastic nuclear stopping excludes separate nuclear reaction transport.",
            "Shared legacy-ion nodes do not establish required source/mass/energy coverage.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    report = run_audit()
    payload = coverage._json(report).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"path": str(args.output), "sha256": hashlib.sha256(payload).hexdigest(),
                      "controls": len(report["selected_controls"]),
                      "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
