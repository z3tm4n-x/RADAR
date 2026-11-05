"""Authenticated exact-node SRIM/CATIMA stopping overlap; never a release gate.

Plan/replay modes do not load a native calculator. The companion CPython-3.12
worker samples the existing pinned CATIMA candidate at the retained SRIM nodes.
No range comparison, table splice, fitting, or external energy extrapolation is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path
import re
from statistics import median
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srim_source_extension as extension  # noqa: E402
from radar.validation.srim_output import SrimRangeOutput  # noqa: E402

EXTENSION = "outputs/srim_source_extension_full_20261101/report.json"
EXTENSION_SHA256 = "65dfd1a31ddc3e522da08b174dd8248d0be2c4ce35b78ca9ffece38401031802"
SUMMARY = "outputs/srim_source_extension_full_summary_20261101.json"
SUMMARY_SHA256 = "658b48c80abfe752a0057cd644dfbfcd8af25d0905025d67f8c71943e6065ce5"
PARENT_PINS = {
    "scripts/run_srim_source_extension.py":
        "34f5120aac44e02c42048a2dc6c8f3393e8bf8f9492008e31dc391ce0fafc92f",
    "src/radar/validation/srim_source_extension.py":
        "5d5eaee0ccfd61c3f7f8ffdbeb6e418da087fe64a3a293133d75873bf53cbb5d",
}
FLAGS = {"status": "INCOMPLETE", "normative_export_allowed": False,
         "physical_accuracy_verified": False, "production_tables_replaced": False,
         "source_energy_domain_covered": False, "source_grids_changed": False,
         "high_energy_model_selected": False, "table_splicing_performed": False,
         "range_comparison_performed": False, "source_spectrum_convolution_performed": False}
LOWER, UPPER = Fraction(1, 100), Fraction(3000)
COMPONENTS = ("electronic", "nuclear", "nonradiative")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def read_pinned(path: Path, sha: str) -> dict[str, Any]:
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("A lowercase SHA256 is required.")
    raw = path.read_bytes()
    if digest(raw) != sha:
        raise ValueError(f"Pinned overlap input changed: {path}")
    # The parent rejects duplicate keys and NaN/Infinity; no native launch.
    from scripts.summarize_srim_source_campaign import _load
    return dict(_load(raw))


def same(actual: Any, expected: Any, label: str) -> None:
    if encode(actual) != encode(expected):
        raise ValueError(f"Changed overlap {label}.")


def reconstruct() -> tuple[dict[str, SrimRangeOutput], Any]:
    print("Authenticating retained SRIM baseline and runtime copies...", file=sys.stderr, flush=True)
    for name, sha in PARENT_PINS.items():
        if digest((ROOT / name).read_bytes()) != sha:
            raise ValueError("Frozen native extension implementation changed.")
    helper, evidence, _, baselines, catalog = extension._context()
    print("Authenticating retained SRIM extensions...", file=sys.stderr, flush=True)
    report, extended = extension.replay_extension(
        ROOT / EXTENSION, EXTENSION_SHA256, helper=helper, evidence=evidence,
        baselines=baselines, catalog=catalog,
    )
    if report["mode"] != "full" or not report["all_native_controls_repeated"]:
        raise ValueError("Complete full repeated extension is required.")
    summary = read_pinned(ROOT / SUMMARY, SUMMARY_SHA256)
    selected = dict(baselines)
    selection = []
    for case in catalog:
        observed = extended.get(case.name)
        if observed is None:
            raise ValueError("A pinned full extension is missing an ion/material.")
        selected[case.base_source_name] = observed
        selection.append({"base_source_name": case.base_source_name, "extension_name": case.name,
                          "baseline_raw_sha256": baselines[case.base_source_name].raw_sha256,
                          "selected_raw_sha256": observed.raw_sha256,
                          "baseline_prefix_identical": True})
    same(selection, summary["table_selection"], "whole-table selection")
    if len(selected) != 168 or len(extended) != 158:
        raise ValueError("Unexpected full-source SRIM roster.")
    evidence.read(ROOT / SUMMARY, SUMMARY_SHA256)
    evidence.verify_unchanged()
    print("Authenticated 168 whole SRIM tables.", file=sys.stderr, flush=True)
    return selected, evidence


def build_inventory(selected: dict[str, SrimRangeOutput]) -> dict[str, Any]:
    """Keep every row identified: matched requests or explicit lower exclusions."""
    if not selected:
        raise ValueError("A nonempty selected SRIM inventory is required.")
    ordered = sorted(selected.items(), key=lambda item: (
        item[1].ion_z, item[1].ion_mass_u, item[1].target_z,
    ))
    seen: set[tuple[int, Decimal, int]] = set()
    requests, excluded, tables = [], [], []
    for name, table in ordered:
        identity = table.ion_z, table.ion_mass_u, table.target_z
        if identity in seen or table.ion_z < 2 or table.target_z not in (13, 14):
            raise ValueError("Repeated or unexpected ion/material identity.")
        seen.add(identity)
        if table.stopping_unit != "MeV/(mg/cm2)" or table.energy_basis != "total_ion_kinetic_energy":
            raise ValueError("Unexpected SRIM units or energy basis.")
        count = 0
        for index, row in enumerate(table.rows):
            energy = Fraction(row.energy_total_mev) / Fraction(table.ion_mass_u)
            record = {"case_name": name, "raw_sha256": table.raw_sha256,
                      "native_row_index": index, "z": table.ion_z,
                      "mass_parameter_u_decimal": format(table.ion_mass_u, "f"),
                      "material_z": table.target_z,
                      "energy_total_mev_decimal": format(row.energy_total_mev, "f"),
                      "energy_mev_per_nucleon": float(energy)}
            if energy < LOWER or energy > UPPER:
                excluded.append({**record, "reason": "below_declared_candidate_overlap"
                                 if energy < LOWER else "above_declared_candidate_overlap"})
            else:
                requests.append(record)
                count += 1
        tables.append({"case_name": name, "raw_sha256": table.raw_sha256,
                       "z": table.ion_z, "mass_parameter_u_decimal": str(table.ion_mass_u),
                       "material_z": table.target_z, "srim_density_g_cm3": str(table.density_g_cm3),
                       "native_row_count": len(table.rows), "matched_row_count": count,
                       "excluded_row_count": len(table.rows) - count,
                       "srim_stopping_unit": table.stopping_unit,
                       "srim_energy_basis": table.energy_basis})
    if not requests:
        raise ValueError("No matched rows in the declared overlap.")
    return {"requests": requests, "excluded_native_rows": excluded, "tables": tables,
            "selected_native_table_count": len(tables), "matched_row_count": len(requests),
            "excluded_row_count": len(excluded), "native_row_count": len(requests) + len(excluded)}


def plan(selected: dict[str, SrimRangeOutput], evidence: Any) -> dict[str, Any]:
    return {"schema_version": "radar.srim_catima_overlap_requests.v1", **FLAGS,
            "extension_sha256": EXTENSION_SHA256, "selection_summary_sha256": SUMMARY_SHA256,
            "source_evidence_count": len(evidence.pins),
            "source_evidence_inventory_sha256": digest(encode(evidence.pins)),
            "planner_sha256": digest(Path(__file__).read_bytes()),
            "energy_lower_mev_per_source_a_inclusive": .01,
            "energy_upper_mev_per_source_a_inclusive": 3000.,
            "mass_policy": "literal_source_A_as_numeric_transport_mass_no_isotope_rounding",
            "stopping_conversion": "SRIM_MeV_cm2_mg_times_1000_no_extra_A_factor",
            "exclusion_policy": "uncompared_native_nodes_not_removed_from_source_spectra",
            **build_inventory(selected)}


def energy_band(energy: Fraction) -> str:
    if not LOWER <= energy <= UPPER:
        raise ValueError("Energy is outside the declared overlap.")
    for boundary, name in ((1, "[0.01,1)"), (10, "[1,10)"), (30, "[10,30)"),
                           (100, "[30,100)"), (1000, "[100,1000)")):
        if energy < boundary:
            return name
    return "[1000,3000]"


def positive(value: Any, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Finite positive stopping required; no clipping.")
    try:
        valid = isfinite(value) and (value >= 0 if zero else value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Finite positive stopping required; no clipping.")
    return float(value)


def compare_rows(selected: dict[str, SrimRangeOutput], rows: Any) -> list[dict[str, Any]]:
    requests = build_inventory(selected)["requests"]
    if not isinstance(rows, list) or len(rows) != len(requests):
        raise ValueError("Missing or extra exact native overlap rows.")
    results = []
    stopping_fields = {f"stopping_{c}_mev_cm2_g" for c in
                       (*COMPONENTS, "total", "radiative")}
    for request, row in zip(requests, rows, strict=True):
        if not isinstance(row, dict) or set(row) != set(request) | stopping_fields:
            raise ValueError("Changed native overlap row schema.")
        same({key: row[key] for key in request}, request, "native row identity/order")
        values = {c: positive(row[f"stopping_{c}_mev_cm2_g"], zero=c in ("nuclear", "radiative"))
                  for c in (*COMPONENTS, "total", "radiative")}
        if (values["radiative"] != 0 or not isclose(values["total"], values["nonradiative"], rel_tol=2e-14)
                or not isclose(values["nonradiative"], values["electronic"] + values["nuclear"],
                               rel_tol=2e-14)):
            raise ValueError("Inconsistent stopping decomposition/radiative term.")
        table = selected[request["case_name"]]
        native_row = table.rows[request["native_row_index"]]
        # Fraction avoids any ambient Decimal-context dependence during conversion.
        srim = {"electronic": float(Fraction(native_row.electronic_mev_cm2_mg) * 1000),
                "nuclear": float(Fraction(native_row.nuclear_mev_cm2_mg) * 1000)}
        srim["nonradiative"] = float((Fraction(native_row.electronic_mev_cm2_mg)
                                     + Fraction(native_row.nuclear_mev_cm2_mg)) * 1000)
        energy = Fraction(native_row.energy_total_mev) / Fraction(table.ion_mass_u)
        components = {c: {"srim_mev_cm2_g": srim[c], "catima_mev_cm2_g": values[c],
                          "catima_over_srim_minus_one": values[c] / srim[c] - 1. if srim[c] else None,
                          "difference_mev_cm2_g": values[c] - srim[c]} for c in COMPONENTS}
        results.append({**request, "energy_band": energy_band(energy), "components": components,
                        "srim_printed_tokens": list(native_row.printed_tokens)})
    return results


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No overlap rows to summarize.")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        material = str(row["material_z"])
        groups[("material", material, "all")].append(row)
        groups[("band", material, row["energy_band"])].append(row)
        groups[("ion_material", str(row["z"]), row["mass_parameter_u_decimal"], material)].append(row)
    result = []
    for group, items in sorted(groups.items()):
        components = {}
        for component in COMPONENTS:
            defined = [r for r in items if r["components"][component]["catima_over_srim_minus_one"]
                       is not None]
            worst = max(defined, key=lambda r: abs(r["components"][component]
                                                  ["catima_over_srim_minus_one"])) if defined else None
            relative = [r["components"][component]["catima_over_srim_minus_one"] for r in defined]
            components[component] = {
                "defined_relative_count": len(defined), "undefined_zero_reference_count": len(items) - len(defined),
                "median_signed_relative_difference": median(relative) if relative else None,
                "median_absolute_relative_difference": median(abs(v) for v in relative) if relative else None,
                "min_signed_relative_difference": min(relative) if relative else None,
                "max_signed_relative_difference": max(relative) if relative else None,
                "worst_absolute": None if worst is None else {
                    key: worst[key] for key in ("z", "mass_parameter_u_decimal", "material_z",
                                                "energy_total_mev_decimal", "energy_mev_per_nucleon",
                                                "case_name", "raw_sha256", "native_row_index")
                } | {"relative_difference": worst["components"][component]["catima_over_srim_minus_one"]},
            }
        result.append({"group": list(group), "row_count": len(items), "components": components})
    return result


def write_new(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(encode(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "compare"), required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--requests-sha256")
    parser.add_argument("--native", type=Path)
    parser.add_argument("--native-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    supplied = (args.requests, args.requests_sha256, args.native, args.native_sha256)
    if args.mode == "plan" and any(supplied) or args.mode == "compare" and not all(supplied):
        raise ValueError("Plan accepts no native inputs; compare requires both input paths/SHA256.")
    selected, evidence = reconstruct()
    request_plan = plan(selected, evidence)
    if args.mode == "plan":
        result = request_plan
    else:
        stored = read_pinned(args.requests, args.requests_sha256)
        same(stored, request_plan, "authenticated request plan")
        native = read_pinned(args.native, args.native_sha256)
        validate_native(native, args.requests, args.requests_sha256, request_plan)
        rows = compare_rows(selected, native["rows"])
        result = {"schema_version": "radar.srim_catima_stopping_overlap.v1", **FLAGS,
                  "requests_sha256": args.requests_sha256, "native_sha256": args.native_sha256,
                  "planner_sha256": request_plan["planner_sha256"],
                  "native_row_count": request_plan["native_row_count"], "matched_row_count": len(rows),
                  "excluded_native_rows": request_plan["excluded_native_rows"],
                  "selected_native_table_count": len(selected),
                  "source_evidence_inventory_sha256": request_plan["source_evidence_inventory_sha256"],
                  "catima_materials": native["materials"], "srim_table_metadata": request_plan["tables"],
                  "difference_definition": "CATIMA/SRIM-1; dimensionless; SRIM is comparator not truth",
                  "summary": summarize(rows), "rows": rows,
                  "limitations": [
                      "Model comparison, not independent experimental validation or OST errata.",
                      "Shared SRIM ancestry at low energy; no claimed source independence.",
                      "Literal source A retained; mass-policy and internal LS mass interpolation remain diagnostic.",
                      "Unchanged native material constants differ; no density, I or mass fitting.",
                      "SRIM printed precision is retained, not interpreted as measurement uncertainty.",
                      "Residual summaries are node-count descriptions, not flux/dose/SEE-weighted quantities.",
                      "No projected-range/CSDA equivalence, spectral convolution or high-energy tail closure.",
                      "No residual tolerance is a physical acceptance criterion; no splice was selected.",
                  ]}
    evidence.verify_unchanged()
    if args.mode == "compare":
        # Detect an input-file mutation during arithmetic, not just on first read.
        same(read_pinned(args.requests, args.requests_sha256), stored, "request after comparison")
        same(read_pinned(args.native, args.native_sha256), native, "native after comparison")
    write_new(args.output, result)
    print(json.dumps({"sha256": digest(args.output.read_bytes()),
                      "matched_rows": result["matched_row_count"], "status": "INCOMPLETE"}))
    return 0


def validate_native(payload: dict[str, Any], path: Path, sha: str, request_plan: dict[str, Any]) -> None:
    """Validate worker metadata without importing/loading its native extension."""
    import generate_srim_overlap_controls as controls
    controls.validate_report(payload, path, sha, request_plan)


if __name__ == "__main__":
    raise SystemExit(main())
