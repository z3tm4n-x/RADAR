"""Exact-node 1000..5000 MeV/source-A SRIM/CATIMA diagnostic; no model selection."""
from __future__ import annotations

import argparse
from fractions import Fraction
from math import isclose
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_srim_source_overlap as shared  # noqa: E402
from radar.shielding.srim_dataset import load_srim_hze_dataset  # noqa: E402
from radar.validation.srim_source_catalog import build_source_catalog, MANIFEST_PATH  # noqa: E402
from radar.validation.srim_output import SrimRangeOutput  # noqa: E402

BUNDLE = ROOT / "outputs/srim_upper_full_20261105/observed_upper_full.zip"
BUNDLE_SHA256 = "d4754dce59446a44d8c96ffe9853feb4102a70ade2e3e839737eaff63289434c"
LOWER, UPPER = Fraction(1000), Fraction(5000)
FLAGS = shared.FLAGS
COMPONENTS = shared.COMPONENTS


def build_inventory(selected: dict[str, SrimRangeOutput], catalog: Any) -> dict[str, Any]:
    expected = {(c.ion_z, c.target_z): c for c in catalog}
    if len(catalog) != 168 or len(expected) != 168 or len(selected) != 168:
        raise ValueError("Exactly 84 authenticated source ions in Al/Si are required.")
    requests: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for name, native in sorted(selected.items(), key=lambda item: (item[1].ion_z, item[1].target_z)):
        key = native.ion_z, native.target_z
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected full-roster ion/material.")
        seen.add(key)
        source = expected[key]
        if (native.ion_mass_u != source.ion_mass_u or native.density_g_cm3 != source.density_g_cm3
                or native.stopping_unit != "MeV/(mg/cm2)" or native.energy_basis != "total_ion_kinetic_energy"):
            raise ValueError("Native source A, material or units differ from the authenticated catalog.")
        count, below, above = 0, 0, 0
        for index, row in enumerate(native.rows):
            energy = Fraction(row.energy_total_mev)/Fraction(native.ion_mass_u)
            if energy < LOWER:
                below += 1
            elif energy > UPPER:
                above += 1
            else:
                requests.append({"case_name": name, "raw_sha256": native.raw_sha256,
                    "native_row_index": index, "z": native.ion_z,
                    "mass_parameter_u_decimal": format(native.ion_mass_u, "f"),
                    "material_z": native.target_z, "energy_total_mev_decimal": format(row.energy_total_mev, "f"),
                    "energy_mev_per_nucleon": float(energy)})
                count += 1
        if not count or above:
            raise ValueError("Each bounded upper table must have high-band support and no node above 5000/source-A.")
        tables.append({"case_name": name, "z": native.ion_z, "material_z": native.target_z,
            "mass_parameter_u_decimal": format(native.ion_mass_u, "f"), "raw_sha256": native.raw_sha256,
            "srim_density_g_cm3": str(native.density_g_cm3), "native_row_count": len(native.rows),
            "matched_row_count": count, "excluded_below_band_count": below, "excluded_above_band_count": above,
            "maximum_mev_per_source_a": float(Fraction(native.rows[-1].energy_total_mev)/Fraction(native.ion_mass_u))})
    if seen != set(expected) or len({z for z, _ in seen}) != 84:
        raise ValueError("Incomplete full source roster.")
    return {"requests": requests, "tables": tables, "selected_native_table_count": 168,
            "ion_count": 84, "matched_row_count": len(requests),
            "native_row_count": sum(t["native_row_count"] for t in tables),
            "excluded_row_count": sum(t["excluded_below_band_count"] + t["excluded_above_band_count"] for t in tables)}


def reconstruct_plan() -> tuple[dict[str, SrimRangeOutput], dict[str, Any]]:
    dataset = load_srim_hze_dataset(BUNDLE, expected_sha256=BUNDLE_SHA256)
    catalog = build_source_catalog(ROOT)
    selected = {}
    for record in dataset.manifest["tables"]:
        pair = dataset.tables_by_z[record["z"]]
        selected[record["case_name"]] = pair.native_al if record["material_z"] == 13 else pair.native_si
    inventory = build_inventory(selected, catalog)
    plan = {"schema_version": "radar.srim_catima_upper_overlap_requests.v1", **FLAGS,
        "bundle_sha256": BUNDLE_SHA256,
        "source_manifest_sha256": shared.digest((ROOT / MANIFEST_PATH).read_bytes()),
        "planner_sha256": shared.digest(Path(__file__).read_bytes()),
        "energy_lower_mev_per_source_a_inclusive": 1000,
        "energy_upper_mev_per_source_a_inclusive": 5000,
        "mass_policy": "literal_source_A_as_numeric_transport_mass_no_isotope_rounding",
        "stopping_conversion": "SRIM_MeV_cm2_mg_times_1000_no_extra_A_factor",
        "exclusion_policy": "uncompared_native_nodes_not_removed_from_source_spectra", **inventory}
    return selected, plan


def compare_rows(selected: dict[str, SrimRangeOutput], requests: list[dict[str, Any]], rows: Any) -> list[dict[str, Any]]:
    import generate_srim_upper_overlap_controls as worker
    worker.validate_requests(requests)
    if not isinstance(rows, list) or len(rows) != len(requests):
        raise ValueError("Missing or extra exact-node samples.")
    result = []
    for request, row in zip(requests, rows, strict=True):
        if not isinstance(row, dict) or set(row) != set(request) | worker.STOPPING_FIELDS:
            raise ValueError("Unexpected native sample fields.")
        shared.same({k: row[k] for k in request}, request, "sample identity/order")
        values = {c: shared.positive(row[f"stopping_{c}_mev_cm2_g"], zero=c in ("nuclear", "radiative"))
                  for c in (*COMPONENTS, "total", "radiative")}
        if (values["radiative"] != 0 or values["total"] != values["nonradiative"] or not isclose(
                values["nonradiative"], values["electronic"]+values["nuclear"], rel_tol=2e-14, abs_tol=0)):
            raise ValueError("Inconsistent stopping decomposition.")
        native = selected[request["case_name"]].rows[request["native_row_index"]]
        srim = {"electronic": float(Fraction(native.electronic_mev_cm2_mg)*1000),
                "nuclear": float(Fraction(native.nuclear_mev_cm2_mg)*1000),
                "nonradiative": float((Fraction(native.electronic_mev_cm2_mg)+Fraction(native.nuclear_mev_cm2_mg))*1000)}
        energy = Fraction(request["energy_total_mev_decimal"])/Fraction(request["mass_parameter_u_decimal"])
        band = "[1000,3000)" if energy < 3000 else "[3000,4400)" if energy < 4400 else "[4400,5000]"
        result.append({**request, "energy_band": band, "srim_printed_tokens": list(native.printed_tokens),
            "components": {c: {"srim_mev_cm2_g": srim[c], "catima_mev_cm2_g": values[c],
                "catima_over_srim_minus_one": values[c]/srim[c]-1 if srim[c] else None,
                "difference_mev_cm2_g": values[c]-srim[c]} for c in COMPONENTS}})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "compare"), required=True)
    for name in ("requests", "native"):
        parser.add_argument(f"--{name}", type=Path)
        parser.add_argument(f"--{name}-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    supplied = (args.requests, args.requests_sha256, args.native, args.native_sha256)
    if (args.mode == "plan" and any(supplied)) or (args.mode == "compare" and not all(supplied)):
        raise ValueError("Compare requires both pinned inputs; plan accepts neither.")
    selected, plan = reconstruct_plan()
    result = plan
    if args.mode == "compare":
        import generate_srim_upper_overlap_controls as worker
        shared.same(shared.read_pinned(args.requests, args.requests_sha256), plan, "reconstructed request plan")
        report = shared.read_pinned(args.native, args.native_sha256)
        worker.validate_report(report, args.requests, args.requests_sha256, plan)
        rows = compare_rows(selected, plan["requests"], report["rows"])
        result = {"schema_version": "radar.srim_catima_upper_stopping_overlap.v1", **FLAGS,
            "bundle_sha256": BUNDLE_SHA256, "requests_sha256": args.requests_sha256,
            "native_sha256": args.native_sha256, "matched_row_count": len(rows),
            "selected_native_table_count": 168, "ion_count": 84, "srim_table_metadata": plan["tables"],
            "catima_materials": report["materials"], "rows": rows, "summary": shared.summarize(rows),
            "difference_definition": "CATIMA/SRIM-1; dimensionless; neither is experimental truth",
            "limitations": ["Model comparison only; no physical acceptance tolerance, model selection or splice.",
                "Literal source A and default LS/Pierce-Blann configuration remain unchanged and unqualified.",
                "Native material constants differ; no I, density, isotope or stopping fit.",
                "No range, spectrum, dose, SEE calculation or high-energy tail closure.",
                "Node-count statistics are not source-spectrum-weighted errors."]}
        shared.read_pinned(args.requests, args.requests_sha256)
        shared.read_pinned(args.native, args.native_sha256)
    _, after = reconstruct_plan()
    shared.same(after, plan, "source evidence after comparison")
    shared.write_new(args.output, result)
    print({"sha256": shared.digest(args.output.read_bytes()), "matched_rows": result["matched_row_count"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
