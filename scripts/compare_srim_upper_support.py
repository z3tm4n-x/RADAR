"""Compare the fixed upper SRIM pilot or full roster without erasing unresolved tails."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from math import isclose
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts import compare_srim_lower_support as shared  # noqa: E402
from radar.shielding.srim_dataset import load_srim_hze_dataset  # noqa: E402
from radar.validation import srim_source_catalog as catalog  # noqa: E402

PILOT_Z = {2, 10, 26, 92}
SHIELDS = (0., .01, .1, 1., 3., 10.)


def _full_roster() -> tuple[dict[int, tuple[str, Decimal]], set[int]]:
    controls = catalog.build_source_catalog(ROOT)
    ions = {c.ion_z: (c.symbol, c.ion_mass_u) for c in controls}
    sep = {r["z"] for r in catalog._source_inventory()["projectiles"]
           if "SEP" in r["source_membership"]}
    if len(ions) != 84 or len(sep) != 24 or not sep <= set(ions):
        raise ValueError("Fixed source inventory must contain 84 GCR and 24 SEP ions.")
    return ions, sep


def compare(*, parent_bundle: Path, parent_sha: str, upper_bundle: Path, upper_sha: str,
            parent_products: Path, parent_products_sha: str,
            upper_products: Path, upper_products_sha: str,
            require_full_inventory: bool = False) -> dict[str, Any]:
    if type(require_full_inventory) is not bool:
        raise ValueError("Full-inventory selection must be boolean.")
    paths: tuple[Path, ...] = (parent_bundle, upper_bundle, parent_products, upper_products,
                              Path(__file__), Path(shared.__file__))
    if require_full_inventory:
        paths += (Path(catalog.__file__), ROOT / catalog.MANIFEST_PATH,
                  *(ROOT / p for p in catalog.SOURCE_HASHES))
    pins = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old = load_srim_hze_dataset(parent_bundle, expected_sha256=parent_sha)
    new = load_srim_hze_dataset(upper_bundle, expected_sha256=upper_sha)
    full_roster, sep_roster = _full_roster() if require_full_inventory else ({}, {2, 10, 26})
    selected = set(full_roster) if require_full_inventory else PILOT_Z
    if require_full_inventory and (set(old.tables_by_z) != selected or set(new.tables_by_z) != selected):
        raise ValueError("Full comparison requires exactly the declared 84 source-ion pairs.")
    if not require_full_inventory and (set(new.tables_by_z) != PILOT_Z or not PILOT_Z <= set(old.tables_by_z)):
        raise ValueError("Expected exactly the four-ion upper pilot and its parent pairs.")
    tables = []
    for z in sorted(selected):
        parent, upper = old.tables_by_z[z], new.tables_by_z[z]
        for target in ("native_al", "native_si"):
            p, n = getattr(parent, target), getattr(upper, target)
            if p.ion_z != z or n.ion_z != z:
                raise ValueError("Native ion does not match its table key.")
            if require_full_inventory and p.ion_mass_u != full_roster[z][1]:
                raise ValueError("Native mass differs from the literal OST source parameter.")
            for field in ("ion_mass_u", "target_z", "density_g_cm3", "atomic_density_cm3",
                          "bragg_correction_percent", "engine_label", "version_label", "conversion_factors"):
                if getattr(p, field) != getattr(n, field):
                    raise ValueError(f"Changed native identity: {field}")
            if len(n.rows) <= len(p.rows) or n.rows[:len(p.rows)] != p.rows:
                raise ValueError("Whole upper table must retain every parent row and token exactly.")
        tables.append({"z": z, "source_mass": str(parent.mass_parameter_u_decimal),
                       "minimum_total_energy_mev": str(parent.native_al.rows[0].energy_total_mev),
                       "old_maximum_total_energy_mev": str(parent.native_al.rows[-1].energy_total_mev),
                       "new_maximum_total_energy_mev": str(upper.native_al.rows[-1].energy_total_mev)})
    a, b = shared._read(parent_products, parent_products_sha), shared._read(upper_products, upper_products_sha)
    if a["bundle_sha256"] != parent_sha or b["bundle_sha256"] != upper_sha:
        raise ValueError("Product reports do not refer to the supplied bundles.")
    for field in ("mission", "source_grid_refinement", "scope"):
        if a[field] != b[field]:
            raise ValueError(f"Changed source calculation: {field}")
    if not (a["implementation_and_input_sha256_before"] == a["implementation_and_input_sha256_after"]
            == b["implementation_and_input_sha256_before"] == b["implementation_and_input_sha256_after"]):
        raise ValueError("Identical implementation/source fingerprints are required.")
    ca, cb = shared._keys(a), shared._keys(b)
    expected = {(source, mode, z, shield)
                for source, roster in (("gcr", selected), ("sep", sep_roster))
                for mode in ("peak", "cumulative") for z in roster for shield in SHIELDS}
    if require_full_inventory:
        if (len(expected) != 1296 or set(ca) != expected or set(cb) != expected
                or any(r.get("source_product_count") != 216
                       or r.get("requested_symbols", "missing") is not None for r in (a, b))):
            raise ValueError("Full comparison requires all 1296 cases and 216 unfiltered source inputs.")
    elif (set(cb) != expected or not expected <= set(ca) or b.get("source_product_count") != 14
            or set(b.get("requested_symbols") or ()) != {"He", "Ne", "Fe", "U"}):
        raise ValueError("Expected all 84 fixed pilot cases, not a partial or full-roster claim.")
    rows = []
    for key, n in cb.items():
        p = ca[key]
        lp, ln = p["primary_ledger"], n["primary_ledger"]
        total = lp["source_integral"]
        tolerance = 2e-11 * total
        def equal(x: float, y: float) -> bool:
            return isclose(x, y, rel_tol=2e-11, abs_tol=tolerance)
        if total < 0 or not equal(total, ln["source_integral"]):
            raise ValueError("Declared source integral changed.")
        recovered = lp["outside_al_domain_integral"] - ln["outside_al_domain_integral"]
        gain = ln["supported_exit_integral"] - lp["supported_exit_integral"]
        low_gain = ln["unresolved_low_residual_integral"] - lp["unresolved_low_residual_integral"]
        if (min(recovered, gain, low_gain) < -tolerance or not equal(recovered, gain + low_gain)):
            raise ValueError("Recovered upper source measure is not conservatively partitioned.")
        for ledger in (lp, ln):
            fields = ("represented_integral", "below_output_window_integral", "above_output_window_integral",
                      "outside_al_domain_integral", "unresolved_low_residual_integral")
            if any(ledger[f] < 0 for f in fields) or not equal(sum(ledger[f] for f in fields), total):
                raise ValueError("Primary integral ledger is invalid.")
            if not equal(ledger["supported_exit_integral"], sum(ledger[f] for f in fields[:3])):
                raise ValueError("Supported primary integral disagrees with its output-window partition.")
        for name in ("energy_product", "let_product"):
            ps, ns = p[name]["spectrum"], n[name]["spectrum"]
            for field in ("x", "bin_edges", "x_unit", "y_unit", "assessment_mode", "quantity",
                          "temporal_basis", "mission_duration_s", "angular_basis", "source", "species"):
                if ps[field] != ns[field]:
                    raise ValueError(f"Changed output grid/unit/source identity: {name}/{field}")
            if (ps["species"]["atomic_number"] != key[2] or ps["source"] != key[0]
                    or ps["assessment_mode"] != key[1]):
                raise ValueError("Product identity differs from its case key.")
            if require_full_inventory and ps["species"]["symbol"] != full_roster[key[2]][0]:
                raise ValueError("Product symbol differs from the fixed source roster.")
            for index, (pv, nv) in enumerate(zip(ps["y"], ns["y"], strict=True)):
                width = ps["bin_edges"][index+1] - ps["bin_edges"][index] if ps["bin_edges"] is not None else 1.
                if (nv-pv)*width < -tolerance:
                    raise ValueError("Previously supported bin contribution was lost.")
                if key[3] == 0 and name == "energy_product" and pv != nv:
                    raise ValueError("Zero-Al energy identity changed.")
        let_gain = n["let_ledger"]["mapped_integral"] - p["let_ledger"]["mapped_integral"]
        # At zero Al shielding energy is already complete: new Si coverage may
        # still recover LET counts even though Al supported gain is zero.
        if let_gain < -tolerance or let_gain > gain + p["let_ledger"]["outside_si_domain_integral"] + tolerance:
            raise ValueError("Unexpected LET gain relative to available source measure.")
        rows.append({"source": key[0], "mode": key[1], "z": key[2], "thickness_g_cm2": key[3],
                     "source_integral": total, "recovered_al_source_integral": recovered,
                     "new_supported_integral": gain, "new_below_anchor_integral": low_gain,
                     "new_mapped_let_integral": let_gain,
                     "new_mapped_let_fraction_of_source": let_gain/total if total else None,
                     "mapped_let_relative_gain": let_gain/p["let_ledger"]["mapped_integral"]
                         if p["let_ledger"]["mapped_integral"] else None,
                     "remaining_outside_al_integral": ln["outside_al_domain_integral"],
                     "remaining_outside_si_integral": n["let_ledger"]["outside_si_domain_integral"]})
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha for p, sha in pins.items()):
        raise ValueError("Comparison evidence changed during calculation.")
    return {"schema_version": "radar.srim_upper_support_comparison.v1", "status": "INCOMPLETE",
            "scope": "full_84_ion_upper_extension_not_orbit_acceptance" if require_full_inventory else
                "four_ion_upper_pilot_not_full_roster", "normative_export_allowed": False,
            "pilot_inventory_verified": not require_full_inventory,
            "full_inventory_verified": require_full_inventory, "case_count": len(rows), "cases": rows,
            "tables": tables, "input_sha256": pins,
            "limitations": ["Counts are not LET/dose/SEE physical-error bounds.",
                            "Remaining high-energy tails and range origin are unresolved.",
                            "No orbital acceptance or source-model qualification is performed."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-bundle", "upper-bundle", "parent-products", "upper-products"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-full-inventory", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(parent_bundle=args.parent_bundle, parent_sha=args.parent_bundle_sha256,
                     upper_bundle=args.upper_bundle, upper_sha=args.upper_bundle_sha256,
                     parent_products=args.parent_products, parent_products_sha=args.parent_products_sha256,
                     upper_products=args.upper_products, upper_products_sha=args.upper_products_sha256,
                     require_full_inventory=args.require_full_inventory)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"cases": result["case_count"], "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
