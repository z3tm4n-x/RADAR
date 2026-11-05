"""Compare a whole-table lower extension with its parent; never qualify stopping at rest.

Inputs are pinned native bundles and external-product reports. No native code
is executed and no range origin or unobserved stopping curve is invented.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from radar.shielding.srim_dataset import load_srim_hze_dataset  # noqa: E402
from radar.validation.srim_output import SrimRangeRow  # noqa: E402


def _constant(value: str) -> Any:
    raise ValueError(f"Nonfinite JSON value: {value}")


def _float(value: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError("Nonfinite JSON number.")
    return result


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _read(path: Path, sha: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Product report fingerprint mismatch.")
    result: dict[str, Any] = json.loads(raw, parse_constant=_constant, parse_float=_float,
                                      object_pairs_hook=_unique)
    if (not isinstance(result, dict)
            or result.get("schema_version") != "radar.srim_hze_external_products.v1"
            or result.get("normative_export_allowed") is not False
            or result.get("status") != "INCOMPLETE"):
        raise ValueError("Expected incomplete external-source product report.")
    return result


def _keys(report: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows = report["cases"]
    keyed = {(r["source"], r["mode"], r["z"], r["thickness_g_cm2"]): r for r in rows}
    if len(keyed) != len(rows) or len(rows) != report["case_count"]:
        raise ValueError("Duplicate cases or inconsistent case count.")
    return keyed


def _full_inventory(a: dict[str, Any], b: dict[str, Any], old_ions: set[int], ions: set[int]) -> None:
    ca, cb = _keys(a), _keys(b)
    sep = {z for source, mode, z, shield in ca if source == "sep"}
    expected = {(source, mode, z, shield)
                for source, roster in (("gcr", ions), ("sep", sep))
                for mode in ("peak", "cumulative") for z in roster
                for shield in (0., .01, .1, 1., 3., 10.)}
    if (len(ions) != 84 or ions != old_ions or len(sep) != 24 or not sep <= ions
            or len(expected) != 1296 or set(ca) != expected or set(cb) != expected
            or a.get("source_product_count") != 216 or b.get("source_product_count") != 216
            or a.get("requested_symbols", "missing") is not None
            or b.get("requested_symbols", "missing") is not None):
        raise ValueError("Full inventory requires 84 pairs, 216 inputs and all 1296 shield cases.")


def _low_residual_statement(count: float, minimum: Decimal, mode: str) -> dict[str, Any]:
    if (type(count) not in (int, float) or not isfinite(count) or count < 0
            or not isinstance(minimum, Decimal) or not minimum.is_finite() or minimum <= 0
            or mode not in ("peak", "cumulative")):
        raise ValueError("Invalid low-residual statement inputs.")
    # Exact decimal product of the REPORTED binary64 count and native decimal
    # endpoint. This is a model-conditional bound, not a physics uncertainty.
    with localcontext() as context:
        context.prec = 1200
        bound = Decimal.from_float(float(count)) * minimum
    return {"classification": "stopped_or_exit_below_observed_energy",
            "observed_minimum_total_energy_mev": str(minimum),
            "unresolved_particle_integral": count,
            "residual_kinetic_energy_upper_bound_decimal": str(bound),
            "residual_kinetic_energy_upper_bound_unit":
                "MeV cm^-2 s^-1" if mode == "peak" else "MeV cm^-2",
            "scope": "reported_unresolved_low_residual_only_in_primary_CSDA",
            "angular_basis": "same_as_source_omnidirectional_not_plane_crossing",
            "stopped_particle_classification_established": False,
            "is_let_dose_or_see_error_bound": False}


def compare(*, parent_bundle: Path, parent_sha: str, lower_bundle: Path, lower_sha: str,
            parent_products: Path, parent_products_sha: str,
            lower_products: Path, lower_products_sha: str,
            require_full_inventory: bool = False) -> dict[str, Any]:
    paths = (parent_bundle, lower_bundle, parent_products, lower_products, Path(__file__))
    pins = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old = load_srim_hze_dataset(parent_bundle, expected_sha256=parent_sha)
    new = load_srim_hze_dataset(lower_bundle, expected_sha256=lower_sha)
    a, b = _read(parent_products, parent_products_sha), _read(lower_products, lower_products_sha)
    if a["bundle_sha256"] != parent_sha or b["bundle_sha256"] != lower_sha:
        raise ValueError("Products do not refer to the supplied bundles.")
    for field in ("mission", "source_grid_refinement", "scope"):
        if a[field] != b[field]:
            raise ValueError(f"Source calculation changed: {field}")
    if a["implementation_and_input_sha256_before"] != a["implementation_and_input_sha256_after"]:
        raise ValueError("Parent inputs changed during calculation.")
    if (b["implementation_and_input_sha256_before"] != b["implementation_and_input_sha256_after"]
            or a["implementation_and_input_sha256_before"] != b["implementation_and_input_sha256_before"]):
        raise ValueError("Compared reports require identical implementation and source inputs.")
    tables = []
    for z, lower in new.tables_by_z.items():
        if z not in old.tables_by_z:
            raise ValueError("Lower extension added an unrelated ion.")
        parent = old.tables_by_z[z]
        for target in ("native_al", "native_si"):
            p, n = getattr(parent, target), getattr(lower, target)
            for field in ("ion_z", "ion_mass_u", "target_z", "density_g_cm3",
                          "atomic_density_cm3", "bragg_correction_percent", "engine_label",
                          "version_label", "conversion_factors"):
                if getattr(p, field) != getattr(n, field):
                    raise ValueError(f"Changed native identity: {field}")
            suffix = tuple(r for r in n.rows if r.energy_total_mev >= p.rows[0].energy_total_mev)
            def stopping(r: SrimRangeRow) -> tuple[Decimal, Decimal, Decimal]:
                return (r.energy_total_mev, r.electronic_mev_cm2_mg, r.nuclear_mev_cm2_mg)
            if (n.rows[0].energy_total_mev >= p.rows[0].energy_total_mev
                    or tuple(map(stopping, suffix)) != tuple(map(stopping, p.rows))):
                raise ValueError("Lower whole table does not preserve the stopping suffix.")
        with localcontext() as context:
            context.prec = 80
            offset = lower.range_at_energy(parent.al_energy[0])
            residual = max(abs(lower.range_at_energy(e) - parent.range_at_energy(e) - offset)
                           for e in parent.al_energy)
            if residual > Decimal("1e-70") * max(Decimal(1), offset):
                raise ValueError("Anchored range translation identity failed.")
        tables.append({"z": z, "source_mass": str(parent.mass_parameter_u_decimal),
                       "old_energy_min_mev_total": str(parent.native_al.rows[0].energy_total_mev),
                       "new_energy_min_mev_total": str(lower.native_al.rows[0].energy_total_mev),
                       "added_path_g_cm2": str(offset),
                       "range_translation_max_residual_g_cm2": str(residual)})
    ca, cb = _keys(a), _keys(b)
    if not cb or not set(cb) <= set(ca):
        raise ValueError("Lower report must contain existing source/ion/mode/shield cases.")
    if require_full_inventory:
        _full_inventory(a, b, set(old.tables_by_z), set(new.tables_by_z))
    rows = []
    for key, n in cb.items():
        p = ca[key]
        z = n["z"]
        if z not in new.tables_by_z:
            raise ValueError("Case has no lower-extension pair.")
        lp, ln = p["primary_ledger"], n["primary_ledger"]
        total = lp["source_integral"]
        tolerance = 2e-11 * total
        def equal(x: float, y: float) -> bool:
            return isclose(x, y, rel_tol=2e-11, abs_tol=tolerance)
        if (not equal(total, ln["source_integral"])
                or not equal(lp["outside_al_domain_integral"], ln["outside_al_domain_integral"])):
            raise ValueError("Source or upper outside-Al contribution changed.")
        gain = ln["supported_exit_integral"] - lp["supported_exit_integral"]
        loss = lp["unresolved_low_residual_integral"] - ln["unresolved_low_residual_integral"]
        if gain < -tolerance or not equal(gain, loss):
            raise ValueError("Supported/unresolved count partition is not monotone/conservative.")
        # Compare source identity at zero shield, and complete display bins above
        # the old lower native energy. Partial crossing bins may legitimately gain.
        for name in ("energy_product", "let_product"):
            ps, ns = p[name]["spectrum"], n[name]["spectrum"]
            for field in ("x", "bin_edges", "x_unit", "y_unit", "assessment_mode",
                          "quantity", "temporal_basis", "mission_duration_s"):
                if ps[field] != ns[field]:
                    raise ValueError(f"Changed output grid or source semantics: {name}/{field}")
            for index, (pv, nv) in enumerate(zip(ps["y"], ns["y"], strict=True)):
                width = (ps["bin_edges"][index+1] - ps["bin_edges"][index]
                         if ps["bin_edges"] is not None else 1.)
                if (nv-pv)*width < -tolerance:
                    raise ValueError("A previously supported output-bin contribution was lost.")
                unchanged = n["thickness_g_cm2"] == 0 or (
                    name == "energy_product" and ps["bin_edges"][index] >= old.tables_by_z[z].al_energy[0])
                if unchanged and abs((nv-pv)*width) > tolerance:
                    raise ValueError("Changed an output bin unaffected by lower support.")
        let_gain = n["let_ledger"]["mapped_integral"] - p["let_ledger"]["mapped_integral"]
        if let_gain < -tolerance or let_gain > gain + tolerance:
            raise ValueError("Unexpected LET gain for matched Al/Si lower support.")
        rows.append({"source": key[0], "mode": key[1], "z": z, "thickness_g_cm2": key[3],
                     "source_integral": total, "new_supported_integral": gain,
                     "new_supported_fraction_of_source": gain / total if total else None,
                     "new_mapped_let_integral": let_gain,
                     "remaining_below_anchor_integral": ln["unresolved_low_residual_integral"],
                     "low_residual_statement": _low_residual_statement(
                         ln["unresolved_low_residual_integral"],
                         new.tables_by_z[z].native_al.rows[0].energy_total_mev, key[1])})
    for path, sha in pins.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("Comparison input changed.")
    return {"schema_version": "radar.srim_lower_support_comparison.v1", "status": "INCOMPLETE",
            "normative_export_allowed": False, "stopped_particle_classification_established": False,
            "full_inventory_required_and_verified": require_full_inventory,
            "tables": tables, "cases": rows, "case_count": len(rows), "input_sha256": pins,
            "limitations": ["Finite lower endpoint does not establish the range to rest.",
                            "Counts are not dose or SEE error bounds; orbital acceptance not performed."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-bundle", "lower-bundle", "parent-products", "lower-products"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-full-inventory", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(parent_bundle=args.parent_bundle, parent_sha=args.parent_bundle_sha256,
                     lower_bundle=args.lower_bundle, lower_sha=args.lower_bundle_sha256,
                     parent_products=args.parent_products, parent_products_sha=args.parent_products_sha256,
                     lower_products=args.lower_products, lower_products_sha=args.lower_products_sha256,
                     require_full_inventory=args.require_full_inventory)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"cases": result["case_count"],
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
