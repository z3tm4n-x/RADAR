"""Matched native stopping controls; no joined transport, projected range or extrapolation.

CPython 3.12 -B is needed only for sampling, not for authenticating the report.
SRIM rows use exact printed E_total and literal A. Above the last SRIM row
there is CATIMA only, on a nested 40/80-per-decade grid plus explicit anchors.
All stopping powers are whole-projectile MeV cm2/g.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
from math import ceil, isclose, log10
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_srim_source_overlap as shared  # noqa: E402
import compare_srim_upper_overlap as inventory  # noqa: E402
import generate_catima_candidate as native  # noqa: E402
import generate_high_energy_anchors as anchors  # noqa: E402
from generate_astar_helium_controls import _number, _configuration_values, _elemental_i_values  # noqa: E402
from generate_iaea_stopping_controls import _material_values  # noqa: E402

ANCHOR_PATH = ROOT / "outputs/high_energy_anchors_final_20261105.json"
ANCHOR_SHA = "466a975f47d66f83662a26a740ecab04488c947f688517525a4bcaf10398d604"
IONS = {2: "4", 10: "20.2", 26: "55.8", 92: "238"}
COMPONENTS = ("total", "nonradiative", "electronic", "nuclear", "radiative")
FIELDS = {f"catima_{c}_mev_cm2_g" for c in COMPONENTS}
FLAGS = {**anchors.FLAGS, "srim_projected_range_used": False, "release_ready": False}
SCHEMA = "radar.paired_stopping_controls.v1"
LOW = Fraction(1, 100)


def upper_nodes(last: float) -> dict[float, bool]:
    """Fine nodes include all coarse nodes, the actual source top and threshold neighbours."""
    top = anchors.TOP
    if not 4400 <= last <= 5000 or top != 79432.82347242821:
        raise ValueError("Unexpected native upper endpoint or GCR last centre.")
    intervals = ceil(40 * log10(top / last))
    nodes = {last * (top / last) ** (i / (2 * intervals)): i % 2 == 0
             for i in range(1, 2 * intervals)}
    nodes.update({e: True for e in anchors.FIXED.values()})
    if len(nodes) != 2 * intervals - 1 + len(anchors.FIXED) or not all(last < e <= top for e in nodes):
        raise ValueError("Degenerate CATIMA-only grid.")
    return dict(sorted(nodes.items()))


def build_requests(selected: dict[str, Any]) -> list[dict[str, Any]]:
    tables = {(t.ion_z, t.target_z): (name, t) for name, t in selected.items() if t.ion_z in IONS}
    if set(tables) != {(z, m) for z in IONS for m in (13, 14)} or sum(t.ion_z in IONS for t in selected.values()) != 8:
        raise ValueError("Expected four unique literal masses in Al and Si.")
    result: list[dict[str, Any]] = []
    for z, mass_text in IONS.items():
        grids = []
        for m in (13, 14):
            name, table = tables[z, m]
            if (table.ion_mass_u != Decimal(mass_text) or table.stopping_unit != "MeV/(mg/cm2)"
                    or table.energy_basis != "total_ion_kinetic_energy"
                    or re.fullmatch(r"[0-9a-f]{64}", table.raw_sha256) is None):
                raise ValueError("Changed source A, units, basis or raw table pin.")
            base = {"z": z, "mass_parameter_u_decimal": mass_text, "material_z": m,
                    "case_name": name, "native_raw_sha256": table.raw_sha256}
            common = []
            for i, row in enumerate(table.rows):
                exact = Fraction(row.energy_total_mev) / Fraction(Decimal(mass_text))
                if exact < LOW:
                    continue
                e = float(exact)
                electronic = float(Fraction(row.electronic_mev_cm2_mg) * 1000)
                nuclear = float(Fraction(row.nuclear_mev_cm2_mg) * 1000)
                _number(electronic, "SRIM electronic", positive=True)
                _number(nuclear, "SRIM nuclear")
                common.append({**base, "energy_mev_per_nucleon": e, "energy_binary64_hex": e.hex(),
                    "region": "common", "upper_coarse_node": True, "native_row_index": i,
                    "native_total_energy_mev_decimal": format(row.energy_total_mev, "f"),
                    "srim_electronic_mev_cm2_g": electronic, "srim_nuclear_mev_cm2_g": nuclear})
            grid = tuple(r["energy_mev_per_nucleon"] for r in common)
            if len(grid) < 2 or grid[0] >= 1 or any(a >= b for a, b in zip(grid, grid[1:])):
                raise ValueError("Incomplete or unordered overlapping grid.")
            grids.append(grid)
            result.extend(common)
            for e, coarse in upper_nodes(grid[-1]).items():
                result.append({**base, "energy_mev_per_nucleon": e, "energy_binary64_hex": e.hex(),
                    "region": "catima_only", "upper_coarse_node": coarse, "native_row_index": None,
                    "native_total_energy_mev_decimal": None,
                    "srim_electronic_mev_cm2_g": None, "srim_nuclear_mev_cm2_g": None})
        if grids[0] != grids[1]:
            raise ValueError("Al and Si must share exact energy nodes to isolate their effects.")
    return result


def reconstruct_plan() -> dict[str, Any]:
    selected, manifest = inventory.reconstruct_plan()
    return {"bundle_sha256": inventory.BUNDLE_SHA256, "source_manifest_sha256": manifest["source_manifest_sha256"],
            "anchor_sha256": ANCHOR_SHA, "requests": build_requests(selected)}


def snapshot() -> dict[str, Any]:
    state = anchors.snapshot()
    state["implementation_sha256"]["scripts/build_paired_stopping_controls.py"] = native.sha256(Path(__file__))
    return state


def generate_samples(backend: Any, requests: list[dict[str, Any]], elemental_i: dict[int, float]) -> dict[str, Any]:
    configs = [native.configuration(backend), native.configuration(backend, no_radiative=True)]
    config_values = [_configuration_values(c, nonradiative=bool(i)) for i, c in enumerate(configs)]
    materials = {m: backend.get_material(m) for m in (13, 14)}
    metadata = {m: _material_values(materials[m], m, elemental_i[m]) for m in materials}
    projectiles = {z: backend.Projectile(float(Decimal(a)), z, z) for z, a in IONS.items()}
    rows = []
    for request in requests:
        z, m, e = request["z"], request["material_z"], request["energy_mev_per_nucleon"]
        p = projectiles[z]
        p.T(e)
        total, nonrad = (_number(backend.dedx(p, materials[m], c), "CATIMA stopping", positive=True) for c in configs)
        nuclear = native.nuclear_stopping(z, float(Decimal(IONS[z])), m, metadata[m]["atomic_mass_u"], e)
        electronic, radiative = nonrad - nuclear, total - nonrad
        if e <= 10 and not isclose(electronic, backend.sezi_dedx_e(p, materials[m], configs[0]),
                                   rel_tol=1e-10, abs_tol=0):
            raise ValueError("Low-energy electronic closure failed.")
        rows.append({**request, **{f"catima_{k}_mev_cm2_g": v for k, v in
                    zip(COMPONENTS, (total, nonrad, electronic, nuclear, radiative), strict=True)}})
    if config_values != [_configuration_values(c, nonradiative=bool(i)) for i, c in enumerate(configs)]:
        raise ValueError("CATIMA configuration changed during sampling.")
    if any(_material_values(materials[m], m, elemental_i[m]) != metadata[m] for m in materials):
        raise ValueError("Target changed during sampling.")
    return {"rows": rows, "backend_configuration": config_values[0], "nonradiative_configuration": config_values[1],
            "materials": list(metadata.values())}


def validate_report(report: Any, plan: dict[str, Any], state: dict[str, Any], baseline: dict[str, Any]) -> None:
    if (not isinstance(report, dict) or report.get("schema_version") != SCHEMA or report.get("status") != "INCOMPLETE"
            or any(report.get(k) is not False for k in FLAGS) or report.get("stopping_units") != "whole_ion_MeV_cm2_g"
            or any(report.get(k) != v for k, v in plan.items() if k != "requests")
            or any(not isinstance(report.get(k), str) or not report[k].strip() for k in ("python", "backend_info_literal"))):
        raise ValueError("Report provenance, unit convention or release status changed.")
    for key in ("snapshot_before", "snapshot_after"):
        shared.same(report[key], state, "paired stopping snapshot")
    for k, nonradiative in (("backend_configuration", False), ("nonradiative_configuration", True)):
        _configuration_values(SimpleNamespace(get=lambda: report[k]), nonradiative=nonradiative)
        shared.same(report[k], baseline[k], "historical anchor configuration")
    shared.same(report["materials"], baseline["materials"], "historical anchor targets")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(plan["requests"]) or type(report.get("row_count")) is not int or report["row_count"] != len(rows):
        raise ValueError("Incomplete stopping samples.")
    indexed = {}
    target_a = {13: 26.9815, 14: 28.0855}
    for row, req in zip(rows, plan["requests"], strict=True):
        if not isinstance(row, dict) or set(row) != set(req) | FIELDS:
            raise ValueError("Unexpected row fields.")
        shared.same({k: row[k] for k in req}, req, "raw native row/extension identity")
        total, nonrad, electronic, nuclear, radiative = (_number(row[f"catima_{c}_mev_cm2_g"], c,
                positive=c not in ("nuclear", "radiative")) for c in COMPONENTS)
        z, m, e = req["z"], req["material_z"], req["energy_mev_per_nucleon"]
        replay = native.nuclear_stopping(z, float(IONS[z]), m, target_a[m], e)
        if (not isclose(nuclear, replay, rel_tol=2e-14) or electronic != nonrad - nuclear or radiative != total - nonrad
                or (radiative > 0) != (e > 50000)):
            raise ValueError("Stopping decomposition, nuclear replay or radiative threshold changed.")
        indexed[z, m, e] = row
    if len(baseline["rows"]) != 48:
        raise ValueError("The pinned 48 independent-run anchor observations are required.")
    for anchor in baseline["rows"]:
        match = indexed[anchor["z"], anchor["material_z"], anchor["energy_mev_per_nucleon"]]
        if any(match[f"catima_{c}_mev_cm2_g"] != anchor[f"stopping_{c}_mev_cm2_g"] for c in COMPONENTS):
            raise ValueError("CATIMA does not reproduce the retained anchor at identical input.")


def generate() -> dict[str, Any]:
    plan, state = reconstruct_plan(), snapshot()
    baseline = shared.read_pinned(ANCHOR_PATH, ANCHOR_SHA)
    backend = native.load_backend()
    samples = generate_samples(backend, plan["requests"], _elemental_i_values())
    shared.same(reconstruct_plan(), plan, "paired control plan after sampling")
    shared.same(shared.read_pinned(ANCHOR_PATH, ANCHOR_SHA), baseline, "baseline anchors after sampling")
    shared.same(snapshot(), state, "native and production snapshot after sampling")
    report = {"schema_version": SCHEMA, "status": "INCOMPLETE", **FLAGS,
        "stopping_units": "whole_ion_MeV_cm2_g", "python": sys.version, "backend_info_literal": backend.catima_info(),
        "snapshot_before": state, "snapshot_after": state, "row_count": len(samples["rows"]),
        **{k: v for k, v in plan.items() if k != "requests"}, **samples,
        "limitations": ["No claim of electronic stopping equivalence to total loss or physical LET qualification.",
            "No ranges, nuclear removal, spectral weighting or automatic splice.",
            "The interpolated CATIMA-only region is not a prediction by SRIM. No data beyond the last source centre."]}
    validate_report(report, plan, state, baseline)
    return report


def read_verified(path: Path, sha: str) -> dict[str, Any]:
    report = shared.read_pinned(path, sha)
    validate_report(report, reconstruct_plan(), snapshot(), shared.read_pinned(ANCHOR_PATH, ANCHOR_SHA))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = generate()
    shared.write_new(args.output, report)
    print({"sha256": native.sha256(args.output), "rows": report["row_count"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
