"""Bounded high-energy CATIMA anchors and one-sided threshold checks, not a splice.

CPython 3.12 -B with the already pinned backend. All stopping values refer to
the whole projectile in MeV cm2/g. No range or spectral operator is evaluated.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
from math import isclose, nextafter
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_srim_upper_overlap_controls as previous  # noqa: E402
import generate_catima_candidate as native  # noqa: E402
import compare_srim_source_overlap as shared  # noqa: E402
import compare_srim_upper_overlap as planner  # noqa: E402
from generate_srim_overlap_controls import _decimal  # noqa: E402
from generate_astar_helium_controls import _number, _configuration_values, _elemental_i_values  # noqa: E402
from generate_iaea_stopping_controls import _material_values  # noqa: E402
from radar.core.constants import DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON  # noqa: E402

IONS = {2: "4", 10: "20.2", 26: "55.8", 92: "238"}
TOP = DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON[-1]
FIXED = {"10000": 10000.0, "radiative_below": nextafter(50000.0, -float("inf")),
         "50000": 50000.0, "radiative_above": nextafter(50000.0, float("inf")),
         "gcr_last_center": TOP}
CORE = frozenset({"native_upper", "10000", "50000", "gcr_last_center"})
REQUEST_FIELDS = frozenset({"z", "mass_parameter_u_decimal", "material_z", "anchor_kind",
    "core_anchor", "energy_mev_per_nucleon", "energy_binary64_hex", "energy_exact_fraction", "srim_reference"})
FLAGS = {**previous.FLAGS, "high_energy_model_selected": False, "table_splicing_performed": False,
         "source_spectrum_convolution_performed": False, "source_grids_changed": False}
IDENTITY = {**previous.IDENTITY, "energy_policy": "exact_Srim_total_A_or_declared_binary64_anchor",
            "native_sampling_policy": "32_declared_core_anchors_and_16_one_sided_threshold_checks"}
IMPLEMENTATION_PATHS = (*previous.IMPLEMENTATION_PATHS, "scripts/generate_high_energy_anchors.py",
                        "src/radar/core/constants.py")


def build_requests(selected: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the actual final SRIM row, never a nominal 5000-MeV/A endpoint."""
    tables = {(table.ion_z, table.target_z): (name, table)
              for name, table in selected.items() if table.ion_z in IONS}
    if len(tables) != 8 or sum(t.ion_z in IONS for t in selected.values()) != 8:
        raise ValueError("Exactly four selected ions in both materials are required.")
    if TOP != 79432.82347242821:
        raise ValueError("Declared finite GCR last centre changed; review anchor scope.")
    requests = []
    for z, mass in IONS.items():
        for target in (13, 14):
            name, table = tables[z, target]
            if (table.ion_mass_u != Decimal(mass) or table.stopping_unit != "MeV/(mg/cm2)"
                    or table.energy_basis != "total_ion_kinetic_energy"):
                raise ValueError("Unexpected native mass or units.")
            last = table.rows[-1]
            exact = Fraction(last.energy_total_mev) / Fraction(table.ion_mass_u)
            reference = {"case_name": name, "raw_sha256": table.raw_sha256,
                         "native_row_index": len(table.rows)-1,
                         "energy_total_mev_decimal": format(last.energy_total_mev, "f"),
                         "electronic_mev_cm2_mg": format(last.electronic_mev_cm2_mg, "f"),
                         "nuclear_mev_cm2_mg": format(last.nuclear_mev_cm2_mg, "f")}
            nodes = {"native_upper": float(exact), **FIXED}
            for kind, energy in sorted(nodes.items(), key=lambda item: item[1]):
                requests.append({"z": z, "mass_parameter_u_decimal": mass, "material_z": target,
                    "anchor_kind": kind, "core_anchor": kind in CORE,
                    "energy_mev_per_nucleon": energy, "energy_binary64_hex": energy.hex(),
                    "energy_exact_fraction": str(exact if kind == "native_upper" else Fraction(energy)),
                    "srim_reference": reference.copy() if kind == "native_upper" else None})
    validate_requests(requests)
    return requests


def validate_requests(requests: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(requests, (list, tuple)) or len(requests) != 48:
        raise ValueError("Exactly 32 core anchors plus 16 one-sided checks are required.")
    expected_order = [(z, target, kind) for z in IONS for target in (13, 14)
                      for kind in ("native_upper", *sorted(FIXED, key=lambda name: FIXED[name]))]
    for row, (z, target, kind) in zip(requests, expected_order, strict=True):
        if (not isinstance(row, dict) or set(row) != REQUEST_FIELDS
                or type(row["z"]) is not int or row["z"] != z
                or type(row["material_z"]) is not int or row["material_z"] != target
                or row["mass_parameter_u_decimal"] != IONS[z] or row["anchor_kind"] != kind
                or row["core_anchor"] is not (kind in CORE)):
            raise ValueError("Changed, missing, duplicated or out-of-order anchor identity.")
        energy = row["energy_mev_per_nucleon"]
        if type(energy) is not float or not 1000 <= energy <= TOP:
            raise ValueError("Anchor energy must be a bounded finite binary64 value.")
        if row["energy_binary64_hex"] != energy.hex():
            raise ValueError("Binary64 anchor coordinate changed.")
        reference = row["srim_reference"]
        if kind == "native_upper":
            if (not isinstance(reference, dict) or set(reference) != {"case_name", "raw_sha256",
                    "native_row_index", "energy_total_mev_decimal", "electronic_mev_cm2_mg", "nuclear_mev_cm2_mg"}
                    or not isinstance(reference["case_name"], str)
                    or re.fullmatch(r"[a-z0-9_]{1,200}", reference["case_name"]) is None
                    or not isinstance(reference["raw_sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", reference["raw_sha256"]) is None
                    or type(reference["native_row_index"]) is not int or not 0 <= reference["native_row_index"] < 100000):
                raise ValueError("Malformed retained native endpoint reference.")
            total = _decimal(reference["energy_total_mev_decimal"], "native energy")
            for component in ("electronic_mev_cm2_mg", "nuclear_mev_cm2_mg"):
                _decimal(reference[component], component)
            exact = Fraction(total) / Fraction(Decimal(IONS[z]))
            if not 4400 <= exact <= 5000 or energy != float(exact):
                raise ValueError("Native endpoint must retain exact E_total/A conversion.")
        else:
            exact = Fraction(FIXED[kind])
            if reference is not None or energy != FIXED[kind]:
                raise ValueError("Declared anchor changed or has a fabricated SRIM reference.")
        if row["energy_exact_fraction"] != str(exact):
            raise ValueError("Exact anchor coordinate changed.")
    return tuple(dict(r) for r in requests)


def reconstruct_plan() -> dict[str, Any]:
    selected, source = planner.reconstruct_plan()
    return {"schema_version": "radar.high_energy_anchors.requests.v1",
            "bundle_sha256": planner.BUNDLE_SHA256, "source_manifest_sha256": source["source_manifest_sha256"],
            "source_grid_definition_sha256": native.sha256(ROOT / "src/radar/core/constants.py"),
            "gcr_last_center_mev_per_source_a": TOP, "gcr_last_center_hex": TOP.hex(),
            "gcr_reference_upper_boundary_not_sampled": 100000.0, "requests": build_requests(selected)}


def snapshot() -> dict[str, Any]:
    return {"implementation_sha256": {p: native.sha256(ROOT / p) for p in IMPLEMENTATION_PATHS},
            "native_source_pins": native.verify_sources(), "production_resources_sha256": native.production_hashes()}


def generate_samples(backend: Any, requests: Any, *, elemental_i_ev: dict[int, float]) -> dict[str, Any]:
    requests = validate_requests(requests)
    if not isinstance(elemental_i_ev, dict) or set(elemental_i_ev) != {13, 14}:
        raise ValueError("Both unchanged elemental potentials are required.")
    configs = [native.configuration(backend), native.configuration(backend, no_radiative=True)]
    config_values = [_configuration_values(c, nonradiative=bool(i)) for i, c in enumerate(configs)]
    materials = {z: backend.get_material(z) for z in (13, 14)}
    metadata = {z: _material_values(m, z, elemental_i_ev[z]) for z, m in materials.items()}
    rows, projectiles = [], {}
    for req in requests:
        z, target, energy = req["z"], req["material_z"], req["energy_mev_per_nucleon"]
        mass = float(Decimal(req["mass_parameter_u_decimal"]))
        if z not in projectiles:
            projectiles[z] = backend.Projectile(mass, z, z)
        projectile = projectiles[z]
        projectile.T(energy)
        total, nonrad = [_number(backend.dedx(projectile, materials[target], c), "stopping", positive=True) for c in configs]
        nuclear = _number(native.nuclear_stopping(z, mass, target, metadata[target]["atomic_mass_u"], energy), "nuclear")
        electronic, radiative = _number(nonrad-nuclear, "electronic", positive=True), _number(total-nonrad, "radiative")
        if energy <= 50000 and radiative != 0:
            raise ValueError("Pinned radiative threshold must be strictly above 50000 MeV/u.")
        rows.append({**req, "stopping_total_mev_cm2_g": total, "stopping_nonradiative_mev_cm2_g": nonrad,
            "stopping_electronic_mev_cm2_g": electronic, "stopping_nuclear_mev_cm2_g": nuclear,
            "stopping_radiative_mev_cm2_g": radiative})
    if config_values != [_configuration_values(c, nonradiative=bool(i)) for i, c in enumerate(configs)]:
        raise ValueError("Native configuration changed.")
    if any(_material_values(materials[z], z, elemental_i_ev[z]) != value for z, value in metadata.items()):
        raise ValueError("Native material changed.")
    return {"rows": rows, "materials": list(metadata.values()), "backend_configuration": config_values[0],
            "nonradiative_configuration": config_values[1]}


def validate_report(report: Any, plan: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> None:
    plan = reconstruct_plan() if plan is None else plan
    state = snapshot() if state is None else state
    if (not isinstance(report, dict) or report.get("schema_version") != "radar.high_energy_anchors.native.v1"
            or report.get("status") != "INCOMPLETE" or type(report.get("request_count")) is not int
            or report["request_count"] != 48 or any(report.get(k) is not False for k in FLAGS)
            or any(report.get(k) != v for k, v in IDENTITY.items())):
        raise ValueError("Changed diagnostic report identity or policy.")
    shared.same(report["plan"], plan, "complete high-energy plan")
    for field in ("snapshot_before", "snapshot_after"):
        shared.same(report[field], state, "native/implementation/production state")
    for field, nonrad in (("backend_configuration", False), ("nonradiative_configuration", True)):
        _configuration_values(SimpleNamespace(get=lambda: report[field]), nonradiative=nonrad)
    expected: list[dict[str, Any]] = [{"z": z, "atomic_mass_u": mass, "density_g_cm3": density,
                 "ionization_potential_override_ev": 0.0, "elemental_ionization_potential_ev": potential,
                 "ionization_potential_policy": "unchanged_backend_internal_elemental_table"}
                for z, mass, density, potential in ((13, 26.9815, 2.699, 166.0), (14, 28.0855, 2.33, 173.0))]
    shared.same(report["materials"], expected, "unchanged Al/Si materials")
    if any(not isinstance(report.get(k), str) or not report[k].strip() for k in ("python", "backend_info_literal")):
        raise ValueError("Missing backend/runtime metadata.")
    requests, rows = validate_requests(plan["requests"]), report.get("rows")
    if not isinstance(rows, list) or len(rows) != 48:
        raise ValueError("Incomplete native anchor sample inventory.")
    masses = {row["z"]: row["atomic_mass_u"] for row in expected}
    for row, req in zip(rows, requests, strict=True):
        if not isinstance(row, dict) or set(row) != REQUEST_FIELDS | previous.STOPPING_FIELDS:
            raise ValueError("Changed native sample fields.")
        shared.same({k: row[k] for k in REQUEST_FIELDS}, req, "sample identity")
        values = {name: _number(row[f"stopping_{name}_mev_cm2_g"], name, positive=name not in ("nuclear", "radiative"))
                  for name in ("total", "nonradiative", "electronic", "nuclear", "radiative")}
        nuclear = native.nuclear_stopping(req["z"], float(Decimal(req["mass_parameter_u_decimal"])),
                    req["material_z"], masses[req["material_z"]], req["energy_mev_per_nucleon"])
        if (not isclose(values["nuclear"], nuclear, rel_tol=2e-14, abs_tol=0)
                or not isclose(values["electronic"]+values["nuclear"], values["nonradiative"], rel_tol=2e-14, abs_tol=0)
                or not isclose(values["nonradiative"]+values["radiative"], values["total"], rel_tol=2e-14, abs_tol=0)
                or (req["energy_mev_per_nucleon"] <= 50000 and values["radiative"] != 0)):
            raise ValueError("Stopping decomposition, nuclear replay or radiative threshold changed.")


def generate_controls() -> dict[str, Any]:
    plan, before = reconstruct_plan(), snapshot()
    backend = native.load_backend()
    samples = generate_samples(backend, plan["requests"], elemental_i_ev=_elemental_i_values())
    shared.same(reconstruct_plan(), plan, "anchor input plan after sampling")
    shared.same(snapshot(), before, "native/implementation/production state after sampling")
    report = {"schema_version": "radar.high_energy_anchors.native.v1", "status": "INCOMPLETE",
              "request_count": 48, "plan": plan, "snapshot_before": before, "snapshot_after": before,
              "python": sys.version, "backend_info_literal": backend.catima_info(), **FLAGS, **IDENTITY, **samples,
              "limitations": ["Stopping samples only, not a high-energy transport selection or physical qualification.",
                  "No independent model, experiment, range, spectral weighting or table splice is asserted.",
                  "Literal OST source A remains a diagnostic numeric mass, not a qualified isotope policy.",
                  "Radiative component is a backend total-minus-nonradiative difference; no local deposition claim.",
                  "Finite GCR last centre only; neither 100 GeV reference boundary nor unknown outer tails covered."]}
    validate_report(report, plan, before)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "run"), default="run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = reconstruct_plan() if args.mode == "plan" else generate_controls()
    shared.write_new(args.output, result)
    print({"sha256": native.sha256(args.output), "request_count": 48})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
