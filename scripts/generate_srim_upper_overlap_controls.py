"""Pinned CATIMA exact high-band samples; CPython 3.12 -B, diagnostic only."""
from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
from math import isclose
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_srim_upper_overlap as planner  # noqa: E402
import compare_srim_source_overlap as shared  # noqa: E402
import generate_srim_overlap_controls as old  # noqa: E402
import generate_catima_candidate as native  # noqa: E402
from generate_astar_helium_controls import _configuration_values, _elemental_i_values, _number  # noqa: E402
from generate_iaea_stopping_controls import _material_values  # noqa: E402

FLAGS, IDENTITY = old.FLAGS, old.IDENTITY
REQUEST_FIELDS, STOPPING_FIELDS = old.REQUEST_FIELDS, old.STOPPING_FIELDS
IMPLEMENTATION_PATHS = (*old.IMPLEMENTATION_PATHS, "scripts/generate_srim_upper_overlap_controls.py",
    "scripts/compare_srim_upper_overlap.py", "scripts/compare_srim_source_overlap.py",
    "scripts/summarize_srim_source_campaign.py", "src/radar/validation/srim_source_catalog.py",
    "src/radar/shielding/srim_dataset.py", "src/radar/shielding/srim_hze.py", "src/radar/validation/srim_output.py")


def validate_requests(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 100_000:
        raise ValueError("A bounded nonempty exact-node request list is required.")
    previous = None
    indices: dict[str, int] = {}
    identities: dict[str, tuple[Any, ...]] = {}
    result = []
    for row in value:
        if not isinstance(row, dict) or set(row) != REQUEST_FIELDS:
            raise ValueError("Unexpected exact-node request fields.")
        z, target = row["z"], row["material_z"]
        if type(z) is not int or not 2 <= z <= 92 or type(target) is not int or target not in (13, 14):
            raise ValueError("Only heavy ions in Al/Si are permitted.")
        mass, total = old._decimal(row["mass_parameter_u_decimal"], "A"), old._decimal(row["energy_total_mev_decimal"], "energy")
        energy = Fraction(total)/Fraction(mass)
        if not z <= mass <= 238 or not planner.LOWER <= energy <= planner.UPPER:
            raise ValueError("Requests are bounded to 1000..5000 MeV/literal-source-A.")
        if type(row["energy_mev_per_nucleon"]) is not float or row["energy_mev_per_nucleon"] != float(energy):
            raise ValueError("Backend energy differs from the exact total/A conversion.")
        sha, index, name = row["raw_sha256"], row["native_row_index"], row["case_name"]
        if (not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None
                or type(index) is not int or not 0 <= index < 100_000
                or not isinstance(name, str) or re.fullmatch(r"[a-z0-9_]{1,200}", name) is None):
            raise ValueError("Invalid retained SRIM row identity.")
        key = z, mass, target, total
        if (previous is not None and key <= previous) or index <= indices.get(name, -1):
            raise ValueError("Requests must have unique ordered exact nodes and increasing native indices.")
        if identities.setdefault(name, (z, mass, target, sha)) != (z, mass, target, sha):
            raise ValueError("A native case changed identity.")
        previous, indices[name] = key, index
        result.append(dict(row))
    return tuple(result)


def snapshot() -> dict[str, Any]:
    return {"implementation_sha256": {p: native.sha256(ROOT / p) for p in IMPLEMENTATION_PATHS},
            "native_source_pins": native.verify_sources(), "production_resources_sha256": native.production_hashes()}


def generate_samples(backend: Any, requests: Any, *, elemental_i_ev: dict[int, float]) -> dict[str, Any]:
    requests = validate_requests(requests)
    if not isinstance(elemental_i_ev, dict) or set(elemental_i_ev) != {13, 14}:
        raise ValueError("Both unchanged elemental ionization potentials are required.")
    config, nonrad = native.configuration(backend), native.configuration(backend, no_radiative=True)
    configs = [_configuration_values(config, nonradiative=False), _configuration_values(nonrad, nonradiative=True)]
    materials = {z: backend.get_material(z) for z in (13, 14)}
    metadata = {z: _material_values(m, z, elemental_i_ev[z]) for z, m in materials.items()}
    rows, projectiles = [], {}
    for req in requests:
        z, target, energy = req["z"], req["material_z"], req["energy_mev_per_nucleon"]
        mass = float(Decimal(req["mass_parameter_u_decimal"]))
        key = z, req["mass_parameter_u_decimal"]
        if key not in projectiles:
            projectiles[key] = backend.Projectile(mass, z, z)
        p, m = projectiles[key], materials[target]
        p.T(energy)
        total = _number(backend.dedx(p, m, config), "total stopping", positive=True)
        nonradiative = _number(backend.dedx(p, m, nonrad), "nonradiative stopping", positive=True)
        nuclear = _number(native.nuclear_stopping(z, mass, target, metadata[target]["atomic_mass_u"], energy), "nuclear")
        electronic = _number(nonradiative-nuclear, "electronic", positive=True)
        radiative = _number(total-nonradiative, "radiative")
        if radiative != 0:
            raise ValueError("Radiative term must remain zero below the pinned 50000 MeV/u threshold.")
        rows.append({**req, "stopping_total_mev_cm2_g": total, "stopping_nonradiative_mev_cm2_g": nonradiative,
            "stopping_electronic_mev_cm2_g": electronic, "stopping_nuclear_mev_cm2_g": nuclear,
            "stopping_radiative_mev_cm2_g": radiative})
    if configs != [_configuration_values(config, nonradiative=False), _configuration_values(nonrad, nonradiative=True)]:
        raise ValueError("Native configuration changed during sampling.")
    if any(_material_values(materials[z], z, elemental_i_ev[z]) != values for z, values in metadata.items()):
        raise ValueError("Native material changed during sampling.")
    return {"rows": rows, "materials": list(metadata.values()), "backend_configuration": configs[0],
            "nonradiative_configuration": configs[1], "low_energy_electronic_closure_count": 0,
            "low_energy_electronic_closure_max_relative_error": 0.0}


def read_plan(path: Path, sha: str) -> dict[str, Any]:
    supplied = shared.read_pinned(path, sha)
    _, expected = planner.reconstruct_plan()
    shared.same(supplied, expected, "complete authenticated full-upper request plan")
    validate_requests(supplied["requests"])
    return dict(supplied)


def generate_controls(path: Path, sha: str) -> dict[str, Any]:
    plan = read_plan(path, sha)
    before = snapshot()
    backend = native.load_backend()
    samples = generate_samples(backend, plan["requests"], elemental_i_ev=_elemental_i_values())
    shared.same(read_plan(path, sha), plan, "requests after sampling")
    shared.same(snapshot(), before, "native/implementation/production snapshot")
    return {"schema_version": "radar.srim_catima_upper_overlap_native.v1", "status": "INCOMPLETE",
        "request_path": str(path.resolve()), "requests_sha256": sha, "bundle_sha256": planner.BUNDLE_SHA256,
        "request_count": len(plan["requests"]), "snapshot_before": before, "snapshot_after": before,
        "python": sys.version, "backend_info_literal": backend.catima_info(), **IDENTITY, **FLAGS, **samples,
        "limitations": ["No independent experimental validation; shared model ancestry remains possible.",
            "Fixed default LS and source A are diagnostic, not fitted or newly qualified.",
            "No source-to-binary build verification, range calculation, model splice or spectral convolution."]}


def validate_report(report: Any, path: Path, sha: str, plan: dict[str, Any]) -> None:
    shared.same(read_plan(path, sha), plan, "caller request plan")
    if (not isinstance(report, dict) or report.get("schema_version") != "radar.srim_catima_upper_overlap_native.v1"
            or report.get("status") != "INCOMPLETE" or any(report.get(k) is not False for k in FLAGS)
            or any(report.get(k) != v for k, v in IDENTITY.items())
            or report.get("request_path") != str(path.resolve()) or report.get("requests_sha256") != sha
            or report.get("bundle_sha256") != planner.BUNDLE_SHA256
            or type(report.get("request_count")) is not int or report["request_count"] != len(plan["requests"])):
        raise ValueError("Native report identity, policies or request count changed.")
    for field in ("snapshot_before", "snapshot_after"):
        shared.same(report[field], snapshot(), "retained native evidence snapshot")
    for field, nonrad in (("backend_configuration", False), ("nonradiative_configuration", True)):
        _configuration_values(SimpleNamespace(get=lambda: report[field]), nonradiative=nonrad)
    expected: list[dict[str, Any]] = [{"z": z, "atomic_mass_u": mass, "density_g_cm3": rho,
        "ionization_potential_override_ev": 0.0, "elemental_ionization_potential_ev": potential,
        "ionization_potential_policy": "unchanged_backend_internal_elemental_table"}
        for z, mass, rho, potential in ((13, 26.9815, 2.699, 166.0), (14, 28.0855, 2.33, 173.0))]
    shared.same(report["materials"], expected, "unchanged native materials")
    if (type(report.get("low_energy_electronic_closure_count")) is not int
            or report["low_energy_electronic_closure_count"] != 0
            or _number(report.get("low_energy_electronic_closure_max_relative_error"), "closure") != 0
            or any(not isinstance(report.get(f), str) or not report[f].strip() for f in ("python", "backend_info_literal"))):
        raise ValueError("Runtime/closure metadata changed.")
    requests, rows = validate_requests(plan["requests"]), report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(requests):
        raise ValueError("Missing or extra exact-node native samples.")
    masses = {v["z"]: v["atomic_mass_u"] for v in expected}
    for row, req in zip(rows, requests, strict=True):
        if not isinstance(row, dict) or set(row) != REQUEST_FIELDS | STOPPING_FIELDS:
            raise ValueError("Native sample fields changed.")
        shared.same({k: row[k] for k in REQUEST_FIELDS}, req, "native row identity")
        values = {c: _number(row[f"stopping_{c}_mev_cm2_g"], c, positive=c not in ("nuclear", "radiative"))
                  for c in ("total", "nonradiative", "electronic", "nuclear", "radiative")}
        nuclear = native.nuclear_stopping(req["z"], float(Decimal(req["mass_parameter_u_decimal"])),
                                         req["material_z"], masses[req["material_z"]], req["energy_mev_per_nucleon"])
        if (values["radiative"] != 0 or values["total"] != values["nonradiative"]
                or not isclose(values["nuclear"], nuclear, rel_tol=2e-14, abs_tol=0)
                or not isclose(values["electronic"]+values["nuclear"], values["nonradiative"], rel_tol=2e-14, abs_tol=0)):
            raise ValueError("Native decomposition or replayed nuclear term changed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_file", type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = generate_controls(args.request_file, args.request_sha256)
    shared.write_new(args.output, report)
    print({"sha256": native.sha256(args.output), "request_count": report["request_count"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
