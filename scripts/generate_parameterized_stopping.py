"""Isolated CATIMA mass-convention experiment; never installs production data.

Run with CPython 3.12 -B. Source-model A is used as a *diagnostic numeric mass
parameter*, not identified as an isotope or a qualified transport convention.
The unchanged pilot isotopes provide a paired comparison at identical energies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import generate_catima_candidate as native


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("source_tabulated_parameter_diagnostic", "explicit_pilot_isotope_diagnostic")
MANIFEST_SOURCE_PATHS = {
    "src/radar/gcr/parameters.py", "src/radar/sep/hze_spectrum.py",
    "src/radar/data/normative/sep_hze/ost_134_1044_2007_b9.csv",
    "src/radar/data/integrity.py", "src/radar/validation/ion_mass_contract.py",
    "scripts/build_stopping_mass_manifest.py",
}


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> Any:
    raise ValueError(f"Nonfinite JSON value: {value}")


def read_pinned_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("A lowercase manifest SHA256 is required.")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Manifest SHA256 mismatch.")
    manifest = json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)
    json.dumps(manifest, allow_nan=False)
    if (not isinstance(manifest, dict)
            or manifest.get("schema_version") != "radar.stopping_mass_manifest.v1"
            or manifest.get("normative_export_allowed") is not False):
        raise ValueError("Expected an explicitly nonqualified mass manifest.")
    for field in ("projectiles", "pilot_isotopes"):
        ions = manifest.get(field)
        if not isinstance(ions, list) or not ions:
            raise ValueError(f"Manifest {field} must be a nonempty array.")
        seen = set()
        for ion in ions:
            keys = {"z", "symbol", "mass_parameter_u"}
            if field == "projectiles":
                keys.add("source_membership")
            if not isinstance(ion, dict) or set(ion) != keys:
                raise ValueError("Malformed projectile identity.")
            z, a = ion["z"], ion["mass_parameter_u"]
            if type(z) is not int or not 2 <= z <= 92 or z in seen:
                raise ValueError("Unique HZE Z in 2..92 required.")
            if (isinstance(a, bool) or not isinstance(a, (int, float))
                    or not math.isfinite(a) or a < z):
                raise ValueError("Finite mass parameter >= Z required.")
            if not isinstance(ion["symbol"], str) or not ion["symbol"].strip():
                raise ValueError("Projectile symbol required.")
            if field == "pilot_isotopes" and a != int(a):
                raise ValueError("Explicit pilot isotope A must be integer-valued.")
            if field == "projectiles" and ion["source_membership"] not in (
                ["GCR"], ["GCR", "SEP"],
            ):
                raise ValueError("Expected positive GCR roster with optional SEP membership.")
            seen.add(z)
    source = {ion["z"]: ion for ion in manifest["projectiles"]}
    for ion in manifest["pilot_isotopes"]:
        if ion["z"] not in source or ion["symbol"] != source[ion["z"]]["symbol"]:
            raise ValueError("Pilot must match a source element, without equating its mass.")
    if len(source) != 84 or sum("SEP" in ion["source_membership"] for ion in source.values()) != 24:
        raise ValueError("Expected the full 84 GCR / 24 SEP positive HZE manifest.")
    verify_manifest_sources(manifest)
    return manifest


def verify_manifest_sources(manifest: dict[str, Any]) -> None:
    before = manifest.get("source_files_sha256_before")
    after = manifest.get("source_files_sha256_after")
    if (not isinstance(before, dict) or set(before) != MANIFEST_SOURCE_PATHS
            or before != after):
        raise ValueError("Manifest source hashes must agree before/after.")
    for relative, expected in before.items():
        if (not isinstance(relative, str) or not isinstance(expected, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise ValueError("Malformed source hash entry.")
        path = (ROOT / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(ROOT):
            raise ValueError("Manifest source path escapes the repository.")
        if native.sha256(path) != expected:
            raise ValueError(f"Manifest source hash mismatch: {relative}")


def experiment_grid(points_per_decade: int) -> tuple[float, ...]:
    return tuple(sorted(set(native.energy_grid(points_per_decade)) | {300.0}))


def generate_dataset(
    backend: Any, ions: list[dict[str, Any]], energies: tuple[float, ...], policy: str,
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError("Unknown mass convention.")
    config = native.configuration(backend)
    nonrad_config = native.configuration(backend, no_radiative=True)
    rows, mass_diagnostics, materials = [], [], []
    closure_count, closure_max = 0, 0.0
    worst: dict[str, Any] = {"relative_error": -1.0}
    for ion in ions:
        z, a = ion["z"], ion["mass_parameter_u"]
        reference_mass = backend.get_material(z).get_element(0).A
        if not math.isfinite(reference_mass) or reference_mass <= 0:
            raise ValueError("Invalid backend elemental mass reference.")
        ls_weight = (a / reference_mass - 1.0) / 0.05
        mass_diagnostics.append({
            "z": z, "mass_parameter_u": a, "backend_reference_mass_u": reference_mass,
            "ls_mass_linear_weight": ls_weight,
            "ls_mass_extrapolation": not 0.0 <= ls_weight <= 1.0,
            "physical_isotope_assignment_verified": False,
        })
    for material_z in (13, 14):
        material = backend.get_material(material_z)
        target = material.get_element(0)
        if material.ncomponents() != 1 or target.Z != material_z or target.stn != 1:
            raise ValueError("Expected an elemental target.")
        materials.append({"z": material_z, "atomic_mass_u": target.A,
                          "density_g_cm3": material.density()})
        for ion in ions:
            z, a = ion["z"], ion["mass_parameter_u"]
            projectile = backend.Projectile(a, z, z)
            for energy in energies:
                projectile.T(energy)
                total = backend.dedx(projectile, material, config)
                nonradiative = backend.dedx(projectile, material, nonrad_config)
                nuclear = native.nuclear_stopping(z, a, material_z, target.A, energy)
                electronic, radiative = nonradiative - nuclear, total - nonradiative
                mass_range = backend.range(projectile, material, config)
                if (any(not math.isfinite(v) or v < 0 for v in
                        (total, electronic, nuclear, radiative, mass_range))
                        or min(total, electronic, mass_range) <= 0):
                    raise ValueError("Invalid backend data; clipping is forbidden.")
                if energy <= 10:
                    check = backend.sezi_dedx_e(projectile, material, config)
                    if not math.isfinite(check) or check <= 0:
                        raise ValueError("Invalid independent electronic stopping check.")
                    closure_count += 1
                    closure_max = max(closure_max, abs(electronic / check - 1.0))
                derivative = backend.dedx_from_range(projectile, material, config)
                if not math.isfinite(derivative) or derivative <= 0:
                    raise ValueError("Invalid native range derivative.")
                error = abs(derivative / total - 1.0)
                if error > worst["relative_error"]:
                    worst = {"z": z, "mass_parameter_u": a, "material_z": material_z,
                             "energy_mev_per_nucleon": energy, "relative_error": error}
                rows.append({
                    "z": z, "mass_parameter_u": a, "material_z": material_z,
                    "energy_mev_per_nucleon": energy,
                    "stopping_electronic_mev_cm2_g": electronic,
                    "stopping_nuclear_mev_cm2_g": nuclear,
                    "stopping_radiative_mev_cm2_g": radiative, "range_g_cm2": mass_range,
                })
    if closure_max > 1e-10:
        raise ValueError("Nuclear replay failed exposed electronic closure.")
    return {
        "mass_policy": policy, "projectiles": ions, "rows": rows, "materials": materials,
        "mass_diagnostics": mass_diagnostics,
        "generation_checks": {"low_energy_closure_count": closure_count,
                              "low_energy_closure_max_relative_error": closure_max,
                              "range_derivative_worst": worst},
        "backend_configuration": config.get(), "nonradiative_configuration": nonrad_config.get(),
        "normative_export_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--points-per-decade", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read_pinned_manifest(args.manifest, args.manifest_sha256)
    energies = experiment_grid(args.points_per_decade)
    pins = native.verify_sources()
    resources = native.production_hashes()
    scripts = {p: native.sha256(ROOT / p) for p in
               ("scripts/generate_parameterized_stopping.py", "scripts/generate_catima_candidate.py")}
    backend = native.load_backend()
    datasets = {policy: generate_dataset(backend, manifest[field], energies, policy)
                for policy, field in zip(POLICIES, ("projectiles", "pilot_isotopes"), strict=True)}
    read_pinned_manifest(args.manifest, args.manifest_sha256)
    if native.verify_sources() != pins or native.production_hashes() != resources:
        raise ValueError("Source or RADAR files changed during generation.")
    if any(native.sha256(ROOT / p) != value for p, value in scripts.items()):
        raise ValueError("Generator implementation changed.")
    report = {
        "schema_version": "radar.parameterized_stopping.raw.v1",
        "manifest": manifest, "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": args.manifest_sha256, "energies_mev_per_nucleon": energies,
        "material_atomic_numbers": [13, 14], "points_per_decade": args.points_per_decade,
        "datasets": datasets, "production_resources_sha256": resources,
        "source_inventory_independently_reconstructed_by_generator": False,
        "source_pins": pins, "generator_hashes": scripts, "python": sys.version,
        "backend_info_literal": backend.catima_info(),
        "range_origin_energy_mev_per_nucleon": 0.001,
        "energy_coordinate_compatibility": "source_MeV_n_numerically_equal_backend_MeV_u_assumed",
        "projectile_charge_input": "Q=Z; pinned PierceBlann uses equilibrium charge, not SEP A/Q",
        "let_convention": "electronic_collision_mass_stopping_divided_by_1000_in_Si",
        "absolute_range_origin_verified": False, "physical_mass_policy_qualified": False,
        "normative_export_allowed": False, "production_tables_replaced": False,
        "limitations": [
            "Literal source A is a diagnostic mass parameter, not a physical isotope assignment.",
            "No isotopes are inferred by rounding and no source parameter is corrected.",
            "Internal LS mass interpolation may extrapolate; each projectile is marked.",
            "The native range grid is fixed; denser output does not refine its integration.",
            "Internal consistency and paired mass sensitivity are not independent physical accuracy.",
            "Low-energy absolute stopping, fragmentation and release eligibility remain unqualified.",
            "Pinned pycatima 1.982 wheel, CATIMA source v1.91; wheel not locally rebuilt.",
            "AGPL-3.0 source; integration and redistribution assessment remain open.",
        ],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": native.sha256(args.output),
                      "rows": {p: len(d["rows"]) for p, d in datasets.items()},
                      "normative_export_allowed": False}))


if __name__ == "__main__":
    main()
