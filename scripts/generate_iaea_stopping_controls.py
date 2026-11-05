"""Exact-energy CATIMA controls for pinned experimental heavy-ion records.

Run the native calculation with CPython 3.12 -B. These diagnostic samples do
not refit a stopping model, replace a production table, or establish that a
retained binary was built from the accompanying source distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import generate_catima_candidate as native
from generate_astar_helium_controls import (
    _configuration_values,
    _elemental_i_values,
    _number,
)
from iaea_stopping_reference import build_inventory, load_archives


ROOT = Path(__file__).resolve().parents[1]
MASS_POLICY = "experimental_isotope_mass_number_numeric_u_assumption"
Request = tuple[int, int, int, float]
_RADEF_SOURCE_PATH = "research/iaea_stopping_20261031/javanainen_esa_2009.pdf"
_RADEF_SOURCE_SHA256 = "fb9cca73e7992e1547afb7eccc3b22afbcd186f96acceb5639517f5134604cc9"
_RADEF_REQUESTS: tuple[Request, ...] = tuple(
    (z, a, 14, 9.3) for z, a in ((7, 15), (10, 20), (14, 30), (18, 40), (26, 56), (36, 82), (54, 131))
)
_IMPLEMENTATION_PATHS = (
    "scripts/generate_iaea_stopping_controls.py",
    "scripts/generate_catima_candidate.py",
    "scripts/generate_astar_helium_controls.py",
    "scripts/iaea_stopping_reference.py",
    "src/radar/validation/nist_astar.py",
    "research/ion_stopping_20261031/runtime/include/catima/data_ionisation_potential.h",
)


def _validate_requests(requests: Any) -> tuple[Request, ...]:
    """Canonical unique keys: (projectile Z, explicit isotope A, target Z, E/A)."""
    if not isinstance(requests, tuple) or not requests or len(requests) > 100_000:
        raise ValueError("An immutable nonempty bounded tuple of native requests is required.")
    result = []
    for request in requests:
        if not isinstance(request, tuple) or len(request) != 4:
            raise ValueError("Every native request must contain exactly Z, A, target Z, and energy.")
        z, a, material_z, value = request
        if (type(z) is not int or not 3 <= z <= 92 or type(a) is not int
                or not z <= a <= 300 or type(material_z) is not int or material_z not in (13, 14)):
            raise ValueError("Explicit heavy-ion isotope and elemental Al/Si target are required.")
        energy = _number(value, "experimental energy", positive=True)
        if not .01 <= energy <= 100_000.:
            raise ValueError("Native controls are bounded to .01..100000 MeV/n.")
        result.append((z, a, material_z, energy))
    canonical = tuple(result)
    if any(left >= right for left, right in zip(canonical, canonical[1:])):
        raise ValueError("Native requests must be unique and lexicographically increasing.")
    return canonical


def _material_values(material: Any, material_z: int, elemental_i: float) -> dict[str, Any]:
    target = material.get_element(0)
    if (type(target.Z) is not int or target.Z != material_z or isinstance(target.stn, bool)
            or target.stn != 1 or type(material.ncomponents()) is not int
            or material.ncomponents() != 1):
        raise ValueError("Expected the unchanged elemental Al/Si material.")
    override = _number(material.I(), "material I override")
    if override != 0:
        raise ValueError("Native material I must not be overridden to fit an experiment.")
    return {
        "z": material_z,
        "atomic_mass_u": _number(target.A, "target atomic mass", positive=True),
        "density_g_cm3": _number(material.density(), "target density", positive=True),
        "ionization_potential_override_ev": override,
        "elemental_ionization_potential_ev": _number(elemental_i, "elemental I", positive=True),
        "ionization_potential_policy": "unchanged_backend_internal_elemental_table",
    }


def generate_samples(
    backend: Any, requests: tuple[Request, ...], *, elemental_i_ev: dict[int, float],
) -> dict[str, Any]:
    """Evaluate each experimental isotope/energy key once, without energy interpolation."""
    requests = _validate_requests(requests)
    if not isinstance(elemental_i_ev, dict) or set(elemental_i_ev) != {13, 14}:
        raise ValueError("Both elemental Al/Si ionization potentials are required.")
    for value in elemental_i_ev.values():
        _number(value, "elemental I", positive=True)
    config = native.configuration(backend)
    nonradiative = native.configuration(backend, no_radiative=True)
    full_before = _configuration_values(config, nonradiative=False)
    nonrad_before = _configuration_values(nonradiative, nonradiative=True)
    materials = {z: backend.get_material(z) for z in (13, 14)}
    material_values = {z: _material_values(m, z, elemental_i_ev[z]) for z, m in materials.items()}
    rows = []
    closure_count, closure_max = 0, 0.0
    projectiles: dict[tuple[int, int], Any] = {}
    previous_range: dict[tuple[int, int, int], float] = {}
    for z, a, material_z, energy in requests:
        material = materials[material_z]
        projectile_key = z, a
        if projectile_key not in projectiles:
            projectiles[projectile_key] = backend.Projectile(float(a), z, z)
        projectile = projectiles[projectile_key]
        projectile.T(energy)
        total = _number(backend.dedx(projectile, material, config), "total stopping", positive=True)
        nonrad = _number(backend.dedx(projectile, material, nonradiative),
                         "nonradiative stopping", positive=True)
        nuclear = _number(native.nuclear_stopping(
            z, float(a), material_z, material_values[material_z]["atomic_mass_u"], energy,
        ), "nuclear stopping")
        electronic = _number(nonrad - nuclear, "electronic stopping", positive=True)
        radiative = _number(total - nonrad, "radiative stopping")
        if energy <= 50_000. and radiative != 0:
            raise ValueError("Radiative stopping is nonzero below the retained backend activation energy.")
        mass_range = _number(backend.range(projectile, material, config),
                             "relative range", positive=True)
        group = z, a, material_z
        if mass_range <= previous_range.get(group, -1.):
            raise ValueError("Native relative range must increase within each isotope/material group.")
        previous_range[group] = mass_range
        if energy <= 10.:
            closure = _number(backend.sezi_dedx_e(projectile, material, config),
                              "low-energy electronic closure", positive=True)
            error = abs(electronic / closure - 1.)
            closure_count += 1
            closure_max = max(closure_max, error)
            if error > 1e-10:
                raise ValueError("Nuclear subtraction failed exposed low-energy electronic closure.")
        rows.append({
            "z": z, "mass_number": a, "material_z": material_z,
            "mass_parameter_u": float(a), "charge_input": z,
            "energy_mev_per_nucleon": energy,
            "energy_mev_total_assuming_numeric_mass_number": a * energy,
            "stopping_total_mev_cm2_g": total,
            "stopping_electronic_mev_cm2_g": electronic,
            "stopping_nuclear_mev_cm2_g": nuclear,
            "stopping_radiative_mev_cm2_g": radiative,
            "range_g_cm2": mass_range,
            "nuclear_fraction_of_nonradiative_stopping": nuclear / nonrad,
        })
    if (_configuration_values(config, nonradiative=False) != full_before
            or _configuration_values(nonradiative, nonradiative=True) != nonrad_before):
        raise ValueError("Backend configuration changed during native experimental controls.")
    if any(_material_values(materials[z], z, elemental_i_ev[z]) != values
           for z, values in material_values.items()):
        raise ValueError("Backend material changed during native experimental controls.")
    return {
        "rows": rows, "materials": list(material_values.values()),
        "backend_configuration": full_before, "nonradiative_configuration": nonrad_before,
        "low_energy_electronic_closure_count": closure_count,
        "low_energy_electronic_closure_max_relative_error": closure_max,
    }


def _implementation_hashes() -> dict[str, str]:
    return {name: native.sha256(ROOT / name) for name in _IMPLEMENTATION_PATHS}


def _load_inventory(data_path: Path, references_path: Path) -> dict[str, Any]:
    return dict(build_inventory(load_archives(data_path, references_path)))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _source_input_hashes(data_path: Path, references_path: Path) -> dict[str, str]:
    radef_path = (ROOT / _RADEF_SOURCE_PATH).resolve(strict=True)
    if native.sha256(radef_path) != _RADEF_SOURCE_SHA256:
        raise ValueError("Pinned RADEF author report SHA256 mismatch.")
    paths = (data_path.resolve(strict=True), references_path.resolve(strict=True), radef_path)
    if len(set(paths)) != 3 or any(not p.is_file() for p in paths):
        raise ValueError("Three distinct source files are required for the exact native controls.")
    return {str(p): native.sha256(p) for p in paths}


def _requests_from_inventory(inventory: dict[str, Any]) -> tuple[tuple[Request, ...], dict[Request, list[int]]]:
    records = inventory.get("selected_records")
    if not isinstance(records, list) or not records or len(records) > 100_000:
        raise ValueError("A bounded nonempty selected experimental record inventory is required.")
    links: dict[Request, list[int]] = {}
    seen_records: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Selected experimental records must be explicit objects.")
        try:
            request = (record["z"], record["mass_number"], record["material_z"],
                       record["energy_mev_per_nucleon"])
            record_number = record["csv_record_number"]
        except KeyError as exc:
            raise ValueError("Selected experimental record lacks its exact native key or CSV identity.") from exc
        key = _validate_requests((request,))[0]
        if type(record_number) is not int or record_number < 2 or record_number in seen_records:
            raise ValueError("Selected CSV record identities must be distinct integers, with header=1.")
        seen_records.add(record_number)
        links.setdefault(key, []).append(record_number)
    for values in links.values():
        values.sort()
    requests = _validate_requests(tuple(sorted(set(links) | set(_RADEF_REQUESTS))))
    return requests, links


def generate_controls(data_path: Path, references_path: Path) -> dict[str, Any]:
    implementation_before = _implementation_hashes()
    source_inputs_before = _source_input_hashes(data_path, references_path)
    inventory_before = _load_inventory(data_path, references_path)
    requests, links = _requests_from_inventory(inventory_before)
    selection_before = _canonical_sha256(inventory_before["selected_records"])
    inventory_hash_before = _canonical_sha256(inventory_before)
    native_sources_before = native.verify_sources()
    production_before = native.production_hashes()
    backend = native.load_backend()
    samples = generate_samples(backend, requests, elemental_i_ev=_elemental_i_values())
    for row in samples["rows"]:
        key = row["z"], row["mass_number"], row["material_z"], row["energy_mev_per_nucleon"]
        row["iaea_csv_record_numbers"] = links.get(key, [])
        row["request_origins"] = (["iaea_selected_experimental_records"] if key in links else [])
        if key in _RADEF_REQUESTS:
            row["request_origins"].append("radef_author_report_slide_6")
    inventory_after = _load_inventory(data_path, references_path)
    selection_after = _canonical_sha256(inventory_after["selected_records"])
    inventory_hash_after = _canonical_sha256(inventory_after)
    native_sources_after = native.verify_sources()
    production_after = native.production_hashes()
    implementation_after = _implementation_hashes()
    source_inputs_after = _source_input_hashes(data_path, references_path)
    if (implementation_before != implementation_after or source_inputs_before != source_inputs_after
            or inventory_hash_before != inventory_hash_after or selection_before != selection_after
            or native_sources_before != native_sources_after or production_before != production_after):
        raise ValueError("Source, selected inventory, implementation or production inputs changed during sampling.")
    return {
        "schema_version": "radar.iaea_experimental_stopping_native_controls.v1",
        "status": "INCOMPLETE",
        "data_archive_path": str(data_path.resolve()),
        "references_archive_path": str(references_path.resolve()),
        "source_inputs_sha256_before": source_inputs_before,
        "source_inputs_sha256_after": source_inputs_after,
        "implementation_sha256_before": implementation_before,
        "implementation_sha256_after": implementation_after,
        "native_source_pins_before": native_sources_before,
        "native_source_pins_after": native_sources_after,
        "production_resources_sha256_before": production_before,
        "production_resources_sha256_after": production_after,
        "inventory_canonical_sha256_before": inventory_hash_before,
        "inventory_canonical_sha256_after": inventory_hash_after,
        "selection_canonical_sha256_before": selection_before,
        "selection_canonical_sha256_after": selection_after,
        "iaea_selected_record_count": len(inventory_before["selected_records"]),
        "iaea_unique_request_count": len(links),
        "radef_request_count": len(_RADEF_REQUESTS),
        "unique_native_request_count": len(requests),
        "native_request_key_fields": ["z", "mass_number", "material_z", "energy_mev_per_nucleon"],
        "python": sys.version, "backend_info_literal": backend.catima_info(),
        "mass_policy": MASS_POLICY,
        "energy_coordinate_compatibility": "source_MeV_n_numerically_equal_backend_MeV_u_assumed",
        "stopping_units": "whole_projectile_MeV_cm2_g_no_mass_number_multiplier",
        "native_sampling_policy": "exact_selected_experimental_energies_not_candidate_table_interpolation",
        "range_origin_energy_mev_per_nucleon": .001,
        "radef_source": {
            "path": _RADEF_SOURCE_PATH, "sha256": _RADEF_SOURCE_SHA256,
            "page_one_based": 6, "request_keys": _RADEF_REQUESTS,
            "measurement_convention": "slide_alone_does_not_establish_explicit_nuclear_subtraction",
            "comparison_policy": "compare_electronic_and_nonradiative_total_separately",
        },
        "normative_export_allowed": False, "physical_accuracy_verified": False,
        "physical_mass_policy_qualified": False, "production_tables_replaced": False,
        "source_to_binary_build_verified": False, "absolute_range_origin_verified": False,
        "source_measurement_classification_verified": False, "reference_data_reuse_rights_qualified": False,
        "stage4_acceptance_matrix_completed": False, "fortran_executed": False,
        "limitations": [
            "Exact isotope mass numbers are numeric backend-u parameters, not measured atomic or nuclear masses.",
            "The pinned backend uses Pierce-Blann and SRIM85 without material-I or source-value refitting.",
            "Experimental data can share historical calibration inputs with semi-empirical models.",
            "IAEA electronic stopping controls do not independently validate nuclear stopping or full CSDA ranges.",
            "RADEF slide values require electronic versus nonradiative-total comparison until the full paper is checked.",
            "The native range begins at .001 MeV/u, not an experimentally verified zero-energy origin.",
            "Native source and wheel identities are pinned, but no independent source-to-binary build is asserted.",
        ],
        **samples,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_archive", type=Path)
    parser.add_argument("references_archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = generate_controls(args.data_archive, args.references_archive)
    write_report(report, args.output)
    print(json.dumps({"output_sha256": native.sha256(args.output),
                      "unique_native_request_count": report["unique_native_request_count"],
                      "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
