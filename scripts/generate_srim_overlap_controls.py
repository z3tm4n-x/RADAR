"""Exact-energy CATIMA controls for an authenticated SRIM overlap request file.

Run with CPython 3.12 -B. The caller reconstructs SRIM requests independently;
this isolated worker checks their pinned identity and bounded numerical shape.
It evaluates stopping only, not projected ranges, CSDA ranges or a table splice.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from math import isclose
from pathlib import Path
import re
import sys
from typing import Any

import generate_catima_candidate as native
from generate_astar_helium_controls import (
    _configuration_values, _elemental_i_values, _number,
)
from generate_iaea_stopping_controls import _material_values


ROOT = Path(__file__).resolve().parents[1]
REQUEST_FIELDS = frozenset({
    "z", "mass_parameter_u_decimal", "material_z", "energy_total_mev_decimal",
    "energy_mev_per_nucleon", "raw_sha256", "native_row_index", "case_name",
})
STOPPING_FIELDS = frozenset(f"stopping_{name}_mev_cm2_g" for name in
                            ("total", "nonradiative", "electronic", "nuclear", "radiative"))
FLAGS = {
    "normative_export_allowed": False, "physical_accuracy_verified": False,
    "physical_mass_policy_qualified": False, "production_tables_replaced": False,
    "source_to_binary_build_verified": False, "source_energy_domain_covered": False,
    "stage4_acceptance_matrix_completed": False, "ranges_computed": False,
    "experimental_validation_performed": False, "independent_models_asserted": False,
    "source_inventory_independently_reconstructed_by_generator": False,
}
IMPLEMENTATION_PATHS = (
    "scripts/generate_srim_overlap_controls.py", "scripts/generate_catima_candidate.py",
    "scripts/generate_astar_helium_controls.py", "scripts/generate_iaea_stopping_controls.py",
    "scripts/iaea_stopping_reference.py", "src/radar/validation/nist_astar.py",
    "research/ion_stopping_20261031/runtime/include/catima/data_ionisation_potential.h",
)
IDENTITY = {
    "mass_policy": "literal_source_A_numeric_u_diagnostic_no_isotope_rounding",
    "energy_policy": "exact_total_energy_divided_by_literal_source_A_then_binary64",
    "stopping_units": "whole_projectile_MeV_cm2_g_no_mass_number_multiplier",
    "native_sampling_policy": "exact_Srim_requested_nodes_no_external_energy_grid_interpolation",
    "projectile_charge_input": "Q=Z; pinned PierceBlann, not SEP A/Q",
}


def _decimal(value: Any, label: str) -> Decimal:
    # Bound parsing work and disallow special values, whitespace and exponent tricks.
    if (not isinstance(value, str) or len(value) > 40
            or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None):
        raise ValueError(f"{label} must be a bounded positive decimal string.")
    result = Decimal(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive.")
    return result


def validate_requests(value: Any) -> tuple[dict[str, Any], ...]:
    """Validate exact source-A coordinates; do not round A or interpolate energy."""
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 100_000:
        raise ValueError("A nonempty bounded list/tuple of requests is required.")
    previous: tuple[int, Decimal, int, Decimal] | None = None
    cases: dict[str, tuple[int, Decimal, int, str]] = {}
    previous_index: dict[str, int] = {}
    result = []
    for request in value:
        if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
            raise ValueError("Request fields differ from the exact overlap schema.")
        z, target = request["z"], request["material_z"]
        if (type(z) is not int or not 2 <= z <= 92 or type(target) is not int
                or target not in (13, 14)):
            raise ValueError("HZE Z=2..92 and elemental Al/Si are required.")
        mass = _decimal(request["mass_parameter_u_decimal"], "source A")
        total = _decimal(request["energy_total_mev_decimal"], "total energy")
        if not z <= mass <= 300:
            raise ValueError("Source A must lie between Z and 300, without isotope rounding.")
        exact_energy = Fraction(total) / Fraction(mass)
        if not Fraction(1, 100) <= exact_energy <= 3000:
            raise ValueError("Overlap controls are bounded to .01..3000 MeV/source-A.")
        energy = request["energy_mev_per_nucleon"]
        if type(energy) is not float or energy != float(exact_energy):
            raise ValueError("Backend energy must equal the exact total-energy/source-A conversion.")
        raw_sha = request["raw_sha256"]
        if not isinstance(raw_sha, str) or re.fullmatch(r"[0-9a-f]{64}", raw_sha) is None:
            raise ValueError("Every request needs a lowercase raw SRIM SHA256.")
        index, case = request["native_row_index"], request["case_name"]
        if type(index) is not int or not 0 <= index < 100_000:
            raise ValueError("A bounded nonnegative native row index is required.")
        if not isinstance(case, str) or not case.strip() or len(case) > 200:
            raise ValueError("A nonempty bounded native case name is required.")
        identity = (z, mass, target, raw_sha)
        if cases.setdefault(case, identity) != identity:
            raise ValueError("A case cannot change its projectile, material or raw source identity.")
        if index <= previous_index.get(case, -1):
            raise ValueError("Native row indices must increase within a case.")
        previous_index[case] = index
        key = z, mass, target, total
        if previous is not None and key <= previous:
            raise ValueError("Requests must be unique and canonically ordered.")
        previous = key
        result.append(dict(request))
    return tuple(result)


def generate_samples(
    backend: Any, requests: Any, *, elemental_i_ev: dict[int, float],
) -> dict[str, Any]:
    """Whole-projectile stopping at requested nodes; never native range calls."""
    requests = validate_requests(requests)
    if not isinstance(elemental_i_ev, dict) or set(elemental_i_ev) != {13, 14}:
        raise ValueError("Both elemental Al/Si ionization potentials are required.")
    for value in elemental_i_ev.values():
        _number(value, "elemental I", positive=True)
    config = native.configuration(backend)
    nonradiative = native.configuration(backend, no_radiative=True)
    full_before = _configuration_values(config, nonradiative=False)
    nonrad_before = _configuration_values(nonradiative, nonradiative=True)
    materials = {z: backend.get_material(z) for z in (13, 14)}
    values = {z: _material_values(m, z, elemental_i_ev[z]) for z, m in materials.items()}
    rows = []
    closure_count, closure_max = 0, 0.0
    projectiles: dict[tuple[int, str], Any] = {}
    for request in requests:
        z, material_z = request["z"], request["material_z"]
        mass = float(Decimal(request["mass_parameter_u_decimal"]))
        energy = request["energy_mev_per_nucleon"]
        key = z, request["mass_parameter_u_decimal"]
        if key not in projectiles:
            projectiles[key] = backend.Projectile(mass, z, z)
        projectile, material = projectiles[key], materials[material_z]
        projectile.T(energy)
        total = _number(backend.dedx(projectile, material, config), "total stopping", positive=True)
        nonrad = _number(backend.dedx(projectile, material, nonradiative),
                         "nonradiative stopping", positive=True)
        nuclear = _number(native.nuclear_stopping(
            z, mass, material_z, values[material_z]["atomic_mass_u"], energy,
        ), "nuclear stopping")
        electronic = _number(nonrad - nuclear, "electronic stopping", positive=True)
        radiative = _number(total - nonrad, "radiative stopping")
        if radiative != 0:
            raise ValueError("Radiative stopping must be zero over this bounded native overlap.")
        if energy <= 10:
            exposed = _number(backend.sezi_dedx_e(projectile, material, config),
                              "exposed electronic stopping", positive=True)
            error = abs(electronic / exposed - 1.)
            closure_count += 1
            closure_max = max(closure_max, error)
            if error > 1e-10:
                raise ValueError("Nuclear subtraction failed exposed low-energy electronic closure.")
        rows.append({
            **request, "stopping_total_mev_cm2_g": total,
            "stopping_nonradiative_mev_cm2_g": nonrad,
            "stopping_electronic_mev_cm2_g": electronic,
            "stopping_nuclear_mev_cm2_g": nuclear,
            "stopping_radiative_mev_cm2_g": radiative,
        })
    if (_configuration_values(config, nonradiative=False) != full_before
            or _configuration_values(nonradiative, nonradiative=True) != nonrad_before):
        raise ValueError("Backend configuration changed during native sampling.")
    if any(_material_values(materials[z], z, elemental_i_ev[z]) != before
           for z, before in values.items()):
        raise ValueError("Backend material changed during native sampling.")
    return {
        "rows": rows, "materials": list(values.values()),
        "backend_configuration": full_before, "nonradiative_configuration": nonrad_before,
        "low_energy_electronic_closure_count": closure_count,
        "low_energy_electronic_closure_max_relative_error": closure_max,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_requests(path: Path, expected_sha256: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("A lowercase request SHA256 is required.")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Request file SHA256 mismatch.")
    payload = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if (not isinstance(payload, dict)
            or payload.get("schema_version") != "radar.srim_catima_overlap_requests.v1"
            or payload.get("status") != "INCOMPLETE"
            or payload.get("normative_export_allowed") is not False):
        raise ValueError("Expected an explicitly unqualified overlap request file.")
    for flag in FLAGS:
        if flag in payload and payload[flag] is not False:
            raise ValueError(f"Request qualification flag must remain false: {flag}")
    return validate_requests(payload.get("requests"))


def _implementation_hashes() -> dict[str, str]:
    return {name: native.sha256(ROOT / name) for name in IMPLEMENTATION_PATHS}


def generate_controls(request_path: Path, expected_sha256: str) -> dict[str, Any]:
    implementation_before = _implementation_hashes()
    requests = read_requests(request_path, expected_sha256)
    native_before, production_before = native.verify_sources(), native.production_hashes()
    backend = native.load_backend()
    samples = generate_samples(backend, requests, elemental_i_ev=_elemental_i_values())
    if read_requests(request_path, expected_sha256) != requests:
        raise ValueError("Request inventory changed during native sampling.")
    native_after, production_after = native.verify_sources(), native.production_hashes()
    implementation_after = _implementation_hashes()
    if (implementation_before != implementation_after or native_before != native_after
            or production_before != production_after):
        raise ValueError("Implementation, native source or production files changed during sampling.")
    return {
        "schema_version": "radar.srim_catima_overlap_native.v1", "status": "INCOMPLETE",
        "request_path": str(request_path.resolve()), "requests_sha256": expected_sha256,
        "request_count": len(requests), "implementation_sha256_before": implementation_before,
        "implementation_sha256_after": implementation_after,
        "native_source_pins_before": native_before, "native_source_pins_after": native_after,
        "production_resources_sha256_before": production_before,
        "production_resources_sha256_after": production_after,
        "python": sys.version, "backend_info_literal": backend.catima_info(),
        **IDENTITY,
        "limitations": [
            "SRIM selection is authenticated and reconstructed by the caller, not this worker.",
            "Source A is a literal numeric parameter, not a newly asserted isotope identity.",
            "CATIMA and SRIM can share source ancestry; this is not independent experimental truth.",
            "Source archives and wheel are pinned; their source-to-binary build is unverified.",
            "Backend target masses/densities/I are unchanged, not fitted to SRIM metadata.",
            "Stopping overlap does not close source tails or validate CSDA/projected range equality.",
            "No external energy extrapolation, table splice, range calculation or production replacement.",
            "The backend's internal LS mass-correction interpolation/extrapolation is retained, not qualified.",
        ], **FLAGS, **samples,
    }


def validate_report(
    payload: Any, request_path: Path, expected_sha256: str, request_plan: Any,
) -> None:
    """Read-only retained-control replay; never import or load a native backend.

    The caller must separately rebuild/authenticate its source request plan.
    Algebraic consistency and nuclear replay are not a re-execution of the
    electronic model. A separately repeated native run remains a distinct check.
    """
    requests = read_requests(request_path, expected_sha256)
    if (not isinstance(request_plan, dict)
            or validate_requests(request_plan.get("requests")) != requests):
        raise ValueError("Caller request plan differs from pinned native input.")
    if (not isinstance(payload, dict)
            or payload.get("schema_version") != "radar.srim_catima_overlap_native.v1"
            or payload.get("status") != "INCOMPLETE"):
        raise ValueError("Unexpected native overlap report schema/status.")
    if any(payload.get(flag) is not False for flag in FLAGS):
        raise ValueError("Every overlap qualification flag must remain explicitly false.")
    if any(payload.get(key) != value for key, value in IDENTITY.items()):
        raise ValueError("Native overlap physical identity/units/policies changed.")
    if (payload.get("request_path") != str(request_path.resolve())
            or payload.get("requests_sha256") != expected_sha256
            or type(payload.get("request_count")) is not int
            or payload["request_count"] != len(requests)):
        raise ValueError("Native overlap request identity/count changed.")
    for name, expected in (
        ("implementation_sha256", _implementation_hashes()),
        ("native_source_pins", native.verify_sources()),
        ("production_resources_sha256", native.production_hashes()),
    ):
        if (payload.get(name + "_before") != expected
                or payload.get(name + "_after") != expected):
            raise ValueError(f"Native overlap snapshot mismatch: {name}")
    for key, correction in (("backend_configuration", 0), ("nonradiative_configuration", 8)):
        config = payload.get(key)
        expected_config = {"z_effective": 1, "corrections": correction, "calculation": 1,
                           "low_energy": 0, "scattering": 255, "scattering_factor": 1.0}
        if not isinstance(config, dict) or config != expected_config:
            raise ValueError("Native overlap configuration changed.")
        for field in expected_config:
            if field == "scattering_factor":
                _number(config[field], field, positive=True)
            elif type(config[field]) is not int:
                raise ValueError("Native configuration options must be integer values, not bool.")
    expected_materials: list[dict[str, Any]] = [
        {"z": z, "atomic_mass_u": mass, "density_g_cm3": density,
         "ionization_potential_override_ev": 0., "elemental_ionization_potential_ev": potential,
         "ionization_potential_policy": "unchanged_backend_internal_elemental_table"}
        for z, mass, density, potential in ((13, 26.9815, 2.699, 166.), (14, 28.0855, 2.33, 173.))
    ]
    materials = payload.get("materials")
    if materials != expected_materials or any(type(item["z"]) is not int for item in materials):
        raise ValueError("Native overlap default material metadata changed.")
    for material in materials:
        for field in ("atomic_mass_u", "density_g_cm3", "elemental_ionization_potential_ev",
                      "ionization_potential_override_ev"):
            _number(material[field], field, positive=field != "ionization_potential_override_ev")
    for field in ("python", "backend_info_literal"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError("Native runtime information must remain explicit.")
    count = sum(request["energy_mev_per_nucleon"] <= 10 for request in requests)
    actual_count = payload.get("low_energy_electronic_closure_count")
    if type(actual_count) is not int or actual_count != count:
        raise ValueError("Native low-energy closure count changed.")
    error = _number(payload.get("low_energy_electronic_closure_max_relative_error"), "closure error")
    if error > 1e-10 or (not count and error != 0):
        raise ValueError("Native low-energy closure is outside its explicit numerical check.")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(requests):
        raise ValueError("Missing or extra native overlap samples.")
    target_masses = {item["z"]: item["atomic_mass_u"] for item in expected_materials}
    for row, request in zip(rows, requests, strict=True):
        if not isinstance(row, dict) or set(row) != REQUEST_FIELDS | STOPPING_FIELDS:
            raise ValueError("Native overlap row fields changed.")
        actual_request = {key: row[key] for key in REQUEST_FIELDS}
        if validate_requests([actual_request])[0] != request:
            raise ValueError("Native overlap row identity/order differs from pinned request.")
        values = {name: _number(row[f"stopping_{name}_mev_cm2_g"], name,
                               positive=name not in ("nuclear", "radiative"))
                  for name in ("total", "nonradiative", "electronic", "nuclear", "radiative")}
        if (values["radiative"] != 0 or values["total"] != values["nonradiative"]
                or not isclose(values["nonradiative"], values["electronic"] + values["nuclear"],
                               rel_tol=2e-14, abs_tol=0.)):
            raise ValueError("Native overlap stopping decomposition changed.")
        expected_nuclear = native.nuclear_stopping(
            request["z"], float(Decimal(request["mass_parameter_u_decimal"])), request["material_z"],
            target_masses[request["material_z"]], request["energy_mev_per_nucleon"],
        )
        if not isclose(values["nuclear"], expected_nuclear, rel_tol=2e-14, abs_tol=0.):
            raise ValueError("Native overlap nuclear term differs from pinned formula replay.")


def write_report(report: dict[str, Any], output: Path) -> None:
    raw = json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True,
                     separators=(",", ":")) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_file", type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = generate_controls(args.request_file, args.request_sha256)
    write_report(report, args.output)
    print(json.dumps({"output_sha256": native.sha256(args.output),
                      "request_count": report["request_count"], "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
