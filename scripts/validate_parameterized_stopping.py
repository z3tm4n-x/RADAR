"""Validate the isolated mass experiment without loading its native calculator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import build_stopping_mass_manifest as manifest_builder
import generate_catima_candidate as native
import generate_parameterized_stopping as experiment
from validate_ion_stopping_candidate import _resource_hashes

from radar.validation.ion_mass_contract import CandidateIsotope, audit_ost_ion_mass_contract
from radar.validation.parameterized_stopping import (
    ParameterizedStoppingRow, validate_parameterized_rows,
)


_ROOT = Path(__file__).resolve().parents[1]
_POLICIES = experiment.POLICIES
_SOURCE_ROOT = "research/ion_stopping_20261031"
_GENERATOR_PATHS = {
    "scripts/generate_parameterized_stopping.py", "scripts/generate_catima_candidate.py",
}
_VALIDATION_PATHS = (
    "scripts/validate_parameterized_stopping.py", "scripts/validate_ion_stopping_candidate.py",
    "scripts/build_stopping_mass_manifest.py",
)
_CONVENTIONS = {
    "energy_coordinate_compatibility": "source_MeV_n_numerically_equal_backend_MeV_u_assumed",
    "projectile_charge_input": "Q=Z; pinned PierceBlann uses equilibrium charge, not SEP A/Q",
    "let_convention": "electronic_collision_mass_stopping_divided_by_1000_in_Si",
}
_RAW_FIELDS = {
    "schema_version", "manifest", "manifest_path", "manifest_sha256",
    "energies_mev_per_nucleon", "material_atomic_numbers", "points_per_decade", "datasets",
    "production_resources_sha256", "source_pins", "generator_hashes", "python",
    "backend_info_literal", "range_origin_energy_mev_per_nucleon",
    "energy_coordinate_compatibility", "projectile_charge_input", "let_convention",
    "absolute_range_origin_verified", "physical_mass_policy_qualified",
    "normative_export_allowed", "production_tables_replaced", "limitations",
    "source_inventory_independently_reconstructed_by_generator",
}
_DATASET_FIELDS = {
    "mass_policy", "projectiles", "rows", "materials", "mass_diagnostics", "generation_checks",
    "backend_configuration", "nonradiative_configuration", "normative_export_allowed",
}
_FALSE_RAW_FLAGS = (
    "absolute_range_origin_verified", "physical_mass_policy_qualified",
    "normative_export_allowed", "production_tables_replaced",
    "source_inventory_independently_reconstructed_by_generator",
)
_CONFIG = {"z_effective": 1, "corrections": 0, "calculation": 1, "low_energy": 0,
           "scattering": 255, "scattering_factor": 1.0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: Any, required: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} must contain exactly the documented schema fields.")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite numeric, not bool.")
    try:
        valid = isfinite(value) and (not positive or value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"Invalid {label}.")
    return float(value)


def _manifest(raw: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path_value = raw["manifest_path"]
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("Manifest path must be nonempty text.")
    path = Path(path_value).resolve(strict=True)
    if not path.is_relative_to(_ROOT.resolve(strict=True)) or not path.is_file():
        raise ValueError("Manifest path must resolve to a file inside this repository.")
    pin = raw["manifest_sha256"]
    if not isinstance(pin, str):
        raise ValueError("Manifest SHA256 must be text.")
    manifest = experiment.read_pinned_manifest(path, pin)
    if manifest != raw["manifest"]:
        raise ValueError("Embedded manifest differs from its pinned file.")
    for flag in ("normative_export_allowed", "physical_mass_policy_qualified",
                 "transport_tables_generated", "production_tables_replaced"):
        if manifest.get(flag) is not False:
            raise ValueError(f"Manifest {flag} must remain false.")
    if manifest.get("source_roster_inventory_complete") is not True:
        raise ValueError("Manifest must declare source inventory completeness, not transport proof.")
    # No isotope is inferred from source A: these integers are the explicitly
    # declared pilot mass values, already checked by read_pinned_manifest.
    pilots = tuple(CandidateIsotope(ion["z"], int(ion["mass_parameter_u"]), ion["symbol"])
                   for ion in manifest["pilot_isotopes"])
    rebuilt = manifest_builder._source_inventory(audit_ost_ion_mass_contract(pilots).rows)
    for key in ("projectiles", "source_rows", "declared_zero_sources", "source_counts"):
        if manifest.get(key) != rebuilt[key]:
            raise ValueError(f"Manifest {key} differs from independently reconstructed OST sources.")
    candidate_value = manifest.get("candidate_path")
    if not isinstance(candidate_value, str) or not candidate_value:
        raise ValueError("Manifest must name its pinned original pilot candidate.")
    candidate_path = Path(candidate_value).resolve(strict=True)
    if not candidate_path.is_relative_to(_ROOT.resolve(strict=True)) or not candidate_path.is_file():
        raise ValueError("Pilot candidate path must resolve to a file inside this repository.")
    expected_candidate = manifest.get("candidate_expected_sha256")
    if not isinstance(expected_candidate, str):
        raise ValueError("Manifest must retain the original pilot candidate SHA256.")
    # Rebuilding also verifies all pilot identities against the actual pinned
    # v1 candidate and every manifest field, not only source roster cardinality.
    candidate_backed = manifest_builder.build_manifest(
        candidate_path, expected_candidate_sha256=expected_candidate,
    )
    if candidate_backed != manifest:
        raise ValueError("Manifest differs from independently rebuilt candidate-backed manifest.")
    return path, manifest


def _check_config(value: Any, *, nonradiative: bool) -> None:
    config = _object(value, set(_CONFIG), "Backend configuration")
    expected = {**_CONFIG, "corrections": 8 if nonradiative else 0}
    for key, target in expected.items():
        if key == "scattering_factor":
            if _number(config[key], key, positive=True) != target:
                raise ValueError("Unexpected backend configuration value.")
        elif type(config[key]) is not int or config[key] != target:
            raise ValueError("Unexpected backend configuration value.")


def _check_materials(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Exactly two Al/Si material records are required.")
    for entry, z in zip(value, (13, 14), strict=True):
        material = _object(entry, {"z", "atomic_mass_u", "density_g_cm3"}, "Material")
        if type(material["z"]) is not int or material["z"] != z:
            raise ValueError("Material records must be ordered Al then Si.")
        _number(material["atomic_mass_u"], "target mass", positive=True)
        _number(material["density_g_cm3"], "target density", positive=True)


def _check_generation_diagnostics(
    dataset: dict[str, Any], ions: list[dict[str, Any]], energies: list[float],
) -> None:
    diagnostics = dataset["mass_diagnostics"]
    if not isinstance(diagnostics, list) or len(diagnostics) != len(ions):
        raise ValueError("Mass diagnostics must cover every declared projectile exactly once.")
    for entry, ion in zip(diagnostics, ions, strict=True):
        item = _object(entry, {
            "z", "mass_parameter_u", "backend_reference_mass_u", "ls_mass_linear_weight",
            "ls_mass_extrapolation", "physical_isotope_assignment_verified",
        }, "Mass diagnostic")
        if (type(item["z"]) is not int or item["z"] != ion["z"]
                or _number(item["mass_parameter_u"], "diagnostic mass", positive=True)
                != ion["mass_parameter_u"]):
            raise ValueError("Mass diagnostic identity mismatch.")
        reference = _number(item["backend_reference_mass_u"], "reference mass", positive=True)
        weight = _number(item["ls_mass_linear_weight"], "LS mass weight")
        expected = (ion["mass_parameter_u"] / reference - 1.0) / 0.05
        if not isclose(weight, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("LS mass weight arithmetic mismatch.")
        if item["ls_mass_extrapolation"] is not (not 0 <= expected <= 1):
            raise ValueError("LS mass extrapolation flag mismatch.")
        if item["physical_isotope_assignment_verified"] is not False:
            raise ValueError("Mass diagnostic must not assert isotope assignment.")
    checks = _object(dataset["generation_checks"], {
        "low_energy_closure_count", "low_energy_closure_max_relative_error", "range_derivative_worst",
    }, "Generation checks")
    count = checks["low_energy_closure_count"]
    if type(count) is not int or count != 2 * len(ions) * sum(e <= 10 for e in energies):
        raise ValueError("Low energy closure count does not match the declared grid.")
    closure = _number(checks["low_energy_closure_max_relative_error"], "closure error")
    if not 0 <= closure <= 1e-10:
        raise ValueError("Invalid claimed low energy closure error.")
    worst = _object(checks["range_derivative_worst"], {
        "z", "mass_parameter_u", "material_z", "energy_mev_per_nucleon", "relative_error",
    }, "Range derivative diagnostic")
    diagnostic_mass = _number(worst["mass_parameter_u"], "derivative diagnostic mass", positive=True)
    diagnostic_energy = _number(
        worst["energy_mev_per_nucleon"], "derivative diagnostic energy", positive=True,
    )
    identity = (worst["z"], diagnostic_mass)
    if (type(worst["z"]) is not int
            or identity not in {(ion["z"], ion["mass_parameter_u"]) for ion in ions}
            or type(worst["material_z"]) is not int or worst["material_z"] not in (13, 14)
            or diagnostic_energy not in energies
            or _number(worst["relative_error"], "range derivative error") < 0):
        raise ValueError("Invalid range derivative diagnostic identity or error.")


def _paired_comparisons(
    rows_by_policy: dict[str, tuple[ParameterizedStoppingRow, ...]],
    manifest: dict[str, Any], energies: list[float],
) -> list[dict[str, Any]]:
    indexed = {policy: {(row.z, row.material_z, row.energy_mev_per_nucleon): row for row in rows}
               for policy, rows in rows_by_policy.items()}
    source_masses = {ion["z"]: ion["mass_parameter_u"] for ion in manifest["projectiles"]}
    fields_to_compare = (
        "range_g_cm2", "stopping_electronic_mev_cm2_g", "stopping_total_mev_cm2_g",
    )
    result = []
    for pilot in manifest["pilot_isotopes"]:
        for material in (13, 14):
            metrics = {}
            for field in fields_to_compare:
                changes = []
                for energy in energies:
                    key = (pilot["z"], material, energy)
                    denominator = getattr(indexed[_POLICIES[1]][key], field)
                    relative = getattr(indexed[_POLICIES[0]][key], field) / denominator - 1.0
                    _number(relative, "paired relative change")
                    changes.append((energy, relative))
                energy, change = max(changes, key=lambda pair: abs(pair[1]))
                metrics[field] = {"max_absolute_relative_change": abs(change),
                                  "signed_relative_change_at_max": change,
                                  "energy_mev_per_nucleon_at_max": energy,
                                  "points_compared": len(energies)}
            result.append({"z": pilot["z"], "symbol": pilot["symbol"], "material_z": material,
                           "source_mass_parameter_u": source_masses[pilot["z"]],
                           "pilot_isotope_mass_parameter_u": pilot["mass_parameter_u"],
                           "relative_change_definition": "source_parameter / explicit_pilot - 1",
                           "metrics": metrics})
    return result


def run_validator(input_path: Path, *, relative_tolerance: float = 0.01) -> dict[str, Any]:
    implementation_before = {relative: _sha(_ROOT / relative) for relative in _VALIDATION_PATHS}
    payload = input_path.read_bytes()
    before = hashlib.sha256(payload).hexdigest()
    raw = json.loads(payload, object_pairs_hook=experiment.unique_object,
                     parse_constant=experiment.reject_constant)
    json.dumps(raw, allow_nan=False)
    raw = _object(raw, _RAW_FIELDS, "Parameterized raw candidate")
    if raw["schema_version"] != "radar.parameterized_stopping.raw.v1":
        raise ValueError("Unsupported parameterized raw schema.")
    if any(raw[flag] is not False for flag in _FALSE_RAW_FLAGS):
        raise ValueError("Raw qualification flags must remain false.")
    if _number(raw["range_origin_energy_mev_per_nucleon"], "range origin") != 0.001:
        raise ValueError("Expected explicitly relative range origin at 0.001 MeV/u.")
    if raw["material_atomic_numbers"] != [13, 14] or any(
        type(value) is not int for value in raw["material_atomic_numbers"]
    ):
        raise ValueError("Exactly Al and Si material atomic numbers are required.")
    if type(raw["points_per_decade"]) is not int:
        raise ValueError("points_per_decade must be integer.")
    energies = raw["energies_mev_per_nucleon"]
    if not isinstance(energies, list) or energies != list(experiment.experiment_grid(
        raw["points_per_decade"],
    )) or any(isinstance(energy, bool) for energy in energies):
        raise ValueError("Energy grid differs from the declared generator grid.")
    for key in ("python", "backend_info_literal", "energy_coordinate_compatibility",
                "projectile_charge_input", "let_convention"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"{key} must be nonempty text.")
    if any(raw[key] != value for key, value in _CONVENTIONS.items()):
        raise ValueError("Raw coordinate, charge or LET convention differs from the generator.")
    if not isinstance(raw["limitations"], list) or any(
        not isinstance(item, str) or not item for item in raw["limitations"]
    ):
        raise ValueError("Limitations must be a text array.")
    manifest_path, manifest = _manifest(raw)
    if not isinstance(raw["generator_hashes"], dict) or set(raw["generator_hashes"]) != _GENERATOR_PATHS:
        raise ValueError("Both known generator implementation hashes are required.")
    if raw["source_pins"] != native.PINS:
        raise ValueError("Source pins must exactly match the retained CATIMA package/source pins.")
    resources_before = _resource_hashes(raw["production_resources_sha256"], _ROOT)
    generators_before = _resource_hashes(raw["generator_hashes"], _ROOT)
    source_before = _resource_hashes(raw["source_pins"], _ROOT / _SOURCE_ROOT)
    datasets = _object(raw["datasets"], set(_POLICIES), "Mass-policy datasets")
    validations, rows_by_policy = {}, {}
    row_fields = {item.name for item in fields(ParameterizedStoppingRow)}
    for policy, manifest_field in zip(_POLICIES, ("projectiles", "pilot_isotopes"), strict=True):
        dataset = _object(datasets[policy], _DATASET_FIELDS, "Mass-policy dataset")
        if dataset["mass_policy"] != policy or dataset["normative_export_allowed"] is not False:
            raise ValueError("Dataset policy or qualification mismatch.")
        ions = manifest[manifest_field]
        if dataset["projectiles"] != ions:
            raise ValueError("Dataset projectile identities differ from the source-checked manifest.")
        _check_materials(dataset["materials"])
        _check_config(dataset["backend_configuration"], nonradiative=False)
        _check_config(dataset["nonradiative_configuration"], nonradiative=True)
        _check_generation_diagnostics(dataset, ions, energies)
        if not isinstance(dataset["rows"], list) or not dataset["rows"]:
            raise ValueError("Rows must be a nonempty array.")
        row_values = [_object(row, row_fields, "Stopping row") for row in dataset["rows"]]
        rows = tuple(ParameterizedStoppingRow(**row) for row in row_values)
        rows_by_policy[policy] = rows
        validation = validate_parameterized_rows(
            rows, required_projectiles=tuple((ion["z"], ion["mass_parameter_u"]) for ion in ions),
            material_atomic_numbers=(13, 14), energy_nodes_mev_per_nucleon=energies,
            relative_tolerance=relative_tolerance,
        )
        validations[policy] = asdict(validation)
    for field in ("materials", "backend_configuration", "nonradiative_configuration"):
        if datasets[_POLICIES[0]][field] != datasets[_POLICIES[1]][field]:
            raise ValueError(f"Paired datasets differ in {field}.")
    comparisons = _paired_comparisons(rows_by_policy, manifest, energies)
    resources_after = _resource_hashes(raw["production_resources_sha256"], _ROOT)
    generators_after = _resource_hashes(raw["generator_hashes"], _ROOT)
    source_after = _resource_hashes(raw["source_pins"], _ROOT / _SOURCE_ROOT)
    manifest_path_after, manifest_after = _manifest(raw)
    implementation_after = {relative: _sha(_ROOT / relative) for relative in _VALIDATION_PATHS}
    if (before != _sha(input_path) or manifest_after != manifest or manifest_path_after != manifest_path
            or resources_before != resources_after or generators_before != generators_after
            or source_before != source_after or implementation_before != implementation_after):
        raise ValueError("Input, manifest, source or implementation changed during validation.")
    return {
        "schema_version": "radar.parameterized_stopping.validation.v1",
        "input_path": str(input_path.resolve()), "input_sha256_before": before,
        "input_sha256_after": _sha(input_path), "manifest_sha256": raw["manifest_sha256"],
        "production_resources_sha256_before": resources_before,
        "production_resources_sha256_after": resources_after,
        "generator_hashes_before": generators_before, "generator_hashes_after": generators_after,
        "source_pins_before": source_before, "source_pins_after": source_after,
        "validation_implementation_sha256_before": implementation_before,
        "validation_implementation_sha256_after": implementation_after,
        "source_inventory_independently_reconstructed": True,
        "source_counts": manifest["source_counts"],
        "dataset_validations": validations, "paired_mass_comparisons": comparisons,
        "generation_diagnostics": {policy: {key: datasets[policy][key]
            for key in ("generation_checks", "mass_diagnostics")} for policy in _POLICIES},
        "numerical_consistency_passed": all(
            validation["numerical_consistency_passed"] for validation in validations.values()
        ),
        "energy_domain_mev_per_nucleon": [energies[0], energies[-1]],
        "range_origin_energy_mev_per_nucleon": 0.001,
        "absolute_range_origin_verified": False, "physical_mass_policy_qualified": False,
        "normative_export_allowed": False, "production_tables_replaced": False,
        "generation_checks_independently_replayed": False,
        "limitations": [
            "Inventory coverage is reconstructed; physical isotope and backend qualification are not.",
            "Only declared file hashes are verified; no native code is loaded by this validator.",
            "Range increment consistency does not establish the zero-energy range origin.",
            "Paired changes use pilot values as denominator and do not measure physical accuracy.",
            "A denser output grid does not refine the native backend range integration grid.",
        ],
    }


def write_report_exclusively(report: dict[str, Any], output: Path) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        report = run_validator(args.input, relative_tolerance=args.relative_tolerance)
        write_report_exclusively(report, args.output)
        print(json.dumps({"output": str(args.output), "sha256": _sha(args.output),
                          "numerical_consistency_passed": report["numerical_consistency_passed"],
                          "normative_export_allowed": False}))
    except (OSError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
