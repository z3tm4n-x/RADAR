"""Compare matched diagnostic grids and replay the pinned historical pilot control."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any

from generate_parameterized_stopping import POLICIES, reject_constant, unique_object
from validate_parameterized_stopping import run_validator, write_report_exclusively


_PHYSICAL_FIELDS = (
    "energy_mev_per_nucleon", "stopping_electronic_mev_cm2_g",
    "stopping_nuclear_mev_cm2_g", "stopping_radiative_mev_cm2_g", "range_g_cm2",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_verified(path: Path, expected_sha256: str) -> dict[str, Any]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Input changed after individual validation or does not match its pin.")
    raw = json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)
    json.dumps(raw, allow_nan=False)
    if not isinstance(raw, dict):
        raise ValueError("Input must be a JSON object.")
    return raw


def _key(row: dict[str, Any], mass_field: str) -> tuple[int, float, int, float]:
    return row["z"], row[mass_field], row["material_z"], row["energy_mev_per_nucleon"]


def _indexed_rows(rows: list[dict[str, Any]], mass_field: str) -> dict[tuple[int, float, int, float], Any]:
    indexed = {_key(row, mass_field): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate physical row identity.")
    return indexed


def _groups(validation: dict[str, Any]) -> dict[tuple[int, float, int], dict[str, Any]]:
    groups = validation["groups"]
    indexed = {(group["z"], group["mass_parameter_u"], group["material_z"]): group for group in groups}
    if len(indexed) != len(groups):
        raise ValueError("Duplicate validated group identity.")
    return indexed


def _legacy_control(
    raw: dict[str, Any], legacy: dict[str, Any], *, label: str,
) -> dict[str, Any]:
    """Check physical pilot bytes, not historical production-implementation hashes."""
    if legacy.get("schema_version") != "radar.ion_stopping_candidate.raw.v1":
        raise ValueError("Historical pilot must use the original raw.v1 schema.")
    dataset = raw["datasets"][POLICIES[1]]
    expected_ions = [{"z": ion["z"], "mass_number": ion["mass_parameter_u"], "symbol": ion["symbol"]}
                     for ion in dataset["projectiles"]]
    if legacy.get("ions") != expected_ions:
        raise ValueError("Historical pilot isotope inventory differs from the explicit control.")
    for field in ("backend_configuration", "nonradiative_configuration"):
        if legacy.get(field) != dataset[field]:
            raise ValueError(f"Historical pilot differs in {field}.")
    if (legacy.get("backend_info_literal") != raw["backend_info_literal"]
            or legacy.get("material_atomic_numbers") != raw["material_atomic_numbers"]
            or legacy.get("source_provenance", {}).get("hashes") != raw["source_pins"]):
        raise ValueError("Historical pilot backend, materials or source pins differ.")
    materials = legacy.get("materials")
    if not isinstance(materials, list) or len(materials) != len(dataset["materials"]):
        raise ValueError("Historical pilot material records are missing.")
    for old, current in zip(materials, dataset["materials"], strict=True):
        if not isinstance(old, dict) or any(old.get(key) != value for key, value in current.items()):
            raise ValueError("Historical pilot physical material parameters differ.")
        if (old.get("ionization_potential_override_ev") != 0.0
                or old.get("ionization_potential_policy") != "pinned_internal_element_table"):
            raise ValueError("Historical pilot has a different ionization-potential policy.")
    old_nodes = legacy.get("energies_mev_per_nucleon")
    nodes = raw["energies_mev_per_nucleon"]
    if (not isinstance(old_nodes, list) or len(old_nodes) < 2
            or any(isinstance(e, bool) or not isinstance(e, (int, float))
                   or not isfinite(e) or e <= 0 for e in old_nodes)
            or any(a >= b for a, b in zip(old_nodes, old_nodes[1:]))
            or old_nodes[0] != nodes[0] or old_nodes[-1] != nodes[-1]):
        raise ValueError("Historical pilot energy domain is malformed or differs.")
    common = set(old_nodes) & set(nodes)
    old_rows = legacy.get("rows")
    expected_fields = {"z", "mass_number", "material_z", *_PHYSICAL_FIELDS}
    if (not isinstance(old_rows, list) or not old_rows
            or any(not isinstance(row, dict) or set(row) != expected_fields for row in old_rows)):
        raise ValueError("Historical pilot rows do not match its physical schema.")
    indexed_old = _indexed_rows(old_rows, "mass_number")
    expected_old = {(ion["z"], ion["mass_number"], material, energy)
                    for ion in expected_ions for material in (13, 14) for energy in old_nodes}
    if set(indexed_old) != expected_old:
        raise ValueError("Historical pilot is missing or has unexpected physical nodes.")
    compared = 0
    for row in dataset["rows"]:
        if row["energy_mev_per_nucleon"] not in common:
            continue
        old = indexed_old[_key(row, "mass_parameter_u")]
        for field in _PHYSICAL_FIELDS:
            if type(old[field]) is bool or old[field] != row[field]:
                raise ValueError(f"Historical pilot physical value differs in {field}.")
        compared += 1
    expected_count = len(expected_ions) * 2 * len(common)
    if compared != expected_count:
        raise ValueError("Historical pilot comparison did not cover every shared node.")
    return {
        "dataset": label, "shared_energy_node_count": len(common),
        "shared_row_count": compared, "shared_physical_values_exactly_equal": True,
        "historical_production_hashes_checked_against_current": False,
        "historical_production_hash_policy": "retained_as_historical_not_current_implementation",
    }


def compare_grids(
    left_path: Path, right_path: Path, *, legacy_pilot: Path, legacy_pilot_sha256: str,
) -> dict[str, Any]:
    if (not isinstance(legacy_pilot_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", legacy_pilot_sha256) is None):
        raise ValueError("A lowercase historical pilot SHA256 pin is required.")
    legacy = _read_verified(legacy_pilot, legacy_pilot_sha256)
    reports = [run_validator(path) for path in (left_path, right_path)]
    left, right = [_read_verified(path, report["input_sha256_after"])
                   for path, report in zip((left_path, right_path), reports, strict=True)]
    for field in ("production_resources_sha256_after", "generator_hashes_after",
                  "source_pins_after", "manifest_sha256"):
        if reports[0][field] != reports[1][field]:
            raise ValueError(f"Different {field} cannot establish a grid-only comparison.")
    for field in ("manifest", "manifest_sha256", "material_atomic_numbers", "backend_info_literal",
                  "range_origin_energy_mev_per_nucleon", "energy_coordinate_compatibility",
                  "projectile_charge_input", "let_convention", "python"):
        if left[field] != right[field]:
            raise ValueError(f"Different {field} cannot establish a grid-only comparison.")
    a, b = left["energies_mev_per_nucleon"], right["energies_mev_per_nucleon"]
    if len(b) <= len(a) or a[0] != b[0] or a[-1] != b[-1] or not set(a) <= set(b):
        raise ValueError("Right grid must strictly refine the left grid with identical endpoints.")
    comparisons = {}
    for policy in POLICIES:
        first, second = left["datasets"][policy], right["datasets"][policy]
        for field in ("mass_policy", "projectiles", "materials", "backend_configuration",
                      "nonradiative_configuration", "mass_diagnostics"):
            if first[field] != second[field]:
                raise ValueError(f"Different {policy} {field} is not a grid-only comparison.")
        right_rows = _indexed_rows(second["rows"], "mass_parameter_u")
        for row in first["rows"]:
            if row != right_rows.get(_key(row, "mass_parameter_u")):
                raise ValueError(f"Shared-node values differ for {policy} {_key(row, 'mass_parameter_u')}.")
        validations = [report["dataset_validations"][policy] for report in reports]
        groups_left, groups_right = (_groups(validation) for validation in validations)
        if set(groups_left) != set(groups_right):
            raise ValueError("Validated projectile/material groups differ.")
        changes = []
        for key, group in sorted(groups_left.items()):
            denominator = group["integrated_delta_range_g_cm2"]
            numerator = groups_right[key]["integrated_delta_range_g_cm2"]
            relative = numerator / denominator - 1.0
            if not isfinite(relative):
                raise ValueError("Nonfinite integrated range change.")
            changes.append({
                "z": key[0], "mass_parameter_u": key[1], "material_z": key[2],
                "left_integrated_delta_range_g_cm2": denominator,
                "right_integrated_delta_range_g_cm2": numerator,
                "signed_relative_change": relative, "absolute_relative_change": abs(relative),
                "left_max_local_discrepancy": group["max_relative_discrepancy"],
                "right_max_local_discrepancy": groups_right[key]["max_relative_discrepancy"],
            })
        maximum = max(item["absolute_relative_change"] for item in changes)
        comparisons[policy] = {
            "shared_rows_exactly_equal": True, "shared_row_count": len(first["rows"]),
            "groups": changes, "max_absolute_relative_change": maximum,
            "table_quadrature_grid_check_passed": maximum < 0.01,
            "left_local_consistency_passed": validations[0]["numerical_consistency_passed"],
            "right_local_consistency_passed": validations[1]["numerical_consistency_passed"],
            "left_max_local_discrepancy": validations[0]["max_relative_discrepancy"],
            "right_max_local_discrepancy": validations[1]["max_relative_discrepancy"],
        }
    historical = [_legacy_control(raw, legacy, label=label)
                  for raw, label in ((left, "left"), (right, "right"))]
    for path, report in zip((left_path, right_path), reports, strict=True):
        _read_verified(path, report["input_sha256_after"])
    if _sha(legacy_pilot) != legacy_pilot_sha256:
        raise ValueError("Historical pilot changed during comparison.")
    return {
        "schema_version": "radar.parameterized_stopping.grid_comparison.v1",
        "left_input_sha256_before": reports[0]["input_sha256_before"],
        "left_input_sha256_after": _sha(left_path),
        "right_input_sha256_before": reports[1]["input_sha256_before"],
        "right_input_sha256_after": _sha(right_path),
        "historical_pilot_path": str(legacy_pilot.resolve()),
        "historical_pilot_sha256_before": legacy_pilot_sha256,
        "historical_pilot_sha256_after": _sha(legacy_pilot),
        "left_energy_node_count": len(a), "right_energy_node_count": len(b),
        "relative_change_definition": "right_integrated_increment / left_integrated_increment - 1",
        "dataset_comparisons": comparisons, "historical_pilot_controls": historical,
        "manifest_sha256": reports[0]["manifest_sha256"],
        "production_resources_sha256": reports[0]["production_resources_sha256_after"],
        "generator_hashes": reports[0]["generator_hashes_after"],
        "source_pins": reports[0]["source_pins_after"],
        "table_quadrature_relative_tolerance": 0.01,
        "normative_export_allowed": False, "physical_mass_policy_qualified": False,
        "spectral_grid_convergence_established": False, "uniform_local_convergence_established": False,
        "limitations": [
            "Whole-interval quadrature changes do not supersede local range/stopping discrepancies.",
            "An output-grid refinement does not refine the native backend range integration grid.",
            "Exact pilot control preserves the prior backend calculation, not physical truth.",
            "Historical implementation hashes are not compared with deliberately changed production code.",
            "No physical mass convention, stopping-table accuracy or normative release is established.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--legacy-pilot", type=Path, required=True)
    parser.add_argument("--legacy-pilot-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        report = compare_grids(args.left, args.right, legacy_pilot=args.legacy_pilot,
                               legacy_pilot_sha256=args.legacy_pilot_sha256)
        write_report_exclusively(report, args.output)
        print(json.dumps({"output": str(args.output), "sha256": _sha(args.output),
                          "normative_export_allowed": False}))
    except (OSError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
