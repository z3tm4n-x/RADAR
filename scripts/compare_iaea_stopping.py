"""Bounded experimental stopping comparisons; never a physical acceptance test.

IAEA compilation observations and author-report RADEF checkpoints remain
separate. No model fitting, isotope rounding, error-bar interpretation, range
validation, production replacement, or normative qualification is performed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import isclose
from pathlib import Path
import re
from statistics import median
from typing import Any

from compare_astar_helium import CONFIG, _configuration, _number
import generate_catima_candidate as native
import generate_iaea_stopping_controls as controls
from iaea_stopping_reference import build_inventory, load_archives, write_report
from validate_ion_stopping_candidate import _reject_constant, _unique_object


ROOT = Path(__file__).resolve().parents[1]
KEYS = ("z", "mass_number", "material_z", "energy_mev_per_nucleon")
FLAGS = (
    "normative_export_allowed", "physical_accuracy_verified", "physical_mass_policy_qualified",
    "production_tables_replaced", "source_to_binary_build_verified", "absolute_range_origin_verified",
    "stage4_acceptance_matrix_completed", "fortran_executed",
    "source_measurement_classification_verified", "reference_data_reuse_rights_qualified",
)
IMPLEMENTATION = (
    "scripts/compare_iaea_stopping.py", "scripts/compare_astar_helium.py",
    "scripts/validate_ion_stopping_candidate.py", *controls._IMPLEMENTATION_PATHS,
)
INDEX = "research/iaea_stopping_20261031/index.html"
INDEX_SHA256 = "61fa5a4ece771421fc376eca6e6dd772614129fa9051a522968c7f416b86338a"
# Manually transcribed from the MEASURED column, visually checked on PDF page 6.
# The slide's SRIM/LETCalc columns are NOT experimental reference values.
RADEF_MEASURED = ((7, 15, "1.88"), (10, 20, "3.64"), (14, 30, "6.73"),
                  (18, 40, "10.1"), (26, 56, "18.5"), (36, 82, "30.2"), (54, 131, "55.3"))


def _key(row: dict[str, Any]) -> controls.Request:
    try:
        request = (row["z"], row["mass_number"], row["material_z"], row["energy_mev_per_nucleon"])
        return controls._validate_requests((request,))[0]
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing or malformed experimental/native identity.") from exc


def _same_snapshot(payload: dict[str, Any], name: str, expected: Any) -> None:
    if payload.get(name + "_before") != expected or payload.get(name + "_after") != expected:
        raise ValueError(f"Native generation snapshot mismatch: {name}")


def _validate_controls(
    payload: dict[str, Any], inventory: dict[str, Any], data_path: Path, references_path: Path,
) -> dict[controls.Request, dict[str, Any]]:
    """Rebuild source selection and reject stale, incomplete or relabelled controls."""
    if (not isinstance(payload, dict)
            or payload.get("schema_version") != "radar.iaea_experimental_stopping_native_controls.v1"
            or payload.get("status") != "INCOMPLETE"):
        raise ValueError("Unexpected native experimental-control schema.")
    expected_identity = {
        "data_archive_path": str(data_path.resolve()), "references_archive_path": str(references_path.resolve()),
        "mass_policy": controls.MASS_POLICY,
        "energy_coordinate_compatibility": "source_MeV_n_numerically_equal_backend_MeV_u_assumed",
        "stopping_units": "whole_projectile_MeV_cm2_g_no_mass_number_multiplier",
        "native_sampling_policy": "exact_selected_experimental_energies_not_candidate_table_interpolation",
        "range_origin_energy_mev_per_nucleon": .001, "native_request_key_fields": list(KEYS),
    }
    if any(payload.get(k) != v for k, v in expected_identity.items()):
        raise ValueError("Native control physical identity, source paths or units differ.")
    if any(payload.get(flag) is not False for flag in FLAGS):
        raise ValueError("All qualification flags must remain explicitly false.")
    _configuration(payload.get("backend_configuration"), CONFIG)
    _configuration(payload.get("nonradiative_configuration"), {**CONFIG, "corrections": 8})
    _same_snapshot(payload, "source_inputs_sha256", controls._source_input_hashes(data_path, references_path))
    _same_snapshot(payload, "implementation_sha256", controls._implementation_hashes())
    _same_snapshot(payload, "native_source_pins", native.verify_sources())
    _same_snapshot(payload, "production_resources_sha256", native.production_hashes())
    _same_snapshot(payload, "inventory_canonical_sha256", controls._canonical_sha256(inventory))
    _same_snapshot(payload, "selection_canonical_sha256", controls._canonical_sha256(inventory["selected_records"]))
    requests, links = controls._requests_from_inventory(inventory)
    counts = {"iaea_selected_record_count": len(inventory["selected_records"]),
              "iaea_unique_request_count": len(links), "radef_request_count": len(controls._RADEF_REQUESTS),
              "unique_native_request_count": len(requests)}
    if any(type(payload.get(k)) is not int or payload[k] != v for k, v in counts.items()):
        raise ValueError("Native request/observation counts differ from rebuilt selection.")
    expected_radef = {
        "path": controls._RADEF_SOURCE_PATH, "sha256": controls._RADEF_SOURCE_SHA256,
        "page_one_based": 6, "request_keys": [list(k) for k in controls._RADEF_REQUESTS],
        "measurement_convention": "slide_alone_does_not_establish_explicit_nuclear_subtraction",
        "comparison_policy": "compare_electronic_and_nonradiative_total_separately",
    }
    if payload.get("radef_source") != expected_radef:
        raise ValueError("RADEF source or physical-comparison convention differs.")
    materials = payload.get("materials")
    expected_materials = [
        {"z": z, "atomic_mass_u": a, "density_g_cm3": density,
         "ionization_potential_override_ev": 0., "elemental_ionization_potential_ev": potential,
         "ionization_potential_policy": "unchanged_backend_internal_elemental_table"}
        for z, a, density, potential in ((13, 26.9815, 2.699, 166.), (14, 28.0855, 2.33, 173.))
    ]
    if materials != expected_materials or any(type(m["z"]) is not int for m in materials):
        raise ValueError("Unchanged elemental Al/Si metadata are required; no I fitting.")
    for material in materials:
        for field in ("atomic_mass_u", "density_g_cm3", "elemental_ionization_potential_ev",
                      "ionization_potential_override_ev"):
            _number(material[field], zero_allowed=field == "ionization_potential_override_ev")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(requests):
        raise ValueError("Missing/extra native observations.")
    row_fields = set(KEYS) | {
        "mass_parameter_u", "charge_input", "energy_mev_total_assuming_numeric_mass_number",
        "stopping_total_mev_cm2_g", "stopping_electronic_mev_cm2_g", "stopping_nuclear_mev_cm2_g",
        "stopping_radiative_mev_cm2_g", "range_g_cm2", "nuclear_fraction_of_nonradiative_stopping",
        "iaea_csv_record_numbers", "request_origins",
    }
    result = {}
    previous_ranges: dict[tuple[int, int, int], float] = {}
    for row, key in zip(rows, requests, strict=True):
        if not isinstance(row, dict) or set(row) != row_fields or _key(row) != key:
            raise ValueError("Native row identity/order/schema differs.")
        z, a, target, energy = key
        if (type(row["charge_input"]) is not int or row["charge_input"] != z
                or _number(row["mass_parameter_u"]) != a
                or _number(row["energy_mev_total_assuming_numeric_mass_number"]) != a * energy):
            raise ValueError("Native mass, charge or total/per-nucleon energy differs.")
        origins = (["iaea_selected_experimental_records"] if key in links else [])
        if key in controls._RADEF_REQUESTS:
            origins.append("radef_author_report_slide_6")
        if row["iaea_csv_record_numbers"] != links.get(key, []) or row["request_origins"] != origins:
            raise ValueError("Native row does not retain exact original CSV identities and origins.")
        if any(type(v) is not int for v in row["iaea_csv_record_numbers"]):
            raise ValueError("CSV identities must be literal integers.")
        total = _number(row["stopping_total_mev_cm2_g"])
        electronic = _number(row["stopping_electronic_mev_cm2_g"])
        nuclear = _number(row["stopping_nuclear_mev_cm2_g"], zero_allowed=True)
        radiative = _number(row["stopping_radiative_mev_cm2_g"], zero_allowed=True)
        if energy <= 50_000 and radiative != 0:
            raise ValueError("Radiative term is not active in the retained experimental domain.")
        if not isclose(total, electronic + nuclear + radiative, rel_tol=2e-14):
            raise ValueError("Electronic/nuclear/radiative stopping decomposition is inconsistent.")
        fraction = _number(row["nuclear_fraction_of_nonradiative_stopping"], zero_allowed=True)
        if not isclose(fraction, nuclear / (electronic + nuclear), rel_tol=2e-14, abs_tol=1e-16):
            raise ValueError("Nuclear fraction metadata mismatch.")
        mass_range = _number(row["range_g_cm2"])
        if mass_range <= previous_ranges.get((z, a, target), -1):
            raise ValueError("Relative ranges must increase in each exact native group.")
        previous_ranges[z, a, target] = mass_range
        result[key] = row
    count = sum(key[3] <= 10 for key in requests)
    if type(payload.get("low_energy_electronic_closure_count")) is not int or payload["low_energy_electronic_closure_count"] != count:
        raise ValueError("Low-energy closure count mismatch.")
    if _number(payload.get("low_energy_electronic_closure_max_relative_error"), zero_allowed=True) > 1e-10:
        raise ValueError("Generator reported a failed electronic/nuclear closure.")
    return result


def _compare_observation(record: dict[str, Any], native_row: dict[str, Any]) -> dict[str, Any]:
    if _key(record) != _key(native_row):
        raise ValueError("Comparison requires identical experimental isotope, target and energy.")
    reference = _number(record["stopping_native_mev_cm2_g"])
    prediction = _number(native_row["stopping_electronic_mev_cm2_g"])
    provider = record["stopping_converted_mev_cm2_g"]
    if provider is not None:
        provider = _number(provider)
    return {
        **record, "native_prediction_electronic_mev_cm2_g": prediction,
        "native_prediction_nuclear_mev_cm2_g": native_row["stopping_nuclear_mev_cm2_g"],
        "native_nuclear_fraction": native_row["nuclear_fraction_of_nonradiative_stopping"],
        "signed_relative_difference": prediction / reference - 1.,
        "signed_relative_difference_using_provider_conversion": prediction / provider - 1. if provider else None,
        "difference_denominator": "independently_converted_native_experimental_stopping",
        "uncertainty_used_as_acceptance_tolerance": False,
        "experimental_electronic_convention": "IAEA_compilation_assumption_not_individual_article_verification",
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive only; repeat records and varying experiments are not weights."""
    if not rows:
        raise ValueError("Cannot summarize an empty observation group.")
    by_raw: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_raw.setdefault(row["duplicate_group_id"], row)
    unique = list(by_raw.values())

    def describe(values: list[dict[str, Any]]) -> dict[str, Any]:
        worst = max(values, key=lambda row: abs(row["signed_relative_difference"]))
        return {
            "record_count": len(values),
            "median_absolute_relative_difference": median(abs(r["signed_relative_difference"]) for r in values),
            "min_signed_relative_difference": min(r["signed_relative_difference"] for r in values),
            "max_signed_relative_difference": max(r["signed_relative_difference"] for r in values),
            "worst": {key: worst[key] for key in (*KEYS, "csv_record_number", "ref_id", "signed_relative_difference")},
        }

    return {
        "all_source_records": describe(rows), "exact_raw_unique_records": describe(unique),
        "exact_duplicate_excess_records": len(rows) - len(unique),
        "independent_experiment_count": None,
        "summary_is_a_physical_acceptance_test": False,
        "energy_min_mev_per_nucleon": min(r["energy_mev_per_nucleon"] for r in rows),
        "energy_max_mev_per_nucleon": max(r["energy_mev_per_nucleon"] for r in rows),
        "reported_uncertainty_missing_count": sum(r["reported_percentage_error_text"] == "" for r in rows),
    }


def _grouped(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys, key, strict=True)), **_summarize(values)} for key, values in sorted(groups.items())]


def _radef_comparisons(samples: dict[controls.Request, dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for z, a, token in RADEF_MEASURED:
        row = samples[z, a, 14, 9.3]
        measured = float(token)
        electronic = row["stopping_electronic_mev_cm2_g"] / 1000
        nonrad = (row["stopping_electronic_mev_cm2_g"] + row["stopping_nuclear_mev_cm2_g"]) / 1000
        results.append({
            "z": z, "mass_number": a, "material_z": 14, "energy_mev_per_nucleon": 9.3,
            "measured_let_printed_token": token, "measured_let_mev_cm2_mg": measured,
            "reported_uncertainty_literal": "sigma_LET_MEAS < 3%",
            "uncertainty_used_as_acceptance_tolerance": False,
            "native_electronic_let_mev_cm2_mg": electronic,
            "native_nonradiative_total_mev_cm2_mg": nonrad,
            "electronic_signed_relative_difference": electronic / measured - 1,
            "nonradiative_total_signed_relative_difference": nonrad / measured - 1,
            "difference_denominator": "author_report_measured_LET_column",
            "nuclear_subtraction_in_measurement_verified": False,
            "independent_of_IAEA_records_established": False,
        })
    return results


def _implementation_hashes() -> dict[str, str]:
    return {name: native.sha256(ROOT / name) for name in IMPLEMENTATION}


def run_comparison(
    data_path: Path, references_path: Path, native_path: Path, *, native_sha256: str,
) -> dict[str, Any]:
    if not isinstance(native_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", native_sha256) is None:
        raise ValueError("Explicit lowercase native-control SHA256 required.")
    if native.sha256(native_path) != native_sha256:
        raise ValueError("Native-control file hash mismatch.")
    if native.sha256(ROOT / INDEX) != INDEX_SHA256:
        raise ValueError("Retained IAEA landing-page provenance differs.")
    before = _implementation_hashes()
    inventory = build_inventory(load_archives(data_path, references_path))
    payload = json.loads(native_path.read_bytes(), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    json.dumps(payload, allow_nan=False)
    samples = _validate_controls(payload, inventory, data_path, references_path)
    rows = [_compare_observation(record, samples[_key(record)]) for record in inventory["selected_records"]]
    for row in rows:
        energy = row["energy_mev_per_nucleon"]
        row["energy_band"] = ("01_[.01,.1)" if energy < .1 else "02_[.1,1)" if energy < 1 else
                              "03_[1,10)" if energy < 10 else "04_[10,30)" if energy < 30 else
                              "05_[30,100)" if energy < 100 else "06_[100,1000)" if energy < 1000 else
                              "07_[1000,100000]")
    report = {
        "schema_version": "radar.iaea_experimental_stopping_comparison.v1", "status": "INCOMPLETE",
        **{flag: False for flag in FLAGS},
        "native_controls_sha256": native_sha256,
        "native_controls_path": str(native_path.resolve()),
        "source_inventory": inventory,
        "source_inventory_canonical_sha256": controls._canonical_sha256(inventory),
        "native_configuration": payload["backend_configuration"],
        "native_materials": payload["materials"], "mass_policy": payload["mass_policy"],
        "compilation_catalog_discrepancy": {
            "landing_page_sha256": INDEX_SHA256, "landing_page_advertised_data_points": 65269,
            "retained_csv_nonmodel_records_not_independent_measurements": inventory["nonmodel_source_record_count_not_independent_measurements"],
            "difference_csv_minus_advertised": inventory["nonmodel_source_record_count_not_independent_measurements"] - 65269,
            "same_release_identity_established": False,
        },
        "rows": rows, "summary": _summarize(rows),
        "by_reference": _grouped(rows, ("ref_id",)),
        "by_isotope_material": _grouped(rows, ("z", "mass_number", "material_z")),
        "by_reference_isotope_material": _grouped(rows, ("ref_id", "z", "mass_number", "material_z")),
        "by_material_energy_band": _grouped(rows, ("material_z", "energy_band")),
        "coverage_by_material": _grouped(rows, ("material_z",)),
        "radef_source": payload["radef_source"], "radef_checkpoints": _radef_comparisons(samples),
        "radef_method_source": "author report page 5: B-TOF and magnetic dipole; Si samples ~1 and ~12 micrometres",
        "limitations": [
            "Only selected Al/Si solid heavy-ion observations with explicit integral isotope labels are compared.",
            "IAEA is a curated compilation: individual articles, conventions and error budgets are not all verified.",
            "Known model references are excluded; unknown compilation classifications cannot be certified here.",
            "Measured data may have been historical inputs to semiempirical models; out-of-sample independence is not established.",
            "Duplicate rows are retained; unique-raw summaries do not imply statistical independence.",
            "Percent errors and missing uncertainties are not acceptance tolerances or assumed Gaussian standard deviations.",
            "Native isotope coordinates are explicit A in numeric u, not the fractional OST model-mass parameters.",
            "Native exact-energy evaluation is not validation of output-table interpolation or protected spectra.",
            "There are no selected Al data above 985 MeV/u or Si data above 90.63 MeV/u; the full 1e5 domain remains unverified.",
            "Electronic stopping agreement does not establish nuclear stopping, charge-state evolution, or absolute CSDA range accuracy.",
            "RADEF slide checkpoints and IAEA observations can overlap and are not pooled as independent evidence.",
            "Model/source redistribution and integration permissions are not resolved by this diagnostic comparison.",
            "No model, coefficient, table or warning is modified on the basis of these residuals.",
        ],
    }
    # Replay the pinned selection at the end as well; changes during comparison fail closed.
    inventory_after = build_inventory(load_archives(data_path, references_path))
    if controls._canonical_sha256(inventory_after) != controls._canonical_sha256(inventory):
        raise ValueError("Compilation selection changed during comparison.")
    _validate_controls(payload, inventory_after, data_path, references_path)
    after = _implementation_hashes()
    if before != after or native.sha256(native_path) != native_sha256 or native.sha256(ROOT / INDEX) != INDEX_SHA256:
        raise ValueError("Inputs or comparison implementation changed during comparison.")
    report["implementation_sha256_before"] = before
    report["implementation_sha256_after"] = after
    report["production_resources_sha256_before"] = payload["production_resources_sha256_before"]
    report["production_resources_sha256_after"] = payload["production_resources_sha256_after"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_archive", type=Path)
    parser.add_argument("references_archive", type=Path)
    parser.add_argument("native_controls", type=Path)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_comparison(args.data_archive, args.references_archive, args.native_controls,
                            native_sha256=args.native_sha256)
    write_report(report, args.output)
    print(json.dumps({"record_count": len(report["rows"]), "output_sha256": native.sha256(args.output),
                      "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
