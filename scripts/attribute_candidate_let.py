"""Attribute diagnostic HZE LET to species and exit-energy bands, without refitting.

This replays the frozen full-roster transport experiment. Fractions describe
supported primary counts, not a bound on physical error or Al path sensitivity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import re
from typing import Any

from compare_parameterized_stopping_grids import _read_verified
from validate_candidate_transport import ENERGY_EDGES, LET_EDGES, SHIELDS, THRESHOLDS, _source_inputs
from validate_ion_stopping_candidate import _resource_hashes
from validate_parameterized_stopping import run_validator, write_report_exclusively

from radar.core.spectra import Spectrum1D
from radar.core.types import AssessmentMode, SpectrumCompleteness
from radar.core.units import Unit
from radar.validation.candidate_let_attribution import attribute_candidate_electronic_let
from radar.validation.candidate_transport import (
    CandidateIonTable, MASS_POLICIES, transport_candidate_primary,
)
from radar.validation.parameterized_stopping import ParameterizedStoppingRow


ROOT = Path(__file__).resolve().parents[1]
EXIT_ENERGY_EDGES = (0.01, 0.03, 0.1, 0.3, 1.0, 10.0, 100.0, 1000.0, 100000.0)
LET_THRESHOLDS = (0.0, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0)
FALSE_FLAGS = (
    "stage4_acceptance_matrix_completed", "physical_accuracy_verified",
    "normative_export_allowed", "production_tables_replaced",
)
IMPLEMENTATION_PATHS = (
    "scripts/attribute_candidate_let.py",
    "scripts/validate_candidate_transport.py",
    "scripts/compare_parameterized_stopping_grids.py",
    "scripts/validate_parameterized_stopping.py",
    "scripts/validate_ion_stopping_candidate.py",
    "scripts/build_stopping_mass_manifest.py",
    "scripts/generate_parameterized_stopping.py",
    "src/radar/validation/candidate_let_attribution.py",
    "src/radar/validation/candidate_transport.py",
    "src/radar/validation/candidate_let.py",
    "src/radar/shielding/primary_let.py",
    "src/radar/validation/parameterized_stopping.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pins() -> dict[str, str]:
    return {p: _sha(ROOT / p) for p in IMPLEMENTATION_PATHS}


def _fingerprints(sources: tuple[Spectrum1D, ...]) -> list[dict[str, Any]]:
    # Match the frozen runner's full source identity, including warnings/units.
    return [{"source": s.source.value,
             "species": asdict(s.species) if s.species is not None else None,
             "mode": s.assessment_mode,
             "sha256": hashlib.sha256(json.dumps(asdict(s), sort_keys=True,
                 ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
             "energy_domain": [s.x[0], s.x[-1]], "point_count": len(s.x),
             "incomplete_reasons": list(s.incomplete_reasons)} for s in sources]


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative integral.")
    return float(value)


def _share(value: float, total: float) -> float | None:
    # A zero denominator is unavailable, never a proven zero relative contribution.
    return value / total if total > 0 else None


def _close(actual: float, expected: Any, name: str) -> None:
    expected = _number(expected)
    if not isclose(actual, expected, rel_tol=2e-9, abs_tol=0):
        raise ValueError(f"Frozen transport reference mismatch: {name}.")


def _index(cases: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed = {tuple(case["key"]): case for case in cases}
    if len(indexed) != len(cases):
        raise ValueError("Duplicate source/species/mode/shield reference case.")
    return indexed


def _case(
    source: Spectrum1D, table: CandidateIonTable, thickness: float, reference: dict[str, Any],
) -> dict[str, Any]:
    transport = transport_candidate_primary(
        source, table, thickness, source_mass_parameter_u=table.mass_parameter_u,
        output_edges=None if thickness == 0 else ENERGY_EDGES,
    )
    if (transport.normative_export_allowed
            or transport.spectrum.completeness is not SpectrumCompleteness.INCOMPLETE):
        raise ValueError("Attribution must not promote diagnostic transport to production.")
    if asdict(transport.ledger) != reference["primary_ledger"]:
        raise ValueError("Frozen primary transport ledger mismatch.")
    expected_let_unit = (Unit.DIFFERENTIAL_LET_FLUX if source.assessment_mode is AssessmentMode.PEAK
                         else Unit.DIFFERENTIAL_LET_FLUENCE)
    if (reference["energy_y_unit"] != transport.spectrum.y_unit.value
            or reference["let_y_unit"] != expected_let_unit.value):
        raise ValueError("Frozen reference energy/LET units mismatch.")
    if set(reference["let_tail_integrals"]) != {str(t) for t in THRESHOLDS}:
        raise ValueError("Frozen reference LET threshold roster mismatch.")
    result = attribute_candidate_electronic_let(
        transport, energy_edges=EXIT_ENERGY_EDGES, let_thresholds=LET_THRESHOLDS,
    )
    if (result.normative_export_allowed or result.physical_table_accuracy_verified
            or result.error_bound_established or result.al_path_accuracy_assessed):
        raise ValueError("Attribution qualification flags must remain false.")
    old = reference["let_ledger"]
    _close(result.outside_si_domain_integral, old["outside_si_domain_integral"], "Si support")
    _close(result.full_supported_tail_integrals[0],
           fsum(old[k] for k in ("mapped_integral", "below_let_window_integral",
                                "above_let_window_integral")), "total Si counts")
    for threshold, value in zip(LET_THRESHOLDS[1:], result.full_supported_tail_integrals[1:]):
        _close(value, reference["let_tail_integrals"][str(threshold)], f"LET >= {threshold}")
    assert source.species is not None and source.assessment_mode is not None
    key = [source.source.value, source.species.atomic_number, source.assessment_mode.value, thickness]
    if key != reference["key"] or table.mass_parameter_u != reference["mass_parameter_u"]:
        raise ValueError("Reference case source/mass identity mismatch.")
    return {
        "key": key, "symbol": source.species.symbol, "mass_parameter_u": table.mass_parameter_u,
        "integral_unit": "cm^-2 s^-1" if source.assessment_mode is AssessmentMode.PEAK else "cm^-2",
        "primary_ledger": asdict(transport.ledger), "attribution": asdict(result),
        "incomplete_reasons": list(transport.spectrum.incomplete_reasons),
    }


def _aggregate(cases: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {(member.lower(), ion["z"], mode.value, shield)
                for ion in inventory for member in ion["source_membership"]
                for mode in AssessmentMode for shield in SHIELDS}
    indexed = _index(cases)
    if set(indexed) != expected:
        raise ValueError("INCOMPLETE species/source/mode/shield inventory; aggregation refused.")
    identities = {ion["z"]: (ion["symbol"], ion["mass_parameter_u"]) for ion in inventory}
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for case in cases:
        source, z, mode, shield = case["key"]
        if (case["symbol"], case["mass_parameter_u"]) != identities[z]:
            raise ValueError("Species symbol or source mass identity changed.")
        expected_unit = "cm^-2 s^-1" if mode == AssessmentMode.PEAK.value else "cm^-2"
        if case["integral_unit"] != expected_unit:
            raise ValueError("Temporal integral unit mismatch.")
        attr = case["attribution"]
        if (tuple(attr["thresholds"]) != LET_THRESHOLDS
                or tuple(attr["energy_edges"]) != EXIT_ENERGY_EDGES):
            raise ValueError("Attribution coordinate mismatch.")
        if (len(attr["full_supported_tail_integrals"]) != len(LET_THRESHOLDS)
                or len(attr["band_tail_integrals"]) != len(EXIT_ENERGY_EDGES) - 1
                or any(len(row) != len(LET_THRESHOLDS) for row in attr["band_tail_integrals"])):
            raise ValueError("Incomplete attribution array shape.")
        groups.setdefault((source, mode, shield), []).append(case)
    output = []
    for key, items in sorted(groups.items()):
        tails = [fsum(_number(c["attribution"]["full_supported_tail_integrals"][j]) for c in items)
                 for j in range(len(LET_THRESHOLDS))]
        bands = [[fsum(_number(c["attribution"]["band_tail_integrals"][i][j]) for c in items)
                  for j in range(len(LET_THRESHOLDS))] for i in range(len(EXIT_ENERGY_EDGES) - 1)]
        contributions = [{
            "symbol": c["symbol"], "z": c["key"][1], "mass_parameter_u": c["mass_parameter_u"],
            "tail_integrals": c["attribution"]["full_supported_tail_integrals"],
            "tail_fraction_of_supported_all_species": [
                _share(v, total) for v, total in zip(
                    c["attribution"]["full_supported_tail_integrals"], tails, strict=True)],
        } for c in sorted(items, key=lambda c: c["key"][1])]
        output.append({
            "key": list(key), "ion_count": len(items), "integral_unit": items[0]["integral_unit"],
            "supported_tail_integrals": tails, "band_tail_integrals": bands,
            "band_fraction_of_supported_all_species": [
                [_share(v, total) for v, total in zip(row, tails, strict=True)] for row in bands],
            "source_integral": fsum(_number(c["primary_ledger"]["source_integral"]) for c in items),
            "outside_al_domain_integral": fsum(_number(
                c["primary_ledger"]["outside_al_domain_integral"]) for c in items),
            "unresolved_low_residual_integral": fsum(_number(
                c["primary_ledger"]["unresolved_low_residual_integral"]) for c in items),
            "outside_si_domain_integral": fsum(_number(
                c["attribution"]["outside_si_domain_integral"]) for c in items),
            "species_contributions": contributions,
        })
    return output


def _changes(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(left) != len(right):
        raise ValueError("Mismatched coarse/dense attribution groups.")
    comparisons = []
    for a, b in zip(left, right, strict=True):
        if a["key"] != b["key"] or a["integral_unit"] != b["integral_unit"]:
            raise ValueError("Mismatched coarse/dense attribution identity.")
        changes = []
        for i, (row_a, row_b) in enumerate(zip(a["band_tail_integrals"], b["band_tail_integrals"],
                                             strict=True)):
            for t, v, w in zip(LET_THRESHOLDS, row_a, row_b, strict=True):
                changes.append({"energy_band_index": i, "let_threshold": t,
                                "coarse_integral": v, "dense_integral": w,
                                "absolute_change": abs(w - v),
                                "relative_to_coarse": _share(abs(w - v), v),
                                "both_zero": v == 0 and w == 0})
        comparisons.append({"key": a["key"], "band_tail_changes": changes})
    return comparisons


def run_attribution(reference_path: Path, *, reference_sha256: str) -> dict[str, Any]:
    if not isinstance(reference_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", reference_sha256):
        raise ValueError("An explicit lowercase SHA256 reference pin is required.")
    before = _pins()
    ref = _read_verified(reference_path, reference_sha256)
    if ref.get("schema_version") != "radar.candidate_primary_electronic_let.experiment.v1":
        raise ValueError("Unsupported frozen transport schema.")
    if any(ref.get(flag) is not False for flag in FALSE_FLAGS):
        raise ValueError("Frozen transport qualification flags must be literally false.")
    if (ref.get("full_positive_inventory_selected") is not True
            or ref.get("numerical_table_consistency_passed") is not True
            or ref.get("all_zero_shield_identities_passed") is not True):
        raise ValueError("A complete positive roster and successful numerical controls are required.")
    if (ref["implementation_sha256_before"] != ref["implementation_sha256_after"]
            or ref["shield_g_cm2"] != list(SHIELDS) or ref["energy_edges"] != list(ENERGY_EDGES)
            or ref["let_edges"] != list(LET_EDGES) or LET_THRESHOLDS[1:] != THRESHOLDS
            or ref["mass_policy"] != MASS_POLICIES[0]):
        raise ValueError("Frozen transport settings or implementation identity mismatch.")
    _resource_hashes(ref["implementation_sha256_after"], ROOT)
    _resource_hashes(ref["production_resources_sha256"], ROOT)
    if len(ref["candidate_paths"]) != 2 or len(ref["candidate_sha256"]) != 2:
        raise ValueError("Exactly two pinned candidates are required.")
    paths = [Path(p) for p in ref["candidate_paths"]]
    raws = [_read_verified(p, sha) for p, sha in zip(paths, ref["candidate_sha256"], strict=True)]
    validations = [run_validator(p) for p in paths]
    policy = MASS_POLICIES[0]
    for raw, validation, sha in zip(raws, validations, ref["candidate_sha256"], strict=True):
        if (validation["input_sha256_after"] != sha
                or validation["numerical_consistency_passed"] is not True
                or validation["production_resources_sha256_after"] != ref["production_resources_sha256"]
                or raw["manifest_sha256"] != ref["manifest_sha256"]
                or raw["datasets"][policy]["projectiles"] != ref["selected_projectiles"]):
            raise ValueError("Validated candidate identity or numerical status mismatch.")
    for key in ("manifest", "source_pins", "energy_coordinate_compatibility", "let_convention",
                "projectile_charge_input", "range_origin_energy_mev_per_nucleon"):
        if raws[0][key] != raws[1][key]:
            raise ValueError(f"Candidate semantics differ: {key}.")
    for key in ("backend_configuration", "nonradiative_configuration", "materials"):
        if raws[0]["datasets"][policy][key] != raws[1]["datasets"][policy][key]:
            raise ValueError(f"Candidate configuration differs: {key}.")
    coarse_grid, dense_grid = [r["energies_mev_per_nucleon"] for r in raws]
    if not (len(coarse_grid) < len(dense_grid) and set(coarse_grid).issubset(dense_grid)):
        raise ValueError("Strictly nested table grid refinement is required.")
    inventory = ref["selected_projectiles"]
    sources = _source_inputs(inventory)
    fingerprints = _fingerprints(sources)
    if fingerprints != ref["source_fingerprints"] or len(sources) != ref["source_mode_count"]:
        raise ValueError("Frozen source fingerprints do not reproduce.")
    grids: list[dict[str, Any]] = []
    for label, raw, validation in zip(("coarse", "dense"), raws, validations, strict=True):
        grouped: dict[int, list[ParameterizedStoppingRow]] = {p["z"]: [] for p in inventory}
        for row in raw["datasets"][policy]["rows"]:
            grouped[row["z"]].append(ParameterizedStoppingRow(**row))
        tables = {z: CandidateIonTable.from_rows(rows, mass_policy=policy,
            dataset_ref=f"diagnostic://parameterized-stopping/sha256/{validation['input_sha256_after']}")
            for z, rows in grouped.items()}
        # Frozen tails stop at ref LET max. Compare to untruncated supported
        # thresholds only after proving the represented Si table lies below it.
        if max(max(t.si_electronic_let) for t in tables.values()) > ref["let_edges"][-1]:
            raise ValueError("Frozen LET window truncates candidate support; reference not comparable.")
        references = _index(ref[f"{label}_cases"])
        expected = {(s.source.value, s.species.atomic_number, s.assessment_mode.value, h)
                    for s in sources if s.species is not None and s.assessment_mode is not None
                    for h in SHIELDS}
        if set(references) != expected or len(references) != ref["case_count_per_grid"]:
            raise ValueError("Incomplete frozen reference case matrix.")
        cases = []
        for i, source in enumerate(sources):
            assert source.species is not None and source.assessment_mode is not None
            z = source.species.atomic_number
            assert z is not None
            for h in SHIELDS:
                key_tuple = (source.source.value, z, source.assessment_mode.value, h)
                cases.append(_case(source, tables[z], h, references[key_tuple]))
            if (i + 1) % 24 == 0:
                print(f"{label}: {i + 1}/{len(sources)} source-mode spectra attributed", flush=True)
        grids.append({"grid": label, "cases": cases, "aggregates": _aggregate(cases, inventory)})
    for path, raw, validation, sha in zip(paths, raws, validations, ref["candidate_sha256"], strict=True):
        if _sha(path) != sha or _sha(Path(raw["manifest_path"])) != raw["manifest_sha256"]:
            raise ValueError("Candidate or manifest changed during attribution.")
        for hashes, base in ((validation["production_resources_sha256_after"], ROOT),
                             (validation["generator_hashes_after"], ROOT),
                             (validation["source_pins_after"], ROOT / "research/ion_stopping_20261031")):
            _resource_hashes(hashes, base)
    _resource_hashes(ref["implementation_sha256_after"], ROOT)
    after = _pins()
    if before != after or _sha(reference_path) != reference_sha256:
        raise ValueError("Reference or attribution implementation changed during execution.")
    return {
        "schema_version": "radar.candidate_let_attribution.v1",
        "scope": "supported_primary_species_and_exit_energy_attribution_not_error_bound",
        "reference_path": str(reference_path.resolve()), "reference_sha256": reference_sha256,
        "candidate_sha256": ref["candidate_sha256"], "mass_policy": policy,
        "selected_projectiles": inventory, "full_positive_inventory_selected": True,
        "source_fingerprints": fingerprints, "source_policy": ref["source_policy"],
        "source_mode_count": len(sources), "case_count_per_grid": len(grids[0]["cases"]),
        "energy_edges_mev_per_nucleon": EXIT_ENERGY_EDGES,
        "let_thresholds_mev_cm2_mg": LET_THRESHOLDS, "shield_g_cm2": SHIELDS,
        "implementation_sha256_before": before, "implementation_sha256_after": after,
        "production_resources_sha256": ref["production_resources_sha256"],
        "grids": grids, "aggregate_grid_changes": _changes(grids[0]["aggregates"], grids[1]["aggregates"]),
        "species_grid_changes": _changes(*[
            [{"key": c["key"], "integral_unit": c["integral_unit"],
              "band_tail_integrals": c["attribution"]["band_tail_integrals"]}
             for c in grid["cases"]] for grid in grids]),
        **dict.fromkeys(FALSE_FLAGS, False), "error_bound_established": False,
        "al_path_accuracy_assessed": False,
        "limitations": [
            "Frozen external source state only; no orbit or geomagnetic/solar-history acceptance.",
            "Fractions use supported primary HZE counts within declared source and Si domains.",
            "Unknown source tails, low residual energies and fragmentation are not zero-proven.",
            "Exit-energy attribution is not Al path visitation or stopping/range sensitivity.",
            "Source mass parameters are unchanged, not isotope-matched to Ag107/Au197/Ne20 experiments.",
            "LET thresholds are inclusive, with no upper LET truncation within represented Si support.",
            "Zero denominators are null; zero coarse/dense pairs are separately identified.",
            "GCR and SEP, peak and cumulative quantities are never pooled; no time factor is applied.",
            "No fitted corrections, physical accuracy tolerance or automatic release criterion follows.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_attribution(args.reference, reference_sha256=args.reference_sha256)
    write_report_exclusively(report, args.output)
    print(json.dumps({"output": str(args.output), "sha256": _sha(args.output),
                      "case_count_per_grid": report["case_count_per_grid"],
                      "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
