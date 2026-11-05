"""Bounded He4 spectrum sensitivity to ASTAR/CATIMA tables, never acceptance.

ASTAR native nodes are used with RADAR's diagnostic piecewise-linear operator;
this is not ASTAR's APOL spline or a replay of online/compiled ASTAR. All source
tails and unsupported residual ranges remain explicitly accounted for. The
four matched-node factorial variants isolate range and electronic-LET table
changes. CATIMA output-grid witnesses are not ASTAR interpolation error bounds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from math import fsum, isclose, isfinite, log10
from pathlib import Path
import re
from typing import Any

from compare_astar_helium import _native_controls, _sample_candidate, _configuration, _number, CONFIG
from compare_parameterized_stopping_grids import _read_verified
import generate_catima_candidate as native
from validate_candidate_transport import _source_inputs
from validate_ion_stopping_candidate import _resource_hashes
from validate_parameterized_stopping import run_validator, write_report_exclusively

from radar.core.spectra import Spectrum1D
from radar.core.types import AssessmentMode, RadiationSource, SpectrumCompleteness
from radar.validation.candidate_let import convert_candidate_electronic_let
from radar.validation.candidate_transport import CandidateIonTable, transport_candidate_primary
from radar.validation.nist_astar import AstarArchive, load_astar_archive
from radar.validation.parameterized_stopping import ParameterizedStoppingRow


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = ROOT / "research/nist_star_20261031/STAR.tar.gz"
POLICY = "source_tabulated_parameter_diagnostic"
VARIANTS = ("CC", "AC", "CA", "AA", "catima_dense", "catima_coarse")
SHIELDS = (0.0, 0.01, 0.1, 1.0, 3.0, 10.0)
ENERGY_THRESHOLDS = (.01, 1., 10., 30., 100.)
LET_THRESHOLDS = (.01, .02, .05, .1, .2, .5, 1., 1.25, 1.4, 1.45, 1.5, 2.)
ENERGY_EDGES = tuple(sorted({.01, 250., *ENERGY_THRESHOLDS,
    *(10 ** (-2 + (log10(250.) + 2) * i / 140) for i in range(1, 140))}))
LET_EDGES = tuple(sorted({0., 10., *LET_THRESHOLDS,
                         *(10 ** (-4 + 5 * i / 100) for i in range(100))}))
IMPLEMENTATION_PATHS = (
    "scripts/compare_astar_helium_impact.py", "scripts/compare_astar_helium.py",
    "scripts/generate_astar_helium_controls.py", "scripts/generate_catima_candidate.py",
    "scripts/validate_candidate_transport.py", "scripts/validate_parameterized_stopping.py",
    "scripts/validate_ion_stopping_candidate.py", "scripts/build_stopping_mass_manifest.py",
    "scripts/compare_parameterized_stopping_grids.py",
    "src/radar/validation/candidate_transport.py", "src/radar/validation/candidate_let.py",
    "src/radar/shielding/primary_let.py",
    "src/radar/validation/nist_astar.py", "src/radar/validation/parameterized_stopping.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_text(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("An explicit lowercase SHA256 is required.")


def _source_sha(source: Spectrum1D) -> str:
    return hashlib.sha256(json.dumps(asdict(source), sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _validate_source_roster(sources: tuple[Spectrum1D, ...]) -> None:
    expected = {(source, mode) for source in (RadiationSource.GCR, RadiationSource.SEP)
                for mode in (AssessmentMode.PEAK, AssessmentMode.CUMULATIVE)}
    if (len(sources) != 4 or any(not isinstance(source, Spectrum1D) for source in sources)
            or any(source.species is None or source.species.atomic_number != 2
                   or source.species.mass_number != 4 or source.species.is_aggregate
                   for source in sources)
            or {(source.source, source.assessment_mode) for source in sources} != expected):
        raise ValueError("Four unique GCR/SEP PEAK/CUMULATIVE He4 source inputs are required.")


def _bounded_candidate_table(
    payload: dict[str, Any], *, dataset_sha256: str, label: str,
) -> CandidateIonTable:
    """Crop validated He4 candidate nodes to [.01,250], with a linear 250 endpoint."""
    _sha_text(dataset_sha256)
    if label not in ("catima_coarse", "catima_dense"):
        raise ValueError("Expected an explicitly labelled CATIMA output-grid witness.")
    dataset = payload["datasets"][POLICY]
    if dataset["mass_policy"] != POLICY:
        raise ValueError("Only the declared source mass-parameter policy is supported.")
    groups = {z: tuple(ParameterizedStoppingRow(**row) for row in dataset["rows"]
                       if row["z"] == 2 and row["mass_parameter_u"] == 4.
                       and row["material_z"] == z) for z in (13, 14)}
    energies: dict[int, tuple[float, ...]] = {}
    ranges: tuple[float, ...] = ()
    lets: tuple[float, ...] = ()
    for material_z, rows in groups.items():
        grid = tuple(sorted({.01, 250., *(r.energy_mev_per_nucleon for r in rows
                                        if .01 < r.energy_mev_per_nucleon < 250.)}))
        # This strict sampler also rejects missing/mixed/unsorted source rows
        # and refuses to extrapolate either endpoint.
        sampled = tuple(_sample_candidate(rows, energy) for energy in grid)
        energies[material_z] = grid
        if material_z == 13:
            ranges = tuple(row["range_g_cm2"] for row in sampled)
        else:
            lets = tuple(row["stopping_electronic_mev_cm2_g"] / 1000 for row in sampled)
    return CandidateIonTable(
        2, 4., POLICY,
        f"diagnostic://astar-helium-impact/{label}?candidate_sha256={dataset_sha256}"
        f"&manifest_sha256={payload['manifest_sha256']}&energy_domain=.01..250_MeV_n",
        energies[13], ranges, energies[14], lets,
    )


def _factorial_tables(
    native_controls: dict[str, Any], archive: AstarArchive, *, native_sha256: str,
) -> dict[str, CandidateIonTable]:
    """CC/AC/CA/AA: first letter Al range, second Si electronic LET, same 101 nodes."""
    _sha_text(native_sha256)
    if (native_controls["projectile"] != {"z": 2, "mass_parameter_u": 4., "charge_input": 2}
            or native_controls["archive_sha256"] != archive.archive_sha256):
        raise ValueError("Factorial controls must be the matched He4/native ASTAR archive.")
    reference = {z: tuple(r for r in archive.material(z).rows if .04 <= r.energy_mev_total <= 1000.)
                 for z in (13, 14)}
    total_grid = tuple(r.energy_mev_total for r in reference[13])
    if (len(total_grid) != 101 or total_grid[0] != .04 or total_grid[-1] != 1000.
            or tuple(r.energy_mev_total for r in reference[14]) != total_grid):
        raise ValueError("Both ASTAR materials must supply the identical 101-node overlap grid.")
    controls = {z: tuple(r for r in native_controls["rows"] if r["material_z"] == z)
                for z in (13, 14)}
    for rows in controls.values():
        if (tuple(r["energy_mev_total"] for r in rows) != total_grid
                or any(r["energy_mev_per_nucleon"] != e / 4
                       for r, e in zip(rows, total_grid, strict=True))):
            raise ValueError("CATIMA exact controls must use E_total/4 at the same ASTAR nodes.")
    grid = tuple(e / 4 for e in total_grid)
    al = {"A": tuple(r.range_csda_g_cm2 for r in reference[13]),
          "C": tuple(r["range_g_cm2"] for r in controls[13])}
    si = {"A": tuple(r.stopping_electronic_mev_cm2_g / 1000 for r in reference[14]),
          "C": tuple(r["stopping_electronic_mev_cm2_g"] / 1000 for r in controls[14])}
    return {label: CandidateIonTable(
        2, 4., POLICY, f"diagnostic://astar-helium-impact/{label}?al={label[0]}&si={label[1]}"
        f"&astar_sha256={archive.archive_sha256}&native_sha256={native_sha256}",
        grid, al[label[0]], grid, si[label[1]],
    ) for label in ("CC", "AC", "CA", "AA")}


def _evaluate_case(source: Spectrum1D, table: CandidateIonTable, thickness: float) -> dict[str, Any]:
    """Preserve full source support; common-window integrals have explicit denominators."""
    _number(thickness, zero_allowed=True)
    if (table.z != 2 or table.mass_parameter_u != 4. or table.mass_policy != POLICY
            or table.al_energy[0] != .01 or table.al_energy[-1] != 250.
            or table.si_energy[0] != .01 or table.si_energy[-1] != 250.):
        raise ValueError("Impact tables must retain exact He4 and the same [.01,250] domains.")
    if min(table.si_electronic_let) <= LET_EDGES[0] or max(table.si_electronic_let) > LET_EDGES[-1]:
        raise ValueError("The chosen LET window must cover every supported electronic-LET value.")
    zero = transport_candidate_primary(source, table, 0., source_mass_parameter_u=4.)
    assert zero.exit_support is not None
    source_total = zero.ledger.source_integral
    coverage = {
        "source_integral": source_total,
        "below_common_energy_integral": zero.integral_on_exit(0., .01),
        "common_energy_integral": zero.integral_on_exit(.01, 250.),
        "above_common_energy_integral": zero.integral_on_exit(250., max(250., zero.exit_support[1])),
    }
    coverage["balance_residual"] = fsum((coverage["below_common_energy_integral"],
        coverage["common_energy_integral"], coverage["above_common_energy_integral"], -source_total))
    if abs(coverage["balance_residual"]) > 1e-10 * source_total:
        raise ValueError("Original source common-domain accounting does not close.")
    transport = (zero if thickness == 0 else transport_candidate_primary(
        source, table, thickness, source_mass_parameter_u=4., output_edges=ENERGY_EDGES,
    ))
    let = convert_candidate_electronic_let(transport, let_edges=LET_EDGES)
    for spectrum in (transport.spectrum, let.spectrum):
        if (spectrum.completeness is not SpectrumCompleteness.INCOMPLETE
                or spectrum.production_contract_violations() != (
                    "incomplete spectra cannot be released as production results",)):
            raise ValueError("Impact spectra must retain typed contracts and INCOMPLETE.")
    energy_counts = [transport.integral_on_exit(a, b) for a, b in zip(ENERGY_EDGES, ENERGY_EDGES[1:])]
    let_counts = [y * (b - a) for y, a, b in zip(let.spectrum.y, LET_EDGES, LET_EDGES[1:])]
    captured = fsum(energy_counts)
    if abs(captured - let.ledger.mapped_integral) > 1e-10 * source_total:
        raise ValueError("Full-window electronic LET must account for the common exit-energy measure.")
    if let.ledger.below_let_window_integral != 0 or let.ledger.above_let_window_integral != 0:
        raise ValueError("Declared full LET window unexpectedly truncates the supported response.")
    assert source.species is not None and source.assessment_mode is not None
    metrics = {
        "energy_common_integral": captured,
        "supported_primary_integral": transport.ledger.supported_exit_integral,
        "unresolved_low_residual_integral": transport.ledger.unresolved_low_residual_integral,
        "outside_al_domain_integral": transport.ledger.outside_al_domain_integral,
        "si_mapped_integral": let.ledger.mapped_integral,
        "outside_si_domain_integral": let.ledger.outside_si_domain_integral,
        **{f"energy_ge_{threshold}_to250": fsum(energy_counts[ENERGY_EDGES.index(threshold):])
           for threshold in ENERGY_THRESHOLDS},
        **{f"let_ge_{threshold}_to10": fsum(let_counts[LET_EDGES.index(threshold):])
           for threshold in LET_THRESHOLDS},
    }
    return {
        "key": [source.source.value, source.assessment_mode.value, thickness],
        "source_sha256": _source_sha(source), "source_integral": source_total,
        "source_common_domain_coverage": coverage,
        "source_common_fraction": coverage["common_energy_integral"] / source_total
            if source_total > 0 else None,
        "source_above_common_fraction": coverage["above_common_energy_integral"] / source_total
            if source_total > 0 else None,
        "dataset_ref": table.dataset_ref, "mass_parameter_u": 4.,
        "energy_y_unit": transport.spectrum.y_unit.value, "let_y_unit": let.spectrum.y_unit.value,
        "primary_ledger": asdict(transport.ledger), "let_ledger": asdict(let.ledger),
        "exit_support": transport.exit_support,
        "energy_bin_integrals": energy_counts, "let_bin_integrals": let_counts,
        "energy_histogram_domain": [.01, 250.], "let_histogram_domain": [0., 10.],
        "energy_histogram_is_full_zero_shield_identity": False if thickness == 0 else None,
        "zero_shield_identity": (source.x == transport.spectrum.x and source.y == transport.spectrum.y
            and source.bin_edges == transport.spectrum.bin_edges
            and source.ordinate_basis == transport.spectrum.ordinate_basis) if thickness == 0 else None,
        "metrics": metrics, "incomplete_reasons": list(let.spectrum.incomplete_reasons),
        "normative_export_allowed": False,
    }


def _matched(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("key", "source_sha256", "source_integral", "source_common_domain_coverage",
                "energy_y_unit", "let_y_unit", "mass_parameter_u"):
        if left[key] != right[key]:
            raise ValueError(f"Unmatched source, units or mass for impact comparison: {key}.")
    if set(left["metrics"]) != set(right["metrics"]):
        raise ValueError("Impact metric coverage differs.")


def _impact(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Right-minus-left; a zero left denominator gives None even for zero/zero."""
    _matched(left, right)
    source = left["source_integral"]
    common = left["source_common_domain_coverage"]["common_energy_integral"]
    metrics = {}
    for name, left_value in left["metrics"].items():
        right_value = right["metrics"][name]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0
               for v in (left_value, right_value, source, common)):
            raise ValueError("Impact metrics and denominators must be finite nonnegative values.")
        difference = right_value - left_value
        metrics[name] = {
            "left": left_value, "right": right_value, "signed_difference": difference,
            "source_normalized_signed_difference": difference / source if source > 0 else None,
            "common_source_normalized_signed_difference": difference / common if common > 0 else None,
            "ratio_right_over_left": right_value / left_value if left_value > 0 else None,
            "relative_change": difference / left_value if left_value > 0 else None,
            "both_zero": left_value == 0 and right_value == 0,
            "left_denominator_zero": left_value == 0,
        }
    return {"key": left["key"], "metrics": metrics,
            "left_dataset_ref": left["dataset_ref"], "right_dataset_ref": right["dataset_ref"],
            "difference_definition": "right_minus_left",
            "zero_denominator_policy": "ratio_and_relative_change_None_even_when_both_zero"}


def _factorial_interaction(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """AA-AC-CA+CC on matched source measures, not a physical error estimator."""
    if not {"CC", "AC", "CA", "AA"} <= set(cases):
        raise ValueError("All four matched-node factorial cases are required.")
    base = cases["CC"]
    for label in ("AC", "CA", "AA"):
        _matched(base, cases[label])
    energy_checks = {
        "CC_equals_CA": cases["CC"]["energy_bin_integrals"] == cases["CA"]["energy_bin_integrals"],
        "AC_equals_AA": cases["AC"]["energy_bin_integrals"] == cases["AA"]["energy_bin_integrals"],
    }
    count_checks = {
        f"{left}_equals_{right}": isclose(cases[left]["metrics"]["si_mapped_integral"],
            cases[right]["metrics"]["si_mapped_integral"], rel_tol=1e-10, abs_tol=0.)
        for left, right in (("CC", "CA"), ("AC", "AA"))
    }
    if not all(energy_checks.values()) or not all(count_checks.values()):
        raise ValueError("Changing only Si LET must not change energy or full-window mapped count.")
    source = base["source_integral"]
    common = base["source_common_domain_coverage"]["common_energy_integral"]
    metrics = {}
    for name in base["metrics"]:
        interaction = fsum((cases["AA"]["metrics"][name], -cases["AC"]["metrics"][name],
                            -cases["CA"]["metrics"][name], cases["CC"]["metrics"][name]))
        metrics[name] = {"signed_interaction": interaction,
            "source_normalized_signed_interaction": interaction / source if source > 0 else None,
            "common_source_normalized_signed_interaction": interaction / common if common > 0 else None}
    return {"key": base["key"], "formula": "AA-AC-CA+CC", "metrics": metrics,
            "energy_independent_of_si_table": energy_checks,
            "full_window_count_independent_of_si_table": count_checks}


def _table_metadata(table: CandidateIonTable) -> dict[str, Any]:
    coordinates = {"al_energy": table.al_energy, "al_range": table.al_range,
                   "si_energy": table.si_energy, "si_electronic_let": table.si_electronic_let}
    peak = max(range(len(table.si_energy)), key=table.si_electronic_let.__getitem__)
    return {"dataset_ref": table.dataset_ref, "mass_parameter_u": table.mass_parameter_u,
        "mass_policy": table.mass_policy, "al_node_count": len(table.al_energy),
        "si_node_count": len(table.si_energy), "al_energy_domain": [table.al_energy[0], table.al_energy[-1]],
        "si_energy_domain": [table.si_energy[0], table.si_energy[-1]],
        "al_range_endpoints_g_cm2": [table.al_range[0], table.al_range[-1]],
        "electronic_let_min": min(table.si_electronic_let),
        "electronic_let_max": table.si_electronic_let[peak],
        "electronic_let_max_energy_mev_per_nucleon": table.si_energy[peak],
        "coordinates_sha256": hashlib.sha256(json.dumps(coordinates, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "interpolation": "RADAR_diagnostic_piecewise_linear_not_ASTAR_APOL_spline",
        "normative_export_allowed": False}


def run_experiment(
    coarse_path: Path, dense_path: Path, native_path: Path, *, native_sha256: str,
    archive_path: Path = ARCHIVE_PATH,
) -> dict[str, Any]:
    implementation_before = {p: _sha(ROOT / p) for p in IMPLEMENTATION_PATHS}
    archive = load_astar_archive(archive_path)
    energies = tuple(row.energy_mev_total for row in archive.material(13).rows
                     if row.energy_mev_total >= .04)
    controls = _native_controls(native_path, native_sha256, archive.archive_sha256, energies)
    if controls["archive_member_sha256"] != dict(archive.member_sha256):
        raise ValueError("Native controls do not name the same ASTAR source members.")
    validations = [run_validator(path) for path in (coarse_path, dense_path)]
    if any(validation.get("numerical_consistency_passed") is not True for validation in validations):
        raise ValueError("Both input candidates must pass their declared range/stopping consistency check.")
    candidate_checks = [{
        "input_sha256": validation["input_sha256_after"],
        "numerical_consistency_passed": True,
        "source_policy_max_interval_relative_discrepancy": _number(
            validation["dataset_validations"][POLICY]["max_relative_discrepancy"], zero_allowed=True),
    } for validation in validations]
    payloads = [_read_verified(path, validation["input_sha256_after"])
                for path, validation in zip((coarse_path, dense_path), validations, strict=True)]
    if (payloads[0]["manifest"] != payloads[1]["manifest"]
            or payloads[0]["manifest_sha256"] != payloads[1]["manifest_sha256"]
            or payloads[0]["points_per_decade"] != 100 or payloads[1]["points_per_decade"] != 200):
        raise ValueError("Matched 100/200 candidate grids and the same source-mass manifest are required.")
    for payload in payloads:
        dataset = payload["datasets"][POLICY]
        _configuration(dataset["backend_configuration"], CONFIG)
        _configuration(dataset["nonradiative_configuration"], {**CONFIG, "corrections": 8})
        for material, control in zip(dataset["materials"], controls["materials"], strict=True):
            if any(material[key] != control[key] for key in ("z", "atomic_mass_u", "density_g_cm3")):
                raise ValueError("Candidate/native target constants differ.")
    inventory = [p for p in payloads[0]["datasets"][POLICY]["projectiles"] if p["symbol"] == "He"]
    if (len(inventory) != 1 or inventory[0]["z"] != 2 or inventory[0]["mass_parameter_u"] != 4.
            or set(inventory[0]["source_membership"]) != {"GCR", "SEP"}):
        raise ValueError("The source manifest must contain one exact positive He4 GCR/SEP entry.")
    sources = _source_inputs(inventory)
    _validate_source_roster(sources)
    tables = _factorial_tables(controls, archive, native_sha256=native_sha256)
    for label, payload, validation in zip(("catima_coarse", "catima_dense"), payloads,
                                          validations, strict=True):
        tables[label] = _bounded_candidate_table(payload, dataset_sha256=validation["input_sha256_after"],
                                                 label=label)
    cases: dict[str, list[dict[str, Any]]] = {label: [] for label in VARIANTS}
    impacts: dict[str, list[dict[str, Any]]] = {name: [] for name in (
        "range_AC_minus_CC", "let_CA_minus_CC", "combined_AA_minus_CC",
        "range_AA_minus_CA", "let_AA_minus_AC", "native101_minus_dense_CATIMA",
        "dense200_minus_coarse100_CATIMA",
    )}
    pairs = (("range_AC_minus_CC", "CC", "AC"), ("let_CA_minus_CC", "CC", "CA"),
             ("combined_AA_minus_CC", "CC", "AA"), ("range_AA_minus_CA", "CA", "AA"),
             ("let_AA_minus_AC", "AC", "AA"),
             ("native101_minus_dense_CATIMA", "catima_dense", "CC"),
             ("dense200_minus_coarse100_CATIMA", "catima_coarse", "catima_dense"))
    interactions = []
    for source in sources:
        for shield in SHIELDS:
            current = {label: _evaluate_case(source, tables[label], shield) for label in VARIANTS}
            for label in VARIANTS:
                cases[label].append(current[label])
            for name, left, right in pairs:
                impacts[name].append(_impact(current[left], current[right]))
            interactions.append(_factorial_interaction(current))
        assert source.assessment_mode is not None
        print(f"Checked {source.source.value}/{source.assessment_mode.value}: six shields/six variants",
              flush=True)
    for path, validation, payload in zip((coarse_path, dense_path), validations, payloads, strict=True):
        if (_sha(path) != validation["input_sha256_after"]
                or _sha(Path(payload["manifest_path"])) != payload["manifest_sha256"]):
            raise ValueError("Candidate or mass manifest changed during impact calculation.")
        for field, base in (("production_resources_sha256_after", ROOT),
                             ("generator_hashes_after", ROOT),
                             ("source_pins_after", native.SOURCE)):
            _resource_hashes(validation[field], base)
    for prefix, base in (("native_source_pins", native.SOURCE),
                         ("production_resources_sha256", ROOT), ("implementation_sha256", ROOT)):
        _resource_hashes(controls[prefix + "_after"], base)
    implementation_after = {p: _sha(ROOT / p) for p in IMPLEMENTATION_PATHS}
    if (implementation_before != implementation_after or _sha(native_path) != native_sha256
            or _sha(archive_path) != archive.archive_sha256):
        raise ValueError("Native control, archive or implementation changed during impact calculation.")
    return {
        "schema_version": "radar.astar_helium_spectral_impact.v1",
        "scope": "He4_common_domain_table_sensitivity_fixed_external_sources_not_physical_acceptance",
        "implementation_sha256_before": implementation_before,
        "implementation_sha256_after": implementation_after,
        "candidate_paths": [str(p.resolve()) for p in (coarse_path, dense_path)],
        "candidate_sha256": [v["input_sha256_after"] for v in validations],
        "candidate_consistency_checks": candidate_checks,
        "native_controls_path": str(native_path.resolve()), "native_controls_sha256": native_sha256,
        "archive_path": str(archive_path.resolve()), "archive_sha256": archive.archive_sha256,
        "archive_member_sha256": dict(archive.member_sha256),
        "source_mass_manifest_sha256": payloads[0]["manifest_sha256"],
        "production_resources_sha256": controls["production_resources_sha256_after"],
        "derivative_notice": "Derived from NIST STAR1.2 tabulations using new RADAR piecewise-linear "
            "transport/LET interpolation; not ASTAR APOL spline, compiled program or online replay.",
        "energy_domain_mev_per_nucleon": [.01, 250.], "energy_edges": ENERGY_EDGES,
        "let_edges": LET_EDGES, "energy_thresholds": ENERGY_THRESHOLDS, "let_thresholds": LET_THRESHOLDS,
        "shield_g_cm2": SHIELDS, "variant_order": VARIANTS,
        "factorial_definition": "first letter Al range; second Si electronic LET; A=ASTAR,C=CATIMA native101",
        "table_metadata": {label: _table_metadata(tables[label]) for label in VARIANTS},
        "source_fingerprints": [{"source_sha256": _source_sha(source),
            "source": source.source.value, "mode": source.assessment_mode,
            "energy_domain": [source.x[0], source.x[-1]], "energy_node_count": len(source.x),
            "model_reference": asdict(source.model_reference) if source.model_reference else None,
            "transformation_chain": [asdict(step) for step in source.transformation_chain],
            "incomplete_reasons": source.incomplete_reasons} for source in sources],
        "source_policy": "Unchanged validate_candidate_transport fixed external GCR/SEP source state; "
            "original full source grids retained, neither orbital attenuation nor new time factors.",
        "case_count_per_variant": len(cases["CC"]), "total_case_count": sum(map(len, cases.values())),
        "cases": cases, "impacts": impacts, "factorial_interactions": interactions,
        "all_zero_shield_identities_passed": all(case["zero_shield_identity"] is not False
                                                  for group in cases.values() for case in group),
        "normative_export_allowed": False, "production_tables_replaced": False,
        "physical_accuracy_verified": False, "physical_mass_policy_qualified": False,
        "stage4_acceptance_matrix_completed": False, "astar_interpolation_error_bounded": False,
        "limitations": [
            "The large original high-energy source tail is retained as a separate coverage category.",
            "Zero shielding retains the full source energy identity; the common-window histogram is not that identity.",
            "Positive-shield contributions from incident energy above250MeV/n are unresolved, not assigned zero.",
            "Residual range below the common lower table bound is unresolved, not proven stopped.",
            "Source-normalized and common-source-normalized changes use explicitly different denominators.",
            "Zero baseline ratios are None even when both integrals vanish; absolute differences remain reported.",
            "CATIMA native101 versus dense200 is sparse-grid sensitivity, not a bound on the unknown ASTAR inter-knot curve.",
            "Factorial mixed tables are diagnostic decompositions, not a physical hybrid recommended for production.",
            "All variants use a central spherical-shell straight-path primary CSDA model, with no added angular multiplier.",
            "No straggling, attenuation, fragmentation, physical mass calibration or normative qualification is established.",
            "The selected He4-only thresholds and finite source interval do not qualify other ions or full orbital results.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("dense", type=Path)
    parser.add_argument("native_controls", type=Path)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_experiment(args.coarse, args.dense, args.native_controls,
                            native_sha256=args.native_sha256, archive_path=args.archive)
    write_report_exclusively(report, args.output)
    print(json.dumps({"output_sha256": _sha(args.output), "total_case_count": report["total_case_count"],
                      "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
