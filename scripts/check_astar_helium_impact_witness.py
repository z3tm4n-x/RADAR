"""Independent incident-energy quadrature for bounded ASTAR/CATIMA He comparisons.

The reference calculation uses NumPy forward maps and explicit domain masks,
not the conservative preimage integrator. It never normalizes a source or
multiplies by mission duration. All outputs remain diagnostic and incomplete.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from math import fsum, isfinite, log
from pathlib import Path
import re
from typing import Any

import numpy as np

from radar.core.spectra import Spectrum1D
from radar.core.types import SpectrumOrdinateBasis
from radar.validation.candidate_transport import CandidateIonTable
from radar.validation.nist_astar import load_astar_archive

import compare_astar_helium_impact as impact
from compare_astar_helium import _native_controls
from generate_parameterized_stopping import reject_constant, unique_object
from validate_candidate_transport import _source_inputs


LET_THRESHOLDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.25, 1.4, 1.45, 1.5)
ENERGY_THRESHOLDS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
_CHUNK_SIZE = 262144
ROOT = Path(__file__).resolve().parents[1]
_OWN_PATH = "scripts/check_astar_helium_impact_witness.py"
_OWN_DEPENDENCIES = (_OWN_PATH, "scripts/generate_parameterized_stopping.py")
_SELECTED = (("sep", "peak", 0.01), ("sep", "peak", 1.0), ("gcr", "peak", 10.0))
_LOCAL_THRESHOLDS = (1.25, 1.4, 1.45, 1.5)


def _validate_samples(values: tuple[int, ...]) -> tuple[int, ...]:
    if (not isinstance(values, tuple) or not values
            or any(type(n) is not int or not 2 <= n <= 20_000_000 for n in values)
            or any(a >= b for a, b in zip(values, values[1:]))):
        raise ValueError("Sample counts must be a nonempty increasing tuple of integers in 2..20000000.")
    return values


def _midpoint_metrics(
    source: Spectrum1D, table: CandidateIonTable, thickness: float, count: int,
) -> dict[str, float]:
    """Integrate j(E) E d(log E) with bounded-memory midpoint quadrature.

    ``mapped_let_integral`` covers all supported Si energies, with no imposed
    LET display window. Thresholds are inclusive, including a possible plateau
    exactly on a threshold. Input coverage is reported even for zero shielding,
    while the actual Al-domain transport exclusion is zero for that identity.
    """
    _validate_samples((count,))
    if (isinstance(thickness, bool) or not isinstance(thickness, (int, float))
            or not isfinite(thickness) or thickness < 0):
        raise ValueError("Shield thickness must be a finite nonnegative real number.")
    if not isinstance(source, Spectrum1D) or not isinstance(table, CandidateIonTable):
        raise ValueError("Typed source and candidate ion table are required.")
    if source.ordinate_basis is SpectrumOrdinateBasis.POINT_SAMPLE:
        nodes = source.x
    elif source.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE:
        if source.bin_edges is None:
            raise ValueError("Bin-average source has no explicit edges.")
        nodes = source.bin_edges
    else:
        raise ValueError("Unsupported source ordinate semantics.")
    if len(nodes) < 2 or nodes[0] <= 0 or nodes[-1] <= nodes[0]:
        raise ValueError("Positive finite nondegenerate source energy support is required.")
    source_x, source_y = np.asarray(nodes, dtype=float), np.asarray(source.y, dtype=float)
    al_e = np.asarray(table.al_energy, dtype=float)
    al_r = np.asarray(table.al_range, dtype=float)
    al_r = al_r - al_r[0]
    si_e = np.asarray(table.si_energy, dtype=float)
    si_l = np.asarray(table.si_electronic_let, dtype=float)
    step = (log(nodes[-1]) - log(nodes[0])) / count
    if not isfinite(step) or step <= 0:
        raise ValueError("Logarithmic source interval collapsed at floating precision.")
    partials: dict[str, list[float]] = {}
    for start in range(0, count, _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, count)
        energy = np.exp(log(nodes[0]) + (np.arange(start, stop, dtype=float) + 0.5) * step)
        if source.ordinate_basis is SpectrumOrdinateBasis.POINT_SAMPLE:
            density = np.interp(energy, source_x, source_y)
        else:
            indices = np.searchsorted(source_x, energy, side="right") - 1
            density = source_y[indices]
        weights = density * energy * step
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Midpoint source weights must be finite and nonnegative.")
        inside_al = (energy >= al_e[0]) & (energy <= al_e[-1])
        if thickness == 0:
            supported = np.ones(energy.shape, dtype=bool)
            unresolved = np.zeros(energy.shape, dtype=bool)
            outside_transport = unresolved
            exit_energy = energy
        else:
            residual = np.interp(energy, al_e, al_r, left=np.nan, right=np.nan) - thickness
            unresolved = inside_al & (residual < 0)
            supported = inside_al & (residual >= 0)
            outside_transport = ~inside_al
            exit_energy = np.full(energy.shape, np.nan)
            exit_energy[supported] = np.interp(residual[supported], al_r, al_e)
        inside_si = supported & (exit_energy >= si_e[0]) & (exit_energy <= si_e[-1])
        let_values = np.full(energy.shape, np.nan)
        let_values[inside_si] = np.interp(exit_energy[inside_si], si_e, si_l)

        def integrate(mask: Any) -> float:
            value = float(np.sum(weights[mask], dtype=np.float64))
            if not isfinite(value) or value < 0:
                raise ValueError("Midpoint integral is not representable as a finite nonnegative value.")
            return value

        metrics = {
            "source_integral": float(np.sum(weights, dtype=np.float64)),
            "inside_al_source_integral": integrate(inside_al),
            "outside_al_source_integral": integrate(~inside_al),
            "outside_al_domain_integral": integrate(outside_transport),
            "unresolved_low_residual_integral": integrate(unresolved),
            "supported_exit_integral": integrate(supported),
            "outside_si_domain_integral": integrate(supported & ~inside_si),
            "mapped_let_integral": integrate(inside_si),
            "exit_below_source_min_integral": integrate(supported & (exit_energy < nodes[0])),
            **{f"let_above_{value}": integrate(inside_si & (let_values >= value))
               for value in LET_THRESHOLDS},
            **{f"energy_above_{value}": integrate(supported & (exit_energy >= value))
               for value in ENERGY_THRESHOLDS},
        }
        for name, value in metrics.items():
            if not isfinite(value) or value < 0:
                raise ValueError("Midpoint integral is not representable as a finite nonnegative value.")
            partials.setdefault(name, []).append(value)
    try:
        results = {name: fsum(values) for name, values in partials.items()}
    except OverflowError as exc:
        raise ValueError("Midpoint integral summation overflowed.") from exc
    if any(not isfinite(value) or value < 0 for value in results.values()):
        raise ValueError("Midpoint totals must be finite and nonnegative.")
    return results


def _localized_high_let_metrics(
    source: Spectrum1D, table: CandidateIonTable, thickness: float, count: int,
) -> dict[str, Any]:
    """Resolve a proven enclosing band, not a whole-source or physical integral.

    All represented LET >= 1.25 must lie within exit energy [.01, 1] MeV/n:
    the lower edge is the exact table support and the upper complement is
    bounded by its linear-interpolation endpoints and knots. Monotone Al range
    maps this broad band into one incident-energy interval. Direct forward
    midpoints integrate that interval, without finding LET roots/preimages.
    """
    _validate_samples((count,))
    if (isinstance(thickness, bool) or not isinstance(thickness, (int, float))
            or not isfinite(thickness) or thickness <= 0):
        raise ValueError("Localized witness requires a positive finite shield thickness.")
    if (not isinstance(source, Spectrum1D) or not isinstance(table, CandidateIonTable)
            or source.ordinate_basis is not SpectrumOrdinateBasis.POINT_SAMPLE):
        raise ValueError("Localized witness requires typed point-sample source and candidate table.")
    lo, hi = .01, 1.
    al_e = np.asarray(table.al_energy, dtype=float)
    al_r = np.asarray(table.al_range, dtype=float)
    al_r = al_r - al_r[0]
    si_e = np.asarray(table.si_energy, dtype=float)
    si_l = np.asarray(table.si_electronic_let, dtype=float)
    if al_e[0] != lo or si_e[0] != lo or min(al_e[-1], si_e[-1]) < hi:
        raise ValueError("The complete exit band must start at the exact Al and Si lower support.")
    outside_bound = float(max(float(np.interp(hi, si_e, si_l)),
                              max(si_l[si_e > hi], default=0.)))
    if not isfinite(outside_bound) or outside_bound >= min(_LOCAL_THRESHOLDS):
        raise ValueError("The represented LET outside the exit band is not below every threshold.")
    targets = np.array((thickness, float(np.interp(hi, al_e, al_r)) + thickness))
    if not np.all(np.isfinite(targets)) or targets[0] < al_r[0] or targets[-1] > al_r[-1]:
        raise ValueError("Al support cannot enclose the complete localized exit band without extrapolation.")
    incident_lo, incident_hi = (float(v) for v in np.interp(targets, al_r, al_e))
    if not source.x[0] <= incident_lo < incident_hi <= source.x[-1]:
        raise ValueError("The original source does not cover the complete localized incident interval.")
    step = (incident_hi - incident_lo) / count
    if not isfinite(step) or step <= 0:
        raise ValueError("Localized incident-energy interval collapsed at floating precision.")
    partials: dict[str, list[float]] = {f"let_above_{v}": [] for v in _LOCAL_THRESHOLDS}
    sample_counts = dict.fromkeys(partials, 0)
    band_integrals = []
    for start in range(0, count, _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, count)
        incident = incident_lo + (np.arange(start, stop, dtype=float) + .5) * step
        weights = np.interp(incident, source.x, source.y) * step
        residual = np.interp(incident, al_e, al_r, left=np.nan, right=np.nan) - thickness
        exit_energy = np.interp(residual, al_r, al_e, left=np.nan, right=np.nan)
        let_values = np.interp(exit_energy, si_e, si_l, left=np.nan, right=np.nan)
        if (not np.all(np.isfinite(weights)) or np.any(weights < 0)
                or not np.all(np.isfinite(let_values))
                or np.any(exit_energy < lo) or np.any(exit_energy > hi)):
            raise ValueError("Localized quadrature left its explicitly supported exit band.")
        band_integrals.append(float(np.sum(weights, dtype=np.float64)))
        for threshold in _LOCAL_THRESHOLDS:
            name = f"let_above_{threshold}"
            mask = let_values >= threshold
            partials[name].append(float(np.sum(weights[mask], dtype=np.float64)))
            sample_counts[name] += int(np.count_nonzero(mask))
    try:
        integrals = {name: fsum(values) for name, values in partials.items()}
        band_integral = fsum(band_integrals)
    except OverflowError as exc:
        raise ValueError("Localized midpoint integral summation overflowed.") from exc
    if any(not isfinite(value) or value < 0 for value in (*integrals.values(), band_integral)):
        raise ValueError("Localized midpoint integrals must be finite and nonnegative.")
    return {
        "exit_energy_band_mev_per_nucleon": [lo, hi],
        "incident_energy_band_mev_per_nucleon": [incident_lo, incident_hi],
        "outside_exit_band_let_upper_bound": outside_bound,
        "outside_exit_band_let_bound_below_threshold": True,
        "minimum_checked_let_threshold": min(_LOCAL_THRESHOLDS),
        "lower_exit_band_is_exact_table_support": True,
        "sample_count": count, "band_source_integral": band_integral,
        "threshold_integrals": integrals, "threshold_sample_counts": sample_counts,
        "source_normalization_applied": False,
        "is_full_source_integral": False,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin_text(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("An explicit lowercase SHA256 is required.")
    return value


def _inside_repo(value: Any) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError("Input path must be explicit text or Path.")
    path = Path(value).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve(strict=True)) or not path.is_file():
        raise ValueError("Witness inputs must be regular files inside this repository.")
    return path


def _read_report(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _pin_text(expected_sha256):
        raise ValueError("Impact report SHA256 mismatch.")
    raw = json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_constant)
    json.dumps(raw, allow_nan=False)
    if (not isinstance(raw, dict)
            or raw.get("schema_version") != "radar.astar_helium_spectral_impact.v1"):
        raise ValueError("Expected a retained He spectral-impact report.")
    for flag in ("normative_export_allowed", "physical_accuracy_verified", "production_tables_replaced",
                 "physical_mass_policy_qualified", "stage4_acceptance_matrix_completed",
                 "astar_interpolation_error_bounded"):
        if raw.get(flag) is not False:
            raise ValueError("Impact report must retain every diagnostic qualification boundary.")
    return raw


def _input_pins(report: dict[str, Any], report_path: Path, report_sha: str) -> dict[str, str]:
    pins = {str(report_path): _pin_text(report_sha)}

    def include(path_value: Any, digest: Any) -> None:
        path = str(_inside_repo(path_value))
        pin = _pin_text(digest)
        if path in pins and pins[path] != pin:
            raise ValueError("Conflicting pins for one witness input.")
        pins[path] = pin

    if (not isinstance(report.get("candidate_paths"), list)
            or not isinstance(report.get("candidate_sha256"), list)
            or len(report["candidate_paths"]) != 2 or len(report["candidate_sha256"]) != 2):
        raise ValueError("Both pinned candidate inputs must remain explicit.")
    for path, digest in zip(report["candidate_paths"], report["candidate_sha256"], strict=True):
        include(path, digest)
    for prefix in ("native_controls", "archive"):
        include(report[f"{prefix}_path"], report[f"{prefix}_sha256"])
    if report["implementation_sha256_before"] != report["implementation_sha256_after"]:
        raise ValueError("Impact report's implementation changed during its calculation.")
    required_implementation = set(impact.IMPLEMENTATION_PATHS)
    if set(report["implementation_sha256_after"]) != required_implementation:
        raise ValueError("Impact report implementation coverage differs from the frozen runner.")
    for field in ("implementation_sha256_after", "production_resources_sha256"):
        hashes = report[field]
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("Nonempty production and implementation hash inventories are required.")
        for relative, digest in hashes.items():
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError("Implementation and production pins must be repository-relative.")
            include(ROOT / relative, digest)
    for path in _OWN_DEPENDENCIES:
        include(ROOT / path, _sha(ROOT / path))
    return pins


def _check_pins(pins: dict[str, str]) -> dict[str, str]:
    actual = {path: _sha(Path(path)) for path in pins}
    if actual != pins:
        raise ValueError("A retained report, input or implementation no longer matches its SHA256.")
    return actual


def _tables_from_inputs(report: dict[str, Any]) -> dict[str, CandidateIonTable]:
    """Reconstruct only CC/AA arrays; no transport or preimage function is called."""
    archive = load_astar_archive(_inside_repo(report["archive_path"]))
    reference = {z: tuple(r for r in archive.material(z).rows if .04 <= r.energy_mev_total <= 1000.)
                 for z in (13, 14)}
    energies = tuple(r.energy_mev_total for r in reference[13])
    native = _native_controls(
        _inside_repo(report["native_controls_path"]), report["native_controls_sha256"],
        archive.archive_sha256, energies,
    )
    if (native["archive_member_sha256"] != dict(archive.member_sha256)
            or report["archive_member_sha256"] != dict(archive.member_sha256)):
        raise ValueError("ASTAR member identities changed.")
    grid = tuple(e / 4.0 for e in energies)
    rows = {z: tuple(r for r in native["rows"] if r["material_z"] == z) for z in (13, 14)}
    for z in (13, 14):
        if tuple(r["energy_mev_per_nucleon"] for r in rows[z]) != grid:
            raise ValueError("Native control energy coordinates differ from ASTAR whole-energy/4.")
    tables = {
        "CC": CandidateIonTable(2, 4.0, impact.POLICY, report["table_metadata"]["CC"]["dataset_ref"],
            grid, tuple(r["range_g_cm2"] for r in rows[13]), grid,
            tuple(r["stopping_electronic_mev_cm2_g"] / 1000.0 for r in rows[14])),
        "AA": CandidateIonTable(2, 4.0, impact.POLICY, report["table_metadata"]["AA"]["dataset_ref"],
            grid, tuple(r.range_csda_g_cm2 for r in reference[13]), grid,
            tuple(r.stopping_electronic_mev_cm2_g / 1000.0 for r in reference[14])),
    }
    for label, table in tables.items():
        coordinates = {"al_energy": table.al_energy, "al_range": table.al_range,
                       "si_energy": table.si_energy, "si_electronic_let": table.si_electronic_let}
        digest = hashlib.sha256(json.dumps(coordinates, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if digest != report["table_metadata"][label]["coordinates_sha256"]:
            raise ValueError("Reconstructed numerical table does not match the impact report.")
        if (table.al_energy[0] != .01 or table.al_energy[-1] != 250.
                or report["let_edges"][0] > min(table.si_electronic_let)
                or report["let_edges"][-1] < max(table.si_electronic_let)):
            raise ValueError("The report's common/LET windows do not cover the witness interpretation.")
    return tables


def _expected_metrics(case: dict[str, Any], report: dict[str, Any], source: Spectrum1D) -> dict[str, float]:
    """Map exact retained metrics; positive shields make energy upper bounds agree."""
    if case["key"][2] <= 0:
        raise ValueError("The retained-real-case witness selects positive shields only.")
    if any(case["let_ledger"][k] != 0 for k in (
        "below_let_window_integral", "above_let_window_integral",
    )):
        raise ValueError("A cropped LET display cannot represent the all-Si witness integral.")
    edges = report["energy_edges"]
    if source.x[0] not in edges:
        raise ValueError("Exact below-source-minimum reference requires an aligned energy edge.")
    coverage = case["source_common_domain_coverage"]
    result = {
        "source_integral": case["source_integral"],
        "inside_al_source_integral": coverage["common_energy_integral"],
        "outside_al_source_integral": fsum((coverage["below_common_energy_integral"],
                                             coverage["above_common_energy_integral"])),
        "outside_al_domain_integral": case["primary_ledger"]["outside_al_domain_integral"],
        "unresolved_low_residual_integral": case["primary_ledger"]["unresolved_low_residual_integral"],
        "supported_exit_integral": case["primary_ledger"]["supported_exit_integral"],
        "outside_si_domain_integral": case["let_ledger"]["outside_si_domain_integral"],
        "mapped_let_integral": case["let_ledger"]["mapped_integral"],
        "exit_below_source_min_integral": fsum(case["energy_bin_integrals"][:edges.index(source.x[0])]),
        **{f"let_above_{t}": case["metrics"][f"let_ge_{t}_to10"] for t in LET_THRESHOLDS},
        **{f"energy_above_{t}": case["metrics"][f"energy_ge_{t}_to250"]
           for t in ENERGY_THRESHOLDS if t in report["energy_thresholds"]},
    }
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0
           for v in result.values()):
        raise ValueError("Reference metrics must be finite and nonnegative.")
    return result


def run_witness(
    report_path: Path, *, expected_sha256: str,
    sample_counts: tuple[int, ...] = (2_000_000, 4_000_000),
) -> dict[str, Any]:
    _validate_samples(sample_counts)
    report_path = _inside_repo(report_path)
    report = _read_report(report_path, expected_sha256)
    expected_pins = _input_pins(report, report_path, expected_sha256)
    before = _check_pins(expected_pins)
    tables = _tables_from_inputs(report)
    sources = _source_inputs([{
        "z": 2, "symbol": "He", "mass_parameter_u": 4.0, "source_membership": ["GCR", "SEP"],
    }])
    source_map = {(s.source.value, s.assessment_mode.value): s for s in sources
                  if s.assessment_mode is not None}
    if len(sources) != 4 or len(source_map) != 4:
        raise ValueError("Expected all four unchanged He4 source/mode definitions.")
    fingerprints = {(r["source"], r["mode"]): r for r in report["source_fingerprints"]}
    if len(fingerprints) != 4:
        raise ValueError("Impact report source identity inventory is incomplete or duplicated.")
    for key, source in source_map.items():
        digest = hashlib.sha256(json.dumps(asdict(source), sort_keys=True, ensure_ascii=False,
                                           allow_nan=False).encode()).hexdigest()
        if key not in fingerprints or digest != fingerprints[key]["source_sha256"]:
            raise ValueError("Reconstructed source differs from its retained impact-report fingerprint.")
    results = []
    for source_name, mode, thickness in _SELECTED:
        source = source_map[(source_name, mode)]
        for label in ("CC", "AA"):
            matches = [c for c in report["cases"][label] if c["key"] == [source_name, mode, thickness]]
            if len(matches) != 1:
                raise ValueError("Selected reference case is missing or duplicated.")
            reference = matches[0]
            if reference["source_sha256"] != fingerprints[(source_name, mode)]["source_sha256"]:
                raise ValueError("Selected case source identity differs from the source inventory.")
            expected = _expected_metrics(reference, report, source)
            comparisons = []
            for count in sample_counts:
                observed = _midpoint_metrics(source, tables[label], thickness, count)
                errors = {key: abs(observed[key] - value) for key, value in expected.items()}
                source_total = expected["source_integral"]
                common_total = expected["inside_al_source_integral"]
                comparisons.append({
                    "sample_count": count, "midpoint_metrics": observed, "absolute_errors": errors,
                    "relative_errors_to_reference": {
                        key: errors[key] / value if value > 0 else None for key, value in expected.items()
                    },
                    "max_absolute_error": max(errors.values()),
                    "max_error_over_full_source": max(errors.values()) / source_total
                        if source_total > 0 else None,
                    "max_error_over_common_source": max(errors.values()) / common_total
                        if common_total > 0 else None,
                })
            results.append({
                "key": [source_name, mode, thickness], "variant": label,
                "source_sha256": reference["source_sha256"],
                "table_coordinates_sha256": report["table_metadata"][label]["coordinates_sha256"],
                "source_domain_mev_per_nucleon": [source.x[0], source.x[-1]],
                "common_domain_mev_per_nucleon": [.01, 250.],
                "reference_metrics": expected, "metric_comparison_count": len(expected),
                "midpoint_comparisons": comparisons,
            })
            print(f"Independent midpoint check: {source_name}/{mode}/{thickness}/{label}", flush=True)
    localized_results = []
    for label in ("CC", "AA"):
        reference = next(c for c in report["cases"][label] if c["key"] == ["gcr", "peak", 10.])
        source = source_map[("gcr", "peak")]
        expected = {f"let_above_{t}": reference["metrics"][f"let_ge_{t}_to10"]
                    for t in _LOCAL_THRESHOLDS}
        comparisons = []
        for count in (100_000, 200_000):
            observed_local = _localized_high_let_metrics(source, tables[label], 10., count)
            errors = {key: abs(observed_local["threshold_integrals"][key] - value)
                      for key, value in expected.items()}
            comparisons.append({
                **observed_local, "absolute_errors": errors,
                "relative_errors_to_reference": {
                    key: errors[key] / value if value > 0 else None for key, value in expected.items()
                },
            })
        localized_results.append({
            "key": ["gcr", "peak", 10.], "variant": label,
            "source_sha256": reference["source_sha256"],
            "table_coordinates_sha256": report["table_metadata"][label]["coordinates_sha256"],
            "reference_metrics": expected, "midpoint_comparisons": comparisons,
            "scope": "complete_supported_Si_LET_tails_above_1.25_not_unknown_source_domains",
        })
        print(f"Localized high-LET midpoint check: gcr/peak/10.0/{label}", flush=True)
    after = _check_pins(expected_pins)
    if before != after:
        raise ValueError("Witness input or implementation changed during independent quadrature.")
    return {
        "schema_version": "radar.astar_helium_spectral_impact.midpoint_witness.v1",
        "report_path": str(report_path), "report_expected_sha256": expected_sha256,
        "input_and_implementation_sha256_before": before,
        "input_and_implementation_sha256_after": after,
        "sample_counts": sample_counts, "case_count": len(results), "cases": results,
        "localized_high_let_checks": localized_results,
        "localized_method": "uniform_incident_energy_midpoints_in_proven_enclosing_exit_band",
        "localized_scope": "supported_high_LET_only_not_full_source_or_unknown_input_residual_domains",
        "method": "uniform_log_incident_energy_midpoints_j(E)*E*dlogE_forward_NumPy_maps",
        "source_interpolation": "piecewise_linear_between_original_samples_no_source_normalization",
        "range_interpolation": "piecewise_linear_with_common_range_origin_removed_before_arithmetic",
        "let_interpolation": "piecewise_linear_electronic_Si_stopping_with_all_forward_branches",
        "domain_masks": "incident_Al_support_then_nonnegative_relative_residual_then_exit_Si_support",
        "mission_duration_multiplier_applied": False,
        "preimage_integrator_used_for_midpoint_estimate": False,
        "normative_export_allowed": False, "physical_accuracy_verified": False,
        "production_tables_replaced": False, "stage4_acceptance_matrix_completed": False,
        "limitations": [
            "Original source grids and scalar values are shared definitions; only quadrature is independent.",
            "Fixed real-source cases do not prove uniform quadrature convergence or physical model accuracy.",
            "Classification discontinuities and narrow LET preimages retain finite sampling error.",
            "All original high-energy source tails remain input coverage outside the Al table, not zero output proof.",
            "LET thresholds integrate supported Si output only; unknown incident/residual domains are not inferred.",
            "Selected shields are positive, so supported output energies do not exceed the common 250-MeV/n bound.",
            "Both full-source and common-source error normalizations are reported separately.",
            "Localized high-LET checks resolve only a proven enclosing supported band; global errors remain visible.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--samples", nargs="+", type=int, default=[2_000_000, 4_000_000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run_witness(args.report, expected_sha256=args.report_sha256,
                         sample_counts=tuple(args.samples))
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    print(json.dumps({"output_sha256": _sha(args.output), "case_count": result["case_count"],
                      "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
