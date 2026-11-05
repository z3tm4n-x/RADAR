"""Bounded local-time correlation counterfactuals, not a real solar history.

Compare a quiet baseline, independent uniform local-time ensembles and two
different correlated time definitions under identical artificial Sun phases.
The centered-dipole axis is borrowed explicitly, not accepted as an OST time
definition. Frozen inertial Sun vectors are not a UTC ephemeris.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from radar.core.project import OrbitConfig
from radar.geomagnetic.local_time import (
    LocalTimeHistory,
    build_centered_dipole_mlt_history,
    build_solar_local_time_history,
    orbit_samples_fingerprint,
)
from radar.geomagnetic.ost_orbit import (
    OST_EARTH_ROTATION_RATE_RAD_S,
    OstOrbitSample,
    sample_ost_orbit,
)
from radar.geomagnetic.ost_penetration import (
    OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
    build_ost_penetration_function_from_samples,
)
from radar.geomagnetic.rigidity import RigidityGrid


_ROOT = Path(__file__).resolve().parent.parent
_MAX_SAMPLES = 400000
_MAX_ORBIT_LOCAL_TIME_STATES = 5000000
_TIME_STATES_PER_NODE = 1 + 24 + 48 + 2 + 2
_TINY_POSITIVE_DAYS = 1e-9
_SUN_MODEL = "artificial_frozen_inertial_sun_declination_zero_v1"
_TIME_REFERENCE = "relative_perigee_t0_Greenwich_angle_zero_not_UTC"
_NORTH_LATITUDE_DEG = 79.0
_NORTH_LONGITUDE_DEG = 291.0
_FAMILIES = ("centered_dipole", "geographic_solar")
_PHASES = (0.0, 90.0)
_OPERATORS = ("quiet", "uniform24", "uniform48", "centered_dipole_phase0",
              "centered_dipole_phase90", "geographic_solar_phase0", "geographic_solar_phase90")
_SUN_REASON = (
    "Frozen inertial Sun with declination zero is an artificial phase counterfactual, "
    "not a solar ephemeris or reconstructed mission history."
)
_AXIS_REASON = (
    "The 79N/291E centered-dipole axis is borrowed from OST D.24 only; "
    "its use as the disturbance model local-time axis is not qualified."
)
_LIMITATIONS = (
    "This experiment compares candidate time definitions and an independent time ensemble.",
    "It does not decide which local-time definition is required by OST or related models.",
    "The two artificial frozen-inertial Sun phases are not real MLT or UTC solar histories.",
    "The centered-dipole 79N/291E axis is borrowed from OST D.24, not a qualified time convention.",
    "The initial orientation is relative Omega0 minus thetaG0, not an established launch epoch.",
    "Source products are fixed for 2026/five years; source-access correlations are not reconstructed.",
    "Only GCR/SEP proton, He and Fe are selected; source incomplete reasons are preserved.",
    "Source energy grids, omitted tails, shielding, LET, dose, SEE and trapped belts are not qualified.",
    "Geometric shadow and retained secular precession remain unqualified model assumptions.",
    "Density comparisons keep the requested duration; whole-cell rounding changes actual duration.",
    "Co-rotation checks coordinate invariance of local-time geometry, not physical cutoff invariance.",
    "No fitted parameters, production defaults or physical qualification are changed by this diagnostic.",
)


def _load_integration_helper() -> ModuleType:
    path = _ROOT / "scripts/validate_heo_mission_window.py"
    spec = importlib.util.spec_from_file_location("_radar_local_time_source_integration_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load local fixed-source integration helper.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_integration = _load_integration_helper()
_source = _integration._helper


@dataclass(frozen=True)
class TraceSpec:
    name: str
    orbit: OrbitConfig
    steps_per_orbit: int
    raan_deg: float = 0.0
    min_days: float = 16.0
    min_orbits: int = 10


def _specifications(case: str) -> tuple[TraceSpec, ...]:
    heo_orbit = _source._orbit(500.0, 39700.0, 63.4)
    heo = tuple(TraceSpec(f"heo_16d_n{steps}_raan{raan}", heo_orbit, steps, float(raan))
                for steps in (240, 480) for raan in (0, 90))
    smoke = (
        TraceSpec("leo_10orbit_smoke", _source._orbit(400.0, 400.0, 51.6), 120,
                  min_days=_TINY_POSITIVE_DAYS),
        TraceSpec("geo_10orbit_smoke", _source._orbit(35786.0, 35786.0, 0.0), 120,
                  min_days=_TINY_POSITIVE_DAYS),
    )
    cases = {"all": (*heo, *smoke), "heo": heo, "smoke": smoke}
    if case not in cases:
        raise ValueError(f"Unknown local-time diagnostic case: {case}")
    return cases[case]


def _preflight(specs: tuple[TraceSpec, ...], max_states: int) -> dict[str, Any]:
    _integration._positive_integer(max_states, "Local-time state limit")
    rows = []
    total = 0
    for spec in specs:
        _integration._positive_integer(spec.steps_per_orbit, "Steps per orbit")
        _integration._positive_integer(spec.min_orbits, "Minimum orbits")
        if spec.steps_per_orbit < 4:
            raise ValueError("Steps per orbit must be at least four.")
        if isinstance(spec.min_days, bool) or not math.isfinite(spec.min_days) or spec.min_days <= 0.0:
            raise ValueError("Minimum days must be positive and finite.")
        if isinstance(spec.raan_deg, bool) or not math.isfinite(spec.raan_deg):
            raise ValueError("Initial relative node orientation must be finite.")
        step = _integration._orbital_period(spec.orbit) / spec.steps_per_orbit
        count = max(spec.min_orbits * spec.steps_per_orbit,
                    math.ceil(spec.min_days * 86400.0 / step))
        if count > _MAX_SAMPLES:
            raise ValueError("Local-time diagnostic sample budget exceeded; no coarsening allowed.")
        states = count * _TIME_STATES_PER_NODE
        total += states
        rows.append({"name": spec.name, "sample_count": count, "step_s": step,
                     "operator_count": len(_OPERATORS), "orbit_local_time_states": states})
    if total > max_states:
        raise ValueError(f"Local-time diagnostic state budget exceeded: {total} > {max_states}")
    return {"trace_count": len(specs), "operator_count": len(specs) * len(_OPERATORS),
            "total_orbit_local_time_states": total, "orbit_local_time_state_limit": max_states,
            "states_per_orbit_node": _TIME_STATES_PER_NODE, "max_samples_per_trace": _MAX_SAMPLES,
            "silent_coarsening": False, "variants": rows}


def _north_axis() -> tuple[float, float, float]:
    latitude, longitude = math.radians(_NORTH_LATITUDE_DEG), math.radians(_NORTH_LONGITUDE_DEG)
    return (math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude), math.sin(latitude))


def _sun_vectors(
    samples: tuple[OstOrbitSample, ...], phase_deg: float,
) -> tuple[tuple[float, float, float], ...]:
    if isinstance(phase_deg, bool) or not math.isfinite(phase_deg):
        raise ValueError("Artificial Sun phase must be finite.")
    phase = math.radians(phase_deg)
    return tuple((math.cos(phase - OST_EARTH_ROTATION_RATE_RAD_S * sample.time_s),
                  math.sin(phase - OST_EARTH_ROTATION_RATE_RAD_S * sample.time_s), 0.0)
                 for sample in samples)


def _history(
    samples: tuple[OstOrbitSample, ...], family: str, phase_deg: float,
    *, suns: tuple[tuple[float, float, float], ...] | None = None,
    north_axis: tuple[float, float, float] | None = None,
) -> LocalTimeHistory:
    parameters = (
        ("artificial_initial_sun_inertial_phase_deg", format(phase_deg, ".17g")),
        ("artificial_sun_declination_deg", "0"),
        ("earth_rotation_rate_rad_s", format(OST_EARTH_ROTATION_RATE_RAD_S, ".17g")),
        ("sun_ecef_phase_rule", "phase0_minus_same_sampler_Earth_rotation_rate_times_global_t"),
        ("candidate_family", family),
    )
    directions = _sun_vectors(samples, phase_deg) if suns is None else suns
    if family == "centered_dipole":
        return build_centered_dipole_mlt_history(
            samples=samples, sun_directions_ecef=directions,
            north_axis_ecef=_north_axis() if north_axis is None else north_axis,
            sun_model=_SUN_MODEL, time_reference=_TIME_REFERENCE,
            parameters=(*parameters, ("axis_provenance", "borrowed_OST_D24_not_qualified_for_MLT")),
            incomplete_reasons=(_SUN_REASON, _AXIS_REASON),
        )
    if family == "geographic_solar":
        if north_axis is not None:
            raise ValueError("Geographic solar hour angle does not accept a magnetic north axis.")
        return build_solar_local_time_history(
            samples=samples, sun_directions_ecef=directions,
            sun_model=_SUN_MODEL, time_reference=_TIME_REFERENCE,
            parameters=parameters, incomplete_reasons=(_SUN_REASON,),
        )
    raise ValueError(f"Unknown local-time candidate family: {family}")


def _histogram(values: tuple[float, ...]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) or not 0.0 <= value < 24.0 for value in values):
        raise ValueError("Local-time histogram needs nonempty finite values within [0,24).")
    counts = [0] * 24
    for value in values:
        counts[math.floor(value)] += 1
    sine = math.fsum(math.sin(value * math.pi / 12.0) for value in values) / len(values)
    cosine = math.fsum(math.cos(value * math.pi / 12.0) for value in values) / len(values)
    resultant = math.hypot(sine, cosine)
    mean = ((math.atan2(sine, cosine) * 12.0 / math.pi) % 24.0
            if resultant > 1e-12 else None)
    return {"bin_edges_hours": tuple(float(value) for value in range(25)),
            "counts": tuple(counts), "fractions": tuple(count / len(values) for count in counts),
            "sample_count": len(values), "weighting": "equal_complete_temporal_cells",
            "circular_resultant_length": resultant, "circular_mean_hours": mean,
            "circular_mean_status": "defined" if mean is not None else "unresolved_near_zero_resultant",
            "minimum_hours": min(values), "maximum_hours": max(values),
            "occupied_bins": sum(count > 0 for count in counts)}


def _rotate_z(vector: tuple[float, float, float], angle_rad: float) -> tuple[float, float, float]:
    x, y, z = vector
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return cosine * x - sine * y, sine * x + cosine * y, z


def _co_rotation_sanity(
    samples: tuple[OstOrbitSample, ...], family: str, phase_deg: float,
) -> dict[str, Any]:
    # This is only a change of coordinate origin. Do not compare cutoff fields,
    # whose geographic tables have not been transformed with this frame.
    indices = tuple(sorted({index * (len(samples) - 1) // 16 for index in range(17)}))
    subset = tuple(samples[index] for index in indices)
    angle_deg = 67.0
    angle_rad = math.radians(angle_deg)
    suns = _sun_vectors(subset, phase_deg)
    original = _history(subset, family, phase_deg, suns=suns)
    rotated_samples = tuple(replace(sample, longitude_deg=(sample.longitude_deg + angle_deg) % 360.0)
                            for sample in subset)
    rotated_suns = tuple(_rotate_z(sun, angle_rad) for sun in suns)
    rotated = _history(
        rotated_samples, family, phase_deg, suns=rotated_suns,
        north_axis=_rotate_z(_north_axis(), angle_rad) if family == "centered_dipole" else None,
    )
    difference = max(abs((right - left + 12.0) % 24.0 - 12.0)
                     for left, right in zip(original.values_hours, rotated.values_hours, strict=True))
    tolerance = 1e-10
    if difference > tolerance:
        raise ArithmeticError("Local-time geometry failed its coordinate-origin co-rotation check.")
    return {"scope": "coordinate_geometry_only_not_cutoff_or_physical_acceptance",
            "sample_indices": indices, "rotation_about_geographic_north_deg": angle_deg,
            "max_circular_hour_difference": difference, "numerical_tolerance_hours": tolerance,
            "coordinate_invariance_check_passed": True,
            "original_subset_fingerprint": original.sample_fingerprint,
            "rotated_subset_fingerprint": rotated.sample_fingerprint}


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _variant(
    spec: TraceSpec, grid: RigidityGrid, prepared: tuple[Any, ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    trace = sample_ost_orbit(
        spec.orbit, raan_deg=spec.raan_deg, min_days=spec.min_days, min_orbits=spec.min_orbits,
        steps_per_orbit=spec.steps_per_orbit, max_samples=_MAX_SAMPLES,
    )
    expected = _preflight((spec,), _MAX_ORBIT_LOCAL_TIME_STATES)["variants"][0]
    if trace.step_s != expected["step_s"] or len(trace.samples) != expected["sample_count"]:
        raise ValueError("Sampler step or count differs from the preflight; no silent coarsening.")
    histories = {f"{family}_phase{phase:g}": _history(trace.samples, family, phase)
                 for family in _FAMILIES for phase in _PHASES}
    history_reports = {}
    for name, bound_history in histories.items():
        bound_history.validate_samples(trace.samples)
        payload = asdict(bound_history)
        history_reports[name] = {"payload": payload, "sha256": _canonical_sha(payload),
                                 "histogram": _histogram(bound_history.values_hours)}
    operators = []
    for name in _OPERATORS:
        history = histories.get(name)
        operator = build_ost_penetration_function_from_samples(
            samples=trace.samples, rigidity_grid=grid, kp=3, apply_disturbance=name != "quiet",
            mlt_sample_count=48 if name == "uniform48" else 24,
            local_time_history=history,
        )
        operators.append({
            "name": f"{spec.name}:{name}", "time_case": name,
            "trace_name": spec.name, "step_s": trace.step_s,
            "requested_window_s": trace.requested_window_s, "actual_window_s": trace.window_s,
            "operator_version": operator.model_version, "operator_parameters": operator.parameters,
            "operator_incomplete_reasons": operator.incomplete_reasons,
            "local_time_history_model": history.model if history is not None else None,
            "local_time_history_sha256": history_reports[name]["sha256"] if history else None,
            "penetration": operator.values,
            "source_weighted_integrals": _integration._integrals(operator.values, prepared),
            "normative_release_eligible": False,
        })
    return {
        "name": spec.name, "requested": asdict(spec), "sample_count": len(trace.samples),
        "sample_fingerprint": orbit_samples_fingerprint(trace.samples),
        "step_s": trace.step_s, "orbital_period_s": trace.orbital_period_s,
        "requested_window_s": trace.requested_window_s, "actual_window_s": trace.window_s,
        "sampling_policy": trace.sampling_policy,
        "first_sample": asdict(trace.samples[0]), "last_sample": asdict(trace.samples[-1]),
        "histories": history_reports, "operators": operators,
        "co_rotation_sanity": {f"{family}_phase{phase:g}": _co_rotation_sanity(
            trace.samples, family, phase,
        ) for family in _FAMILIES for phase in _PHASES},
        "elapsed_s": time.perf_counter() - started, "normative_release_eligible": False,
    }


def _pair(left: dict[str, Any], right: dict[str, Any], axis: str) -> dict[str, Any]:
    pair: dict[str, Any] = _integration._comparison(
        left, right, left["penetration"], right["penetration"], axis,
    )
    pair.update({
        "left_local_time_model": left["local_time_history_model"],
        "right_local_time_model": right["local_time_history_model"],
        "left_requested_window_s": left["requested_window_s"],
        "right_requested_window_s": right["requested_window_s"],
        "left_actual_window_s": left["actual_window_s"],
        "right_actual_window_s": right["actual_window_s"],
        "actual_window_change_s": right["actual_window_s"] - left["actual_window_s"],
        "actual_window_equal": left["actual_window_s"] == right["actual_window_s"],
        "density_comparison_includes_rounding_window_change": (
            axis == "orbital_density_at_fixed_requested_window_and_local_time_candidate"
            and left["actual_window_s"] != right["actual_window_s"]
        ),
        "candidate_selection_performed": False,
    })
    return pair


def _comparisons(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    by_trace = {variant["name"]: {item["time_case"]: item for item in variant["operators"]}
                for variant in variants}
    for rows in by_trace.values():
        comparisons.append(_pair(rows["uniform24"], rows["uniform48"],
                                 "independent_local_time_quadrature_24_to_48"))
        comparisons.append(_pair(rows["quiet"], rows["uniform48"],
                                 "quiet_vs_disturbed_independent_ensemble"))
        for family in _FAMILIES:
            for phase in _PHASES:
                comparisons.append(_pair(rows["uniform48"], rows[f"{family}_phase{phase:g}"],
                                         "independent_ensemble_vs_correlated_candidate"))
            comparisons.append(_pair(rows[f"{family}_phase0"], rows[f"{family}_phase90"],
                                     "artificial_frozen_sun_initial_phase_0_to_90"))
        for phase in _PHASES:
            comparisons.append(_pair(rows[f"geographic_solar_phase{phase:g}"],
                                     rows[f"centered_dipole_phase{phase:g}"],
                                     "candidate_time_definition_at_identical_artificial_sun_phase"))
    for raan in (0, 90):
        left, right = f"heo_16d_n240_raan{raan}", f"heo_16d_n480_raan{raan}"
        if left in by_trace and right in by_trace:
            comparisons.extend(_pair(by_trace[left][name], by_trace[right][name],
                                     "orbital_density_at_fixed_requested_window_and_local_time_candidate")
                               for name in _OPERATORS)
    for steps in (240, 480):
        left, right = f"heo_16d_n{steps}_raan0", f"heo_16d_n{steps}_raan90"
        if left in by_trace and right in by_trace:
            comparisons.extend(_pair(by_trace[left][name], by_trace[right][name],
                                     "initial_relative_node_orientation_at_fixed_local_time_candidate")
                               for name in _OPERATORS)
    return comparisons


def _model_hashes() -> dict[str, str]:
    hashes: dict[str, str] = _integration._model_hashes()
    own_path = Path(__file__).resolve()
    hashes[own_path.relative_to(_ROOT).as_posix()] = _source._sha(own_path)
    return hashes


def build_report(
    case: str = "all", *, max_states: int = _MAX_ORBIT_LOCAL_TIME_STATES,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    specs = _specifications(case)
    preflight = _preflight(specs, max_states)
    models_before, inputs_before = _model_hashes(), _source._input_hashes()
    products, source_scenario = _source._source_products()
    source_duration = _integration._validate_source_duration(products)
    mission = source_scenario["calculation_config"]["mission"]
    if (mission["launch_year"] != 2026 or mission["lifetime_years"] != 5
            or source_scenario["sep_exceedance_probability"] != 0.1):
        raise ValueError("Fixed sources must remain the declared 2026 five-year mission with SEP P=0.1.")
    source_payload = [asdict(product) for product in products]
    grid = _source._rigidity_grid(products)
    prepared = _integration._prepare_products(products, grid)
    variants = []
    for spec in specs:
        result = _variant(spec, grid, prepared)
        variants.append(result)
        if progress is not None:
            progress({"trace": spec.name, "completed_traces": len(variants),
                      "total_traces": len(specs), "elapsed_s": result["elapsed_s"]})
    models_after, inputs_after = _model_hashes(), _source._input_hashes()
    if models_before != models_after or inputs_before != inputs_after:
        raise ValueError("Source data or implementation changed during diagnostics; discard run.")
    return {
        "schema": "radar_local_time_correlation_diagnostic_v1", "case": case,
        "normative_release_eligible": False, "mission_radiation_history_qualified": False,
        "stage4_spectral_convergence_established": False, "candidate_selection_performed": False,
        "source_roster_complete": False, "parameters_fitted": False,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "production_operator_version": OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
        "preflight": preflight, "source_scenario": source_scenario,
        "source_mission_duration_s": source_duration,
        "source_products": source_payload, "source_products_sha256": _canonical_sha(source_payload),
        "model_hash_scope": "all_src_radar_python_and_both_imported_helpers_and_this_runner",
        "model_sha256_before": models_before, "model_sha256_after": models_after,
        "input_sha256_before": inputs_before, "input_sha256_after": inputs_after,
        "rigidity_gv": grid.values_gv, "variants": variants, "comparisons": _comparisons(variants),
        "conventions": {
            "kp": 3, "source_operator": "fixed_source_times_dimensionless_psi_no_extra_time_factor",
            "source_integral": "finite_energy_linear_trapezoid_existing_centers_no_tail",
            "source_time": "2026_five_year_model_products_fixed_once_not_window_fluences",
            "initial_orientation": "relative_Omega0_minus_thetaG0_not_UTC_launch_epoch",
            "sun_model": _SUN_MODEL, "sun_phases_deg": _PHASES,
            "sun_declination_deg": 0.0,
            "sun_ecef_rotation": "negative_same_sampler_Earth_rotation_rate_times_global_t",
            "time_reference": _TIME_REFERENCE,
            "centered_dipole_axis": {"latitude_deg": _NORTH_LATITUDE_DEG,
                                     "longitude_east_deg": _NORTH_LONGITUDE_DEG,
                                     "ecef_unit": _north_axis(),
                                     "provenance": "borrowed_OST_D24_not_qualified_for_MLT"},
            "solar_hour_angle_axis": "physical_geographic_north_0_0_1",
            "window_weights": "equal_complete_temporal_cells_no_partial_cell",
            "candidate_selection": "none_no_time_definition_claimed_normative",
        },
        "limitations": _LIMITATIONS,
    }


def write_report_exclusively(report: dict[str, Any], output: Path) -> None:
    _source.write_report_exclusively(report, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "heo", "smoke"), default="all")
    parser.add_argument("--max-states", type=int, default=_MAX_ORBIT_LOCAL_TIME_STATES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        print(json.dumps({"preflight": _preflight(_specifications(args.case), args.max_states)},
                         ensure_ascii=False, allow_nan=False), flush=True)
        report = build_report(args.case, max_states=args.max_states, progress=lambda item: print(
            json.dumps({"progress": item}, ensure_ascii=False, allow_nan=False), flush=True,
        ))
        write_report_exclusively(report, args.output)
    except (OSError, ValueError, ArithmeticError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
