"""Bounded finite-mission access diagnostic; not a mission radiation history.

Nonoverlapping temporal blocks preserve the sampler's global phase. Actual
five-year GCR/SEP source products are held fixed while access is averaged over
the declared duration. The last cell is clipped, not rounded past the mission.
Neither block results nor their changes are a physical source time series.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from radar.core.products import SpectrumProduct
from radar.core.project import OrbitConfig
from radar.core.types import AssessmentMode, RadiationProductKind, RadiationSource
from radar.gcr.model import GCR_MODEL_SECONDS_PER_YEAR
from radar.geomagnetic.ost import EARTH_RADIUS_KM
from radar.geomagnetic.ost_orbit import (
    OST_GRAVITATIONAL_PARAMETER_KM3_S2,
    OST_ORBIT_SAMPLING_MODEL_VERSION,
    OST_ORBIT_TIME_ORIGIN_MODEL_VERSION,
    TWO_PI,
    sample_ost_orbit,
)
from radar.geomagnetic.ost_penetration import (
    OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
    build_ost_penetration_function_from_samples,
)
from radar.geomagnetic.rigidity import RigidityGrid


_ROOT = Path(__file__).resolve().parent.parent
_MISSION_DURATION_S = 5.0 * GCR_MODEL_SECONDS_PER_YEAR
_MAX_SAMPLES_PER_BLOCK = 400000
_MAX_ORBIT_MLT_STATES = 50000000
_TINY_POSITIVE_DAYS = 1e-9
_METRIC_SEMANTICS = (
    "fixed_full_mission_source_product_times_mean_access_not_actual_block_fluence_or_history"
)
_LIMITATIONS = (
    "Full temporal coverage qualifies only coverage of the retained kinematic model.",
    "Fixed mission source products are not correlated with time-dependent access.",
    "Block metrics use the same full-mission sources; they are not block fluences or a timeline.",
    "Only GCR/SEP proton, He and Fe are retained; the diagnostic ion roster is incomplete.",
    "Source incomplete reasons remain applicable to every weighted metric.",
    "Geometric shadow, uniform MLT and printed secular precession remain unqualified.",
    "The initial orientation is Omega0 minus thetaG0, with implicit Greenwich angle zero.",
    "The relative time origin is not a reconstructed UTC launch epoch.",
    "Fixed energy grids and omitted spectral tails are not qualified by this diagnostic.",
    "No trapped-belt, shielding, LET, dose or SEE acceptance matrix is calculated.",
    "No density-convergence or normative-release acceptance follows from temporal coverage.",
)


def _load_source_helper() -> ModuleType:
    path = _ROOT / "scripts/validate_heo_orbit_sampling.py"
    spec = importlib.util.spec_from_file_location("_radar_heo_mission_source_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned local source diagnostic helper.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_helper = _load_source_helper()


@dataclass(frozen=True)
class MissionSpec:
    name: str
    orbit: OrbitConfig
    duration_s: float
    steps_per_orbit: int
    raan_deg: float
    mlt_sample_count: int = 24


@dataclass(frozen=True)
class BlockPlan:
    index: int
    start_sample_index: int
    used_cell_count: int
    generated_orbits: int
    generated_cell_count: int
    start_time_s: float
    end_time_s: float
    represented_duration_s: float
    duration_fraction: float
    last_cell_duration_s: float
    last_cell_fraction: float


@dataclass(frozen=True)
class PreparedProduct:
    key: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    rigidity_indices: tuple[int, ...]
    metadata: dict[str, Any]


def _positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")


def _orbital_period(orbit: OrbitConfig) -> float:
    # Keep the exact production operation order: an equivalent reciprocal sqrt
    # can differ by an ULP and change ceil decisions at a cell boundary.
    apogee_radius = EARTH_RADIUS_KM + orbit.apogee_altitude_km
    perigee_radius = EARTH_RADIUS_KM + orbit.perigee_altitude_km
    semi_major_axis = 0.5 * (apogee_radius + perigee_radius)
    mean_motion = math.sqrt(OST_GRAVITATIONAL_PARAMETER_KM3_S2 / semi_major_axis**3)
    return TWO_PI / mean_motion


def _cell_count(duration_s: float, step_s: float) -> int:
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0.0
           for value in (duration_s, step_s)):
        raise ValueError("Duration and step must be positive finite numbers.")
    quotient = duration_s / step_s
    if not math.isfinite(quotient) or math.ulp(duration_s) > step_s:
        raise ValueError("Temporal resolution cannot represent the requested cell boundaries.")
    count = max(1, math.ceil(quotient))
    # Division may round nextafter(edge,+inf) back onto the integer boundary.
    while count * step_s < duration_s:
        count += 1
    while count > 1 and (count - 1) * step_s >= duration_s:
        count -= 1
    return count


def _block_plan(
    duration_s: float, step_s: float, steps_per_orbit: int, block_orbits: int,
) -> tuple[BlockPlan, ...]:
    _positive_integer(steps_per_orbit, "Steps per orbit")
    _positive_integer(block_orbits, "Orbits per block")
    capacity = block_orbits * steps_per_orbit
    if capacity > _MAX_SAMPLES_PER_BLOCK:
        raise ValueError("Per-block sample budget exceeded; temporal resolution is not reduced.")
    count = _cell_count(duration_s, step_s)
    blocks = []
    for index, start in enumerate(range(0, count, capacity)):
        used = min(capacity, count - start)
        generated_orbits = (used + steps_per_orbit - 1) // steps_per_orbit
        end = min(duration_s, (start + used) * step_s)
        begin = start * step_s
        last_begin = (start + used - 1) * step_s
        last_full_end = (start + used) * step_s
        last_duration = end - last_begin
        if not 0.0 < last_duration <= last_full_end - last_begin:
            raise ValueError("Last temporal cell is empty or exceeds its full grid interval.")
        blocks.append(BlockPlan(
            index=index, start_sample_index=start, used_cell_count=used,
            generated_orbits=generated_orbits,
            generated_cell_count=generated_orbits * steps_per_orbit,
            start_time_s=begin, end_time_s=end, represented_duration_s=end - begin,
            duration_fraction=(end - begin) / duration_s,
            last_cell_duration_s=last_duration,
            last_cell_fraction=last_duration / (last_full_end - last_begin),
        ))
    if blocks[0].start_time_s != 0.0 or blocks[-1].end_time_s != duration_s:
        raise ValueError("Block plan does not cover the exact finite duration.")
    if any(left.end_time_s != right.start_time_s
           or left.start_sample_index + left.used_cell_count != right.start_sample_index
           for left, right in zip(blocks, blocks[1:])):
        raise ValueError("Temporal blocks overlap or contain a gap.")
    represented = math.fsum(block.represented_duration_s for block in blocks)
    if abs(represented - duration_s) > 4.0 * math.ulp(duration_s):
        raise ValueError("Represented block durations do not close to the requested duration.")
    return tuple(blocks)


def _cell_weights(block: BlockPlan, step_s: float) -> tuple[float, ...]:
    weights = tuple(
        min(block.end_time_s, (global_index + 1) * step_s) - global_index * step_s
        for global_index in range(block.start_sample_index,
                                  block.start_sample_index + block.used_cell_count)
    )
    if any(weight <= 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("Every represented temporal cell must have positive finite duration.")
    if abs(math.fsum(weights) - block.represented_duration_s) > (
        4.0 * math.ulp(block.represented_duration_s)
    ):
        raise ValueError("Cell durations do not close to their represented block duration.")
    return weights


def _specifications(case: str) -> tuple[MissionSpec, ...]:
    if case not in ("smoke", "n60", "n120", "n240"):
        raise ValueError(f"Unknown finite-mission diagnostic case: {case}")
    orbit = _helper._orbit(500.0, 39700.0, 63.4)
    steps = 16 if case == "smoke" else int(case[1:])
    duration = 2.3 * _orbital_period(orbit) if case == "smoke" else _MISSION_DURATION_S
    return tuple(MissionSpec(f"heo_{case}_raan{raan}", orbit, duration, steps, float(raan))
                 for raan in (0, 90))


def _preflight(
    specs: tuple[MissionSpec, ...], block_orbits: int, max_states: int,
) -> dict[str, Any]:
    _positive_integer(max_states, "Orbit-MLT state limit")
    _positive_integer(block_orbits, "Orbits per block")
    rows: list[dict[str, Any]] = []
    total_states = 0
    # Check total work before allocating plans, loading source files or sampling.
    for spec in specs:
        _positive_integer(spec.steps_per_orbit, "Steps per orbit")
        if spec.steps_per_orbit < 4:
            raise ValueError("Steps per orbit must be at least four, as required by the sampler.")
        _positive_integer(spec.mlt_sample_count, "MLT sample count")
        step = _orbital_period(spec.orbit) / spec.steps_per_orbit
        count = _cell_count(spec.duration_s, step)
        total_states += count * spec.mlt_sample_count
        rows.append({"name": spec.name, "step_s": step, "used_cells": count,
                     "orbit_mlt_states": count * spec.mlt_sample_count})
    if total_states > max_states:
        raise ValueError(f"Mission diagnostic state budget exceeded: {total_states} > {max_states}")
    for spec, row in zip(specs, rows, strict=True):
        plan = _block_plan(spec.duration_s, row["step_s"], spec.steps_per_orbit, block_orbits)
        row.update({"block_count": len(plan),
                    "generated_cells": sum(block.generated_cell_count for block in plan),
                    "max_generated_cells_per_block": max(b.generated_cell_count for b in plan)})
    return {"variant_count": len(specs), "total_orbit_mlt_states": total_states,
            "orbit_mlt_state_limit": max_states, "max_samples_per_block": _MAX_SAMPLES_PER_BLOCK,
            "block_orbits": block_orbits, "silent_coarsening": False, "variants": rows}


def _validate_source_duration(products: tuple[SpectrumProduct, ...]) -> float:
    if len(products) != 12 or len({_helper._product_key(p) for p in products}) != 12:
        raise ValueError("Exactly twelve unique fixed source products are required.")
    cumulative_count = 0
    for product in products:
        product.validate_calculation_contract()
        spectrum = product.spectrum
        if product.kind is RadiationProductKind.MISSION_FLUENCE:
            cumulative_count += 1
            if (spectrum.mission_duration_s != _MISSION_DURATION_S
                    or spectrum.assessment_mode is not AssessmentMode.CUMULATIVE):
                raise ValueError("All cumulative source mission durations must match the plan.")
        elif (product.kind not in (RadiationProductKind.MAXIMUM_FLUX,
                                   RadiationProductKind.PEAK_FLUX)
              or spectrum.mission_duration_s is not None
              or spectrum.assessment_mode is not AssessmentMode.PEAK):
            raise ValueError("Peak source products must have no mission-duration multiplier.")
    if cumulative_count != 6:
        raise ValueError("Both source modes are required for every selected source/species pair.")
    return _MISSION_DURATION_S


def _prepare_products(
    products: tuple[SpectrumProduct, ...], grid: RigidityGrid,
) -> tuple[PreparedProduct, ...]:
    index_by_rigidity = {value: index for index, value in enumerate(grid.values_gv)}
    prepared = []
    for product in products:
        spectrum = product.spectrum
        # Validate each energy grid and cache exact rigidity indices once per run.
        integral = _helper._finite_trapezoid(spectrum.x, spectrum.y)
        indices = tuple(index_by_rigidity[r] for r in _helper._source_rigidities(product))
        prepared.append(PreparedProduct(
            key=_helper._product_key(product), x=spectrum.x, y=spectrum.y,
            rigidity_indices=indices, metadata={
                "unit": ("cm^-2" if product.kind is RadiationProductKind.MISSION_FLUENCE
                         else "cm^-2 s^-1"),
                "source_quantity": spectrum.quantity.value,
                "source_differential_unit": spectrum.y_unit.value,
                "source_angular_basis": spectrum.angular_basis.value
                if spectrum.angular_basis else None,
                "assessment_mode": spectrum.assessment_mode.value
                if spectrum.assessment_mode else None,
                "temporal_basis": spectrum.temporal_basis.value if spectrum.temporal_basis else None,
                "mission_duration_s": spectrum.mission_duration_s,
                "time_multiplier": "none",
                "energy_lower_bound": spectrum.x[0], "energy_upper_bound": spectrum.x[-1],
                "energy_unit": spectrum.x_unit.value,
                "source_completeness": spectrum.completeness.value
                if spectrum.completeness else None,
                "source_incomplete_reasons": spectrum.incomplete_reasons,
                "source_integral_same_grid": integral,
                "rigidity_mass_to_charge_used": _helper._mass_to_charge(product),
                "rigidity_mass_to_charge_convention": (
                    "SEP_B9_A_over_Q_production_helper" if spectrum.source is RadiationSource.SEP
                    else "GCR_parameter_A_over_Z_production_helper"
                ),
                "metric_semantics": _METRIC_SEMANTICS,
            },
        ))
    return tuple(prepared)


def _integrals(values: tuple[float, ...], products: tuple[PreparedProduct, ...]) -> dict[str, Any]:
    return {product.key: {
        **product.metadata,
        "value": _helper._finite_trapezoid(product.x, tuple(
            source * values[index]
            for source, index in zip(product.y, product.rigidity_indices, strict=True)
        )),
    } for product in products}


class _DurationAccumulator:
    """Neumaier accumulation of duration*psi, divided only after all blocks."""

    def __init__(self, size: int) -> None:
        _positive_integer(size, "Accumulator size")
        self.total = [0.0] * size
        self.correction = [0.0] * size
        self.durations: list[float] = []

    def add(self, values: tuple[float, ...], duration_s: float) -> None:
        if len(values) != len(self.total) or not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("Invalid block shape or represented duration for accumulation.")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Block access must be finite and within zero to one.")
        self.durations.append(duration_s)
        for index, value in enumerate(values):
            term = value * duration_s
            previous = self.total[index]
            current = previous + term
            self.correction[index] += ((previous - current) + term if abs(previous) >= abs(term)
                                       else (term - current) + previous)
            self.total[index] = current

    def values(self, duration_s: float) -> tuple[float, ...]:
        if isinstance(duration_s, bool) or not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("Aggregate duration must be positive and finite.")
        if abs(math.fsum(self.durations) - duration_s) > 4.0 * math.ulp(duration_s):
            raise ValueError("Aggregate block durations do not close to the declared duration.")
        values = tuple((total + correction) / duration_s
                       for total, correction in zip(self.total, self.correction, strict=True))
        tolerance = 8.0 * math.ulp(1.0)
        if any(not math.isfinite(value) or not -tolerance <= value <= 1.0 + tolerance
               for value in values):
            raise ArithmeticError("Accumulated access escaped its physical bounds.")
        if any(right < left - tolerance for left, right in zip(values, values[1:])):
            raise ArithmeticError("Accumulated access is not nondecreasing with rigidity.")
        # Match the production eight-ULP bound policy; do not smooth the curve
        # or repair a physical excursion/monotonicity failure.
        return tuple(min(1.0, max(0.0, value)) for value in values)


def _comparison(
    left: dict[str, Any], right: dict[str, Any], left_values: tuple[float, ...],
    right_values: tuple[float, ...], axis: str,
) -> dict[str, Any]:
    left_metrics, right_metrics = left["source_weighted_integrals"], right["source_weighted_integrals"]
    if left_metrics.keys() != right_metrics.keys():
        raise ValueError("Comparison source identities differ.")
    return {
        "axis": axis, "left": left["name"], "right": right["name"],
        "max_absolute_psi_change": max(abs(b - a) for a, b in
                                       zip(left_values, right_values, strict=True)),
        "psi_change_is_dimensionless_not_relative_flux_error": True,
        "source_integral_changes": {key: _helper._relative_difference(
            right_metrics[key]["value"], left_metrics[key]["value"],
        ) for key in left_metrics},
        "physical_source_time_series": False,
        "normative_acceptance_established": False,
    }


def _variant(
    spec: MissionSpec, block_orbits: int, grid: RigidityGrid,
    prepared: tuple[PreparedProduct, ...], progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    period = _orbital_period(spec.orbit)
    step = period / spec.steps_per_orbit
    plan = _block_plan(spec.duration_s, step, spec.steps_per_orbit, block_orbits)
    accumulator = _DurationAccumulator(len(grid.values_gv))
    records: list[dict[str, Any]] = []
    comparisons = []
    previous_values: tuple[float, ...] | None = None
    all_reasons: set[str] = set()
    for block in plan:
        trace = sample_ost_orbit(
            spec.orbit, raan_deg=spec.raan_deg, min_days=_TINY_POSITIVE_DAYS,
            min_orbits=block.generated_orbits, start_sample_index=block.start_sample_index,
            steps_per_orbit=spec.steps_per_orbit, max_samples=_MAX_SAMPLES_PER_BLOCK,
        )
        if (trace.step_s != step or trace.orbital_period_s != period
                or len(trace.samples) != block.generated_cell_count
                or trace.start_time_s != block.start_time_s):
            raise ValueError("Sampler step, global time or cell count differs from the plan.")
        samples = trace.samples[:block.used_cell_count]
        weights = _cell_weights(block, step)
        operator = build_ost_penetration_function_from_samples(
            samples=samples, rigidity_grid=grid, kp=3, apply_disturbance=True,
            mlt_sample_count=spec.mlt_sample_count, sample_weights=weights,
        )
        accumulator.add(operator.values, block.represented_duration_s)
        all_reasons.update(operator.incomplete_reasons)
        record: dict[str, Any] = {
            "name": f"{spec.name}_block{block.index:04d}", **asdict(block),
            "step_s": trace.step_s, "first_node_time_s": samples[0].time_s,
            "last_node_time_s": samples[-1].time_s,
            "generated_end_time_s": trace.end_time_s,
            "discarded_generated_cells": len(trace.samples) - len(samples),
            "cell_duration_sum_s": math.fsum(weights),
            "cell_duration_closure_error_s": math.fsum(weights) - block.represented_duration_s,
            "operator_version": operator.model_version, "operator_parameters": operator.parameters,
            "operator_incomplete_reasons": operator.incomplete_reasons,
            "source_weighted_integrals": _integrals(operator.values, prepared),
            "normative_release_eligible": False,
        }
        if previous_values is not None:
            comparison = _comparison(records[-1], record, previous_values, operator.values,
                                     "consecutive_nonoverlap_blocks_fixed_source_phase_coverage")
            comparison.update({
                "left_start_time_s": records[-1]["start_time_s"],
                "boundary_time_s": block.start_time_s,
                "right_end_time_s": block.end_time_s,
                "contiguous_without_overlap": records[-1]["end_time_s"] == block.start_time_s,
                "left_duration_s": records[-1]["represented_duration_s"],
                "right_duration_s": block.represented_duration_s,
            })
            comparisons.append(comparison)
        records.append(record)
        previous_values = operator.values
        if progress is not None and (block.index == 0 or (block.index + 1) % 10 == 0
                                     or block.index + 1 == len(plan)):
            progress({"variant": spec.name, "completed_blocks": block.index + 1,
                      "total_blocks": len(plan), "covered_duration_s": block.end_time_s,
                      "elapsed_s": time.perf_counter() - started})
    values = accumulator.values(spec.duration_s)
    duration_sum = math.fsum(block.represented_duration_s for block in plan)
    return {
        "name": spec.name, "requested": asdict(spec), "orbital_period_s": period, "step_s": step,
        "sampling_policy": OST_ORBIT_SAMPLING_MODEL_VERSION,
        "time_origin_policy": OST_ORBIT_TIME_ORIGIN_MODEL_VERSION,
        "used_cell_count": sum(block.used_cell_count for block in plan),
        "generated_cell_count": sum(block.generated_cell_count for block in plan),
        "block_count": len(plan), "represented_duration_s": duration_sum,
        "duration_closure_error_s": duration_sum - spec.duration_s,
        "duration_fraction_sum": math.fsum(block.duration_fraction for block in plan),
        "coverage_start_time_s": plan[0].start_time_s, "coverage_end_time_s": plan[-1].end_time_s,
        "exact_configured_mission_temporal_coverage": spec.duration_s == _MISSION_DURATION_S,
        "contiguous_without_overlap": True, "global_phases_preserved": True,
        "penetration": values, "source_weighted_integrals": _integrals(values, prepared),
        "operator_incomplete_reasons": tuple(sorted(all_reasons)),
        "blocks": records, "consecutive_block_comparisons": comparisons,
        "elapsed_s": time.perf_counter() - started,
        "mission_radiation_history_qualified": False, "normative_release_eligible": False,
    }


def _model_hashes() -> dict[str, str]:
    hashes: dict[str, str] = _helper._model_hashes()
    own_path = Path(__file__).resolve()
    hashes[own_path.relative_to(_ROOT).as_posix()] = _helper._sha(own_path)
    return hashes


def build_report(
    case: str = "n60", *, block_orbits: int | None = None,
    max_states: int = _MAX_ORBIT_MLT_STATES,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    specs = _specifications(case)
    actual_block_orbits = (2 if case == "smoke" else 32) if block_orbits is None else block_orbits
    preflight = _preflight(specs, actual_block_orbits, max_states)
    models_before, inputs_before = _model_hashes(), _helper._input_hashes()
    products, source_scenario = _helper._source_products()
    source_duration = _validate_source_duration(products)
    mission = source_scenario["calculation_config"]["mission"]
    if mission["launch_year"] != 2026 or mission["lifetime_years"] != 5:
        raise ValueError("Source scenario must remain the declared 2026 five-year mission.")
    source_payload = [asdict(product) for product in products]
    source_hash = hashlib.sha256(json.dumps(
        source_payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    grid = _helper._rigidity_grid(products)
    prepared = _prepare_products(products, grid)
    variants = [_variant(spec, actual_block_orbits, grid, prepared, progress) for spec in specs]
    models_after, inputs_after = _model_hashes(), _helper._input_hashes()
    if models_before != models_after or inputs_before != inputs_after:
        raise ValueError("Source data or implementation changed during diagnostics; discard run.")
    return {
        "schema": "radar_heo_finite_mission_window_diagnostic_v1", "case": case,
        "normative_release_eligible": False, "mission_radiation_history_qualified": False,
        "stage4_spectral_convergence_established": False, "source_roster_complete": False,
        "parameters_fitted": False, "source_mission_duration_s": source_duration,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "production_operator_version": OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
        "preflight": preflight, "source_scenario": source_scenario,
        "source_products": source_payload, "source_products_sha256": source_hash,
        "model_hash_scope": "all_src_radar_python_files_source_helper_and_this_runner",
        "model_sha256_before": models_before, "model_sha256_after": models_after,
        "input_sha256_before": inputs_before, "input_sha256_after": inputs_after,
        "rigidity_gv": grid.values_gv, "variants": variants,
        "comparisons": [_comparison(
            variants[0], variants[1], variants[0]["penetration"], variants[1]["penetration"],
            "initial_relative_node_orientation_at_fixed_finite_duration",
        )],
        "conventions": {
            "kp": 3, "source_operator": "fixed_source_times_dimensionless_mean_psi_no_extra_time",
            "time": "model_year_365_25_days_not_exact_UTC_calendar_interval",
            "time_origin": "relative_perigee_passage_Greenwich_angle_zero_not_UTC_launch_epoch",
            "initial_orientation": "raan_deg_represents_Omega0_minus_thetaG0",
            "grid": "union_401_log_rigidity_nodes_and_exact_source_rigidities",
            "source_integral": "finite_energy_linear_trapezoid_between_existing_centers_no_tail",
            "temporal_rule": "left_endpoints_global_integer_grid_last_cell_clipped_to_duration",
            "block_generation": "whole_orbits_trim_unused_final_cells_without_resetting_phase",
            "block_weights": "positive_cell_durations_normalized_as_relative_quadrature_only",
            "aggregation": "Neumaier_sum_block_psi_times_block_duration_divided_by_total_duration",
            "aggregate_roundoff_guard": "8_ULP_at_unity_bounds_and_monotonicity_no_smoothing",
            "metric_semantics": _METRIC_SEMANTICS,
            "mlt": "uniform_ensemble_not_reconstructed_mission_history",
            "smoke": "short_interval_only_fixed_full_mission_sources_remain_unchanged",
        },
        "limitations": _LIMITATIONS,
    }


def write_report_exclusively(report: dict[str, Any], output: Path) -> None:
    _helper.write_report_exclusively(report, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("smoke", "n60", "n120", "n240"), default="n60")
    parser.add_argument("--block-orbits", type=int)
    parser.add_argument("--max-states", type=int, default=_MAX_ORBIT_MLT_STATES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        block_orbits = (2 if args.case == "smoke" else 32
                        ) if args.block_orbits is None else args.block_orbits
        print(json.dumps({"preflight": _preflight(
            _specifications(args.case), block_orbits, args.max_states,
        )}, ensure_ascii=False, allow_nan=False), flush=True)
        report = build_report(
            args.case, block_orbits=block_orbits, max_states=args.max_states,
            progress=lambda item: print(json.dumps({"progress": item}, allow_nan=False), flush=True),
        )
        write_report_exclusively(report, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
