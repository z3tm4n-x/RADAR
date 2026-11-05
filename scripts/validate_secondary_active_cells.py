"""Preflight real GCR secondary-proton partitions without computing responses.

Default matrix: GEO, 2026 launch, five years, 81/161 energy nodes, five Al
thicknesses. Both PEAK and CUMULATIVE sources use the explicit 60-month OST
solar timeline. Only the proton component is generated; its existing ion-
roster warning is preserved. All ten retained TENDL products remain present.

JSON goes to stdout (or --output) and progress goes to stderr. This script performs no
secondary-response integration and makes no numerical/normative closure claim.
Its work-count forecast uses the actual floating-point GL4 abscissae and the
current complete-or-explicit-fallback partitions, not merely knot counts.
With --prequadrature-screening, the retained expensive calls are compared with
an independently constructed unscreened partition. Proven zeros and unavailable
contributions are counted separately; neither is a convergence result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from bisect import bisect_right
from dataclasses import asdict
from datetime import datetime, timezone
from math import fsum, isfinite, sqrt
from pathlib import Path
from time import perf_counter
from typing import Any

from radar.core.constants import DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON
from radar.core.spectra import Spectrum1D
from radar.core.types import AssessmentMode, RadiationProductKind, SpectrumQuantity
from radar.gcr.model import GcrModelInput, OstGcrModel
from radar.gcr.penetration import apply_ost_geomagnetic_penetration_to_gcr_products
from radar.shielding import proton_al_secondary as secondary
from radar.shielding.proton_al import ProtonAlRangeEnergyTable, log_bin_edges_from_centers
from radar.shielding.resources import (
    load_normative_proton_al_range_table,
    load_normative_secondary_proton_kernel,
)
from radar.solar_activity.model import build_calculation_solar_activity
from radar.validation.stage4_spectra import (
    STAGE4_REFERENCE_SCENARIOS,
    build_stage4_reference_config,
)


_ROOT = Path(__file__).resolve().parent.parent
_FILES = (
    "scripts/validate_secondary_active_cells.py",
    "src/radar/shielding/proton_al_secondary.py",
    "src/radar/shielding/proton_al.py",
    "src/radar/shielding/proton_al_survival.py",
    "src/radar/nuclear/tendl.py",
    "src/radar/core/constants.py",
    "src/radar/core/project.py",
    "src/radar/gcr/model.py",
    "src/radar/gcr/parameters.py",
    "src/radar/gcr/source_spectra.py",
    "src/radar/gcr/products.py",
    "src/radar/gcr/penetration.py",
    "src/radar/solar_activity/model.py",
    "src/radar/solar_activity/ost.py",
    "src/radar/solar_activity/cycle.py",
    "src/radar/geomagnetic/ost_penetration.py",
    "src/radar/geomagnetic/penetration.py",
    "src/radar/geomagnetic/rigidity.py",
    "src/radar/geomagnetic/spectrum.py",
    "src/radar/data/normative/ost_134_1044_2007/table_g_1_wolf_numbers.csv",
    "src/radar/data/normative/stopping/proton_al_range.csv",
    "src/radar/data/normative/tendl/p_al27_mf3_xs.csv",
    "src/radar/data/normative/tendl/p_al27_mf6_proton_yield.csv",
    "src/radar/data/normative/tendl/p_al27_mf6_proton_pdf.csv",
    "src/radar/data/normative/tendl/p-Al027.tendl.txt",
)


def _hashes() -> dict[str, str]:
    result = {}
    for name in _FILES:
        with (_ROOT / name).open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def _grid(nodes: int) -> tuple[float, ...]:
    base = DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON
    if nodes == 81:
        assert len(base) == 81
        return base
    return tuple(value for lo, hi in zip(base, base[1:]) for value in (lo, sqrt(lo * hi))) + (
        base[-1],
    )


def _source_record(source: Spectrum1D) -> dict[str, Any]:
    assert source.assessment_mode is not None and source.temporal_basis is not None
    assert source.completeness is not None
    assert source.location is not None
    record = {
        "energy_mev": source.x,
        "ordinates": source.y,
        "x_unit": source.x_unit.value,
        "y_unit": source.y_unit.value,
        "quantity": source.quantity.value,
        "assessment_mode": source.assessment_mode.value,
        "temporal_basis": source.temporal_basis.value,
        "mission_duration_s": source.mission_duration_s,
        "location": source.location.value,
        "completeness": source.completeness.value,
        "incomplete_reasons": source.incomplete_reasons,
        "model_reference": asdict(source.model_reference) if source.model_reference else None,
        "transformation_chain": tuple(asdict(step) for step in source.transformation_chain),
    }
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()
    return {**record, "canonical_record_sha256": digest}


def _sources(nodes: int) -> tuple[dict[str, Spectrum1D], dict[str, Any]]:
    scenario = next(item for item in STAGE4_REFERENCE_SCENARIOS if item.name == "GEO")
    config = build_stage4_reference_config(scenario, AssessmentMode.CUMULATIVE)
    started = perf_counter()
    solar_activity = build_calculation_solar_activity(config)
    model = OstGcrModel(
        symbols=("H",),
        energy_grid_mev_per_nucleon=_grid(nodes),
        solar_activity=solar_activity,
    )
    result = model.calculate(GcrModelInput(mission=config.mission))
    source_seconds = perf_counter() - started
    started = perf_counter()
    orbit = apply_ost_geomagnetic_penetration_to_gcr_products(
        config=config, products=result.products
    )
    orbit_seconds = perf_counter() - started
    spectra = {}
    for product in orbit.products:
        source = product.spectrum
        if (
            product.kind is RadiationProductKind.MAXIMUM_FLUX
            and source.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
        ):
            spectra["peak"] = source
        elif (
            product.kind is RadiationProductKind.MISSION_FLUENCE
            and source.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
        ):
            spectra["cumulative"] = source
    assert set(spectra) == {"peak", "cumulative"}
    peak, cumulative = spectra["peak"], spectra["cumulative"]
    assert peak.x == cumulative.x
    assert tuple(value > 0 for value in peak.y) == tuple(value > 0 for value in cumulative.y)
    assert peak.mission_duration_s is None and cumulative.mission_duration_s == 157788000.0
    return spectra, {
        "grid_nodes": nodes,
        "source_generation_seconds": source_seconds,
        "orbit_penetration_seconds": orbit_seconds,
        "solar_months": solar_activity.mission_month_count,
        "method_metadata": result.method_metadata,
        "positive_support_identical_for_both_modes": True,
        "sources": {mode: _source_record(source) for mode, source in spectra.items()},
    }


def _source_value(source: Spectrum1D, energy: float) -> float:
    index = min(bisect_right(source.x, energy) - 1, len(source.x) - 2)
    lo, hi = source.x[index : index + 2]
    fraction = (energy - lo) / (hi - lo)
    return (1 - fraction) * source.y[index] + fraction * source.y[index + 1]


def _work_counts(
    source: Spectrum1D,
    boundaries: tuple[float, ...],
    stopping: ProtonAlRangeEnergyTable,
    *,
    screening: secondary._SecondaryScreeningPlan | None = None,
) -> tuple[int, int, int]:
    coarse_nodes: set[float] = set()
    first_check_nodes: set[float] = set()
    active_panels = 0
    for lo, hi in zip(boundaries, boundaries[1:]):
        if screening is not None and screening.for_interval(lo, hi).classification != "evaluate":
            continue
        if _source_value(source, lo) <= 0 and _source_value(source, hi) <= 0:
            continue
        if lo < stopping.energy_mev[0] or hi > stopping.energy_mev[-1]:
            continue
        active_panels += 1
        coarse_nodes.update(energy for energy, _ in secondary._gauss_legendre_4_samples(lo, hi))
        middle = 0.5 * (lo + hi)
        first_check_nodes.update(
            energy
            for a, b in ((lo, middle), (middle, hi))
            for energy, _ in secondary._gauss_legendre_4_samples(a, b)
        )
    return active_panels, len(coarse_nodes), len(coarse_nodes | first_check_nodes)


_SCREENING_CLASSES = ("source_zero", "proven_zero", "omitted_unsupported", "evaluate")


def _screening_record(
    source: Spectrum1D,
    partition: secondary._IncidentOutputPartition,
    *,
    include_intervals: bool,
) -> dict[str, Any] | None:
    plan = partition.screening
    if plan is None:
        return None
    base_counts = {kind: 0 for kind in _SCREENING_CLASSES}
    final_counts = {kind: 0 for kind in _SCREENING_CLASSES}
    basis_counts: dict[str, int] = {}
    weights: dict[str, list[float]] = {kind: [] for kind in _SCREENING_CLASSES}
    for interval in plan.intervals:
        base_counts[interval.classification] += 1
        weights[interval.classification].append(interval.source_integral)
        basis_counts[interval.basis] = basis_counts.get(interval.basis, 0) + 1
    for lo, hi in zip(partition.boundaries_mev, partition.boundaries_mev[1:]):
        final_counts[plan.for_interval(lo, hi).classification] += 1
    assert sum(base_counts.values()) == len(plan.intervals)
    assert sum(final_counts.values()) == len(partition.boundaries_mev) - 1
    return {
        "base_interval_counts": base_counts,
        "final_interval_counts": final_counts,
        "classification_basis_counts": basis_counts,
        "base_source_integral_by_classification": {
            kind: fsum(values) for kind, values in weights.items()
        },
        "source_integral_dimension": f"({source.y_unit.value}) * ({source.x_unit.value})",
        "source_weight_assessment_mode": (
            source.assessment_mode.value if source.assessment_mode is not None else None
        ),
        "source_weight_scope": (
            "Input-spectrum integral classified for this single output bin. The same incident "
            "particles recur in other bins; these weights must not be summed across output "
            "bins as a physical source flux/fluence. Unknown contributions are not zero."
        ),
        "evaluation_range_intervals_g_cm2": plan.evaluation_range_intervals_g_cm2,
        "omitted_physical_contribution_error": (
            "unknown" if base_counts["omitted_unsupported"] else "not_omitted_by_this_screen"
        ),
        "base_intervals": [asdict(interval) for interval in plan.intervals]
        if include_intervals
        else None,
    }


def _case(
    *,
    source: Spectrum1D,
    thickness: float,
    depth_options: secondary.DepthCrossingQuadrature,
    response_budget: int,
    prequadrature_screening: bool = False,
    include_screening_intervals: bool = False,
) -> dict[str, Any]:
    case_started = perf_counter()
    stopping = load_normative_proton_al_range_table()
    kernel = load_normative_secondary_proton_kernel()
    support = kernel.incident_energy_support
    assert support is not None
    started = perf_counter()
    knots = secondary._active_cell_incident_energy_knots(kernel=kernel, stopping_table=stopping)
    ranges = secondary._positive_source_range_intervals(source, stopping)
    cells = secondary._prepare_primary_pdf_crossing_cells(
        incident_energy_knots=knots,
        source_range_intervals=ranges,
        thickness_g_cm2=thickness,
        stopping_table=stopping,
        kernel=kernel,
    )
    common_seconds = perf_counter() - started
    output_edges = log_bin_edges_from_centers(source.x)
    records: list[dict[str, Any]] = []
    for index in range(len(source.x)):
        started = perf_counter()
        reference_part = secondary._prepare_incident_output_partition(
            output_indices=(index,),
            source_nodes=source.x,
            output_edges=output_edges,
            incident_energy_knots=knots,
            stopping_table=stopping,
            kernel=kernel,
            thickness_g_cm2=thickness,
            depth_options=depth_options,
            primary_pdf_cells=cells,
            source_range_intervals=ranges,
        )
        reference_preparation_seconds = perf_counter() - started
        reference_panels, reference_coarse, reference_first_check = _work_counts(
            source, reference_part.boundaries_mev, stopping
        )
        started = perf_counter()
        part = (
            secondary._prepare_incident_output_partition(
                output_indices=(index,),
                source_nodes=source.x,
                output_edges=output_edges,
                incident_energy_knots=knots,
                stopping_table=stopping,
                kernel=kernel,
                thickness_g_cm2=thickness,
                depth_options=depth_options,
                primary_pdf_cells=cells,
                source_range_intervals=ranges,
                screening_source=source,
            )
            if prequadrature_screening
            else reference_part
        )
        preparation_seconds = (
            perf_counter() - started if prequadrature_screening else reference_preparation_seconds
        )
        active_panels, coarse_calls, first_check_calls = (
            _work_counts(source, part.boundaries_mev, stopping, screening=part.screening)
            if prequadrature_screening
            else (reference_panels, reference_coarse, reference_first_check)
        )
        records.append(
            {
                "bin_index": index,
                "preparation_seconds": preparation_seconds,
                "unfiltered_preparation_seconds": reference_preparation_seconds,
                "common_depth_knots": len(part.depth.knots_g_cm2),
                "unique_depth_knots": part.depth.unique_depth_count,
                "active_cells": len(part.depth.active_cells),
                "active_cell_knot_records": sum(
                    len(cell.knots_g_cm2) for cell in part.depth.active_cells
                ),
                "source_intervals": len(part.boundaries_mev) - 1,
                "nonzero_source_intervals": active_panels,
                "unique_coarse_response_calls": coarse_calls,
                "unique_coarse_plus_first_check_calls": first_check_calls,
                "unfiltered_nonzero_source_intervals": reference_panels,
                "unfiltered_unique_coarse_response_calls": reference_coarse,
                "unfiltered_unique_coarse_plus_first_check_calls": reference_first_check,
                "unfiltered_depth_partition_budget_exceeded": reference_part.depth.budget_exceeded,
                "unfiltered_coupled_partition_budget_exceeded": reference_part.coupled_budget_exceeded,
                "screening": _screening_record(
                    source, part, include_intervals=include_screening_intervals
                ),
                "depth_partition_budget_exceeded": part.depth.budget_exceeded,
                "coupled_partition_budget_exceeded": part.coupled_budget_exceeded,
            }
        )
    coarse_calls = sum(item["unique_coarse_response_calls"] for item in records)
    initial_check_calls = sum(item["unique_coarse_plus_first_check_calls"] for item in records)
    unfiltered_coarse_calls = sum(item["unfiltered_unique_coarse_response_calls"] for item in records)
    unfiltered_first_check_calls = sum(
        item["unfiltered_unique_coarse_plus_first_check_calls"] for item in records
    )
    partition_failures = [
        item["bin_index"]
        for item in records
        if item["depth_partition_budget_exceeded"] or item["coupled_partition_budget_exceeded"]
    ]
    reference_failures = [
        item["bin_index"] for item in records
        if item["unfiltered_depth_partition_budget_exceeded"]
        or item["unfiltered_coupled_partition_budget_exceeded"]
    ]
    screening_base_counts = {
        kind: sum(
            item["screening"]["base_interval_counts"][kind]
            for item in records if item["screening"] is not None
        )
        for kind in _SCREENING_CLASSES
    }
    screening_final_counts = {
        kind: sum(
            item["screening"]["final_interval_counts"][kind]
            for item in records if item["screening"] is not None
        )
        for kind in _SCREENING_CLASSES
    }
    maximum_range = stopping.range_at_energy(support.maximum_mev) + thickness
    max_known_entrance = (
        stopping.energy_at_range(maximum_range)
        if maximum_range <= stopping.max_range_g_cm2
        else None
    )
    nonzero_source_intervals = [
        (lo, hi)
        for lo, hi, left, right in zip(source.x, source.x[1:], source.y, source.y[1:])
        if left > 0 or right > 0
    ]
    return {
        "grid_nodes": len(source.x),
        "thickness_g_cm2": thickness,
        "strategy": "per_output_bin_active_cells",
        "prequadrature_screening": prequadrature_screening,
        "modes_sharing_geometry": ["peak", "cumulative"],
        "positive_source_range_intervals_g_cm2": ranges,
        "primary_incident_knots": len(knots),
        "reachable_primary_pdf_cells": len(cells),
        "primary_pdf_cells_with_missing_support": sum(cell.pdf_support_missing for cell in cells),
        "common_preparation_seconds": common_seconds,
        "preflight_wall_seconds_including_work_count_forecast": perf_counter() - case_started,
        "output_partition_preparation_seconds": sum(
            item["preparation_seconds"] for item in records
        ),
        "unfiltered_output_partition_preparation_seconds": sum(
            item["unfiltered_preparation_seconds"] for item in records
        ),
        "partition_budget_failures_bin_indices": partition_failures,
        "all_crossing_partitions_included": not partition_failures,
        "unfiltered_partition_budget_failures_bin_indices": reference_failures,
        "unfiltered_all_crossing_partitions_included": not reference_failures,
        "common_depth_knots_sum": sum(item["common_depth_knots"] for item in records),
        "unique_depth_knots_sum_across_bins": sum(item["unique_depth_knots"] for item in records),
        "unique_depth_knots_max_per_bin": max(item["unique_depth_knots"] for item in records),
        "active_cell_knot_records_sum": sum(item["active_cell_knot_records"] for item in records),
        "active_cell_knot_records_max_per_bin": max(
            item["active_cell_knot_records"] for item in records
        ),
        "nonzero_source_panels_sum_across_bins": sum(
            item["nonzero_source_intervals"] for item in records
        ),
        "complete_coarse_pass_unique_response_calls": coarse_calls,
        "complete_initial_error_check_unique_response_calls": initial_check_calls,
        "unfiltered_complete_coarse_pass_unique_response_calls": unfiltered_coarse_calls,
        "unfiltered_complete_initial_error_check_unique_response_calls": unfiltered_first_check_calls,
        "coarse_call_reduction_fraction": (
            1.0 - coarse_calls / unfiltered_coarse_calls if unfiltered_coarse_calls else None
        ),
        "screening_base_interval_counts_sum_across_bins": screening_base_counts,
        "screening_final_interval_counts_sum_across_bins": screening_final_counts,
        "screening_counts_scope": "Numerical work intervals, repeated per output bin; not physical counts",
        "unknown_omitted_physical_contribution": screening_base_counts["omitted_unsupported"] > 0,
        "work_count_meaning": (
            "Unique scalar-bin GL4 evaluations for a full coarse pass and its first half-panel "
            "check using these actual partitions. Refinement can add work. When screening is "
            "enabled, only evaluate-class panels require response nodes; proven-zero and "
            "omitted-unsupported classes are reported separately. Unfiltered fields use a "
            "separately prepared reference partition. No responses are calculated here."
        ),
        "configured_global_response_budget": response_budget,
        "coarse_pass_exceeds_response_budget": coarse_calls > response_budget,
        "initial_error_check_exceeds_response_budget": initial_check_calls > response_budget,
        "maximum_entrance_with_any_represented_tendl_path_mev": max_known_entrance,
        "represented_path_entrance_bound_status": (
            "supported" if max_known_entrance is not None else "beyond_stopping_table_domain"
        ),
        "nonzero_source_intervals_wholly_above_represented_path": (
            sum(lo >= max_known_entrance for lo, _ in nonzero_source_intervals)
            if max_known_entrance is not None
            else None
        ),
        "source_has_positive_ordinates_above_tendl_support": any(
            value > 0 and energy > support.maximum_mev for energy, value in zip(source.x, source.y)
        ),
        "unknown_high_energy_is_not_zero": True,
        "secondary_response_evaluations_performed": 0,
        "numerical_closure_claim": False,
        "bins": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", nargs="+", type=int, choices=(81, 161), default=(81, 161))
    parser.add_argument("--thickness", nargs="+", type=float, default=(0.01, 0.1, 1.0, 3.0, 10.0))
    parser.add_argument("--max-depth-knots", type=int, default=8192)
    parser.add_argument("--max-coupled-incident-knots", type=int, default=65536)
    parser.add_argument("--max-active-cell-records", type=int, default=32768)
    parser.add_argument("--max-response-evaluations", type=int, default=4096)
    parser.add_argument(
        "--prequadrature-screening", action="store_true",
        help="Forecast exact zero/unsupported screening separately from the unfiltered reference",
    )
    parser.add_argument(
        "--include-screening-intervals", action="store_true",
        help="Include every base screening interval; potentially large JSON output",
    )
    parser.add_argument("--output", type=Path, help="Write the complete JSON artifact to this file")
    args = parser.parse_args()
    if any(not isfinite(value) or value <= 0 for value in args.thickness):
        parser.error("--thickness values must be finite and positive; zero identity is out of scope")
    if args.include_screening_intervals and not args.prequadrature_screening:
        parser.error("--include-screening-intervals requires --prequadrature-screening")
    try:
        secondary.IncidentEnergyQuadrature(
            max_response_evaluations=args.max_response_evaluations,
            partition_strategy="per_output_bin_active_cells",
            prequadrature_screening=args.prequadrature_screening,
        )
    except ValueError as exc:
        parser.error(str(exc))
    options = secondary.DepthCrossingQuadrature(
        max_depth_knots=args.max_depth_knots,
        max_coupled_incident_knots=args.max_coupled_incident_knots,
        max_active_cell_records=args.max_active_cell_records,
    )
    before_hashes = _hashes()
    sources, cases = [], []
    for nodes in args.grids:
        spectra, source_record = _sources(nodes)
        sources.append(source_record)
        for thickness in args.thickness:
            print(
                f"Preflight {nodes} nodes, t={thickness:g} g/cm2; no responses",
                file=sys.stderr,
                flush=True,
            )
            result = _case(
                source=spectra["peak"],
                thickness=thickness,
                depth_options=options,
                response_budget=args.max_response_evaluations,
                prequadrature_screening=args.prequadrature_screening,
                include_screening_intervals=args.include_screening_intervals,
            )
            cases.append(result)
            print(
                f"Prepared in {result['common_preparation_seconds'] + result['output_partition_preparation_seconds']:.3f}s; "
                f"coarse calls={result['complete_coarse_pass_unique_response_calls']} "
                f"(unfiltered={result['unfiltered_complete_coarse_pass_unique_response_calls']}); "
                f"partition fallbacks={len(result['partition_budget_failures_bin_indices'])}",
                file=sys.stderr,
                flush=True,
            )
    after_hashes = _hashes()
    assert [(case["grid_nodes"], case["thickness_g_cm2"]) for case in cases] == [
        (nodes, thickness) for nodes in args.grids for thickness in args.thickness
    ]
    assert all(
        [record["bin_index"] for record in case["bins"]] == list(range(case["grid_nodes"]))
        for case in cases
    )
    report = {
        "schema": "radar.secondary-prequadrature-preflight.v1"
        if args.prequadrature_screening else "radar.secondary-active-cells-preflight.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Actual OST GCR proton component, GEO 35786 km / 0 degrees, 2026 launch, "
            "5-year mission with explicit monthly solar activity. Full source energy "
            "domain and full TENDL kernel retained. Geometry/work preflight only; no "
            "secondary numerical integration or normative qualification. H-only source "
            "roster warning and all unavailable high-energy contributions are not cleared."
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "command_arguments": sys.argv[1:],
        "depth_partition_options": asdict(options),
        "global_response_budget_for_forecast": args.max_response_evaluations,
        "prequadrature_screening": args.prequadrature_screening,
        "include_screening_intervals": args.include_screening_intervals,
        "screening_classification_meaning": {
            "source_zero": "Both ordinates of the exact linear source segment are zero",
            "proven_zero": "Supported full-interval analytical zero proof; not sampled-response agreement",
            "omitted_unsupported": "Unknown physical contribution without a represented operator path; never a physical zero",
            "evaluate": "Retained expensive response work; absence of a zero proof is not a positivity claim",
        },
        "response_work_unit": "scalar_exit_bin_response_globally_across_all_bins",
        "file_sha256_at_start": before_hashes,
        "file_sha256_at_end": after_hashes,
        "source_files_changed_during_run": before_hashes != after_hashes,
        "source_generation": sources,
        "cases": cases,
        "secondary_response_evaluations_performed": 0,
        "normative_certification": False,
    }
    serialized = json.dumps(report, ensure_ascii=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote {len(cases)} complete preflight cases to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
