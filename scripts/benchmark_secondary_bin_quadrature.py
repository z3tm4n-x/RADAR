"""Bounded, reproducible algorithm comparison; this is not a normative reference.

Run from the repository's configured Python environment, for example::

    python scripts/benchmark_secondary_bin_quadrature.py --thickness 1e-5 1e-4

JSON is written to stdout by default; --output saves a complete UTF-8 report
after checking that all recorded inputs stayed unchanged. Existing output
files are not overwritten. Short progress messages go to stderr. Run a
single thickness/strategy under an external process timeout when a strict
wall-clock limit is required. A partition-budget fallback is rejected before
calculation rather than compared with a full partition as an exact reference.
The measured calculations retain all physical/data/depth-accuracy blockers.
Use --compare-birth-partition to compare the same incident quadrature and
screening settings with exact birth-survival partition reuse disabled/enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding import proton_al_secondary as secondary
from radar.shielding.proton_al import log_bin_edges_from_centers
from radar.shielding.resources import (
    load_normative_proton_al_range_table,
    load_normative_secondary_proton_kernel,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_X = (1.0, 3.0, 33.0, 33.01, 33.02)
_SOURCE_Y = (0.0, 0.0, 0.0, 1.0, 0.0)
PartitionStrategy = Literal["global_union", "per_output_bin", "per_output_bin_active_cells"]
_INPUT_FILES = (
    "src/radar/shielding/proton_al_secondary.py",
    "src/radar/shielding/proton_al.py",
    "src/radar/shielding/proton_al_survival.py",
    "src/radar/shielding/resources.py",
    "src/radar/nuclear/tendl.py",
    "src/radar/core/spectra.py",
    "src/radar/core/spectrum_ops.py",
    "src/radar/core/types.py",
    "src/radar/core/units.py",
    "src/radar/data/integrity.py",
    "src/radar/data/normative/integrity_manifest.json",
    "src/radar/data/normative/stopping/proton_al_range.csv",
    "src/radar/data/normative/tendl/p_al27_mf3_xs.csv",
    "src/radar/data/normative/tendl/p_al27_mf6_proton_yield.csv",
    "src/radar/data/normative/tendl/p_al27_mf6_proton_pdf.csv",
    "src/radar/data/normative/tendl/p-Al027.tendl.txt",
    "scripts/benchmark_secondary_bin_quadrature.py",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thickness", nargs="+", type=float, default=(1e-5, 1e-4))
    parser.add_argument(
        "--strategy",
        choices=("both", "global_union", "per_output_bin", "per_output_bin_active_cells"),
        default="both",
    )
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-response-evaluations", type=int, default=4096)
    parser.add_argument("--max-refinement-depth", type=int, default=6)
    parser.add_argument("--max-unproven-zero-refinement-depth", type=int, default=2)
    parser.add_argument("--max-depth-knots", type=int, default=8192)
    parser.add_argument("--max-coupled-incident-knots", type=int, default=65536)
    parser.add_argument("--max-active-cell-records", type=int, default=32768)
    parser.add_argument("--depth-steps", type=int, default=1)
    parser.add_argument(
        "--prequadrature-screening", action="store_true",
        help="Enable opt-in proof-based screening (active-cells strategy only).",
    )
    parser.add_argument(
        "--compare-screening", action="store_true",
        help="Compare unscreened v7 and screened v8 for each thickness (active-cells only).",
    )
    parser.add_argument(
        "--reuse-birth-survival-partition", action="store_true",
        help="Enable exact mapping-local reuse of the birth-survival partition (v9).",
    )
    parser.add_argument(
        "--compare-birth-partition", action="store_true",
        help="Compare reuse disabled/enabled at unchanged screening and quadrature settings.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Write the complete validated JSON report to this new file instead of stdout.",
    )
    return parser


def _case(
    *,
    thickness: float,
    strategy: PartitionStrategy,
    args: argparse.Namespace,
    screening: bool | None = None,
    reuse_birth_partition: bool | None = None,
) -> dict[str, Any]:
    if screening is None:
        screening = bool(args.prequadrature_screening)
    if reuse_birth_partition is None:
        reuse_birth_partition = bool(args.reuse_birth_survival_partition)
    stopping = load_normative_proton_al_range_table()
    kernel = load_normative_secondary_proton_kernel()
    source = Spectrum1D(
        x=_SOURCE_X,
        y=_SOURCE_Y,
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.GCR,
        model="synthetic_triangle_33_to_33p02_MeV_with_full_actual_TENDL_kernel",
    )
    incident_options = secondary.IncidentEnergyQuadrature(
        relative_tolerance=args.relative_tolerance,
        max_response_evaluations=args.max_response_evaluations,
        max_refinement_depth=args.max_refinement_depth,
        max_unproven_zero_refinement_depth=args.max_unproven_zero_refinement_depth,
        partition_strategy=strategy,
        prequadrature_screening=screening,
        reuse_birth_survival_partition=reuse_birth_partition,
    )
    depth_options = secondary.DepthCrossingQuadrature(
        max_depth_knots=args.max_depth_knots,
        max_coupled_incident_knots=args.max_coupled_incident_knots,
        max_active_cell_records=args.max_active_cell_records,
    )
    source_edges = log_bin_edges_from_centers(_SOURCE_X)
    groups = (
        tuple((index,) for index in range(len(_SOURCE_X)))
        if strategy != "global_union"
        else (tuple(range(len(_SOURCE_X))),)
    )
    started = perf_counter()
    active = strategy == "per_output_bin_active_cells"
    incident_knots = (
        secondary._active_cell_incident_energy_knots(kernel=kernel, stopping_table=stopping)
        if active else secondary._secondary_incident_energy_knots(kernel=kernel, stopping_table=stopping)
    )
    source_ranges = secondary._positive_source_range_intervals(source, stopping) if active else None
    cells = (
        secondary._prepare_primary_pdf_crossing_cells(
            incident_energy_knots=incident_knots, source_range_intervals=source_ranges,
            thickness_g_cm2=thickness, stopping_table=stopping, kernel=kernel,
        ) if source_ranges is not None else None
    )
    partitions = tuple(
        secondary._prepare_incident_output_partition(
            output_indices=indices,
            source_nodes=_SOURCE_X,
            output_edges=source_edges,
            incident_energy_knots=incident_knots,
            stopping_table=stopping,
            kernel=kernel,
            thickness_g_cm2=thickness,
            depth_options=depth_options,
            primary_pdf_cells=cells,
            source_range_intervals=source_ranges,
            screening_source=source if screening else None,
        )
        for indices in groups
    )
    preflight_seconds = perf_counter() - started
    if any(part.depth.budget_exceeded or part.coupled_budget_exceeded for part in partitions):
        raise RuntimeError(
            "Full crossing partition exceeds the requested budgets; refused a fallback benchmark."
        )

    original_response = secondary._single_incident_secondary_response
    response_seconds = 0.0
    response_calls = 0
    response_bin_calls = 0

    def timed_response(**kwargs: Any) -> secondary._SingleIncidentResponse:
        nonlocal response_seconds, response_calls, response_bin_calls
        response_started = perf_counter()
        try:
            return original_response(**kwargs)
        finally:
            response_seconds += perf_counter() - response_started
            response_calls += 1
            response_bin_calls += len(kwargs["output_centers_mev"])

    secondary._single_incident_secondary_response = timed_response
    started = perf_counter()
    try:
        result = secondary.calculate_secondary_proton_spectrum_through_al(
            incident_spectrum=source,
            stopping_table=stopping,
            kernel=kernel,
            thickness_g_cm2=thickness,
            depth_steps=args.depth_steps,
            survival_integration_steps=64,
            incident_quadrature=incident_options,
            depth_quadrature=depth_options,
        )
    finally:
        elapsed = perf_counter() - started
        secondary._single_incident_secondary_response = original_response
    assert result.bin_edges is not None
    assert result.completeness is not None
    counts = tuple(
        ordinate * (upper - lower)
        for ordinate, lower, upper in zip(result.y, result.bin_edges, result.bin_edges[1:])
    )
    return {
        "strategy": strategy,
        "prequadrature_screening": screening,
        "reuse_birth_survival_partition": reuse_birth_partition,
        "algorithm_label": (
            "v9_reused_birth_survival_partition" if reuse_birth_partition
            else "v8_screened_active_cells" if screening
            else "v7_unscreened_active_cells" if active
            else strategy
        ),
        "transformation_version": result.transformation_chain[-1].version,
        "thickness_g_cm2": thickness,
        "incident_options": asdict(incident_options),
        "depth_options": asdict(depth_options),
        "depth_steps": args.depth_steps,
        "survival_integration_steps": 64,
        "preflight_seconds": preflight_seconds,
        "calculation_seconds_including_internal_preparation": elapsed,
        "single_response_seconds": response_seconds,
        "observed_response_calls": response_calls,
        "observed_response_bin_calls": response_bin_calls,
        "depth_knots_by_partition": [part.depth.unique_depth_count for part in partitions],
        "active_cell_records_by_partition": [
            sum(len(cell.knots_g_cm2) for cell in part.depth.active_cells) for part in partitions
        ],
        "source_intervals_by_partition": [len(part.boundaries_mev) - 1 for part in partitions],
        "all_crossing_partitions_included": True,
        "crossing_partition_scope": (
            "evaluable_screened_intervals_only" if screening else "all_positive_source_intervals"
        ),
        "retained_kernel_product_count": len(kernel.products),
        "output_bin_edges_mev": result.bin_edges,
        "output_ordinates": result.y,
        "output_counts_by_bin": counts,
        "total_count": sum(counts),
        "quantity": result.quantity.value,
        "y_unit": result.y_unit.value,
        "completeness": result.completeness.value,
        "incomplete_reasons": result.incomplete_reasons,
        "parameters": dict(result.transformation_chain[-1].parameters),
    }


def _compare_cases(
    first: dict[str, Any], second: dict[str, Any], *, screening_comparison: bool,
    birth_partition_comparison: bool = False,
) -> dict[str, Any]:
    if screening_comparison and birth_partition_comparison:
        raise ValueError("A comparison must vary only one optimization.")
    for field in ("thickness_g_cm2", "output_bin_edges_mev", "quantity", "y_unit"):
        if first[field] != second[field]:
            raise ValueError(f"Cannot compare cases with different {field}.")
    if birth_partition_comparison:
        for field in (
            "strategy", "prequadrature_screening", "depth_options", "depth_steps",
            "survival_integration_steps",
        ):
            if first.get(field) != second.get(field):
                raise ValueError(f"Birth-partition comparison requires unchanged {field}.")
        if (
            first["reuse_birth_survival_partition"] is not False
            or second["reuse_birth_survival_partition"] is not True
        ):
            raise ValueError("Birth-partition comparison requires reuse disabled then enabled.")
        reference_options = dict(first.get("incident_options", {}))
        candidate_options = dict(second.get("incident_options", {}))
        reference_options.pop("reuse_birth_survival_partition", None)
        candidate_options.pop("reuse_birth_survival_partition", None)
        if reference_options != candidate_options:
            raise ValueError("Birth-partition comparison requires unchanged incident options.")
    reference, candidate = first["output_counts_by_bin"], second["output_counts_by_bin"]
    absolute = [abs(right - left) for left, right in zip(reference, candidate, strict=True)]
    candidate_seconds = second["calculation_seconds_including_internal_preparation"]
    wall_ratio = (
        first["calculation_seconds_including_internal_preparation"] / candidate_seconds
        if candidate_seconds > 0 else None
    )
    full_statuses = [
        item["parameters"]["incident_quadrature_status"] for item in (first, second)
    ]
    supported_statuses = [
        item["parameters"].get("incident_quadrature_supported_status", full_status)
        for item, full_status in zip((first, second), full_statuses, strict=True)
    ]
    comparison = {
        "comparison_kind": (
            "birth_survival_partition_reuse" if birth_partition_comparison
            else "screening_v7_vs_v8" if screening_comparison else "partition_strategies"
        ),
        "thickness_g_cm2": first["thickness_g_cm2"],
        "reference_label": first["algorithm_label"],
        "candidate_label": second["algorithm_label"],
        "reference_over_candidate_wall_time_ratio": wall_ratio,
        "absolute_count_difference_by_bin": absolute,
        "count_vectors_exactly_equal": all(error == 0.0 for error in absolute),
        "relative_count_difference_by_nonzero_reference_bin": [
            error / abs(value) if value else None
            for error, value in zip(absolute, reference, strict=True)
        ],
        "l1_count_relative_difference": sum(absolute) / sum(reference) if sum(reference) else None,
        "reference_full_quadrature_status": full_statuses[0],
        "candidate_full_quadrature_status": full_statuses[1],
        "reference_supported_quadrature_status": supported_statuses[0],
        "candidate_supported_quadrature_status": supported_statuses[1],
        "both_outer_integrals_converged": all(status == "converged" for status in full_statuses),
        "both_retained_outer_integrals_converged": all(
            status == "converged" for status in supported_statuses
        ),
        "agreement_scope": (
            "retained bin counts and outer quadrature estimators only; missing physical domains "
            "and model/depth-accuracy blockers are not validated by this comparison"
        ),
        "normative_certification": False,
    }
    if not screening_comparison and not birth_partition_comparison:
        comparison["global_over_per_bin_wall_time_ratio"] = wall_ratio
    return comparison


def _input_hashes() -> dict[str, str]:
    return {name: _sha256(_PROJECT_ROOT / name) for name in _INPUT_FILES}


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.compare_screening and (
        args.compare_birth_partition or args.reuse_birth_survival_partition
    ):
        parser.error("--compare-screening cannot be combined with birth-partition options")
    if (args.prequadrature_screening or args.compare_screening) and (
        args.strategy != "per_output_bin_active_cells"
    ):
        parser.error("screening options require --strategy per_output_bin_active_cells")
    if args.output is not None:
        output_path = args.output.resolve()
        if output_path in {(_PROJECT_ROOT / name).resolve() for name in _INPUT_FILES}:
            parser.error("--output must not name a benchmark input file")
        if output_path.exists():
            parser.error("--output must name a new file; existing files are not overwritten")
    started_at = datetime.now(timezone.utc).isoformat()
    start_hashes = _input_hashes()
    strategies: tuple[PartitionStrategy, ...] = (
        ("global_union", "per_output_bin") if args.strategy == "both" else (args.strategy,)
    )
    cases = []
    comparisons = []
    for thickness in args.thickness:
        groups: tuple[tuple[tuple[PartitionStrategy, bool, bool], ...], ...]
        if args.compare_screening:
            groups = ((("per_output_bin_active_cells", False, False),
                       ("per_output_bin_active_cells", True, False)),)
        elif args.compare_birth_partition:
            groups = tuple(
                ((strategy, args.prequadrature_screening, False),
                 (strategy, args.prequadrature_screening, True))
                for strategy in strategies
            )
        else:
            groups = (tuple(
                (strategy, args.prequadrature_screening, args.reuse_birth_survival_partition)
                for strategy in strategies
            ),)
        for configurations in groups:
            group_cases = []
            for strategy, screening, reuse in configurations:
                print(
                    f"Starting {strategy}, screening={screening}, birth_partition_reuse={reuse}, "
                    f"t={thickness:g} g/cm2",
                    file=sys.stderr, flush=True,
                )
                case = _case(
                    thickness=thickness, strategy=strategy, args=args, screening=screening,
                    reuse_birth_partition=reuse,
                )
                case["comparison_role"] = (
                    ("candidate" if reuse else "reference") if args.compare_birth_partition
                    else ("candidate" if screening else "reference") if args.compare_screening
                    else "standalone"
                )
                print(
                    f"Finished {strategy} in {case['calculation_seconds_including_internal_preparation']:.3f}s; "
                    f"outer status={case['parameters']['incident_quadrature_status']}",
                    file=sys.stderr,
                    flush=True,
                )
                cases.append(case)
                group_cases.append(case)
            if len(group_cases) == 2:
                comparisons.append(_compare_cases(
                    *group_cases, screening_comparison=args.compare_screening,
                    birth_partition_comparison=args.compare_birth_partition,
                ))
    end_hashes = _input_hashes()
    changed_inputs = sorted(name for name in start_hashes if start_hashes[name] != end_hashes[name])
    if changed_inputs:
        raise RuntimeError(
            "Benchmark inputs changed during the run; report was not published: "
            + ", ".join(changed_inputs)
        )
    report = {
        "schema": "radar.secondary-partition-benchmark.v2",
        "started_at_utc": started_at,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Synthetic triangular incident fluence on 33..33.02 MeV, all retained actual TENDL "
            "products and the current provisional Al range table. Thin-shield algorithm "
            "comparison only, not a GCR environment benchmark or normative reference. "
            "Wall times are local measurements, not general speedup guarantees."
        ),
        "source_energy_mev": _SOURCE_X,
        "source_ordinates": _SOURCE_Y,
        "source_interpolation": "linear_on_true_source_intervals_no_extrapolation",
        "python_version": sys.version,
        "platform": platform.platform(),
        "command_arguments": sys.argv[1:] if argv is None else argv,
        "file_sha256": start_hashes,
        "file_sha256_at_start": start_hashes,
        "file_sha256_at_end": end_hashes,
        "input_hashes_consistent": True,
        "hash_validation_scope": (
            "listed files agree at start and end; this does not detect edits reverted during "
            "the run or certify unlisted dependencies"
        ),
        "normative_certification": False,
        "cases": cases,
        "comparisons": comparisons,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        print(f"Saved complete benchmark report: {args.output}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
