"""Bounded actual secondary-proton calculation on the full GEO GCR source.

This is an execution/cost diagnostic, not a full-spectrum convergence run.
The complete 81/161-node source and all ten retained TENDL products are passed
to the operator, including unavailable high-energy domains. A response budget
can leave unresolved intervals and bins. Their missing contributions are
unknown, never physical zeros. The reported retained_diagnostic_count_sum is
therefore not a completed radiation-environment result.

Use an external process timeout when a strict wall-clock bound is needed:
the response-evaluation budget does not cap preparation or single-call cost.
Reports require a new --output file and unchanged pinned inputs. No existing
file is overwritten and no additional mission-time factor is applied.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from math import fsum, isfinite
from pathlib import Path
import platform
import sys
from time import perf_counter
from types import ModuleType
from typing import Any, cast

from radar.core.spectra import Spectrum1D
from radar.shielding import proton_al_secondary as secondary
from radar.shielding.resources import (
    load_normative_proton_al_range_table,
    load_normative_secondary_proton_kernel,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_preflight_runner() -> ModuleType:
    # Explicit sibling loading also works when tests import this script by path;
    # neither sys.path nor the application's import configuration is changed.
    path = Path(__file__).with_name("validate_secondary_active_cells.py")
    spec = spec_from_file_location("radar_full_source_preflight_dependency", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the existing full-source preflight helper.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PREFLIGHT = _load_preflight_runner()
_INPUT_FILES = tuple(dict.fromkeys((
    *cast(tuple[str, ...], _PREFLIGHT._FILES),
    "scripts/validate_secondary_full_source.py",
    "src/radar/shielding/resources.py",
    "src/radar/core/spectra.py",
    "src/radar/core/spectrum_ops.py",
    "src/radar/core/types.py",
    "src/radar/core/units.py",
    "src/radar/core/products.py",
    "src/radar/core/profiles.py",
    "src/radar/validation/stage4_spectra.py",
    "src/radar/data/integrity.py",
    "src/radar/data/normative/integrity_manifest.json",
)))


def _input_hashes() -> dict[str, str]:
    hashes = {}
    for name in _INPUT_FILES:
        with (_PROJECT_ROOT / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def _sources(nodes: int) -> tuple[dict[str, Spectrum1D], dict[str, Any]]:
    return cast(tuple[dict[str, Spectrum1D], dict[str, Any]], _PREFLIGHT._sources(nodes))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, choices=(81, 161), default=81)
    parser.add_argument("--mode", choices=("peak", "cumulative"), default="peak")
    parser.add_argument("--thickness", type=float, required=True)
    parser.add_argument("--depth-steps", type=int, default=1)
    parser.add_argument("--max-response-evaluations", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True, help="New exclusive JSON output file")
    return parser


def _incident_options(args: argparse.Namespace) -> secondary.IncidentEnergyQuadrature:
    return secondary.IncidentEnergyQuadrature(
        relative_tolerance=1e-3,
        max_response_evaluations=args.max_response_evaluations,
        partition_strategy="per_output_bin_active_cells",
        prequadrature_screening=True,
        reuse_birth_survival_partition=True,
    )


def _calculate(args: argparse.Namespace) -> dict[str, Any]:
    source_started = perf_counter()
    sources, source_generation = _sources(args.grid)
    source = sources[args.mode]
    source_seconds = perf_counter() - source_started
    if len(source.x) != args.grid:
        raise RuntimeError("The full source grid does not match the requested node count.")
    resource_started = perf_counter()
    stopping = load_normative_proton_al_range_table()
    kernel = load_normative_secondary_proton_kernel()
    resource_seconds = perf_counter() - resource_started
    if len(kernel.products) != 10:
        raise RuntimeError("This pinned diagnostic requires all ten retained TENDL products.")
    incident_options = _incident_options(args)
    depth_options = secondary.DepthCrossingQuadrature(
        max_depth_knots=8192,
        max_coupled_incident_knots=65536,
        max_active_cell_records=32768,
    )
    original_response = secondary._single_incident_secondary_response
    response_calls = 0
    response_bin_calls = 0
    response_seconds = 0.0
    calls_by_bin = [0] * len(source.x)
    seconds_by_bin = [0.0] * len(source.x)
    bin_index_by_center = {center: index for index, center in enumerate(source.x)}

    def timed_response(**kwargs: Any) -> secondary._SingleIncidentResponse:
        nonlocal response_calls, response_bin_calls, response_seconds
        started = perf_counter()
        try:
            return original_response(**kwargs)
        finally:
            elapsed = perf_counter() - started
            centers = kwargs["output_centers_mev"]
            response_calls += 1
            response_bin_calls += len(centers)
            response_seconds += elapsed
            for center in centers:
                index = bin_index_by_center[center]
                calls_by_bin[index] += 1
                seconds_by_bin[index] += elapsed

    secondary._single_incident_secondary_response = timed_response
    calculation_started = perf_counter()
    try:
        result = secondary.calculate_secondary_proton_spectrum_through_al(
            incident_spectrum=source,
            stopping_table=stopping,
            kernel=kernel,
            thickness_g_cm2=args.thickness,
            depth_steps=args.depth_steps,
            survival_integration_steps=64,
            incident_quadrature=incident_options,
            depth_quadrature=depth_options,
        )
    finally:
        calculation_seconds = perf_counter() - calculation_started
        secondary._single_incident_secondary_response = original_response

    if result.bin_edges is None or result.x != source.x:
        raise RuntimeError("The diagnostic output must retain every original source/output bin.")
    if any(reason not in result.incomplete_reasons for reason in source.incomplete_reasons):
        raise RuntimeError("The calculation discarded an inherited source warning.")
    for field in (
        "quantity", "x_unit", "y_unit", "assessment_mode", "temporal_basis", "mission_duration_s",
    ):
        if getattr(result, field) != getattr(source, field):
            raise RuntimeError(f"The diagnostic changed the source time/unit contract: {field}.")
    parameters = dict(result.transformation_chain[-1].parameters)
    reported_calls = int(parameters["incident_quadrature_response_evaluations"])
    if reported_calls != response_calls or response_calls != response_bin_calls:
        raise RuntimeError("Observed response calls do not match the scalar operator accounting.")
    if response_calls > args.max_response_evaluations:
        raise RuntimeError("The calculation exceeded its global response-evaluation budget.")
    counts = tuple(
        value * (upper - lower)
        for value, lower, upper in zip(
            result.y, result.bin_edges[:-1], result.bin_edges[1:], strict=True,
        )
    )
    support = kernel.incident_energy_support
    return {
        "grid_nodes": args.grid,
        "mode": args.mode,
        "thickness_g_cm2": args.thickness,
        "depth_steps": args.depth_steps,
        "survival_integration_steps": 64,
        "incident_options": asdict(incident_options),
        "depth_options": asdict(depth_options),
        "source_generation": source_generation,
        "input_spectrum": asdict(source),
        "source_domain_policy": "complete_original_source_no_trimming_or_ordinate_rescaling",
        "implicit_time_factor_applied": False,
        "retained_kernel_product_count": len(kernel.products),
        "retained_kernel_products": [
            {"mt": product.mt, "product_index": product.product_index}
            for product in kernel.products
        ],
        "tendl_incident_energy_support": asdict(support) if support is not None else None,
        "source_preparation_seconds": source_seconds,
        "resource_loading_seconds": resource_seconds,
        "calculation_seconds_including_internal_preparation": calculation_seconds,
        "single_response_seconds": response_seconds,
        "observed_response_calls": response_calls,
        "observed_response_bin_calls": response_bin_calls,
        "observed_response_calls_by_output_bin": calls_by_bin,
        "single_response_seconds_by_output_bin": seconds_by_bin,
        "response_budget_reached": response_calls >= args.max_response_evaluations,
        "output_spectrum": asdict(result),
        "output_energy_mev": result.x,
        "output_bin_edges_mev": result.bin_edges,
        "output_ordinates": result.y,
        "output_counts_by_bin": counts,
        "retained_diagnostic_count_sum": fsum(counts),
        "retained_diagnostic_count_dimension": f"({result.y_unit.value}) * ({result.x_unit.value})",
        "retained_diagnostic_count_scope": (
            "Only retained numerical contributions. Budget-unresolved intervals and unavailable "
            "physical contributions are unknown, not zero. This is not a complete spectrum, "
            "dose, event count or convergence certificate. PEAK remains per second; CUMULATIVE "
            "retains its already integrated mission fluence without another time multiplier."
        ),
        "completeness": result.completeness.value if result.completeness is not None else None,
        "incomplete_reasons": result.incomplete_reasons,
        "parameters": parameters,
        "full_spectrum_convergence_claim": False,
        "normative_certification": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if not isfinite(args.thickness) or args.thickness <= 0:
        parser.error("--thickness must be finite and positive; zero identity is not this diagnostic")
    if args.depth_steps < 1:
        parser.error("--depth-steps must be a positive integer")
    try:
        _incident_options(args)
    except ValueError as error:
        parser.error(str(error))
    output_path = args.output.resolve()
    if output_path in {(_PROJECT_ROOT / name).resolve() for name in _INPUT_FILES}:
        parser.error("--output must not name a pinned input")
    if output_path.exists():
        parser.error("--output must name a new file; existing files are not overwritten")
    started_at = datetime.now(timezone.utc).isoformat()
    start_hashes = _input_hashes()
    print(
        f"Bounded full-source GEO GCR: {args.grid} nodes, {args.mode}, "
        f"Al={args.thickness:g} g/cm2, response budget={args.max_response_evaluations}; "
        "partial diagnostic only",
        file=sys.stderr, flush=True,
    )
    calculation = _calculate(args)
    end_hashes = _input_hashes()
    if start_hashes != end_hashes:
        raise RuntimeError("Pinned inputs changed during calculation; report was not published.")
    report = {
        "schema": "radar.secondary-full-source-bounded-diagnostic.v1",
        "started_at_utc": started_at,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Actual OST GCR proton source, GEO 35786 km / 0 degrees, 2026 launch, five years "
            "and the existing explicit 60-month solar timeline. Only H is generated; its "
            "incomplete ion-roster warning remains. All source energies and ten retained "
            "TENDL products remain in scope. Bounded actual numerical execution, not a "
            "full-source accuracy or normative validation."
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "command_arguments": sys.argv[1:] if argv is None else argv,
        "file_sha256_at_start": start_hashes,
        "file_sha256_at_end": end_hashes,
        "input_hashes_consistent": True,
        "hash_validation_scope": (
            "Listed files match at start/end; reverted intermediate edits and unlisted "
            "dependencies are not certified."
        ),
        "calculation": calculation,
        "full_spectrum_convergence_claim": False,
        "normative_certification": False,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    print(
        f"Saved partial diagnostic: {args.output}; "
        f"responses={calculation['observed_response_calls']}; "
        f"status={calculation['parameters']['incident_quadrature_status']}",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
