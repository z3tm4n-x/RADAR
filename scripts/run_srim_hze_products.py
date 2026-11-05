"""Calculate opt-in SRIM energy/LET products from real OST external sources.

This command isolates the transport integration, not LEO/GEO/HEO orbit
acceptance. Library callers may pass actual on-orbit products to the same API.
No dose/SEE calculation or normative release is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from math import fsum, sqrt
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.core.constants import (  # noqa: E402
    DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
)
from radar.core.products import SpectrumProduct  # noqa: E402
from radar.core.project import MissionConfig, MethodologyConfig  # noqa: E402
from radar.core.types import AssessmentMode, Particle, SpectrumQuantity  # noqa: E402
from radar.gcr.model import GcrModelInput, OstGcrModel  # noqa: E402
from radar.pipelines.srim_hze import calculate_srim_hze_products  # noqa: E402
from radar.sep.model import OstSepModel, SepModelInput  # noqa: E402
from radar.shielding.srim_dataset import load_srim_hze_dataset  # noqa: E402
from radar.solar_activity.model import build_mission_solar_activity  # noqa: E402
from radar.solar_activity.ost import ost_wolf_number_cycle_table  # noqa: E402

SHIELDS = (0., .01, .1, 1., 3., 10.)
LET_EDGES = tuple(sorted({0., *(10 ** (-4 + 7 * i / 140) for i in range(141))}))
ENERGY_EDGES = tuple(10 ** (-3 + 8 * i / 160) for i in range(161))


def _refine(grid: tuple[float, ...], factor: int) -> tuple[float, ...]:
    if factor not in (1, 2):
        raise ValueError("Source grid refinement must be 1 or 2.")
    return grid if factor == 1 else tuple(sorted({*grid, *(sqrt(a*b) for a, b in zip(grid, grid[1:]))}))


def source_products(mission: MissionConfig, refinement: int) -> tuple[SpectrumProduct, ...]:
    policy = MethodologyConfig()
    solar = build_mission_solar_activity(
        mission, ost_wolf_number_cycle_table(mission.solar_activity_level),
        policy.solar_cycle_reference_start_year,
        reference_cycle_number=policy.solar_cycle_reference_number,
        wolf_number_version=policy.solar_activity_wolf_version,
        monthly_policy=policy.solar_activity_monthly_policy,
    )
    gcr = OstGcrModel(energy_grid_mev_per_nucleon=_refine(
        DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON, refinement,
    )).calculate(GcrModelInput(mission, solar))
    sep_grid = _refine(DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON, refinement)
    sep = OstSepModel(energy_grid_mev=sep_grid, hze_energy_grid_mev_per_nucleon=sep_grid,
                      monthly_smoothed_wolf_numbers=tuple(p.wolf_number for p in solar.months)
                      ).calculate(SepModelInput(mission))
    return (*gcr.products, *sep.products)


def _code_pins() -> dict[str, str]:
    files = set((ROOT / "src/radar").rglob("*.py"))
    files.update(p for p in (ROOT / "src/radar/data/normative").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts)
    files.add(Path(__file__))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def run(bundle: Path, sha: str, *, mission: MissionConfig, refinement: int,
        symbols: tuple[str, ...] | None = None) -> dict[str, Any]:
    before = _code_pins()
    dataset = load_srim_hze_dataset(bundle, expected_sha256=sha)
    print(f"Loaded {len(dataset.tables_by_z)} SRIM ion pairs; calculating OST sources...", flush=True)
    products = source_products(mission, refinement)
    selected, zero = [], []
    for product in products:
        source = product.spectrum
        if (source.particle is not Particle.HZE or source.assessment_mode not in tuple(AssessmentMode)
                or source.quantity not in (SpectrumQuantity.DIFFERENTIAL_FLUENCE,
                                           SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
                                           SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX)):
            continue
        assert source.species is not None
        if symbols is not None and source.species.symbol not in symbols:
            continue
        if not any(source.y):
            zero.append({"source": source.source.value, "mode": source.assessment_mode.value,
                         "symbol": source.species.symbol,
                         "reason": "exact_zero_on_the_declared_source_grid_only"})
            continue
        selected.append(product)
    present = {p.spectrum.species.symbol for p in selected if p.spectrum.species is not None}
    if symbols is not None and (not symbols or len(set(symbols)) != len(symbols)
                                or not set(symbols).issubset(present)):
        raise ValueError("Requested symbols must be unique positive source elements.")
    rows = []
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for index, product in enumerate(selected, 1):
        source = product.spectrum
        assert source.assessment_mode is not None and source.species is not None
        result = calculate_srim_hze_products(
            products=(product,), tables=dataset.tables_by_z, thicknesses_g_cm2=SHIELDS,
            assessment_mode=source.assessment_mode, let_edges=LET_EDGES,
            output_energy_edges=ENERGY_EDGES,
        )
        for case in result.cases:
            row: dict[str, Any] = {
                "source": case.source.value, "mode": source.assessment_mode.value,
                "symbol": source.species.symbol, "z": case.z, "thickness_g_cm2": case.thickness_g_cm2,
                "energy_product": None if case.energy_product is None else asdict(case.energy_product),
                "let_product": None if case.let_product is None else asdict(case.let_product),
                "primary_ledger": None if case.transport is None else asdict(case.transport.ledger),
                "let_ledger": None if case.let_result is None else asdict(case.let_result.ledger),
                "incomplete_reasons": case.incomplete_reasons,
                "normative_export_allowed": False,
            }
            rows.append(row)
            grouped[case.source.value, source.assessment_mode.value, case.thickness_g_cm2].append(row)
        if index % 12 == 0 or index == len(selected):
            print(f"Calculated {index}/{len(selected)} source/ion/mode inputs ({len(rows)} shield cases)", flush=True)
    summaries = []
    for (source_name, mode, shield), cases in sorted(grouped.items()):
        known = [r for r in cases if r["primary_ledger"] is not None and r["let_ledger"] is not None]
        primary = {key: fsum(r["primary_ledger"][key] for r in known) for key in (
            "source_integral", "supported_exit_integral", "represented_integral",
            "outside_al_domain_integral", "unresolved_low_residual_integral",
            "below_output_window_integral", "above_output_window_integral",
        )}
        let = {key: fsum(r["let_ledger"][key] for r in known) for key in (
            "mapped_integral", "outside_si_domain_integral", "below_let_window_integral",
            "above_let_window_integral",
        )}
        total = primary["source_integral"]
        summaries.append({"source": source_name, "mode": mode, "thickness_g_cm2": shield,
                          "case_count": len(cases), "unavailable_cases": len(cases) - len(known),
                          "primary": primary, "let": let,
                          "outside_al_fraction_of_declared_source": primary["outside_al_domain_integral"] / total
                          if total else None,
                          "unresolved_low_fraction_of_declared_source": primary["unresolved_low_residual_integral"] / total
                          if total else None})
    after = _code_pins()
    if before != after or hashlib.sha256(bundle.read_bytes()).hexdigest() != sha:
        raise ValueError("Implementation, normative inputs or bundle changed during calculation.")
    return {"schema_version": "radar.srim_hze_external_products.v1", "status": "INCOMPLETE",
            "scope": "OST_external_source_transport_integration_not_orbit_acceptance",
            "mission": asdict(mission), "bundle_sha256": sha, "source_grid_refinement": refinement,
            "requested_symbols": symbols, "source_product_count": len(selected), "case_count": len(rows),
            "zero_sampled_sources_not_transported": zero, "summaries": summaries, "cases": rows,
            "normative_export_allowed": False, "orbit_calculation_performed": False,
            "dose_or_see_calculated": False, "default_profile_replaced": False,
            "implementation_and_input_sha256_before": before,
            "implementation_and_input_sha256_after": after,
            "limitations": [
                "Source integrals cover declared finite sampled intervals, not unmodelled tails beyond them.",
                "Outside-Al and unresolved-low fractions are source-count fractions, not dose/SEE error bounds.",
                "At zero shield Al is an identity; missing Si support is reported separately in the LET ledger.",
                "Output-bin subdivision is not an independent refinement of physical SRIM stopping data.",
            ]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch-year", type=int, default=2026)
    parser.add_argument("--lifetime-years", type=int, default=5)
    parser.add_argument("--sep-probability", type=float, default=.1)
    parser.add_argument("--source-grid-refinement", type=int, choices=(1, 2), default=1)
    parser.add_argument("--symbols", help="Optional comma-separated positive-source elements; default all")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.bundle, args.bundle_sha256, mission=MissionConfig(
        launch_year=args.launch_year, lifetime_years=args.lifetime_years,
        sep_exceedance_probability=args.sep_probability,
    ), refinement=args.source_grid_refinement,
        symbols=tuple(args.symbols.split(",")) if args.symbols else None)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "cases": result["case_count"], "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
