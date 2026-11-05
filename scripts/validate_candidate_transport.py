"""Replay a bounded two-grid primary HZE/electronic-LET experiment, not orbit acceptance."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from math import fsum
from pathlib import Path
from typing import Any

from compare_parameterized_stopping_grids import _read_verified
from validate_parameterized_stopping import run_validator, write_report_exclusively
from validate_ion_stopping_candidate import _resource_hashes

from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import AssessmentMode, Particle, SpectrumCompleteness
from radar.gcr.components import OstGcrSolarState
from radar.gcr.products import GcrFluxTimeBin, calculate_gcr_mission_products
from radar.gcr.source_spectra import calculate_ost_gcr_source_spectra_outside_magnetosphere
from radar.sep.model import OstSepModel, SepModelInput
from radar.validation.candidate_let import convert_candidate_electronic_let
from radar.validation.candidate_transport import (
    CandidateIonTable, MASS_POLICIES, transport_candidate_primary,
)
from radar.validation.parameterized_stopping import ParameterizedStoppingRow


ROOT = Path(__file__).resolve().parents[1]
SHIELDS = (0.0, 0.01, 0.1, 1.0, 3.0, 10.0)
THRESHOLDS = (0.01, 0.1, 1.0, 10.0, 50.0, 100.0)
ENERGY_EDGES = tuple(10 ** (-2 + 7 * i / 140) for i in range(141))
LET_EDGES = tuple(sorted({*(10 ** (-4 + 7 * i / 140) for i in range(141)), *THRESHOLDS}))
IMPLEMENTATION_PATHS = (
    "scripts/validate_candidate_transport.py",
    "scripts/compare_parameterized_stopping_grids.py",
    "scripts/validate_parameterized_stopping.py",
    "scripts/validate_ion_stopping_candidate.py",
    "scripts/build_stopping_mass_manifest.py",
    "src/radar/validation/candidate_transport.py",
    "src/radar/validation/candidate_let.py",
    "src/radar/shielding/primary_let.py",
    "src/radar/validation/parameterized_stopping.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_inputs(projectiles: list[dict[str, Any]]) -> tuple[Spectrum1D, ...]:
    """Fixed source grid/state isolates table interpolation, not mission/orbit sampling."""
    gcr = calculate_ost_gcr_source_spectra_outside_magnetosphere(
        energy_grid_mev_per_nucleon=tuple(10 ** (i / 200) for i in range(1001)),
        solar_state=OstGcrSolarState(
            wolf_current=60, wolf_lagged=50, wolf_min=10, wolf_max=100,
            cycle_number=24, after_polarity_reversal=True,
        ), nek_wolf_number=60, akl_years_from_cycle_start=2,
        symbols=tuple(p["symbol"] for p in projectiles),
    )
    sources: list[Spectrum1D] = []
    for ion in gcr.ions:
        products = calculate_gcr_mission_products((GcrFluxTimeBin(ion.total, 5 * 365.25 * 86400),))
        sources.extend((products.maximum_flux.spectrum, products.mission_fluence.spectrum))
    sep_grid = tuple(10 ** (i / 200) for i in range(821))
    sep = OstSepModel(
        energy_grid_mev=sep_grid, hze_energy_grid_mev_per_nucleon=sep_grid,
        monthly_smoothed_wolf_numbers=(60.0,) * 60,
    ).calculate(SepModelInput(MissionConfig(
        launch_year=2026, lifetime_years=5, sep_exceedance_probability=0.1,
    )))
    selected = {p["z"] for p in projectiles}
    for product in sep.products:
        spectrum = product.spectrum
        if (spectrum.particle is Particle.HZE and spectrum.species is not None
                and spectrum.species.atomic_number in selected
                and spectrum.assessment_mode in (AssessmentMode.PEAK, AssessmentMode.CUMULATIVE)
                and any(spectrum.y)):
            sources.append(spectrum)
    expected = {(membership.lower(), p["z"], mode.value)
                for p in projectiles for membership in p["source_membership"]
                for mode in AssessmentMode}
    actual = {(s.source.value, s.species.atomic_number, s.assessment_mode.value)
              for s in sources if s.species is not None and s.assessment_mode is not None}
    if len(sources) != len(expected) or actual != expected:
        raise ValueError("Missing, duplicate or unexpected positive HZE source/mode.")
    return tuple(sources)


def _case(source: Spectrum1D, table: CandidateIonTable, thickness: float) -> dict[str, Any]:
    transport = transport_candidate_primary(
        source, table, thickness, source_mass_parameter_u=table.mass_parameter_u,
        output_edges=None if thickness == 0 else ENERGY_EDGES,
    )
    let = convert_candidate_electronic_let(transport, let_edges=LET_EDGES)
    if any(s.completeness is not SpectrumCompleteness.INCOMPLETE
           for s in (transport.spectrum, let.spectrum)):
        raise ValueError("Diagnostic candidate must remain INCOMPLETE.")
    if any(s.production_contract_violations() != (
        "incomplete spectra cannot be released as production results",
    ) for s in (transport.spectrum, let.spectrum)):
        raise ValueError("Candidate output lost its typed physical contract.")
    energy_counts = [transport.integral_on_exit(a, b) for a, b in zip(ENERGY_EDGES, ENERGY_EDGES[1:])]
    let_counts = [y * (b - a) for y, a, b in zip(let.spectrum.y, LET_EDGES, LET_EDGES[1:])]
    assert source.species is not None and source.assessment_mode is not None
    return {
        "key": [source.source.value, source.species.atomic_number,
                source.assessment_mode.value, thickness],
        "symbol": source.species.symbol, "mass_parameter_u": table.mass_parameter_u,
        "primary_ledger": asdict(transport.ledger), "let_ledger": asdict(let.ledger),
        "exit_support": transport.exit_support, "energy_bin_integrals": energy_counts,
        "let_bin_integrals": let_counts,
        "let_tail_integrals": {str(t): fsum(let_counts[LET_EDGES.index(t):]) for t in THRESHOLDS},
        "energy_y_unit": transport.spectrum.y_unit.value, "let_y_unit": let.spectrum.y_unit.value,
        "incomplete_reasons": list(let.spectrum.incomplete_reasons),
        "zero_shield_identity": (transport.spectrum.x == source.x and transport.spectrum.y == source.y
                                 and transport.spectrum.bin_edges == source.bin_edges
                                 and transport.spectrum.ordinate_basis == source.ordinate_basis
                                 if thickness == 0 else None),
    }


def _relative(left: float, right: float) -> float | None:
    return abs(right - left) / left if left > 0 else (0.0 if right == 0 else None)


def _compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if any(left[key] != right[key] for key in (
        "key", "mass_parameter_u", "energy_y_unit", "let_y_unit",
    )):
        raise ValueError("Unmatched source/mode/shield, mass or units comparison.")
    checks = {
        "supported_primary": _relative(left["primary_ledger"]["supported_exit_integral"],
                                       right["primary_ledger"]["supported_exit_integral"]),
        "mapped_let": _relative(left["let_ledger"]["mapped_integral"],
                               right["let_ledger"]["mapped_integral"]),
        **{f"let_above_{t}": _relative(v, right["let_tail_integrals"][t])
           for t, v in left["let_tail_integrals"].items()},
    }
    histogram_checks = {}
    for axis in ("energy", "let"):
        a, b = left[f"{axis}_bin_integrals"], right[f"{axis}_bin_integrals"]
        difference = fsum(abs(x - y) for x, y in zip(a, b, strict=True))
        histogram_checks[axis] = (difference / fsum(a) if fsum(a) > 0
                                 else (0.0 if difference == 0 else None))
    return {"key": left["key"], "relative_integral_changes": checks,
            "histogram_l1_over_coarse_integral": histogram_checks,
            "selected_integral_grid_check_passed": all(v is not None and v < 0.01
                                                       for v in checks.values())}


def run_experiment(
    coarse_path: Path, dense_path: Path, *, symbols: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    before = {p: _sha(ROOT / p) for p in IMPLEMENTATION_PATHS}
    validations = [run_validator(p) for p in (coarse_path, dense_path)]
    raw = [_read_verified(p, v["input_sha256_after"])
           for p, v in zip((coarse_path, dense_path), validations, strict=True)]
    for key in ("manifest", "manifest_sha256", "source_pins", "energy_coordinate_compatibility",
                "let_convention", "projectile_charge_input", "range_origin_energy_mev_per_nucleon"):
        if raw[0][key] != raw[1][key]:
            raise ValueError(f"Candidates differ in {key}.")
    coarse_e, dense_e = [r["energies_mev_per_nucleon"] for r in raw]
    if len(coarse_e) >= len(dense_e) or not set(coarse_e).issubset(dense_e):
        raise ValueError("A strictly denser nested grid is required.")
    policy = MASS_POLICIES[0]
    for key in ("projectiles", "backend_configuration", "nonradiative_configuration", "materials"):
        if raw[0]["datasets"][policy][key] != raw[1]["datasets"][policy][key]:
            raise ValueError(f"Source-policy candidates differ in {key}.")
    inventory = raw[0]["datasets"][policy]["projectiles"]
    projectiles = inventory
    if symbols is not None:
        if (not symbols or len(set(symbols)) != len(symbols)
                or not set(symbols).issubset(p["symbol"] for p in inventory)):
            raise ValueError("Requested symbols must be unique positive source entries.")
        projectiles = [p for p in inventory if p["symbol"] in symbols]
    sources = _source_inputs(projectiles)
    selected_z = {p["z"] for p in projectiles}
    cases = []
    for label, payload, validation in zip(("coarse", "dense"), raw, validations, strict=True):
        grouped: dict[int, list[ParameterizedStoppingRow]] = {z: [] for z in selected_z}
        for row in payload["datasets"][policy]["rows"]:
            if row["z"] in selected_z:
                grouped[row["z"]].append(ParameterizedStoppingRow(**row))
        ref = (f"diagnostic://parameterized-stopping/sha256/{validation['input_sha256_after']}"
               f"?manifest_sha256={payload['manifest_sha256']}&mass_policy={policy}")
        tables = {z: CandidateIonTable.from_rows(rows, mass_policy=policy, dataset_ref=ref)
                  for z, rows in grouped.items()}
        current = []
        for i, source in enumerate(sources):
            assert source.species is not None and source.species.atomic_number is not None
            for thickness in SHIELDS:
                current.append(_case(source, tables[source.species.atomic_number], thickness))
            if (i + 1) % 24 == 0:
                print(f"{label}: {i + 1}/{len(sources)} source-mode spectra checked", flush=True)
        cases.append(current)
    comparisons = [_compare(a, b) for a, b in zip(*cases, strict=True)]
    for p, validation in zip((coarse_path, dense_path), validations, strict=True):
        if _sha(p) != validation["input_sha256_after"]:
            raise ValueError("Candidate changed during transport experiment.")
        for hashes, base in ((validation["production_resources_sha256_after"], ROOT),
                             (validation["generator_hashes_after"], ROOT),
                             (validation["source_pins_after"], ROOT / "research/ion_stopping_20261031")):
            _resource_hashes(hashes, base)
    for payload in raw:
        if _sha(Path(payload["manifest_path"])) != payload["manifest_sha256"]:
            raise ValueError("Candidate manifest changed during transport experiment.")
    after = {p: _sha(ROOT / p) for p in IMPLEMENTATION_PATHS}
    if before != after:
        raise ValueError("Transport/validation implementation changed during replay.")
    finite_changes = [(value, c["key"], metric) for c in comparisons
                      for metric, value in c["relative_integral_changes"].items() if value is not None]
    return {
        "schema_version": "radar.candidate_primary_electronic_let.experiment.v1",
        "scope": "fixed_external_source_state_and_grid_no_orbit_no_physical_qualification",
        "candidate_paths": [str(p.resolve()) for p in (coarse_path, dense_path)],
        "candidate_sha256": [v["input_sha256_after"] for v in validations],
        "manifest_sha256": raw[0]["manifest_sha256"], "mass_policy": policy,
        "implementation_sha256_before": before, "implementation_sha256_after": after,
        "production_resources_sha256": validations[0]["production_resources_sha256_after"],
        "selected_projectiles": projectiles, "full_positive_inventory_selected": symbols is None,
        "source_mode_count": len(sources), "case_count_per_grid": len(cases[0]),
        "source_fingerprints": [{"source": s.source.value,
            "species": asdict(s.species) if s.species is not None else None,
            "mode": s.assessment_mode, "sha256": hashlib.sha256(json.dumps(asdict(s), sort_keys=True,
                ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
            "energy_domain": [s.x[0], s.x[-1]], "point_count": len(s.x),
            "incomplete_reasons": s.incomplete_reasons} for s in sources],
        "source_policy": {
            "GCR": "fixed W=60,lag=50,min=10,max=100,cycle24,after_reversal,AKL_year=2; single 5yr bin",
            "SEP": "2026,5yr,p=0.1,explicit monthly W=60 for all60 months; existing peak/fluence models",
            "source_density_interpolation": "piecewise_linear_between_declared_samples_no_tails",
            "source_grid_refined_here": False, "table_interpolation": "piecewise_linear",
            "mission_time_factor_in_adapter": "none",
        },
        "shield_g_cm2": SHIELDS, "energy_edges": ENERGY_EDGES, "let_edges": LET_EDGES,
        "coarse_cases": cases[0], "dense_cases": cases[1], "comparisons": comparisons,
        "maximum_selected_relative_integral_change": max(finite_changes, default=None),
        "failed_selected_integral_case_count": sum(
            not c["selected_integral_grid_check_passed"] for c in comparisons),
        "selected_integral_grid_check_passed": all(c["selected_integral_grid_check_passed"]
                                                   for c in comparisons),
        "all_zero_shield_identities_passed": all(c["zero_shield_identity"] is not False
                                                  for grid in cases for c in grid),
        "numerical_table_consistency_passed": all(v["numerical_consistency_passed"] for v in validations),
        "stage4_acceptance_matrix_completed": False, "physical_accuracy_verified": False,
        "normative_export_allowed": False, "production_tables_replaced": False,
        "limitations": [
            "No orbit, geomagnetic attenuation or mission solar-time validation is performed.",
            "Counts beyond declared source support are unknown, not extrapolated or zero-proven.",
            "No physical low-energy stopping, fragmentation or mass-policy qualification follows.",
            "Only exported stopping-table grid is refined, not the native backend or source grid.",
            "Integral and histogram metrics do not establish pointwise or uniform convergence.",
            "LET tail integrals stop at the declared LET maximum; outside-window ledger is separate.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("dense", type=Path)
    parser.add_argument("--symbols", help="Optional comma-separated subset; omitted means all positive ions")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_experiment(args.coarse, args.dense,
                            symbols=tuple(args.symbols.split(","))
                            if args.symbols is not None else None)
    write_report_exclusively(report, args.output)
    print(json.dumps({k: report[k] for k in ("case_count_per_grid",
        "failed_selected_integral_case_count", "maximum_selected_relative_integral_change",
        "all_zero_shield_identities_passed", "normative_export_allowed")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
