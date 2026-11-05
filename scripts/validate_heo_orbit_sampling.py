"""Bounded orbit-window, temporal-density and MLT diagnostics for Appendix Zh.

The source products are calculated once for a declared 2026 five-year mission,
then held fixed while only the access quadrature changes. No shielding, LET,
dose, SEE or normative convergence/release qualification is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from radar.core.products import SpectrumProduct
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.core.types import OrbitType, Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.gcr.model import GcrModelInput, OstGcrModel
from radar.gcr.penetration import gcr_product_mass_to_charge
from radar.geomagnetic.ost import EARTH_RADIUS_KM
from radar.geomagnetic.ost_orbit import (
    OST_GRAVITATIONAL_PARAMETER_KM3_S2, OstOrbitTrace, sample_ost_orbit,
)
from radar.geomagnetic.ost_penetration import (
    OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
    build_ost_penetration_function_from_samples,
)
from radar.geomagnetic.rigidity import RigidityGrid
from radar.physics.rigidity import ion_kinetic_energy_per_nucleon_to_rigidity_gv
from radar.pipelines.execution import build_pipeline_source_models
from radar.pipelines.sep_source_products import (
    _hze_ion_record_for_product, _load_ost_sep_hze_ion_records_for_penetration,
)
from radar.sep.model import SepModelInput
from radar.solar_activity.model import build_calculation_solar_activity


_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_RESOURCE_PATHS = (
    "src/radar/data/normative/index.json",
    "src/radar/data/normative/integrity_manifest.json",
    "src/radar/data/normative/ost_134_1044_2007/manifest.json",
    "src/radar/data/normative/ost_134_1044_2007/table_g_1_wolf_numbers.csv",
    "src/radar/data/normative/sep_protons/ost_134_1044_2007_coefficients.csv",
    "src/radar/data/normative/sep_protons/ost_134_1044_2007_metadata.json",
    "src/radar/data/normative/sep_hze/ost_134_1044_2007_b9.csv",
)
_CRITICAL_INCLINATION_DEG = math.degrees(math.acos(1.0 / math.sqrt(5.0)))
_PRIMARY_DENSITIES = (60, 120, 240, 480)
_EXTENSION_DAYS = (16, 32, 64, 128, 256)
_MAX_SAMPLES = 400000
_MAX_ORBIT_MLT_STATES = 12000000
_TINY_POSITIVE_DAYS = 1e-9


@dataclass(frozen=True)
class VariantSpec:
    name: str
    orbit: OrbitConfig
    min_days: float = 16.0
    min_orbits: int = 10
    steps_per_orbit: int = 240
    mlt_sample_count: int = 24
    raan_deg: float = 0.0


def _orbit(perigee: float, apogee: float, inclination: float) -> OrbitConfig:
    return OrbitConfig(
        orbit_type=OrbitType.CIRCULAR if perigee == apogee else OrbitType.ELLIPTIC,
        perigee_altitude_km=perigee, apogee_altitude_km=apogee,
        inclination_deg=inclination, argument_of_perigee_deg=0.0,
    )


def _specifications(scope: str) -> tuple[VariantSpec, ...]:
    heo = _orbit(500.0, 39700.0, 63.4)
    primary = (
        *(VariantSpec(f"heo_16d_n{steps}_m24", heo, steps_per_orbit=steps)
          for steps in _PRIMARY_DENSITIES),
        *(VariantSpec(f"heo_{orbits}orbits_n480_m24", heo,
                      min_days=_TINY_POSITIVE_DAYS, min_orbits=orbits, steps_per_orbit=480)
          for orbits in (10, 20, 40)),
        VariantSpec("heo_32d_n480_m24", heo, min_days=32.0, steps_per_orbit=480),
        *(VariantSpec(f"heo_16d_n480_m{mlt}", heo,
                      steps_per_orbit=480, mlt_sample_count=mlt) for mlt in (48, 96)),
        *(VariantSpec(f"heo_16d_n480_raan{raan}", heo,
                      steps_per_orbit=480, raan_deg=float(raan)) for raan in (90, 180, 270)),
    )
    critical = tuple(
        VariantSpec(f"heo_critical_{label}", _orbit(
            500.0, 39700.0, _CRITICAL_INCLINATION_DEG + offset,
        )) for label, offset in (("minus", -1e-6), ("exact", 0.0), ("plus", 1e-6))
    )
    smoke = (
        VariantSpec("leo_smoke", _orbit(400.0, 400.0, 51.6), steps_per_orbit=120),
        VariantSpec("geo_smoke", _orbit(35786.0, 35786.0, 0.0), steps_per_orbit=120),
    )
    window_extension = tuple(
        VariantSpec(f"heo_extension_{days}d_n240_raan{raan}", heo,
                    min_days=float(days), steps_per_orbit=240, raan_deg=float(raan))
        for raan in (0, 90) for days in _EXTENSION_DAYS
    )
    groups = {"heo": primary, "critical": critical, "smoke": smoke,
              "all": (*primary, *critical, *smoke), "window_extension": window_extension}
    if scope not in groups:
        raise ValueError(f"Unknown sampling diagnostic scope: {scope}")
    return groups[scope]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_hashes() -> dict[str, str]:
    # Include lazy source-model dependencies, not only the modules imported by
    # this runner before the first calculation. No external user files are read.
    source_root = (_ROOT / "src/radar").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "radar" or name.startswith("radar."):
            filename = getattr(module, "__file__", None)
            if filename is not None and not Path(filename).resolve().is_relative_to(source_root):
                raise ValueError(f"Loaded RADAR module is outside this worktree: {name}")
    paths = sorted(source_root.rglob("*.py"))
    paths.append(Path(__file__).resolve())
    return {path.relative_to(_ROOT).as_posix(): _sha(path) for path in paths}


def _input_hashes() -> dict[str, str]:
    return {name: _sha(_ROOT / name) for name in _SOURCE_RESOURCE_PATHS}


def _source_products() -> tuple[tuple[SpectrumProduct, ...], dict[str, Any]]:
    config = CalculationConfig(
        mission=MissionConfig(launch_year=2026, lifetime_years=5),
        orbit=_orbit(500.0, 39700.0, 63.4),
    )
    solar = build_calculation_solar_activity(config)
    model = OstGcrModel(symbols=("H", "He", "Fe"), solar_activity=solar)
    result = model.calculate(GcrModelInput(mission=config.mission, solar_activity=solar))
    sep_model = build_pipeline_source_models(config).sep_model
    sep_result = sep_model.calculate(SepModelInput(mission=config.mission))
    kinds = (RadiationProductKind.MISSION_FLUENCE, RadiationProductKind.MAXIMUM_FLUX,
             RadiationProductKind.PEAK_FLUX)
    quantities = (SpectrumQuantity.DIFFERENTIAL_FLUENCE, SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
                  SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX)
    products = tuple(product for product in (*result.products, *sep_result.products)
                     if product.kind in kinds and product.spectrum.quantity in quantities
                     and product.spectrum.species is not None
                     and product.spectrum.species.atomic_number in (1, 2, 26))
    if len(products) != 12 or {p.kind for p in products} != set(kinds):
        raise ValueError("Diagnostic requires GCR/SEP proton, He and Fe in both source modes.")
    keys = tuple(_product_key(product) for product in products)
    if len(set(keys)) != 12:
        raise ValueError("Diagnostic source product identities must be unique.")
    for product in products:
        product.validate_calculation_contract()
    return products, {
        "calculation_config": asdict(config),
        "solar_activity": asdict(solar),
        "model": result.model,
        "document": result.document,
        "method_metadata": result.method_metadata,
        "sep_model": sep_result.model,
        "sep_model_version": sep_model.metadata.version,
        "sep_document": sep_result.document,
        "sep_exceedance_probability": sep_result.exceedance_probability,
        "sep_full_source_products_calculated": len(sep_result.products),
        "selected_species_after_source_calculation": "proton_He_Fe_only",
        "source_location": "outside_magnetosphere",
        "source_products_evaluated_once": True,
        "orbit_window_changes_do_not_recompute_source_time_products": True,
    }


def _product_key(product: SpectrumProduct) -> str:
    species = product.spectrum.species
    if species is None or species.is_aggregate:
        raise ValueError("Source-weighted diagnostic requires explicit single-ion identity.")
    return f"{product.spectrum.source.value}:{species.symbol}:{product.kind.value}"


def _mass_to_charge(product: SpectrumProduct) -> float:
    if product.spectrum.source is RadiationSource.GCR:
        return gcr_product_mass_to_charge(product)
    if product.spectrum.source is RadiationSource.SEP:
        if product.spectrum.particle is Particle.PROTON:
            return 1.0
        return _hze_ion_record_for_product(
            product, _load_ost_sep_hze_ion_records_for_penetration(),
        ).mass_to_charge
    raise ValueError("Source-weighted diagnostic supports only GCR and SEP.")


def _source_rigidities(product: SpectrumProduct) -> tuple[float, ...]:
    species = product.spectrum.species
    if species is None or species.mass_number is None or species.atomic_number is None:
        raise ValueError("Source-weighted diagnostic requires A and Z.")
    mass_to_charge = _mass_to_charge(product)
    return tuple(ion_kinetic_energy_per_nucleon_to_rigidity_gv(
        energy, mass_to_charge,
    ) for energy in product.spectrum.x)


def _rigidity_grid(products: tuple[SpectrumProduct, ...]) -> RigidityGrid:
    diagnostic_nodes = tuple(10.0 ** (-4.0 + 6.0 * i / 400) for i in range(401))
    exact_source_nodes = tuple(r for product in products for r in _source_rigidities(product))
    return RigidityGrid(tuple(sorted(set((*diagnostic_nodes, *exact_source_nodes)))))


def _finite_trapezoid(x: tuple[float, ...], y: tuple[float, ...]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Finite integral requires matching arrays with at least two nodes.")
    if any(not math.isfinite(v) or v < 0.0 for v in (*x, *y)):
        raise ValueError("Finite diagnostic arrays must be non-negative and finite.")
    if any(right <= left for left, right in zip(x, x[1:])):
        raise ValueError("Finite diagnostic integration grid must increase strictly.")
    value = math.fsum((right - left) * (yl + yr) / 2.0
                      for left, right, yl, yr in zip(x, x[1:], y, y[1:]))
    if not math.isfinite(value):
        raise ValueError("Finite diagnostic integral overflowed.")
    return value


def _relative_difference(value: float, reference: float) -> dict[str, Any]:
    if any(not math.isfinite(item) or item < 0.0 for item in (value, reference)):
        raise ValueError("Diagnostic comparisons require non-negative finite values.")
    difference = abs(value - reference)
    relative = difference / reference if reference > 0.0 else None
    if relative is not None and not math.isfinite(relative):
        raise ValueError("Relative diagnostic comparison overflowed.")
    return {
        "value": value, "reference": reference, "absolute_change": difference,
        "relative_change": relative,
        "relative_change_status": "defined" if reference > 0.0 else "zero_reference",
        "below_one_percent_for_this_fixed_metric_only": (
            relative < 0.01 if relative is not None else None
        ),
    }


def _legacy_window_estimate(trace: OstOrbitTrace) -> dict[str, Any]:
    """Analytical counterfactual only: never allocate the old long orbit trace."""
    rate = abs(trace.perigee_precession_rad_s)
    period = 2.0 * math.pi / rate if rate > 0.0 else None
    window = 16.0 * 86400.0
    if trace.eccentricity > 1e-4 and period is not None:
        window = max(window, 2.5 * period)
    step = max(trace.orbital_period_s / 60.0, window / _MAX_SAMPLES)
    return {
        "role": "analytical_counterfactual_not_sampled",
        "minimum_days": 16.0, "requested_steps_per_orbit": 60,
        "sample_limit": _MAX_SAMPLES,
        "perigee_drift_period_s": period,
        "legacy_window_s": window,
        "legacy_step_s": step,
        "legacy_step_over_orbital_period": step / trace.orbital_period_s,
        "legacy_sample_cap_would_coarsen": step > trace.orbital_period_s / 60.0,
    }


def _preflight(specs: tuple[VariantSpec, ...]) -> dict[str, Any]:
    rows = []
    states = 0
    for spec in specs:
        a = EARTH_RADIUS_KM + (
            spec.orbit.perigee_altitude_km + spec.orbit.apogee_altitude_km
        ) / 2.0
        period = 2.0 * math.pi * math.sqrt(a**3 / OST_GRAVITATIONAL_PARAMETER_KM3_S2)
        count = max(spec.min_orbits * spec.steps_per_orbit,
                    math.ceil(spec.min_days * 86400.0 / (period / spec.steps_per_orbit)))
        if count > _MAX_SAMPLES:
            raise ValueError(f"Diagnostic orbit budget exceeded by {spec.name}: {count}")
        rows.append({"variant": spec.name, "samples": count,
                     "orbit_mlt_states": count * spec.mlt_sample_count})
        states += count * spec.mlt_sample_count
    if states > _MAX_ORBIT_MLT_STATES:
        raise ValueError(f"Diagnostic state budget exceeded: {states}")
    return {"variant_count": len(specs), "total_orbit_mlt_states": states,
            "orbit_mlt_state_limit": _MAX_ORBIT_MLT_STATES,
            "max_samples_per_variant": _MAX_SAMPLES,
            "legacy_orbit_traces_allocated": 0, "variants": rows}


def _variant(
    spec: VariantSpec, grid: RigidityGrid, products: tuple[SpectrumProduct, ...],
) -> dict[str, Any]:
    trace = sample_ost_orbit(
        spec.orbit, raan_deg=spec.raan_deg, min_days=spec.min_days, min_orbits=spec.min_orbits,
        steps_per_orbit=spec.steps_per_orbit, max_samples=_MAX_SAMPLES,
    )
    if trace.step_s > trace.orbital_period_s / spec.steps_per_orbit * (1.0 + 1e-12):
        raise ValueError("Sampling diagnostic refuses silently coarsened orbit traces.")
    if not math.isclose(trace.window_s, len(trace.samples) * trace.step_s, rel_tol=1e-12):
        raise ValueError("Equal quadrature weights require a whole-cell sampling window.")
    operator = build_ost_penetration_function_from_samples(
        samples=trace.samples, rigidity_grid=grid, kp=3, apply_disturbance=True,
        mlt_sample_count=spec.mlt_sample_count,
    )
    value_by_rigidity = dict(zip(grid.values_gv, operator.values, strict=True))
    integrals: dict[str, Any] = {}
    for product in products:
        spectrum = product.spectrum
        factors = tuple(value_by_rigidity[r] for r in _source_rigidities(product))
        penetrated = tuple(y * factor for y, factor in zip(spectrum.y, factors, strict=True))
        integral = _finite_trapezoid(spectrum.x, penetrated)
        integrals[_product_key(product)] = {
            "value": integral,
            "unit": ("cm^-2" if product.kind is RadiationProductKind.MISSION_FLUENCE
                     else "cm^-2 s^-1"),
            "source_quantity": spectrum.quantity.value,
            "source_differential_unit": spectrum.y_unit.value,
            "source_angular_basis": spectrum.angular_basis.value if spectrum.angular_basis else None,
            "assessment_mode": spectrum.assessment_mode.value if spectrum.assessment_mode else None,
            "temporal_basis": spectrum.temporal_basis.value if spectrum.temporal_basis else None,
            "mission_duration_s": spectrum.mission_duration_s,
            "time_multiplier": "none",
            "energy_lower_bound": spectrum.x[0], "energy_upper_bound": spectrum.x[-1],
            "energy_unit": spectrum.x_unit.value,
            "source_completeness": spectrum.completeness.value if spectrum.completeness else None,
            "source_incomplete_reasons": spectrum.incomplete_reasons,
            "rigidity_mass_to_charge_used": _mass_to_charge(product),
            "rigidity_mass_to_charge_convention": (
                "SEP_B9_A_over_Q_production_helper" if spectrum.source is RadiationSource.SEP
                else "GCR_parameter_A_over_Z_production_helper"
            ),
            "source_integral_same_grid": _finite_trapezoid(spectrum.x, spectrum.y),
        }
    return {
        "name": spec.name, "requested": asdict(spec),
        "sample_count": len(trace.samples), "sample_limit": _MAX_SAMPLES,
        "window_s": trace.window_s, "step_s": trace.step_s,
        "requested_window_s": trace.requested_window_s,
        "minimum_orbits": trace.minimum_orbits,
        "sampling_policy": trace.sampling_policy,
        "orbital_period_s": trace.orbital_period_s,
        "step_over_orbital_period": trace.step_s / trace.orbital_period_s,
        "node_precession_rad_s": trace.node_precession_rad_s,
        "perigee_precession_rad_s": trace.perigee_precession_rad_s,
        "operator_version": operator.model_version,
        "operator_parameters": operator.parameters,
        "operator_incomplete_reasons": operator.incomplete_reasons,
        "penetration": operator.values, "source_weighted_integrals": integrals,
        "legacy_window_estimate": _legacy_window_estimate(trace),
    }


def _pair(left: dict[str, Any], right: dict[str, Any], axis: str) -> dict[str, Any]:
    differences = tuple(b - a for a, b in zip(left["penetration"], right["penetration"], strict=True))
    source_keys = left["source_weighted_integrals"].keys()
    if source_keys != right["source_weighted_integrals"].keys():
        raise ValueError("Comparison source identities do not match.")
    actual_window_equal = left["window_s"] == right["window_s"]
    return {
        "axis": axis, "left": left["name"], "right": right["name"],
        "left_requested_window_s": left["requested_window_s"],
        "right_requested_window_s": right["requested_window_s"],
        "left_actual_window_s": left["window_s"],
        "right_actual_window_s": right["window_s"],
        "actual_window_change_s": right["window_s"] - left["window_s"],
        "actual_window_equal": actual_window_equal,
        "density_comparison_includes_rounding_window_change": (
            axis.startswith("orbital_density_") and not actual_window_equal
        ),
        "max_absolute_psi_change": max(abs(value) for value in differences),
        "psi_change_is_dimensionless_not_relative_flux_error": True,
        "source_weighted_integral_changes": {
            key: _relative_difference(right["source_weighted_integrals"][key]["value"],
                                      left["source_weighted_integrals"][key]["value"])
            for key in source_keys
        },
        "reference_denominator": "left_variant",
        "stage4_acceptance_claim": False,
    }


def _comparisons(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {variant["name"]: variant for variant in variants}
    chains = (
        ("orbital_density_at_fixed_requested_window_and_mlt",
         tuple(f"heo_16d_n{n}_m24" for n in _PRIMARY_DENSITIES)),
        ("window_length_at_fixed_orbit_density_and_mlt",
         tuple(f"heo_{n}orbits_n480_m24" for n in (10, 20, 40))),
        ("window_length_at_fixed_orbit_density_and_mlt",
         ("heo_16d_n480_m24", "heo_32d_n480_m24")),
        ("mlt_density_at_fixed_window_and_orbit_density",
         tuple(f"heo_16d_n480_m{n}" for n in (24, 48, 96))),
        ("inclination_continuity_not_grid_convergence",
         tuple(f"heo_critical_{label}" for label in ("minus", "exact", "plus"))),
    )
    pairs = [_pair(by_name[left], by_name[right], axis)
             for axis, names in chains for left, right in zip(names, names[1:])
             if left in by_name and right in by_name]
    base = "heo_16d_n480_m24"
    for raan in (90, 180, 270):
        alternative = f"heo_16d_n480_raan{raan}"
        if base in by_name and alternative in by_name:
            pairs.append(_pair(by_name[base], by_name[alternative],
                               "initial_raan_sensitivity_not_grid_convergence"))
    for raan in (0, 90):
        names = tuple(f"heo_extension_{days}d_n240_raan{raan}" for days in _EXTENSION_DAYS)
        for left, right in zip(names, names[1:]):
            if left in by_name and right in by_name:
                pairs.append(_pair(
                    by_name[left], by_name[right],
                    "phase_coverage_window_extension_not_mission_qualification",
                ))
    for days in _EXTENSION_DAYS:
        left = f"heo_extension_{days}d_n240_raan0"
        right = f"heo_extension_{days}d_n240_raan90"
        if left in by_name and right in by_name:
            pairs.append(_pair(
                by_name[left], by_name[right],
                "phase_coverage_initial_raan_sensitivity_not_mission_qualification",
            ))
    return pairs


def build_report(scope: str = "all") -> dict[str, Any]:
    specs = _specifications(scope)
    preflight = _preflight(specs)
    models_before, inputs_before = _model_hashes(), _input_hashes()
    products, source_scenario = _source_products()
    source_payload = [asdict(product) for product in products]
    source_hash = hashlib.sha256(json.dumps(
        source_payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    grid = _rigidity_grid(products)
    variants = [_variant(spec, grid, products) for spec in specs]
    models_after, inputs_after = _model_hashes(), _input_hashes()
    if models_before != models_after or inputs_before != inputs_after:
        raise ValueError("Source data or implementation changed during diagnostics; discard run.")
    return {
        "schema": "radar_heo_orbit_sampling_diagnostic_v1", "scope": scope,
        "normative_release_eligible": False, "stage4_spectral_convergence_established": False,
        "source_roster_complete": False, "parameters_fitted": False,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "production_operator_version": OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
        "preflight": preflight,
        "model_hash_scope": "all_src_radar_python_files_and_this_runner",
        "model_sha256_before": models_before, "model_sha256_after": models_after,
        "input_sha256_before": inputs_before, "input_sha256_after": inputs_after,
        "source_scenario": source_scenario, "source_products": source_payload,
        "source_products_sha256": source_hash,
        "rigidity_gv": grid.values_gv,
        "critical_inclination_deg": _CRITICAL_INCLINATION_DEG,
        "variants": variants, "comparisons": _comparisons(variants),
        "conventions": {
            "kp": 3, "initial_raan_deg": "per_variant_explicit_zero_except_raan_sweep",
            "source_operator": "source_product_times_dimensionless_psi_no_extra_time_factor",
            "grid": "union_401_log_rigidity_nodes_and_exact_source_rigidities",
            "source_integral": "finite_energy_linear_trapezoid_between_existing_centers_no_tail",
            "orbit_weights": "equal_complete_temporal_cells",
            "window_policy": "max_min_days_min_orbits_rounded_up_to_complete_cells",
            "cell_rounding_window_change": "reported_per_comparison_not_pure_fixed_window_refinement",
            "mlt": "uniform_ensemble_not_reconstructed_mission_history",
            "legacy": "analytical_window_and_step_estimate_only_no_legacy_trace_allocated",
        },
        "limitations": [
            "This is a local-window sampling diagnostic, not a reconstructed mission history.",
            "Only proton, He and Fe GCR/SEP products are analyzed; diagnostic ion roster is incomplete.",
            "Source-model incomplete reasons remain applicable to every weighted metric.",
            "The energy grid is held fixed; its density and omitted boundary pieces are not tested.",
            "No trapped-belt, shielding, LET, dose or SEE acceptance matrix is calculated.",
            "A small fixed-source integral change does not qualify the full physical spectra.",
            "Temporal-window variation is distinct from quadrature-density convergence.",
            "Orbital density comparisons hold the requested window, not exactly the rounded window.",
            "Window-extension scope explores phase coverage without selecting or fitting a production window.",
            "Geometric shadow, uniform MLT and printed precession assumptions remain unqualified.",
            "No model constants or source coefficients are fitted to COSRAD or any target.",
        ],
    }


def write_report_exclusively(report: dict[str, Any], output: Path) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "heo", "critical", "smoke", "window_extension"),
                        default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        print(json.dumps({"preflight": _preflight(_specifications(args.case))},
                         ensure_ascii=False, allow_nan=False), flush=True)
        write_report_exclusively(build_report(args.case), args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
