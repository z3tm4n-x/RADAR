"""Compare protected external spectra; diagnostic factorial experiment, not a model splice."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from math import fsum, isclose, isfinite
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(directory))

import compare_srim_source_overlap as io  # noqa: E402
import run_srim_hze_products as sources  # noqa: E402
from radar.core.project import MissionConfig  # noqa: E402
from radar.core.spectra import Spectrum1D  # noqa: E402
from radar.core.types import AssessmentMode, Particle, SpectrumLocation, SpectrumQuantity  # noqa: E402
from radar.core.units import Unit  # noqa: E402

IONS = {2: 4., 10: 20.2, 26: 55.8, 92: 238.}
SHIELDS = (0., .01, .1, 1., 3., 10.)
SOURCE_FILES = {
    1: ("srim_upper_products_full_20261105.json", "d35e107ac7b6d406ea8e31d3dca0b6558d138cc5dfd698742d44c6c68f33d786"),
    2: ("srim_upper_products_dense_20261105.json", "6329581471422c66bc9510b438d16e22217802536147207d78cf4a1679b53c4b"),
}
EXPECTED = {(kind, mode.value, z) for kind in ("gcr", "sep") for mode in AssessmentMode
            for z in IONS if (kind, z) != ("sep", 92)}
FLAGS = {"status": "INCOMPLETE", "normative_export_allowed": False, "orbit_calculation_performed": False,
         "production_tables_replaced": False, "table_splicing_performed": False,
         "dose_or_see_calculated": False, "physical_accuracy_verified": False}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(refinement: int) -> tuple[dict[tuple[str, str, int], Spectrum1D], dict[str, str]]:
    if type(refinement) is not int or refinement not in SOURCE_FILES:
        raise ValueError("Expected declared source refinement 1 or 2.")
    filename, sha = SOURCE_FILES[refinement]
    retained = io.read_pinned(ROOT / "outputs" / filename, sha)
    mission = MissionConfig(launch_year=2026, lifetime_years=5, sep_exceedance_probability=.1)
    baseline_pins = sources._code_pins()
    io.same(asdict(mission), retained["mission"], "external source mission settings")
    for name in ("implementation_and_input_sha256_before", "implementation_and_input_sha256_after"):
        io.same(baseline_pins, retained[name], "OST input/code state")
    if (retained["schema_version"] != "radar.srim_hze_external_products.v1"
            or retained["source_grid_refinement"] != refinement or retained["orbit_calculation_performed"] is not False
            or retained["normative_export_allowed"] is not False):
        raise ValueError("Unexpected external spectra evidence.")
    reference = {}
    for row in retained["cases"]:
        if row["z"] not in IONS or row["thickness_g_cm2"] != 0:
            continue
        key = row["source"], row["mode"], row["z"]
        if key in reference:
            raise ValueError("Duplicate retained source.")
        reference[key] = row["energy_product"]["spectrum"]
    if reference.keys() != EXPECTED:
        raise ValueError("Missing or unexpected retained source.")
    result = {}
    for product in sources.source_products(mission, refinement):
        spectrum = product.spectrum
        if (spectrum.particle is not Particle.HZE or spectrum.species is None
                or spectrum.species.atomic_number not in IONS or spectrum.assessment_mode is None
                or spectrum.quantity not in (SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
                    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX, SpectrumQuantity.DIFFERENTIAL_FLUENCE)):
            continue
        key = spectrum.source.value, spectrum.assessment_mode.value, spectrum.species.atomic_number
        if (key not in reference or key in result or spectrum.location is not SpectrumLocation.OUTSIDE_MAGNETOSPHERE
                or spectrum.species.is_aggregate or not any(spectrum.y)):
            raise ValueError("Unexpected, repeated or empty pilot source.")
        actual = asdict(spectrum)
        for field in ("species", "x", "y", "x_unit", "y_unit", "assessment_mode", "quantity", "angular_basis",
                      "ordinate_basis", "temporal_basis", "axis_kind", "bin_edges", "mission_duration_s"):
            io.same(actual[field], reference[key][field], f"zero-shield reference {key}/{field}")
        masses = [float(v) for step in spectrum.transformation_chain for k, v in step.parameters
                  if k == "source_tabulated_mass_parameter"]
        if not masses or any(v != IONS[key[2]] for v in masses):
            raise ValueError("Literal source mass is absent or changed.")
        result[key] = spectrum
    if result.keys() != EXPECTED or len(result) != 14:
        raise ValueError("Exactly 14 positive GCR/SEP source-mode-ion inputs required.")
    return result, baseline_pins


def decomposition(values: dict[str, float], source_total: float) -> dict[str, Any]:
    """SS is the reference. Keep interaction and a second denominator for rare bins."""
    if (set(values) != {"SS", "CS", "SC", "CC"} or not isfinite(source_total) or source_total <= 0
            or any(not isfinite(v) or v < 0 for v in values.values())):
        raise ValueError("Finite nonnegative factorial counts and positive source integral required.")
    ss, cs, sc, cc = (values[k] for k in ("SS", "CS", "SC", "CC"))
    delta = {"Al": cs-ss, "Si": sc-ss, "interaction": fsum((cc, -cs, -sc, ss)), "combined": cc-ss}
    if not isclose(fsum((delta["Al"], delta["Si"], delta["interaction"])), delta["combined"],
                   rel_tol=1e-12, abs_tol=1e-13 * source_total):
        raise ValueError("Factorial balance does not close.")
    return {"delta": delta, "percent_of_full_source": {k: 100*v/source_total for k, v in delta.items()},
            "percent_of_SS": {k: (100*v/ss if ss > 0 else None) for k, v in delta.items()}}


def prepare(controls: dict[str, Any], z: int, tail_grid: int) -> Any:
    from paired_stopping_impact import prepare_pair
    if z not in IONS or type(tail_grid) is not int or tail_grid not in (40, 80):
        raise ValueError("Unexpected ion or tail grid.")
    array = {t: [r for r in controls["rows"] if r["z"] == z and r["material_z"] == t
             and (tail_grid == 80 or r["upper_coarse_node"])] for t in (13, 14)}
    common = {t: [r for r in array[t] if r["region"] == "common"] for t in array}
    def col(rows: list[dict[str, Any]], key: str) -> tuple[float, ...]:
        return tuple(r[key] for r in rows)
    for field in ("energy_mev_per_nucleon", "region", "mass_parameter_u_decimal"):
        if [r[field] for r in array[13]] != [r[field] for r in array[14]]:
            raise ValueError("Paired Al/Si controls must use identical physical nodes.")
    return prepare_pair(z, IONS[z], col(common[13], "energy_mev_per_nucleon"),
        tuple(r["srim_electronic_mev_cm2_g"] + r["srim_nuclear_mev_cm2_g"] for r in common[13]),
        col(common[13], "catima_total_mev_cm2_g"), col(common[14], "srim_electronic_mev_cm2_g"),
        col(common[14], "catima_electronic_mev_cm2_g"), col(array[13], "energy_mev_per_nucleon"),
        col(array[13], "catima_total_mev_cm2_g"), col(array[14], "catima_electronic_mev_cm2_g"))


def run(path: Path, sha: str, source_grid: int, tail_grid: int) -> dict[str, Any]:
    import build_paired_stopping_controls as control
    import paired_stopping_impact as impact
    def implementation() -> dict[str, str]:
        return {name: _hash(ROOT / "scripts" / name) for name in
                (Path(__file__).name, "build_paired_stopping_controls.py", "paired_stopping_impact.py")}
    pins = implementation()
    controls = control.read_verified(path, sha)
    pairs = {z: prepare(controls, z, tail_grid) for z in IONS}
    spectra, source_pins = load_sources(source_grid)
    rows = []
    for (kind, mode, z), source in sorted(spectra.items()):
        for thickness in SHIELDS:
            case = impact.evaluate_case(source, pairs[z], thickness)
            variants = case["variants"]
            if set(variants) != {"SS", "CS", "SC", "CC"}:
                raise ValueError("Missing paired model variant.")
            case["factorial"] = {axis: {t: decomposition({v: variants[v][axis][t] for v in variants},
                case["source_integral"]) for t in variants["SS"][axis]} for axis in ("energy_ge", "let_ge")}
            rows.append({"source": kind, "mode": mode, "z": z, "thickness_g_cm2": thickness,
                "energy_density_unit": source.y_unit.value, "let_density_unit": (Unit.DIFFERENTIAL_LET_FLUX
                    if mode == "peak" else Unit.DIFFERENTIAL_LET_FLUENCE).value, **case})
    io.same(implementation(), pins, "pilot code after evaluation")
    io.same(sources._code_pins(), source_pins, "source/production pins after evaluation")
    if (_hash(path) != sha or _hash(ROOT / "outputs" / SOURCE_FILES[source_grid][0]) != SOURCE_FILES[source_grid][1]
            or len(rows) != 84):
        raise ValueError("Evidence changed or a pilot case is missing.")
    return {"schema_version": "radar.paired_protected_external_spectra.v1", **FLAGS,
        "scope": "four_selected_ions_external_primary_only_not_orbit_acceptance",
        "controls_sha256": sha, "source_reference_sha256": SOURCE_FILES[source_grid][1],
        "source_refinement": source_grid, "upper_grid_points_per_decade_setting": tail_grid,
        "implementation_sha256": pins, "production_and_source_sha256": source_pins,
        "source_count": len(spectra), "case_count": len(rows), "shields_g_cm2": SHIELDS,
        "input_spectra": [asdict(s) for _, s in sorted(spectra.items())],
        "source_absences": [{"source": "sep", "z": 92, "reason": "not_in_OST_B9_roster_not_a_zero_spectrum"}],
        "energy_edges_mev_per_source_A": impact.ENERGY_EDGES, "let_edges_mev_cm2_mg": impact.LET_EDGES,
        "energy_thresholds": impact.ENERGY_THRESHOLDS, "let_thresholds": impact.LET_THRESHOLDS,
        "interpretation": ["Variants are alternative predictions for the same common incident interval.",
            "CATIMA-only upper incident contributions are computed on an all-CATIMA path, not a joined model.",
            "Source intervals outside the declared sampled grid are unknown; no extra time multiplication.",
            "No physical uncertainty bound, nuclear attenuation, fragmentation, dose or SEE estimate."],
        "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-grid", type=int, choices=(1, 2), default=1)
    parser.add_argument("--upper-grid", type=int, choices=(40, 80), default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.controls, args.sha256, args.source_grid, args.upper_grid)
    io.write_new(args.output, result)
    print(f"{result['case_count']} cases, SHA256 {_hash(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
