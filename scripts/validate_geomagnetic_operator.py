"""Bounded before/after and density diagnostics for the geomagnetic operator.

No source spectrum, shield transport or PEAK/CUMULATIVE qualification is inferred.
The legacy threshold operator is reproduced locally on the CURRENT orbit trace;
this does not reproduce the historical secular-perigee window or capped trace.
The new operator is called from RADAR.
Both use the same orbit samples, Kp and geometric shadow at the baseline density.
COSRAD inputs are independent comparison data, never fitting targets or truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from radar.core.project import OrbitConfig
from radar.core.types import OrbitType
from radar.geomagnetic.ost_disturbance import ost_disturbed_cutoff_rigidity_mlt_average_gv
from radar.geomagnetic.ost_orbit import (
    OST_ORBIT_SAMPLING_MODEL_VERSION, OstOrbitTrace, sample_ost_orbit,
)
from radar.geomagnetic.ost_penetration import (
    OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
    ost_penetration_values_from_samples,
    ost_sample_cutoff_rigidity_gv,
    ost_sample_shadow_weight,
)
from radar.geomagnetic.rigidity import RigidityGrid


_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CAPTURES = _ROOT / "outputs/cosrad_reference_20261030"
_MODEL_MODULES = (
    "radar.core.constants", "radar.core.project", "radar.core.types",
    "radar.geomagnetic.ost", "radar.geomagnetic.ost_orbit",
    "radar.geomagnetic.ost_disturbance", "radar.geomagnetic.ost_penetration",
    "radar.geomagnetic.penetration", "radar.geomagnetic.rigidity",
)
_LADDER = ((60, 24), (120, 24), (240, 24), (240, 48), (240, 96))
_CASE_NAMES = ("leo", "geo", "heo", "cosrad_eq", "cosrad_incl")


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    orbit: OrbitConfig
    rigidity_grid: RigidityGrid
    max_samples: int
    cosrad_values: tuple[float, ...] | None = None
    capture_directory: Path | None = None
    input_files: tuple[Path, ...] = ()
    parsed_input_sha256: tuple[tuple[str, str], ...] = ()
    raw_cutoff_header: tuple[str, ...] = ()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _orbit(perigee: float, apogee: float, inclination: float) -> OrbitConfig:
    return OrbitConfig(
        orbit_type=OrbitType.CIRCULAR if perigee == apogee else OrbitType.ELLIPTIC,
        perigee_altitude_km=perigee, apogee_altitude_km=apogee,
        inclination_deg=inclination, argument_of_perigee_deg=0.0,
    )


def _cutoff_reader() -> ModuleType:
    """Reuse the reviewed observed-format reader without changing the old runner."""
    name = "radar_geomagnetic_diagnostic_cutoff_reader"
    path = _ROOT / "scripts/compare_cosrad_cutoff.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the captured COSRAD cutoff reader.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _case(name: str, capture_root: Path) -> DiagnosticCase:
    if name not in _CASE_NAMES:
        raise ValueError(f"Unknown diagnostic case: {name}")
    grid = RigidityGrid(tuple(10.0 ** (-4.0 + 6.0 * i / 400) for i in range(401)))
    if name == "leo":
        return DiagnosticCase(name, _orbit(400.0, 400.0, 51.6), grid, 400000)
    if name == "geo":
        return DiagnosticCase(name, _orbit(35786.0, 35786.0, 0.0), grid, 400000)
    if name == "heo":
        return DiagnosticCase(name, _orbit(500.0, 39700.0, 63.4), grid, 30000)
    inclination = 0.0 if name == "cosrad_eq" else 51.6
    directory = capture_root / (
        "capture_002_equatorial" if name == "cosrad_eq"
        else "capture_001_actual_inclination_51p6"
    )
    reader = _cutoff_reader()
    cutoff_path, orbit_path = directory / "cutoff.txt", directory / "orbit.txt"
    cutoff_raw, orbit_raw = cutoff_path.read_bytes(), orbit_path.read_bytes()
    table = reader.parse_cutoff(cutoff_raw, expected_inclination_deg=inclination)
    expected = _orbit(35786.0, 35786.0, inclination)
    if table.orbit != expected:
        raise ValueError("Captured orbit differs from the declared diagnostic case.")
    orbit_header = reader._orbit_from_header(orbit_raw.decode("ascii").splitlines())
    if orbit_header != table.orbit:
        raise ValueError("Captured orbit.txt and cutoff.txt headers disagree.")
    return DiagnosticCase(
        name, expected, table.rigidity_grid, 400000,
        cosrad_values=table.values, capture_directory=directory.resolve(),
        input_files=(cutoff_path.resolve(), orbit_path.resolve()),
        parsed_input_sha256=(
            (str(cutoff_path.resolve()), hashlib.sha256(cutoff_raw).hexdigest()),
            (str(orbit_path.resolve()), hashlib.sha256(orbit_raw).hexdigest()),
        ),
        raw_cutoff_header=table.raw_header,
    )


def _model_hashes() -> dict[str, str]:
    paths = (
        "scripts/validate_geomagnetic_operator.py", "scripts/compare_cosrad_cutoff.py",
    )
    result = {name: _sha(_ROOT / name) for name in paths}
    for name in _MODEL_MODULES:
        module = importlib.import_module(name)
        relative = "src/" + name.replace(".", "/") + ".py"
        expected = (_ROOT / relative).resolve()
        if module.__file__ is None or Path(module.__file__).resolve() != expected:
            raise ValueError(f"Loaded model is outside this worktree: {name}")
        result[relative] = _sha(expected)
    return result


def _legacy_values(
    trace: OstOrbitTrace, grid: RigidityGrid, mlt_count: int,
) -> tuple[float, ...]:
    """Pre-change Rc/mean(delta), inclusive cutoff; not a normative reference."""
    pairs = sorted(
        (
            ost_disturbed_cutoff_rigidity_mlt_average_gv(
                cutoff_rigidity_gv=ost_sample_cutoff_rigidity_gv(sample),
                kp=3, mlt_sample_count=mlt_count,
            ),
            ost_sample_shadow_weight(sample),
        ) for sample in trace.samples
    )
    total = 0.0
    index = 0
    values = []
    for rigidity in grid.values_gv:
        while index < len(pairs) and pairs[index][0] <= rigidity:
            total += pairs[index][1]
            index += 1
        values.append(total / len(pairs))
    return tuple(values)


def _log_r_area(grid: RigidityGrid, values: tuple[float, ...]) -> float:
    """Normalized trapezoid area in log(R); not an integrated particle spectrum."""
    x = tuple(math.log(value) for value in grid.values_gv)
    return math.fsum(
        (right - left) * (yl + yr) / 2.0
        for left, right, yl, yr in zip(x, x[1:], values, values[1:])
    ) / (x[-1] - x[0])


def _difference(
    left: tuple[float, ...], right: tuple[float, ...], grid: RigidityGrid,
) -> dict[str, Any]:
    differences = tuple(a - b for a, b in zip(left, right, strict=True))
    index = max(range(len(differences)), key=lambda i: abs(differences[i]))
    return {
        "left_minus_right": differences,
        "max_absolute_change": abs(differences[index]),
        "at_rigidity_gv": grid.values_gv[index],
        "left_log_r_area": _log_r_area(grid, left),
        "right_log_r_area": _log_r_area(grid, right),
        "log_r_area_is_not_a_physical_spectrum_integral": True,
    }


def _variant(case: DiagnosticCase, steps: int, mlt_count: int) -> dict[str, Any]:
    trace = sample_ost_orbit(
        case.orbit, raan_deg=0.0, min_days=16.0,
        steps_per_orbit=steps, max_samples=case.max_samples,
    )
    requested_step = trace.orbital_period_s / steps
    capped = trace.step_s > requested_step * (1.0 + 1e-12)
    values = ost_penetration_values_from_samples(
        samples=trace.samples, rigidity_grid=case.rigidity_grid, kp=3,
        apply_disturbance=True, mlt_sample_count=mlt_count,
    )
    variant: dict[str, Any] = {
        "requested_steps_per_orbit": steps,
        "mlt_sample_count": mlt_count,
        "sample_count": len(trace.samples),
        "sample_limit": case.max_samples,
        "sampling_capped": capped,
        "requested_step_s": requested_step,
        "actual_step_s": trace.step_s,
        "window_s": trace.window_s,
        "requested_window_s": trace.requested_window_s,
        "orbit_sampling_policy": trace.sampling_policy,
        "orbital_period_s": trace.orbital_period_s,
        "effective_steps_per_orbit": trace.orbital_period_s / trace.step_s,
        "new_pointwise_penetration": values,
        "new_log_r_area": _log_r_area(case.rigidity_grid, values),
    }
    if (steps, mlt_count) == _LADDER[0]:
        legacy = _legacy_values(trace, case.rigidity_grid, mlt_count)
        variant["legacy_mean_delta_penetration"] = legacy
        variant["new_minus_legacy_same_samples"] = _difference(values, legacy, case.rigidity_grid)
        if case.cosrad_values is not None:
            variant["legacy_minus_cosrad"] = _difference(
                legacy, case.cosrad_values, case.rigidity_grid,
            )
    if case.cosrad_values is not None:
        variant["new_minus_cosrad"] = _difference(values, case.cosrad_values, case.rigidity_grid)
    return variant


def _case_report(case: DiagnosticCase) -> dict[str, Any]:
    variants = [_variant(case, steps, mlt) for steps, mlt in _LADDER]
    pairs = []
    for left, right in zip(variants, variants[1:]):
        axis = (
            "orbit_sampling" if left["mlt_sample_count"] == right["mlt_sample_count"]
            else "uniform_mlt_quadrature"
        )
        actual_orbit_changed = (
            left["sample_count"] != right["sample_count"]
            or left["actual_step_s"] != right["actual_step_s"]
        )
        informative = axis != "orbit_sampling" or actual_orbit_changed
        pairs.append({
            "axis": axis,
            "left_requested": [left["requested_steps_per_orbit"], left["mlt_sample_count"]],
            "right_requested": [right["requested_steps_per_orbit"], right["mlt_sample_count"]],
            "actual_orbit_sampling_changed": actual_orbit_changed,
            "density_comparison_informative": informative,
            "limitation": None if informative else "sample_cap_prevents_density_refinement",
            **_difference(
                right["new_pointwise_penetration"], left["new_pointwise_penetration"],
                case.rigidity_grid,
            ),
        })
    return {
        "case": case.name, "orbit": asdict(case.orbit),
        "rigidity_gv": case.rigidity_grid.values_gv,
        "grid_kind": "native_cosrad_nodes" if case.cosrad_values is not None else "401_log_nodes",
        "capture_directory": str(case.capture_directory) if case.capture_directory else None,
        "raw_cutoff_header": case.raw_cutoff_header,
        "cosrad_penetration": case.cosrad_values,
        "variants": variants, "density_comparisons": pairs,
    }


def build_report(case_name: str, *, capture_root: Path = _DEFAULT_CAPTURES) -> dict[str, Any]:
    """Measure fixed cases, fail if any used input or implementation changes."""
    names = _CASE_NAMES if case_name == "all" else (case_name,)
    models_before = _model_hashes()
    cases = tuple(_case(name, capture_root) for name in names)
    inputs_before = dict(item for case in cases for item in case.parsed_input_sha256)
    if inputs_before != {str(path): _sha(path) for case in cases for path in case.input_files}:
        raise ValueError("Captured input changed after parsing; discard the result.")
    reports = [_case_report(case) for case in cases]
    inputs_after = {str(path): _sha(path) for case in cases for path in case.input_files}
    models_after = _model_hashes()
    if inputs_before != inputs_after or models_before != models_after:
        raise ValueError("Input or model files changed during diagnostics; discard the result.")
    return {
        "schema": "radar_geomagnetic_operator_diagnostic_v2",
        "evidence_role": "bounded_operator_and_independent_model_comparison_only",
        "normative_release_eligible": False, "parameters_fitted": False,
        "stage4_spectral_convergence_established": False,
        "peak_operator_qualified": False, "cumulative_operator_qualified": False,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "production_operator_version": OST_GEOMAGNETIC_PENETRATION_MODEL_VERSION,
        "orbit_sampling_model_version": OST_ORBIT_SAMPLING_MODEL_VERSION,
        "model_sha256_before": models_before, "model_sha256_after": models_after,
        "input_sha256_before": inputs_before, "input_sha256_after": inputs_after,
        "conventions": {
            "kp": 3, "raan_deg": 0.0, "minimum_sampling_days": 16.0,
            "actual_window_policy": "bounded_min_days_and_10_orbits_rounded_up_to_whole_cells",
            "budget_policy": "fail_before_sampling_never_coarsen_or_truncate",
            "shadow": "unchanged_geometric_opaque_sphere_not_literal_zh9",
            "orbit_weights": "equal_sample_weights_not_extra_mission_duration",
            "mlt": "uniform_0_to_24h_ensemble_not_reconstructed_spacecraft_history",
            "ladder": _LADDER,
            "legacy": "mean_time(mu*I(Rc/mean_MLT(delta)<=R))",
            "new": "mean_time(mean_MLT(mu*I(R>Rc/delta)))",
            "baseline_operator_comparison": "identical_samples_and_24_MLT_nodes",
            "legacy_implementation": "old_threshold_on_current_trace_not_historical_sampler_replay",
        },
        "limitations": [
            "No source spectra, absorbed dose, LET, SEE or shielding are calculated.",
            "Log-rigidity area is a numerical diagnostic, not a physical flux integral.",
            "No one-percent Stage-4 convergence criterion is certified by this report.",
            "A capped unchanged orbit grid is not evidence of orbit-density convergence.",
            "COSRAD Kp, longitude, MLT history and internal shadow remain unestablished.",
            "COSRAD is not an error-free reference; discrepancies do not assign correctness.",
            "Equal scalar orbit inputs do not prove equal trajectories or physical assumptions.",
            "Uniform MLT weighting is an explicit diagnostic approximation, not observed history.",
            "No instantaneous peak operator or mission source-time factorization is qualified.",
            "Finite-window sampling is not automatically representative of a whole mission.",
        ],
        "cases": reports,
    }


def write_report_exclusively(report: dict[str, Any], output: Path) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*_CASE_NAMES, "all"), default="all")
    parser.add_argument("--capture-root", type=Path, default=_DEFAULT_CAPTURES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        report = build_report(args.case, capture_root=args.capture_root)
        write_report_exclusively(report, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
