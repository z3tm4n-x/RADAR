"""Compare one captured COSRAD cutoff table with fixed RADAR OST conventions.

This is a diagnostic on identical rigidity nodes, not a normative acceptance
test or a parameter fit. Orbit geometry comes from the validated text header;
COSRAD Kp, longitude history and averaging convention remain unestablished.
Only the observed ASCII format (six header lines, 55 data rows) is supported.
The required output is created exclusively, never overwriting existing data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from radar.core.project import OrbitConfig
from radar.core.types import OrbitType
from radar.geomagnetic.ost import ost_earth_shadow_factor
from radar.geomagnetic.ost_disturbance import ost_disturbed_cutoff_rigidity_gv
from radar.geomagnetic.ost_orbit import OST_ORBIT_SAMPLING_MODEL_VERSION, sample_ost_orbit
from radar.geomagnetic.ost_penetration import (
    build_ost_penetration_function_from_samples,
    ost_sample_cutoff_rigidity_gv,
)
from radar.geomagnetic.rigidity import RigidityGrid


_ROOT = Path(__file__).resolve().parent.parent
_MODEL_MODULES = (
    "radar.core.constants", "radar.core.project", "radar.core.types",
    "radar.geomagnetic.ost", "radar.geomagnetic.ost_orbit",
    "radar.geomagnetic.ost_disturbance", "radar.geomagnetic.ost_penetration",
    "radar.geomagnetic.penetration", "radar.geomagnetic.rigidity",
)
_STEPS = (60, 120, 240)
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_ORBIT_HEADER = (
    rf"Apogee[ \t]*=[ \t]*({_NUM})[ \t]+km\.",
    rf"Perigee[ \t]*=[ \t]*({_NUM})[ \t]+km\.",
    rf"Inclination[ \t]*=[ \t]*({_NUM})[ \t]+deg\.",
    rf"Argument of perigee[ \t]*=[ \t]*({_NUM})[ \t]+deg\.",
)


@dataclass(frozen=True)
class CutoffTable:
    orbit: OrbitConfig
    rigidity_grid: RigidityGrid
    values: tuple[float, ...]
    raw_header: tuple[str, ...]
    source_sha256: str


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked_float(token: str) -> float:
    """Reject overflow and nonzero decimal input rounded to floating-point zero."""
    if re.fullmatch(_NUM, token) is None:
        raise ValueError(f"Unsupported numeric token: {token!r}")
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("Numeric input must be finite.")
    mantissa = token.lower().split("e", maxsplit=1)[0]
    if value == 0.0 and any(digit in "123456789" for digit in mantissa):
        raise ValueError(f"Nonzero decimal input underflows to zero: {token!r}")
    return value


def _orbit_from_header(lines: list[str]) -> OrbitConfig:
    if len(lines) < 4:
        raise ValueError("Missing four-line orbit header.")
    values = []
    for pattern, line in zip(_ORBIT_HEADER, lines[:4], strict=True):
        match = re.fullmatch(pattern, line)
        if match is None:
            raise ValueError(f"Unsupported orbit header: {line!r}")
        values.append(_checked_float(match.group(1)))
    apogee, perigee, inclination, argument = values
    return OrbitConfig(
        orbit_type=OrbitType.CIRCULAR if apogee == perigee else OrbitType.ELLIPTIC,
        perigee_altitude_km=perigee,
        apogee_altitude_km=apogee,
        inclination_deg=inclination,
        argument_of_perigee_deg=argument,
    )


def parse_cutoff(data: bytes, *, expected_inclination_deg: float | None = None) -> CutoffTable:
    """Reject malformed/partial tables; never infer values from the filename."""
    lines = data.decode("ascii").splitlines()
    if len(lines) != 61:
        raise ValueError("Expected exactly six header lines and 55 cutoff rows.")
    orbit = _orbit_from_header(lines)
    if expected_inclination_deg is not None:
        if not math.isfinite(expected_inclination_deg):
            raise ValueError("Expected inclination must be finite.")
        if orbit.inclination_deg != expected_inclination_deg:
            raise ValueError("Expected inclination does not match the captured header.")
    if lines[4:6] != ["1 col.- rigidity [GV]", "2 col.- penitration function"]:
        raise ValueError("Unsupported cutoff column labels or units.")
    rigidity: list[float] = []
    values: list[float] = []
    for line in lines[6:]:
        fields = line.split()
        if len(fields) != 2 or any(re.fullmatch(_NUM, token) is None for token in fields):
            raise ValueError("Each cutoff row must contain exactly two numeric fields.")
        r, value = map(_checked_float, fields)
        if not 0.0 <= value <= 1.0:
            raise ValueError("Penetration must be in [0, 1].")
        if values and value < values[-1]:
            raise ValueError("Penetration must be nondecreasing.")
        rigidity.append(r)
        values.append(value)
    return CutoffTable(
        orbit=orbit, rigidity_grid=RigidityGrid(tuple(rigidity)), values=tuple(values),
        raw_header=tuple(lines[:6]), source_sha256=_sha(data),
    )


def _model_hashes() -> dict[str, str]:
    result = {"scripts/compare_cosrad_cutoff.py": _sha(Path(__file__).read_bytes())}
    for name in _MODEL_MODULES:
        module = importlib.import_module(name)
        relative = "src/" + name.replace(".", "/") + ".py"
        expected = (_ROOT / relative).resolve()
        if module.__file__ is None or Path(module.__file__).resolve() != expected:
            raise ValueError(f"Loaded model does not belong to this worktree: {name}")
        result[relative] = _sha(expected.read_bytes())
    return result


def _pointwise_cutoff_range(
    quiet_cutoffs: tuple[float, ...], *, mlt_count: int = 24,
) -> tuple[float, float]:
    """Extrema over the same orbit x uniform-MLT states as the current operator."""
    if not quiet_cutoffs or mlt_count < 1:
        raise ValueError("Pointwise cutoff extrema require orbit and MLT samples.")
    minimum, maximum = math.inf, -math.inf
    for quiet in quiet_cutoffs:
        for index in range(mlt_count):
            value = ost_disturbed_cutoff_rigidity_gv(
                cutoff_rigidity_gv=quiet, kp=3,
                magnetic_local_time_hours=24.0 * index / mlt_count,
            )
            minimum, maximum = min(minimum, value), max(maximum, value)
    return minimum, maximum


def _variant(table: CutoffTable, steps: int) -> dict[str, Any]:
    trace = sample_ost_orbit(
        table.orbit, raan_deg=0.0, min_days=16.0,
        steps_per_orbit=steps, max_samples=400000,
    )
    penetration = build_ost_penetration_function_from_samples(
        samples=trace.samples, rigidity_grid=table.rigidity_grid,
        kp=3, apply_disturbance=True, mlt_sample_count=24,
    )
    quiet = tuple(ost_sample_cutoff_rigidity_gv(sample) for sample in trace.samples)
    disturbed_range = _pointwise_cutoff_range(quiet)
    differences = tuple(a - b for a, b in zip(penetration.values, table.values, strict=True))
    largest = max(range(len(differences)), key=lambda i: abs(differences[i]))
    return {
        "steps_per_orbit": steps,
        "sample_count": len(trace.samples),
        "orbital_period_s": trace.orbital_period_s,
        "window_s": trace.window_s,
        "requested_window_s": trace.requested_window_s,
        "orbit_sampling_policy": trace.sampling_policy,
        "step_s": trace.step_s,
        "altitude_range_km": [min(trace.altitudes_km()), max(trace.altitudes_km())],
        "latitude_range_deg": [min(trace.latitudes_deg()), max(trace.latitudes_deg())],
        "longitude_wrapped_range_deg": [min(trace.longitudes_deg()), max(trace.longitudes_deg())],
        "quiet_cutoff_range_gv": [min(quiet), max(quiet)],
        "disturbed_cutoff_range_gv": disturbed_range,
        "disturbed_cutoff_range_definition": "all_orbit_x_uniform_24_MLT_pointwise_states",
        "earth_shadow_weight_range": [
            min(ost_earth_shadow_factor(s.altitude_km) for s in trace.samples),
            max(ost_earth_shadow_factor(s.altitude_km) for s in trace.samples),
        ],
        "radar_penetration": penetration.values,
        "operator_model": penetration.model,
        "operator_version": penetration.model_version,
        "operator_parameters": dict(penetration.parameters),
        "operator_incomplete_reasons": penetration.incomplete_reasons,
        "radar_minus_cosrad": differences,
        "largest_absolute_discrepancy": {
            "node_index_zero_based": largest,
            "rigidity_gv": table.rigidity_grid.values_gv[largest],
            "cosrad": table.values[largest],
            "radar": penetration.values[largest],
            "signed_difference": differences[largest],
            "absolute_difference": abs(differences[largest]),
        },
    }


def build_comparison(
    capture_dir: Path, *, expected_inclination_deg: float | None = None,
) -> dict[str, Any]:
    """Compare the observed nodes; fail if inputs/model files change during work."""
    capture_dir = capture_dir.resolve(strict=True)
    raw = {"cutoff.txt": (capture_dir / "cutoff.txt").read_bytes()}
    table = parse_cutoff(raw["cutoff.txt"], expected_inclination_deg=expected_inclination_deg)
    orbit_path = capture_dir / "orbit.txt"
    orbit_present = orbit_path.exists()
    if orbit_present:
        raw["orbit.txt"] = orbit_path.read_bytes()
        if _orbit_from_header(raw["orbit.txt"].decode("ascii").splitlines()) != table.orbit:
            raise ValueError("orbit.txt and cutoff.txt orbital headers disagree.")
    before = {name: _sha(value) for name, value in raw.items()}
    models_before = _model_hashes()
    variants = [_variant(table, steps) for steps in _STEPS]
    after = {name: _sha((capture_dir / name).read_bytes()) for name in raw}
    models_after = _model_hashes()
    if before != after or models_before != models_after or orbit_path.exists() != orbit_present:
        raise ValueError("Input or model files changed during the comparison; discard the run.")
    return {
        "schema": "radar_cosrad_cutoff_comparison_v3",
        "orbit_sampling_model_version": OST_ORBIT_SAMPLING_MODEL_VERSION,
        "evidence_role": "independent_model_comparison_only",
        "normative_release_eligible": False,
        "parameters_fitted": False,
        "operator_qualification": {
            "status": "INCOMPLETE",
            "scope": "diagnostic_mean_access_not_qualified_peak_or_cumulative_source_operator",
            "incomplete_reasons": tuple(dict.fromkeys(
                reason for variant in variants for reason in variant["operator_incomplete_reasons"]
            )),
        },
        "scope": "cutoff_only_on_observed_rigidity_nodes_no_source_spectra_no_L_B_validation",
        "capture_dir": str(capture_dir),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "orbit_from_cutoff_header": asdict(table.orbit),
        "raw_cutoff_header": table.raw_header,
        "orbit_header_crosscheck": "matched" if orbit_present else "orbit_txt_not_present",
        "input_sha256_before": before, "input_sha256_after": after,
        "model_sha256_before": models_before, "model_sha256_after": models_after,
        "rigidity_gv": table.rigidity_grid.values_gv,
        "cosrad_penetration": table.values,
        "cosrad_precision": "native_printed_values_no_dequantization_or_interpolation",
        "radar_conventions": {
            "kp": 3, "apply_disturbance": True, "mlt_sample_count": 24,
            "raan_deg": 0.0, "minimum_sampling_days": 16.0,
            "steps_per_orbit_variants": _STEPS, "max_samples": 400000,
            "trajectory": "kepler_earth_rotation_secular_precession_not_RAAN_ensemble",
            "window_policy": "bounded_min_days_and_10_orbits_rounded_up_to_whole_cells",
            "budget_policy": "fail_before_sampling_never_coarsen_or_truncate",
            "averaging": "psi=mean_time(mean_MLT(mu_i*I(R>Rc_i/delta_i(tMLT))))",
            "cutoff_condition": "R>Rc_star",
            "mlt_policy": "uniform_ensemble_not_actual_spacecraft_history",
            "shadow_model": "opaque_sphere_geometric_not_literal_empirical_zh9",
            "grid": "exact_printed_cosrad_rigidity_nodes_no_interpolation",
        },
        "unestablished_cosrad_conditions": [
            "Kp", "longitude_history_and_initial_longitude", "sampling_duration_and_weights",
            "MLT_and_disturbance_averaging_convention", "earth_shadow_implementation",
            "solar_history_for_any_subsequent_source_spectrum_comparison",
        ],
        "limitations": [
            "COSRAD is an independent comparison, not an error-free reference.",
            "Matched orbital scalars do not prove matched trajectories or physical models.",
            "Density checks on these nodes do not establish general spectral convergence.",
            "Finite-window sampling is not automatically representative of a whole mission.",
            "Discrepancies do not identify which implementation is correct.",
            "This runner does not edit production models; hashes identify the actual revision.",
            "The mean-access operator is not qualified for PEAK or CUMULATIVE source use.",
            "Uniform MLT and geometric shadow remain explicit model-qualification limitations.",
            "Wrapped longitude min/max is not a continuous longitude coverage measure.",
        ],
        "variants": variants,
        "density_pair_max_absolute_changes": [
            {
                "steps_per_orbit": [left["steps_per_orbit"], right["steps_per_orbit"]],
                "max_absolute_change": max(
                    abs(a - b) for a, b in zip(
                        left["radar_penetration"], right["radar_penetration"], strict=True,
                    )
                ),
            } for left, right in zip(variants, variants[1:])
        ],
    }


def write_report_exclusively(report: dict[str, Any], output: Path) -> None:
    """Serialize strictly before creating the output; never replace existing files."""
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--inclination-deg", type=_checked_float,
        help="Optional expected inclination assertion; never overrides captured geometry.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        report = build_comparison(args.capture_dir, expected_inclination_deg=args.inclination_deg)
        write_report_exclusively(report, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
