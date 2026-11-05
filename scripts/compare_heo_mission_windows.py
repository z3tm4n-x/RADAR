"""Compare pinned finite-mission density runs without conferring qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


_SCHEMA = "radar_heo_finite_mission_window_diagnostic_v1"


def _relative_change(value: float, reference: float) -> dict[str, Any]:
    if any(not math.isfinite(x) or x < 0.0 for x in (value, reference)):
        raise ValueError("Comparison metrics must be nonnegative finite numbers.")
    absolute = abs(value - reference)
    relative = absolute / reference if reference > 0.0 else None
    if relative is not None and not math.isfinite(relative):
        raise ValueError("Relative comparison overflowed.")
    return {"reference": reference, "value": value, "absolute_change": absolute,
            "relative_change": relative,
            "relative_change_status": "defined" if relative is not None else "zero_reference"}


def _variants(report: dict[str, Any]) -> dict[float, dict[str, Any]]:
    if report["schema"] != _SCHEMA:
        raise ValueError("Unsupported finite-mission report schema.")
    if report["normative_release_eligible"] or report["mission_radiation_history_qualified"]:
        raise ValueError("Unexpected qualification claims in diagnostic input.")
    for category in ("model", "input"):
        if report[f"{category}_sha256_before"] != report[f"{category}_sha256_after"]:
            raise ValueError("Diagnostic inputs changed during the recorded run.")
    variants = {item["requested"]["raan_deg"]: item for item in report["variants"]}
    if len(variants) != len(report["variants"]) or set(variants) != {0.0, 90.0}:
        raise ValueError("Expected unique initial orientations 0 and 90 degrees.")
    for item in variants.values():
        duration = item["requested"]["duration_s"]
        represented = item["represented_duration_s"]
        step = item["step_s"]
        steps = item["requested"]["steps_per_orbit"]
        if (any(isinstance(v, bool) or not math.isfinite(v) or v <= 0.0
                for v in (duration, represented, step))
                or isinstance(steps, bool) or not isinstance(steps, int) or steps < 4):
            raise ValueError("Temporal durations, steps and density must be finite and positive.")
        if (not item["exact_configured_mission_temporal_coverage"]
                or not item["contiguous_without_overlap"] or not item["global_phases_preserved"]
                or item["coverage_start_time_s"] != 0.0
                or item["coverage_end_time_s"] != duration
                or duration != report["source_mission_duration_s"]
                or abs(represented - duration) > 4 * math.ulp(duration)):
            raise ValueError("A full common interval is required for a density comparison.")
    return variants


def compare_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_variants, right_variants = _variants(left), _variants(right)
    for key in ("source_products_sha256", "source_products", "source_scenario",
                "model_sha256_before", "input_sha256_before", "rigidity_gv",
                "conventions", "production_operator_version", "source_mission_duration_s"):
        if left[key] != right[key]:
            raise ValueError(f"Density comparison inputs differ: {key}")
    comparisons = []
    for orientation in sorted(left_variants):
        a, b = left_variants[orientation], right_variants[orientation]
        for key in ("orbit", "duration_s", "raan_deg", "mlt_sample_count"):
            if a["requested"][key] != b["requested"][key]:
                raise ValueError(f"Non-density scenario input differs: {key}")
        if b["requested"]["steps_per_orbit"] != 2 * a["requested"]["steps_per_orbit"]:
            raise ValueError("The right run must exactly double orbital sampling density.")
        if a["step_s"] != 2.0 * b["step_s"]:
            raise ValueError("Actual temporal steps must reflect the requested density doubling.")
        av, bv = a["penetration"], b["penetration"]
        if len(av) != len(bv) or len(av) != len(left["rigidity_gv"]):
            raise ValueError("Penetration arrays must match the common rigidity grid.")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (*av, *bv)):
            raise ValueError("Penetration values must be finite and within zero to one.")
        am, bm = a["source_weighted_integrals"], b["source_weighted_integrals"]
        if am.keys() != bm.keys() or len(am) != 12:
            raise ValueError("Expected the same twelve selected source metrics.")
        metrics = {}
        for key in am:
            if ({k: v for k, v in am[key].items() if k != "value"}
                    != {k: v for k, v in bm[key].items() if k != "value"}):
                raise ValueError(f"Metric units, semantics or source metadata differ: {key}")
            metrics[key] = {**_relative_change(bm[key]["value"], am[key]["value"]),
                            "unit": am[key]["unit"]}
        comparisons.append({
            "left": a["name"], "right": b["name"], "initial_relative_node_deg": orientation,
            "duration_s": a["requested"]["duration_s"],
            "left_steps_per_orbit": a["requested"]["steps_per_orbit"],
            "right_steps_per_orbit": b["requested"]["steps_per_orbit"],
            "max_absolute_psi_change": max(abs(y - x) for x, y in zip(av, bv, strict=True)),
            "source_integral_changes": metrics,
        })
    return {"schema": "radar_heo_finite_mission_density_comparison_v1",
            "axis": "orbital_density_at_identical_exact_model_duration",
            "reference_denominator": "left_coarser_variant",
            "source_products_sha256": left["source_products_sha256"],
            "comparisons": comparisons, "normative_release_eligible": False,
            "stage4_spectral_convergence_established": False,
            "limitations": ["Only twelve fixed-source finite-energy integrals are compared.",
                            "These differences are not guaranteed quadrature error bounds.",
                            "This does not qualify MLT, initial phase, precession or source history."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        raw_left, raw_right = args.left.read_bytes(), args.right.read_bytes()
        report = compare_reports(json.loads(raw_left), json.loads(raw_right))
        report["inputs"] = {str(path.resolve()): hashlib.sha256(raw).hexdigest()
                            for path, raw in ((args.left, raw_left), (args.right, raw_right))}
        report["comparison_implementation_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes(),
        ).hexdigest()
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
