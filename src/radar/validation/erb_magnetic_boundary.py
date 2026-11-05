"""Prepare deterministic L-B inputs and RADAR diagnostics for a future COSRAD check.

This bounded runner does not contain COSRAD values and does not establish a
numerical gold baseline or resolve the OST Appendix D model boundary.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import hashlib
from importlib import resources
import json
from math import acos, atan2, cos, degrees, isfinite, pi, radians, sin, sqrt, ulp
from pathlib import Path

from radar.erb.field import (
    ErbFieldLineSample,
    ost_l_shell_b_over_b0,
    shifted_dipole_l_shell_b_over_b0,
    trace_l_shell_b_over_b0,
)
from radar.erb.igrf import IgrfCoefficients
from radar.erb.orbit import ErbOrbitSample
from radar.erb.resources import ERB_IGRF14_RESOURCE_NAME, load_igrf14_coefficients
from radar.version import __version__

ERB_MAGNETIC_BOUNDARY_SCHEMA = "radar.stage4.erb-magnetic-boundary.v1"
ERB_MAGNETIC_BOUNDARY_EPOCHS = (1985.0, 2026.0)
_L_VALUES = (5.49, 5.5, 5.51)
_LATITUDES_DEG = (0.0, 30.0)
_AZIMUTHS = (("0", 0.0), ("0.9", 0.9), ("pi", pi))
_STEPS_RE = (0.06, 0.03)
_BOUNDARY_L = 5.5
# Independent transcription for fixture geometry, not production field constants.
_REFERENCE_RADIUS_KM = 6371.2


def _inverse_geometry(l_shell: float, latitude_deg: float, azimuth: float) -> ErbOrbitSample:
    """Construct known shifted-dipole coordinates by an independent rigid rotation."""
    tilt, longitude = radians(11.0), radians(291.0)
    axis = (sin(tilt) * cos(longitude), sin(tilt) * sin(longitude), cos(tilt))
    first = (cos(tilt) * cos(longitude), cos(tilt) * sin(longitude), -sin(tilt))
    second = (-sin(longitude), cos(longitude), 0.0)
    latitude = radians(latitude_deg)
    radius = l_shell * _REFERENCE_RADIUS_KM * cos(latitude) ** 2
    offset_latitude, offset_longitude = radians(15.5), radians(150.5)
    offset = (
        430.0 * cos(offset_latitude) * cos(offset_longitude),
        430.0 * cos(offset_latitude) * sin(offset_longitude),
        430.0 * sin(offset_latitude),
    )
    point = tuple(
        offset[index]
        + radius
        * (
            cos(latitude) * (cos(azimuth) * first[index] + sin(azimuth) * second[index])
            + sin(latitude) * axis[index]
        )
        for index in range(3)
    )
    geocentric_radius = sqrt(sum(value * value for value in point))
    return ErbOrbitSample(
        radius_km=geocentric_radius,
        theta_rad=acos(point[2] / geocentric_radius),
        longitude_rad=atan2(point[1], point[0]),
    )


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and isfinite(value) else None


def _field_summary(result: ErbFieldLineSample) -> dict[str, object]:
    return {
        "valid": result.valid,
        "method": result.method,
        "l_shell": _finite_or_none(result.l_shell),
        "b_over_b0": _finite_or_none(result.b_over_b0),
        "local_b_nt": _finite_or_none(result.local_b_nt),
        "minimum_b0_nt": _finite_or_none(result.mirror_b0_nt),
        "incomplete_reasons": list(result.incomplete_reasons),
        "classifier_l_shell": _finite_or_none(result.classifier_l_shell),
        "alternative_l_shell": _finite_or_none(result.alternative_l_shell),
    }


def _relative_differences(
    candidate: ErbFieldLineSample, reference: ErbFieldLineSample
) -> dict[str, float | None]:
    pairs = {
        "l_shell": (candidate.l_shell, reference.l_shell),
        "b_over_b0": (candidate.b_over_b0, reference.b_over_b0),
        "local_b_nt": (candidate.local_b_nt, reference.local_b_nt),
        "minimum_b0_nt": (candidate.mirror_b0_nt, reference.mirror_b0_nt),
    }
    return {
        key: (left / right - 1.0)
        if candidate.valid and reference.valid and isfinite(left) and isfinite(right) and right != 0
        else None
        for key, (left, right) in pairs.items()
    }


def _coefficient_fingerprint(coefficients: IgrfCoefficients) -> str:
    values = {
        "epoch": coefficients.epoch,
        "nmax": coefficients.nmax,
        "g": [[n, m, value] for (n, m), value in sorted(coefficients.g.items())],
        "h": [[n, m, value] for (n, m), value in sorted(coefficients.h.items())],
    }
    rendered = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def run_erb_magnetic_boundary_validation(
    *, epochs: Sequence[float], max_steps: int = 1500
) -> dict[str, object]:
    """Evaluate exactly 18 fixture points per explicitly selected supported epoch.

    IGRF is independently evaluated at 0.06 and 0.03 reference radii. The
    production OST selector is also invoked, so its branch and warnings are
    recorded rather than inferred by this runner. No boundary snapping occurs.
    """
    if not epochs or any(epoch not in ERB_MAGNETIC_BOUNDARY_EPOCHS for epoch in epochs):
        raise ValueError("Select at least one supported IGRF epoch: 1985 or 2026.")
    if type(max_steps) is not int or not 1 <= max_steps <= 1500:
        raise ValueError("Magnetic-boundary max_steps must be an integer from 1 to 1500.")
    selected_epochs = tuple(sorted({float(epoch) for epoch in epochs}))
    resource_bytes = (
        resources.files("radar.data.normative").joinpath(ERB_IGRF14_RESOURCE_NAME).read_bytes()
    )
    cases: list[dict[str, object]] = []
    epoch_metadata: list[dict[str, object]] = []
    selected_methods: Counter[str] = Counter()
    invalid_candidates = 0
    invalid_selections = 0
    diagnostic_selections = 0
    for epoch in selected_epochs:
        coefficients = load_igrf14_coefficients(epoch=epoch)
        epoch_metadata.append(
            {
                "requested_epoch": epoch,
                "loaded_epoch": coefficients.epoch,
                "nmax": coefficients.nmax,
                "loaded_coefficients_sha256": _coefficient_fingerprint(coefficients),
            }
        )
        for l_shell in _L_VALUES:
            for latitude in _LATITUDES_DEG:
                for azimuth_name, azimuth in _AZIMUTHS:
                    point = _inverse_geometry(l_shell, latitude, azimuth)
                    dipole = shifted_dipole_l_shell_b_over_b0(point)
                    coarse, fine = (
                        trace_l_shell_b_over_b0(
                            point,
                            coefficients,
                            step_km=step * _REFERENCE_RADIUS_KM,
                            max_steps=max_steps,
                        )
                        for step in _STEPS_RE
                    )
                    selected = ost_l_shell_b_over_b0(
                        point,
                        coefficients,
                        step_km=_STEPS_RE[0] * _REFERENCE_RADIUS_KM,
                        max_steps=max_steps,
                        compare_outer_igrf=True,
                    )
                    selected_methods[selected.method] += 1
                    invalid_candidates += sum(not result.valid for result in (coarse, fine, dipole))
                    invalid_selections += not selected.valid
                    diagnostic_selections += bool(selected.incomplete_reasons)
                    recomputed_l = _finite_or_none(dipole.l_shell)
                    cases.append(
                        {
                            "case_id": f"epoch{epoch:g}_L{l_shell:g}_lat{latitude:g}_az{azimuth_name}",
                            "igrf_epoch": epoch,
                            "target_magnetic_coordinates": {
                                "shifted_dipole_l_shell": l_shell,
                                "latitude_deg": latitude,
                                "azimuth_rad": azimuth,
                            },
                            "geographic_input": {
                                "radius_km": point.radius_km,
                                "geocentric_latitude_deg": 90.0 - degrees(point.theta_rad),
                                "longitude_deg": degrees(point.longitude_rad),
                                "colatitude_rad": point.theta_rad,
                                "longitude_rad": point.longitude_rad,
                                "coordinate_system": "geocentric_not_geodetic",
                            },
                            "candidates": {
                                "igrf_step_0.06_re": _field_summary(coarse),
                                "igrf_step_0.03_re": _field_summary(fine),
                                "shifted_dipole": _field_summary(dipole),
                            },
                            "production_selection": _field_summary(selected),
                            "relative_differences": {
                                "igrf_0.06_re_vs_0.03_re": _relative_differences(coarse, fine),
                                "shifted_dipole_vs_igrf_0.03_re": _relative_differences(
                                    dipole, fine
                                ),
                            },
                            "boundary_roundoff": {
                                "nominal_l_equals_boundary": l_shell == _BOUNDARY_L,
                                "recomputed_l_minus_nominal": (
                                    recomputed_l - l_shell if recomputed_l is not None else None
                                ),
                                "recomputed_l_minus_nominal_in_ulps": (
                                    (recomputed_l - l_shell) / ulp(l_shell)
                                    if recomputed_l is not None
                                    else None
                                ),
                                "nominal_region": "outer" if l_shell > _BOUNDARY_L else "inner",
                                "recomputed_classifier_region": (
                                    ("outer" if recomputed_l > _BOUNDARY_L else "inner")
                                    if recomputed_l is not None
                                    else None
                                ),
                                "snapping_applied": False,
                            },
                        }
                    )
    return {
        "schema": ERB_MAGNETIC_BOUNDARY_SCHEMA,
        "program_version": __version__,
        "report_kind": "cosrad_magnetic_boundary_comparison_preparation",
        "intended_use": {
            "diagnostic_only": True,
            "numerical_gold": False,
            "normative_export_allowed": False,
        },
        "cosrad_reference": {
            "status": "not_supplied",
            "comparison_performed": False,
            "values": None,
        },
        "source": {
            "igrf_resource": ERB_IGRF14_RESOURCE_NAME,
            "igrf_resource_sha256": hashlib.sha256(resource_bytes).hexdigest(),
            "epochs": epoch_metadata,
        },
        "method": {
            "reference_radius_km": _REFERENCE_RADIUS_KM,
            "trace_steps_re": list(_STEPS_RE),
            "max_steps_per_trace": max_steps,
            "inverse_geometry": "independent_rigid_rotation_of_OST_D_shifted_dipole",
            "shifted_dipole_offset_km": 430.0,
            "shifted_dipole_offset_latitude_deg": 15.5,
            "shifted_dipole_offset_longitude_deg": 150.5,
            "shifted_dipole_axis_latitude_deg": 79.0,
            "shifted_dipole_axis_longitude_deg": 291.0,
            "shifted_dipole_moment_scale": 1.0,
            "region_selection_convention": "L_shifted_dipole_gt_5.5_outer_else_inner",
            "boundary_classifier_modified": False,
            "boundary_smoothing": "none",
            "relative_difference_definition": "candidate / reference - 1; dimensionless, signed",
            "relative_differences_are_accuracy_estimates": False,
        },
        "matrix": {
            "epochs": list(selected_epochs),
            "target_l_shells": list(_L_VALUES),
            "magnetic_latitudes_deg": list(_LATITUDES_DEG),
            "magnetic_azimuths_rad": [value for _, value in _AZIMUTHS],
            "case_count": len(cases),
        },
        "summary": {
            "selected_methods": dict(sorted(selected_methods.items())),
            "invalid_candidate_count": invalid_candidates,
            "invalid_selection_count": invalid_selections,
            "diagnostic_selection_count": diagnostic_selections,
        },
        "cases": cases,
        "release_gate": {
            "report_eligible_for_normative_export": False,
            "blockers": [
                "Independent COSRAD L-B reference values have not been supplied.",
                "The pilot-L region-selection convention and model junction remain unresolved.",
                "Step halving and model differences are diagnostics, not independent validation.",
            ],
        },
    }


def erb_magnetic_boundary_validation_json(report: dict[str, object]) -> str:
    """Serialize finite JSON without timestamps or environment-dependent paths."""
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the bounded preparation report, with explicit epoch selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epoch",
        type=float,
        choices=ERB_MAGNETIC_BOUNDARY_EPOCHS,
        action="append",
        required=True,
        help="IGRF epoch; repeat to include both 1985 and 2026.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout.")
    parser.add_argument("--max-steps", type=int, default=1500, help="Trace budget, 1..1500.")
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.max_steps <= 1500:
        parser.error("--max-steps must be from 1 to 1500")
    report = run_erb_magnetic_boundary_validation(
        epochs=arguments.epoch, max_steps=arguments.max_steps
    )
    rendered = erb_magnetic_boundary_validation_json(report)
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0


__all__ = [
    "ERB_MAGNETIC_BOUNDARY_EPOCHS",
    "ERB_MAGNETIC_BOUNDARY_SCHEMA",
    "erb_magnetic_boundary_validation_json",
    "main",
    "run_erb_magnetic_boundary_validation",
]


if __name__ == "__main__":
    raise SystemExit(main())
