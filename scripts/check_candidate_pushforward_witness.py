"""Replay a bounded, independent midpoint check of candidate primary/LET measures.

The comparison path samples in original incident energy and uses NumPy's
linear interpolator. It never invokes the adapter's preimage integration.
This synthetic numerical witness is not a stopping-table accuracy check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
from numpy.typing import NDArray

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationSource, SpectrumAxisKind,
    SpectrumCompleteness, SpectrumLocation, SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit
from radar.validation.candidate_let import convert_candidate_electronic_let
from radar.validation.candidate_transport import CandidateIonTable, transport_candidate_primary


_ROOT = Path(__file__).resolve().parents[1]
_PATHS = (
    "scripts/check_candidate_pushforward_witness.py",
    "src/radar/validation/candidate_transport.py",
    "src/radar/validation/candidate_let.py",
    "src/radar/shielding/primary_let.py",
    "src/radar/validation/parameterized_stopping.py",
    "src/radar/core/spectra.py", "src/radar/core/spectrum_ops.py",
    "src/radar/core/types.py", "src/radar/core/units.py",
)
_SOURCE_E = (0.2, 1.0, 3.0, 8.0, 12.0)
_SOURCE_Y = (0.5, 5.0, 1.0, 4.0, 0.25)
_AL_E = (0.5, 2.0, 5.0, 10.0)
_AL_R = (3.1, 4.0, 15.0, 40.0)
_SI_E = (0.8, 1.4, 3.0, 4.0, 8.0)
_SI_L = (1.0, 5.0, 2.0, 2.0, 7.0)
_SHIELD = 6.3
_ENERGY_EDGES = (1.1, 3.0, 5.0, 7.0)
_LET_EDGES = (1.5, 2.0, 3.5, 6.0)
_ANALYTIC_SOURCE_AREA = 29.2
_METRICS = (
    "source_integral", "outside_al_domain_integral", "unresolved_low_residual_integral",
    "supported_exit_integral", "energy_represented_integral", "below_energy_window_integral",
    "above_energy_window_integral", "let_mapped_integral", "outside_si_domain_integral",
    "below_let_window_integral", "above_let_window_integral", "let_bin_0_integral",
    "let_bin_1_integral", "let_bin_2_integral",
)


def _hashes() -> dict[str, str]:
    return {path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest() for path in _PATHS}


def _adapter_metrics() -> dict[str, float]:
    source = Spectrum1D(
        x=_SOURCE_E, y=_SOURCE_Y, x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE, particle=Particle.HZE,
        source=RadiationSource.SEP, model="synthetic_witness",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI, temporal_basis=TemporalBasis.MISSION_TOTAL,
        assessment_mode=AssessmentMode.CUMULATIVE, mission_duration_s=1234567.0,
        location=SpectrumLocation.ON_ORBIT, species=ParticleSpecies("Fe", 26, 56),
        model_reference=ModelReference("synthetic_witness", "1",
                                       ("Synthetic analytic witness, not a normative source",),
                                       ("synthetic source coordinates in this script",)),
        transformation_chain=(TransformationStep("synthetic_source", "1", (
            ("source_tabulated_mass_parameter", "55.8"),
        )),), completeness=SpectrumCompleteness.COMPLETE,
    )
    table = CandidateIonTable(
        z=26, mass_parameter_u=55.8, mass_policy="source_tabulated_parameter_diagnostic",
        dataset_ref="synthetic candidate coordinates in this script",
        al_energy=_AL_E, al_range=_AL_R, si_energy=_SI_E, si_electronic_let=_SI_L,
    )
    primary = transport_candidate_primary(
        source, table, _SHIELD, source_mass_parameter_u=55.8, output_edges=_ENERGY_EDGES,
    )
    let = convert_candidate_electronic_let(primary, let_edges=_LET_EDGES)
    p, ell = primary.ledger, let.ledger
    if (primary.normative_export_allowed or let.normative_export_allowed
            or let.physical_table_accuracy_verified):
        raise ValueError("Synthetic witness must not claim physical or normative qualification.")
    return dict(zip(_METRICS, (
        p.source_integral, p.outside_al_domain_integral, p.unresolved_low_residual_integral,
        p.supported_exit_integral, p.represented_integral, p.below_output_window_integral,
        p.above_output_window_integral, ell.mapped_integral, ell.outside_si_domain_integral,
        ell.below_let_window_integral, ell.above_let_window_integral,
        *(y * (b - a) for y, a, b in zip(let.spectrum.y, _LET_EDGES, _LET_EDGES[1:])),
    ), strict=True))


def _midpoint_metrics(count: int) -> dict[str, float]:
    width = (_SOURCE_E[-1] - _SOURCE_E[0]) / count
    incident = _SOURCE_E[0] + (np.arange(count, dtype=np.float64) + 0.5) * width
    weights = np.interp(incident, _SOURCE_E, _SOURCE_Y) * width
    outside_al = (incident < _AL_E[0]) | (incident > _AL_E[-1])
    residual = np.interp(incident, _AL_E, _AL_R) - _SHIELD
    unresolved = (~outside_al) & (residual < _AL_R[0])
    supported = (~outside_al) & (~unresolved)
    # np.interp clamps irrelevant out-of-domain coordinates; every associated
    # count is explicitly excluded by the independent domain masks below.
    exit_energy = np.interp(residual, _AL_R, _AL_E)
    inside_si = supported & (exit_energy >= _SI_E[0]) & (exit_energy <= _SI_E[-1])
    let_value = np.interp(exit_energy, _SI_E, _SI_L)

    def integral(mask: NDArray[np.bool_]) -> float:
        return float(weights[mask].sum())

    bins = tuple(integral(
        inside_si & (let_value >= a)
        & ((let_value < b) if b < _LET_EDGES[-1] else (let_value <= b)),
    ) for a, b in zip(_LET_EDGES, _LET_EDGES[1:]))
    return dict(zip(_METRICS, (
        float(weights.sum()), integral(outside_al), integral(unresolved), integral(supported),
        integral(supported & (exit_energy >= _ENERGY_EDGES[0])
                 & (exit_energy <= _ENERGY_EDGES[-1])),
        integral(supported & (exit_energy < _ENERGY_EDGES[0])),
        integral(supported & (exit_energy > _ENERGY_EDGES[-1])),
        integral(inside_si & (let_value >= _LET_EDGES[0]) & (let_value <= _LET_EDGES[-1])),
        integral(supported & ~inside_si),
        integral(inside_si & (let_value < _LET_EDGES[0])),
        integral(inside_si & (let_value > _LET_EDGES[-1])), *bins,
    ), strict=True))


def run_witness(sample_counts: tuple[int, ...] = (2_000_000, 4_000_000)) -> dict[str, Any]:
    if (not sample_counts or any(type(n) is not int or not 2 <= n <= 20_000_000
                                 for n in sample_counts)
            or any(a >= b for a, b in zip(sample_counts, sample_counts[1:]))):
        raise ValueError("Sample counts must be unique increasing integers in 2..20000000.")
    before = _hashes()
    expected = _adapter_metrics()
    comparisons = []
    for count in sample_counts:
        observed = _midpoint_metrics(count)
        errors = {name: abs(observed[name] - expected[name]) for name in _METRICS}
        comparisons.append({
            "incident_midpoint_count": count,
            "incident_step_mev_per_nucleon": (_SOURCE_E[-1] - _SOURCE_E[0]) / count,
            "metrics": observed, "absolute_errors": errors,
            "max_absolute_error": max(errors.values()),
            "max_error_divided_by_analytic_source_area": max(errors.values()) / _ANALYTIC_SOURCE_AREA,
            "source_area_absolute_error": abs(observed["source_integral"] - _ANALYTIC_SOURCE_AREA),
        })
    after = _hashes()
    if before != after:
        raise ValueError("Source implementation changed during independent witness execution.")
    return {
        "schema_version": "radar.candidate_pushforward.midpoint_witness.v1",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "source_implementation_sha256_before": before,
        "source_implementation_sha256_after": after,
        "input_configuration": {
            "source_energy_mev_per_nucleon": _SOURCE_E, "source_ordinate": _SOURCE_Y,
            "source_node_count": len(_SOURCE_E),
            "source_ordinate_unit": Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY.value,
            "source_interpolation": "piecewise_linear_no_tails",
            "source_assessment_mode": "cumulative", "source_mission_duration_s": 1234567.0,
            "mission_duration_multiplier_applied": False,
            "source_mass_parameter_u": 55.8, "source_species_integer_label": 56, "z": 26,
            "al_energy_mev_per_nucleon": _AL_E, "al_relative_range_g_cm2": _AL_R,
            "si_energy_mev_per_nucleon": _SI_E, "si_electronic_let_mev_cm2_mg": _SI_L,
            "shield_g_cm2": _SHIELD, "energy_window_edges_mev_per_nucleon": _ENERGY_EDGES,
            "let_window_edges_mev_cm2_mg": _LET_EDGES,
            "plateau_bin_rule": "left_closed_right_open_final_upper_inclusive",
        },
        "analytic_source_area_cm_minus_2": _ANALYTIC_SOURCE_AREA,
        "adapter_metrics": expected, "midpoint_comparisons": comparisons,
        "normative_export_allowed": False, "physical_table_accuracy_verified": False,
        "production_tables_replaced": False, "all_spectra_convergence_established": False,
        "limitations": [
            "Fixed synthetic case; not an independent physical validation of CATIMA or OST.",
            "Midpoint comparison samples original incident energy without adapter integrals.",
            "Discontinuous classification masks introduce finite-grid quadrature error.",
            "No unknown source tails or residual ranges are counted as physically stopped.",
            "Raw spectral inputs are already fluences; no time or solid-angle factor is applied.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[2_000_000, 4_000_000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_witness(tuple(args.samples))
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "metrics_checked": len(_METRICS), "samples": args.samples}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
