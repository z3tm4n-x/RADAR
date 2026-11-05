"""Independent analytic witnesses for the opt-in primary candidate measure."""

from dataclasses import replace
from math import inf, nan, nextafter

import pytest

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationSource, SpectrumAxisKind,
    SpectrumCompleteness, SpectrumLocation, SpectrumOrdinateBasis, SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit
from radar.validation.candidate_transport import CandidateIonTable, transport_candidate_primary
from radar.validation.parameterized_stopping import ParameterizedStoppingRow


def _table(**changes: object) -> CandidateIonTable:
    values: dict[str, object] = {
        "z": 26, "mass_parameter_u": 55.8,
        "mass_policy": "source_tabulated_parameter_diagnostic",
        "dataset_ref": "synthetic-candidate@sha256:analytic-test-only",
        "al_energy": (1.0, 10.0, 30.0, 100.0, 200.0),
        "al_range": (1.0, 10.0, 30.0, 100.0, 200.0),
        "si_energy": (1.0, 10.0, 100.0, 200.0),
        "si_electronic_let": (1.0, 3.0, 2.0, 4.0),
    }
    values.update(changes)
    return CandidateIonTable(**values)  # type: ignore[arg-type]


def _source(*, peak: bool = False, **changes: object) -> Spectrum1D:
    values: dict[str, object] = {
        "x": (10.0, 30.0, 100.0), "y": (2.0, 2.0, 2.0),
        "x_unit": Unit.MEV_PER_NUCLEON,
        "y_unit": (Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY if peak
                   else Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY),
        "quantity": (SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX if peak
                     else SpectrumQuantity.DIFFERENTIAL_FLUENCE),
        "particle": Particle.HZE, "source": RadiationSource.SEP, "model": "analytic_source",
        "axis_kind": SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        "angular_basis": AngularBasis.OMNIDIRECTIONAL_4PI,
        "temporal_basis": TemporalBasis.SEP_PEAK if peak else TemporalBasis.MISSION_TOTAL,
        "assessment_mode": AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE,
        "mission_duration_s": None if peak else 1234567.0,
        "location": SpectrumLocation.ON_ORBIT,
        "species": ParticleSpecies("Fe", 26, 56),
        "model_reference": ModelReference(
            "analytic_source", "1", ("synthetic analytic witness",), ("synthetic source",),
        ),
        "transformation_chain": (TransformationStep("synthetic_source", "1", (
            ("source_tabulated_mass_parameter", "55.8"),
        )),),
        "completeness": SpectrumCompleteness.COMPLETE,
    }
    values.update(changes)
    return Spectrum1D(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("peak", [False, True])
def test_full_supported_lower_exits_recovered_with_known_count_ledger(peak: bool) -> None:
    source = _source(peak=peak)
    result = transport_candidate_primary(source, _table(), 20, source_mass_parameter_u=55.8)
    assert result.exit_support == pytest.approx((1, 80))
    assert result.ledger.source_integral == pytest.approx(180)
    assert result.ledger.supported_exit_integral == pytest.approx(158)
    assert result.ledger.represented_integral == pytest.approx(158)
    assert result.ledger.unresolved_low_residual_integral == pytest.approx(22)
    assert result.ledger.outside_al_domain_integral == 0
    assert result.ledger.below_output_window_integral == 0
    assert result.ledger.above_output_window_integral == 0
    assert result.spectrum.integral_conserved() == pytest.approx(158)
    assert result.spectrum.y == pytest.approx((2, 2, 10 / 7, 0))
    assert result.spectrum.y_unit is source.y_unit
    assert result.spectrum.quantity is source.quantity
    assert result.spectrum.temporal_basis is source.temporal_basis
    assert result.spectrum.angular_basis is source.angular_basis
    assert result.spectrum.mission_duration_s == source.mission_duration_s
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
    assert result.normative_export_allowed is False
    assert result.ledger.stopped_particle_classification_established is False
    assert result.ledger.source_tails_known is False
    assert result.spectrum.production_contract_violations() == (
        "incomplete spectra cannot be released as production results",
    )


@pytest.mark.parametrize("offset", [137.0, 1e6])
def test_supported_mapping_does_not_depend_on_additive_range_origin(offset: float) -> None:
    table = _table()
    shifted = replace(table, al_range=tuple(x + offset for x in table.al_range))
    a = transport_candidate_primary(_source(), table, 20, source_mass_parameter_u=55.8)
    b = transport_candidate_primary(_source(), shifted, 20, source_mass_parameter_u=55.8)
    assert a.spectrum.y == pytest.approx(b.spectrum.y)
    assert a.ledger == b.ledger
    assert a.exit_support == b.exit_support


@pytest.mark.parametrize("thickness", [0.01, 0.1, 1.0])
def test_large_common_origin_does_not_round_away_a_resolvable_range_increment(
    thickness: float,
) -> None:
    source = _source(x=(1.0, 2.0, 4.0), y=(1.0, 1.0, 1.0))
    table = _table(al_energy=(1.0, 2.0, 4.0),
                   al_range=(1e15 + 1, 1e15 + 2, 1e15 + 4))
    # The three supplied differences are exactly representable. Subtract the
    # arbitrary common origin before applying a much smaller shield increment.
    result = transport_candidate_primary(source, table, thickness, source_mass_parameter_u=55.8)
    assert result.exit_support == pytest.approx((1, 4 - thickness), abs=1e-13)
    assert result.ledger.unresolved_low_residual_integral == pytest.approx(thickness, abs=1e-13)
    assert result.ledger.represented_integral == pytest.approx(3 - thickness, abs=1e-13)
    assert result.integral_on_exit(1.25, 2.25) == pytest.approx(1, abs=1e-13)


def test_nextafter_supported_sliver_is_not_snapped_to_zero_or_negative() -> None:
    source = _source(x=(1.0, 2.0, 4.0), y=(1.0, 1.0, 1.0))
    table = _table(al_energy=(1.0, 2.0, 4.0), al_range=(1.0, 2.0, 4.0))
    thickness = nextafter(3, 0)
    result = transport_candidate_primary(source, table, thickness, source_mass_parameter_u=55.8)
    assert result.ledger.represented_integral == 3 - thickness
    assert result.ledger.represented_integral > 0
    assert result.ledger.unresolved_low_residual_integral == thickness
    assert result.ledger.balance_residual == 0
    above = transport_candidate_primary(source, table, nextafter(3, inf),
                                         source_mass_parameter_u=55.8)
    assert above.exit_support is None
    assert above.ledger.represented_integral == 0
    assert above.ledger.unresolved_low_residual_integral == 3


def test_display_window_is_not_confused_with_supported_measure() -> None:
    result = transport_candidate_primary(
        _source(), _table(), 20, source_mass_parameter_u=55.8, output_edges=(5.0, 15.0, 60.0),
    )
    assert result.ledger.represented_integral == pytest.approx(110)
    assert result.ledger.below_output_window_integral == pytest.approx(8)
    assert result.ledger.above_output_window_integral == pytest.approx(40)
    assert result.ledger.supported_exit_integral == pytest.approx(158)
    assert result.integral_on_exit(1, 80) == pytest.approx(158)
    assert result.integral_on_exit(0, 1) == 0
    assert result.integral_on_exit(80, 1000) == 0
    assert result.integral_on_exit(12, 12) == 0
    assert any("display window" in reason for reason in result.spectrum.incomplete_reasons)


def test_nonuniform_range_and_nonconstant_source_use_exact_preimage_integral() -> None:
    # R=E on[1,3], R=3E-6 on[3,8].  Source j(E)=2E+1 has primitive E^2+E.
    # t=2: supported inputs[3,8], exits[1,22/3]. Exit[2,4] maps to[10/3,14/3].
    source = _source(x=(1.0, 4.0, 8.0), y=(3.0, 9.0, 17.0))
    table = _table(al_energy=(1.0, 3.0, 8.0), al_range=(1.0, 3.0, 18.0))
    result = transport_candidate_primary(
        source, table, 2, source_mass_parameter_u=55.8, output_edges=(1.0, 2.0, 4.0, 8.0),
    )
    assert result.ledger.source_integral == pytest.approx(70)
    assert result.ledger.supported_exit_integral == pytest.approx(60)
    assert result.ledger.unresolved_low_residual_integral == pytest.approx(10)
    assert result.exit_support == pytest.approx((1, 22 / 3))
    assert result.integral_on_exit(2, 4) == pytest.approx(12)
    assert result.spectrum.integral_conserved() == pytest.approx(60)


def test_bin_average_input_uses_declared_edges_and_constant_within_each_bin() -> None:
    source = _source(
        x=(2.0, 5.0, 8.0), y=(2.0, 4.0, 1.0),
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE, bin_edges=(1.0, 3.0, 7.0, 9.0),
    )
    result = transport_candidate_primary(source, _table(), 1, source_mass_parameter_u=55.8)
    # Original mass22. Inputs[1,2] unresolved2; retained[2,9] has mass20.
    assert result.ledger.source_integral == 22
    assert result.ledger.unresolved_low_residual_integral == 2
    assert result.ledger.represented_integral == 20
    assert result.integral_on_exit(1, 4) == 10  # inputs[2,5]: 1*2 + 2*4
    assert result.integral_on_exit(5, 8) == 6  # inputs[6,9]: 1*4 + 2*1


@pytest.mark.parametrize("peak", [False, True])
def test_zero_shield_is_exact_identity_without_range_table_coverage(peak: bool) -> None:
    source = _source(peak=peak, x=(0.1, 10.0, 1000.0), y=(3.0, 2.0, 1.0))
    result = transport_candidate_primary(source, _table(), 0, source_mass_parameter_u=55.8)
    assert result.spectrum.x is source.x
    assert result.spectrum.y is source.y
    assert result.spectrum.bin_edges is source.bin_edges
    assert result.spectrum.ordinate_basis is source.ordinate_basis
    assert result.exit_support == (0.1, 1000.0)
    assert result.ledger.outside_al_domain_integral == 0
    assert result.ledger.unresolved_low_residual_integral == 0
    assert result.integral_on_exit(0.1, 1000) == source.integral_conserved()
    with pytest.raises(ValueError, match="identity"):
        transport_candidate_primary(source, _table(), 0,
                                    source_mass_parameter_u=55.8, output_edges=(1.0, 2.0))


def test_cumulative_duration_metadata_is_not_applied_again() -> None:
    first = transport_candidate_primary(_source(), _table(), 20, source_mass_parameter_u=55.8)
    second = transport_candidate_primary(
        _source(mission_duration_s=999999999.0), _table(), 20, source_mass_parameter_u=55.8,
    )
    assert first.spectrum.y == second.spectrum.y
    assert first.ledger == second.ledger


@pytest.mark.parametrize("peak", [False, True])
def test_source_amplitude_scales_every_mass_partition(peak: bool) -> None:
    source = _source(peak=peak)
    a = transport_candidate_primary(source, _table(), 20, source_mass_parameter_u=55.8)
    b = transport_candidate_primary(replace(source, y=(6.0, 6.0, 6.0)), _table(), 20,
                                    source_mass_parameter_u=55.8)
    assert b.spectrum.y == pytest.approx(tuple(3 * value for value in a.spectrum.y))
    assert b.ledger.represented_integral == pytest.approx(3 * a.ledger.represented_integral)
    assert b.ledger.unresolved_low_residual_integral == pytest.approx(
        3 * a.ledger.unresolved_low_residual_integral,
    )


def test_nested_output_grid_aggregates_to_identical_parent_counts() -> None:
    table = _table(al_energy=(1.0, 3.0, 8.0), al_range=(1.0, 3.0, 18.0))
    source = _source(x=(1.0, 4.0, 8.0), y=(3.0, 9.0, 17.0))
    coarse = transport_candidate_primary(source, table, 2, source_mass_parameter_u=55.8,
                                          output_edges=(1.0, 4.0, 8.0))
    fine = transport_candidate_primary(source, table, 2, source_mass_parameter_u=55.8,
                                        output_edges=(1.0, 2.0, 4.0, 6.0, 8.0))
    assert coarse.spectrum.y[0] * 3 == pytest.approx(fine.spectrum.y[0] + fine.spectrum.y[1] * 2)
    assert coarse.spectrum.y[1] * 4 == pytest.approx(sum(fine.spectrum.y[2:]) * 2)


@pytest.mark.parametrize("thickness", [199.0, 200.0, 1000.0])
def test_no_supported_exit_is_unknown_residual_not_stopped(thickness: float) -> None:
    result = transport_candidate_primary(_source(), _table(), thickness,
                                         source_mass_parameter_u=55.8)
    assert result.exit_support is None
    assert result.ledger.represented_integral == 0
    assert result.ledger.unresolved_low_residual_integral == 180
    assert result.integral_on_exit(1, 100) == 0
    assert result.ledger.stopped_particle_classification_established is False


@pytest.mark.parametrize("energies", [(0.1, 0.2, 0.3), (300.0, 400.0, 500.0)])
def test_source_wholly_outside_al_domain_is_not_zero_environment(
    energies: tuple[float, ...],
) -> None:
    source = _source(x=energies)
    result = transport_candidate_primary(source, _table(), 1, source_mass_parameter_u=55.8)
    assert result.ledger.outside_al_domain_integral == source.integral_conserved()
    assert result.exit_support is None
    assert result.ledger.unresolved_low_residual_integral == 0
    assert result.ledger.represented_integral == 0
    assert any("outside Al" in reason for reason in result.spectrum.incomplete_reasons)


def test_partial_table_domain_has_distinct_outside_and_low_residual_counts() -> None:
    table = _table(al_energy=(20.0, 30.0, 80.0), al_range=(20.0, 30.0, 80.0))
    result = transport_candidate_primary(_source(), table, 20, source_mass_parameter_u=55.8)
    assert result.ledger.source_integral == 180
    assert result.ledger.outside_al_domain_integral == 60
    assert result.ledger.unresolved_low_residual_integral == 40
    assert result.ledger.represented_integral == 80
    assert result.exit_support == pytest.approx((20, 60))


def test_zero_source_retains_diagnostic_status_without_inventing_missing_counts() -> None:
    result = transport_candidate_primary(_source(y=(0.0, 0.0, 0.0)), _table(), 20,
                                         source_mass_parameter_u=55.8)
    assert result.ledger.source_integral == result.ledger.represented_integral == 0
    assert result.ledger.balance_residual == 0
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE


def test_fractional_mass_binding_does_not_require_false_integer_isotope_identity() -> None:
    source = _source()
    assert source.species is not None and source.species.mass_number == 56
    result = transport_candidate_primary(source, _table(), 1, source_mass_parameter_u=55.8)
    parameters = dict(result.spectrum.transformation_chain[-1].parameters)
    assert parameters["mass_parameter_u"] == "55.8"
    assert parameters["mass_policy"] == "source_tabulated_parameter_diagnostic"
    assert result.spectrum.model_reference is not None
    assert _table().dataset_ref in result.spectrum.model_reference.dataset_refs
    assert "stopping/hze_al_range.csv" not in result.spectrum.model_reference.dataset_refs


@pytest.mark.parametrize("mass", [56, 55.800000000001, True, nan, inf, -1])
def test_source_mass_must_match_exact_declared_parameter(mass: float) -> None:
    with pytest.raises(ValueError):
        transport_candidate_primary(_source(), _table(), 1, source_mass_parameter_u=mass)


def test_conflicting_source_trace_is_not_ignored() -> None:
    source = _source(transformation_chain=(TransformationStep("source", "1", (
        ("source_tabulated_mass_parameter", "56"),
    )),))
    with pytest.raises(ValueError, match="trace"):
        transport_candidate_primary(source, _table(), 1, source_mass_parameter_u=55.8)


def test_upstream_incomplete_reasons_are_preserved() -> None:
    source = _source(completeness=SpectrumCompleteness.INCOMPLETE,
                     incomplete_reasons=("upstream uncertainty",))
    result = transport_candidate_primary(source, _table(), 1, source_mass_parameter_u=55.8)
    assert "upstream uncertainty" in result.spectrum.incomplete_reasons


@pytest.mark.parametrize("changes", [
    {"particle": Particle.PROTON}, {"species": None},
    {"source": RadiationSource.ERB},
    {"species": ParticleSpecies("Ni", 28, 58)},
    {"species": ParticleSpecies("all HZE", is_aggregate=True)},
    {"axis_kind": None}, {"x_unit": Unit.MEV}, {"angular_basis": None},
    {"temporal_basis": None}, {"assessment_mode": None}, {"location": None},
    {"location": SpectrumLocation.BEHIND_SHIELD}, {"model_reference": None},
    {"mission_duration_s": None}, {"quantity": SpectrumQuantity.DIFFERENTIAL_FLUX},
    {"y_unit": Unit.DIFFERENTIAL_FLUX}, {"completeness": None},
    {"transformation_chain": ()}, {"x": (0.0, 30.0, 100.0)},
    {"x": (True, 30.0, 100.0)}, {"y": (True, 2.0, 2.0)},
])
def test_invalid_source_contract_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        transport_candidate_primary(_source(**changes), _table(), 1,  # type: ignore[arg-type]
                                    source_mass_parameter_u=55.8)


@pytest.mark.parametrize("thickness", [True, -1.0, nan, inf, 10**500])
def test_invalid_thickness_rejected(thickness: float) -> None:
    with pytest.raises(ValueError):
        transport_candidate_primary(_source(), _table(), thickness, source_mass_parameter_u=55.8)


@pytest.mark.parametrize("lower,upper", [(2, 1), (-1, 1), (nan, 1), (0, inf), (False, 1)])
def test_invalid_exit_interval_rejected(lower: float, upper: float) -> None:
    result = transport_candidate_primary(_source(), _table(), 1, source_mass_parameter_u=55.8)
    with pytest.raises(ValueError):
        result.integral_on_exit(lower, upper)


@pytest.mark.parametrize("changes", [
    {"z": True}, {"z": 1}, {"z": 93}, {"z": 26.0},
    {"mass_parameter_u": True}, {"mass_parameter_u": nan}, {"mass_parameter_u": 25},
    {"mass_policy": "isotope_inferred_from_rounded_mass"}, {"dataset_ref": " "},
    {"al_energy": [1.0, 2.0]}, {"al_energy": (1.0,)},
    {"al_energy": (1.0, 1.0, 30.0, 100.0, 200.0)},
    {"al_energy": (1.0, 10.0, 30.0, 100.0, inf)},
    {"al_range": (1.0, 2.0)}, {"al_range": (0.0, 10.0, 30.0, 100.0, 200.0)},
    {"al_range": (1.0, 30.0, 10.0, 100.0, 200.0)},
    {"si_energy": [1.0, 2.0, 3.0, 4.0]}, {"si_energy": (1.0, 2.0)},
    {"si_electronic_let": (1.0, -1.0, 2.0, 4.0)},
    {"si_electronic_let": (1.0, 0.0, 2.0, 4.0)},
    {"si_electronic_let": (1.0, nan, 2.0, 4.0)},
    {"si_electronic_let": [1.0, 3.0, 2.0, 4.0]},
])
def test_invalid_table_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _table(**changes)


def _rows() -> tuple[ParameterizedStoppingRow, ...]:
    return tuple(
        ParameterizedStoppingRow(26, 55.8, material, energy, 7000 + energy, 2000, energy, 3000)
        for material in (13, 14) for energy in (1.0, 2.0, 3.0)
    )


def test_from_rows_selects_electronic_si_only_and_converts_gram_to_milligram() -> None:
    table = CandidateIonTable.from_rows(_rows(), mass_policy="source_tabulated_parameter_diagnostic",
                                       dataset_ref="synthetic rows")
    assert table.si_electronic_let == pytest.approx((7.001, 7.002, 7.003))
    assert table.al_range == (1, 2, 3)
    assert table.mass_parameter_u == 55.8
    assert table.normative_export_allowed is False


@pytest.mark.parametrize("kind", ["empty", "one_material", "unsorted", "duplicate", "mixed_mass"])
def test_from_rows_does_not_sort_repair_or_mix_ions(kind: str) -> None:
    rows = _rows()
    if kind == "empty":
        rows = ()
    elif kind == "one_material":
        rows = rows[:3]
    elif kind == "unsorted":
        rows = rows[::-1]
    elif kind == "duplicate":
        rows = (rows[0], *rows)
    elif kind == "mixed_mass":
        rows = (replace(rows[0], mass_parameter_u=56), *rows[1:])
    with pytest.raises(ValueError):
        CandidateIonTable.from_rows(rows, mass_policy="source_tabulated_parameter_diagnostic",
                                    dataset_ref="synthetic rows")
