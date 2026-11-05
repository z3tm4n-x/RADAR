"""Analytic measure-pushforward witnesses, independent of CATIMA tables."""

from __future__ import annotations

from dataclasses import replace
from math import inf, nan

import pytest

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis,
    AssessmentMode,
    LetSpectrumRole,
    Particle,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit
from radar.validation.candidate_let import convert_candidate_electronic_let
from radar.validation.candidate_transport import (
    CandidateIonTable,
    DiagnosticPrimaryTransport,
    transport_candidate_primary,
)


def _source(
    *, x: tuple[float, ...] = (1.0, 2.0, 3.0),
    y: tuple[float, ...] = (2.0, 2.0, 2.0), peak: bool = True,
) -> Spectrum1D:
    return Spectrum1D(
        x=x, y=y, x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=(Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY if peak
                else Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY),
        quantity=(SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX if peak
                  else SpectrumQuantity.DIFFERENTIAL_FLUENCE),
        particle=Particle.HZE, source=RadiationSource.GCR, model="analytic",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=TemporalBasis.SOURCE_MAXIMUM if peak else TemporalBasis.MISSION_TOTAL,
        location=SpectrumLocation.ON_ORBIT,
        species=ParticleSpecies("He", atomic_number=2, mass_number=4),
        assessment_mode=AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE,
        mission_duration_s=None if peak else 123456789.0,
        model_reference=ModelReference("analytic", "1", ("analytic witness",), ("synthetic",)),
        transformation_chain=(TransformationStep("analytic_source", "1"),),
        completeness=SpectrumCompleteness.COMPLETE,
    )


def _table(
    *, energies: tuple[float, ...] = (1.0, 2.0, 3.0),
    lets: tuple[float, ...] = (1.0, 3.0, 1.0),
) -> CandidateIonTable:
    return CandidateIonTable(
        z=2, mass_parameter_u=4.0, mass_policy="source_tabulated_parameter_diagnostic",
        dataset_ref="synthetic:electronic-only:sha256:test",
        al_energy=(0.5, 1.0, 2.0, 3.0, 4.0, 10.0),
        al_range=(0.5, 1.0, 2.0, 3.0, 4.0, 10.0),
        si_energy=energies, si_electronic_let=lets,
    )


def _transport(
    source: Spectrum1D | None = None, table: CandidateIonTable | None = None,
    thickness: float = 0.0, output_edges: tuple[float, ...] | None = None,
) -> DiagnosticPrimaryTransport:
    return transport_candidate_primary(
        _source() if source is None else source,
        _table() if table is None else table, thickness,
        source_mass_parameter_u=4.0, output_edges=output_edges,
    )


def test_nonmonotone_peak_counts_both_branches_and_preserves_contract() -> None:
    transport = _transport()
    result = convert_candidate_electronic_let(transport, let_edges=(1.0, 2.0, 3.0))
    # The two preimages of [1,2] have width 0.5 each; j(E)=2.
    assert result.spectrum.y == pytest.approx((2.0, 2.0))
    assert result.spectrum.integral_conserved() == pytest.approx(4.0)
    assert result.ledger.supported_energy_integral == pytest.approx(4.0)
    assert result.ledger.mapped_integral == pytest.approx(4.0)
    assert result.ledger.balance_residual == pytest.approx(0.0, abs=1e-14)
    assert result.spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUX
    assert result.spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUX
    assert result.spectrum.x_unit is Unit.LET
    assert result.spectrum.let_role is LetSpectrumRole.HZE_SPECIES
    assert result.spectrum.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE
    assert result.spectrum.bin_edges == (1.0, 2.0, 3.0)
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
    assert result.spectrum.model_reference is not None
    refs = result.spectrum.model_reference.dataset_refs
    assert transport.table.dataset_ref in refs
    assert "stopping/hze_si_let.csv" not in refs
    assert not result.normative_export_allowed
    assert not result.physical_table_accuracy_verified


@pytest.mark.parametrize("plateau, expected", [
    (1.0, (4.0, 0.0)), (2.0, (0.0, 4.0)), (3.0, (0.0, 4.0)),
])
def test_plateau_deterministic_half_open_ownership(
    plateau: float, expected: tuple[float, float],
) -> None:
    table = _table(lets=(plateau,) * 3)
    result = convert_candidate_electronic_let(_transport(table=table), let_edges=(1., 2., 3.))
    assert result.spectrum.y == expected
    assert result.ledger.balance_residual == 0.0


@pytest.mark.parametrize("plateau, below, above", [(0.5, 4.0, 0.0), (4.0, 0.0, 4.0)])
def test_plateau_outside_window_is_counted_not_lost(
    plateau: float, below: float, above: float,
) -> None:
    result = convert_candidate_electronic_let(
        _transport(table=_table(lets=(plateau,) * 3)), let_edges=(1., 2., 3.),
    )
    assert result.spectrum.y == (0.0, 0.0)
    assert result.ledger.below_let_window_integral == below
    assert result.ledger.above_let_window_integral == above
    assert result.ledger.balance_residual == 0.0
    assert any("outside requested" in reason for reason in result.spectrum.incomplete_reasons)


def test_both_window_tails_and_outside_si_domain_have_separate_balances() -> None:
    # j=2 on .5..4. Si valid 1..3 with L(E)=E; requested L=1.5..2.5.
    result = convert_candidate_electronic_let(
        _transport(source=_source(x=(0.5, 2.0, 4.0)), table=_table(lets=(1., 2., 3.))),
        let_edges=(1.5, 2.0, 2.5),
    )
    ledger = result.ledger
    assert ledger.supported_energy_integral == 7.0
    assert ledger.mapped_integral == 2.0
    assert ledger.below_let_window_integral == 1.0
    assert ledger.above_let_window_integral == 1.0
    assert ledger.outside_si_domain_integral == 3.0
    assert ledger.balance_residual == 0.0
    assert any("outside candidate Si" in reason for reason in result.spectrum.incomplete_reasons)


def test_linear_density_exact_integrals_not_energy_bin_center_assignment() -> None:
    # R=E, shield .5: j_exit(E)=E+.5 over E=1..3.
    # The coarse energy display consists of one bin; its average is 2.5.
    source = _source(x=(1.5, 2.5, 3.5), y=(1.5, 2.5, 3.5))
    transport = _transport(source, _table(lets=(1., 2., 3.)), .5, (1., 3.))
    assert len(transport.spectrum.x) == 1
    result = convert_candidate_electronic_let(transport, let_edges=(1., 2., 3.))
    assert result.spectrum.y == pytest.approx((2.0, 3.0))
    assert result.ledger.balance_residual == pytest.approx(0.0, abs=1e-14)


def test_requested_energy_display_window_does_not_crop_let_measure() -> None:
    source = _source(x=(1.5, 2.5, 3.5))
    transport = _transport(source, _table(), .5, (1.5, 2.5))
    assert transport.spectrum.integral_conserved() == pytest.approx(2.0)
    result = convert_candidate_electronic_let(transport, let_edges=(1., 2., 3.))
    assert result.spectrum.integral_conserved() == pytest.approx(4.0)


def test_peak_and_cumulative_keep_own_units_without_mission_multiplier() -> None:
    peak = convert_candidate_electronic_let(_transport(), let_edges=(1., 2., 3.))
    cumulative = convert_candidate_electronic_let(
        _transport(_source(peak=False)), let_edges=(1., 2., 3.),
    )
    assert cumulative.spectrum.y == peak.spectrum.y
    assert cumulative.spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
    assert cumulative.spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUENCE
    assert cumulative.spectrum.mission_duration_s == 123456789.0
    assert cumulative.spectrum.temporal_basis is TemporalBasis.MISSION_TOTAL
    assert peak.spectrum.mission_duration_s is None
    assert peak.spectrum.assessment_mode is AssessmentMode.PEAK
    assert peak.spectrum.temporal_basis is TemporalBasis.SOURCE_MAXIMUM


def test_source_amplitude_scales_counts_once() -> None:
    base = convert_candidate_electronic_let(_transport(), let_edges=(1., 2., 3.))
    scaled = convert_candidate_electronic_let(
        _transport(_source(y=(6., 6., 6.))), let_edges=(1., 2., 3.),
    )
    assert scaled.spectrum.y == tuple(3 * value for value in base.spectrum.y)


def test_inherited_incomplete_reason_survives() -> None:
    source = replace(
        _source(), completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=("independent source qualification pending",),
    )
    result = convert_candidate_electronic_let(_transport(source), let_edges=(1., 2., 3.))
    assert source.incomplete_reasons[0] in result.spectrum.incomplete_reasons


def test_no_supported_exit_is_not_fabricated_as_zero_let_event() -> None:
    transport = _transport(thickness=1000.)
    result = convert_candidate_electronic_let(transport, let_edges=(0., 1., 3.))
    assert result.spectrum.y == (0., 0.)
    assert result.ledger.supported_energy_integral == 0.0
    assert result.ledger.mapped_integral == 0.0
    assert result.ledger.balance_residual == 0.0
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE


def test_all_source_outside_si_gives_known_domain_loss() -> None:
    source = _source(x=(4., 5., 6.))
    result = convert_candidate_electronic_let(_transport(source), let_edges=(1., 2., 3.))
    assert result.ledger.outside_si_domain_integral == 4.0
    assert result.ledger.mapped_integral == 0.0
    assert result.ledger.balance_residual == 0.0


@pytest.mark.parametrize("edges", [
    (), (1.,), (2., 1.), (1., 1.), (-1., 2.), (True, 2.), (1., False),
    (1., nan), (1., inf), (-inf, 1.), (1., 10**1000),
])
def test_invalid_let_edges_rejected(edges: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        convert_candidate_electronic_let(_transport(), let_edges=edges)


def test_bin_average_input_remains_constant_only_on_its_declared_bins() -> None:
    source = replace(
        _source(x=(1.5, 2.5), y=(2., 4.)),
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE, bin_edges=(1., 2., 3.),
    )
    result = convert_candidate_electronic_let(
        _transport(source, _table(lets=(1., 2., 3.))), let_edges=(1., 2., 3.),
    )
    assert result.spectrum.y == (2., 4.)
    assert result.ledger.balance_residual == 0.0


def test_refinement_preserves_total_and_coarse_bin_counts() -> None:
    transport = _transport(_source(y=(1., 7., 2.)))
    coarse = convert_candidate_electronic_let(transport, let_edges=(1., 2., 3.))
    fine = convert_candidate_electronic_let(transport, let_edges=(1., 1.5, 2., 2.5, 3.))
    assert coarse.spectrum.y == pytest.approx((
        .5 * sum(fine.spectrum.y[:2]), .5 * sum(fine.spectrum.y[2:]),
    ))
    assert coarse.ledger.mapped_integral == pytest.approx(fine.ledger.mapped_integral)


@pytest.mark.parametrize("supported", [0.0, 3.9, 4.1, 1e100])
def test_false_supported_integral_cannot_bypass_balance_guard(supported: float) -> None:
    transport = _transport()
    forged = replace(transport, ledger=replace(
        transport.ledger, supported_exit_integral=supported,
    ))
    with pytest.raises(ValueError, match="integral balance failed"):
        convert_candidate_electronic_let(forged, let_edges=(1., 2., 3.))


@pytest.mark.parametrize("supported", [-1.0, nan, inf, -inf, True, 10**1000])
def test_invalid_supported_integral_rejected(supported: float) -> None:
    transport = _transport()
    forged = replace(transport, ledger=replace(
        transport.ledger, supported_exit_integral=supported,
    ))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        convert_candidate_electronic_let(forged, let_edges=(1., 2., 3.))


@pytest.mark.parametrize("value", [-1.0, nan, inf, -inf, True])
def test_invalid_partition_integral_rejected_before_summation(
    value: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DiagnosticPrimaryTransport, "integral_on_exit", lambda *_: value)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        convert_candidate_electronic_let(_transport(), let_edges=(1., 2., 3.))


def test_partition_sum_overflow_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport()
    monkeypatch.setattr(DiagnosticPrimaryTransport, "integral_on_exit", lambda *_: 1e308)
    with pytest.raises(ValueError, match="sum exceeds finite"):
        convert_candidate_electronic_let(transport, let_edges=(1., 2., 3.))


def test_narrow_bin_density_overflow_fails_explicitly() -> None:
    transport = _transport(table=_table(lets=(1e-309,) * 3))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        convert_candidate_electronic_let(transport, let_edges=(0., 1e-308))
