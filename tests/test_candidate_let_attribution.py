"""Independent analytic exit-energy/LET partitions, without native tables."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from math import inf, nan
from typing import Any

import pytest

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationSource, SpectrumAxisKind,
    SpectrumCompleteness, SpectrumLocation, SpectrumOrdinateBasis, SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit
from radar.validation.candidate_let_attribution import attribute_candidate_electronic_let
from radar.validation.candidate_transport import (
    CandidateIonTable, DiagnosticPrimaryTransport, transport_candidate_primary,
)


def _source(
    *, x: tuple[float, ...] = (1., 2., 3.), y: tuple[float, ...] = (2., 2., 2.),
    peak: bool = True,
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
    *, energies: tuple[float, ...] = (1., 2., 3.),
    lets: tuple[float, ...] = (1., 3., 1.),
) -> CandidateIonTable:
    return CandidateIonTable(
        z=2, mass_parameter_u=4.0, mass_policy="source_tabulated_parameter_diagnostic",
        dataset_ref="synthetic:attribution:sha256:test",
        al_energy=(.5, 1., 2., 3., 4., 10.), al_range=(.5, 1., 2., 3., 4., 10.),
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


def test_nonmonotone_branches_are_attributed_to_exit_bands() -> None:
    result = attribute_candidate_electronic_let(
        _transport(), energy_edges=(1., 1.5, 2.5, 3.), let_thresholds=(0., 1., 2., 3., 4.),
    )
    assert result.energy_edges == (1., 1.5, 2.5, 3.)
    assert result.thresholds == (0., 1., 2., 3., 4.)
    assert result.band_tail_integrals == (
        (1., 1., 0., 0., 0.), (2., 2., 2., 0., 0.), (1., 1., 0., 0., 0.),
    )
    assert result.full_supported_tail_integrals == (4., 4., 2., 0., 0.)
    assert result.supported_si_domain_integral == result.supported_exit_integral == 4.
    assert result.outside_si_domain_integral == 0.
    assert result.balance_residuals == (0.,) * 5
    assert result.si_domain_balance_residual == 0.
    assert not result.normative_export_allowed
    assert not result.physical_table_accuracy_verified
    assert not result.error_bound_established
    assert not result.al_path_accuracy_assessed
    with pytest.raises(FrozenInstanceError):
        result.outside_si_domain_integral = 9.  # type: ignore[misc]


def test_many_turning_points_include_all_preimages() -> None:
    transport = _transport(
        _source(x=(1., 3., 5.)),
        _table(energies=(1., 2., 3., 4., 5.), lets=(1., 3., 1., 3., 1.)),
    )
    result = attribute_candidate_electronic_let(
        transport, energy_edges=(1., 3., 5.), let_thresholds=(1., 2., 2.5, 3.),
    )
    assert result.band_tail_integrals == ((4., 2., 1., 0.),) * 2
    assert result.full_supported_tail_integrals == (8., 4., 2., 0.)


@pytest.mark.parametrize("plateau, expected", [
    (.5, (4., 0., 0., 0., 0.)), (1., (4., 4., 0., 0., 0.)),
    (2., (4., 4., 4., 0., 0.)), (3., (4., 4., 4., 4., 0.)),
])
def test_plateau_threshold_is_inclusive(
    plateau: float, expected: tuple[float, ...],
) -> None:
    result = attribute_candidate_electronic_let(
        _transport(table=_table(lets=(plateau,) * 3)),
        energy_edges=(1., 1.5, 2.5, 3.), let_thresholds=(0., 1., 2., 3., 4.),
    )
    assert result.full_supported_tail_integrals == expected
    assert result.band_tail_integrals[0] == tuple(value / 4 for value in expected)
    assert result.band_tail_integrals[1] == tuple(value / 2 for value in expected)
    assert result.band_tail_integrals[2] == tuple(value / 4 for value in expected)


def test_plateau_connected_to_slopes_does_not_duplicate_knots() -> None:
    result = attribute_candidate_electronic_let(
        _transport(_source(x=(1., 2.5, 4.)),
                   _table(energies=(1., 2., 3., 4.), lets=(1., 2., 2., 1.))),
        energy_edges=(1., 2., 3., 4.), let_thresholds=(0., 2., 2.00000001),
    )
    assert result.full_supported_tail_integrals == (6., 2., 0.)
    assert result.band_tail_integrals == ((2., 0., 0.), (2., 2., 0.), (2., 0., 0.))


def test_window_tails_and_unknown_si_domain_are_separate() -> None:
    result = attribute_candidate_electronic_let(
        _transport(_source(x=(.5, 2., 4.)), _table(lets=(1., 2., 3.))),
        energy_edges=(1.5, 2., 2.5), let_thresholds=(0., 1.75, 2.5, 3.1),
    )
    assert result.band_tail_integrals == ((1., .5, 0., 0.), (1., 1., 0., 0.))
    assert result.full_supported_tail_integrals == (4., 2.5, 1., 0.)
    assert result.below_energy_window_tail_integrals == (1., 0., 0., 0.)
    assert result.above_energy_window_tail_integrals == (1., 1., 1., 0.)
    assert result.outside_si_domain_integral == 3.
    assert result.supported_si_domain_integral == 4.
    assert result.supported_exit_integral == 7.
    assert result.balance_residuals == (0.,) * 4
    assert result.si_domain_balance_residual == 0.


def test_partial_exit_support_clips_inside_both_si_segments() -> None:
    result = attribute_candidate_electronic_let(
        _transport(_source(x=(1.25, 2., 2.75))),
        energy_edges=(0., 2., 9.), let_thresholds=(0., 2., 2.5, 3.),
    )
    assert result.full_supported_tail_integrals == (3., 2., 1., 0.)
    assert result.band_tail_integrals == ((1.5, 1., .5, 0.),) * 2


@pytest.mark.parametrize("edges, below, above", [
    ((4., 5.), (4., 2.), (0., 0.)), ((0., .5), (0., 0.), (4., 2.)),
])
def test_energy_window_disjoint_from_si_support(
    edges: tuple[float, ...], below: tuple[float, ...], above: tuple[float, ...],
) -> None:
    result = attribute_candidate_electronic_let(
        _transport(), energy_edges=edges, let_thresholds=(0., 2.),
    )
    assert result.band_tail_integrals == ((0., 0.),)
    assert result.below_energy_window_tail_integrals == below
    assert result.above_energy_window_tail_integrals == above


def test_positive_shield_uses_original_density_not_one_display_bin() -> None:
    source = _source(x=(1.5, 2.5, 3.5), y=(1.5, 2.5, 3.5))
    transport = _transport(source, _table(lets=(1., 2., 3.)), .5, (1., 3.))
    assert transport.spectrum.y == (2.5,)
    result = attribute_candidate_electronic_let(
        transport, energy_edges=(1., 2., 3.), let_thresholds=(0., 2., 2.5, 3.),
    )
    # j_exit(E)=E+.5, hence primitive E^2/2+.5E, not a constant 2.5.
    assert result.band_tail_integrals == ((2., 0., 0., 0.), (3., 3., 1.625, 0.))
    assert result.full_supported_tail_integrals == (5., 3., 1.625, 0.)


def test_display_energy_window_does_not_crop_original_measure() -> None:
    transport = _transport(_source(x=(1.5, 2.5, 3.5)), _table(), .5, (1.5, 2.5))
    assert transport.spectrum.integral_conserved() == 2.
    result = attribute_candidate_electronic_let(
        transport, energy_edges=(1., 3.), let_thresholds=(0., 2.),
    )
    assert result.full_supported_tail_integrals == (4., 2.)


def test_nonlinear_al_mapping_changes_exit_preimage_not_source_normalization() -> None:
    # Piecewise-linear R nodes from E^2: at exit E=2, R=4; plus shield 3
    # gives incident R=7, E=2.6 on [2,3]. j=2 from incident E=2..4.
    table = replace(_table(energies=(.5, 2., 4.), lets=(.5, 2., 4.)),
                    al_range=(.25, 1., 4., 9., 16., 100.))
    result = attribute_candidate_electronic_let(
        _transport(_source(x=(2., 3., 4.)), table, 3.),
        energy_edges=(0., 2., 4.), let_thresholds=(0., 2.),
    )
    assert result.full_supported_tail_integrals == pytest.approx((4., 2.8))
    assert result.band_tail_integrals[0] == pytest.approx((1.2, 0.))
    assert result.band_tail_integrals[1] == pytest.approx((2.8, 2.8))


def test_peak_and_fluence_values_unchanged_without_time_or_angular_factor() -> None:
    peak = _transport()
    cumulative = _transport(_source(peak=False))
    kwargs = {"energy_edges": (1., 2., 3.), "let_thresholds": (0., 2.)}
    assert attribute_candidate_electronic_let(peak, **kwargs) == (
        attribute_candidate_electronic_let(cumulative, **kwargs)
    )
    assert peak.source.y_unit is Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY
    assert cumulative.source.y_unit is Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY
    assert cumulative.source.mission_duration_s == 123456789.
    assert peak.source.mission_duration_s is None
    later = _transport(replace(_source(peak=False), mission_duration_s=2 * 123456789.))
    assert attribute_candidate_electronic_let(later, **kwargs) == (
        attribute_candidate_electronic_let(cumulative, **kwargs)
    )


def test_bin_average_source_preserves_its_declared_bins() -> None:
    source = replace(
        _source(x=(1.5, 2.5), y=(2., 4.)),
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE, bin_edges=(1., 2., 3.),
    )
    result = attribute_candidate_electronic_let(
        _transport(source), energy_edges=(1., 1.5, 2.5, 3.), let_thresholds=(0., 2.),
    )
    assert result.band_tail_integrals == ((1., 0.), (3., 3.), (2., 0.))
    assert result.full_supported_tail_integrals == (6., 3.)


def test_range_origin_offset_does_not_change_attribution() -> None:
    table = _table()
    shifted = replace(table, al_range=tuple(2.**40 + value for value in table.al_range))
    source = _source(x=(1.5, 2.5, 3.5), y=(1.5, 2.5, 3.5))
    kwargs = {"energy_edges": (0., 1., 2., 3., 4.), "let_thresholds": (0., 1., 2., 3.)}
    assert attribute_candidate_electronic_let(_transport(source, table, .5), **kwargs) == (
        attribute_candidate_electronic_let(_transport(source, shifted, .5), **kwargs)
    )


def test_empty_exits_remain_empty_not_zero_let_particles() -> None:
    transport = _transport(thickness=1000.)
    result = attribute_candidate_electronic_let(
        transport, energy_edges=(0., 1., 10.), let_thresholds=(0., 2.),
    )
    assert transport.ledger.unresolved_low_residual_integral == 4.
    assert result.full_supported_tail_integrals == (0., 0.)
    assert result.band_tail_integrals == ((0., 0.),) * 2
    assert result.outside_si_domain_integral == result.supported_si_domain_integral == 0.
    assert result.balance_residuals == (0., 0.)


@pytest.mark.parametrize("x", [(.1, .2, .3), (4., 5., 6.)])
def test_source_wholly_outside_si_has_unknown_let(x: tuple[float, ...]) -> None:
    result = attribute_candidate_electronic_let(
        _transport(_source(x=x)), energy_edges=(0., 10.), let_thresholds=(0., 2.),
    )
    assert result.full_supported_tail_integrals == (0., 0.)
    assert result.supported_si_domain_integral == 0.
    assert result.outside_si_domain_integral == pytest.approx(2 * (x[-1] - x[0]))


def test_zero_density_source_does_not_create_counts() -> None:
    result = attribute_candidate_electronic_let(
        _transport(_source(y=(0., 0., 0.))),
        energy_edges=(0., 10.), let_thresholds=(0., 2.),
    )
    assert result.full_supported_tail_integrals == (0., 0.)
    assert result.si_domain_balance_residual == 0.


def test_source_amplitude_scales_each_attributed_count_once() -> None:
    kwargs = {"energy_edges": (1.25, 1.75, 2.25, 2.75), "let_thresholds": (0., 2.)}
    base = attribute_candidate_electronic_let(_transport(), **kwargs)
    scaled = attribute_candidate_electronic_let(_transport(_source(y=(6., 6., 6.))), **kwargs)
    assert scaled.full_supported_tail_integrals == tuple(
        3 * value for value in base.full_supported_tail_integrals
    )
    assert scaled.band_tail_integrals == tuple(
        tuple(3 * value for value in row) for row in base.band_tail_integrals
    )
    assert scaled.below_energy_window_tail_integrals == tuple(
        3 * value for value in base.below_energy_window_tail_integrals
    )
    assert scaled.above_energy_window_tail_integrals == tuple(
        3 * value for value in base.above_energy_window_tail_integrals
    )


@pytest.mark.parametrize("values", [
    (), (1.,), (2., 1.), (1., 1.), (-1., 2.), (True, 2.), (1., False),
    (1., nan), (1., inf), (-inf, 1.), (1., 10**1000), ("1", 2.),
    (1., 2j), (1., None), [1., 2.], None, (2**53, 2**53 + 1),
])
def test_invalid_energy_edges_rejected(values: Any) -> None:
    with pytest.raises(ValueError):
        attribute_candidate_electronic_let(
            _transport(), energy_edges=values, let_thresholds=(0.,),
        )


@pytest.mark.parametrize("values", [
    (), (2., 1.), (1., 1.), (-1.,), (True,), (nan,), (inf,), (-inf,),
    (10**1000,), ("1",), (2j,), (None,), [1.], None, (2**53, 2**53 + 1),
])
def test_invalid_thresholds_rejected(values: Any) -> None:
    with pytest.raises(ValueError):
        attribute_candidate_electronic_let(
            _transport(), energy_edges=(0., 4.), let_thresholds=values,
        )


def test_single_zero_threshold_and_integer_grid_are_valid() -> None:
    result = attribute_candidate_electronic_let(
        _transport(), energy_edges=(0, 4), let_thresholds=(0,),
    )
    assert result.band_tail_integrals == ((4.,),)
    assert result.full_supported_tail_integrals == (result.supported_si_domain_integral,)


@pytest.mark.parametrize("transport", [None, 1., True, "transport"])
def test_invalid_transport_type(transport: Any) -> None:
    with pytest.raises(ValueError, match="diagnostic primary transport"):
        attribute_candidate_electronic_let(
            transport, energy_edges=(0., 4.), let_thresholds=(0.,),
        )


@pytest.mark.parametrize("value", [-1., nan, inf, -inf, True, 10**1000])
def test_invalid_supported_integral_is_rejected(value: float) -> None:
    transport = _transport()
    forged = replace(transport, ledger=replace(transport.ledger, supported_exit_integral=value))
    with pytest.raises(ValueError, match="finite nonnegative"):
        attribute_candidate_electronic_let(
            forged, energy_edges=(0., 4.), let_thresholds=(0.,),
        )


@pytest.mark.parametrize("value", [0., 3.9, 4.1, 1e100])
def test_false_supported_integral_cannot_bypass_domain_balance(value: float) -> None:
    transport = _transport()
    forged = replace(transport, ledger=replace(transport.ledger, supported_exit_integral=value))
    with pytest.raises(ValueError, match="balance failed"):
        attribute_candidate_electronic_let(
            forged, energy_edges=(0., 4.), let_thresholds=(2.,),
        )


@pytest.mark.parametrize("value", [-1., nan, inf, -inf, True])
def test_invalid_measured_piece_rejected(value: float, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport()
    monkeypatch.setattr(DiagnosticPrimaryTransport, "integral_on_exit", lambda *_: value)
    with pytest.raises(ValueError, match="finite nonnegative"):
        attribute_candidate_electronic_let(
            transport, energy_edges=(0., 4.), let_thresholds=(0.,),
        )


def test_piece_sum_overflow_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport()
    monkeypatch.setattr(DiagnosticPrimaryTransport, "integral_on_exit", lambda *_: 1e308)
    with pytest.raises(ValueError, match="sum exceeds finite"):
        attribute_candidate_electronic_let(
            transport, energy_edges=(0., 4.), let_thresholds=(0.,),
        )


def test_inconsistent_interval_measure_fails_threshold_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _transport()
    original = DiagnosticPrimaryTransport.integral_on_exit

    def corrupted(self: DiagnosticPrimaryTransport, lower: float, upper: float) -> float:
        value = original(self, lower, upper)
        return value + .1 if (lower, upper) == (1.5, 2.) else value

    monkeypatch.setattr(DiagnosticPrimaryTransport, "integral_on_exit", corrupted)
    with pytest.raises(ValueError, match="balance failed"):
        attribute_candidate_electronic_let(
            transport, energy_edges=(1., 1.5, 3.), let_thresholds=(0.,),
        )
