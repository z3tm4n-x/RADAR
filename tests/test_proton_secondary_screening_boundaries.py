"""Independent support and accounting boundaries for early interval screening.

Synthetic R(E)=2E fixes the units analytically. These tests exercise omission
versus proven zero, without treating missing physical data as a zero response.
"""

from dataclasses import replace
import json
from math import inf, nextafter

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit
from radar.nuclear.tendl import TendlIncidentEnergySupport
from radar.shielding.proton_al import ProtonAlRangeEnergyTable
import radar.shielding.proton_al_secondary as secondary
from radar.shielding.proton_al_secondary import (
    Al27CrossSectionTable,
    IncidentEnergyQuadrature,
    SecondaryProtonKernel,
    SecondaryProtonProductDistribution,
)
from radar.shielding.proton_al_survival import Al27NonelasticCrossSectionTable


def _stopping() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 20.0), range_g_cm2=(2.0, 40.0),
        stopping_mev_cm2_g=(0.5, 0.5),
    )


def _kernel() -> SecondaryProtonKernel:
    return SecondaryProtonKernel(
        products=(SecondaryProtonProductDistribution(
            mt=5, product_index=0, incident_energy_mev=(1.0, 20.0),
            yield_value=(1.0, 1.0), emitted_energy_mev_by_incident=((2.0, 3.0),) * 2,
            pdf_per_mev_by_incident=((1.0, 0.0),) * 2,
        ),),
        cross_sections_by_mt={5: Al27CrossSectionTable(
            mt=5, energy_mev=(1.0, 20.0), sigma_barn=(2.0, 2.0),
        )},
        nonelastic_cross_section=Al27NonelasticCrossSectionTable(
            mt=3, energy_mev=(1.0, 20.0), sigma_barn=(3.0, 3.0),
        ),
    )


def _source(*, peak: bool = False) -> Spectrum1D:
    return Spectrum1D(
        x=(5.0, 6.0), y=(2.0, 4.0), x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX if peak else Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX if peak
        else SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        temporal_basis=TemporalBasis.SEP_PEAK if peak else TemporalBasis.MISSION_TOTAL,
        mission_duration_s=None if peak else 123_456.0,
        particle=Particle.PROTON, source=RadiationSource.SEP,
        model="independent_screening_source",
        completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=("independent inherited warning",),
    )


def _screen(
    lower: float,
    upper: float,
    *,
    edges: tuple[float, float] = (1.0, 19.0),
    thickness: float = 3.0,
    kernel: SecondaryProtonKernel | None = None,
) -> secondary._ScreenedIncidentInterval:
    return secondary._screen_secondary_incident_interval(
        lower_mev=lower, upper_mev=upper, output_edges_mev=edges,
        thickness_g_cm2=thickness, stopping_table=_stopping(),
        kernel=_kernel() if kernel is None else kernel,
        source_integral=7.5,
    )


def _calculate(source: Spectrum1D, kernel: SecondaryProtonKernel) -> Spectrum1D:
    return secondary.calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=source, stopping_table=_stopping(), kernel=kernel,
        thickness_g_cm2=0.5, depth_steps=1,
        incident_quadrature=IncidentEnergyQuadrature(
            partition_strategy="per_output_bin_active_cells", prequadrature_screening=True,
            max_response_evaluations=4, max_refinement_depth=1,
        ),
    )


def _forbid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(**kwargs):
        pytest.fail("A completely screened interval spent an expensive response call.")
    monkeypatch.setattr(secondary, "_single_incident_secondary_response", forbidden)


def test_no_full_birth_mapping_uses_mass_range_not_mev_as_thickness() -> None:
    # R(out_hi)=38, Rmax=40, t=3 -> mapping needs d >= 1 g/cm2.
    # At E0<=1.4, dmax=2*1.4-2=.8: no complete mapping at any depth.
    item = _screen(1.1, 1.4)
    assert item.classification == "omitted_unsupported"
    assert item.basis == "no_supported_full_birth_mapping"
    assert item.range_domain_missing
    assert item.path_energies == (1.0, 1.4)
    assert item.source_integral == 7.5


def test_narrow_supported_birth_mapping_strip_must_remain_evaluable() -> None:
    # dmax=1.2 now leaves a genuinely supported depth interval [1,1.2].
    item = _screen(1.1, 1.6)
    assert item.classification == "evaluate"
    assert item.range_domain_missing  # The primary still stops below Rmin.


@pytest.mark.parametrize("upper", [nextafter(1.5, -inf), 1.5, nextafter(1.5, inf)])
def test_birth_support_equality_and_float_neighbors_are_not_called_wholly_missing(upper) -> None:
    # Equality at dmax=1 must not be turned into a strict unsupported claim.
    assert _screen(1.1, upper).classification == "evaluate"


@pytest.mark.parametrize("edges", [(0.9, 1.5), (19.0, 20.1)])
def test_partial_output_bin_is_an_explicit_omission_even_for_zero_reaction(edges) -> None:
    kernel = _kernel()
    kernel = replace(kernel, cross_sections_by_mt={5: replace(
        kernel.cross_sections_by_mt[5], sigma_barn=(0.0, 0.0),
    )})
    item = _screen(5.0, 6.0, edges=edges, thickness=0.5, kernel=kernel)
    assert item.classification == "omitted_unsupported"
    assert item.basis == "output_range_domain"
    assert item.range_domain_missing


@pytest.mark.parametrize("lower,upper", [(0.5, 0.9), (19.0, 21.0), (21.0, 22.0)])
def test_unsupported_incident_interval_never_calls_a_range_interpolator(
    monkeypatch: pytest.MonkeyPatch, lower, upper,
) -> None:
    def forbidden(*args):
        pytest.fail("Unsupported incident energy was passed to range interpolation.")
    monkeypatch.setattr(ProtonAlRangeEnergyTable, "range_at_energy", forbidden)
    monkeypatch.setattr(ProtonAlRangeEnergyTable, "energy_at_range", forbidden)
    item = _screen(lower, upper)
    assert item.classification == "omitted_unsupported"
    assert item.basis == "incident_range_domain"
    assert item.path_energies == (lower, upper)


@pytest.mark.parametrize("lower,upper,expected", [
    (11.0, 12.0, "omitted_unsupported"),
    (10.25, 10.75, "evaluate"),
    (3.0, 4.0, "omitted_unsupported"),
])
def test_primary_support_uses_full_depth_envelope_not_only_entrance_energy(
    lower, upper, expected,
) -> None:
    kernel = replace(_kernel(), incident_energy_support=TendlIncidentEnergySupport(
        minimum_mev=5.0, maximum_mev=10.0, dataset_ref="synthetic_primary_support",
    ))
    item = _screen(lower, upper, edges=(2.0, 3.0), thickness=1.0, kernel=kernel)
    assert item.classification == expected
    if expected == "omitted_unsupported":
        assert item.basis == "all_primary_depths_outside_tendl_support"


@pytest.mark.parametrize("direction", [-inf, None, inf])
def test_primary_support_touch_and_float_neighbors_remain_evaluable(direction) -> None:
    # R=2E and t=1 give minimum primary E=lower-.5, touching Kmax=10.
    lower = 10.5 if direction is None else nextafter(10.5, direction)
    kernel = replace(_kernel(), incident_energy_support=TendlIncidentEnergySupport(
        minimum_mev=5.0, maximum_mev=10.0, dataset_ref="synthetic_primary_touch",
    ))
    assert _screen(lower, 11.0, edges=(2.0, 3.0), thickness=1.0, kernel=kernel).classification == "evaluate"


def test_proven_zero_retains_full_primary_path_extrema_and_source_weight() -> None:
    item = _screen(5.0, 6.0, edges=(7.0, 8.0), thickness=0.5)
    assert item.classification == "proven_zero"
    assert item.basis == "supported_positive_pdf_segments_do_not_overlap"
    assert item.path_energies == (4.75, 6.0)
    assert item.source_integral == 7.5
    assert not item.range_domain_missing


def test_missing_active_pdf_prevents_zero_even_when_stored_pdf_edges_are_disjoint() -> None:
    kernel = _kernel()
    product = replace(kernel.products[0], pdf_incident_energy_mev=(5.0, 20.0))
    item = _screen(5.0, 6.0, edges=(7.0, 8.0), thickness=0.5,
                   kernel=replace(kernel, products=(product,)))
    assert item.classification == "evaluate"


def test_base_plan_proof_inheritance_does_not_recheck_descendant_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = secondary._prepare_secondary_screening_plan(
        incident_spectrum=_source(), output_edges_mev=(7.0, 8.0),
        incident_energy_knots=(1.0, 5.25, 20.0), stopping_table=_stopping(),
        kernel=_kernel(), thickness_g_cm2=0.5,
    )
    assert plan.boundaries_mev == (5.0, 5.25, 5.5, 6.0)
    assert all(item.classification == "proven_zero" for item in plan.intervals)
    assert plan.evaluation_range_intervals_g_cm2 == ()
    assert sum(item.source_integral for item in plan.intervals) == 3.0

    def forbidden(**kwargs):
        pytest.fail("A known parent proof was unnecessarily recomputed.")
    monkeypatch.setattr(secondary, "_screen_secondary_incident_interval", forbidden)
    for lower, upper, parent_index in [(5.0, 5.125, 0), (5.25, 5.375, 1), (5.75, 6.0, 2)]:
        assert plan.for_interval(lower, upper) is plan.intervals[parent_index]
    # A crossing or out-of-parent interval cannot borrow either parent's proof.
    for lower, upper in [(5.125, 5.375), (4.9, 5.125), (5.75, nextafter(6.0, inf))]:
        item = plan.for_interval(lower, upper)
        assert item.classification == "evaluate"
        assert item.basis == "not_contained_in_base_cell"


def test_source_zero_panel_and_positive_to_zero_panel_have_distinct_classifications() -> None:
    source = replace(_source(), x=(5.0, 5.5, 6.0, 6.5), y=(0.0, 0.0, 2.0, 0.0))
    plan = secondary._prepare_secondary_screening_plan(
        incident_spectrum=source, output_edges_mev=(7.0, 8.0),
        incident_energy_knots=(), stopping_table=_stopping(), kernel=_kernel(), thickness_g_cm2=0.5,
    )
    assert [item.classification for item in plan.intervals] == [
        "source_zero", "proven_zero", "proven_zero",
    ]
    assert [item.source_integral for item in plan.intervals] == [0.0, 0.5, 0.5]


@pytest.mark.parametrize("peak", [False, True])
def test_source_integral_is_not_multiplied_by_duration_or_number_of_output_bins(
    monkeypatch: pytest.MonkeyPatch, peak: bool,
) -> None:
    _forbid_response(monkeypatch)
    source = _source(peak=peak)
    result = _calculate(source, _kernel())
    parameters = dict(result.transformation_chain[-1].parameters)
    integrals = json.loads(parameters["prequadrature_source_integrals_by_output_bin"])
    assert parameters["incident_quadrature_response_evaluations"] == "0"
    assert integrals["proven_zero"] == [3.0, 3.0]
    for index in range(len(source.x)):
        assert sum(values[index] for values in integrals.values()) == 3.0
    assert parameters["prequadrature_integral_scope"] == "per_output_bin_source_weights_not_summed_across_bins"
    assert result.y_unit is source.y_unit
    assert result.quantity is source.quantity
    assert result.temporal_basis is source.temporal_basis
    assert result.mission_duration_s == source.mission_duration_s
    assert source.incomplete_reasons[0] in result.incomplete_reasons


def test_one_source_has_different_screening_categories_in_different_output_bins() -> None:
    source = replace(_source(), x=(0.8, 2.5, 5.0, 6.0), y=(0.0, 0.0, 2.0, 4.0))
    result = _calculate(source, _kernel())
    parameters = dict(result.transformation_chain[-1].parameters)
    integrals = json.loads(parameters["prequadrature_source_integrals_by_output_bin"])
    # Source area is 2.5*(0+2)/2 + 1*(2+4)/2 = 5.5, independent of bin.
    assert integrals["omitted_unsupported"][0] == 5.5  # Partial output-bin support.
    assert integrals["evaluate"][1] == 5.5             # Contains the birth histogram.
    assert integrals["proven_zero"][2:] == [5.5, 5.5]  # Disjoint supported output bins.
    for index in range(len(source.x)):
        assert sum(values[index] for values in integrals.values()) == 5.5
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"


def test_empty_response_cache_keeps_primary_path_mt3_coverage_not_just_source_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_response(monkeypatch)
    kernel = _kernel()
    assert kernel.nonelastic_cross_section is not None
    mt3 = replace(kernel.nonelastic_cross_section, incident_energy_support=TendlIncidentEnergySupport(
        minimum_mev=4.9, maximum_mev=20.0, dataset_ref="synthetic_mt3_path_gap",
    ))
    result = _calculate(_source(), replace(kernel, nonelastic_cross_section=mt3))
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_response_evaluations"] == "0"
    # Source [5,6] is inside MT3 support; path minimum 4.75 is not.
    assert any("nonelastic attenuation" in reason and "[4.9, 20]" in reason
               for reason in result.incomplete_reasons)
    assert result.completeness is SpectrumCompleteness.INCOMPLETE


def test_all_omitted_without_responses_still_has_unknown_full_error_and_source_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_response(monkeypatch)
    kernel = replace(_kernel(), incident_energy_support=TendlIncidentEnergySupport(
        minimum_mev=1.0, maximum_mev=4.0, dataset_ref="synthetic_all_missing",
    ))
    result = _calculate(_source(), kernel)
    parameters = dict(result.transformation_chain[-1].parameters)
    integrals = json.loads(parameters["prequadrature_source_integrals_by_output_bin"])
    assert integrals["omitted_unsupported"] == [3.0, 3.0]
    assert integrals["proven_zero"] == [0.0, 0.0]
    assert parameters["incident_quadrature_response_evaluations"] == "0"
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"
    assert parameters["incident_quadrature_estimated_relative_error"] == "unavailable"
    assert parameters["incident_quadrature_supported_status"] == "empty_supported_domain"
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    assert _source().incomplete_reasons[0] in result.incomplete_reasons
