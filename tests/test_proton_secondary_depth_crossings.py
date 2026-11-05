"""Independent analytic tests for coupled secondary birth/exit crossings."""

from dataclasses import replace
from math import exp, sqrt

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.nuclear.tendl import TendlIncidentEnergySupport
from radar.shielding.proton_al import ProtonAlRangeEnergyTable, log_bin_edges_from_centers
from radar.shielding.proton_al_secondary import (
    PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
    PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON,
    PROTON_AL_SECONDARY_DEPTH_CROSSING_QUADRATURE_REASON,
    PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
    PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON,
    PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON,
    PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
    Al27CrossSectionTable,
    DepthCrossingQuadrature,
    IncidentEnergyQuadrature,
    SecondaryProtonKernel,
    SecondaryProtonProductDistribution,
    _depth_quadrature_intervals,
    _point_source_quadrature_boundaries,
    _secondary_exit_depth_knots,
    _secondary_incident_energy_knots,
    calculate_secondary_proton_spectrum_through_al,
)
from radar.shielding.proton_al_survival import (
    AL27_MOLAR_MASS_G_MOL,
    AVOGADRO_CONSTANT_MOL_INV,
    BARN_CM2,
    Al27CrossSectionDiscontinuity,
    Al27NonelasticCrossSectionTable,
)


def _stopping(*, range_scale: float = 1.0) -> ProtonAlRangeEnergyTable:
    # Exactly consistent R(E)=range_scale*E, S(E)=1/range_scale.
    return ProtonAlRangeEnergyTable(
        energy_mev=(0.1, 20.0),
        range_g_cm2=(range_scale * 0.1, range_scale * 20.0),
        stopping_mev_cm2_g=(1.0 / range_scale,) * 2,
    )


def _kernel(
    *, birth_edges: tuple[float, ...] = (0.39, 0.3901)
) -> SecondaryProtonKernel:
    pdf = (1.0 / (birth_edges[1] - birth_edges[0]),) + (0.0,) * (len(birth_edges) - 1)
    return SecondaryProtonKernel(
        products=(
            SecondaryProtonProductDistribution(
                mt=5,
                product_index=0,
                incident_energy_mev=(0.1, 20.0),
                yield_value=(1.0, 1.0),
                emitted_energy_mev_by_incident=(birth_edges,) * 2,
                pdf_per_mev_by_incident=(pdf,) * 2,
            ),
        ),
        cross_sections_by_mt={
            5: Al27CrossSectionTable(
                mt=5,
                energy_mev=(0.1, 5.0, 20.0),
                sigma_barn=(0.0, 2.0, 2.0),
                discontinuities=(
                    Al27CrossSectionDiscontinuity(
                        energy_mev=5.0,
                        left_sigma_barn=0.0,
                        right_sigma_barn=2.0,
                    ),
                ),
            ),
        },
        nonelastic_cross_section=Al27NonelasticCrossSectionTable(
            mt=3, energy_mev=(0.1, 20.0), sigma_barn=(3.0, 3.0)
        ),
    )


def _source(*, peak: bool = False, refined: bool = False) -> Spectrum1D:
    return Spectrum1D(
        x=(1.0, 7.0, 7.99, 8.0) if refined else (1.0, 7.0, 8.0),
        y=(0.0, 0.0, 0.99, 1.0) if refined else (0.0, 0.0, 1.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX if peak else Unit.DIFFERENTIAL_FLUENCE,
        quantity=(
            SpectrumQuantity.DIFFERENTIAL_FLUX
            if peak
            else SpectrumQuantity.DIFFERENTIAL_FLUENCE
        ),
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="analytic_narrow_exit_window",
    )


def _counts(spectrum: Spectrum1D) -> float:
    assert spectrum.bin_edges is not None
    return sum(
        density * (upper - lower)
        for density, lower, upper in zip(
            spectrum.y, spectrum.bin_edges, spectrum.bin_edges[1:]
        )
    )


def _analytic_narrow_count() -> float:
    # Exit lower edge is unchanged by the added collinear source point.
    # For u=8-E0 and w=Ebirth-Eexit_min, the source/depth integral is
    # integral_0^w (1-u)*(w-u) du = w**2/2-w**3/6.
    lower_w = 0.39 - 1.0 / sqrt(7.0)
    upper_w = 0.3901 - 1.0 / sqrt(7.0)
    averaged_integral = (
        (upper_w**3 - lower_w**3) / 6.0
        - (upper_w**4 - lower_w**4) / 24.0
    ) / (0.3901 - 0.39)
    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    attenuation = exp(-number_per_gram * 3.0 * BARN_CM2 * 3.0)
    return number_per_gram * 2.0 * BARN_CM2 * attenuation * averaged_integral


@pytest.mark.parametrize("depth_steps", [1, 2, 8])
@pytest.mark.parametrize("peak", [False, True])
@pytest.mark.parametrize("refined", [False, True])
def test_narrow_response_matches_analytic_integral_without_dense_depth_grid(
    depth_steps: int, peak: bool, refined: bool
) -> None:
    source = _source(peak=peak, refined=refined)
    result = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=source,
        stopping_table=_stopping(),
        kernel=_kernel(),
        thickness_g_cm2=3.0,
        depth_steps=depth_steps,
        incident_quadrature=IncidentEnergyQuadrature(
            relative_tolerance=1e-7,
            max_response_evaluations=2048,
            max_refinement_depth=4,
        ),
    )
    exact_count = _analytic_narrow_count()
    assert exact_count == pytest.approx(2.6559853528780986e-6, rel=2e-12)
    assert _counts(result) == pytest.approx(exact_count, rel=1e-7)
    assert result.y_unit is source.y_unit
    assert result.quantity is source.quantity
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    assert {
        PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
        PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
        PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
    }.issubset(result.incomplete_reasons)


def test_narrow_birth_edges_generate_both_depth_and_coupled_incident_events() -> None:
    source = _source()
    stopping = _stopping()
    kernel = _kernel()
    output_edges = log_bin_edges_from_centers(source.x)
    exit_events = _secondary_exit_depth_knots(
        output_edges_mev=output_edges,
        thickness_g_cm2=3.0,
        stopping_table=stopping,
        kernel=kernel,
    )
    assert exit_events == tuple(sorted(set(exit_events)))
    assert all(0.0 < depth < 3.0 for depth in exit_events)
    expected_depths = tuple(3.0 - (birth - 1.0 / sqrt(7.0)) for birth in (0.39, 0.3901))
    for depth in expected_depths:
        assert any(abs(actual - depth) < 1e-12 for actual in exit_events)

    incident_knots = _secondary_incident_energy_knots(kernel=kernel, stopping_table=stopping)
    boundaries = _point_source_quadrature_boundaries(
        source_nodes=source.x,
        incident_energy_knots=incident_knots,
        stopping_table=stopping,
        thickness_g_cm2=3.0,
        exit_depth_knots_g_cm2=exit_events,
    )
    for depth in expected_depths:
        # R(E)=E: when the reaction threshold at E=5 crosses the exit event,
        # the incident energy is E0=5+depth, inside the old missed interval.
        assert any(abs(actual - (5.0 + depth)) < 1e-12 for actual in boundaries)

    intervals = _depth_quadrature_intervals(
        entrance_energy_mev=8.0,
        thickness_g_cm2=3.0,
        depth_steps=1,
        stopping_table=stopping,
        incident_energy_knots_mev=incident_knots,
        exit_depth_knots_g_cm2=exit_events,
    )
    assert sum(upper - lower for lower, upper in intervals) == pytest.approx(3.0)
    for depth in expected_depths:
        assert not any(lower < depth - 1e-12 and upper > depth + 1e-12 for lower, upper in intervals)


def test_crossing_events_use_mass_ranges_not_energy_differences() -> None:
    stopping = _stopping(range_scale=2.0)
    kernel = _kernel(birth_edges=(2.0, 3.0))
    events = _secondary_exit_depth_knots(
        output_edges_mev=(1.0, 1.5),
        thickness_g_cm2=4.0,
        stopping_table=stopping,
        kernel=kernel,
    )
    # t-[2*Ebirth-2*Eexit] gives 2,3,0,1; endpoints are not interior events.
    assert all(0.0 < depth < 4.0 for depth in events)
    for depth in (1.0, 2.0, 3.0):
        assert any(abs(actual - depth) < 1e-12 for actual in events)
    boundaries = _point_source_quadrature_boundaries(
        source_nodes=(5.0, 8.0),
        incident_energy_knots=(0.1, 5.0, 20.0),
        stopping_table=stopping,
        thickness_g_cm2=4.0,
        exit_depth_knots_g_cm2=events,
    )
    for energy in (5.5, 6.0, 6.5):
        assert any(abs(actual - energy) < 1e-12 for actual in boundaries)


def test_exit_events_do_not_extrapolate_range_and_retain_upper_mapping_boundary() -> None:
    events = _secondary_exit_depth_knots(
        output_edges_mev=(0.05, 0.1, 0.3, 19.5, 20.0, 21.0),
        thickness_g_cm2=1.0,
        stopping_table=_stopping(),
        kernel=_kernel(birth_edges=(0.01, 0.2, 30.0)),
    )
    # Out-of-domain PDF/output edges must be ignored, never sent into strict
    # range interpolation. Still preserve .2 -> .1 and Rmax=20 -> exit=19.5.
    for depth in (0.9, 0.5):
        assert any(abs(actual - depth) < 1e-12 for actual in events)
    assert all(0.0 < depth < 1.0 for depth in events)


@pytest.mark.parametrize("depth_steps", [1, 2, 8])
def test_exit_events_are_clipped_to_active_primary_depth(depth_steps: int) -> None:
    intervals = _depth_quadrature_intervals(
        entrance_energy_mev=0.6,
        thickness_g_cm2=3.0,
        depth_steps=depth_steps,
        stopping_table=_stopping(),
        incident_energy_knots_mev=(0.1, 5.0, 20.0),
        exit_depth_knots_g_cm2=(0.0, 0.25, 0.5, 0.75, 3.0),
    )
    # The supported primary path stops at R(.6)-R(.1)=.5, not at the slab exit.
    assert intervals[0][0] == 0.0
    assert intervals[-1][1] == pytest.approx(0.5)
    assert all(0.0 <= lower < upper <= 0.5 for lower, upper in intervals)
    assert sum(upper - lower for lower, upper in intervals) == pytest.approx(0.5)
    assert not any(lower < 0.25 < upper for lower, upper in intervals)


def test_exit_events_include_every_independent_pdf_incident_slice() -> None:
    kernel = _kernel(birth_edges=(2.0, 3.0))
    product = replace(
        kernel.products[0],
        pdf_incident_energy_mev=(0.1, 8.0, 20.0),
        emitted_energy_mev_by_incident=((2.0, 3.0), (1.75, 2.25), (2.0, 3.0)),
        pdf_per_mev_by_incident=((1.0, 0.0), (2.0, 0.0), (1.0, 0.0)),
    )
    events = _secondary_exit_depth_knots(
        output_edges_mev=(1.0, 1.5),
        thickness_g_cm2=2.0,
        stopping_table=_stopping(),
        kernel=replace(kernel, products=(product,)),
    )
    # These events arise only from the middle PDF slice, not the yield grid.
    for depth in (0.75, 1.25, 1.75):
        assert any(abs(actual - depth) < 1e-12 for actual in events)


@pytest.mark.parametrize("outer_quadrature", [False, True])
def test_depth_knot_budget_falls_back_completely_without_retaining_arbitrary_prefix(
    outer_quadrature: bool,
) -> None:
    source = _source()
    stopping = _stopping()
    kernel = _kernel()
    events = _secondary_exit_depth_knots(
        output_edges_mev=log_bin_edges_from_centers(source.x),
        thickness_g_cm2=3.0,
        stopping_table=stopping,
        kernel=kernel,
    )
    assert len(events) > 1
    results = []
    for limit in (0, 1):
        result = calculate_secondary_proton_spectrum_through_al(
            incident_spectrum=source,
            stopping_table=stopping,
            kernel=kernel,
            thickness_g_cm2=3.0,
            depth_steps=1,
            depth_quadrature=DepthCrossingQuadrature(max_depth_knots=limit),
            incident_quadrature=(
                IncidentEnergyQuadrature(
                    max_response_evaluations=256, max_refinement_depth=2
                )
                if outer_quadrature
                else None
            ),
        )
        parameters = dict(result.transformation_chain[-1].parameters)
        assert parameters["depth_output_crossing_knots"] == (
            "budget_exceeded_legacy_partition_retained"
        )
        assert parameters["depth_output_crossing_count"] == "0"
        assert parameters["depth_output_crossing_max_knots"] == str(limit)
        assert PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON in result.incomplete_reasons
        assert PROTON_AL_SECONDARY_DEPTH_CROSSING_QUADRATURE_REASON in result.incomplete_reasons
        assert PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON in result.incomplete_reasons
        assert result.completeness is SpectrumCompleteness.INCOMPLETE
        if outer_quadrature:
            assert parameters["incident_quadrature_status"] == "incomplete"
            assert parameters["incident_crossing_partition"] == "incomplete_depth_partition"
            assert (
                PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON
                in result.incomplete_reasons
            )
        results.append(result)
    # Both limits are insufficient, so neither may retain a different prefix.
    assert results[0].y == results[1].y


def test_coupled_knot_budget_falls_back_only_outer_partition_without_prefix() -> None:
    source = _source()
    stopping = _stopping()
    kernel = _kernel()
    incident_knots = _secondary_incident_energy_knots(kernel=kernel, stopping_table=stopping)
    depth_events = _secondary_exit_depth_knots(
        output_edges_mev=log_bin_edges_from_centers(source.x),
        thickness_g_cm2=3.0,
        stopping_table=stopping,
        kernel=kernel,
    )
    base_boundaries = _point_source_quadrature_boundaries(
        source_nodes=source.x,
        incident_energy_knots=incident_knots,
        stopping_table=stopping,
        thickness_g_cm2=3.0,
    )
    full_boundaries = _point_source_quadrature_boundaries(
        source_nodes=source.x,
        incident_energy_knots=incident_knots,
        stopping_table=stopping,
        thickness_g_cm2=3.0,
        exit_depth_knots_g_cm2=depth_events,
    )
    assert len(full_boundaries) > len(base_boundaries) + 1
    results = []
    for limit in (0, 1):
        result = calculate_secondary_proton_spectrum_through_al(
            incident_spectrum=source,
            stopping_table=stopping,
            kernel=kernel,
            thickness_g_cm2=3.0,
            depth_steps=1,
            depth_quadrature=DepthCrossingQuadrature(max_coupled_incident_knots=limit),
            incident_quadrature=IncidentEnergyQuadrature(
                max_response_evaluations=256, max_refinement_depth=2
            ),
        )
        parameters = dict(result.transformation_chain[-1].parameters)
        assert parameters["depth_output_crossing_knots"] == (
            "all_supported_pdf_range_attenuation_crossings"
        )
        assert int(parameters["depth_output_crossing_count"]) == len(depth_events)
        assert parameters["incident_crossing_partition"] == (
            "budget_exceeded_legacy_partition_retained"
        )
        assert int(parameters["incident_partition_intervals"]) == len(base_boundaries) - 1
        assert parameters["incident_quadrature_status"] == "incomplete"
        assert PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON in result.incomplete_reasons
        assert (
            PROTON_AL_SECONDARY_DEPTH_CROSSING_QUADRATURE_REASON
            not in result.incomplete_reasons
        )
        assert (
            PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON
            in result.incomplete_reasons
        )
        assert PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON in result.incomplete_reasons
        results.append(result)
    assert results[0].y == results[1].y


@pytest.mark.parametrize("outer_quadrature", [False, True])
def test_zero_thickness_does_not_activate_partition_budgets_or_blockers(
    outer_quadrature: bool,
) -> None:
    source = _source()
    result = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=source,
        stopping_table=_stopping(),
        kernel=_kernel(),
        thickness_g_cm2=0.0,
        depth_quadrature=DepthCrossingQuadrature(
            max_depth_knots=0, max_coupled_incident_knots=0
        ),
        incident_quadrature=IncidentEnergyQuadrature() if outer_quadrature else None,
    )
    assert result.y == (0.0,) * len(source.x)
    assert result.incomplete_reasons == ()
    assert result.completeness is source.completeness
    assert result.model_reference is source.model_reference
    assert result.transformation_chain[-1].parameters == (("thickness_g_cm2", "0"),)


@pytest.mark.parametrize("field", ["max_depth_knots", "max_coupled_incident_knots"])
@pytest.mark.parametrize("value", [-1, -0.1, 1.5, float("nan"), float("inf"), True, "1"])
def test_crossing_budget_options_require_nonnegative_integers(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        DepthCrossingQuadrature(**{field: value})  # type: ignore[arg-type]


def test_declared_mt3_support_boundaries_are_events_even_when_not_tabulated_nodes() -> None:
    stopping = _stopping()
    kernel = _kernel(birth_edges=(2.0, 3.0))
    assert kernel.nonelastic_cross_section is not None
    kernel = replace(
        kernel,
        nonelastic_cross_section=replace(
            kernel.nonelastic_cross_section,
            incident_energy_support=TendlIncidentEnergySupport(
                minimum_mev=0.1,
                maximum_mev=2.75,
                dataset_ref="synthetic_declared_mt3_subset_of_tabulated_domain",
            ),
        ),
    )
    events = _secondary_exit_depth_knots(
        output_edges_mev=(1.0, 1.5),
        thickness_g_cm2=3.0,
        stopping_table=stopping,
        kernel=kernel,
    )
    # MT3 support changes at a boundary absent from its (.1,20) data grid.
    # The current all-or-nothing attenuation mapping must split there too.
    for depth in (3.0 - (2.75 - 1.0), 3.0 - (2.75 - 1.5)):
        assert any(abs(actual - depth) < 1e-12 for actual in events)
    incident_knots = _secondary_incident_energy_knots(kernel=kernel, stopping_table=stopping)
    assert 2.75 in incident_knots
