"""Independent geometry checks for primary-cell-conditioned PDF crossings.

The analytic reference is R(E)=cE: a PDF edge b crossing output edge e
occurs at d=t-c(b-e), and its entrance activation endpoints are a+d/c.
These are partition checks, not validation of the transport physics.
"""

from dataclasses import replace
from math import inf, nextafter

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.nuclear.tendl import TendlIncidentEnergySupport
from radar.shielding.proton_al import ProtonAlRangeEnergyTable
import radar.shielding.proton_al_secondary as secondary
from radar.shielding.proton_al_secondary import (
    Al27CrossSectionTable,
    DepthCrossingQuadrature,
    SecondaryProtonKernel,
    SecondaryProtonProductDistribution,
)
from radar.shielding.proton_al_survival import Al27NonelasticCrossSectionTable


def _stopping(scale: float = 1.0) -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(0.125, 20.0),
        range_g_cm2=(0.125 * scale, 20.0 * scale),
        stopping_mev_cm2_g=(1.0 / scale,) * 2,
    )


def _kernel() -> SecondaryProtonKernel:
    return SecondaryProtonKernel(
        products=(
            SecondaryProtonProductDistribution(
                mt=5,
                product_index=0,
                incident_energy_mev=(0.125, 20.0),
                yield_value=(1.0, 1.0),
                pdf_incident_energy_mev=(0.125, 4.0, 6.0, 20.0),
                emitted_energy_mev_by_incident=(
                    (0.5, 1.5), (2.0, 3.0), (8.0, 9.0), (12.0, 13.0),
                ),
                pdf_per_mev_by_incident=((1.0, 0.0),) * 4,
            ),
        ),
        cross_sections_by_mt={
            5: Al27CrossSectionTable(mt=5, energy_mev=(0.125, 20.0), sigma_barn=(2.0, 2.0)),
        },
    )


def _source(x: tuple[float, ...], y: tuple[float, ...]) -> Spectrum1D:
    return Spectrum1D(
        x=x, y=y, x_unit=Unit.MEV, y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON, source=RadiationSource.SEP, model="synthetic_boundary_source",
    )


def _cells(
    *,
    knots: tuple[float, ...],
    source_ranges: tuple[tuple[float, float], ...] = ((0.125, 20.0),),
    kernel: SecondaryProtonKernel | None = None,
    scale: float = 1.0,
    thickness: float = 1.0,
) -> tuple[secondary._PrimaryPdfCrossingCell, ...]:
    return secondary._prepare_primary_pdf_crossing_cells(
        incident_energy_knots=knots, source_range_intervals=source_ranges,
        thickness_g_cm2=thickness, stopping_table=_stopping(scale),
        kernel=_kernel() if kernel is None else kernel,
    )


def _depth_boundaries(
    entrance: float,
    cells: tuple[secondary._PrimaryEnergyDepthCrossings, ...],
    *,
    table: ProtonAlRangeEnergyTable | None = None,
    thickness: float = 1.3,
) -> set[float]:
    intervals = secondary._depth_quadrature_intervals(
        entrance_energy_mev=entrance, thickness_g_cm2=thickness, depth_steps=1,
        stopping_table=_stopping() if table is None else table,
        incident_energy_knots_mev=(), active_exit_cells=cells,
    )
    return {boundary for interval in intervals for boundary in interval}


def test_positive_source_support_keeps_linear_zero_endpoint_segments_and_gaps() -> None:
    source = _source(
        (0.0625, 0.125, 1.0, 2.0, 3.0, 4.0, 21.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0),
    )
    # The [2,3] interval is exactly zero, but [1,2] and [3,4] are not.
    # Range support clips the second interval at E=20; units are g/cm2.
    assert secondary._positive_source_range_intervals(source, _stopping(2.0)) == (
        (0.25, 4.0), (6.0, 40.0),
    )


def test_zero_source_has_no_positive_support() -> None:
    assert secondary._positive_source_range_intervals(
        _source((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)), _stopping()
    ) == ()


def test_source_reachability_uses_mass_range_not_energy_minus_thickness() -> None:
    cells = _cells(
        knots=(0.125, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0),
        source_ranges=((10.0, 12.0),), scale=2.0, thickness=2.0,
    )
    # E0=[5,6], R=2E, t=2 -> primary support [4,6], not [3,6].
    # Endpoint-touching cells may be retained conservatively.
    assert all(cell.lower_range_g_cm2 >= 6.0 for cell in cells)
    assert any((cell.lower_range_g_cm2, cell.upper_range_g_cm2) == (8.0, 10.0)
               for cell in cells)
    assert not _cells(knots=(0.125, 20.0), source_ranges=())


def test_active_pdf_hats_exclude_remote_slices_but_keep_both_local_neighbors() -> None:
    cell, = _cells(knots=(4.5, 5.5))
    assert cell.birth_knots_mev == (2.0, 3.0, 8.0, 9.0)
    assert not cell.pdf_support_missing


def test_exact_pdf_grid_endpoint_does_not_activate_a_remote_hat() -> None:
    cell, = _cells(knots=(4.0, 6.0))
    assert cell.birth_knots_mev == (2.0, 3.0, 8.0, 9.0)


def test_all_original_primary_cell_endpoints_survive_identical_pdf_pairs() -> None:
    cells = _cells(knots=(4.0, 4.5, 5.5, 6.0), scale=2.0)
    assert [(cell.lower_range_g_cm2, cell.upper_range_g_cm2) for cell in cells] == [
        (8.0, 9.0), (9.0, 11.0), (11.0, 12.0),
    ]
    assert len({cell.birth_knots_mev for cell in cells}) == 1


def test_missing_pdf_support_keeps_all_stored_edges_and_explicit_unknown() -> None:
    kernel = _kernel()
    product = replace(kernel.products[0], pdf_incident_energy_mev=(3.0, 4.0, 6.0, 20.0))
    cell, = _cells(knots=(1.0, 2.0), kernel=replace(kernel, products=(product,)))
    assert cell.pdf_support_missing
    assert cell.birth_knots_mev == (0.5, 1.5, 2.0, 3.0, 8.0, 9.0, 12.0, 13.0)


def test_proven_inactive_channel_may_skip_its_missing_pdf_support() -> None:
    kernel = _kernel()
    product = replace(kernel.products[0], pdf_incident_energy_mev=(3.0, 4.0, 6.0, 20.0))
    cross_section = replace(kernel.cross_sections_by_mt[5], sigma_barn=(0.0, 0.0))
    assert _cells(
        knots=(1.0, 2.0),
        kernel=replace(kernel, products=(product,), cross_sections_by_mt={5: cross_section}),
    ) == ()


def test_adjacent_float_primary_cell_with_positive_linear_yield_is_not_pruned() -> None:
    lower, upper = 1.0, nextafter(1.0, inf)
    kernel = _kernel()
    product = replace(
        kernel.products[0], incident_energy_mev=(lower, upper, 20.0),
        yield_value=(0.0, 1.0, 1.0),
    )
    # The mathematical linear yield is positive on (lower, upper), although
    # a floating midpoint rounds to lower. A sample there cannot prove zero.
    cells = _cells(knots=(lower, upper), kernel=replace(kernel, products=(product,)))
    assert len(cells) == 1
    assert cells[0].birth_knots_mev == (0.5, 1.5, 2.0, 3.0)


def test_individual_declared_domains_are_explicit_primary_knots() -> None:
    kernel = _kernel()
    product = replace(kernel.products[0], incident_energy_support=TendlIncidentEnergySupport(
        minimum_mev=1.25, maximum_mev=18.5, dataset_ref="synthetic_product_subset",
    ))
    cross_section = replace(
        kernel.cross_sections_by_mt[5], incident_energy_support=TendlIncidentEnergySupport(
            minimum_mev=2.25, maximum_mev=17.5, dataset_ref="synthetic_mf3_subset",
        ),
    )
    kernel = replace(
        kernel, products=(product,), cross_sections_by_mt={5: cross_section},
        incident_energy_support=None,
    )
    knots = secondary._active_cell_incident_energy_knots(kernel=kernel, stopping_table=_stopping())
    assert {1.25, 2.25, 17.5, 18.5} <= set(knots)


def test_conditional_depths_and_outer_endpoints_use_mass_range_units() -> None:
    cells = (secondary._PrimaryPdfCrossingCell(10.0, 12.0, (2.0, 3.0), False),)
    partition = secondary._prepare_active_exit_depth_partition(
        output_edges_mev=(1.0, 1.5), thickness_g_cm2=4.0, stopping_table=_stopping(2.0),
        kernel=_kernel(), options=DepthCrossingQuadrature(), primary_cells=cells,
        source_range_intervals=((10.0, 20.0),),
    )
    assert not partition.budget_exceeded
    assert partition.knots_g_cm2 == ()
    assert partition.active_cells == (
        secondary._PrimaryEnergyDepthCrossings(10.0, 12.0, (1.0, 2.0, 3.0)),
    )
    assert partition.unique_depth_count == 3
    # A single d=1.25 g/cm2 yields E0=(Rcell+1.25)/2, not Ecell+1.25.
    boundaries = secondary._point_source_quadrature_boundaries(
        source_nodes=(5.0, 10.0), incident_energy_knots=(), stopping_table=_stopping(2.0),
        thickness_g_cm2=4.0,
        active_exit_cells=(secondary._PrimaryEnergyDepthCrossings(10.0, 12.0, (1.25,)),),
    )
    assert boundaries == (5.0, 5.625, 6.625, 10.0)


def test_adjacent_cells_with_same_event_keep_their_shared_outer_endpoint() -> None:
    cells = (
        secondary._PrimaryEnergyDepthCrossings(10.0, 12.0, (1.25,)),
        secondary._PrimaryEnergyDepthCrossings(12.0, 14.0, (1.25,)),
    )
    boundaries = secondary._point_source_quadrature_boundaries(
        source_nodes=(5.0, 10.0), incident_energy_knots=(), stopping_table=_stopping(2.0),
        thickness_g_cm2=4.0, active_exit_cells=cells,
    )
    assert boundaries == pytest.approx((5.0, 5.625, 6.625, 7.625, 10.0), rel=0, abs=2e-15)


@pytest.mark.parametrize("entrance,expected", [(4.75, True), (7.0, False), (8.75, True)])
def test_same_depth_event_in_disjoint_cells_is_not_globally_activated(entrance, expected) -> None:
    cells = (
        secondary._PrimaryEnergyDepthCrossings(4.0, 5.0, (0.5,)),
        secondary._PrimaryEnergyDepthCrossings(8.0, 9.0, (0.5,)),
    )
    assert (0.5 in _depth_boundaries(entrance, cells)) is expected


def test_outer_couplings_prune_zero_source_gap_but_keep_original_source_nodes() -> None:
    boundaries = secondary._point_source_quadrature_boundaries(
        source_nodes=(1.0, 3.0, 5.0, 7.0), incident_energy_knots=(), stopping_table=_stopping(),
        thickness_g_cm2=1.0, source_range_intervals=((1.0, 3.0), (5.0, 7.0)),
        active_exit_cells=(
            secondary._PrimaryEnergyDepthCrossings(1.0, 2.0, (0.25,)),
            secondary._PrimaryEnergyDepthCrossings(3.0, 4.0, (0.25,)),
            secondary._PrimaryEnergyDepthCrossings(5.0, 6.0, (0.25,)),
        ),
    )
    assert boundaries == (1.0, 1.25, 2.25, 3.0, 5.0, 5.25, 6.25, 7.0)


@pytest.mark.parametrize("edges,expected", [((1.0, 1.5), (1.25, 1.75)),
                                           ((18.0, 19.5), (1.0, 2.5))])
def test_mt3_declared_support_and_range_max_remain_unconditional_without_pdf_cells(
    edges, expected,
) -> None:
    kernel = replace(_kernel(), nonelastic_cross_section=Al27NonelasticCrossSectionTable(
        mt=3, energy_mev=(0.125, 20.0), sigma_barn=(2.0, 2.0),
        incident_energy_support=TendlIncidentEnergySupport(
            minimum_mev=0.125, maximum_mev=2.75, dataset_ref="synthetic_mt3_subset",
        ),
    ))
    partition = secondary._prepare_active_exit_depth_partition(
        output_edges_mev=edges, thickness_g_cm2=3.0, stopping_table=_stopping(),
        kernel=kernel, options=DepthCrossingQuadrature(), primary_cells=(),
        source_range_intervals=(),
    )
    assert not partition.budget_exceeded
    assert partition.knots_g_cm2 == expected
    assert partition.active_cells == ()


def test_common_depth_events_couple_to_every_primary_knot_not_only_active_pdf_cells() -> None:
    boundaries = secondary._point_source_quadrature_boundaries(
        source_nodes=(1.0, 10.0), incident_energy_knots=(2.0, 4.0, 6.0),
        stopping_table=_stopping(), thickness_g_cm2=2.0,
        exit_depth_knots_g_cm2=(0.375,), active_exit_cells=(),
    )
    assert {2.375, 4.375, 6.375} <= set(boundaries)


def test_pdf_edge_crossing_an_internal_exit_survival_knot_is_not_lost() -> None:
    kernel = replace(_kernel(), nonelastic_cross_section=Al27NonelasticCrossSectionTable(
        mt=3, energy_mev=(0.125, 1.5, 20.0), sigma_barn=(1.0, 4.0, 2.0),
    ))
    partition = secondary._prepare_active_exit_depth_partition(
        output_edges_mev=(1.0, 3.0), thickness_g_cm2=2.0, stopping_table=_stopping(),
        kernel=kernel, options=DepthCrossingQuadrature(),
        primary_cells=(secondary._PrimaryPdfCrossingCell(4.0, 5.0, (2.75,), False),),
        source_range_intervals=((4.0, 7.0),),
    )
    assert not partition.budget_exceeded
    # PDF birth edge 2.75 crosses exit-bin edge 1 at d=.25 and the
    # internal MT3 knot 1.5 at d=.75, although neither is a primary knot.
    cell, = partition.active_cells
    assert cell.knots_g_cm2 == (0.25, 0.75)
    assert partition.knots_g_cm2 == (1.5,)


def test_same_unique_event_in_two_cells_counts_as_two_active_records() -> None:
    # d=1 and d=1.5 occur in both cells; unique count=2, records=4.
    cells = (
        secondary._PrimaryPdfCrossingCell(4.0, 5.0, (2.0,), False),
        secondary._PrimaryPdfCrossingCell(8.0, 9.0, (2.0,), False),
    )
    kwargs = dict(
        output_edges_mev=(1.0, 1.5), thickness_g_cm2=2.0, stopping_table=_stopping(),
        kernel=_kernel(), primary_cells=cells, source_range_intervals=((0.125, 20.0),),
    )
    complete = secondary._prepare_active_exit_depth_partition(
        **kwargs, options=DepthCrossingQuadrature(max_depth_knots=2, max_active_cell_records=4),
    )
    assert not complete.budget_exceeded
    assert complete.unique_depth_count == 2
    assert len(complete.active_cells) == 2
    overflow = secondary._prepare_active_exit_depth_partition(
        **kwargs, options=DepthCrossingQuadrature(max_depth_knots=2, max_active_cell_records=3),
    )
    assert overflow.budget_exceeded
    assert overflow.knots_g_cm2 == ()
    assert overflow.active_cells == ()  # Never silently keep an arbitrary prefix.


@pytest.mark.parametrize("cell_side", ["lower", "upper"])
@pytest.mark.parametrize("direction", [-inf, None, inf])
def test_float_neighbors_of_closed_cell_endpoint_keep_event(cell_side, direction) -> None:
    cell = secondary._PrimaryEnergyDepthCrossings(4.0, 5.0, (0.5,))
    entrance = 4.5 if cell_side == "lower" else 5.5
    if direction is not None:
        entrance = nextafter(entrance, direction)
    assert 0.5 in _depth_boundaries(entrance, (cell,))


@pytest.mark.parametrize("entrance", [4.5 - 1e-9, 5.5 + 1e-9])
def test_clearly_outside_cell_does_not_gain_event_from_endpoint_tolerance(entrance) -> None:
    cell = secondary._PrimaryEnergyDepthCrossings(4.0, 5.0, (0.5,))
    assert 0.5 not in _depth_boundaries(entrance, (cell,))


@pytest.mark.parametrize("endpoint,depth", [(4.0, 1e-8), (400.0, 1e-10), (1e9, 1e-5)])
@pytest.mark.parametrize("cell_side", ["lower", "upper"])
def test_tiny_depth_next_to_large_range_survives_inverse_mapping_cancellation(
    endpoint, depth, cell_side,
) -> None:
    table = ProtonAlRangeEnergyTable(
        energy_mev=(0.1, 2e9), range_g_cm2=(0.1, 2e9), stopping_mev_cm2_g=(1.0, 1.0),
    )
    cell = secondary._PrimaryEnergyDepthCrossings(
        endpoint if cell_side == "lower" else endpoint - 1.0,
        endpoint + 1.0 if cell_side == "lower" else endpoint,
        (depth,),
    )
    # Form an entrance boundary through the same R-inverse map used by the
    # outer partition; expanding only the final small difference is unsafe.
    entrance = table.energy_at_range(endpoint + depth)
    assert depth in _depth_boundaries(entrance, (cell,), table=table)


@pytest.mark.parametrize("direction", [-inf, None, inf])
def test_source_range_endpoint_neighbors_are_not_pruned(direction) -> None:
    endpoint = 4.5 if direction is None else nextafter(4.5, direction)
    assert secondary._ranges_intersect(endpoint, endpoint, ((4.5, 5.5),))
    assert not secondary._ranges_intersect(4.5 - 1e-9, 4.5 - 1e-9, ((4.5, 5.5),))
