"""Independent support tests for the conservative per-bin zero-response proof.

For the synthetic R(E)=E table, t=1 and E_exit in [1,2] imply that
every possible birth energy is in [1,3]. Incident E0 in [5,6] traverses
the complete supported primary-energy interval [4,6]. No quadrature
sampling or production cross-section zero is needed for these proofs.
"""

from dataclasses import replace

import pytest

from radar.nuclear.tendl import TendlIncidentEnergySupport
from radar.shielding.proton_al import ProtonAlRangeEnergyTable
import radar.shielding.proton_al_secondary as secondary
from radar.shielding.proton_al_secondary import (
    Al27CrossSectionTable,
    SecondaryProtonKernel,
    SecondaryProtonProductDistribution,
)
from radar.shielding.proton_al_survival import (
    Al27CrossSectionDiscontinuity,
    Al27NonelasticCrossSectionTable,
)


def _stopping() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(0.1, 20.0),
        range_g_cm2=(0.1, 20.0),
        stopping_mev_cm2_g=(1.0, 1.0),
    )


def _kernel(
    *,
    birth_edges: tuple[float, ...] = (10.0, 11.0),
    densities: tuple[float, ...] = (1.0, 0.0),
    pdf_incident: tuple[float, ...] = (0.1, 20.0),
) -> SecondaryProtonKernel:
    return SecondaryProtonKernel(
        products=(
            SecondaryProtonProductDistribution(
                mt=5,
                product_index=0,
                incident_energy_mev=(0.1, 20.0),
                yield_value=(1.0, 1.0),
                pdf_incident_energy_mev=pdf_incident,
                emitted_energy_mev_by_incident=(birth_edges,) * len(pdf_incident),
                pdf_per_mev_by_incident=(densities,) * len(pdf_incident),
            ),
        ),
        cross_sections_by_mt={
            5: Al27CrossSectionTable(mt=5, energy_mev=(0.1, 20.0), sigma_barn=(2.0, 2.0)),
        },
    )


def _prove(
    kernel: SecondaryProtonKernel | None = None,
    *,
    entrance_lower: float = 5.0,
    entrance_upper: float = 6.0,
    output_edges: tuple[float, float] = (1.0, 2.0),
    thickness: float = 1.0,
    stopping: ProtonAlRangeEnergyTable | None = None,
) -> bool:
    return secondary._secondary_bin_geometry_is_proven_zero(
        entrance_lower_mev=entrance_lower,
        entrance_upper_mev=entrance_upper,
        output_edges_mev=output_edges,
        thickness_g_cm2=thickness,
        stopping_table=_stopping() if stopping is None else stopping,
        kernel=_kernel() if kernel is None else kernel,
    )


@pytest.mark.parametrize("birth_edges", [(10.0, 11.0), (0.2, 0.4)])
def test_fully_supported_disjoint_positive_histogram_proves_zero(birth_edges) -> None:
    assert _prove(_kernel(birth_edges=birth_edges))


@pytest.mark.parametrize("birth_edges", [(1.5, 2.5), (2.9999999999, 4.0), (0.5, 1.0000000001)])
def test_any_positive_width_overlap_cannot_be_certified_zero(birth_edges) -> None:
    assert not _prove(_kernel(birth_edges=birth_edges))


def test_intermediate_independent_pdf_slice_prevents_false_zero() -> None:
    kernel = _kernel(pdf_incident=(0.1, 5.5, 20.0))
    product = replace(
        kernel.products[0],
        emitted_energy_mev_by_incident=((10.0, 11.0), (2.0, 2.1), (10.0, 11.0)),
        pdf_per_mev_by_incident=((1.0, 0.0), (10.0, 0.0), (1.0, 0.0)),
    )
    assert not _prove(replace(kernel, products=(product,)))


def test_all_products_are_checked_not_just_first_disjoint_product() -> None:
    kernel = _kernel()
    second = replace(
        kernel.products[0],
        product_index=1,
        emitted_energy_mev_by_incident=((2.0, 2.1),) * 2,
        pdf_per_mev_by_incident=((10.0, 0.0),) * 2,
    )
    assert not _prove(replace(kernel, products=(*kernel.products, second)))


def test_missing_primary_incident_pdf_support_is_not_a_geometry_zero() -> None:
    kernel = _kernel(pdf_incident=(4.5, 20.0))
    assert kernel.incident_energy_support is not None
    assert kernel.incident_energy_support.contains(4.0)
    # The kernel support says yes but the reachable interval [4,4.5) has
    # no tabulated/interpolatable PDF: its possible birth support is unknown.
    assert not _prove(kernel)


def test_pdf_support_endpoint_is_closed_and_small_uncovered_tail_is_not_ignored() -> None:
    assert _prove(_kernel(pdf_incident=(4.0, 20.0)))
    assert not _prove(_kernel(pdf_incident=(4.0000000001, 20.0)))


def test_missing_pdf_support_in_second_product_also_prevents_proof() -> None:
    kernel = _kernel()
    second = replace(kernel.products[0], product_index=1, pdf_incident_energy_mev=(4.5, 20.0))
    assert not _prove(replace(kernel, products=(*kernel.products, second)))


@pytest.mark.parametrize(
    "minimum,maximum",
    [(4.5, 20.0), (0.1, 5.5)],
)
def test_primary_path_must_stay_inside_declared_kernel_support(minimum, maximum) -> None:
    kernel = replace(
        _kernel(),
        incident_energy_support=TendlIncidentEnergySupport(
            minimum_mev=minimum, maximum_mev=maximum, dataset_ref="synthetic_subset"
        ),
    )
    assert not _prove(kernel)


@pytest.mark.parametrize(
    "entrance_lower,entrance_upper",
    [(0.05, 6.0), (5.0, 21.0), (0.5, 1.0)],
)
def test_unsupported_entrance_or_stopped_primary_is_not_proven_zero(
    entrance_lower, entrance_upper
) -> None:
    assert not _prove(entrance_lower=entrance_lower, entrance_upper=entrance_upper)


@pytest.mark.parametrize("output_edges", [(0.05, 1.0), (19.0, 21.0)])
def test_whole_output_bin_must_be_supported_without_clipping(output_edges) -> None:
    assert not _prove(output_edges=output_edges)


def test_full_birth_upper_bound_must_remain_inside_range_domain() -> None:
    assert _prove(output_edges=(18.0, 19.0))  # R(19)+t = Rmax exactly.
    assert not _prove(output_edges=(18.0, 19.5))  # A partial birth path would be missing.


@pytest.mark.parametrize("birth_edges", [(3.0, 4.0), (0.5, 1.0)])
def test_histogram_boundary_touching_has_zero_measure(birth_edges) -> None:
    assert _prove(_kernel(birth_edges=birth_edges))


def test_zero_density_gap_is_allowed_despite_overlapping_total_histogram_envelope() -> None:
    # Positive [0.2,1) and [3,4), but exactly zero on the birth envelope [1,3].
    kernel = _kernel(birth_edges=(0.2, 1.0, 3.0, 4.0), densities=(1.0, 0.0, 1.0, 0.0))
    assert _prove(kernel)


def test_last_pdf_ordinate_is_not_an_extra_histogram_interval() -> None:
    # A final ordinate has no following edge, hence no area or interval.
    kernel = _kernel(birth_edges=(0.2, 1.0, 4.0), densities=(1.0, 0.0, 99.0))
    assert _prove(kernel)


def test_geometry_zero_does_not_require_mt3_attenuation_data() -> None:
    kernel = _kernel()
    assert kernel.nonelastic_cross_section is None
    assert _prove(kernel)
    assert _prove(
        replace(
            kernel,
            nonelastic_cross_section=Al27NonelasticCrossSectionTable(
                mt=3, energy_mev=(0.1, 20.0), sigma_barn=(3.0, 3.0)
            ),
        )
    )


def test_birth_envelope_uses_mass_ranges_not_energy_plus_thickness() -> None:
    # R=2E, t=1 => max birth from Eexit_hi=2 is 2.5, not 3 MeV.
    stopping = ProtonAlRangeEnergyTable(
        energy_mev=(0.1, 20.0),
        range_g_cm2=(0.2, 40.0),
        stopping_mev_cm2_g=(0.5, 0.5),
    )
    assert _prove(_kernel(birth_edges=(2.6, 2.8)), stopping=stopping)
    assert not _prove(_kernel(birth_edges=(2.4, 2.6)), stopping=stopping)


@pytest.mark.parametrize("overlapping_pdf_index", [0, 1, 4, 5])
def test_pdf_hat_outside_whole_reachable_interval_cannot_prevent_zero_proof(
    overlapping_pdf_index: int,
) -> None:
    # Reachable [4,6] uses only the rows at 4 and 7. The row at 3 has
    # exactly zero weight at the lower endpoint, not an active interval.
    grid = (0.1, 3.0, 4.0, 7.0, 10.0, 20.0)
    kernel = _kernel(pdf_incident=grid)
    grids = list(kernel.products[0].emitted_energy_mev_by_incident)
    grids[overlapping_pdf_index] = (2.0, 2.1)
    product = replace(kernel.products[0], emitted_energy_mev_by_incident=tuple(grids))
    assert _prove(replace(kernel, products=(product,)))


@pytest.mark.parametrize("overlapping_pdf_index", [2, 3])
def test_every_pdf_hat_intersecting_reachable_interval_must_be_retained(
    overlapping_pdf_index: int,
) -> None:
    # Row 7 is outside the reachable [4,6] but has positive interpolation
    # weight inside it; filtering only tabulated energies in [4,6] is unsafe.
    grid = (0.1, 3.0, 4.0, 7.0, 10.0, 20.0)
    kernel = _kernel(pdf_incident=grid)
    grids = list(kernel.products[0].emitted_energy_mev_by_incident)
    grids[overlapping_pdf_index] = (2.0, 2.1)
    product = replace(kernel.products[0], emitted_energy_mev_by_incident=tuple(grids))
    assert not _prove(replace(kernel, products=(product,)))


def test_upper_pdf_support_hat_touching_at_zero_weight_can_be_excluded() -> None:
    grid = (0.1, 4.0, 7.0, 10.0, 20.0)
    kernel = _kernel(pdf_incident=grid)
    grids = list(kernel.products[0].emitted_energy_mev_by_incident)
    grids[3] = (2.0, 2.1)  # Hat at 10 has zero weight at E=7.
    product = replace(kernel.products[0], emitted_energy_mev_by_incident=tuple(grids))
    modified = replace(kernel, products=(product,))
    assert _prove(modified, entrance_upper=7.0)
    assert not _prove(modified, entrance_upper=7.0000000001)


def test_identically_zero_yield_allows_inactive_product_without_reachable_pdf() -> None:
    kernel = _kernel()
    inactive = replace(
        kernel.products[0],
        product_index=1,
        yield_value=(0.0, 0.0),
        pdf_incident_energy_mev=(10.0, 20.0),
    )
    assert _prove(replace(kernel, products=(*kernel.products, inactive)))


def _kernel_with_threshold_inactive_pdf_product() -> SecondaryProtonKernel:
    kernel = _kernel()
    extra = replace(
        kernel.products[0], mt=16, product_index=1, pdf_incident_energy_mev=(10.0, 20.0)
    )
    cross_section = Al27CrossSectionTable(
        mt=16,
        energy_mev=(0.1, 7.0, 20.0),
        sigma_barn=(0.0, 2.0, 2.0),
        discontinuities=(
            Al27CrossSectionDiscontinuity(
                energy_mev=7.0, left_sigma_barn=0.0, right_sigma_barn=2.0
            ),
        ),
    )
    return replace(
        kernel,
        products=(*kernel.products, extra),
        cross_sections_by_mt={**kernel.cross_sections_by_mt, 16: cross_section},
    )


@pytest.mark.parametrize("entrance_upper", [6.0, 7.0])
def test_reaction_below_one_sided_threshold_needs_no_missing_pdf(entrance_upper: float) -> None:
    assert _prove(_kernel_with_threshold_inactive_pdf_product(), entrance_upper=entrance_upper)


def test_any_active_width_above_threshold_requires_pdf_support() -> None:
    assert not _prove(_kernel_with_threshold_inactive_pdf_product(), entrance_upper=7.0000000001)


def test_narrow_positive_yield_between_zero_samples_prevents_inactive_product_proof() -> None:
    kernel = _kernel()
    product = replace(
        kernel.products[0],
        incident_energy_mev=(0.1, 4.8, 4.9, 5.0, 20.0),
        yield_value=(0.0, 0.0, 1.0, 0.0, 0.0),
        pdf_incident_energy_mev=(10.0, 20.0),
    )
    # Sampling just endpoints and midpoint of [4,6] sees zeros, but an
    # entire positive triangular yield lies inside that reachable interval.
    assert not _prove(replace(kernel, products=(product,)))


@pytest.mark.parametrize("limited_factor", ["yield", "cross_section"])
def test_declared_factor_domain_is_checked_before_an_inactive_channel_is_skipped(
    limited_factor: str,
) -> None:
    kernel = _kernel()
    limited_support = TendlIncidentEnergySupport(
        minimum_mev=4.5, maximum_mev=20.0, dataset_ref="synthetic_limited_factor"
    )
    broad_kernel_support = TendlIncidentEnergySupport(
        minimum_mev=0.1, maximum_mev=20.0, dataset_ref="synthetic_explicit_kernel_domain"
    )
    if limited_factor == "yield":
        product = replace(
            kernel.products[0], yield_value=(0.0, 0.0), incident_energy_support=limited_support
        )
        kernel = replace(kernel, products=(product,), incident_energy_support=broad_kernel_support)
    else:
        cross_section = replace(
            kernel.cross_sections_by_mt[5],
            sigma_barn=(0.0, 0.0),
            incident_energy_support=limited_support,
        )
        kernel = replace(
            kernel,
            cross_sections_by_mt={5: cross_section},
            incident_energy_support=broad_kernel_support,
        )
    # The actual factor is undefined on [4,4.5); neither a false proof nor
    # an out-of-domain interpolation exception is an acceptable outcome.
    assert not _prove(kernel)
