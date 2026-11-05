"""Independent tests of exact, mapping-local birth survival partition reuse."""

from dataclasses import replace
from math import exp, inf, nextafter
from typing import Any

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import SpectrumCompleteness, TemporalBasis
from radar.nuclear.tendl import TendlIncidentEnergySupport
from radar.shielding import proton_al_secondary as secondary
from radar.shielding.proton_al_secondary import DepthCrossingQuadrature, IncidentEnergyQuadrature
from radar.shielding.proton_al_survival import (
    AL27_MOLAR_MASS_G_MOL,
    AVOGADRO_CONSTANT_MOL_INV,
    BARN_CM2,
    Al27CrossSectionDiscontinuity,
    Al27NonelasticCrossSectionTable,
)
from test_proton_secondary_depth_crossings import (
    _analytic_narrow_count, _counts, _kernel, _source, _stopping,
)


STRATEGIES = ("global_union", "per_output_bin", "per_output_bin_active_cells")


def _calculate(
    source: Spectrum1D,
    *,
    reuse: bool,
    strategy: str = "per_output_bin_active_cells",
    screening: bool = False,
    kernel: secondary.SecondaryProtonKernel | None = None,
    thickness: float = 3.0,
    budget: int = 16_384,
) -> Spectrum1D:
    return secondary.calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=source,
        stopping_table=_stopping(),
        kernel=kernel or _kernel(),
        thickness_g_cm2=thickness,
        depth_steps=2,
        depth_quadrature=DepthCrossingQuadrature(
            max_depth_knots=4096, max_coupled_incident_knots=16_384,
        ),
        incident_quadrature=IncidentEnergyQuadrature(
            partition_strategy=strategy,
            prequadrature_screening=screening,
            reuse_birth_survival_partition=reuse,
            relative_tolerance=1e-7,
            max_response_evaluations=budget,
            max_refinement_depth=4,
        ),
    )


def _assert_unchanged(old: Spectrum1D, new: Spectrum1D) -> None:
    # This acceleration claims exact reuse, not a different/refined quadrature.
    assert new.x == old.x
    assert new.y == old.y
    assert new.bin_edges == old.bin_edges
    assert new.y_unit is old.y_unit
    assert new.quantity is old.quantity
    assert new.temporal_basis is old.temporal_basis
    assert new.mission_duration_s == old.mission_duration_s
    assert new.completeness is old.completeness
    assert new.incomplete_reasons == old.incomplete_reasons
    old_parameters = dict(old.transformation_chain[-1].parameters)
    new_parameters = dict(new.transformation_chain[-1].parameters)
    assert old_parameters.pop("birth_survival_partition") == "per_pdf_histogram_overlap"
    assert new_parameters.pop("birth_survival_partition") == "per_mapping_exact_reuse"
    assert new_parameters == old_parameters
    assert new.transformation_chain[-1].version == "9_diagnostic_birth_survival_partition_reuse"


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("peak", [False, True])
def test_public_reuse_preserves_analytic_narrow_counts_and_temporal_contract(
    strategy: str, peak: bool,
) -> None:
    source = replace(
        _source(peak=peak),
        temporal_basis=TemporalBasis.SEP_PEAK if peak else TemporalBasis.MISSION_TOTAL,
        mission_duration_s=None if peak else 456_789.0,
    )
    old = _calculate(source, reuse=False, strategy=strategy)
    new = _calculate(source, reuse=True, strategy=strategy)
    _assert_unchanged(old, new)
    assert _counts(new) == pytest.approx(_analytic_narrow_count(), rel=1e-7)
    assert new.completeness is SpectrumCompleteness.INCOMPLETE


def _jump_kernel() -> secondary.SecondaryProtonKernel:
    kernel = _kernel(birth_edges=(2.0, 7.0))
    product = replace(
        kernel.products[0],
        incident_energy_mev=(0.1, 6.0, 11.0, 20.0),
        yield_value=(0.2, 1.1, 0.4, 1.2),
        pdf_incident_energy_mev=(0.1, 5.5, 6.5, 20.0),
        emitted_energy_mev_by_incident=((2.0, 7.0), (1.1, 2.5, 6.0), (3.0, 5.0), (2.0, 7.0)),
        pdf_per_mev_by_incident=((0.2, 0.0), (0.3, 0.1, 0.0), (0.5, 0.0), (0.2, 0.0)),
    )
    return replace(
        kernel,
        products=(product,),
        nonelastic_cross_section=Al27NonelasticCrossSectionTable(
            mt=3,
            energy_mev=(0.1, 2.0, 2.5, 4.0, 20.0),
            sigma_barn=(3.0, 10.0, 50.0, 20.0, 20.0),
            discontinuities=(Al27CrossSectionDiscontinuity(
                energy_mev=2.5, left_sigma_barn=1.0, right_sigma_barn=50.0,
            ),),
        ),
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("peak", [False, True])
def test_reuse_keeps_nonlinear_survival_jump_and_independent_pdf_grids_identical(
    strategy: str, peak: bool,
) -> None:
    source = replace(
        _source(peak=peak),
        x=(1.0, 2.0, 4.0, 8.0, 10.0, 12.0),
        y=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    kernel = _jump_kernel()
    old = _calculate(source, reuse=False, strategy=strategy, kernel=kernel)
    new = _calculate(source, reuse=True, strategy=strategy, kernel=kernel)
    _assert_unchanged(old, new)
    assert _counts(new) > 0.0


def _mapping(
    *, lower: float, upper: float, residual: float, supported: bool = True,
) -> secondary._SecondaryExitMapping:
    return secondary._SecondaryExitMapping(
        output_index=0,
        output_width_mev=upper - lower,
        output_lower_mev=lower - residual,
        output_upper_mev=upper - residual,
        birth_lower_mev=lower,
        birth_upper_mev=upper,
        residual_thickness_g_cm2=residual,
        attenuation_supported=supported,
        attenuation_energies_mev=(lower - residual, upper - residual, lower, upper),
    )


@pytest.mark.parametrize("edge", [nextafter(2.5, -inf), 2.5, nextafter(2.5, inf)])
def test_strict_reused_slices_preserve_every_original_gl4_segment_at_float_neighbors(
    monkeypatch: pytest.MonkeyPatch, edge: float,
) -> None:
    kernel = _jump_kernel()
    assert kernel.nonelastic_cross_section is not None
    table = _stopping()
    optical = secondary._build_nonelastic_optical_depth_table(
        stopping_table=table, cross_section_table=kernel.nonelastic_cross_section,
    )
    mapping = _mapping(lower=1.0, upper=5.0, residual=0.5)
    product = replace(
        _kernel().products[0],
        emitted_energy_mev_by_incident=((1.3, edge, 3.0, 3.2, 4.8),) * 2,
        pdf_per_mev_by_incident=((0.2, 0.4, 1e-20, 0.3, 999.0),) * 2,
    )
    original = secondary._integrate_exit_survival_over_birth_segment
    calls = []

    def recorded_segment(**kwargs: Any) -> float:
        calls.append((kwargs["birth_lower_mev"], kwargs["birth_upper_mev"]))
        return original(**kwargs)

    monkeypatch.setattr(secondary, "_integrate_exit_survival_over_birth_segment", recorded_segment)
    kwargs = dict(
        pdf_index=0, mapping=mapping, stopping_table=table,
        cross_section_table=kernel.nonelastic_cross_section, optical_depth_table=optical,
    )
    old = secondary._integrate_tabulated_pdf_with_exit_survival(
        product, survival_integral_cache={}, **kwargs,
    )
    old_segments = tuple(calls)
    calls.clear()
    all_knots = secondary._birth_survival_segment_boundaries(
        lower_mev=mapping.birth_lower_mev, upper_mev=mapping.birth_upper_mev,
        mapping=mapping, stopping_table=table, optical_depth_table=optical,
    )
    new = secondary._integrate_tabulated_pdf_with_exit_survival(
        product, survival_integral_cache={}, birth_survival_knots=all_knots, **kwargs,
    )
    assert new == old
    assert tuple(calls) == old_segments
    assert all(lo < hi for lo, hi in calls)
    assert (3.0, 3.2) in calls  # Tiny positive density is not rounded to zero.
    assert all(hi <= 4.8 for _lo, hi in calls)  # Last ordinate is not an extra bin.


def test_mapping_local_reuse_does_not_leak_between_depths_or_mt3_datasets() -> None:
    kernel = _kernel(birth_edges=(2.0, 3.0))
    active = secondary._active_secondary_products_at_incident_energy(
        kernel=kernel, incident_energy_mev=10.0,
    )
    table = _stopping()
    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    for sigma, residual in ((3.0, 0.2), (3.0, 0.8), (30.0, 0.2), (0.0, 0.8), (3.0, 0.2)):
        mt3 = Al27NonelasticCrossSectionTable(
            mt=3, energy_mev=(0.1, 20.0), sigma_barn=(sigma, sigma),
        )
        optical = secondary._build_nonelastic_optical_depth_table(
            stopping_table=table, cross_section_table=mt3,
        )
        kwargs = dict(
            active_products=active, nonelastic_cross_section=mt3, optical_depth_table=optical,
            mapping=_mapping(lower=2.0, upper=3.0, residual=residual), stopping_table=table,
        )
        old = secondary._integrated_secondary_source_to_exit_per_g_cm2(**kwargs)
        new = secondary._integrated_secondary_source_to_exit_per_g_cm2(
            reuse_birth_survival_partition=True, **kwargs,
        )
        expected = number_per_gram * 2.0 * BARN_CM2 * exp(
            -number_per_gram * sigma * BARN_CM2 * residual,
        )
        assert new == old
        assert new == pytest.approx(expected, rel=1e-13)


def test_reuse_retains_tiny_positive_histogram_tail_without_cdf_cancellation() -> None:
    kernel = _kernel()
    product = replace(
        kernel.products[0],
        emitted_energy_mev_by_incident=((1.0, 2.0, 3.0),) * 2,
        pdf_per_mev_by_incident=((1.0, 1e-20, 0.0),) * 2,
    )
    kernel = replace(kernel, products=(product,), nonelastic_cross_section=None)
    active = secondary._active_secondary_products_at_incident_energy(
        kernel=kernel, incident_energy_mev=10.0,
    )
    kwargs = dict(
        active_products=active, nonelastic_cross_section=None, optical_depth_table=None,
        mapping=_mapping(lower=2.25, upper=2.75, residual=0.5, supported=False),
        stopping_table=_stopping(),
    )
    old = secondary._integrated_secondary_source_to_exit_per_g_cm2(**kwargs)
    new = secondary._integrated_secondary_source_to_exit_per_g_cm2(
        reuse_birth_survival_partition=True, **kwargs,
    )
    expected = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL * 2.0 * BARN_CM2 * 5e-21
    assert new == old
    assert new > 0.0
    assert new == pytest.approx(expected, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("kind", ["missing_mt3", "partial_mt3", "partial_reaction", "missing_pdf"])
def test_reuse_preserves_nonzero_supported_counts_and_all_missing_data_statuses(kind: str) -> None:
    kernel = _kernel(birth_edges=(4.0, 5.0))
    if kind == "missing_mt3":
        kernel = replace(kernel, nonelastic_cross_section=None)
    elif kind == "partial_mt3":
        assert kernel.nonelastic_cross_section is not None
        kernel = replace(kernel, nonelastic_cross_section=replace(
            kernel.nonelastic_cross_section,
            incident_energy_support=TendlIncidentEnergySupport(2.5, 20.0, "partial MT3"),
        ))
    elif kind == "partial_reaction":
        kernel = replace(kernel, incident_energy_support=TendlIncidentEnergySupport(
            0.1, 7.5, "partial production",
        ))
    else:
        kernel = replace(kernel, products=(replace(
            kernel.products[0], pdf_incident_energy_mev=(6.0, 20.0),
        ),))
    source = replace(
        _source(), completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=("retain this source warning",),
    )
    old = _calculate(source, reuse=False, screening=True, kernel=kernel)
    new = _calculate(source, reuse=True, screening=True, kernel=kernel)
    _assert_unchanged(old, new)
    assert _counts(new) > 0.0
    assert "retain this source warning" in new.incomplete_reasons
    assert new.completeness is SpectrumCompleteness.INCOMPLETE


def test_response_budget_and_unresolved_error_are_not_changed_by_reuse() -> None:
    old = _calculate(_source(), reuse=False, budget=4)
    new = _calculate(_source(), reuse=True, budget=4)
    _assert_unchanged(old, new)
    parameters = dict(new.transformation_chain[-1].parameters)
    assert int(parameters["incident_quadrature_response_evaluations"]) <= 4
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"


def test_reuse_reduces_actual_partition_builds_without_changing_survival_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _jump_kernel()
    active = secondary._active_secondary_products_at_incident_energy(
        kernel=kernel, incident_energy_mev=6.0,
    )
    assert kernel.nonelastic_cross_section is not None
    table = _stopping()
    optical = secondary._build_nonelastic_optical_depth_table(
        stopping_table=table, cross_section_table=kernel.nonelastic_cross_section,
    )
    original = secondary._birth_survival_segment_boundaries
    builds = []

    def counted_build(**kwargs: Any) -> tuple[float, ...]:
        builds.append((kwargs["lower_mev"], kwargs["upper_mev"]))
        return original(**kwargs)

    monkeypatch.setattr(secondary, "_birth_survival_segment_boundaries", counted_build)
    kwargs = dict(
        active_products=active, nonelastic_cross_section=kernel.nonelastic_cross_section,
        optical_depth_table=optical, mapping=_mapping(lower=1.0, upper=6.0, residual=0.5),
        stopping_table=table,
    )
    old = secondary._integrated_secondary_source_to_exit_per_g_cm2(**kwargs)
    old_builds = len(builds)
    builds.clear()
    new = secondary._integrated_secondary_source_to_exit_per_g_cm2(
        reuse_birth_survival_partition=True, **kwargs,
    )
    assert new == old
    assert old_builds > len(builds) == 1


def test_reuse_is_disabled_by_default() -> None:
    assert IncidentEnergyQuadrature().reuse_birth_survival_partition is False


@pytest.mark.parametrize("value", [None, 0, 1, "true", (), float("nan")])
def test_reuse_option_requires_a_strict_boolean(value: object) -> None:
    with pytest.raises(ValueError):
        IncidentEnergyQuadrature(reuse_birth_survival_partition=value)  # type: ignore[arg-type]
