"""Independent numerical checks of complete-primary-cell crossing pruning."""

from dataclasses import replace
from math import exp
from typing import Any

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import SpectrumCompleteness
from radar.shielding import proton_al_secondary as secondary_module
from radar.shielding.proton_al_secondary import (
    PROTON_AL_SECONDARY_ACTIVE_PDF_SUPPORT_REASON,
    PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
    PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON,
    PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
    PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON,
    PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON,
    PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
    DepthCrossingQuadrature,
)
from radar.shielding.proton_al_survival import (
    AL27_MOLAR_MASS_G_MOL,
    AVOGADRO_CONSTANT_MOL_INV,
    BARN_CM2,
)
from test_proton_secondary_bin_quadrature import _assert_same_bins, _calculate
from test_proton_secondary_depth_crossings import (
    _analytic_narrow_count,
    _counts,
    _kernel,
    _source,
)


ACTIVE_STRATEGY = "per_output_bin_active_cells"


@pytest.mark.parametrize("peak", [False, True])
@pytest.mark.parametrize("refined", [False, True])
def test_complete_active_cells_keep_narrow_analytic_response_and_reference_bins(
    peak: bool, refined: bool
) -> None:
    source = _source(peak=peak, refined=refined)
    reference = _calculate(source, strategy="global_union")
    result = _calculate(source, strategy=ACTIVE_STRATEGY)
    _assert_same_bins(result, reference)
    assert _counts(result) == pytest.approx(_analytic_narrow_count(), rel=1e-7)
    assert result.y_unit is source.y_unit
    assert result.quantity is source.quantity
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    assert {
        PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
        PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
        PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
        PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON,
    }.issubset(result.incomplete_reasons)


@pytest.mark.parametrize("peak", [False, True])
def test_independent_pdf_grid_keeps_upper_interpolation_slice_outside_reachable_domain(
    peak: bool,
) -> None:
    source = _source(peak=peak)
    kernel = _kernel()
    product = replace(
        kernel.products[0],
        incident_energy_mev=(0.1, 20.0),
        yield_value=(0.5, 1.5),
        pdf_incident_energy_mev=(0.1, 5.5, 6.5, 20.0),
        emitted_energy_mev_by_incident=(
            (0.39, 0.3901), (2.0, 3.0), (0.7, 1.7), (5.0, 6.0)
        ),
        pdf_per_mev_by_incident=((10_000.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)),
    )
    kernel = replace(kernel, products=(product,))
    # Reachable incident energies are 4..8. The PDF at 20 is still the right
    # interpolation bracket on 6.5..8 and must not be mistaken for irrelevant.
    reference = _calculate(source, strategy="global_union", kernel=kernel, depth_steps=2)
    result = _calculate(source, strategy=ACTIVE_STRATEGY, kernel=kernel, depth_steps=2)
    _assert_same_bins(result, reference)
    assert _counts(result) > 0.0


@pytest.mark.parametrize("peak", [False, True])
def test_active_cells_with_nonconstant_mf3_yield_and_survival_agree_after_depth_refinement(
    peak: bool,
) -> None:
    source = replace(
        _source(peak=peak),
        x=(1.0, 2.0, 4.0, 8.0, 10.0, 12.0),
        y=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    kernel = _kernel(birth_edges=(2.0, 7.0))
    assert kernel.nonelastic_cross_section is not None
    kernel = replace(
        kernel,
        products=(replace(
            kernel.products[0],
            incident_energy_mev=(0.1, 6.0, 11.0, 20.0),
            yield_value=(0.2, 1.1, 0.4, 1.2),
        ),),
        cross_sections_by_mt={5: replace(
            kernel.cross_sections_by_mt[5],
            energy_mev=(0.1, 5.0, 9.0, 20.0),
            sigma_barn=(0.0, 2.0, 3.0, 1.0),
        )},
        nonelastic_cross_section=replace(
            kernel.nonelastic_cross_section,
            energy_mev=(0.1, 5.0, 20.0),
            sigma_barn=(100.0, 3.0, 100.0),
        ),
    )
    results = {
        (depth, strategy): _calculate(source, strategy=strategy, kernel=kernel, depth_steps=depth)
        for depth in (4, 8)
        for strategy in ("global_union", ACTIVE_STRATEGY)
    }
    for strategy in ("global_union", ACTIVE_STRATEGY):
        assert results[4, strategy].y == pytest.approx(
            results[8, strategy].y, rel=1e-7, abs=1e-13
        )
    _assert_same_bins(results[8, ACTIVE_STRATEGY], results[8, "global_union"])
    assert _counts(results[8, ACTIVE_STRATEGY]) > 0.0


def test_distant_pdf_cells_reduce_actual_depth_work_without_changing_supported_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    kernel = _kernel()
    far_edges = tuple(0.4 + 0.07 * index for index in range(81))
    far_density = (1.0 / (far_edges[-1] - far_edges[0]),) * 80 + (0.0,)
    product = replace(
        kernel.products[0],
        incident_energy_mev=(0.1, 300.0),
        pdf_incident_energy_mev=(0.1, 8.0, 200.0, 300.0),
        emitted_energy_mev_by_incident=(
            (0.39, 0.3901), (0.39, 0.3901), far_edges, far_edges
        ),
        pdf_per_mev_by_incident=(
            (10_000.0, 0.0), (10_000.0, 0.0), far_density, far_density
        ),
        incident_energy_support=None,
    )
    kernel = replace(kernel, products=(product,))
    original = secondary_module._single_incident_secondary_response
    observed_depth_samples = []

    def counted_response(**kwargs: Any) -> Any:
        response = original(**kwargs)
        observed_depth_samples.append(response.production_samples)
        return response

    monkeypatch.setattr(secondary_module, "_single_incident_secondary_response", counted_response)
    reference = _calculate(source, strategy="per_output_bin", kernel=kernel, budget=65_536)
    reference_work = sum(observed_depth_samples)
    observed_depth_samples.clear()
    result = _calculate(source, strategy=ACTIVE_STRATEGY, kernel=kernel, budget=65_536)
    active_work = sum(observed_depth_samples)
    parameters = dict(result.transformation_chain[-1].parameters)
    assert int(parameters["incident_quadrature_supported_depth_samples"]) == active_work
    _assert_same_bins(result, reference)
    assert _counts(result) == pytest.approx(_analytic_narrow_count(), rel=1e-7)
    assert 0 < active_work < reference_work


@pytest.mark.parametrize("limit_kind", ["depth", "coupled", "active_cells"])
def test_active_partition_budget_exhaustion_never_claims_complete_crossings(limit_kind: str) -> None:
    options = (
        DepthCrossingQuadrature(max_depth_knots=0, max_coupled_incident_knots=16_384)
        if limit_kind == "depth"
        else DepthCrossingQuadrature(max_depth_knots=4096, max_coupled_incident_knots=0)
        if limit_kind == "coupled"
        else DepthCrossingQuadrature(
            max_depth_knots=4096, max_coupled_incident_knots=16_384, max_active_cell_records=0
        )
    )
    result = _calculate(_source(), strategy=ACTIVE_STRATEGY, depth_options=options)
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON in result.incomplete_reasons
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON in result.incomplete_reasons
    assert result.completeness is SpectrumCompleteness.INCOMPLETE


def test_active_strategy_does_not_reset_response_budget_between_cells_or_output_bins() -> None:
    result = _calculate(_source(), strategy=ACTIVE_STRATEGY, budget=4)
    parameters = dict(result.transformation_chain[-1].parameters)
    assert int(parameters["incident_quadrature_response_evaluations"]) <= 4
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON in result.incomplete_reasons


def test_zero_shield_with_active_cells_has_no_partition_or_physics_blockers() -> None:
    source = _source()
    result = _calculate(source, strategy=ACTIVE_STRATEGY, thickness=0.0)
    assert result.y == (0.0,) * len(source.x)
    assert result.incomplete_reasons == ()
    assert result.ordinate_basis is source.ordinate_basis


def test_active_pruning_retains_incoming_source_incomplete_reason() -> None:
    source: Spectrum1D = replace(
        _source(),
        completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=("synthetic unverified incident-source input",),
    )
    result = _calculate(source, strategy=ACTIVE_STRATEGY)
    assert "synthetic unverified incident-source input" in result.incomplete_reasons
    assert result.completeness is SpectrumCompleteness.INCOMPLETE


def test_active_cells_keep_thin_yield_peak_between_other_transport_knots() -> None:
    source = _source()
    kernel = _kernel(birth_edges=(3.0, 4.0))
    kernel = replace(
        kernel,
        products=(replace(
            kernel.products[0],
            incident_energy_mev=(0.1, 5.5, 5.5001, 5.5002, 20.0),
            yield_value=(0.0, 0.0, 1.0, 0.0, 0.0),
        ),),
    )
    reference = _calculate(source, strategy="global_union", kernel=kernel)
    result = _calculate(source, strategy=ACTIVE_STRATEGY, kernel=kernel)
    _assert_same_bins(result, reference)
    # Every entrance E0 in 7..8 traverses the complete narrow yield triangle.
    # R=E gives depth integral equal to triangle area 0.0001; every emitted
    # proton lies within the output support after the remaining shielding.
    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    expected = (
        number_per_gram * 2.0 * BARN_CM2
        * exp(-number_per_gram * 3.0 * BARN_CM2 * 3.0)
        * 0.5 * 0.5 * (5.5002 - 5.5)
    )
    assert _counts(result) == pytest.approx(expected, rel=1e-7)


def test_missing_pdf_cells_remain_incomplete_without_losing_supported_production() -> None:
    source = _source()
    kernel = _kernel(birth_edges=(4.0, 5.0))
    kernel = replace(
        kernel,
        products=(replace(
            kernel.products[0],
            pdf_incident_energy_mev=(6.0, 20.0),
        ),),
    )
    # Positive production exists on primary energies 5..6, but its PDF is
    # unavailable. The known part on 6..E0 is independently calculable.
    reference = _calculate(source, strategy="global_union", kernel=kernel)
    result = _calculate(source, strategy=ACTIVE_STRATEGY, kernel=kernel)
    parameters = dict(result.transformation_chain[-1].parameters)
    _assert_same_bins(result, reference)
    assert parameters["depth_active_pdf_support_missing"] == "true"
    assert PROTON_AL_SECONDARY_ACTIVE_PDF_SUPPORT_REASON in result.incomplete_reasons
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    # All emitted protons in 4..5 exit inside the output bins. Integrating the
    # known primary depth E0-6 against source E0-7 on E0=7..8 gives 5/6.
    expected = (
        number_per_gram * 2.0 * BARN_CM2
        * exp(-number_per_gram * 3.0 * BARN_CM2 * 3.0)
        * (5.0 / 6.0)
    )
    assert _counts(result) == pytest.approx(expected, rel=1e-7)
