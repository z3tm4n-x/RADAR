"""Independent public checks of proof-based pre-quadrature screening."""

from dataclasses import replace
import json
from math import exp
from typing import Any

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import SpectrumCompleteness, TemporalBasis
from radar.nuclear.tendl import TendlIncidentEnergySupport
from radar.shielding import proton_al_secondary as secondary_module
from radar.shielding.proton_al_secondary import (
    PROTON_AL_SECONDARY_ACTIVE_PDF_SUPPORT_REASON,
    PROTON_AL_SECONDARY_ATTENUATION_OPERATOR,
    PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON,
    DepthCrossingQuadrature,
    IncidentEnergyQuadrature,
    SecondaryProtonKernel,
    calculate_secondary_proton_spectrum_through_al,
)
from radar.shielding.proton_al_survival import (
    AL27_MOLAR_MASS_G_MOL,
    AVOGADRO_CONSTANT_MOL_INV,
    BARN_CM2,
)
from test_proton_secondary_bin_quadrature import _assert_same_bins
from test_proton_secondary_depth_crossings import (
    _analytic_narrow_count,
    _counts,
    _kernel,
    _source,
    _stopping,
)


def _calculate(
    source: Spectrum1D,
    *,
    screening: bool = True,
    kernel: SecondaryProtonKernel | None = None,
    thickness: float = 3.0,
    budget: int = 16_384,
) -> Spectrum1D:
    return calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=source,
        stopping_table=_stopping(),
        kernel=kernel or _kernel(),
        thickness_g_cm2=thickness,
        depth_steps=1,
        depth_quadrature=DepthCrossingQuadrature(
            max_depth_knots=4096,
            max_coupled_incident_knots=16_384,
        ),
        incident_quadrature=IncidentEnergyQuadrature(
            partition_strategy="per_output_bin_active_cells",
            prequadrature_screening=screening,
            relative_tolerance=1e-7,
            max_response_evaluations=budget,
            max_refinement_depth=4,
        ),
    )


def _count_response_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, tuple[float, ...]]]:
    original = secondary_module._single_incident_secondary_response
    calls: list[tuple[float, tuple[float, ...]]] = []

    def counted_response(**kwargs: Any) -> Any:
        calls.append((kwargs["entrance_energy_mev"], kwargs["output_centers_mev"]))
        return original(**kwargs)

    monkeypatch.setattr(secondary_module, "_single_incident_secondary_response", counted_response)
    return calls


def _assert_per_bin_source_weights_conserved(result: Spectrum1D, source: Spectrum1D) -> None:
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["prequadrature_screening"] == "enabled"
    weights = json.loads(parameters["prequadrature_source_integrals_by_output_bin"])
    categories = {"evaluate", "proven_zero", "omitted_unsupported", "source_zero"}
    assert set(weights) == categories
    source_integral = sum(
        0.5 * (left_y + right_y) * (right_x - left_x)
        for left_x, right_x, left_y, right_y in zip(source.x, source.x[1:], source.y, source.y[1:])
    )
    for category in categories:
        assert len(weights[category]) == len(result.x)
        assert all(value >= 0.0 for value in weights[category])
    # Each output partition classifies the SAME source independently. Never
    # add these weights across output bins as if they were a physical fluence.
    for index in range(len(result.x)):
        assert sum(weights[category][index] for category in categories) == pytest.approx(
            source_integral, rel=1e-12, abs=1e-15,
        )
    assert weights["source_zero"] == [0.0] * len(result.x)
    assert parameters["prequadrature_integral_scope"] == (
        "per_output_bin_source_weights_not_summed_across_bins"
    )


@pytest.mark.parametrize("peak", [False, True])
@pytest.mark.parametrize("refined", [False, True])
def test_screening_preserves_narrow_analytic_response_units_and_source_warning(
    peak: bool, refined: bool,
) -> None:
    source = replace(
        _source(peak=peak, refined=refined),
        temporal_basis=TemporalBasis.SEP_PEAK if peak else TemporalBasis.MISSION_TOTAL,
        mission_duration_s=None if peak else 123_456.0,
        completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=("synthetic source warning must survive screening",),
    )
    reference = _calculate(source, screening=False)
    result = _calculate(source)
    _assert_same_bins(result, reference)
    assert _counts(result) == pytest.approx(_analytic_narrow_count(), rel=1e-7)
    assert result.quantity is source.quantity
    assert result.y_unit is source.y_unit
    assert result.temporal_basis is source.temporal_basis
    assert result.mission_duration_s == source.mission_duration_s
    assert source.incomplete_reasons[0] in result.incomplete_reasons
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    _assert_per_bin_source_weights_conserved(result, source)


@pytest.mark.parametrize("zero_kind", ["geometry", "reaction"])
def test_supported_zero_proofs_bypass_every_expensive_response_call(
    monkeypatch: pytest.MonkeyPatch, zero_kind: str,
) -> None:
    kernel = _kernel(birth_edges=(9.0, 10.0))
    if zero_kind == "reaction":
        kernel = replace(
            _kernel(birth_edges=(3.0, 4.0)),
            cross_sections_by_mt={5: replace(
                kernel.cross_sections_by_mt[5],
                sigma_barn=(0.0, 0.0, 0.0),
                discontinuities=(),
            )},
        )
    calls = _count_response_calls(monkeypatch)
    reference = _calculate(_source(), screening=False, kernel=kernel, thickness=0.2)
    assert calls
    calls.clear()
    result = _calculate(_source(), kernel=kernel, thickness=0.2)
    assert calls == []
    _assert_same_bins(result, reference)
    assert result.y == (0.0,) * len(result.x)
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_response_evaluations"] == "0"
    assert parameters["incident_quadrature_estimated_absolute_error"] == "0"
    assert int(parameters["prequadrature_proven_zero_base_intervals"]) > 0
    assert parameters["prequadrature_omitted_unsupported_base_intervals"] == "0"
    assert parameters["prequadrature_evaluate_base_intervals"] == "0"
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON not in result.incomplete_reasons
    _assert_per_bin_source_weights_conserved(result, _source())


@pytest.mark.parametrize("domain", [(0.1, 6.0), (9.0, 20.0)])
def test_wholly_missing_tendl_domain_is_omission_not_a_converged_zero(
    monkeypatch: pytest.MonkeyPatch, domain: tuple[float, float],
) -> None:
    kernel = replace(
        _kernel(birth_edges=(3.0, 4.0)),
        incident_energy_support=TendlIncidentEnergySupport(
            minimum_mev=domain[0], maximum_mev=domain[1], dataset_ref="synthetic restricted TENDL",
        ),
    )
    calls = _count_response_calls(monkeypatch)
    result = _calculate(_source(), kernel=kernel, thickness=0.2)
    parameters = dict(result.transformation_chain[-1].parameters)
    assert calls == []
    assert result.y == (0.0,) * len(result.x)
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"
    assert parameters["incident_quadrature_estimated_relative_error"] == "unavailable"
    assert int(parameters["prequadrature_omitted_unsupported_base_intervals"]) > 0
    assert parameters["prequadrature_proven_zero_base_intervals"] == "0"
    assert parameters["prequadrature_evaluate_base_intervals"] == "0"
    assert parameters["incident_quadrature_supported_status"] == "empty_supported_domain"
    assert parameters["incident_quadrature_supported_estimated_absolute_error"] == "0"
    assert any("support" in reason.lower() for reason in result.incomplete_reasons)
    _assert_per_bin_source_weights_conserved(result, _source())


def test_missing_output_bins_are_not_evaluated_and_supported_counts_are_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = replace(
        _source(), x=(0.02, 0.04, 1.0, 7.0, 8.0), y=(0.0, 0.0, 0.0, 0.0, 1.0),
    )
    kernel = _kernel(birth_edges=(3.0, 4.0))
    reference = _calculate(source, screening=False, kernel=kernel, thickness=0.2)
    calls = _count_response_calls(monkeypatch)
    result = _calculate(source, kernel=kernel, thickness=0.2)
    assert calls
    assert all(centers != (0.02,) for _energy, centers in calls)
    assert result.y[0] == 0.0
    assert _counts(result) > 0.0
    _assert_same_bins(result, reference)
    parameters = dict(result.transformation_chain[-1].parameters)
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"
    _assert_per_bin_source_weights_conserved(result, source)


def test_entry_above_tendl_support_can_still_reach_supported_reaction_depths() -> None:
    kernel = replace(
        _kernel(birth_edges=(3.0, 4.0)),
        incident_energy_support=TendlIncidentEnergySupport(
            minimum_mev=0.1, maximum_mev=7.5, dataset_ref="synthetic upper-bound crossing",
        ),
    )
    reference = _calculate(_source(), screening=False, kernel=kernel, thickness=0.2)
    result = _calculate(_source(), kernel=kernel, thickness=0.2)
    _assert_same_bins(result, reference)
    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    # Known depth is 0.2 on E0=7..7.5, then 7.7-E0 on E0=7.5..7.7.
    # Its source-weighted integral is 109/3000, including E0 > 7.5.
    expected = (
        number_per_gram * 2.0 * BARN_CM2
        * exp(-number_per_gram * 3.0 * BARN_CM2 * 0.2)
        * (109.0 / 3000.0)
    )
    assert _counts(result) == pytest.approx(expected, rel=1e-7)
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"
    _assert_per_bin_source_weights_conserved(result, _source())


def test_partial_tendl_paths_keep_unknown_full_error_even_without_wholly_omitted_cells() -> None:
    source = replace(_source(), x=(1.0, 7.6, 7.65), y=(0.0, 0.0, 1.0))
    kernel = replace(
        _kernel(birth_edges=(3.0, 4.0)),
        incident_energy_support=TendlIncidentEnergySupport(
            minimum_mev=0.1, maximum_mev=7.5, dataset_ref="synthetic partial reaction paths",
        ),
    )
    result = _calculate(source, kernel=kernel, thickness=0.2)
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["prequadrature_omitted_unsupported_base_intervals"] == "0"
    assert int(parameters["prequadrature_evaluate_base_intervals"]) > 0
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"
    assert parameters["incident_quadrature_estimated_relative_error"] == "unavailable"
    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    # Integral on 7.6..7.65: source (E0-7.6)/0.05 times known depth 7.7-E0.
    expected = (
        number_per_gram * 2.0 * BARN_CM2
        * exp(-number_per_gram * 3.0 * BARN_CM2 * 0.2) / 600.0
    )
    assert _counts(result) == pytest.approx(expected, rel=1e-7)
    _assert_per_bin_source_weights_conserved(result, source)


def test_skipping_zero_reactions_retains_mt3_depth_coverage_warning_without_response_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _kernel(birth_edges=(3.0, 4.0))
    assert kernel.nonelastic_cross_section is not None
    kernel = replace(
        kernel,
        cross_sections_by_mt={5: replace(
            kernel.cross_sections_by_mt[5], sigma_barn=(0.0, 0.0, 0.0), discontinuities=(),
        )},
        nonelastic_cross_section=replace(
            kernel.nonelastic_cross_section,
            incident_energy_support=TendlIncidentEnergySupport(
                minimum_mev=6.5, maximum_mev=20.0, dataset_ref="synthetic missing MT3 depth",
            ),
        ),
    )
    calls = _count_response_calls(monkeypatch)
    result = _calculate(_source(), kernel=kernel)
    assert calls == []
    assert result.y == (0.0,) * len(result.x)
    assert any(
        reason.startswith(PROTON_AL_SECONDARY_ATTENUATION_OPERATOR)
        for reason in result.incomplete_reasons
    )
    # Entrance source endpoints 7..8 are covered; only full depth-path evidence
    # exposes missing MT3 below 6.5. No numerical response cache exists here.
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    _assert_per_bin_source_weights_conserved(result, _source())


def test_unavailable_pdf_is_not_zero_and_retains_the_analytically_known_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _kernel(birth_edges=(4.0, 5.0))
    kernel = replace(
        kernel,
        products=(replace(kernel.products[0], pdf_incident_energy_mev=(6.0, 20.0)),),
    )
    reference = _calculate(_source(), screening=False, kernel=kernel)
    calls = _count_response_calls(monkeypatch)
    result = _calculate(_source(), kernel=kernel)
    assert calls
    _assert_same_bins(result, reference)
    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    # The known reaction depth is E0-6; integral_7^8 (E0-7)*(E0-6) dE0 = 5/6.
    expected = (
        number_per_gram * 2.0 * BARN_CM2
        * exp(-number_per_gram * 3.0 * BARN_CM2 * 3.0)
        * (5.0 / 6.0)
    )
    assert _counts(result) == pytest.approx(expected, rel=1e-7)
    assert PROTON_AL_SECONDARY_ACTIVE_PDF_SUPPORT_REASON in result.incomplete_reasons
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["depth_active_pdf_support_missing"] == "true"
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"


def test_screened_response_budget_is_shared_across_all_nonzero_partitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = replace(
        _source(),
        x=(0.5, 1.0, 2.0, 4.0, 7.0, 8.0),
        y=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    calls = _count_response_calls(monkeypatch)
    result = _calculate(source, kernel=_kernel(birth_edges=(1.1, 4.5)), thickness=0.2, budget=4)
    parameters = dict(result.transformation_chain[-1].parameters)
    assert 0 < len(calls) <= 4
    assert int(parameters["incident_quadrature_response_evaluations"]) == len(calls)
    assert all(len(centers) == 1 for _energy, centers in calls)
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON in result.incomplete_reasons


@pytest.mark.parametrize("kind", ["zero_source", "zero_shield"])
def test_screening_does_not_spend_response_budget_on_zero_input_or_zero_shield(
    monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    source = _source()
    if kind == "zero_source":
        source = replace(source, y=(0.0,) * len(source.x))
    calls = _count_response_calls(monkeypatch)
    result = _calculate(source, thickness=0.0 if kind == "zero_shield" else 3.0)
    assert calls == []
    assert result.y == (0.0,) * len(source.x)
    assert result.y_unit is source.y_unit
    if kind == "zero_shield":
        assert result.incomplete_reasons == ()


def test_screening_remains_disabled_by_default() -> None:
    assert IncidentEnergyQuadrature().prequadrature_screening is False


@pytest.mark.parametrize("strategy", ["global_union", "per_output_bin"])
def test_screening_rejects_unsupported_partition_strategy(strategy: str) -> None:
    with pytest.raises(ValueError):
        IncidentEnergyQuadrature(partition_strategy=strategy, prequadrature_screening=True)


@pytest.mark.parametrize("value", [None, 0, 1, "true", (), float("nan")])
def test_screening_requires_a_real_boolean(value: object) -> None:
    with pytest.raises(ValueError):
        IncidentEnergyQuadrature(
            partition_strategy="per_output_bin_active_cells",
            prequadrature_screening=value,  # type: ignore[arg-type]
        )
