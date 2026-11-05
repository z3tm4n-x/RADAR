"""Independent checks of per-output-bin secondary incident integration."""

from dataclasses import replace
from math import exp
from typing import Any

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import SpectrumCompleteness, TemporalBasis
from radar.shielding import proton_al_secondary as secondary_module
from radar.shielding.proton_al import log_bin_edges_from_centers
from radar.shielding.proton_al_secondary import (
    PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
    PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON,
    PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
    PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON,
    PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON,
    PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
    PROTON_AL_SECONDARY_TRANSPORT_DOMAIN_REASON_PREFIX,
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
    strategy: str,
    kernel: SecondaryProtonKernel | None = None,
    thickness: float = 3.0,
    budget: int = 16_384,
    depth_steps: int = 1,
    depth_options: DepthCrossingQuadrature | None = None,
) -> Spectrum1D:
    return calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=source,
        stopping_table=_stopping(),
        kernel=kernel or _kernel(),
        thickness_g_cm2=thickness,
        depth_steps=depth_steps,
        depth_quadrature=depth_options or DepthCrossingQuadrature(
            max_depth_knots=4096, max_coupled_incident_knots=16_384
        ),
        incident_quadrature=IncidentEnergyQuadrature(
            partition_strategy=strategy,
            relative_tolerance=1e-7,
            max_response_evaluations=budget,
            max_refinement_depth=4,
        ),
    )


def _assert_same_bins(left: Spectrum1D, right: Spectrum1D) -> None:
    assert left.x == right.x
    assert left.bin_edges == right.bin_edges
    assert left.ordinate_basis is right.ordinate_basis
    assert left.y == pytest.approx(right.y, rel=1e-7, abs=1e-13)


@pytest.mark.parametrize("peak", [False, True])
@pytest.mark.parametrize("refined", [False, True])
def test_scalar_partitions_match_complete_vector_reference_and_narrow_analytic_result(
    peak: bool, refined: bool
) -> None:
    source = replace(
        _source(peak=peak, refined=refined),
        temporal_basis=TemporalBasis.SEP_PEAK if peak else TemporalBasis.MISSION_TOTAL,
        mission_duration_s=None if peak else 123_456.0,
    )
    vector = _calculate(source, strategy="global_union")
    scalar = _calculate(source, strategy="per_output_bin")
    _assert_same_bins(scalar, vector)
    assert _counts(scalar) == pytest.approx(_analytic_narrow_count(), rel=1e-7)
    assert scalar.y_unit is source.y_unit
    assert scalar.quantity is source.quantity
    assert scalar.temporal_basis is source.temporal_basis
    assert scalar.mission_duration_s == source.mission_duration_s
    assert scalar.completeness is SpectrumCompleteness.INCOMPLETE
    assert scalar.bin_edges == log_bin_edges_from_centers(source.x)
    scalar_parameters = dict(scalar.transformation_chain[-1].parameters)
    vector_parameters = dict(vector.transformation_chain[-1].parameters)
    assert scalar_parameters["incident_quadrature_partition_strategy"] == "per_output_bin"
    assert vector_parameters["incident_quadrature_partition_strategy"] == "global_union"
    assert scalar_parameters["incident_quadrature_response_evaluation_unit"] == (
        "scalar_exit_bin_response"
    )
    assert vector_parameters["incident_quadrature_response_evaluation_unit"] == (
        "vector_exit_spectrum_response"
    )
    assert int(scalar_parameters["incident_quadrature_response_bin_evaluations"]) == int(
        scalar_parameters["incident_quadrature_response_evaluations"]
    )
    assert int(vector_parameters["incident_quadrature_response_bin_evaluations"]) == len(
        source.x
    ) * int(vector_parameters["incident_quadrature_response_evaluations"])


def _integrated_uniform_birth_mass(
    *, lower: float, upper: float, birth_lower: float, birth_upper: float, thickness: float
) -> float:
    """Exact depth integral of a unit-normalized uniform emitted-energy PDF."""

    def integrated_cdf(edge: float) -> float:
        return 0.5 * (
            max(0.0, edge + thickness - birth_lower) ** 2
            - max(0.0, edge - birth_lower) ** 2
            - max(0.0, edge + thickness - birth_upper) ** 2
            + max(0.0, edge - birth_upper) ** 2
        ) / (birth_upper - birth_lower)

    return integrated_cdf(upper) - integrated_cdf(lower)


@pytest.mark.parametrize("peak", [False, True])
def test_each_original_output_bin_matches_independent_uniform_birth_integral(peak: bool) -> None:
    source = replace(
        _source(peak=peak),
        x=(0.5, 1.0, 2.0, 4.0, 7.0, 8.0),
        y=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    kernel = _kernel(birth_edges=(1.1, 4.5))
    thickness = 0.2
    vector = _calculate(source, strategy="global_union", kernel=kernel, thickness=thickness)
    scalar = _calculate(source, strategy="per_output_bin", kernel=kernel, thickness=thickness)
    _assert_same_bins(scalar, vector)
    assert scalar.bin_edges is not None
    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    # Source integral on 7..8 is 1/2. Total primary+secondary path is t.
    factor = number_per_gram * 2.0 * BARN_CM2 * 0.5 * exp(
        -number_per_gram * 3.0 * BARN_CM2 * thickness
    )
    expected_counts = tuple(
        factor
        * _integrated_uniform_birth_mass(
            lower=lower,
            upper=upper,
            birth_lower=1.1,
            birth_upper=4.5,
            thickness=thickness,
        )
        for lower, upper in zip(scalar.bin_edges, scalar.bin_edges[1:])
    )
    actual_counts = tuple(
        density * (upper - lower)
        for density, lower, upper in zip(
            scalar.y, scalar.bin_edges, scalar.bin_edges[1:]
        )
    )
    assert actual_counts == pytest.approx(expected_counts, rel=1e-7, abs=1e-13)
    assert _counts(scalar) == pytest.approx(factor * thickness, rel=1e-7)
    assert sum(value > 1e-12 for value in actual_counts) >= 3


@pytest.mark.parametrize("peak", [False, True])
def test_unsupported_output_bins_stay_zero_without_losing_supported_counts(peak: bool) -> None:
    source = replace(
        _source(peak=peak),
        x=(0.05, 0.5, 1.0, 2.0, 4.0, 7.0, 8.0, 25.0),
        y=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    kernel = _kernel(birth_edges=(1.1, 4.5))
    vector = _calculate(source, strategy="global_union", kernel=kernel, thickness=0.2)
    scalar = _calculate(source, strategy="per_output_bin", kernel=kernel, thickness=0.2)
    _assert_same_bins(scalar, vector)
    assert scalar.bin_edges is not None
    for index, (lower, upper) in enumerate(zip(scalar.bin_edges, scalar.bin_edges[1:])):
        if lower < 0.1 or upper > 20.0:
            assert scalar.y[index] == 0.0
    assert _counts(scalar) > 0.0
    assert scalar.completeness is SpectrumCompleteness.INCOMPLETE
    assert any(
        reason.startswith(PROTON_AL_SECONDARY_TRANSPORT_DOMAIN_REASON_PREFIX)
        for reason in scalar.incomplete_reasons
    )


@pytest.mark.parametrize("peak", [False, True])
def test_zero_source_produces_exact_zero_in_both_partition_strategies(peak: bool) -> None:
    source = _source(peak=peak)
    source = replace(source, y=(0.0,) * len(source.x))
    vector = _calculate(source, strategy="global_union")
    scalar = _calculate(source, strategy="per_output_bin")
    _assert_same_bins(scalar, vector)
    assert scalar.y == (0.0,) * len(source.x)
    assert scalar.y_unit is source.y_unit
    assert scalar.quantity is source.quantity
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON not in scalar.incomplete_reasons


@pytest.mark.parametrize("strategy", ["global_union", "per_output_bin"])
def test_zero_shield_does_not_activate_partition_strategy(strategy: str) -> None:
    source = _source()
    result = _calculate(source, strategy=strategy, thickness=0.0)
    assert result.y == (0.0,) * len(source.x)
    assert result.incomplete_reasons == ()
    assert result.ordinate_basis is source.ordinate_basis
    assert result.model_reference is source.model_reference
    assert result.transformation_chain[-1].parameters == (("thickness_g_cm2", "0"),)


def test_per_bin_evaluation_budget_is_global_not_reset_for_each_output_bin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = secondary_module._single_incident_secondary_response
    evaluated_group_sizes = []

    def counted_response(**kwargs: Any) -> Any:
        evaluated_group_sizes.append(len(kwargs["output_centers_mev"]))
        return original(**kwargs)

    monkeypatch.setattr(secondary_module, "_single_incident_secondary_response", counted_response)
    source = replace(
        _source(),
        x=(0.5, 1.0, 2.0, 4.0, 7.0, 8.0),
        y=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    result = _calculate(
        source, strategy="per_output_bin", kernel=_kernel(birth_edges=(1.1, 4.5)),
        thickness=0.2, budget=4,
    )
    parameters = dict(result.transformation_chain[-1].parameters)
    assert int(parameters["incident_quadrature_response_evaluations"]) == len(evaluated_group_sizes)
    assert 0 < len(evaluated_group_sizes) <= 4
    assert all(size == 1 for size in evaluated_group_sizes)
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON in result.incomplete_reasons


@pytest.mark.parametrize("value", ["", "per_input_bin", "GLOBAL_UNION", None, 1, True])
def test_partition_strategy_rejects_unknown_values(value: object) -> None:
    with pytest.raises(ValueError):
        IncidentEnergyQuadrature(partition_strategy=value)  # type: ignore[arg-type]


def test_default_partition_strategy_remains_global_union() -> None:
    assert IncidentEnergyQuadrature().partition_strategy == "global_union"


@pytest.mark.parametrize("peak", [False, True])
def test_nonlinear_survival_strategies_agree_after_independent_depth_refinement(
    peak: bool,
) -> None:
    """Different complete partitions need not have identical finite-GL4 errors.

    The source is a piecewise-linear triangle on 8..12 MeV, not a single
    entrance-energy sample. With R(E)=E and piecewise-linear MT3, optical
    depth is piecewise quadratic; its exponential makes GL4 non-exact.
    MT3 remains at least 3 barn, above the 2 barn production channel.
    """
    source = replace(
        _source(peak=peak),
        x=(1.0, 2.0, 4.0, 8.0, 10.0, 12.0),
        y=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    kernel = _kernel(birth_edges=(2.0, 7.0))
    assert kernel.nonelastic_cross_section is not None
    kernel = replace(
        kernel,
        nonelastic_cross_section=replace(
            kernel.nonelastic_cross_section,
            energy_mev=(0.1, 5.0, 20.0),
            sigma_barn=(100.0, 3.0, 100.0),
        ),
    )
    results = {
        (steps, strategy): _calculate(
            source, strategy=strategy, kernel=kernel, depth_steps=steps
        )
        for steps in (1, 8, 16)
        for strategy in ("global_union", "per_output_bin")
    }
    widths = tuple(
        upper - lower
        for lower, upper in zip(
            log_bin_edges_from_centers(source.x),
            log_bin_edges_from_centers(source.x)[1:],
        )
    )

    def count_difference(left: Spectrum1D, right: Spectrum1D) -> float:
        return sum(abs(a - b) * width for a, b, width in zip(left.y, right.y, widths))

    coarse_difference = count_difference(
        results[1, "global_union"], results[1, "per_output_bin"]
    )
    fine_difference = count_difference(
        results[8, "global_union"], results[8, "per_output_bin"]
    )
    reference_count = _counts(results[16, "global_union"])
    assert reference_count > 0.0
    assert fine_difference <= max(coarse_difference * 0.01, 1e-10 * reference_count)
    for strategy in ("global_union", "per_output_bin"):
        refined = results[8, strategy]
        finer = results[16, strategy]
        assert refined.y == pytest.approx(finer.y, rel=1e-8, abs=1e-13)
        assert finer.y_unit is source.y_unit
        assert finer.quantity is source.quantity
        assert finer.completeness is SpectrumCompleteness.INCOMPLETE
    assert results[8, "per_output_bin"].y == pytest.approx(
        results[8, "global_union"].y, rel=1e-9, abs=1e-13
    )


def _mixed_birth_kernel() -> SecondaryProtonKernel:
    kernel = _kernel()
    broad_product = replace(
        kernel.products[0],
        product_index=1,
        emitted_energy_mev_by_incident=((3.0, 4.0),) * 2,
        pdf_per_mev_by_incident=((1.0, 0.0),) * 2,
    )
    return replace(kernel, products=(*kernel.products, broad_product))


@pytest.mark.parametrize("limited_partition", ["depth", "incident"])
def test_per_bin_knot_fallback_does_not_discard_unaffected_partition(
    limited_partition: str,
) -> None:
    source = _source()
    kernel = _mixed_birth_kernel()
    reference = _calculate(source, strategy="per_output_bin", kernel=kernel)
    # Exact geometry has depth counts (5,2,0) and extra coupled counts (9,4,0).
    # Each selected limit therefore fails only for the first exit interval.
    depth_options = (
        DepthCrossingQuadrature(max_depth_knots=2, max_coupled_incident_knots=16_384)
        if limited_partition == "depth"
        else DepthCrossingQuadrature(max_depth_knots=4096, max_coupled_incident_knots=4)
    )
    result = _calculate(
        source, strategy="per_output_bin", kernel=kernel, depth_options=depth_options
    )
    parameters = dict(result.transformation_chain[-1].parameters)
    exceeded_key = (
        "depth_output_crossing_budget_exceeded_by_partition"
        if limited_partition == "depth"
        else "incident_crossing_budget_exceeded_by_partition"
    )
    assert parameters[exceeded_key] == "true,false,false"
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON in result.incomplete_reasons
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON in result.incomplete_reasons
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    assert result.y[1] > 0.0
    assert result.y[1] == pytest.approx(reference.y[1], rel=1e-10)
    assert result.y[2] == reference.y[2] == 0.0
    if limited_partition == "depth":
        assert parameters["depth_output_crossing_counts_by_partition"] == "0,2,0"
    else:
        assert parameters["depth_output_crossing_counts_by_partition"] == "5,2,0"
        assert parameters["incident_partition_intervals_by_partition"].split(",")[0] == "4"


def test_early_positive_count_survives_global_budget_exhaustion_for_later_bins() -> None:
    source = _source()
    kernel = _mixed_birth_kernel()
    reference = _calculate(source, strategy="per_output_bin", kernel=kernel)
    result = _calculate(source, strategy="per_output_bin", kernel=kernel, budget=4)
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_response_evaluations"] == "4"
    assert 0.0 < result.y[0] <= reference.y[0] * (1.0 + 1e-9)
    assert reference.y[1] > 0.0
    assert result.y[1] == 0.0
    unresolved = tuple(
        int(value)
        for value in parameters["incident_quadrature_unresolved_intervals_by_partition"].split(",")
    )
    assert unresolved[1] > 0
    assert sum(unresolved) == int(parameters["incident_quadrature_unresolved_intervals"])
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"
    assert parameters["incident_quadrature_estimated_relative_error"] == "unavailable"
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON in result.incomplete_reasons


def test_converged_per_bin_estimator_does_not_remove_absent_physics_or_depth_validation() -> None:
    result = _calculate(_source(), strategy="per_output_bin", kernel=_mixed_birth_kernel())
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_status"] == "converged"
    assert parameters["incident_quadrature_unresolved_intervals"] == "0"
    assert float(parameters["incident_quadrature_estimated_relative_error"]) < 1e-7
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    assert {
        PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
        PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
        PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
        PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON,
    }.issubset(result.incomplete_reasons)
