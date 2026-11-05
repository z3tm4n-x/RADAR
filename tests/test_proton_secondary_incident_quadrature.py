"""Independent checks of the opt-in outer secondary-proton source integral."""

from dataclasses import replace
from math import exp

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.shielding.proton_al import ProtonAlRangeEnergyTable
from radar.shielding.proton_al_secondary import (
    PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
    PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON,
    PROTON_AL_SECONDARY_DEPTH_CROSSING_QUADRATURE_REASON,
    PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON,
    PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
    PROTON_AL_SECONDARY_INCIDENT_BIN_QUADRATURE_REASON,
    PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON,
    PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
    PROTON_AL_SECONDARY_TRANSPORT_DOMAIN_REASON_PREFIX,
    Al27CrossSectionTable,
    IncidentEnergyQuadrature,
    SecondaryProtonKernel,
    SecondaryProtonProductDistribution,
    _point_source_quadrature_boundaries,
    _secondary_production_is_proven_zero,
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
from radar.shielding.resources import (
    load_normative_proton_al_range_table,
    load_normative_secondary_proton_kernel,
)


def _stopping() -> ProtonAlRangeEnergyTable:
    # An exactly consistent synthetic range/stopping pair: dR/dE = 1/S = 1.
    return ProtonAlRangeEnergyTable(
        energy_mev=(0.1, 10.0),
        range_g_cm2=(0.1, 10.0),
        stopping_mev_cm2_g=(1.0, 1.0),
    )


def _kernel(*, threshold: bool = False) -> SecondaryProtonKernel:
    return SecondaryProtonKernel(
        products=(
            SecondaryProtonProductDistribution(
                mt=5,
                product_index=0,
                incident_energy_mev=(0.1, 10.0),
                yield_value=(1.0, 1.0),
                emitted_energy_mev_by_incident=((1.5, 2.5),) * 2,
                pdf_per_mev_by_incident=((1.0, 0.0),) * 2,
            ),
        ),
        cross_sections_by_mt={
            5: Al27CrossSectionTable(
                mt=5,
                energy_mev=(0.1, 3.5, 10.0) if threshold else (0.1, 10.0),
                sigma_barn=(0.0, 2.0, 2.0) if threshold else (2.0, 2.0),
                discontinuities=(
                    Al27CrossSectionDiscontinuity(
                        energy_mev=3.5,
                        left_sigma_barn=0.0,
                        right_sigma_barn=2.0,
                    ),
                )
                if threshold
                else (),
            )
        },
        nonelastic_cross_section=Al27NonelasticCrossSectionTable(
            mt=3,
            energy_mev=(0.1, 10.0),
            sigma_barn=(3.0, 3.0),
        ),
    )


def _source(*, refined: bool = False, peak: bool = False) -> Spectrum1D:
    x = tuple(float(index) for index in range(1, 9))
    y = (0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 0.0, 0.0)
    if refined:
        pairs = [
            pair
            for index in range(len(x) - 1)
            for pair in (
                (x[index], y[index]),
                ((x[index] + x[index + 1]) / 2, (y[index] + y[index + 1]) / 2),
            )
        ] + [(x[-1], y[-1])]
        x, y = tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
    return Spectrum1D(
        x=x,
        y=y,
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX if peak else Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUX
        if peak
        else SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="synthetic_linear_source",
    )


def _counts(spectrum: Spectrum1D) -> float:
    assert spectrum.bin_edges is not None
    return sum(
        density * (upper - lower)
        for density, lower, upper in zip(spectrum.y, spectrum.bin_edges, spectrum.bin_edges[1:])
    )


@pytest.mark.parametrize("threshold,integrated_source_depth", [(False, 1.8), (True, 2069 / 1200)])
@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize("peak", [False, True])
def test_incident_quadrature_matches_independent_source_depth_integral(
    threshold: bool,
    integrated_source_depth: float,
    refined: bool,
    peak: bool,
) -> None:
    source = _source(refined=refined, peak=peak)
    result = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=source,
        stopping_table=_stopping(),
        kernel=_kernel(threshold=threshold),
        thickness_g_cm2=0.1,
        depth_steps=1,
        incident_quadrature=IncidentEnergyQuadrature(),
    )
    n = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    # Uniform birth PDF is fully retained; primary + secondary path = t.
    # Threshold removes integral over E0 < 3.5 + depth, yielding 2069/1200.
    expected = n * 2.0 * BARN_CM2 * exp(-n * 3.0 * BARN_CM2 * 0.1) * integrated_source_depth
    assert _counts(result) == pytest.approx(expected, rel=1e-10)
    assert result.y_unit is source.y_unit
    assert result.quantity is source.quantity
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_status"] == "converged"
    assert parameters["incident_source_extrapolation"] == "none"
    assert parameters["incident_source_support_mev"] == "1..8"
    assert PROTON_AL_SECONDARY_INCIDENT_BIN_QUADRATURE_REASON not in result.incomplete_reasons
    assert (
        PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON not in result.incomplete_reasons
    )
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    assert set(result.incomplete_reasons) == {
        PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
        PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
        PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
        PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON,
    }


def test_incident_boundaries_include_direct_and_mapped_thresholds_and_independent_grids() -> None:
    kernel = _kernel(threshold=True)
    product = replace(
        kernel.products[0],
        incident_energy_mev=(0.1, 4.2, 10.0),
        yield_value=(1.0,) * 3,
        pdf_incident_energy_mev=(0.1, 5.3, 10.0),
        emitted_energy_mev_by_incident=((1.5, 2.5),) * 3,
        pdf_per_mev_by_incident=((1.0, 0.0),) * 3,
    )
    kernel = replace(kernel, products=(product,))
    boundaries = _point_source_quadrature_boundaries(
        source_nodes=(0.15, 8.0),
        incident_energy_knots=_secondary_incident_energy_knots(
            kernel=kernel, stopping_table=_stopping()
        ),
        stopping_table=_stopping(),
        thickness_g_cm2=0.1,
    )
    assert boundaries == pytest.approx((0.15, 0.2, 3.5, 3.6, 4.2, 4.3, 5.3, 5.4, 8.0))


def test_response_budget_exhaustion_keeps_partial_counts_and_an_explicit_blocker() -> None:
    result = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_source(),
        stopping_table=_stopping(),
        kernel=_kernel(),
        thickness_g_cm2=0.1,
        depth_steps=1,
        incident_quadrature=IncidentEnergyQuadrature(max_response_evaluations=4),
    )
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_response_evaluations"] == "4"
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert parameters["incident_quadrature_estimated_absolute_error"] == "unavailable"
    assert int(parameters["incident_quadrature_unresolved_intervals"]) > 0
    assert _counts(result) > 0
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON in result.incomplete_reasons
    assert PROTON_AL_SECONDARY_INCIDENT_BIN_QUADRATURE_REASON not in result.incomplete_reasons


def test_refinement_limit_does_not_claim_requested_accuracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radar.shielding.proton_al_secondary as secondary

    def nonlinear_response(**kwargs):
        energy = kwargs["entrance_energy_mev"]
        return secondary._SingleIncidentResponse(
            counts=(exp(energy),) + (0.0,) * (len(kwargs["output_centers_mev"]) - 1),
            reaction_energies=(energy, energy),
            attenuation_energies=(energy, energy),
            stopped_below_range_support=False,
            range_domain_mappings=0,
            production_samples=1,
            attenuation_missing=False,
        )

    # A smooth non-polynomial response still needs refinement after all knots
    # are split. The old threshold example is now integrated exactly.
    monkeypatch.setattr(secondary, "_single_incident_secondary_response", nonlinear_response)
    result = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_source(),
        stopping_table=_stopping(),
        kernel=_kernel(threshold=True),
        thickness_g_cm2=0.1,
        depth_steps=1,
        incident_quadrature=IncidentEnergyQuadrature(
            relative_tolerance=1e-12,
            max_refinement_depth=0,
        ),
    )
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert float(parameters["incident_quadrature_estimated_absolute_error"]) > 0
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON in result.incomplete_reasons


def test_nonzero_source_endpoints_do_not_create_half_bin_tails() -> None:
    source = replace(_source(), y=(1.0,) * 8)
    result = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=source,
        stopping_table=_stopping(),
        kernel=_kernel(),
        thickness_g_cm2=0.1,
        depth_steps=1,
        incident_quadrature=IncidentEnergyQuadrature(),
    )
    n = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    expected = n * 2 * BARN_CM2 * exp(-n * 3 * BARN_CM2 * 0.1) * 0.7
    assert _counts(result) == pytest.approx(expected, rel=1e-10)


def test_point_source_tails_are_not_invented_and_outside_range_is_not_extrapolated() -> None:
    source = replace(_source(), x=(1.0, 3.0, 5.0, 8.0, 12.0), y=(0.0, 0.0, 6.0, 1.0, 1.0))
    result = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=source,
        stopping_table=_stopping(),
        kernel=_kernel(),
        thickness_g_cm2=0.1,
        depth_steps=1,
        incident_quadrature=IncidentEnergyQuadrature(),
    )
    assert _counts(result) > 0
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_status"] == "converged_on_supported_domain"
    assert int(parameters["incident_omitted_range_intervals"]) > 0
    assert any(
        reason.startswith(PROTON_AL_SECONDARY_TRANSPORT_DOMAIN_REASON_PREFIX)
        for reason in result.incomplete_reasons
    )


def test_quadrature_zero_thickness_remains_exact_and_adds_no_transport_data() -> None:
    result = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_source(),
        stopping_table=_stopping(),
        kernel=_kernel(),
        thickness_g_cm2=0.0,
        incident_quadrature=IncidentEnergyQuadrature(),
    )
    assert result.y == (0.0,) * 8
    assert result.incomplete_reasons == ()
    assert result.ordinate_basis is SpectrumOrdinateBasis.POINT_SAMPLE
    assert result.transformation_chain[-1].parameters == (("thickness_g_cm2", "0"),)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relative_tolerance": 0.0},
        {"relative_tolerance": float("nan")},
        {"relative_tolerance": 1.0},
        {"max_response_evaluations": 3},
        {"max_response_evaluations": float("nan")},
        {"max_response_evaluations": 4.5},
        {"max_refinement_depth": -1},
        {"max_refinement_depth": float("nan")},
    ],
)
def test_quadrature_rejects_invalid_options(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        IncidentEnergyQuadrature(**kwargs)  # type: ignore[arg-type]


def test_quadrature_does_not_silently_reinterpret_declared_bin_averages() -> None:
    source = replace(
        _source(),
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=tuple(float(index) + 0.5 for index in range(9)),
    )
    with pytest.raises(ValueError, match="POINT_SAMPLE"):
        calculate_secondary_proton_spectrum_through_al(
            incident_spectrum=source,
            stopping_table=_stopping(),
            kernel=_kernel(),
            thickness_g_cm2=0.1,
            incident_quadrature=IncidentEnergyQuadrature(),
        )


@pytest.mark.parametrize("thickness", [float("nan"), float("inf"), -float("inf")])
def test_secondary_partition_rejects_nonfinite_thickness(thickness: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate_secondary_proton_spectrum_through_al(
            incident_spectrum=_source(),
            stopping_table=_stopping(),
            kernel=_kernel(),
            thickness_g_cm2=thickness,
            incident_quadrature=IncidentEnergyQuadrature(),
        )


def test_zero_reaction_proof_uses_whole_depth_energy_domain_and_threshold_limits() -> None:
    arguments = dict(
        stopping_table=_stopping(),
        kernel=_kernel(threshold=True),
        thickness_g_cm2=0.1,
    )
    assert _secondary_production_is_proven_zero(
        entrance_lower_mev=3.0,
        entrance_upper_mev=3.5,
        **arguments,
    )
    assert not _secondary_production_is_proven_zero(
        entrance_lower_mev=3.0,
        entrance_upper_mev=3.51,
        **arguments,
    )


@pytest.mark.parametrize("quadrature", [None, IncidentEnergyQuadrature()])
def test_completed_crossing_partitions_do_not_claim_general_depth_accuracy(
    quadrature: IncidentEnergyQuadrature | None,
) -> None:
    result = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_source(),
        stopping_table=_stopping(),
        kernel=_kernel(),
        thickness_g_cm2=0.1,
        depth_steps=1,
        incident_quadrature=quadrature,
    )
    parameters = dict(result.transformation_chain[-1].parameters)
    assert (
        parameters["depth_output_crossing_knots"] == "all_supported_pdf_range_attenuation_crossings"
    )
    assert parameters["depth_output_crossing_accuracy"] == "not_established_by_refinement"
    assert PROTON_AL_SECONDARY_DEPTH_ACCURACY_REASON in result.incomplete_reasons
    assert PROTON_AL_SECONDARY_DEPTH_CROSSING_QUADRATURE_REASON not in result.incomplete_reasons


def test_real_tendl_narrow_source_is_invariant_to_collinear_incident_refinement() -> None:
    """Actual tables, synthetic 30--40 MeV triangle, not a full GCR benchmark.

    Only active-source intervals are doubled; exit-domain limits remain fixed.
    Depth convergence and absent physics are deliberately not certified here.
    """
    kernel = load_normative_secondary_proton_kernel()
    stopping = load_normative_proton_al_range_table()
    totals = []
    for x in (
        (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 35.0, 40.0, 100.0),
        (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 32.5, 35.0, 37.5, 40.0, 100.0),
    ):
        source = replace(
            _source(),
            x=x,
            y=tuple(max(0.0, 1 - abs(e - 35) / 5) for e in x),
            source=RadiationSource.GCR,
            model="synthetic_triangle_30_40_MeV_with_actual_TENDL",
        )
        result = calculate_secondary_proton_spectrum_through_al(
            incident_spectrum=source,
            stopping_table=stopping,
            kernel=kernel,
            thickness_g_cm2=0.01,
            depth_steps=2,
            incident_quadrature=IncidentEnergyQuadrature(max_response_evaluations=1024),
        )
        parameters = dict(result.transformation_chain[-1].parameters)
        assert parameters["incident_quadrature_status"] == "incomplete"
        assert (
            parameters["depth_output_crossing_knots"] == "budget_exceeded_legacy_partition_retained"
        )
        assert PROTON_AL_SECONDARY_PARTITION_BUDGET_REASON in result.incomplete_reasons
        assert float(parameters["incident_quadrature_estimated_relative_error"]) < 1e-3
        assert _counts(result) > 0
        assert result.completeness is SpectrumCompleteness.INCOMPLETE
        assert PROTON_AL_SECONDARY_INCIDENT_BIN_QUADRATURE_REASON not in result.incomplete_reasons
        totals.append(_counts(result))
    assert totals[1] == pytest.approx(totals[0], rel=1e-3)
