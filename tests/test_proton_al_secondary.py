from pathlib import Path

import pytest

from radar.core.spectra import ModelReference, Spectrum1D
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.nuclear.tendl import TendlIncidentEnergySupport
from radar.shielding.proton_al import (
    ProtonAlRangeEnergyTable,
    shield_proton_spectrum_primary_through_al,
)
from radar.shielding.proton_al_secondary import (
    PROTON_AL_SECONDARY_ATTENUATION_OPERATOR,
    PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON,
    PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON,
    PROTON_AL_SECONDARY_INCIDENT_BIN_QUADRATURE_REASON,
    PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON,
    Al27CrossSectionTable,
    PdfIncidentAvailability,
    SecondaryProtonKernel,
    SecondaryProtonProductDistribution,
    _nonelastic_survival_piecewise_linear,
    calculate_secondary_proton_spectrum_through_al,
    combine_primary_and_secondary_proton_spectra,
    load_al27_cross_section_tables,
    load_secondary_proton_kernel,
    load_secondary_proton_products,
)
from radar.shielding.proton_al_survival import (
    Al27NonelasticCrossSectionTable,
    IncidentEnergyLimit,
    apply_proton_nonelastic_survival_to_primary_spectrum,
    proton_nonelastic_survival_factor,
)
from radar.shielding.resources import (
    load_normative_proton_al_range_table,
    load_normative_secondary_proton_kernel,
)


def _stopping_table() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
        range_g_cm2=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
        stopping_mev_cm2_g=(100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0),
    )


def _spectrum() -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="sep_proton_orbit_spectrum",
    )


def _kernel(
    *,
    sigma_barn: float = 1000.0,
    nonelastic_sigma_barn: float = 0.0,
) -> SecondaryProtonKernel:
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(1.0, 1000.0),
        yield_value=(1.0, 1.0),
        emitted_energy_mev_by_incident=((1.0, 1000.0), (1.0, 1000.0)),
        pdf_per_mev_by_incident=((1.0e-3, 1.0e-3), (1.0e-3, 1.0e-3)),
    )
    xs = Al27CrossSectionTable(
        mt=5,
        energy_mev=(1.0, 1000.0),
        sigma_barn=(sigma_barn, sigma_barn),
    )
    return SecondaryProtonKernel(
        products=(product,),
        cross_sections_by_mt={5: xs},
        nonelastic_cross_section=Al27NonelasticCrossSectionTable(
            mt=3,
            energy_mev=(1.0, 1000.0),
            sigma_barn=(nonelastic_sigma_barn, nonelastic_sigma_barn),
        ),
    )


def test_empty_diagnostic_kernel_is_valid_but_cannot_transport_positive_thickness() -> None:
    kernel = SecondaryProtonKernel(products=(), cross_sections_by_mt={})

    zero = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=kernel,
        thickness_g_cm2=0.0,
    )

    assert kernel.incident_energy_support is None
    assert zero.y == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="requires a populated MF=3/MF=6"):
        calculate_secondary_proton_spectrum_through_al(
            incident_spectrum=_spectrum(),
            stopping_table=_stopping_table(),
            kernel=kernel,
            thickness_g_cm2=1.0,
        )
    with pytest.raises(ValueError, match="cannot evaluate a source"):
        kernel.source_density_per_g_cm2(
            incident_energy_mev=10.0,
            emitted_energy_mev=1.0,
        )


def test_empty_diagnostic_kernel_rejects_inconsistent_data_and_support() -> None:
    populated = _kernel()
    with pytest.raises(ValueError, match="must not contain MF=3 production"):
        SecondaryProtonKernel(
            products=(),
            cross_sections_by_mt=populated.cross_sections_by_mt,
        )
    with pytest.raises(ValueError, match="must not contain an MF=3/MT=3"):
        SecondaryProtonKernel(
            products=(),
            cross_sections_by_mt={},
            nonelastic_cross_section=populated.nonelastic_cross_section,
        )
    with pytest.raises(ValueError, match="must not declare incident-energy support"):
        SecondaryProtonKernel(
            products=(),
            cross_sections_by_mt={},
            incident_energy_support=TendlIncidentEnergySupport(
                minimum_mev=0.0,
                maximum_mev=200.0,
                dataset_ref="invalid empty kernel",
            ),
        )


def test_secondary_product_interpolates_yield_and_pdf() -> None:
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(10.0, 20.0),
        yield_value=(1.0, 3.0),
        emitted_energy_mev_by_incident=((1.0, 10.0), (1.0, 10.0)),
        pdf_per_mev_by_incident=((2.0, 4.0), (6.0, 8.0)),
    )

    assert product.yield_at_incident_energy(15.0) == pytest.approx(2.0)
    assert product.pdf_at(incident_energy_mev=15.0, emitted_energy_mev=5.5) == pytest.approx(4.0)


def test_secondary_product_uses_histogram_left_at_threshold() -> None:
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(10.0, 20.0),
        yield_value=(1.0, 1.0),
        emitted_energy_mev_by_incident=((0.0, 1.0, 2.0), (0.0, 1.0, 2.0)),
        pdf_per_mev_by_incident=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )

    assert product.tabulated_pdf_integral(0) == pytest.approx(1.0)
    assert product.tabulated_pdf_integral(0) != pytest.approx(0.5)
    assert product.pdf_at(incident_energy_mev=10.0, emitted_energy_mev=0.5) == pytest.approx(1.0)
    assert product.pdf_at(incident_energy_mev=10.0, emitted_energy_mev=1.0) == 0.0


def test_interpolated_histogram_pdf_integral_uses_all_outgoing_edges() -> None:
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(10.0, 20.0),
        yield_value=(1.0, 1.0),
        emitted_energy_mev_by_incident=((0.0, 1.0, 3.0), (0.0, 2.0, 3.0)),
        pdf_per_mev_by_incident=((1.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
    )

    exact_integral = product.pdf_integral(
        incident_energy_mev=15.0,
        emitted_energy_lower_mev=0.5,
        emitted_energy_upper_mev=1.5,
    )
    midpoint_rectangle = product.pdf_at(
        incident_energy_mev=15.0,
        emitted_energy_mev=1.0,
    )

    assert exact_integral == pytest.approx(0.5)
    assert midpoint_rectangle == pytest.approx(0.25)


def test_all_normative_secondary_pdfs_are_histogram_normalized() -> None:
    kernel = load_normative_secondary_proton_kernel()
    integrals = [
        product.tabulated_pdf_integral(index)
        for product in kernel.products
        for index in range(len(product.pdf_incident_energy_mev))
    ]

    assert len(integrals) == 106
    assert integrals == pytest.approx([1.0] * len(integrals), abs=2.0e-6)


def test_normative_kernel_retains_mt5_and_independent_pdf_grid() -> None:
    kernel = load_normative_secondary_proton_kernel()
    mt5 = next(product for product in kernel.products if product.mt == 5)

    assert len(kernel.products) == 10
    assert len(mt5.incident_energy_mev) == 22
    assert len(mt5.pdf_incident_energy_mev) == 15
    assert mt5.incident_energy_mev.count(30.0) == 1
    assert mt5.yield_at_incident_energy(30.0) == pytest.approx(1.225536)
    assert mt5.pdf_availability_at_incident_energy(60.0) is PdfIncidentAvailability.TABULATED
    assert (
        mt5.pdf_availability_at_incident_energy(65.0)
        is PdfIncidentAvailability.INTERPOLATABLE
    )
    assert (
        mt5.pdf_availability_at_incident_energy(250.0)
        is PdfIncidentAvailability.OUTSIDE_EVALUATED_SUPPORT
    )
    with pytest.raises(ValueError, match=r"outside TENDL support \[0, 200\] MeV"):
        mt5.pdf_at(incident_energy_mev=250.0, emitted_energy_mev=1.0)

    with pytest.raises(ValueError, match=r"outside TENDL support \[0, 200\] MeV"):
        kernel.source_density_per_g_cm2(
            incident_energy_mev=250.0,
            emitted_energy_mev=1.0,
        )


def test_normative_cross_sections_preserve_one_sided_thresholds() -> None:
    kernel = load_normative_secondary_proton_kernel()
    mt5 = kernel.cross_sections_by_mt[5]
    mt103 = kernel.cross_sections_by_mt[103]

    assert mt5.sigma_at_energy(10.0) == 0.0
    assert mt5.sigma_at_energy(29.999) == 0.0
    assert mt5.sigma_at_energy(30.0, limit=IncidentEnergyLimit.LEFT) == 0.0
    assert mt5.sigma_at_energy(30.0) == pytest.approx(0.7151679)
    assert mt5.sigma_at_energy(30.001) < 0.7151679
    assert mt5.discontinuities[0].energy_mev == 30.0
    assert mt103.sigma_at_energy(
        30.0,
        limit=IncidentEnergyLimit.LEFT,
    ) == pytest.approx(0.1089363)
    assert mt103.sigma_at_energy(30.0) == 0.0
    assert mt103.sigma_at_energy(100.0) == 0.0


def test_normative_intermediate_incident_pdf_is_used_directly() -> None:
    kernel = load_normative_secondary_proton_kernel()
    product = next(product for product in kernel.products if product.mt == 103)
    incident_index = product.pdf_incident_energy_mev.index(10.0)
    emitted_grid = product.emitted_energy_mev_by_incident[incident_index]
    midpoint = (emitted_grid[0] + emitted_grid[1]) / 2.0

    assert 10.0 not in product.incident_energy_mev
    assert product.pdf_availability_at_incident_energy(10.0) is PdfIncidentAvailability.TABULATED
    assert product.pdf_at(incident_energy_mev=10.0, emitted_energy_mev=midpoint) == pytest.approx(
        product.pdf_per_mev_by_incident[incident_index][0]
    )


def test_secondary_spectrum_is_zero_for_zero_thickness() -> None:
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=0.0,
    )

    assert secondary.y == (0.0, 0.0, 0.0)
    assert secondary.ordinate_basis is _spectrum().ordinate_basis
    assert secondary.bin_edges == _spectrum().bin_edges
    assert PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON not in (
        secondary.incomplete_reasons
    )
    assert PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON not in (
        secondary.incomplete_reasons
    )
    assert PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON not in (
        secondary.incomplete_reasons
    )
    assert PROTON_AL_SECONDARY_INCIDENT_BIN_QUADRATURE_REASON not in (
        secondary.incomplete_reasons
    )


def test_zero_thickness_secondary_combines_with_point_sample_primary() -> None:
    spectrum = _spectrum()
    primary = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_stopping_table(),
        thickness_g_cm2=0.0,
    )
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=0.0,
    )

    total = combine_primary_and_secondary_proton_spectra(
        primary=primary,
        secondary=secondary,
    )

    assert total.y == primary.y
    assert total.ordinate_basis is spectrum.ordinate_basis


def test_zero_thickness_secondary_preserves_source_dataset_provenance_only() -> None:
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "model_reference": ModelReference(
                name="source",
                version="test",
                dataset_refs=("source-only.csv",),
            ),
        }
    )

    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=0.0,
    )

    assert secondary.model_reference is not None
    assert secondary.model_reference.dataset_refs == ("source-only.csv",)
    assert secondary.transformation_chain[-1].parameters == (
        ("thickness_g_cm2", "0"),
    )


def test_positive_thickness_secondary_records_all_used_datasets() -> None:
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "model_reference": ModelReference(
                name="source",
                version="test",
                dataset_refs=("source-only.csv",),
            ),
        }
    )

    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=1.0,
        depth_steps=4,
    )

    assert secondary.model_reference is not None
    assert secondary.model_reference.dataset_refs == (
        "source-only.csv",
        "stopping/proton_al_range.csv",
        "tendl/p_al27_mf3_xs.csv",
        "tendl/p_al27_mf6_proton_yield.csv",
        "tendl/p_al27_mf6_proton_pdf.csv",
    )


def test_secondary_spectrum_is_nonnegative_and_nonzero_for_positive_kernel() -> None:
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=1.0,
        depth_steps=4,
    )

    assert all(value >= 0.0 for value in secondary.y)
    assert any(value > 0.0 for value in secondary.y)
    assert secondary.completeness is SpectrumCompleteness.INCOMPLETE
    assert PROTON_AL_SECONDARY_ANGULAR_TRANSPORT_REASON in (
        secondary.incomplete_reasons
    )
    assert PROTON_AL_SECONDARY_REFERENCE_FRAME_REASON in (
        secondary.incomplete_reasons
    )
    assert PROTON_AL_SECONDARY_HIGHER_GENERATION_REASON in (
        secondary.incomplete_reasons
    )
    assert PROTON_AL_SECONDARY_INCIDENT_BIN_QUADRATURE_REASON in (
        secondary.incomplete_reasons
    )
    parameters = dict(secondary.transformation_chain[-1].parameters)
    assert parameters["secondary_generation"] == "single_collision_generation"
    assert parameters["higher_generation_cascade"] == "not_included"
    assert (
        parameters["incident_bin_operator"]
        == "point_sample_center_times_derived_log_bin_width"
    )


def test_secondary_operator_integrates_histogram_not_birth_center_rectangle() -> None:
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(1.0, 1000.0),
        yield_value=(1.0, 1.0),
        emitted_energy_mev_by_incident=((1.0, 2.0, 1000.0),) * 2,
        pdf_per_mev_by_incident=((1.0, 0.0, 0.0),) * 2,
    )
    kernel = SecondaryProtonKernel(
        products=(product,),
        cross_sections_by_mt={
            5: Al27CrossSectionTable(
                mt=5,
                energy_mev=(1.0, 1000.0),
                sigma_barn=(1000.0, 1000.0),
            )
        },
        nonelastic_cross_section=Al27NonelasticCrossSectionTable(
            mt=3,
            energy_mev=(1.0, 1000.0),
            sigma_barn=(0.0, 0.0),
        ),
    )
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x": (3.0, 10.0, 30.0),
            "y": (0.0, 1.0, 0.0),
            "ordinate_basis": SpectrumOrdinateBasis.BIN_AVERAGE,
            "bin_edges": (1.0, 5.0, 20.0, 50.0),
        }
    )

    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=_stopping_table(),
        kernel=kernel,
        thickness_g_cm2=0.01,
        depth_steps=4,
    )

    # The mapped birth centre of the first output bin is above 3 MeV, where
    # this histogram is zero.  Only integration over the full mapped interval
    # can retain the nonzero [1, 2) MeV contribution.
    assert secondary.y[0] > 0.0
    assert secondary.y[1:] == pytest.approx((0.0, 0.0))


def test_secondary_preserves_explicit_input_bin_edges() -> None:
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "ordinate_basis": SpectrumOrdinateBasis.BIN_AVERAGE,
            "bin_edges": (5.0, 20.0, 50.0, 200.0),
        }
    )

    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=1.0,
        depth_steps=4,
    )

    assert secondary.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE
    assert secondary.bin_edges == spectrum.bin_edges
    assert PROTON_AL_SECONDARY_INCIDENT_BIN_QUADRATURE_REASON not in (
        secondary.incomplete_reasons
    )
    parameters = dict(secondary.transformation_chain[-1].parameters)
    assert parameters["incident_bin_operator"] == "declared_bin_average_times_bin_width"


def test_secondary_spectrum_skips_below_minimum_residual_range() -> None:
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=1.0,
        depth_steps=10,
    )

    assert all(value >= 0.0 for value in secondary.y)


def test_secondary_spectrum_is_zero_for_zero_cross_section() -> None:
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(sigma_barn=0.0),
        thickness_g_cm2=1.0,
        depth_steps=4,
    )

    assert secondary.y == pytest.approx((0.0, 0.0, 0.0))


def test_mt3_survival_reduces_secondary_production_and_exit_flux() -> None:
    unattenuated = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(nonelastic_sigma_barn=0.0),
        thickness_g_cm2=1.0,
        depth_steps=16,
    )
    attenuated = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(nonelastic_sigma_barn=10_000.0),
        thickness_g_cm2=1.0,
        depth_steps=16,
    )

    assert sum(attenuated.y) < sum(unattenuated.y)
    parameters = dict(attenuated.transformation_chain[-1].parameters)
    assert parameters["primary_survival_to_depth"] == "TENDL_MF3_MT3"
    assert parameters["secondary_survival_to_exit"] == "TENDL_MF3_MT3"


def test_piecewise_linear_secondary_survival_matches_refined_reference() -> None:
    kernel = load_normative_secondary_proton_kernel()
    cross_section = kernel.nonelastic_cross_section
    assert cross_section is not None
    stopping_table = load_normative_proton_al_range_table()

    exact = _nonelastic_survival_piecewise_linear(
        output_energy_mev=30.0,
        input_energy_mev=100.0,
        stopping_table=stopping_table,
        cross_section_table=cross_section,
    )
    refined = proton_nonelastic_survival_factor(
        output_energy_mev=30.0,
        input_energy_mev=100.0,
        stopping_table=stopping_table,
        cross_section_table=cross_section,
        integration_steps=4096,
    )

    assert exact == pytest.approx(refined, rel=1.0e-7)


def test_secondary_without_mt3_survival_is_explicitly_incomplete() -> None:
    complete_kernel = _kernel()
    diagnostic_kernel = SecondaryProtonKernel(
        products=complete_kernel.products,
        cross_sections_by_mt=complete_kernel.cross_sections_by_mt,
    )

    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=diagnostic_kernel,
        thickness_g_cm2=1.0,
        depth_steps=4,
    )

    assert secondary.completeness is SpectrumCompleteness.INCOMPLETE
    assert any("MF=3/MT=3" in reason for reason in secondary.incomplete_reasons)


def test_single_generation_secondary_depth_quadrature_converges() -> None:
    coarse = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(nonelastic_sigma_barn=1000.0),
        thickness_g_cm2=1.0,
        depth_steps=64,
    )
    refined = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(nonelastic_sigma_barn=1000.0),
        thickness_g_cm2=1.0,
        depth_steps=128,
    )

    assert coarse.integral_conserved() == pytest.approx(
        refined.integral_conserved(),
        rel=0.01,
    )


def test_normative_tendl_secondary_depth_quadrature_converges() -> None:
    kernel = load_normative_secondary_proton_kernel()
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x": (20.0, 40.0, 80.0),
            "y": (0.0, 1.0, 0.0),
        }
    )
    stopping_table = load_normative_proton_al_range_table()
    coarse = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=stopping_table,
        kernel=kernel,
        thickness_g_cm2=1.0,
        depth_steps=24,
    )
    refined = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=stopping_table,
        kernel=kernel,
        thickness_g_cm2=1.0,
        depth_steps=96,
    )

    assert coarse.integral_conserved() == pytest.approx(
        refined.integral_conserved(),
        rel=0.01,
    )


def test_secondary_rejects_partial_stopping_table_bins_without_clipping() -> None:
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x": (1.0, 10.0, 100.0),
            "y": (10.0, 10.0, 10.0),
        }
    )

    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=0.1,
        depth_steps=4,
    )

    assert secondary.y[0] == 0.0
    assert secondary.completeness is SpectrumCompleteness.INCOMPLETE
    assert any(
        "output bin(s) exceed the aluminium range-table energy support" in reason
        for reason in secondary.incomplete_reasons
    )


def test_secondary_marks_birth_to_exit_range_mapping_loss_incomplete() -> None:
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x": (100.0, 300.0, 600.0),
            "y": (10.0, 10.0, 10.0),
        }
    )

    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=80.0,
        depth_steps=4,
    )

    assert secondary.completeness is SpectrumCompleteness.INCOMPLETE
    assert any(
        "output-bin/depth mapping(s) require an aluminium range above" in reason
        for reason in secondary.incomplete_reasons
    )


def _support_limited_kernel() -> SecondaryProtonKernel:
    support = TendlIncidentEnergySupport(
        minimum_mev=0.0,
        maximum_mev=200.0,
        dataset_ref="test TENDL",
    )
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(1.0, 200.0, 1000.0),
        yield_value=(1.0, 1.0, 1.0),
        emitted_energy_mev_by_incident=(
            (1.0, 1000.0),
            (1.0, 1000.0),
            (1.0, 1000.0),
        ),
        pdf_per_mev_by_incident=(
            (1.0e-3, 1.0e-3),
            (1.0e-3, 1.0e-3),
            (1.0e-3, 1.0e-3),
        ),
        incident_energy_support=support,
    )
    xs = Al27CrossSectionTable(
        mt=5,
        energy_mev=(1.0, 200.0, 1000.0),
        sigma_barn=(1.0, 1.0, 1.0),
        incident_energy_support=support,
    )
    return SecondaryProtonKernel(
        products=(product,),
        cross_sections_by_mt={5: xs},
        nonelastic_cross_section=Al27NonelasticCrossSectionTable(
            mt=3,
            energy_mev=(1.0, 200.0, 1000.0),
            sigma_barn=(1.0, 1.0, 1.0),
            incident_energy_support=support,
        ),
        incident_energy_support=support,
    )


def _spectrum_with_high_energy_tail(*, high_energy_value: float) -> Spectrum1D:
    return Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x": (10.0, 100.0, 300.0),
            "y": (10.0, 10.0, high_energy_value),
        }
    )


def test_secondary_operator_marks_nonzero_tail_above_support_incomplete() -> None:
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum_with_high_energy_tail(high_energy_value=10.0),
        stopping_table=_stopping_table(),
        kernel=_support_limited_kernel(),
        thickness_g_cm2=1.0,
        depth_steps=4,
    )

    reason = (
        "TENDL p+Al-27 secondary-proton production is incomplete: nonzero "
        "transported proton contribution exists outside TENDL incident-energy "
        "support [0, 200] MeV"
    )
    assert secondary.completeness is SpectrumCompleteness.INCOMPLETE
    assert reason in secondary.incomplete_reasons


def test_secondary_operator_does_not_flag_zero_tail_above_support() -> None:
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum_with_high_energy_tail(high_energy_value=0.0),
        stopping_table=_stopping_table(),
        kernel=_support_limited_kernel(),
        thickness_g_cm2=0.1,
        depth_steps=4,
    )

    assert not any(
        reason.startswith("TENDL p+Al-27 secondary-proton production is incomplete")
        for reason in secondary.incomplete_reasons
    )


def test_secondary_attenuation_checks_complete_incident_bin_support() -> None:
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(1.0, 1000.0),
        yield_value=(1.0, 1.0),
        emitted_energy_mev_by_incident=((1.0, 10.0, 1000.0),) * 2,
        pdf_per_mev_by_incident=((1.0 / 9.0, 0.0, 0.0),) * 2,
    )
    production_cross_section = Al27CrossSectionTable(
        mt=5,
        energy_mev=(1.0, 1000.0),
        sigma_barn=(1.0, 1.0),
    )
    attenuation_support = TendlIncidentEnergySupport(
        minimum_mev=0.0,
        maximum_mev=200.0,
        dataset_ref="test MT3",
    )
    kernel = SecondaryProtonKernel(
        products=(product,),
        cross_sections_by_mt={5: production_cross_section},
        nonelastic_cross_section=Al27NonelasticCrossSectionTable(
            mt=3,
            energy_mev=(1.0, 200.0, 1000.0),
            sigma_barn=(1.0, 1.0, 1.0),
            incident_energy_support=attenuation_support,
        ),
    )
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x": (5.0, 50.0, 190.0),
            "y": (0.0, 0.0, 1.0),
            "ordinate_basis": SpectrumOrdinateBasis.BIN_AVERAGE,
            "bin_edges": (1.0, 10.0, 100.0, 220.0),
        }
    )

    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=_stopping_table(),
        kernel=kernel,
        thickness_g_cm2=0.1,
        depth_steps=4,
    )

    assert secondary.integral_conserved() > 0.0
    assert any(
        reason.startswith(f"{PROTON_AL_SECONDARY_ATTENUATION_OPERATOR} is incomplete")
        and "[0, 200] MeV" in reason
        for reason in secondary.incomplete_reasons
    )


def test_combine_primary_and_secondary_proton_spectra() -> None:
    primary = _spectrum()
    secondary = Spectrum1D(
        x=primary.x,
        y=(1.0, 2.0, 3.0),
        x_unit=primary.x_unit,
        y_unit=primary.y_unit,
        quantity=primary.quantity,
        particle=Particle.PROTON,
        source=primary.source,
        model="secondary",
    )

    total = combine_primary_and_secondary_proton_spectra(
        primary=primary,
        secondary=secondary,
    )

    assert total.y == pytest.approx((11.0, 12.0, 13.0))


def test_load_secondary_proton_kernel(tmp_path: Path) -> None:
    xs_path = tmp_path / "xs.csv"
    yield_path = tmp_path / "yield.csv"
    pdf_path = tmp_path / "pdf.csv"

    xs_path.write_text(
        "mt,energy_mev,sigma_barn\n"
        "5,1.0,10.0\n"
        "5,100.0,20.0\n",
        encoding="utf-8",
    )
    yield_path.write_text(
        "mt,product_index,incident_energy_mev,yield\n"
        "5,0,1.0,1.0\n"
        "5,0,100.0,2.0\n",
        encoding="utf-8",
    )
    pdf_path.write_text(
        "mt,product_index,incident_energy_mev,emitted_energy_mev,pdf_per_mev\n"
        "5,0,1.0,1.0,0.1\n"
        "5,0,1.0,10.0,0.2\n"
        "5,0,100.0,1.0,0.3\n"
        "5,0,100.0,10.0,0.4\n",
        encoding="utf-8",
    )

    kernel = load_secondary_proton_kernel(
        xs_path=xs_path,
        yield_path=yield_path,
        pdf_path=pdf_path,
    )

    assert len(kernel.products) == 1
    assert set(kernel.cross_sections_by_mt) == {5}
    assert kernel.nonelastic_cross_section is None
    assert kernel.source_density_per_g_cm2(
        incident_energy_mev=50.0,
        emitted_energy_mev=5.0,
    ) > 0.0


def test_production_loader_rejects_empty_mf6_kernel(tmp_path: Path) -> None:
    yield_path = tmp_path / "empty_yield.csv"
    pdf_path = tmp_path / "empty_pdf.csv"
    yield_path.write_text(
        "mt,product_index,incident_energy_mev,yield\n",
        encoding="utf-8",
    )
    pdf_path.write_text(
        "mt,product_index,incident_energy_mev,emitted_energy_mev,pdf_per_mev\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="found no MF=6 proton products"):
        load_secondary_proton_kernel(
            xs_path=tmp_path / "not-read.csv",
            yield_path=yield_path,
            pdf_path=pdf_path,
        )


def test_cross_section_loader_retains_conflicting_duplicate_as_jump(
    tmp_path: Path,
) -> None:
    xs_path = tmp_path / "xs.csv"
    xs_path.write_text(
        "mt,energy_mev,sigma_barn\n"
        "5,0.25,0.0\n"
        "5,30.0,0.0\n"
        "5,30.0,0.7151679\n"
        "5,35.0,0.6802687\n",
        encoding="utf-8",
    )

    table = load_al27_cross_section_tables(xs_path)[5]

    assert table.energy_mev == (0.25, 30.0, 35.0)
    assert table.sigma_at_energy(10.0) == 0.0
    assert table.sigma_at_energy(30.0, limit=IncidentEnergyLimit.LEFT) == 0.0
    assert table.sigma_at_energy(30.0) == pytest.approx(0.7151679)


def _write_duplicate_yield_fixture(
    tmp_path: Path,
    *,
    duplicate_value: float,
) -> tuple[Path, Path]:
    yield_path = tmp_path / f"yield_{duplicate_value}.csv"
    pdf_path = tmp_path / f"pdf_{duplicate_value}.csv"
    yield_path.write_text(
        "mt,product_index,incident_energy_mev,yield\n"
        "5,1,30.0,1.225536\n"
        f"5,1,30.0,{duplicate_value}\n"
        "5,1,40.0,1.5\n",
        encoding="utf-8",
    )
    pdf_path.write_text(
        "mt,product_index,incident_energy_mev,emitted_energy_mev,pdf_per_mev\n"
        "5,1,30.0,0.0,1.0\n"
        "5,1,30.0,1.0,0.0\n"
        "5,1,40.0,0.0,1.0\n"
        "5,1,40.0,1.0,0.0\n",
        encoding="utf-8",
    )
    return yield_path, pdf_path


def test_exact_duplicate_yield_is_safely_deduplicated(tmp_path: Path) -> None:
    yield_path, pdf_path = _write_duplicate_yield_fixture(
        tmp_path,
        duplicate_value=1.225536,
    )

    products = load_secondary_proton_products(yield_path=yield_path, pdf_path=pdf_path)

    assert len(products) == 1
    assert products[0].incident_energy_mev == (30.0, 40.0)
    assert products[0].yield_value == pytest.approx((1.225536, 1.5))


def test_conflicting_duplicate_yield_is_rejected(tmp_path: Path) -> None:
    yield_path, pdf_path = _write_duplicate_yield_fixture(
        tmp_path,
        duplicate_value=1.225537,
    )

    with pytest.raises(ValueError, match="Conflicting duplicate MF=6 yield"):
        load_secondary_proton_products(yield_path=yield_path, pdf_path=pdf_path)


def test_primary_survived_plus_secondary_total_path() -> None:
    spectrum = _spectrum()
    stopping_table = _stopping_table()
    xs = Al27NonelasticCrossSectionTable(
        mt=3,
        energy_mev=(1.0, 1000.0),
        sigma_barn=(1000.0, 1000.0),
    )

    primary = apply_proton_nonelastic_survival_to_primary_spectrum(
        spectrum=spectrum,
        stopping_table=stopping_table,
        cross_section_table=xs,
        thickness_g_cm2=1.0,
    )
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=stopping_table,
        kernel=_kernel(),
        thickness_g_cm2=1.0,
        depth_steps=4,
    )
    total = combine_primary_and_secondary_proton_spectra(
        primary=primary,
        secondary=secondary,
    )

    assert all(t >= p for t, p in zip(total.y, primary.y, strict=True))
