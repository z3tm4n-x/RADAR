"""A missed narrow exit window must not certify incident-energy convergence."""

from dataclasses import replace
from math import exp, sqrt

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.proton_al import ProtonAlRangeEnergyTable
from radar.shielding.proton_al_secondary import (
    PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON,
    Al27CrossSectionTable,
    DepthCrossingQuadrature,
    IncidentEnergyQuadrature,
    SecondaryProtonKernel,
    SecondaryProtonProductDistribution,
    calculate_secondary_proton_spectrum_through_al,
)
from radar.shielding.proton_al_survival import (
    AL27_MOLAR_MASS_G_MOL,
    AVOGADRO_CONSTANT_MOL_INV,
    BARN_CM2,
    Al27CrossSectionDiscontinuity,
    Al27NonelasticCrossSectionTable,
)


def _counts(spectrum: Spectrum1D) -> float:
    assert spectrum.bin_edges is not None
    return sum(
        value * (upper - lower)
        for value, lower, upper in zip(spectrum.y, spectrum.bin_edges, spectrum.bin_edges[1:])
    )


@pytest.mark.parametrize("peak", [False, True])
def test_zero_quadrature_samples_do_not_prove_absent_narrow_secondary_response(
    peak: bool,
) -> None:
    """Both initial GL4 rules miss E0 approximately 7.988..8 MeV.

    R(E)=E and S=1. A reaction is possible only above 5 MeV, and its
    uniform birth PDF occupies 0.39..0.3901 MeV. To reach the first output
    edge 1/sqrt(7), production must occur near the exit of the 3 g/cm2 slab.
    The positive response window is not a direct or full-depth reaction knot.
    """
    stopping = ProtonAlRangeEnergyTable(
        energy_mev=(0.1, 20.0),
        range_g_cm2=(0.1, 20.0),
        stopping_mev_cm2_g=(1.0, 1.0),
    )
    kernel = SecondaryProtonKernel(
        products=(
            SecondaryProtonProductDistribution(
                mt=5,
                product_index=0,
                incident_energy_mev=(0.1, 20.0),
                yield_value=(1.0, 1.0),
                emitted_energy_mev_by_incident=((0.39, 0.3901),) * 2,
                pdf_per_mev_by_incident=((10_000.0, 0.0),) * 2,
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
            mt=3,
            energy_mev=(0.1, 20.0),
            sigma_barn=(3.0, 3.0),
        ),
    )
    source = Spectrum1D(
        x=(1.0, 7.0, 8.0),
        y=(0.0, 0.0, 1.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX if peak else Unit.DIFFERENTIAL_FLUENCE,
        quantity=(
            SpectrumQuantity.DIFFERENTIAL_FLUX if peak else SpectrumQuantity.DIFFERENTIAL_FLUENCE
        ),
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="synthetic_narrow_secondary_exit_window",
    )
    options = IncidentEnergyQuadrature(max_response_evaluations=256, max_refinement_depth=2)

    def calculate(spectrum: Spectrum1D) -> Spectrum1D:
        return calculate_secondary_proton_spectrum_through_al(
            incident_spectrum=spectrum,
            stopping_table=stopping,
            kernel=kernel,
            thickness_g_cm2=3.0,
            depth_steps=128,
            incident_quadrature=options,
            # Preserve the zero-estimator regression on the explicit legacy
            # fallback. Complete crossing partitions have their own exact tests.
            depth_quadrature=DepthCrossingQuadrature(max_depth_knots=0),
        )

    result = calculate(source)
    parameters = dict(result.transformation_chain[-1].parameters)
    assert parameters["incident_quadrature_status"] == "incomplete"
    assert PROTON_AL_SECONDARY_INCIDENT_QUADRATURE_UNRESOLVED_REASON in result.incomplete_reasons
    assert result.y_unit is source.y_unit
    assert result.quantity is source.quantity

    # Adding this exactly collinear source point exposes the missed window.
    # The first output edge remains 1/sqrt(7), so output support is unchanged.
    exposed = calculate(replace(source, x=(1.0, 7.0, 7.99, 8.0), y=(0.0, 0.0, 0.99, 1.0)))
    assert _counts(exposed) > 0.0

    # Independent analytic source/depth/birth integral: with u=8-E0 and
    # w=E_birth-E_exit_min, integrate (1-u)*(w-u) on 0<=u<=w, then average
    # over the uniform birth PDF. Total primary+secondary path is exactly 3.
    lower_w = 0.39 - 1.0 / sqrt(7.0)
    upper_w = 0.3901 - 1.0 / sqrt(7.0)
    source_depth_integral = (
        (upper_w**3 - lower_w**3) / 6.0 - (upper_w**4 - lower_w**4) / 24.0
    ) / 0.0001
    n = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
    exact_count = n * 2.0 * BARN_CM2 * exp(-n * 3.0 * BARN_CM2 * 3.0) * source_depth_integral
    assert exact_count == pytest.approx(2.6559853528780986e-6, rel=1e-12)
    # This test does not certify depth accuracy: the missing depth crossing
    # partitions are a separate issue. It establishes that zero is false.
    assert _counts(exposed) > 0.5 * exact_count
