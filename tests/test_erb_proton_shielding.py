import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.erb.shielding import (
    ERB_PROTON_AL_CENTERED_SPHERICAL_SHELL,
    ERB_PROTON_AL_NO_HIDDEN_4PI,
    ERB_PROTON_AL_SHIELDING_MODEL,
    calculate_erb_proton_shielding_product,
    calculate_erb_proton_shielding_spectrum,
)
from radar.shielding.hze_al import HzeAlRangeTable
from radar.shielding.hze_si_let import HzeSiLetTable
from radar.shielding.proton_al import ProtonAlRangeEnergyTable
from radar.shielding.proton_al_secondary import (
    Al27CrossSectionTable,
    SecondaryProtonKernel,
    SecondaryProtonProductDistribution,
)
from radar.shielding.proton_al_survival import Al27NonelasticCrossSectionTable
from radar.shielding.proton_si_let import ProtonSiLetTable
from radar.shielding.resources import ShieldingNormativeTables


def _proton_al_table() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
        range_g_cm2=(0.001, 0.02, 0.06, 0.2, 0.6, 2.0, 6.0, 20.0),
        stopping_mev_cm2_g=(50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
    )


def _proton_si_table() -> ProtonSiLetTable:
    return ProtonSiLetTable(
        energy_mev=(1.0, 10.0, 100.0),
        let_total_mev_cm2_mg=(10.0, 5.0, 2.0),
        let_electronic_mev_cm2_mg=(8.0, 4.0, 1.5),
        let_nuclear_mev_cm2_mg=(2.0, 1.0, 0.5),
    )


def _hze_al_table() -> HzeAlRangeTable:
    return HzeAlRangeTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=(1.0, 10.0, 100.0),
        range_g_cm2=(0.1, 1.0, 10.0),
    )


def _hze_si_table() -> HzeSiLetTable:
    return HzeSiLetTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=(1.0, 10.0, 100.0),
        let_total_mev_cm2_mg=(100.0, 50.0, 20.0),
        let_electronic_mev_cm2_mg=(80.0, 40.0, 15.0),
        let_nuclear_mev_cm2_mg=(20.0, 10.0, 5.0),
    )


def _secondary_kernel(*, sigma_barn: float = 0.0) -> SecondaryProtonKernel:
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
    )


def _tables(
    *,
    survival_sigma_barn: float = 0.0,
    secondary_sigma_barn: float = 0.0,
) -> ShieldingNormativeTables:
    return ShieldingNormativeTables(
        proton_al_range=_proton_al_table(),
        proton_si_let=_proton_si_table(),
        hze_al_ranges_by_z={26: _hze_al_table()},
        hze_si_let_by_z={26: _hze_si_table()},
        al27_nonelastic_xs=Al27NonelasticCrossSectionTable(
            mt=5,
            energy_mev=(1.0, 1000.0),
            sigma_barn=(survival_sigma_barn, survival_sigma_barn),
        ),
        secondary_proton_kernel=_secondary_kernel(sigma_barn=secondary_sigma_barn),
    )


def _spectrum(
    *,
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.ERB,
    quantity: SpectrumQuantity = SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=source,
        model="erb_proton_on_orbit",
    )


def test_erb_proton_shielding_zero_thickness_preserves_total_spectrum() -> None:
    result = calculate_erb_proton_shielding_spectrum(
        spectrum=_spectrum(),
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert result.primary.y == pytest.approx((10.0, 10.0, 10.0))
    assert result.primary_survived.y == pytest.approx((10.0, 10.0, 10.0))
    assert result.secondary.y == pytest.approx((0.0, 0.0, 0.0))
    assert result.total.y == pytest.approx((10.0, 10.0, 10.0))
    assert result.nonelastic_survival is True
    assert result.secondary_protons is True


def test_erb_proton_shielding_total_includes_secondary_when_enabled() -> None:
    result_without_secondary = calculate_erb_proton_shielding_spectrum(
        spectrum=_spectrum(),
        tables=_tables(survival_sigma_barn=0.0, secondary_sigma_barn=1000.0),
        thickness_g_cm2=1.0,
        include_secondary_protons=False,
    )
    result_with_secondary = calculate_erb_proton_shielding_spectrum(
        spectrum=_spectrum(),
        tables=_tables(survival_sigma_barn=0.0, secondary_sigma_barn=1000.0),
        thickness_g_cm2=1.0,
        include_secondary_protons=True,
        secondary_depth_steps=4,
    )

    assert sum(result_with_secondary.secondary.y) > 0.0
    assert sum(result_with_secondary.total.y) > sum(result_without_secondary.total.y)


def test_erb_proton_shielding_can_disable_nonelastic_survival() -> None:
    without_survival = calculate_erb_proton_shielding_spectrum(
        spectrum=_spectrum(),
        tables=_tables(survival_sigma_barn=1000.0),
        thickness_g_cm2=1.0,
        include_nonelastic_survival=False,
    )
    with_survival = calculate_erb_proton_shielding_spectrum(
        spectrum=_spectrum(),
        tables=_tables(survival_sigma_barn=1000.0),
        thickness_g_cm2=1.0,
        include_nonelastic_survival=True,
    )

    assert sum(with_survival.primary_survived.y) < sum(without_survival.primary_survived.y)


def test_erb_proton_shielding_product_preserves_product_kind_and_units() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.MEAN_FLUX,
        spectrum=_spectrum(),
        label="ERB proton mean flux",
    )

    shielded = calculate_erb_proton_shielding_product(
        product=product,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert shielded.kind is RadiationProductKind.MEAN_FLUX
    assert shielded.spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert shielded.spectrum.y_unit is Unit.DIFFERENTIAL_FLUX
    assert shielded.spectrum.particle is Particle.PROTON
    assert shielded.spectrum.source is RadiationSource.ERB
    assert "behind Al shield 0" in shielded.label


def test_erb_proton_shielding_product_supports_mission_fluence() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=_spectrum(
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
        ),
        label="ERB proton mission fluence",
    )

    shielded = calculate_erb_proton_shielding_product(
        product=product,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert shielded.kind is RadiationProductKind.MISSION_FLUENCE
    assert shielded.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert shielded.spectrum.y_unit is Unit.DIFFERENTIAL_FLUENCE


def test_erb_proton_shielding_model_records_method_flags() -> None:
    result = calculate_erb_proton_shielding_spectrum(
        spectrum=_spectrum(),
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert ERB_PROTON_AL_SHIELDING_MODEL in result.total.model
    assert ERB_PROTON_AL_CENTERED_SPHERICAL_SHELL in result.total.model
    assert ERB_PROTON_AL_NO_HIDDEN_4PI in result.total.model
    assert "proton_nonelastic_survival_included" in result.total.model
    assert "secondary_protons_included" in result.total.model


def test_erb_proton_shielding_rejects_electron_spectrum() -> None:
    with pytest.raises(ValueError, match="proton spectrum"):
        calculate_erb_proton_shielding_spectrum(
            spectrum=_spectrum(particle=Particle.ELECTRON),
            tables=_tables(),
            thickness_g_cm2=0.0,
        )


def test_erb_proton_shielding_rejects_non_erb_spectrum() -> None:
    with pytest.raises(ValueError, match="ERB model spectrum source"):
        calculate_erb_proton_shielding_spectrum(
            spectrum=_spectrum(source=RadiationSource.SEP),
            tables=_tables(),
            thickness_g_cm2=0.0,
        )


def test_erb_proton_shielding_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        calculate_erb_proton_shielding_spectrum(
            spectrum=_spectrum(),
            tables=_tables(),
            thickness_g_cm2=-1.0,
        )
