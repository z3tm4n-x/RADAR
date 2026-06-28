import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.erb.shielding import (
    ERB_ELECTRON_AL_NO_BREMSSTRAHLUNG_TRANSPORT,
    ERB_ELECTRON_AL_NO_SECONDARY_PARTICLES,
    ERB_ELECTRON_AL_SHIELDING_MODEL,
    ERB_PROTON_AL_CENTERED_SPHERICAL_SHELL,
    ERB_PROTON_AL_NO_HIDDEN_4PI,
    calculate_erb_electron_shielding_product,
    calculate_erb_electron_shielding_spectrum,
)
from radar.shielding.electron_al import ElectronAlRangeEnergyTable
from radar.shielding.hze_al import HzeAlRangeTable
from radar.shielding.hze_si_let import HzeSiLetTable
from radar.shielding.proton_al import ProtonAlRangeEnergyTable
from radar.shielding.proton_al_secondary import SecondaryProtonKernel
from radar.shielding.proton_al_survival import Al27NonelasticCrossSectionTable
from radar.shielding.proton_si_let import ProtonSiLetTable
from radar.shielding.resources import ShieldingNormativeTables


def _electron_al_table() -> ElectronAlRangeEnergyTable:
    return ElectronAlRangeEnergyTable(
        energy_mev=(1.0, 10.0, 100.0),
        range_g_cm2=(0.1, 1.0, 10.0),
        stopping_mev_cm2_g=(100.0, 10.0, 1.0),
    )


def _proton_al_table() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 10.0),
        range_g_cm2=(0.1, 1.0),
        stopping_mev_cm2_g=(10.0, 1.0),
    )


def _proton_si_table() -> ProtonSiLetTable:
    return ProtonSiLetTable(
        energy_mev=(1.0, 10.0),
        let_total_mev_cm2_mg=(10.0, 5.0),
        let_electronic_mev_cm2_mg=(8.0, 4.0),
        let_nuclear_mev_cm2_mg=(2.0, 1.0),
    )


def _hze_al_table() -> HzeAlRangeTable:
    return HzeAlRangeTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=(1.0, 10.0),
        range_g_cm2=(0.1, 1.0),
    )


def _hze_si_table() -> HzeSiLetTable:
    return HzeSiLetTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=(1.0, 10.0),
        let_total_mev_cm2_mg=(100.0, 50.0),
        let_electronic_mev_cm2_mg=(80.0, 40.0),
        let_nuclear_mev_cm2_mg=(20.0, 10.0),
    )


def _tables(
    *,
    electron_al_range: ElectronAlRangeEnergyTable | None = _electron_al_table(),
) -> ShieldingNormativeTables:
    return ShieldingNormativeTables(
        proton_al_range=_proton_al_table(),
        proton_si_let=_proton_si_table(),
        hze_al_ranges_by_z={26: _hze_al_table()},
        hze_si_let_by_z={26: _hze_si_table()},
        al27_nonelastic_xs=Al27NonelasticCrossSectionTable(
            mt=5,
            energy_mev=(1.0, 10.0),
            sigma_barn=(0.0, 0.0),
        ),
        secondary_proton_kernel=SecondaryProtonKernel(
            products=(),
            cross_sections_by_mt={},
        ),
        electron_al_range=electron_al_range,
    )


def _spectrum(
    *,
    particle: Particle = Particle.ELECTRON,
    source: RadiationSource = RadiationSource.ERB,
    quantity: SpectrumQuantity = SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
) -> Spectrum1D:
    return Spectrum1D(
        x=(3.1622776601683795, 10.0, 31.622776601683793),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=source,
        model="erb_electron_on_orbit",
    )


def test_erb_electron_shielding_zero_thickness_preserves_spectrum() -> None:
    result = calculate_erb_electron_shielding_spectrum(
        spectrum=_spectrum(),
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert result.primary.y == pytest.approx((10.0, 10.0, 10.0))
    assert result.primary.particle is Particle.ELECTRON
    assert result.primary.source is RadiationSource.ERB
    assert result.thickness_g_cm2 == pytest.approx(0.0)


def test_erb_electron_shielding_positive_thickness_changes_spectrum() -> None:
    result = calculate_erb_electron_shielding_spectrum(
        spectrum=_spectrum(),
        tables=_tables(),
        thickness_g_cm2=0.1,
    )

    assert result.primary.x == _spectrum().x
    assert result.primary.y != pytest.approx(_spectrum().y)
    assert all(value >= 0.0 for value in result.primary.y)


def test_erb_electron_shielding_requires_electron_al_table() -> None:
    with pytest.raises(ValueError, match="electron Al range table"):
        calculate_erb_electron_shielding_spectrum(
            spectrum=_spectrum(),
            tables=_tables(electron_al_range=None),
            thickness_g_cm2=0.0,
        )


def test_erb_electron_shielding_product_preserves_product_kind_and_units() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.MEAN_FLUX,
        spectrum=_spectrum(),
        label="ERB electron mean flux",
    )

    shielded = calculate_erb_electron_shielding_product(
        product=product,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert shielded.kind is RadiationProductKind.MEAN_FLUX
    assert shielded.spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert shielded.spectrum.y_unit is Unit.DIFFERENTIAL_FLUX
    assert shielded.spectrum.particle is Particle.ELECTRON
    assert shielded.spectrum.source is RadiationSource.ERB
    assert "behind Al shield 0" in shielded.label


def test_erb_electron_shielding_product_supports_mission_fluence() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=_spectrum(
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
        ),
        label="ERB electron mission fluence",
    )

    shielded = calculate_erb_electron_shielding_product(
        product=product,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert shielded.kind is RadiationProductKind.MISSION_FLUENCE
    assert shielded.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert shielded.spectrum.y_unit is Unit.DIFFERENTIAL_FLUENCE


def test_erb_electron_shielding_model_records_method_flags() -> None:
    result = calculate_erb_electron_shielding_spectrum(
        spectrum=_spectrum(),
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert ERB_ELECTRON_AL_SHIELDING_MODEL in result.primary.model
    assert ERB_PROTON_AL_CENTERED_SPHERICAL_SHELL in result.primary.model
    assert ERB_ELECTRON_AL_NO_BREMSSTRAHLUNG_TRANSPORT in result.primary.model
    assert ERB_ELECTRON_AL_NO_SECONDARY_PARTICLES in result.primary.model
    assert ERB_PROTON_AL_NO_HIDDEN_4PI in result.primary.model


def test_erb_electron_shielding_rejects_proton_spectrum() -> None:
    with pytest.raises(ValueError, match="electron spectrum"):
        calculate_erb_electron_shielding_spectrum(
            spectrum=_spectrum(particle=Particle.PROTON),
            tables=_tables(),
            thickness_g_cm2=0.0,
        )


def test_erb_electron_shielding_rejects_non_erb_spectrum() -> None:
    with pytest.raises(ValueError, match="ERB model spectrum source"):
        calculate_erb_electron_shielding_spectrum(
            spectrum=_spectrum(source=RadiationSource.SEP),
            tables=_tables(),
            thickness_g_cm2=0.0,
        )


def test_erb_electron_shielding_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        calculate_erb_electron_shielding_spectrum(
            spectrum=_spectrum(),
            tables=_tables(),
            thickness_g_cm2=-1.0,
        )
