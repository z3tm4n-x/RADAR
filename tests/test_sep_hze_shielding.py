import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.hze_al import HzeAlRangeTable
from radar.shielding.hze_si_let import HzeSiLetTable
from radar.shielding.proton_al import ProtonAlRangeEnergyTable
from radar.shielding.proton_al_secondary import SecondaryProtonKernel
from radar.shielding.proton_al_survival import Al27NonelasticCrossSectionTable
from radar.shielding.proton_si_let import ProtonSiLetTable
from radar.shielding.resources import ShieldingNormativeTables
from radar.shielding.sep_hze import (
    SEP_HZE_AL_CENTERED_SPHERICAL_SHELL,
    SEP_HZE_AL_FRAGMENTATION_NOT_INCLUDED,
    SEP_HZE_AL_NO_HIDDEN_4PI,
    SEP_HZE_AL_NUCLEAR_ATTENUATION_NOT_INCLUDED,
    SEP_HZE_AL_SHIELDING_MODEL,
    calculate_sep_hze_shielding_product,
    calculate_sep_hze_shielding_spectra,
    calculate_sep_hze_shielding_spectrum,
)


def _proton_al_table() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 10.0),
        range_g_cm2=(0.1, 1.0),
        stopping_mev_cm2_g=(10.0, 10.0),
    )


def _proton_si_table() -> ProtonSiLetTable:
    return ProtonSiLetTable(
        energy_mev=(1.0, 10.0),
        let_total_mev_cm2_mg=(10.0, 5.0),
        let_electronic_mev_cm2_mg=(8.0, 4.0),
        let_nuclear_mev_cm2_mg=(2.0, 1.0),
    )


def _hze_al_table(
    *,
    z: int = 26,
    symbol: str = "Fe",
    mass_number: float = 56.0,
) -> HzeAlRangeTable:
    return HzeAlRangeTable(
        z=z,
        symbol=symbol,
        mass_number=mass_number,
        mass_to_charge=mass_number / z,
        energy_mev_per_nucleon=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
        range_g_cm2=(0.02, 0.06, 0.2, 0.6, 2.0, 6.0, 20.0),
    )


def _hze_si_table(
    *,
    z: int = 26,
    symbol: str = "Fe",
    mass_number: float = 56.0,
) -> HzeSiLetTable:
    return HzeSiLetTable(
        z=z,
        symbol=symbol,
        mass_number=mass_number,
        mass_to_charge=mass_number / z,
        energy_mev_per_nucleon=(1.0, 10.0, 100.0),
        let_total_mev_cm2_mg=(100.0, 50.0, 20.0),
        let_electronic_mev_cm2_mg=(80.0, 40.0, 15.0),
        let_nuclear_mev_cm2_mg=(20.0, 10.0, 5.0),
    )


def _tables() -> ShieldingNormativeTables:
    return ShieldingNormativeTables(
        proton_al_range=_proton_al_table(),
        proton_si_let=_proton_si_table(),
        hze_al_ranges_by_z={
            8: _hze_al_table(z=8, symbol="O", mass_number=16.0),
            26: _hze_al_table(z=26, symbol="Fe", mass_number=56.0),
        },
        hze_si_let_by_z={
            8: _hze_si_table(z=8, symbol="O", mass_number=16.0),
            26: _hze_si_table(z=26, symbol="Fe", mass_number=56.0),
        },
        al27_nonelastic_xs=Al27NonelasticCrossSectionTable(
            mt=5,
            energy_mev=(1.0, 10.0),
            sigma_barn=(0.0, 0.0),
        ),
        secondary_proton_kernel=SecondaryProtonKernel(
            products=(),
            cross_sections_by_mt={},
        ),
    )


def _spectrum(
    *,
    particle: Particle = Particle.HZE,
    source: RadiationSource = RadiationSource.SEP,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
    model: str = "sep_hze_fe_penetrated",
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=source,
        model=model,
    )


def test_sep_hze_shielding_zero_thickness_preserves_spectrum() -> None:
    result = calculate_sep_hze_shielding_spectrum(
        z=26,
        spectrum=_spectrum(),
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert result.z == 26
    assert result.symbol == "Fe"
    assert result.spectrum.y == pytest.approx((10.0, 10.0, 10.0))
    assert result.nuclear_attenuation is False
    assert result.fragmentation is False


def test_sep_hze_shielding_positive_thickness_changes_spectrum() -> None:
    result = calculate_sep_hze_shielding_spectrum(
        z=26,
        spectrum=_spectrum(),
        tables=_tables(),
        thickness_g_cm2=0.5,
    )

    assert result.spectrum.x == (10.0, 30.0, 100.0)
    assert result.spectrum.y != pytest.approx((10.0, 10.0, 10.0))


def test_sep_hze_shielding_multiple_ions() -> None:
    results = calculate_sep_hze_shielding_spectra(
        spectra_by_z={
            8: _spectrum(model="sep_hze_o_penetrated"),
            26: _spectrum(model="sep_hze_fe_penetrated"),
        },
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert set(results) == {8, 26}
    assert results[8].symbol == "O"
    assert results[26].symbol == "Fe"


def test_sep_hze_shielding_product_preserves_product_kind_and_units() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=_spectrum(),
        label="SEP Fe mission fluence",
    )

    shielded = calculate_sep_hze_shielding_product(
        z=26,
        product=product,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert shielded.kind is RadiationProductKind.MISSION_FLUENCE
    assert shielded.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert shielded.spectrum.y_unit is Unit.DIFFERENTIAL_FLUENCE
    assert shielded.spectrum.particle is Particle.HZE
    assert "Z=26 Fe" in shielded.label
    assert "behind Al shield 0" in shielded.label


def test_sep_hze_shielding_product_supports_peak_flux_product() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.PEAK_FLUX,
        spectrum=_spectrum(
            quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        label="SEP Fe peak flux",
    )

    shielded = calculate_sep_hze_shielding_product(
        z=26,
        product=product,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert shielded.kind is RadiationProductKind.PEAK_FLUX
    assert shielded.spectrum.quantity is SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX
    assert shielded.spectrum.y_unit is Unit.DIFFERENTIAL_FLUX


def test_sep_hze_shielding_model_records_method_flags() -> None:
    result = calculate_sep_hze_shielding_spectrum(
        z=26,
        spectrum=_spectrum(),
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert SEP_HZE_AL_SHIELDING_MODEL in result.spectrum.model
    assert SEP_HZE_AL_CENTERED_SPHERICAL_SHELL in result.spectrum.model
    assert SEP_HZE_AL_NUCLEAR_ATTENUATION_NOT_INCLUDED in result.spectrum.model
    assert SEP_HZE_AL_FRAGMENTATION_NOT_INCLUDED in result.spectrum.model
    assert SEP_HZE_AL_NO_HIDDEN_4PI in result.spectrum.model


def test_sep_hze_shielding_rejects_non_hze_spectrum() -> None:
    with pytest.raises(ValueError, match="HZE spectrum"):
        calculate_sep_hze_shielding_spectrum(
            z=26,
            spectrum=_spectrum(particle=Particle.PROTON),
            tables=_tables(),
            thickness_g_cm2=0.0,
        )


def test_sep_hze_shielding_rejects_non_sep_spectrum() -> None:
    with pytest.raises(ValueError, match="SEP spectrum"):
        calculate_sep_hze_shielding_spectrum(
            z=26,
            spectrum=_spectrum(source=RadiationSource.GCR),
            tables=_tables(),
            thickness_g_cm2=0.0,
        )


def test_sep_hze_shielding_rejects_missing_z_table() -> None:
    with pytest.raises(ValueError, match="Z=6"):
        calculate_sep_hze_shielding_spectrum(
            z=6,
            spectrum=_spectrum(),
            tables=_tables(),
            thickness_g_cm2=0.0,
        )


def test_sep_hze_shielding_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        calculate_sep_hze_shielding_spectrum(
            z=26,
            spectrum=_spectrum(),
            tables=_tables(),
            thickness_g_cm2=-1.0,
        )
