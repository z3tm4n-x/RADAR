import pytest

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
from radar.shielding.sep_hze import SepHzeShieldingResult
from radar.shielding.sep_let import (
    SEP_LET_HZE_PRIMARY_AFTER_AL,
    SEP_LET_NO_HIDDEN_4PI,
    SEP_LET_PRODUCTS_MODEL,
    SEP_LET_PROTON_TOTAL_AFTER_AL,
    SEP_LET_SILICON_TARGET,
    calculate_sep_combined_let_product,
    calculate_sep_let_products,
)
from radar.shielding.sep_proton import SepProtonShieldingResult


def _energy_spectrum(
    *,
    particle: Particle,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
    model: str,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 100.0, 1000.0),
        y=(10.0, 20.0, 30.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=RadiationSource.SEP,
        model=model,
    )


def _proton_result(
    *,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
) -> SepProtonShieldingResult:
    total = _energy_spectrum(
        particle=Particle.PROTON,
        quantity=quantity,
        y_unit=y_unit,
        model="sep_proton_total_after_al",
    )
    zero = Spectrum1D(
        x=total.x,
        y=(0.0, 0.0, 0.0),
        x_unit=total.x_unit,
        y_unit=total.y_unit,
        quantity=total.quantity,
        particle=Particle.PROTON,
        source=total.source,
        model="sep_proton_secondary_zero",
    )

    return SepProtonShieldingResult(
        primary=total,
        primary_survived=total,
        secondary=zero,
        total=total,
        thickness_g_cm2=1.0,
        nonelastic_survival=True,
        secondary_protons=False,
    )


def _hze_result(
    *,
    z: int = 26,
    symbol: str = "Fe",
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
) -> SepHzeShieldingResult:
    return SepHzeShieldingResult(
        z=z,
        symbol=symbol,
        spectrum=_energy_spectrum(
            particle=Particle.HZE,
            quantity=quantity,
            y_unit=y_unit,
            model=f"sep_hze_{symbol}_after_al",
        ),
        thickness_g_cm2=1.0,
        nuclear_attenuation=False,
        fragmentation=False,
    )


def _proton_al_table() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 10.0),
        range_g_cm2=(0.1, 1.0),
        stopping_mev_cm2_g=(10.0, 10.0),
    )


def _proton_si_table() -> ProtonSiLetTable:
    return ProtonSiLetTable(
        energy_mev=(1.0, 10.0, 100.0, 1000.0),
        let_total_mev_cm2_mg=(20.0, 10.0, 5.0, 2.0),
        let_electronic_mev_cm2_mg=(16.0, 8.0, 4.0, 1.6),
        let_nuclear_mev_cm2_mg=(4.0, 2.0, 1.0, 0.4),
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
        energy_mev_per_nucleon=(1.0, 10.0, 100.0, 1000.0),
        let_total_mev_cm2_mg=(100.0, 50.0, 20.0, 10.0),
        let_electronic_mev_cm2_mg=(80.0, 40.0, 15.0, 7.0),
        let_nuclear_mev_cm2_mg=(20.0, 10.0, 5.0, 3.0),
    )


def _tables() -> ShieldingNormativeTables:
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
    )


def test_sep_let_products_build_proton_hze_and_combined_let() -> None:
    result = calculate_sep_let_products(
        proton=_proton_result(),
        hze_by_z={26: _hze_result()},
        tables=_tables(),
        let_centers=(1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0),
    )

    assert result.proton_let.x == result.let_centers
    assert result.hze_let_by_z[26].x == result.let_centers
    assert result.combined_let.x == result.let_centers
    assert result.proton_let.x_unit is Unit.LET
    assert result.combined_let.y_unit is Unit.DIFFERENTIAL_LET_FLUENCE
    assert result.combined_let.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE

    expected = tuple(
        result.proton_let.y[index] + result.hze_let_by_z[26].y[index]
        for index in range(len(result.let_centers))
    )
    assert result.combined_let.y == pytest.approx(expected)


def test_sep_combined_let_product_for_mission_fluence() -> None:
    product = calculate_sep_combined_let_product(
        proton=_proton_result(),
        hze_by_z={26: _hze_result()},
        tables=_tables(),
        let_centers=(1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0),
    )

    assert product.kind is RadiationProductKind.MISSION_LET_FLUENCE
    assert product.spectrum.x_unit is Unit.LET
    assert product.spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUENCE
    assert product.spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
    assert product.spectrum.particle is Particle.HZE


def test_sep_combined_let_product_for_peak_flux() -> None:
    product = calculate_sep_combined_let_product(
        proton=_proton_result(
            quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        hze_by_z={
            26: _hze_result(
                quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
                y_unit=Unit.DIFFERENTIAL_FLUX,
            )
        },
        tables=_tables(),
        let_centers=(1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0),
    )

    assert product.kind is RadiationProductKind.PEAK_LET_FLUX
    assert product.spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUX
    assert product.spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUX


def test_sep_let_products_reject_quantity_mismatch() -> None:
    with pytest.raises(ValueError, match="same energy quantity"):
        calculate_sep_let_products(
            proton=_proton_result(
                quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
                y_unit=Unit.DIFFERENTIAL_FLUX,
            ),
            hze_by_z={
                26: _hze_result(
                    quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
                    y_unit=Unit.DIFFERENTIAL_FLUX,
                )
            },
            tables=_tables(),
            let_centers=(1.0, 10.0, 100.0),
        )


def test_sep_let_products_reject_missing_hze_let_table() -> None:
    with pytest.raises(ValueError, match="Z=8"):
        calculate_sep_let_products(
            proton=_proton_result(),
            hze_by_z={8: _hze_result(z=8, symbol="O")},
            tables=_tables(),
            let_centers=(1.0, 10.0, 100.0),
        )


def test_sep_let_products_model_records_method_flags() -> None:
    result = calculate_sep_let_products(
        proton=_proton_result(),
        hze_by_z={26: _hze_result()},
        tables=_tables(),
        let_centers=(1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0),
    )

    assert SEP_LET_PRODUCTS_MODEL in result.combined_let.model
    assert SEP_LET_PROTON_TOTAL_AFTER_AL in result.combined_let.model
    assert SEP_LET_HZE_PRIMARY_AFTER_AL in result.combined_let.model
    assert SEP_LET_SILICON_TARGET in result.combined_let.model
    assert SEP_LET_NO_HIDDEN_4PI in result.combined_let.model


def test_sep_let_products_reject_empty_hze_results() -> None:
    with pytest.raises(ValueError, match="at least one HZE"):
        calculate_sep_let_products(
            proton=_proton_result(),
            hze_by_z={},
            tables=_tables(),
            let_centers=(1.0, 10.0, 100.0),
        )


def test_sep_combined_let_product_accepts_explicit_product_kind() -> None:
    product = calculate_sep_combined_let_product(
        proton=_proton_result(
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        hze_by_z={
            26: _hze_result(
                quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
                y_unit=Unit.DIFFERENTIAL_FLUX,
            )
        },
        tables=_tables(),
        let_centers=(1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0),
        product_kind=RadiationProductKind.MEAN_LET_FLUX,
        label="custom LET product",
    )

    assert product.kind is RadiationProductKind.MEAN_LET_FLUX
    assert product.label == "custom LET product"
