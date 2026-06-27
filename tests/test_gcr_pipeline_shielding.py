from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.gcr.pipeline_shielding import (
    calculate_gcr_shielding_let_products,
    calculate_gcr_shielding_let_products_for_thickness,
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
        energy_mev=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
        range_g_cm2=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
        stopping_mev_cm2_g=(100.0, 80.0, 60.0, 40.0, 30.0, 20.0),
    )


def _proton_si_table() -> ProtonSiLetTable:
    return ProtonSiLetTable(
        energy_mev=(1.0, 10.0, 30.0, 100.0, 300.0),
        let_total_mev_cm2_mg=(12.0, 8.0, 5.0, 2.0, 1.0),
        let_electronic_mev_cm2_mg=(10.0, 7.0, 4.0, 1.5, 0.8),
        let_nuclear_mev_cm2_mg=(2.0, 1.0, 1.0, 0.5, 0.2),
    )


def _hze_al_table() -> HzeAlRangeTable:
    return HzeAlRangeTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
        range_g_cm2=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
    )


def _hze_si_table() -> HzeSiLetTable:
    return HzeSiLetTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=(1.0, 10.0, 30.0, 100.0, 300.0),
        let_total_mev_cm2_mg=(120.0, 90.0, 60.0, 25.0, 12.0),
        let_electronic_mev_cm2_mg=(100.0, 75.0, 50.0, 20.0, 10.0),
        let_nuclear_mev_cm2_mg=(20.0, 15.0, 10.0, 5.0, 2.0),
    )


def _secondary_kernel() -> SecondaryProtonKernel:
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(1.0, 300.0),
        yield_value=(1.0, 1.0),
        emitted_energy_mev_by_incident=((1.0, 300.0), (1.0, 300.0)),
        pdf_per_mev_by_incident=((0.0, 0.0), (0.0, 0.0)),
    )
    xs = Al27CrossSectionTable(
        mt=5,
        energy_mev=(1.0, 300.0),
        sigma_barn=(0.0, 0.0),
    )

    return SecondaryProtonKernel(
        products=(product,),
        cross_sections_by_mt={5: xs},
    )


def _tables() -> ShieldingNormativeTables:
    return ShieldingNormativeTables(
        proton_al_range=_proton_al_table(),
        proton_si_let=_proton_si_table(),
        hze_al_ranges_by_z={26: _hze_al_table()},
        hze_si_let_by_z={26: _hze_si_table()},
        al27_nonelastic_xs=Al27NonelasticCrossSectionTable(
            mt=5,
            energy_mev=(1.0, 300.0),
            sigma_barn=(0.0, 0.0),
        ),
        secondary_proton_kernel=_secondary_kernel(),
    )


def _spectrum(
    *,
    particle: Particle,
    model: str,
    quantity: SpectrumQuantity,
    y_unit: Unit,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=RadiationSource.GCR,
        model=model,
    )


def _product(
    *,
    particle: Particle,
    symbol: str,
    kind: RadiationProductKind,
    quantity: SpectrumQuantity,
    y_unit: Unit,
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=_spectrum(
            particle=particle,
            model=(
                "gost_gcr_source_outside_magnetosphere:"
                f"{symbol}:total:mission_products:{kind.value}+penetration"
            ),
            quantity=quantity,
            y_unit=y_unit,
        ),
        label=f"GCR {symbol} {kind.value}",
    )


def _products() -> tuple[SpectrumProduct, ...]:
    return (
        _product(
            particle=Particle.PROTON,
            symbol="H",
            kind=RadiationProductKind.MEAN_FLUX,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        _product(
            particle=Particle.PROTON,
            symbol="H",
            kind=RadiationProductKind.MAXIMUM_FLUX,
            quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        _product(
            particle=Particle.PROTON,
            symbol="H",
            kind=RadiationProductKind.MISSION_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
        ),
        _product(
            particle=Particle.HZE,
            symbol="Fe",
            kind=RadiationProductKind.MEAN_FLUX,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        _product(
            particle=Particle.HZE,
            symbol="Fe",
            kind=RadiationProductKind.MAXIMUM_FLUX,
            quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        _product(
            particle=Particle.HZE,
            symbol="Fe",
            kind=RadiationProductKind.MISSION_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
        ),
    )


def test_gcr_pipeline_shielding_let_products_for_one_thickness() -> None:
    result = calculate_gcr_shielding_let_products_for_thickness(
        products=_products(),
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert len(result.shielded_products) == 6
    assert len(result.let_products) == 3
    assert len(result.products) == 9

    assert set(result.proton_by_kind) == {
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MISSION_FLUENCE,
    }
    assert set(result.hze_by_kind) == {
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MISSION_FLUENCE,
    }

    assert tuple(product.kind for product in result.let_products) == (
        RadiationProductKind.MEAN_LET_FLUX,
        RadiationProductKind.MAXIMUM_LET_FLUX,
        RadiationProductKind.MISSION_LET_FLUENCE,
    )


def test_gcr_pipeline_shielding_let_products_for_multiple_thicknesses() -> None:
    results = calculate_gcr_shielding_let_products(
        products=_products(),
        tables=_tables(),
        thicknesses_g_cm2=(0.0, 0.1),
    )

    assert tuple(result.thickness_g_cm2 for result in results) == (0.0, 0.1)
    assert all(len(result.shielded_products) == 6 for result in results)
    assert all(len(result.let_products) == 3 for result in results)


def test_gcr_pipeline_shielding_skips_hze_without_tables_but_keeps_protons() -> None:
    product = _product(
        particle=Particle.HZE,
        symbol="U",
        kind=RadiationProductKind.MEAN_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
    )

    result = calculate_gcr_shielding_let_products_for_thickness(
        products=(_products()[0], product),
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert len(result.shielded_products) == 1
    assert result.shielded_products[0].spectrum.particle is Particle.PROTON
    assert result.let_products == ()



def test_gcr_pipeline_shielding_ignores_integral_energy_products() -> None:
    result = calculate_gcr_shielding_let_products_for_thickness(
        products=(
            _product(
                particle=Particle.PROTON,
                symbol="H",
                kind=RadiationProductKind.MEAN_FLUX,
                quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
                y_unit=Unit.DIFFERENTIAL_FLUX,
            ),
            _product(
                particle=Particle.PROTON,
                symbol="H",
                kind=RadiationProductKind.MEAN_FLUX,
                quantity=SpectrumQuantity.MEAN_INTEGRAL_FLUX,
                y_unit=Unit.INTEGRAL_FLUX,
            ),
        ),
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert len(result.shielded_products) == 1
    assert result.shielded_products[0].spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert result.let_products == ()
