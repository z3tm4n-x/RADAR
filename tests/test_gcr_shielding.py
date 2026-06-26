import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.gcr.shielding import (
    GCR_HZE_FRAGMENTATION_NOT_INCLUDED,
    GCR_HZE_NUCLEAR_ATTENUATION_NOT_INCLUDED,
    GCR_NO_HIDDEN_4PI,
    GCR_PROTON_AL_SHIELDING_MODEL,
    calculate_gcr_combined_let_product,
    calculate_gcr_hze_shielding_product,
    calculate_gcr_hze_shielding_spectrum,
    calculate_gcr_proton_shielding_product,
    calculate_gcr_proton_shielding_spectrum,
    calculate_gcr_shielding_products,
    gcr_hze_charge_number_from_product,
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


def _secondary_kernel(*, sigma_barn: float = 0.0) -> SecondaryProtonKernel:
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(1.0, 300.0),
        yield_value=(1.0, 1.0),
        emitted_energy_mev_by_incident=((1.0, 300.0), (1.0, 300.0)),
        pdf_per_mev_by_incident=((1.0e-3, 1.0e-3), (1.0e-3, 1.0e-3)),
    )
    xs = Al27CrossSectionTable(
        mt=5,
        energy_mev=(1.0, 300.0),
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
            energy_mev=(1.0, 300.0),
            sigma_barn=(survival_sigma_barn, survival_sigma_barn),
        ),
        secondary_proton_kernel=_secondary_kernel(sigma_barn=secondary_sigma_barn),
    )


def _gcr_product(
    *,
    particle: Particle,
    model: str,
    kind: RadiationProductKind = RadiationProductKind.MEAN_FLUX,
    quantity: SpectrumQuantity = SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
    y: tuple[float, ...] = (10.0, 10.0, 10.0),
    source: RadiationSource = RadiationSource.GCR,
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=Spectrum1D(
            x=(10.0, 30.0, 100.0),
            y=y,
            x_unit=Unit.MEV,
            y_unit=y_unit,
            quantity=quantity,
            particle=particle,
            source=source,
            model=model,
        ),
        label="GCR test product",
    )


def _proton_product(
    *,
    kind: RadiationProductKind = RadiationProductKind.MEAN_FLUX,
    quantity: SpectrumQuantity = SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
) -> SpectrumProduct:
    return _gcr_product(
        particle=Particle.PROTON,
        model="gost_gcr_source_outside_magnetosphere:H:total:mission_products:mean_flux",
        kind=kind,
        quantity=quantity,
        y_unit=y_unit,
    )


def _hze_product(
    *,
    kind: RadiationProductKind = RadiationProductKind.MEAN_FLUX,
    quantity: SpectrumQuantity = SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
) -> SpectrumProduct:
    return _gcr_product(
        particle=Particle.HZE,
        model="gost_gcr_source_outside_magnetosphere:Fe:total:mission_products:mean_flux",
        kind=kind,
        quantity=quantity,
        y_unit=y_unit,
    )


def test_gcr_proton_shielding_zero_thickness_preserves_total_spectrum() -> None:
    result = calculate_gcr_proton_shielding_spectrum(
        spectrum=_proton_product().spectrum,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert result.primary.y == pytest.approx((10.0, 10.0, 10.0))
    assert result.primary_survived.y == pytest.approx((10.0, 10.0, 10.0))
    assert result.secondary.y == pytest.approx((0.0, 0.0, 0.0))
    assert result.total.y == pytest.approx((10.0, 10.0, 10.0))
    assert result.total.source is RadiationSource.GCR
    assert GCR_PROTON_AL_SHIELDING_MODEL in result.total.model
    assert GCR_NO_HIDDEN_4PI in result.total.model


def test_gcr_proton_shielding_product_preserves_maximum_flux_semantics() -> None:
    product = _proton_product(
        kind=RadiationProductKind.MAXIMUM_FLUX,
        quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    )

    shielded = calculate_gcr_proton_shielding_product(
        product=product,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert shielded.kind is RadiationProductKind.MAXIMUM_FLUX
    assert shielded.spectrum.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
    assert shielded.spectrum.y_unit is Unit.DIFFERENTIAL_FLUX
    assert shielded.spectrum.source is RadiationSource.GCR


def test_gcr_hze_shielding_product_uses_symbol_tag_and_records_limitations() -> None:
    product = _hze_product()

    assert gcr_hze_charge_number_from_product(product) == 26

    shielded = calculate_gcr_hze_shielding_product(
        product=product,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert shielded.kind is RadiationProductKind.MEAN_FLUX
    assert shielded.spectrum.particle is Particle.HZE
    assert shielded.spectrum.source is RadiationSource.GCR
    assert shielded.spectrum.y == pytest.approx(product.spectrum.y)
    assert "Z26_Fe" in shielded.spectrum.model
    assert GCR_HZE_NUCLEAR_ATTENUATION_NOT_INCLUDED in shielded.spectrum.model
    assert GCR_HZE_FRAGMENTATION_NOT_INCLUDED in shielded.spectrum.model


def test_gcr_shielding_products_handles_proton_and_hze_products() -> None:
    proton = _proton_product()
    hze = _hze_product()

    shielded = calculate_gcr_shielding_products(
        products=(proton, hze),
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    assert len(shielded) == 2
    assert shielded[0].kind is RadiationProductKind.MEAN_FLUX
    assert shielded[0].spectrum.particle is Particle.PROTON
    assert shielded[1].kind is RadiationProductKind.MEAN_FLUX
    assert shielded[1].spectrum.particle is Particle.HZE


def test_gcr_combined_let_product_builds_mean_let_flux() -> None:
    proton = calculate_gcr_proton_shielding_spectrum(
        spectrum=_proton_product().spectrum,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )
    hze = calculate_gcr_hze_shielding_spectrum(
        z=26,
        spectrum=_hze_product().spectrum,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    let_product = calculate_gcr_combined_let_product(
        proton=proton,
        hze_by_z={26: hze},
        tables=_tables(),
        let_centers=(1.0, 10.0, 100.0),
    )

    assert let_product.kind is RadiationProductKind.MEAN_LET_FLUX
    assert let_product.spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUX
    assert let_product.spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUX
    assert let_product.spectrum.source is RadiationSource.GCR
    assert let_product.spectrum.particle is Particle.HZE
    assert sum(let_product.spectrum.y) > 0.0


def test_gcr_combined_let_product_builds_mission_let_fluence() -> None:
    proton_product = _proton_product(
        kind=RadiationProductKind.MISSION_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
    )
    hze_product = _hze_product(
        kind=RadiationProductKind.MISSION_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
    )
    proton = calculate_gcr_proton_shielding_spectrum(
        spectrum=proton_product.spectrum,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )
    hze = calculate_gcr_hze_shielding_spectrum(
        z=26,
        spectrum=hze_product.spectrum,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    let_product = calculate_gcr_combined_let_product(
        proton=proton,
        hze_by_z={26: hze},
        tables=_tables(),
        let_centers=(1.0, 10.0, 100.0),
    )

    assert let_product.kind is RadiationProductKind.MISSION_LET_FLUENCE
    assert let_product.spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
    assert let_product.spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUENCE


def test_gcr_shielding_rejects_non_gcr_product() -> None:
    product = _gcr_product(
        particle=Particle.PROTON,
        model="sep_like_product",
        source=RadiationSource.SEP,
    )

    with pytest.raises(ValueError, match="GCR spectrum"):
        calculate_gcr_proton_shielding_product(
            product=product,
            tables=_tables(),
            thickness_g_cm2=0.0,
        )


def test_gcr_let_rejects_mismatched_energy_quantities() -> None:
    proton = calculate_gcr_proton_shielding_spectrum(
        spectrum=_proton_product().spectrum,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )
    hze = calculate_gcr_hze_shielding_spectrum(
        z=26,
        spectrum=_hze_product(
            kind=RadiationProductKind.MAXIMUM_FLUX,
            quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        ).spectrum,
        tables=_tables(),
        thickness_g_cm2=0.0,
    )

    with pytest.raises(ValueError, match="same energy quantity"):
        calculate_gcr_combined_let_product(
            proton=proton,
            hze_by_z={26: hze},
            tables=_tables(),
            let_centers=(1.0, 10.0, 100.0),
        )
