from radar.core.types import RadiationProductKind
from radar.output_tables import OutputTableKind
from radar.sep.output_tables import (
    SEP_OUTPUT_DOSE_SEE_DEFERRED_REASON,
    SEP_OUTPUT_LOCATION_BEHIND_SHIELDING,
    SEP_OUTPUT_LOCATION_ON_ORBIT,
    sep_hze_shielding_output_tables,
    sep_let_output_tables,
    sep_on_orbit_product_output_tables,
    sep_proton_shielding_output_tables,
    sep_shielding_let_output_tables,
)
from radar.sep.pipeline_shielding import (
    calculate_sep_shielding_let_products_for_thickness,
)

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
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


def _hze_al_table(*, z: int = 26, symbol: str = "Fe") -> HzeAlRangeTable:
    return HzeAlRangeTable(
        z=z,
        symbol=symbol,
        mass_number=56.0 if z == 26 else 16.0,
        mass_to_charge=(56.0 / 26.0) if z == 26 else 2.0,
        energy_mev_per_nucleon=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
        range_g_cm2=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
    )


def _hze_si_table(*, z: int = 26, symbol: str = "Fe") -> HzeSiLetTable:
    return HzeSiLetTable(
        z=z,
        symbol=symbol,
        mass_number=56.0 if z == 26 else 16.0,
        mass_to_charge=(56.0 / 26.0) if z == 26 else 2.0,
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
        source=RadiationSource.SEP,
        model=model,
    )


def _model_tag(
    *,
    particle: Particle,
    z: int,
    symbol: str,
    kind: RadiationProductKind,
) -> str:
    if particle is Particle.PROTON:
        return f"ost_sep_model:proton:{kind.value}"

    return f"ost_sep_model:hze:z{z:02d}_{symbol.lower()}:{kind.value}"


def _product(
    *,
    particle: Particle,
    kind: RadiationProductKind,
    quantity: SpectrumQuantity,
    y_unit: Unit,
    z: int = 26,
    symbol: str = "Fe",
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=_spectrum(
            particle=particle,
            model=_model_tag(
                particle=particle,
                z=z,
                symbol=symbol,
                kind=kind,
            ),
            quantity=quantity,
            y_unit=y_unit,
        ),
        label=f"SEP {symbol if particle is Particle.HZE else 'H'} {kind.value}",
    )


def _products() -> tuple[SpectrumProduct, ...]:
    return (
        _product(
            particle=Particle.PROTON,
            kind=RadiationProductKind.PEAK_FLUX,
            quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        _product(
            particle=Particle.PROTON,
            kind=RadiationProductKind.MEAN_FLUX,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        _product(
            particle=Particle.PROTON,
            kind=RadiationProductKind.MISSION_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
        ),
        _product(
            particle=Particle.HZE,
            kind=RadiationProductKind.PEAK_FLUX,
            quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        _product(
            particle=Particle.HZE,
            kind=RadiationProductKind.MEAN_FLUX,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        _product(
            particle=Particle.HZE,
            kind=RadiationProductKind.MISSION_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
        ),
    )




def _metadata(table, key: str) -> str:
    return dict(table.metadata)[key]


def test_sep_on_orbit_product_output_tables() -> None:
    tables = sep_on_orbit_product_output_tables(_products())

    assert len(tables) == 6
    assert tuple(table.kind for table in tables) == (OutputTableKind.SPECTRUM,) * 6
    assert tables[0].table_id == "sep_proton_peak_flux_on_orbit"
    assert tables[3].table_id == "sep_hze_z26_peak_flux_on_orbit"

    table = tables[0]
    assert _metadata(table, "stage") == "source_model"
    assert _metadata(table, "location") == SEP_OUTPUT_LOCATION_ON_ORBIT
    assert _metadata(table, "shielding") == "not_applied"
    assert _metadata(table, "product_kind") == RadiationProductKind.PEAK_FLUX.value
    assert _metadata(table, "dose_see_dependency") == SEP_OUTPUT_DOSE_SEE_DEFERRED_REASON

    hze_table = tables[3]
    assert _metadata(hze_table, "particle_group") == "hze"
    assert _metadata(hze_table, "z") == "26"
    assert _metadata(hze_table, "symbol") == "Fe"


def test_sep_proton_shielding_output_tables_default_total_only() -> None:
    shielding_result = calculate_sep_shielding_let_products_for_thickness(
        products=_products(),
        tables=_tables(),
        thickness_g_cm2=0.1,
    )
    proton = shielding_result.proton_by_kind[RadiationProductKind.PEAK_FLUX]

    tables = sep_proton_shielding_output_tables(proton)

    assert tuple(table.table_id for table in tables) == (
        "sep_proton_total_energy_behind_al",
    )
    table = tables[0]
    assert _metadata(table, "stage") == "shielding"
    assert _metadata(table, "location") == SEP_OUTPUT_LOCATION_BEHIND_SHIELDING
    assert _metadata(table, "shielding") == "applied"
    assert _metadata(table, "shield_material") == "Al"
    assert _metadata(table, "thickness_g_cm2") == "0.1"
    assert _metadata(table, "component") == "total"
    assert _metadata(table, "proton_nonelastic_survival") == "true"
    assert _metadata(table, "secondary_protons") == "true"


def test_sep_proton_shielding_output_tables_can_include_components() -> None:
    shielding_result = calculate_sep_shielding_let_products_for_thickness(
        products=_products(),
        tables=_tables(),
        thickness_g_cm2=0.1,
    )
    proton = shielding_result.proton_by_kind[RadiationProductKind.PEAK_FLUX]

    tables = sep_proton_shielding_output_tables(
        proton,
        include_components=True,
    )

    assert tuple(table.table_id for table in tables) == (
        "sep_proton_total_energy_behind_al",
        "sep_proton_primary_energy_behind_al",
        "sep_proton_primary_survived_energy_behind_al",
        "sep_proton_secondary_energy_behind_al",
    )


def test_sep_hze_shielding_output_tables() -> None:
    shielding_result = calculate_sep_shielding_let_products_for_thickness(
        products=_products(),
        tables=_tables(),
        thickness_g_cm2=0.1,
    )
    hze_by_z = shielding_result.hze_by_kind[RadiationProductKind.PEAK_FLUX]

    tables = sep_hze_shielding_output_tables(hze_by_z)

    assert tuple(table.table_id for table in tables) == (
        "sep_hze_z26_energy_behind_al",
    )
    table = tables[0]
    assert _metadata(table, "stage") == "shielding"
    assert _metadata(table, "component") == "hze_primary_csda"
    assert _metadata(table, "z") == "26"
    assert _metadata(table, "symbol") == "Fe"
    assert _metadata(table, "heavy_ion_nuclear_attenuation") == "not_included"
    assert _metadata(table, "heavy_ion_fragmentation") == "not_included"


def test_sep_let_output_tables() -> None:
    shielding_result = calculate_sep_shielding_let_products_for_thickness(
        products=_products(),
        tables=_tables(),
        thickness_g_cm2=0.1,
    )
    let_result = shielding_result.let_results_by_kind[RadiationProductKind.PEAK_FLUX]

    tables = sep_let_output_tables(
        result=let_result,
        thickness_g_cm2=shielding_result.thickness_g_cm2,
    )

    assert tuple(table.table_id for table in tables) == (
        "sep_proton_let_behind_al",
        "sep_hze_z26_let_behind_al",
        "sep_combined_let_behind_al",
    )
    assert _metadata(tables[0], "stage") == "let"
    assert _metadata(tables[0], "component") == "proton_let"
    assert _metadata(tables[1], "component") == "hze_let"
    assert _metadata(tables[2], "component") == "combined_let"
    assert _metadata(tables[2], "hze_z_values") == "26"
    assert _metadata(tables[2], "let_target") == "Si"


def test_sep_shielding_let_output_tables() -> None:
    shielding_result = calculate_sep_shielding_let_products_for_thickness(
        products=_products(),
        tables=_tables(),
        thickness_g_cm2=0.1,
    )
    product_kind = RadiationProductKind.PEAK_FLUX
    tables = sep_shielding_let_output_tables(
        proton=shielding_result.proton_by_kind[product_kind],
        hze_by_z=shielding_result.hze_by_kind[product_kind],
        let_products=shielding_result.let_results_by_kind[product_kind],
    )

    assert tuple(table.table_id for table in tables) == (
        "sep_proton_total_energy_behind_al",
        "sep_hze_z26_energy_behind_al",
        "sep_proton_let_behind_al",
        "sep_hze_z26_let_behind_al",
        "sep_combined_let_behind_al",
    )
