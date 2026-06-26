from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.gcr.output_tables import (
    GCR_OUTPUT_DOSE_SEE_DEFERRED_REASON,
    GCR_OUTPUT_DOSE_STATUS_NOT_CALCULATED,
    GCR_OUTPUT_LOCATION_BEHIND_SHIELDING,
    GCR_OUTPUT_LOCATION_ON_ORBIT,
    GCR_OUTPUT_LOCATION_OUTSIDE_MAGNETOSPHERE,
    GCR_OUTPUT_MAIN_COMPONENT_GOST,
    GCR_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED,
    gcr_hze_shielding_output_tables,
    gcr_let_output_tables,
    gcr_on_orbit_product_output_tables,
    gcr_proton_shielding_output_tables,
    gcr_shielding_let_output_tables,
    gcr_source_spectra_output_tables,
)
from radar.gcr.shielding import (
    GcrHzeShieldingResult,
    GcrLetProductsResult,
    GcrProtonShieldingResult,
)
from radar.gcr.source_spectra import GcrIonSourceSpectra, GcrSourceSpectra
from radar.output_tables import OutputTableKind


def _energy_spectrum(
    *,
    particle: Particle,
    model: str,
    quantity: SpectrumQuantity = SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=RadiationSource.GCR,
        model=model,
    )


def _let_spectrum(
    *,
    particle: Particle,
    model: str,
    quantity: SpectrumQuantity = SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
    y_unit: Unit = Unit.DIFFERENTIAL_LET_FLUX,
) -> Spectrum1D:
    return Spectrum1D(
        x=(1.0, 10.0, 100.0),
        y=(0.1, 0.2, 0.3),
        x_unit=Unit.LET,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=RadiationSource.GCR,
        model=model,
    )


def _ion_source_spectra() -> GcrIonSourceSpectra:
    return GcrIonSourceSpectra(
        symbol="H",
        z=1,
        main=_energy_spectrum(
            particle=Particle.PROTON,
            model="gost_gcr_source_outside_magnetosphere:H:main",
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
        ),
        nek=_energy_spectrum(
            particle=Particle.PROTON,
            model="gost_gcr_source_outside_magnetosphere:H:nek",
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
        ),
        akl=_energy_spectrum(
            particle=Particle.PROTON,
            model="gost_gcr_source_outside_magnetosphere:H:akl",
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
        ),
        total=_energy_spectrum(
            particle=Particle.PROTON,
            model="gost_gcr_source_outside_magnetosphere:H:total",
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
        ),
    )


def _source_spectra() -> GcrSourceSpectra:
    return GcrSourceSpectra(
        model="gost_gcr_source_outside_magnetosphere",
        ions=(_ion_source_spectra(),),
    )


def _product() -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MEAN_FLUX,
        spectrum=_energy_spectrum(
            particle=Particle.PROTON,
            model=(
                "gost_gcr_source_outside_magnetosphere:"
                "H:total:mission_products:mean_flux"
                "+ost_134_1044_2007_geomagnetic_penetration"
            ),
        ),
        label="GCR H mean flux",
    )


def _proton_result() -> GcrProtonShieldingResult:
    primary = _energy_spectrum(
        particle=Particle.PROTON,
        model=(
            "gost_gcr_source_outside_magnetosphere:"
            "H:total:mission_products:mean_flux+penetration+primary"
        ),
    )
    primary_survived = _energy_spectrum(
        particle=Particle.PROTON,
        model=(
            "gost_gcr_source_outside_magnetosphere:"
            "H:total:mission_products:mean_flux+penetration+survived"
        ),
    )
    secondary = _energy_spectrum(
        particle=Particle.PROTON,
        model=(
            "gost_gcr_source_outside_magnetosphere:"
            "H:total:mission_products:mean_flux+penetration+secondary"
        ),
    )
    total = _energy_spectrum(
        particle=Particle.PROTON,
        model=(
            "gost_gcr_source_outside_magnetosphere:"
            "H:total:mission_products:mean_flux+penetration+shielding"
        ),
    )

    return GcrProtonShieldingResult(
        primary=primary,
        primary_survived=primary_survived,
        secondary=secondary,
        total=total,
        thickness_g_cm2=0.01,
        nonelastic_survival=True,
        secondary_protons=True,
    )


def _hze_result(*, z: int = 26, symbol: str = "Fe") -> GcrHzeShieldingResult:
    return GcrHzeShieldingResult(
        z=z,
        symbol=symbol,
        spectrum=_energy_spectrum(
            particle=Particle.HZE,
            model=(
                "gost_gcr_source_outside_magnetosphere:"
                "Fe:total:mission_products:mean_flux+penetration+shielding"
            ),
        ),
        thickness_g_cm2=0.01,
        nuclear_attenuation=False,
        fragmentation=False,
    )


def _let_products() -> GcrLetProductsResult:
    return GcrLetProductsResult(
        proton_let=_let_spectrum(
            particle=Particle.PROTON,
            model=(
                "gost_gcr_source_outside_magnetosphere:"
                "H:total:mission_products:mean_flux+penetration+shielding+proton_let"
            ),
        ),
        hze_let_by_z={
            26: _let_spectrum(
                particle=Particle.HZE,
                model=(
                    "gost_gcr_source_outside_magnetosphere:"
                    "Fe:total:mission_products:mean_flux+penetration+shielding+hze_let"
                ),
            )
        },
        combined_let=_let_spectrum(
            particle=Particle.HZE,
            model=(
                "gost_gcr_source_outside_magnetosphere:"
                "H:total:mission_products:mean_flux+penetration+shielding+combined_let"
            ),
        ),
        let_centers=(1.0, 10.0, 100.0),
    )


def test_gcr_source_output_tables_include_normative_metadata() -> None:
    tables = gcr_source_spectra_output_tables(
        _source_spectra(),
        include_components=True,
    )

    assert tuple(table.table_id for table in tables) == (
        "gcr_gost_h_main_source_outside_magnetosphere",
        "gcr_gost_h_nek_source_outside_magnetosphere",
        "gcr_gost_h_akl_source_outside_magnetosphere",
        "gcr_gost_h_total_source_outside_magnetosphere",
    )

    total = tables[-1]
    assert total.kind is OutputTableKind.SPECTRUM
    assert total.columns[0].title == "Энергия"
    assert total.columns[1].title == "Дифференциальный поток"
    assert ("location", GCR_OUTPUT_LOCATION_OUTSIDE_MAGNETOSPHERE) in total.metadata
    assert ("main_component", GCR_OUTPUT_MAIN_COMPONENT_GOST) in total.metadata
    assert ("nek", "included") in total.metadata
    assert ("akl", "included") in total.metadata
    assert ("geomagnetic_penetration", "not_applied") in total.metadata
    assert ("shielding", "not_applied") in total.metadata
    assert ("dose_status", GCR_OUTPUT_DOSE_STATUS_NOT_CALCULATED) in total.metadata
    assert (
        "single_event_effects_status",
        GCR_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED,
    ) in total.metadata
    assert ("dose_see_dependency", GCR_OUTPUT_DOSE_SEE_DEFERRED_REASON) in total.metadata


def test_gcr_on_orbit_output_tables_record_geomagnetic_penetration() -> None:
    table = gcr_on_orbit_product_output_tables((_product(),))[0]

    assert table.table_id == "gcr_h_mean_flux_on_orbit"
    assert table.kind is OutputTableKind.SPECTRUM
    assert ("location", GCR_OUTPUT_LOCATION_ON_ORBIT) in table.metadata
    assert ("stage", "geomagnetic_penetration") in table.metadata
    assert ("geomagnetic_penetration", "applied") in table.metadata
    assert ("shielding", "not_applied") in table.metadata
    assert ("product_kind", "mean_flux") in table.metadata


def test_gcr_proton_shielding_output_tables_can_include_components() -> None:
    tables = gcr_proton_shielding_output_tables(
        _proton_result(),
        include_components=True,
    )

    assert tuple(table.table_id for table in tables) == (
        "gcr_proton_total_energy_behind_al",
        "gcr_proton_primary_energy_behind_al",
        "gcr_proton_primary_survived_energy_behind_al",
        "gcr_proton_secondary_energy_behind_al",
    )

    total = tables[0]
    assert ("location", GCR_OUTPUT_LOCATION_BEHIND_SHIELDING) in total.metadata
    assert ("shielding", "applied") in total.metadata
    assert ("shield_material", "Al") in total.metadata
    assert ("thickness_g_cm2", "0.01") in total.metadata
    assert ("proton_nonelastic_survival", "true") in total.metadata
    assert ("secondary_protons", "true") in total.metadata


def test_gcr_hze_shielding_output_table_records_limitations() -> None:
    table = gcr_hze_shielding_output_tables({26: _hze_result()})[0]

    assert table.table_id == "gcr_hze_z26_energy_behind_al"
    assert ("component", "hze_primary_csda") in table.metadata
    assert ("z", "26") in table.metadata
    assert ("symbol", "Fe") in table.metadata
    assert ("heavy_ion_nuclear_attenuation", "not_included") in table.metadata
    assert ("heavy_ion_fragmentation", "not_included") in table.metadata


def test_gcr_let_output_tables_include_proton_hze_and_combined() -> None:
    tables = gcr_let_output_tables(
        result=_let_products(),
        thickness_g_cm2=0.01,
    )

    assert tuple(table.table_id for table in tables) == (
        "gcr_proton_let_behind_al",
        "gcr_hze_z26_let_behind_al",
        "gcr_combined_let_behind_al",
    )

    combined = tables[-1]
    assert combined.columns[0].title == "ЛПЭ"
    assert combined.columns[1].title == "Дифференциальный ЛПЭ-поток"
    assert ("stage", "let") in combined.metadata
    assert ("component", "combined_let") in combined.metadata
    assert ("hze_z_values", "26") in combined.metadata
    assert ("let_target", "Si") in combined.metadata
    assert ("dose_status", GCR_OUTPUT_DOSE_STATUS_NOT_CALCULATED) in combined.metadata


def test_gcr_shielding_let_output_tables_do_not_create_dose_or_see_tables() -> None:
    tables = gcr_shielding_let_output_tables(
        proton=_proton_result(),
        hze_by_z={26: _hze_result()},
        let_products=_let_products(),
        include_proton_components=True,
    )

    assert all(table.kind is OutputTableKind.SPECTRUM for table in tables)
    assert all(table.kind is not OutputTableKind.DOSE for table in tables)
    assert all(table.kind is not OutputTableKind.SINGLE_EVENT for table in tables)
    assert len(tables) == 8
