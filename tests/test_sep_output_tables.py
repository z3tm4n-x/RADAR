from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.output_tables import OutputTableKind
from radar.shielding.sep_hze import SepHzeShieldingResult
from radar.shielding.sep_let import SepLetProductsResult
from radar.shielding.sep_output_tables import (
    SEP_OUTPUT_DOSE_SEE_DEFERRED_REASON,
    SEP_OUTPUT_DOSE_STATUS_NOT_CALCULATED,
    SEP_OUTPUT_LOCATION_BEHIND_SHIELDING,
    SEP_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED,
    sep_hze_shielding_output_tables,
    sep_let_output_tables,
    sep_proton_shielding_output_tables,
    sep_shielding_let_output_tables,
)
from radar.shielding.sep_proton import SepProtonShieldingResult


def _energy_spectrum(
    *,
    particle: Particle,
    model: str,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        particle=particle,
        source=RadiationSource.SEP,
        model=model,
    )


def _let_spectrum(
    *,
    particle: Particle,
    model: str,
) -> Spectrum1D:
    return Spectrum1D(
        x=(1.0, 10.0, 100.0),
        y=(0.1, 0.2, 0.3),
        x_unit=Unit.LET,
        y_unit=Unit.DIFFERENTIAL_LET_FLUX,
        quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
        particle=particle,
        source=RadiationSource.SEP,
        model=model,
    )


def _proton_result() -> SepProtonShieldingResult:
    primary = _energy_spectrum(particle=Particle.PROTON, model="primary")
    primary_survived = _energy_spectrum(particle=Particle.PROTON, model="survived")
    secondary = _energy_spectrum(particle=Particle.PROTON, model="secondary")
    total = _energy_spectrum(particle=Particle.PROTON, model="total")

    return SepProtonShieldingResult(
        primary=primary,
        primary_survived=primary_survived,
        secondary=secondary,
        total=total,
        thickness_g_cm2=0.01,
        nonelastic_survival=True,
        secondary_protons=True,
    )


def _hze_result(*, z: int = 26, symbol: str = "Fe") -> SepHzeShieldingResult:
    return SepHzeShieldingResult(
        z=z,
        symbol=symbol,
        spectrum=_energy_spectrum(particle=Particle.HZE, model=f"hze_z{z}"),
        thickness_g_cm2=0.01,
        nuclear_attenuation=False,
        fragmentation=False,
    )


def _let_products() -> SepLetProductsResult:
    return SepLetProductsResult(
        proton_let=_let_spectrum(particle=Particle.PROTON, model="proton_let"),
        hze_let_by_z={26: _let_spectrum(particle=Particle.HZE, model="hze_let")},
        combined_let=_let_spectrum(particle=Particle.HZE, model="combined_let"),
        let_centers=(1.0, 10.0, 100.0),
    )


def test_sep_proton_shielding_output_table_metadata() -> None:
    table = sep_proton_shielding_output_tables(_proton_result())[0]

    assert table.kind is OutputTableKind.SPECTRUM
    assert table.table_id == "sep_proton_total_energy_behind_al"
    assert table.columns[0].title == "Энергия"
    assert table.columns[1].title == "Пиковый дифференциальный поток"
    assert ("component", "total") in table.metadata
    assert ("location", SEP_OUTPUT_LOCATION_BEHIND_SHIELDING) in table.metadata
    assert ("thickness_g_cm2", "0.01") in table.metadata
    assert ("dose_status", SEP_OUTPUT_DOSE_STATUS_NOT_CALCULATED) in table.metadata
    assert (
        "single_event_effects_status",
        SEP_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED,
    ) in table.metadata
    assert ("dose_see_dependency", SEP_OUTPUT_DOSE_SEE_DEFERRED_REASON) in table.metadata


def test_sep_proton_shielding_output_tables_can_include_components() -> None:
    tables = sep_proton_shielding_output_tables(
        _proton_result(),
        include_components=True,
    )

    assert tuple(table.table_id for table in tables) == (
        "sep_proton_total_energy_behind_al",
        "sep_proton_primary_energy_behind_al",
        "sep_proton_primary_survived_energy_behind_al",
        "sep_proton_secondary_energy_behind_al",
    )


def test_sep_hze_shielding_output_table_metadata() -> None:
    table = sep_hze_shielding_output_tables({26: _hze_result()})[0]

    assert table.table_id == "sep_hze_z26_energy_behind_al"
    assert ("component", "hze_primary_csda") in table.metadata
    assert ("z", "26") in table.metadata
    assert ("symbol", "Fe") in table.metadata
    assert ("heavy_ion_nuclear_attenuation", "not_included") in table.metadata
    assert ("heavy_ion_fragmentation", "not_included") in table.metadata


def test_sep_let_output_tables_include_proton_hze_and_combined() -> None:
    tables = sep_let_output_tables(
        result=_let_products(),
        thickness_g_cm2=0.01,
    )

    assert tuple(table.table_id for table in tables) == (
        "sep_proton_let_behind_al",
        "sep_hze_z26_let_behind_al",
        "sep_combined_let_behind_al",
    )

    combined = tables[-1]
    assert combined.columns[0].title == "ЛПЭ"
    assert combined.columns[1].title == "Дифференциальный ЛПЭ-поток"
    assert ("component", "combined_let") in combined.metadata
    assert ("hze_z_values", "26") in combined.metadata
    assert ("let_target", "Si") in combined.metadata


def test_sep_shielding_let_output_tables_do_not_create_dose_or_see_tables() -> None:
    tables = sep_shielding_let_output_tables(
        proton=_proton_result(),
        hze_by_z={26: _hze_result()},
        let_products=_let_products(),
        include_proton_components=True,
    )

    assert all(table.kind is OutputTableKind.SPECTRUM for table in tables)
    assert all(table.kind is not OutputTableKind.DOSE for table in tables)
    assert all(table.kind is not OutputTableKind.SINGLE_EVENT for table in tables)
    assert len(tables) == 8
