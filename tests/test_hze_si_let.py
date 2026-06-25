from pathlib import Path

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.hze_si_let import (
    HzeSiLetTable,
    hze_energy_spectrum_to_let_spectrum,
    load_hze_si_let_tables,
)


def _table() -> HzeSiLetTable:
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


def _spectrum(
    *,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
    particle: Particle = Particle.HZE,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 100.0, 1000.0),
        y=(10.0, 20.0, 30.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=RadiationSource.SEP,
        model="hze_fe_after_al",
    )


def test_hze_si_let_table_interpolates() -> None:
    table = _table()

    assert table.total_let_at_energy(55.0) == pytest.approx(35.0)


def test_hze_energy_spectrum_to_let_fluence_spectrum() -> None:
    let_spectrum = hze_energy_spectrum_to_let_spectrum(
        spectrum=_spectrum(),
        table=_table(),
        let_centers=(10.0, 20.0, 50.0),
    )

    assert let_spectrum.x_unit is Unit.LET
    assert let_spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUENCE
    assert let_spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
    assert let_spectrum.particle is Particle.HZE
    assert let_spectrum.source is RadiationSource.SEP
    assert any(value > 0.0 for value in let_spectrum.y)


def test_hze_energy_spectrum_to_let_flux_spectrum() -> None:
    let_spectrum = hze_energy_spectrum_to_let_spectrum(
        spectrum=_spectrum(
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        table=_table(),
        let_centers=(10.0, 20.0, 50.0),
    )

    assert let_spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUX
    assert let_spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUX


def test_hze_let_conversion_rejects_non_hze_particles() -> None:
    with pytest.raises(ValueError, match="HZE spectrum"):
        hze_energy_spectrum_to_let_spectrum(
            spectrum=_spectrum(particle=Particle.PROTON),
            table=_table(),
            let_centers=(10.0, 20.0, 50.0),
        )


def test_load_hze_si_let_tables(tmp_path: Path) -> None:
    csv_path = tmp_path / "hze_si_let.csv"
    csv_path.write_text(
        "z,symbol,mass_number,mass_to_charge,energy_mev_per_nucleon,"
        "let_electronic_mev_cm2_mg,let_nuclear_mev_cm2_mg,let_total_mev_cm2_mg,"
        "electronic_source_sheet,nuclear_source_sheet\n"
        "26,Fe,56,2.1538461538,10.0,40.0,10.0,50.0,i_Si_e,i_Si_n\n"
        "26,Fe,56,2.1538461538,100.0,15.0,5.0,20.0,i_Si_e,i_Si_n\n"
        "26,Fe,56,2.1538461538,1000.0,7.0,3.0,10.0,i_Si_e,i_Si_n\n"
        "8,O,16,2.0,10.0,20.0,5.0,25.0,i_Si_e,i_Si_n\n"
        "8,O,16,2.0,100.0,8.0,2.0,10.0,i_Si_e,i_Si_n\n"
        "8,O,16,2.0,1000.0,3.0,1.0,4.0,i_Si_e,i_Si_n\n",
        encoding="utf-8",
    )

    tables = load_hze_si_let_tables(csv_path)

    assert set(tables) == {8, 26}
    assert tables[26].symbol == "Fe"
    assert tables[26].total_let_at_energy(55.0) == pytest.approx(35.0)
    assert tables[8].total_let_at_energy(55.0) == pytest.approx(17.5)
