from pathlib import Path

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.hze_al import (
    HZE_AL_NO_FRAGMENTATION_NOTE,
    HzeAlRangeTable,
    load_hze_al_range_tables,
    shield_hze_spectrum_primary_through_al,
)


def _table() -> HzeAlRangeTable:
    return HzeAlRangeTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
        range_g_cm2=(0.02, 0.06, 0.2, 0.6, 2.0, 6.0, 20.0),
    )


def _spectrum(
    *,
    y: tuple[float, ...] = (10.0, 10.0, 10.0),
    particle: Particle = Particle.HZE,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=y,
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=particle,
        source=RadiationSource.SEP,
        model="sep_fe_orbit_spectrum",
    )


def test_hze_al_table_interpolates_in_both_directions() -> None:
    table = _table()

    assert table.range_at_energy(20.0) == pytest.approx(0.4)
    assert table.energy_at_range(0.4) == pytest.approx(20.0)


def test_hze_zero_thickness_preserves_flat_spectrum() -> None:
    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=_spectrum(),
        table=_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.x == (10.0, 30.0, 100.0)
    assert shielded.y == pytest.approx((10.0, 10.0, 10.0))
    assert shielded.particle is Particle.HZE
    assert HZE_AL_NO_FRAGMENTATION_NOTE in shielded.model


def test_hze_positive_thickness_moves_to_higher_input_energy() -> None:
    spectrum = _spectrum(y=(1.0, 2.0, 3.0))

    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_table(),
        thickness_g_cm2=0.1,
    )

    assert shielded.y[0] > 0.0
    assert shielded.y[0] != pytest.approx(spectrum.y[0])
    assert all(value >= 0.0 for value in shielded.y)


def test_hze_bins_without_available_range_are_zero() -> None:
    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=_spectrum(),
        table=_table(),
        thickness_g_cm2=1000.0,
    )

    assert shielded.y == (0.0, 0.0, 0.0)


def test_hze_rejects_non_hze_spectrum() -> None:
    with pytest.raises(ValueError, match="HZE spectrum"):
        shield_hze_spectrum_primary_through_al(
            spectrum=_spectrum(particle=Particle.PROTON),
            table=_table(),
            thickness_g_cm2=0.1,
        )


def test_hze_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        shield_hze_spectrum_primary_through_al(
            spectrum=_spectrum(),
            table=_table(),
            thickness_g_cm2=-1.0,
        )


def test_load_hze_al_range_tables(tmp_path: Path) -> None:
    csv_path = tmp_path / "hze_al_range.csv"
    csv_path.write_text(
        "z,symbol,mass_number,mass_to_charge,energy_mev_per_nucleon,range_g_cm2,range_mm,source_sheet\n"
        "26,Fe,56,2.1538461538,10.0,0.2,0.7407,i_R_Al\n"
        "26,Fe,56,2.1538461538,30.0,0.6,2.2222,i_R_Al\n"
        "26,Fe,56,2.1538461538,100.0,2.0,7.4074,i_R_Al\n"
        "8,O,16,2.0,10.0,0.4,1.4815,i_R_Al\n"
        "8,O,16,2.0,30.0,1.2,4.4444,i_R_Al\n"
        "8,O,16,2.0,100.0,4.0,14.8148,i_R_Al\n",
        encoding="utf-8",
    )

    tables = load_hze_al_range_tables(csv_path)

    assert set(tables) == {8, 26}
    assert tables[26].symbol == "Fe"
    assert tables[26].range_at_energy(20.0) == pytest.approx(0.4)
    assert tables[8].energy_at_range(2.0) == pytest.approx(50.0)
