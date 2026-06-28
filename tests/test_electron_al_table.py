from pathlib import Path

import pytest

from radar.shielding.electron_al import (
    ElectronAlRangeEnergyTable,
    load_electron_al_range_energy_table,
)


def _range_table() -> ElectronAlRangeEnergyTable:
    return ElectronAlRangeEnergyTable(
        energy_mev=(1.0, 10.0, 100.0),
        range_g_cm2=(0.1, 1.0, 10.0),
        stopping_mev_cm2_g=(100.0, 10.0, 1.0),
    )


def test_electron_al_table_interpolates_in_loglog_space() -> None:
    table = _range_table()

    assert table.range_at_energy(10.0) == pytest.approx(1.0)
    assert table.energy_at_range(1.0) == pytest.approx(10.0)
    assert table.stopping_at_energy(10.0) == pytest.approx(10.0)
    assert table.range_at_energy(31.622776601683793) == pytest.approx(3.1622776601683795)


def test_electron_al_table_rejects_non_monotonic_range() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ElectronAlRangeEnergyTable(
            energy_mev=(1.0, 10.0, 100.0),
            range_g_cm2=(0.1, 0.1, 10.0),
            stopping_mev_cm2_g=(100.0, 10.0, 1.0),
        )


def test_load_electron_al_range_energy_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "electron_al_range.csv"
    csv_path.write_text(
        "energy_mev,stopping_mev_cm2_g,range_g_cm2,range_mg_cm2,source_sheet\n"
        "10.0,10.0,1.0,1000.0,e_Al\n"
        "1.0,100.0,0.1,100.0,e_Al\n"
        "100.0,1.0,10.0,10000.0,e_Al\n",
        encoding="utf-8",
    )

    table = load_electron_al_range_energy_table(csv_path)

    assert table.energy_mev == (1.0, 10.0, 100.0)
    assert table.range_at_energy(10.0) == pytest.approx(1.0)
    assert table.energy_at_range(1.0) == pytest.approx(10.0)
    assert table.stopping_at_energy(10.0) == pytest.approx(10.0)
