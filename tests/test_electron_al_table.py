from pathlib import Path

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.electron_al import (
    ELECTRON_AL_PRIMARY_CSDA_MODEL,
    ElectronAlRangeEnergyTable,
    integrate_spectrum_loglog,
    load_electron_al_range_energy_table,
    log_bin_edges_from_centers,
    shield_electron_spectrum_primary_through_al,
)


def _range_table() -> ElectronAlRangeEnergyTable:
    return ElectronAlRangeEnergyTable(
        energy_mev=(1.0, 10.0, 100.0),
        range_g_cm2=(0.1, 1.0, 10.0),
        stopping_mev_cm2_g=(100.0, 10.0, 1.0),
    )


def _electron_spectrum(
    *,
    y: tuple[float, ...] = (10.0, 10.0, 10.0),
    particle: Particle = Particle.ELECTRON,
    source: RadiationSource = RadiationSource.ERB,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
) -> Spectrum1D:
    return Spectrum1D(
        x=(3.1622776601683795, 10.0, 31.622776601683793),
        y=y,
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=quantity,
        particle=particle,
        source=source,
        model="erb_electron_orbit_spectrum",
    )


def test_log_bin_edges_from_centers_returns_geometric_edges() -> None:
    edges = log_bin_edges_from_centers((10.0, 100.0, 1000.0))

    assert len(edges) == 4
    assert edges[1] == pytest.approx(10.0**1.5)
    assert edges[2] == pytest.approx(10.0**2.5)


def test_electron_al_table_interpolates_in_loglog_space() -> None:
    table = _range_table()

    assert table.range_at_energy(10.0) == pytest.approx(1.0)
    assert table.energy_at_range(1.0) == pytest.approx(10.0)
    assert table.stopping_at_energy(10.0) == pytest.approx(10.0)
    assert table.range_at_energy(31.622776601683793) == pytest.approx(3.1622776601683795)


def test_integrate_spectrum_loglog_uses_power_law_segments() -> None:
    spectrum = Spectrum1D(
        x=(1.0, 10.0),
        y=(1.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.ELECTRON,
        source=RadiationSource.ERB,
        model="linear_in_loglog",
    )

    assert integrate_spectrum_loglog(spectrum, (1.0,), (10.0,)) == pytest.approx((49.5,))


def test_primary_electron_zero_thickness_preserves_spectrum() -> None:
    shielded = shield_electron_spectrum_primary_through_al(
        spectrum=_electron_spectrum(),
        table=_range_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.x == _electron_spectrum().x
    assert shielded.y == pytest.approx(_electron_spectrum().y)
    assert shielded.x_unit is Unit.MEV
    assert shielded.y_unit is Unit.DIFFERENTIAL_FLUENCE
    assert shielded.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert shielded.particle is Particle.ELECTRON
    assert shielded.source is RadiationSource.ERB
    assert shielded.model == f"erb_electron_orbit_spectrum+{ELECTRON_AL_PRIMARY_CSDA_MODEL}"


def test_primary_electron_positive_thickness_remaps_input_bins() -> None:
    spectrum = _electron_spectrum(y=(1.0, 2.0, 3.0))

    shielded = shield_electron_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_range_table(),
        thickness_g_cm2=0.1,
    )

    assert shielded.x == spectrum.x
    assert shielded.y[0] > 0.0
    assert shielded.y[0] != pytest.approx(spectrum.y[0])
    assert all(value >= 0.0 for value in shielded.y)


def test_primary_electron_bins_without_available_range_are_zero() -> None:
    shielded = shield_electron_spectrum_primary_through_al(
        spectrum=_electron_spectrum(),
        table=_range_table(),
        thickness_g_cm2=1000.0,
    )

    assert shielded.y == (0.0, 0.0, 0.0)


def test_primary_electron_rejects_non_electron_spectrum() -> None:
    with pytest.raises(ValueError, match="electron"):
        shield_electron_spectrum_primary_through_al(
            spectrum=_electron_spectrum(particle=Particle.PROTON),
            table=_range_table(),
            thickness_g_cm2=1.0,
        )


def test_primary_electron_rejects_non_erb_spectrum() -> None:
    with pytest.raises(ValueError, match="ERB"):
        shield_electron_spectrum_primary_through_al(
            spectrum=_electron_spectrum(source=RadiationSource.SEP),
            table=_range_table(),
            thickness_g_cm2=1.0,
        )


def test_primary_electron_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        shield_electron_spectrum_primary_through_al(
            spectrum=_electron_spectrum(),
            table=_range_table(),
            thickness_g_cm2=-1.0,
        )


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
