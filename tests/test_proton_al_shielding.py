from pathlib import Path

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.proton_al import (
    PROTON_AL_PRIMARY_CSDA_MODEL,
    ProtonAlRangeEnergyTable,
    load_proton_al_range_energy_table,
    log_bin_edges_from_centers,
    shield_proton_spectrum_primary_through_al,
)


def _range_table() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
        range_g_cm2=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
        stopping_mev_cm2_g=(100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0),
    )


def _proton_spectrum(
    *,
    y: tuple[float, ...] = (10.0, 10.0, 10.0),
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.SEP,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=y,
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=quantity,
        particle=particle,
        source=source,
        model="sep_proton_orbit_spectrum",
    )


def test_log_bin_edges_from_centers_returns_geometric_edges() -> None:
    edges = log_bin_edges_from_centers((10.0, 100.0, 1000.0))

    assert len(edges) == 4
    assert edges[1] == pytest.approx(10.0**1.5)
    assert edges[2] == pytest.approx(10.0**2.5)


def test_proton_al_table_interpolates_in_both_directions() -> None:
    table = _range_table()

    assert table.range_at_energy(20.0) == pytest.approx(2.0)
    assert table.energy_at_range(2.0) == pytest.approx(20.0)
    assert table.stopping_at_energy(20.0) == pytest.approx(50.0)


def test_primary_proton_zero_thickness_preserves_flat_spectrum() -> None:
    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=_proton_spectrum(),
        table=_range_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.x == (10.0, 30.0, 100.0)
    assert shielded.y == pytest.approx((10.0, 10.0, 10.0))
    assert shielded.x_unit is Unit.MEV
    assert shielded.y_unit is Unit.DIFFERENTIAL_FLUENCE
    assert shielded.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert shielded.particle is Particle.PROTON
    assert shielded.source is RadiationSource.SEP
    assert shielded.model == f"sep_proton_orbit_spectrum+{PROTON_AL_PRIMARY_CSDA_MODEL}"


def test_primary_proton_positive_thickness_moves_to_higher_input_energy() -> None:
    spectrum = _proton_spectrum(y=(1.0, 2.0, 3.0))

    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_range_table(),
        thickness_g_cm2=1.0,
    )

    assert shielded.y[0] > 0.0
    assert shielded.y[0] != pytest.approx(spectrum.y[0])
    assert all(value >= 0.0 for value in shielded.y)


def test_primary_proton_bins_without_available_range_are_zero() -> None:
    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=_proton_spectrum(),
        table=_range_table(),
        thickness_g_cm2=1000.0,
    )

    assert shielded.y == (0.0, 0.0, 0.0)


def test_primary_proton_accepts_erb_spectrum() -> None:
    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=_proton_spectrum(source=RadiationSource.ERB),
        table=_range_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.source is RadiationSource.ERB


def test_primary_proton_rejects_non_proton_spectrum() -> None:
    with pytest.raises(ValueError, match="proton"):
        shield_proton_spectrum_primary_through_al(
            spectrum=_proton_spectrum(particle=Particle.HZE),
            table=_range_table(),
            thickness_g_cm2=1.0,
        )


def test_primary_proton_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        shield_proton_spectrum_primary_through_al(
            spectrum=_proton_spectrum(),
            table=_range_table(),
            thickness_g_cm2=-1.0,
        )


def test_load_proton_al_range_energy_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "proton_al_range.csv"
    csv_path.write_text(
        "energy_mev,stopping_mev_cm2_g,range_g_cm2,source_sheet\n"
        "1.0,100.0,0.1,p_Al\n"
        "10.0,60.0,1.0,p_Al\n"
        "30.0,40.0,3.0,p_Al\n",
        encoding="utf-8",
    )

    table = load_proton_al_range_energy_table(csv_path)

    assert table.range_at_energy(20.0) == pytest.approx(2.0)
    assert table.energy_at_range(2.0) == pytest.approx(20.0)
    assert table.stopping_at_energy(20.0) == pytest.approx(50.0)
