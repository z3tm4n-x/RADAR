from pathlib import Path

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.proton_si_let import (
    ProtonSiLetTable,
    load_proton_si_let_table,
    proton_energy_spectrum_to_let_spectrum,
)


def _table() -> ProtonSiLetTable:
    return ProtonSiLetTable(
        energy_mev=(1.0, 10.0, 100.0, 1000.0),
        let_total_mev_cm2_mg=(10.0, 5.0, 2.0, 1.0),
        let_electronic_mev_cm2_mg=(8.0, 4.0, 1.5, 0.7),
        let_nuclear_mev_cm2_mg=(2.0, 1.0, 0.5, 0.3),
    )


def _spectrum(
    *,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
    particle: Particle = Particle.PROTON,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 100.0, 1000.0),
        y=(10.0, 20.0, 30.0),
        x_unit=Unit.MEV,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=RadiationSource.SEP,
        model="proton_total_after_al",
    )


def test_proton_si_let_table_interpolates() -> None:
    table = _table()

    assert table.total_let_at_energy(55.0) == pytest.approx(3.5)


def test_proton_energy_spectrum_to_let_fluence_spectrum() -> None:
    let_spectrum = proton_energy_spectrum_to_let_spectrum(
        spectrum=_spectrum(),
        table=_table(),
        let_centers=(1.0, 2.0, 5.0),
    )

    assert let_spectrum.x_unit is Unit.LET
    assert let_spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUENCE
    assert let_spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
    assert let_spectrum.particle is Particle.PROTON
    assert let_spectrum.source is RadiationSource.SEP
    assert any(value > 0.0 for value in let_spectrum.y)


def test_proton_energy_spectrum_to_let_flux_spectrum() -> None:
    let_spectrum = proton_energy_spectrum_to_let_spectrum(
        spectrum=_spectrum(
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        table=_table(),
        let_centers=(1.0, 2.0, 5.0),
    )

    assert let_spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUX
    assert let_spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUX


def test_proton_let_conversion_rejects_non_protons() -> None:
    with pytest.raises(ValueError, match="proton spectrum"):
        proton_energy_spectrum_to_let_spectrum(
            spectrum=_spectrum(particle=Particle.HZE),
            table=_table(),
            let_centers=(1.0, 2.0, 5.0),
        )


def test_load_proton_si_let_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "proton_si_let.csv"
    csv_path.write_text(
        "energy_mev,let_electronic_mev_cm2_mg,let_nuclear_mev_cm2_mg,let_total_mev_cm2_mg,source_sheet\n"
        "1.0,8.0,2.0,10.0,p_Si\n"
        "10.0,4.0,1.0,5.0,p_Si\n"
        "100.0,1.5,0.5,2.0,p_Si\n",
        encoding="utf-8",
    )

    table = load_proton_si_let_table(csv_path)

    assert table.total_let_at_energy(55.0) == pytest.approx(3.5)
