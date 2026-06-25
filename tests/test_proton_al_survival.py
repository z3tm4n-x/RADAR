from pathlib import Path

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.proton_al import (
    ProtonAlRangeEnergyTable,
    shield_proton_spectrum_primary_through_al,
)
from radar.shielding.proton_al_survival import (
    PROTON_AL_NONELASTIC_SURVIVAL_MODEL,
    Al27NonelasticCrossSectionTable,
    apply_proton_nonelastic_survival_to_primary_spectrum,
    load_al27_nonelastic_cross_section_table,
    proton_nonelastic_survival_factor,
)


def _stopping_table() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
        range_g_cm2=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
        stopping_mev_cm2_g=(100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0),
    )


def _xs_table(*, sigma_barn: float = 1000.0) -> Al27NonelasticCrossSectionTable:
    return Al27NonelasticCrossSectionTable(
        mt=5,
        energy_mev=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
        sigma_barn=tuple(sigma_barn for _ in range(7)),
    )


def _spectrum() -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="sep_proton_orbit_spectrum",
    )


def test_al27_cross_section_table_interpolates() -> None:
    table = Al27NonelasticCrossSectionTable(
        mt=5,
        energy_mev=(1.0, 10.0),
        sigma_barn=(2.0, 20.0),
    )

    assert table.sigma_at_energy(5.5) == pytest.approx(11.0)


def test_nonelastic_survival_is_one_for_zero_path() -> None:
    factor = proton_nonelastic_survival_factor(
        output_energy_mev=10.0,
        input_energy_mev=10.0,
        stopping_table=_stopping_table(),
        cross_section_table=_xs_table(),
    )

    assert factor == 1.0


def test_nonelastic_survival_is_between_zero_and_one_for_positive_path() -> None:
    factor = proton_nonelastic_survival_factor(
        output_energy_mev=10.0,
        input_energy_mev=30.0,
        stopping_table=_stopping_table(),
        cross_section_table=_xs_table(),
    )

    assert 0.0 < factor < 1.0


def test_apply_nonelastic_survival_reduces_primary_spectrum() -> None:
    spectrum = _spectrum()
    stopping_table = _stopping_table()

    primary = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=stopping_table,
        thickness_g_cm2=1.0,
    )
    survived = apply_proton_nonelastic_survival_to_primary_spectrum(
        spectrum=spectrum,
        stopping_table=stopping_table,
        cross_section_table=_xs_table(),
        thickness_g_cm2=1.0,
    )

    assert survived.x == primary.x
    assert all(0.0 <= y <= x for x, y in zip(primary.y, survived.y, strict=True))
    assert any(y < x for x, y in zip(primary.y, survived.y, strict=True))
    assert survived.model.endswith(PROTON_AL_NONELASTIC_SURVIVAL_MODEL)


def test_apply_zero_cross_section_preserves_primary_spectrum() -> None:
    spectrum = _spectrum()
    stopping_table = _stopping_table()

    primary = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=stopping_table,
        thickness_g_cm2=1.0,
    )
    survived = apply_proton_nonelastic_survival_to_primary_spectrum(
        spectrum=spectrum,
        stopping_table=stopping_table,
        cross_section_table=_xs_table(sigma_barn=0.0),
        thickness_g_cm2=1.0,
    )

    assert survived.y == pytest.approx(primary.y)


def test_load_al27_nonelastic_cross_section_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "p_al27_mf3_xs.csv"
    csv_path.write_text(
        "mt,energy_mev,sigma_barn\n"
        "4,1.0,99.0\n"
        "5,1.0,10.0\n"
        "5,10.0,20.0\n",
        encoding="utf-8",
    )

    table = load_al27_nonelastic_cross_section_table(csv_path, mt=5)

    assert table.mt == 5
    assert table.sigma_at_energy(5.5) == pytest.approx(15.0)
