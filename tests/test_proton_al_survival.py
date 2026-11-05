from pathlib import Path

from math import exp

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.nuclear.tendl import TendlIncidentEnergySupport
from radar.shielding.proton_al import (
    ProtonAlRangeEnergyTable,
    shield_proton_spectrum_primary_through_al,
)
from radar.shielding.proton_al_survival import (
    AL27_MOLAR_MASS_G_MOL,
    AVOGADRO_CONSTANT_MOL_INV,
    BARN_CM2,
    DEFAULT_AL27_NONELASTIC_MT,
    PROTON_AL_NONELASTIC_SURVIVAL_MODEL,
    Al27CrossSectionDiscontinuity,
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
        mt=3,
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
        mt=3,
        energy_mev=(1.0, 10.0),
        sigma_barn=(2.0, 20.0),
    )

    assert table.sigma_at_energy(5.5) == pytest.approx(11.0)


def test_al27_cross_section_rejects_extrapolation_above_evaluated_support() -> None:
    table = Al27NonelasticCrossSectionTable(
        mt=3,
        energy_mev=(1.0, 200.0),
        sigma_barn=(2.0, 20.0),
    )

    with pytest.raises(ValueError, match=r"outside TENDL support \[0, 200\] MeV"):
        table.sigma_at_energy(201.0)


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


def test_survival_quadrature_splits_one_sided_cross_section_jump() -> None:
    stopping_table = ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 10.0, 20.0),
        range_g_cm2=(1.0, 10.0, 20.0),
        stopping_mev_cm2_g=(1.0, 1.0, 1.0),
    )
    cross_section = Al27NonelasticCrossSectionTable(
        mt=3,
        energy_mev=(1.0, 10.0, 20.0),
        sigma_barn=(0.0, 1000.0, 1000.0),
        discontinuities=(
            Al27CrossSectionDiscontinuity(
                energy_mev=10.0,
                left_sigma_barn=0.0,
                right_sigma_barn=1000.0,
            ),
        ),
    )
    attenuation_per_mev = (
        AVOGADRO_CONSTANT_MOL_INV
        / AL27_MOLAR_MASS_G_MOL
        * 1000.0
        * BARN_CM2
    )

    jump_between_grid_nodes = proton_nonelastic_survival_factor(
        output_energy_mev=1.0,
        input_energy_mev=20.0,
        stopping_table=stopping_table,
        cross_section_table=cross_section,
        integration_steps=3,
    )
    jump_on_grid_node = proton_nonelastic_survival_factor(
        output_energy_mev=1.0,
        input_energy_mev=19.0,
        stopping_table=stopping_table,
        cross_section_table=cross_section,
        integration_steps=2,
    )

    assert jump_between_grid_nodes == pytest.approx(
        exp(-attenuation_per_mev * 10.0),
        rel=1.0e-12,
    )
    assert jump_on_grid_node == pytest.approx(
        exp(-attenuation_per_mev * 9.0),
        rel=1.0e-12,
    )


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


@pytest.mark.parametrize("last_center", (100.0, 120.0))
@pytest.mark.parametrize("sigma_barn", (0.0, 3.0))
def test_survival_preserves_supported_fraction_when_bin_center_is_unsupported(
    last_center: float,
    sigma_barn: float,
) -> None:
    # R(E)=E and S(E)=1 make every valid ten-gram path lose exactly 10 MeV.
    # Thus constant sigma has an independently known common attenuation
    # exp(-N_A/A * sigma * 1e-24 * 10), including a partially supported bin.
    table = ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 10.0, 30.0, 100.0),
        range_g_cm2=(1.0, 10.0, 30.0, 100.0),
        stopping_mev_cm2_g=(1.0, 1.0, 1.0, 1.0),
    )
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x": (10.0, 30.0, last_center),
            "y": (1.0, 1.0, 1.0),
        }
    )
    cross_section = Al27NonelasticCrossSectionTable(
        mt=3,
        energy_mev=(1.0, 100.0),
        sigma_barn=(sigma_barn, sigma_barn),
    )
    primary = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=table,
        thickness_g_cm2=10.0,
    )
    survived = apply_proton_nonelastic_survival_to_primary_spectrum(
        spectrum=spectrum,
        stopping_table=table,
        cross_section_table=cross_section,
        thickness_g_cm2=10.0,
    )

    factor = exp(
        -AVOGADRO_CONSTANT_MOL_INV
        / AL27_MOLAR_MASS_G_MOL
        * sigma_barn
        * BARN_CM2
        * 10.0
    )
    assert primary.y[-1] > 0.0
    assert survived.y == pytest.approx(tuple(value * factor for value in primary.y))
    assert survived.completeness is SpectrumCompleteness.INCOMPLETE
    assert set(primary.incomplete_reasons).issubset(survived.incomplete_reasons)


def _support_limited_xs() -> Al27NonelasticCrossSectionTable:
    return Al27NonelasticCrossSectionTable(
        mt=3,
        energy_mev=(1.0, 200.0, 1000.0),
        sigma_barn=(1.0, 1.0, 1.0),
        incident_energy_support=TendlIncidentEnergySupport(
            minimum_mev=0.0,
            maximum_mev=200.0,
            dataset_ref="test TENDL",
        ),
    )


def _spectrum_with_high_energy_tail(*, high_energy_value: float) -> Spectrum1D:
    return Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x": (10.0, 100.0, 300.0),
            "y": (10.0, 10.0, high_energy_value),
        }
    )


def test_nonelastic_survival_marks_nonzero_tail_above_support_incomplete() -> None:
    survived = apply_proton_nonelastic_survival_to_primary_spectrum(
        spectrum=_spectrum_with_high_energy_tail(high_energy_value=10.0),
        stopping_table=_stopping_table(),
        cross_section_table=_support_limited_xs(),
        thickness_g_cm2=1.0,
    )

    reason = (
        "TENDL p+Al-27 nonelastic survival is incomplete: nonzero transported "
        "proton contribution exists outside TENDL incident-energy support [0, 200] MeV"
    )
    assert survived.completeness is SpectrumCompleteness.INCOMPLETE
    assert reason in survived.incomplete_reasons
    with pytest.raises(ValueError, match="incomplete spectra"):
        survived.validate_production_contract()


def test_nonelastic_survival_flags_point_sample_segment_ending_at_zero() -> None:
    survived = apply_proton_nonelastic_survival_to_primary_spectrum(
        spectrum=_spectrum_with_high_energy_tail(high_energy_value=0.0),
        stopping_table=_stopping_table(),
        cross_section_table=_support_limited_xs(),
        thickness_g_cm2=0.1,
    )

    assert any(
        "TENDL incident-energy support" in reason
        for reason in survived.incomplete_reasons
    )


def test_nonelastic_survival_does_not_flag_two_point_zero_tail_above_support() -> None:
    point_sample = _spectrum_with_high_energy_tail(high_energy_value=0.0)
    spectrum = Spectrum1D(
        **{
            **point_sample.__dict__,
            "y": (10.0, 0.0, 0.0),
        }
    )
    survived = apply_proton_nonelastic_survival_to_primary_spectrum(
        spectrum=spectrum,
        stopping_table=_stopping_table(),
        cross_section_table=_support_limited_xs(),
        thickness_g_cm2=0.1,
    )

    assert not any(
        "TENDL incident-energy support" in reason
        for reason in survived.incomplete_reasons
    )


def test_load_al27_nonelastic_cross_section_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "p_al27_mf3_xs.csv"
    csv_path.write_text(
        "mt,energy_mev,sigma_barn\n"
        "4,1.0,99.0\n"
        "3,1.0,10.0\n"
        "3,10.0,20.0\n",
        encoding="utf-8",
    )

    table = load_al27_nonelastic_cross_section_table(csv_path)

    assert DEFAULT_AL27_NONELASTIC_MT == 3
    assert table.mt == 3
    assert table.sigma_at_energy(5.5) == pytest.approx(15.0)


def test_nonelastic_loader_rejects_mt5_reaction_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "p_al27_mf3_xs.csv"
    csv_path.write_text(
        "mt,energy_mev,sigma_barn\n"
        "5,0.25,0.0\n"
        "5,30.0,0.0\n"
        "5,30.0,0.7151679\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must load TENDL MF=3/MT=3"):
        load_al27_nonelastic_cross_section_table(csv_path, mt=5)
