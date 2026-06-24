import math
from pathlib import Path

import pytest

from radar.sep.hze_spectrum import (
    OST_SEP_HZE_GAMMA1_MULTIPLIER,
    OST_SEP_HZE_GAMMA2_MULTIPLIER,
    OST_SEP_HZE_REFERENCE_ENERGY_MEV_PER_NUCLEON,
    OstSepHzeIonRecord,
    evaluate_ost_sep_hze_spectrum,
    evaluate_ost_sep_hze_spectrum_value,
    load_ost_sep_hze_ion_records,
    ost_sep_hze_coefficients_from_proton,
)
from radar.sep.proton_spectrum import (
    SepProtonSpectrumCoefficients,
    proton_momentum_mev,
)


def _proton_coefficients() -> SepProtonSpectrumCoefficients:
    return SepProtonSpectrumCoefficients(
        log10_c=2.0,
        break_energy_mev=10.0,
        gamma1=2.0,
        gamma2=3.0,
    )


def _helium_record() -> OstSepHzeIonRecord:
    return OstSepHzeIonRecord(
        model="ost_134_1044_2007",
        z=2,
        symbol="He",
        mass_number=4.0,
        mass_to_charge=2.0,
        relative_abundance=1.23e-2,
        source_table="B.9",
    )


def test_load_ost_sep_hze_ion_records_from_b9_resource() -> None:
    records = load_ost_sep_hze_ion_records(
        Path("src/radar/data/normative/sep_hze/ost_134_1044_2007_b9.csv")
    )
    records_by_z = {record.z: record for record in records}

    assert len(records) == 27
    assert records_by_z[2].symbol == "He"
    assert records_by_z[2].mass_to_charge == pytest.approx(2.0)
    assert records_by_z[2].relative_abundance == pytest.approx(1.23e-2)
    assert records_by_z[26].symbol == "Fe"
    assert records_by_z[26].relative_abundance == pytest.approx(2.70e-5)


def test_hze_coefficients_follow_appendix_b_relations() -> None:
    proton = _proton_coefficients()
    ion = _helium_record()

    hze = ost_sep_hze_coefficients_from_proton(
        proton_coefficients=proton,
        ion=ion,
    )

    expected_gamma1 = OST_SEP_HZE_GAMMA1_MULTIPLIER * proton.gamma1**0.57
    expected_gamma2 = OST_SEP_HZE_GAMMA2_MULTIPLIER * proton.gamma2**0.62
    expected_break_energy = proton.break_energy_mev / ion.mass_to_charge
    expected_c = (
        ion.relative_abundance
        * 10.0**proton.log10_c
        * (
            proton_momentum_mev(OST_SEP_HZE_REFERENCE_ENERGY_MEV_PER_NUCLEON)
            / proton_momentum_mev(proton.break_energy_mev)
        )
        ** (-2.0 * proton.gamma2)
        * (
            proton_momentum_mev(expected_break_energy)
            / proton_momentum_mev(OST_SEP_HZE_REFERENCE_ENERGY_MEV_PER_NUCLEON)
        )
        ** (-2.0 * expected_gamma2)
    )

    assert hze.c == pytest.approx(expected_c)
    assert hze.break_energy_mev_per_nucleon == pytest.approx(expected_break_energy)
    assert hze.gamma1 == pytest.approx(expected_gamma1)
    assert hze.gamma2 == pytest.approx(expected_gamma2)
    assert hze.ion is ion


def test_hze_gamma2_uses_corrected_multiplier() -> None:
    hze = ost_sep_hze_coefficients_from_proton(
        proton_coefficients=_proton_coefficients(),
        ion=_helium_record(),
    )

    printed_typo_value = 1.34 * _proton_coefficients().gamma2**0.62

    assert hze.gamma2 == pytest.approx(1.68 * _proton_coefficients().gamma2**0.62)
    assert hze.gamma2 != pytest.approx(printed_typo_value)


def test_evaluate_hze_spectrum_value_below_and_above_break() -> None:
    hze = ost_sep_hze_coefficients_from_proton(
        proton_coefficients=_proton_coefficients(),
        ion=_helium_record(),
    )

    below_energy = hze.break_energy_mev_per_nucleon / 2.0
    above_energy = hze.break_energy_mev_per_nucleon * 2.0

    expected_below = hze.c * (below_energy / hze.break_energy_mev_per_nucleon) ** (
        -hze.gamma1
    )
    expected_above = hze.c * (
        proton_momentum_mev(above_energy)
        / proton_momentum_mev(hze.break_energy_mev_per_nucleon)
    ) ** (-2.0 * hze.gamma2)

    assert evaluate_ost_sep_hze_spectrum_value(
        below_energy,
        hze,
    ) == pytest.approx(expected_below)
    assert evaluate_ost_sep_hze_spectrum_value(
        above_energy,
        hze,
    ) == pytest.approx(expected_above)


def test_evaluate_hze_spectrum_returns_tuple_for_grid() -> None:
    hze = ost_sep_hze_coefficients_from_proton(
        proton_coefficients=_proton_coefficients(),
        ion=_helium_record(),
    )
    values = evaluate_ost_sep_hze_spectrum((1.0, 10.0, 100.0), hze)

    assert len(values) == 3
    assert all(math.isfinite(value) for value in values)
    assert all(value >= 0.0 for value in values)


def test_zero_abundance_ion_yields_zero_spectrum() -> None:
    ion = OstSepHzeIonRecord(
        model="ost_134_1044_2007",
        z=6,
        symbol="C",
        mass_number=12.0,
        mass_to_charge=2.07,
        relative_abundance=0.0,
        source_table="B.9",
    )
    hze = ost_sep_hze_coefficients_from_proton(
        proton_coefficients=_proton_coefficients(),
        ion=ion,
    )

    assert hze.c == 0.0
    assert evaluate_ost_sep_hze_spectrum((1.0, 10.0), hze) == (0.0, 0.0)


def test_hze_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        OstSepHzeIonRecord(
            model="ost_134_1044_2007",
            z=1,
            symbol="H",
            mass_number=1.0,
            mass_to_charge=1.0,
            relative_abundance=1.0,
            source_table="B.9",
        )

    with pytest.raises(ValueError, match="positive"):
        evaluate_ost_sep_hze_spectrum_value(0.0, ost_sep_hze_coefficients_from_proton(
            proton_coefficients=_proton_coefficients(),
            ion=_helium_record(),
        ))
