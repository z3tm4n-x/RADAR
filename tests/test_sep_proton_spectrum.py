import math
import csv
from pathlib import Path

import pytest

from radar.sep.proton_spectrum import (
    SEP_PROTON_EVENT_COUNTS,
    SEP_PROTON_PROBABILITIES,
    SepProtonCoefficientName,
    SepProtonSpectrumCoefficients,
    SepProtonSpectrumProduct,
    directional_flux_to_omnidirectional_flux,
    directional_flux_to_omnidirectional_flux_value,
    evaluate_sep_proton_spectrum,
    evaluate_sep_proton_spectrum_value,
    load_sep_proton_coefficient_records,
    lookup_sep_proton_coefficients_exact,
    lookup_sep_proton_coefficients_interpolated,
    proton_momentum_mev,
)

DATA_ROOT = Path("src/radar/data/normative/sep_protons")
OST_MODEL = "ost_134_1044_2007"
GOST_2025_MODEL = "gost_r_25645_165_2025"


def _ost_records():
    return load_sep_proton_coefficient_records(
        DATA_ROOT / "ost_134_1044_2007_coefficients.csv"
    )


def _gost_records():
    return load_sep_proton_coefficient_records(
        DATA_ROOT / "gost_r_25645_165_2025_coefficients.csv"
    )


@pytest.mark.parametrize(
    ("filename", "model"),
    (
        ("ost_134_1044_2007_coefficients.csv", OST_MODEL),
        ("gost_r_25645_165_2025_coefficients.csv", GOST_2025_MODEL),
    ),
)
def test_sep_proton_coefficient_tables_have_complete_grid(
    filename: str,
    model: str,
) -> None:
    records = load_sep_proton_coefficient_records(DATA_ROOT / filename)

    assert len(records) == (
        2
        * 4
        * len(SEP_PROTON_EVENT_COUNTS)
        * len(SEP_PROTON_PROBABILITIES)
    )
    assert {record.model for record in records} == {model}

    for product in SepProtonSpectrumProduct:
        for parameter in SepProtonCoefficientName:
            rows = [
                record
                for record in records
                if record.product is product and record.parameter is parameter
            ]
            assert len(rows) == len(SEP_PROTON_EVENT_COUNTS) * len(
                SEP_PROTON_PROBABILITIES
            )


def test_sep_proton_coefficient_empty_cell_is_preserved() -> None:
    records = _ost_records()

    matching = [
        record
        for record in records
        if record.product is SepProtonSpectrumProduct.FLUENCE
        and record.parameter is SepProtonCoefficientName.LOG10_C
        and record.event_count == 1
        and record.probability == pytest.approx(0.9)
    ]

    assert len(matching) == 1
    assert matching[0].value is None


def test_lookup_rejects_unavailable_table_point() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        lookup_sep_proton_coefficients_exact(
            _ost_records(),
            model=OST_MODEL,
            product=SepProtonSpectrumProduct.FLUENCE,
            event_count=1,
            probability=0.9,
        )


def test_lookup_rejects_missing_table_row(tmp_path: Path) -> None:
    source_path = DATA_ROOT / "ost_134_1044_2007_coefficients.csv"
    target_path = tmp_path / "broken_coefficients.csv"

    with source_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    rows = [
        row
        for row in rows
        if not (
            row["product"] == "fluence"
            and row["parameter"] == "gamma2"
            and row["n"] == "1"
            and row["probability"] == "0.5"
        )
    ]

    with target_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "model",
                "product",
                "parameter",
                "n",
                "probability",
                "value",
                "source_table",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Missing SEP proton coefficient row"):
        lookup_sep_proton_coefficients_exact(
            load_sep_proton_coefficient_records(target_path),
            model=OST_MODEL,
            product=SepProtonSpectrumProduct.FLUENCE,
            event_count=1,
            probability=0.5,
        )


@pytest.mark.parametrize(
    ("product", "parameter", "expected_table"),
    (
        (SepProtonSpectrumProduct.FLUENCE, SepProtonCoefficientName.LOG10_C, "B.1"),
        (
            SepProtonSpectrumProduct.FLUENCE,
            SepProtonCoefficientName.BREAK_ENERGY_MEV,
            "B.2",
        ),
        (SepProtonSpectrumProduct.FLUENCE, SepProtonCoefficientName.GAMMA1, "B.3"),
        (SepProtonSpectrumProduct.FLUENCE, SepProtonCoefficientName.GAMMA2, "B.4"),
        (SepProtonSpectrumProduct.PEAK_FLUX, SepProtonCoefficientName.LOG10_C, "B.5"),
        (
            SepProtonSpectrumProduct.PEAK_FLUX,
            SepProtonCoefficientName.BREAK_ENERGY_MEV,
            "B.6",
        ),
        (SepProtonSpectrumProduct.PEAK_FLUX, SepProtonCoefficientName.GAMMA1, "B.7"),
        (SepProtonSpectrumProduct.PEAK_FLUX, SepProtonCoefficientName.GAMMA2, "B.8"),
    ),
)
def test_ost_coefficient_source_tables_are_recorded(
    product: SepProtonSpectrumProduct,
    parameter: SepProtonCoefficientName,
    expected_table: str,
) -> None:
    records = _ost_records()

    matching = [
        record
        for record in records
        if record.product is product and record.parameter is parameter
    ]

    assert matching
    assert {record.source_table for record in matching} == {expected_table}


@pytest.mark.parametrize(
    ("product", "parameter", "expected_table"),
    (
        (SepProtonSpectrumProduct.FLUENCE, SepProtonCoefficientName.LOG10_C, "table_1"),
        (
            SepProtonSpectrumProduct.FLUENCE,
            SepProtonCoefficientName.BREAK_ENERGY_MEV,
            "table_2",
        ),
        (SepProtonSpectrumProduct.FLUENCE, SepProtonCoefficientName.GAMMA1, "table_3"),
        (SepProtonSpectrumProduct.FLUENCE, SepProtonCoefficientName.GAMMA2, "table_4"),
        (
            SepProtonSpectrumProduct.PEAK_FLUX,
            SepProtonCoefficientName.LOG10_C,
            "table_5",
        ),
        (
            SepProtonSpectrumProduct.PEAK_FLUX,
            SepProtonCoefficientName.BREAK_ENERGY_MEV,
            "table_6",
        ),
        (
            SepProtonSpectrumProduct.PEAK_FLUX,
            SepProtonCoefficientName.GAMMA1,
            "table_7",
        ),
        (
            SepProtonSpectrumProduct.PEAK_FLUX,
            SepProtonCoefficientName.GAMMA2,
            "table_8",
        ),
    ),
)
def test_gost_coefficient_source_tables_are_recorded(
    product: SepProtonSpectrumProduct,
    parameter: SepProtonCoefficientName,
    expected_table: str,
) -> None:
    records = _gost_records()

    matching = [
        record
        for record in records
        if record.product is product and record.parameter is parameter
    ]

    assert matching
    assert {record.source_table for record in matching} == {expected_table}


def test_gost_fluence_coefficients_control_point() -> None:
    coefficients = lookup_sep_proton_coefficients_exact(
        _gost_records(),
        model=GOST_2025_MODEL,
        product=SepProtonSpectrumProduct.FLUENCE,
        event_count=1,
        probability=0.5,
    )

    assert coefficients.log10_c == pytest.approx(6.041)
    assert coefficients.break_energy_mev == pytest.approx(9.715)
    assert coefficients.gamma1 == pytest.approx(1.681)
    assert coefficients.gamma2 == pytest.approx(3.218)


def test_gost_peak_flux_coefficients_control_point() -> None:
    coefficients = lookup_sep_proton_coefficients_exact(
        _gost_records(),
        model=GOST_2025_MODEL,
        product=SepProtonSpectrumProduct.PEAK_FLUX,
        event_count=1,
        probability=0.5,
    )

    assert coefficients.log10_c == pytest.approx(0.190)
    assert coefficients.break_energy_mev == pytest.approx(9.872)
    assert coefficients.gamma1 == pytest.approx(1.650)
    assert coefficients.gamma2 == pytest.approx(3.224)


def test_ost_peak_gamma2_control_point() -> None:
    coefficients = lookup_sep_proton_coefficients_exact(
        _ost_records(),
        model=OST_MODEL,
        product=SepProtonSpectrumProduct.PEAK_FLUX,
        event_count=512,
        probability=0.001,
    )

    assert coefficients.gamma2 == pytest.approx(2.300)


def test_proton_momentum_uses_rest_energy() -> None:
    assert proton_momentum_mev(10.0) == pytest.approx((10.0 * (10.0 + 1876.0)) ** 0.5)


def test_spectrum_formula_returns_c_at_break_energy() -> None:
    coefficients = SepProtonSpectrumCoefficients(
        log10_c=2.0,
        break_energy_mev=10.0,
        gamma1=1.5,
        gamma2=2.5,
    )

    assert evaluate_sep_proton_spectrum_value(10.0, coefficients) == pytest.approx(100.0)


def test_spectrum_formula_uses_energy_branch_below_break() -> None:
    coefficients = SepProtonSpectrumCoefficients(
        log10_c=2.0,
        break_energy_mev=10.0,
        gamma1=1.5,
        gamma2=2.5,
    )

    expected = 100.0 * (5.0 / 10.0) ** -1.5
    assert evaluate_sep_proton_spectrum_value(5.0, coefficients) == pytest.approx(expected)


def test_spectrum_formula_uses_momentum_branch_above_break() -> None:
    coefficients = SepProtonSpectrumCoefficients(
        log10_c=2.0,
        break_energy_mev=10.0,
        gamma1=1.5,
        gamma2=2.5,
    )

    expected = 100.0 * (proton_momentum_mev(20.0) / proton_momentum_mev(10.0)) ** -5.0
    assert evaluate_sep_proton_spectrum_value(20.0, coefficients) == pytest.approx(expected)


def test_spectrum_evaluation_rejects_non_positive_energy() -> None:
    coefficients = SepProtonSpectrumCoefficients(
        log10_c=2.0,
        break_energy_mev=10.0,
        gamma1=1.5,
        gamma2=2.5,
    )

    with pytest.raises(ValueError, match="kinetic energy"):
        evaluate_sep_proton_spectrum_value(0.0, coefficients)


def test_coefficients_reject_non_positive_break_energy() -> None:
    with pytest.raises(ValueError, match="break energy"):
        SepProtonSpectrumCoefficients(
            log10_c=2.0,
            break_energy_mev=0.0,
            gamma1=1.5,
            gamma2=2.5,
        )


def test_spectrum_grid_evaluation() -> None:
    coefficients = SepProtonSpectrumCoefficients(
        log10_c=1.0,
        break_energy_mev=10.0,
        gamma1=1.0,
        gamma2=1.0,
    )

    values = evaluate_sep_proton_spectrum((5.0, 10.0, 20.0), coefficients)

    assert len(values) == 3
    assert values[1] == pytest.approx(10.0)


def test_spectrum_grid_rejects_empty_grid() -> None:
    coefficients = SepProtonSpectrumCoefficients(
        log10_c=1.0,
        break_energy_mev=10.0,
        gamma1=1.0,
        gamma2=1.0,
    )

    with pytest.raises(ValueError, match="energy grid"):
        evaluate_sep_proton_spectrum((), coefficients)



def test_interpolated_lookup_matches_exact_lookup_at_available_table_point() -> None:
    exact = lookup_sep_proton_coefficients_exact(
        _gost_records(),
        model=GOST_2025_MODEL,
        product=SepProtonSpectrumProduct.FLUENCE,
        event_count=2,
        probability=0.5,
    )
    interpolated = lookup_sep_proton_coefficients_interpolated(
        _gost_records(),
        model=GOST_2025_MODEL,
        product=SepProtonSpectrumProduct.FLUENCE,
        event_count=2.0,
        probability=0.5,
    )

    assert interpolated == exact


def test_interpolated_lookup_uses_log2_event_count_coordinate() -> None:
    coefficients = lookup_sep_proton_coefficients_interpolated(
        _gost_records(),
        model=GOST_2025_MODEL,
        product=SepProtonSpectrumProduct.FLUENCE,
        event_count=2.0**0.5,
        probability=0.5,
    )

    assert coefficients.log10_c == pytest.approx((6.041 + 7.041) / 2.0)


def test_interpolated_lookup_uses_linear_probability_coordinate() -> None:
    coefficients = lookup_sep_proton_coefficients_interpolated(
        _gost_records(),
        model=GOST_2025_MODEL,
        product=SepProtonSpectrumProduct.FLUENCE,
        event_count=2.0,
        probability=0.625,
    )

    assert coefficients.log10_c == pytest.approx((7.041 + 6.076) / 2.0)


def test_interpolated_lookup_uses_bilinear_interpolation() -> None:
    coefficients = lookup_sep_proton_coefficients_interpolated(
        _gost_records(),
        model=GOST_2025_MODEL,
        product=SepProtonSpectrumProduct.FLUENCE,
        event_count=8.0**0.5,
        probability=0.625,
    )

    expected = (7.041 + 7.870 + 6.076 + 6.972) / 4.0

    assert coefficients.log10_c == pytest.approx(expected)


def test_interpolated_lookup_rejects_unavailable_exact_table_point() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        lookup_sep_proton_coefficients_interpolated(
            _gost_records(),
            model=GOST_2025_MODEL,
            product=SepProtonSpectrumProduct.FLUENCE,
            event_count=1.0,
            probability=0.75,
        )


def test_interpolated_lookup_rejects_interpolation_through_unavailable_point() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        lookup_sep_proton_coefficients_interpolated(
            _gost_records(),
            model=GOST_2025_MODEL,
            product=SepProtonSpectrumProduct.FLUENCE,
            event_count=2.0**0.5,
            probability=0.75,
        )


@pytest.mark.parametrize("event_count", (0.0, -1.0))
def test_interpolated_lookup_rejects_non_positive_event_count(event_count: float) -> None:
    with pytest.raises(ValueError, match="event count"):
        lookup_sep_proton_coefficients_interpolated(
            _gost_records(),
            model=GOST_2025_MODEL,
            product=SepProtonSpectrumProduct.FLUENCE,
            event_count=event_count,
            probability=0.5,
        )


@pytest.mark.parametrize("event_count", (0.5, 1024.0))
def test_interpolated_lookup_rejects_event_count_outside_table_range(
    event_count: float,
) -> None:
    with pytest.raises(ValueError, match="outside"):
        lookup_sep_proton_coefficients_interpolated(
            _gost_records(),
            model=GOST_2025_MODEL,
            product=SepProtonSpectrumProduct.FLUENCE,
            event_count=event_count,
            probability=0.5,
        )


@pytest.mark.parametrize("probability", (0.0001, 0.95))
def test_interpolated_lookup_rejects_probability_outside_table_range(
    probability: float,
) -> None:
    with pytest.raises(ValueError, match="outside"):
        lookup_sep_proton_coefficients_interpolated(
            _gost_records(),
            model=GOST_2025_MODEL,
            product=SepProtonSpectrumProduct.FLUENCE,
            event_count=2.0,
            probability=probability,
        )


def test_interpolated_lookup_works_for_ost_table() -> None:
    coefficients = lookup_sep_proton_coefficients_interpolated(
        _ost_records(),
        model=OST_MODEL,
        product=SepProtonSpectrumProduct.PEAK_FLUX,
        event_count=8.0**0.5,
        probability=0.625,
    )

    expected = (1.097 + 1.744 + 0.238 + 1.111) / 4.0

    assert coefficients.log10_c == pytest.approx(expected)



def test_directional_flux_value_is_converted_to_omnidirectional_flux() -> None:
    assert directional_flux_to_omnidirectional_flux_value(2.0) == pytest.approx(
        8.0 * math.pi
    )


def test_directional_flux_spectrum_is_converted_to_omnidirectional_flux() -> None:
    values = directional_flux_to_omnidirectional_flux((1.0, 2.0, 3.0))

    assert values == pytest.approx((4.0 * math.pi, 8.0 * math.pi, 12.0 * math.pi))


def test_directional_flux_conversion_accepts_zero() -> None:
    assert directional_flux_to_omnidirectional_flux_value(0.0) == pytest.approx(0.0)


def test_directional_flux_conversion_rejects_negative_value() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        directional_flux_to_omnidirectional_flux_value(-1.0)


@pytest.mark.parametrize("bad_value", (math.nan, math.inf, -math.inf))
def test_directional_flux_conversion_rejects_non_finite_value(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        directional_flux_to_omnidirectional_flux_value(bad_value)


def test_directional_flux_spectrum_conversion_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        directional_flux_to_omnidirectional_flux(())



def test_ost_fluence_break_energy_control_point_n1_p05() -> None:
    records = load_sep_proton_coefficient_records(
        Path("src/radar/data/normative/sep_protons/ost_134_1044_2007_coefficients.csv")
    )

    coefficients = lookup_sep_proton_coefficients_exact(
        records,
        model="ost_134_1044_2007",
        product=SepProtonSpectrumProduct.FLUENCE,
        event_count=1,
        probability=0.5,
    )

    assert coefficients.break_energy_mev == pytest.approx(9.715)


def test_sep_proton_interpolation_rejects_event_count_below_table_range() -> None:
    records = load_sep_proton_coefficient_records(
        Path("src/radar/data/normative/sep_protons/ost_134_1044_2007_coefficients.csv")
    )

    with pytest.raises(ValueError, match="tabulated range \\[1, 512\\]"):
        lookup_sep_proton_coefficients_interpolated(
            records,
            model="ost_134_1044_2007",
            product=SepProtonSpectrumProduct.FLUENCE,
            event_count=0.5,
            probability=0.5,
        )
