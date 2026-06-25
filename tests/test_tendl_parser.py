from pathlib import Path

import pytest

from radar.nuclear.tendl import (
    _endf_float,
    convert_tendl_p_al27_to_csvs,
    parse_mf3_cross_section_points,
    parse_mf6_secondary_proton_points,
)


def _float_field(value: float) -> str:
    text = f"{value: .6E}"
    mantissa, exponent = text.split("E")
    return f"{mantissa}{int(exponent):+d}".rjust(11)


def _int_field(value: int) -> str:
    return f"{value:11d}"


def _line(
    fields: tuple[float | int, ...],
    *,
    mf: int,
    mt: int,
    ns: int,
    mat: int = 1325,
) -> str:
    encoded_fields: list[str] = []

    for value in fields:
        if isinstance(value, int):
            encoded_fields.append(_int_field(value))
        else:
            encoded_fields.append(_float_field(value))

    while len(encoded_fields) < 6:
        encoded_fields.append(" " * 11)

    return "".join(encoded_fields[:6]) + f"{mat:4d}{mf:2d}{mt:3d}{ns:5d}"


def _sample_tendl(path: Path) -> None:
    lines = [
        _line((13027.0, 26.98, 0, 0, 0, 0), mf=3, mt=5, ns=1),
        _line((0.0, 0.0, 0, 0, 1, 2), mf=3, mt=5, ns=2),
        _line((2, 2), mf=3, mt=5, ns=3),
        _line((1.0e6, 0.1, 2.0e6, 0.2), mf=3, mt=5, ns=4),
        _line((13027.0, 26.98, 0, 0, 1, 0), mf=6, mt=5, ns=1),
        _line((1001.0, 1.0, 0, 1, 1, 2), mf=6, mt=5, ns=2),
        _line((2, 2), mf=6, mt=5, ns=3),
        _line((1.0e6, 0.5, 2.0e6, 0.75), mf=6, mt=5, ns=4),
        _line((0.0, 0.0, 2, 2, 1, 2), mf=6, mt=5, ns=5),
        _line((2, 2), mf=6, mt=5, ns=6),
        _line((0.0, 1.0e6, 0, 0, 4, 2), mf=6, mt=5, ns=7),
        _line((1.0e5, 2.0e-6, 2.0e5, 3.0e-6), mf=6, mt=5, ns=8),
        _line((0.0, 2.0e6, 0, 0, 4, 2), mf=6, mt=5, ns=9),
        _line((1.0e5, 4.0e-6, 2.0e5, 5.0e-6), mf=6, mt=5, ns=10),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_endf_float_accepts_no_e_exponent() -> None:
    assert _endf_float(" 1.234560+6") == pytest.approx(1.23456e6)
    assert _endf_float("-2.500000-3") == pytest.approx(-2.5e-3)
    assert _endf_float("           ") == 0.0


def test_parse_mf3_cross_section_points(tmp_path: Path) -> None:
    tendl_path = tmp_path / "sample.endf"
    _sample_tendl(tendl_path)

    points = parse_mf3_cross_section_points(tendl_path)

    assert [(point.mt, point.energy_mev, point.sigma_barn) for point in points] == [
        (5, 1.0, 0.1),
        (5, 2.0, 0.2),
    ]


def test_parse_mf6_secondary_proton_points(tmp_path: Path) -> None:
    tendl_path = tmp_path / "sample.endf"
    _sample_tendl(tendl_path)

    yield_points, pdf_points = parse_mf6_secondary_proton_points(tendl_path)

    assert len(yield_points) == 2
    assert yield_points[0].mt == 5
    assert yield_points[0].incident_energy_mev == pytest.approx(1.0)
    assert yield_points[0].yield_value == pytest.approx(0.5)

    assert len(pdf_points) == 4
    assert pdf_points[0].incident_energy_mev == pytest.approx(1.0)
    assert pdf_points[0].emitted_energy_mev == pytest.approx(0.1)
    assert pdf_points[0].pdf_per_mev == pytest.approx(2.0)


def test_convert_tendl_p_al27_to_csvs(tmp_path: Path) -> None:
    tendl_path = tmp_path / "sample.endf"
    output_dir = tmp_path / "csv"
    _sample_tendl(tendl_path)

    output = convert_tendl_p_al27_to_csvs(
        tendl_path=tendl_path,
        output_dir=output_dir,
    )

    assert output.mf3_xs.exists()
    assert output.mf6_proton_yield.exists()
    assert output.mf6_proton_pdf.exists()

    assert output.mf3_xs.read_text(encoding="utf-8").splitlines()[0] == (
        "mt,energy_mev,sigma_barn"
    )
    assert output.mf6_proton_yield.read_text(encoding="utf-8").splitlines()[0] == (
        "mt,product_index,incident_energy_mev,yield"
    )
    assert output.mf6_proton_pdf.read_text(encoding="utf-8").splitlines()[0] == (
        "mt,product_index,incident_energy_mev,emitted_energy_mev,pdf_per_mev"
    )
