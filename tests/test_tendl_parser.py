from pathlib import Path

from importlib import resources

import pytest

from radar.data.integrity import (
    NormativeResourceStatus,
    validate_normative_resource_integrity,
)
from radar.nuclear.tendl import (
    EndfInterpolationLaw,
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


def _sample_tendl(
    path: Path,
    *,
    mf3_interpolation: int = 2,
    mf6_yield_interpolation: int = 2,
    mf6_incident_interpolation: int = 2,
    mf6_outgoing_interpolation: int = 1,
) -> None:
    lines = [
        _line((13027.0, 26.98, 0, 0, 0, 0), mf=3, mt=5, ns=1),
        _line((0.0, 0.0, 0, 0, 1, 2), mf=3, mt=5, ns=2),
        _line((2, mf3_interpolation), mf=3, mt=5, ns=3),
        _line((1.0e6, 0.1, 2.0e6, 0.2), mf=3, mt=5, ns=4),
        _line((13027.0, 26.98, 0, 0, 1, 0), mf=6, mt=5, ns=1),
        _line((1001.0, 1.0, 0, 1, 1, 2), mf=6, mt=5, ns=2),
        _line((2, mf6_yield_interpolation), mf=6, mt=5, ns=3),
        _line((1.0e6, 0.5, 2.0e6, 0.75), mf=6, mt=5, ns=4),
        _line((0.0, 0.0, 2, mf6_outgoing_interpolation, 1, 2), mf=6, mt=5, ns=5),
        _line((2, mf6_incident_interpolation), mf=6, mt=5, ns=6),
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
    assert yield_points[0].interpolation_law is EndfInterpolationLaw.LINEAR_LINEAR

    assert len(pdf_points) == 4
    assert pdf_points[0].incident_energy_mev == pytest.approx(1.0)
    assert pdf_points[0].emitted_energy_mev == pytest.approx(0.1)
    assert pdf_points[0].pdf_per_mev == pytest.approx(2.0)
    assert (
        pdf_points[0].incident_energy_interpolation_law
        is EndfInterpolationLaw.LINEAR_LINEAR
    )
    assert (
        pdf_points[0].outgoing_energy_interpolation_law
        is EndfInterpolationLaw.HISTOGRAM_LEFT
    )


@pytest.mark.parametrize(
    ("mf3_int", "yield_int", "incident_int", "outgoing_int", "parser_name", "match"),
    [
        (3, 2, 2, 1, "mf3", "MF=3 MT=5 uses unsupported ENDF interpolation"),
        (2, 3, 2, 1, "mf6", "product=0 yield uses unsupported ENDF"),
        (
            2,
            2,
            3,
            1,
            "mf6",
            "product=0 incident-energy PDF uses unsupported ENDF",
        ),
        (2, 2, 2, 3, "mf6", r"only histogram-left \(LEP=1\)"),
    ],
)
def test_parser_rejects_unimplemented_endf_interpolation_flags(
    tmp_path: Path,
    mf3_int: int,
    yield_int: int,
    incident_int: int,
    outgoing_int: int,
    parser_name: str,
    match: str,
) -> None:
    tendl_path = tmp_path / "unsupported.endf"
    _sample_tendl(
        tendl_path,
        mf3_interpolation=mf3_int,
        mf6_yield_interpolation=yield_int,
        mf6_incident_interpolation=incident_int,
        mf6_outgoing_interpolation=outgoing_int,
    )

    parser = (
        parse_mf3_cross_section_points
        if parser_name == "mf3"
        else parse_mf6_secondary_proton_points
    )
    with pytest.raises(ValueError, match=match):
        parser(tendl_path)


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
        "mt,product_index,incident_energy_mev,yield,yield_interpolation"
    )
    assert output.mf6_proton_pdf.read_text(encoding="utf-8").splitlines()[0] == (
        "mt,product_index,incident_energy_mev,emitted_energy_mev,pdf_per_mev,"
        "pdf_incident_interpolation,pdf_outgoing_interpolation"
    )


def test_bundled_tendl_tables_reproduce_from_retained_official_endf(
    tmp_path: Path,
) -> None:
    package = resources.files("radar.data.normative")
    source_relative_path = "tendl/p-Al027.tendl.txt"
    source_passport = validate_normative_resource_integrity(source_relative_path)

    assert source_passport.status is NormativeResourceStatus.VERIFIED_UPSTREAM_EXACT
    assert source_passport.sha256 == (
        "f2218940baef62cf9fb72ed3ab4a0326e69f92faabb0b7ee69fed7b0bb4c41a7"
    )
    assert source_passport.source_ref == (
        "https://tendl.imperial.ac.uk/tendl_2025/proton_files/Al/Al027/lib/"
        "endf/p-Al027.tendl"
    )

    with resources.as_file(package.joinpath(source_relative_path)) as source_path:
        output = convert_tendl_p_al27_to_csvs(
            tendl_path=source_path,
            output_dir=tmp_path / "regenerated",
        )

    generated_by_relative_path = {
        "tendl/p_al27_mf3_xs.csv": output.mf3_xs,
        "tendl/p_al27_mf6_proton_yield.csv": output.mf6_proton_yield,
        "tendl/p_al27_mf6_proton_pdf.csv": output.mf6_proton_pdf,
    }
    for relative_path, generated_path in generated_by_relative_path.items():
        expected = package.joinpath(relative_path).read_text(encoding="utf-8")
        generated = generated_path.read_text(encoding="utf-8")
        assert generated.replace("\r\n", "\n") == expected.replace("\r\n", "\n")

        passport = validate_normative_resource_integrity(relative_path)
        assert passport.status is NormativeResourceStatus.VERIFIED_DERIVED
        assert passport.release_eligible
