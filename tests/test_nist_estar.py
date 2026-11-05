"""Strict ESTAR parsing with independent synthetic and native-output controls.

fixtures/nist_estar_al_1_2.txt is the unchanged output of official STAR 1.2
ESTAR.f, built from the pinned archive with GNU Fortran by the local replay.
Derived from NIST data; permitted redistribution and derivative notice:
https://physics.nist.gov/PhysRefData/Star/Text/download.html
The fixture proves parser compatibility, not authenticity or physics by itself.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
from decimal import Decimal, localcontext
import hashlib
from pathlib import Path
from typing import Any

import pytest

from radar.validation.nist_estar import (
    DEFAULT_ENERGIES_MEV, EstarAlOutput, compare_electron_al_csv,
    half_printed_unit, parse_estar_output,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures/nist_estar_al_1_2.txt"
FIXTURE_SHA = "e2806353a9eff3c1e3902dee93ac61414e0857fd22225c5941c9dc2b862063b4"
HEADER = """     Electrons in  ALUMINUM

Density, g/cm3       I, eV
2.6989E+00 166.0

KINETIC STOPPING POWER CSDA RADIATION DENSITY
ENERGY COLLISION RADIATIVE TOTAL RANGE YIELD EFFECT
DELTA
MeV MeV cm2/g MeV cm2/g MeV cm2/g g/cm2
"""
CSV_HEADER = "energy_mev,stopping_mev_cm2_g,range_g_cm2,range_mg_cm2,source_sheet"
# Independent explicit exported grid, not generated from the parser constant.
ENERGIES = tuple(Decimal(x) for x in """
.01 .0125 .015 .0175 .02 .025 .03 .035 .04 .045 .05 .055 .06 .07 .08 .09
.1 .125 .15 .175 .2 .25 .3 .35 .4 .45 .5 .55 .6 .7 .8 .9
1 1.25 1.5 1.75 2 2.5 3 3.5 4 4.5 5 5.5 6 7 8 9
10 12.5 15 17.5 20 25 30 35 40 45 50 55 60 70 80 90
100 125 150 175 200 250 300 350 400 450 500 550 600 700 800 900 1000
""".split())


def _number(value: Decimal) -> str:
    mantissa, exponent = f"{value:.3E}".split("E")
    return f"{mantissa}E{int(exponent):+03d}"


def _rows() -> list[list[str]]:
    return [[_number(e), "1.000E+00", "1.000E+00", "2.000E+00", _number(e),
             "2.000E-01", "1.000E-01"] for e in ENERGIES]


def _output(rows: list[list[str]] | None = None) -> bytes:
    rows = _rows() if rows is None else rows
    return (HEADER + "\n".join("  ".join(row) for row in rows[:40]) + "\n\f\n"
            + HEADER + "\n".join("  ".join(row) for row in rows[40:]) + "\n").encode("ascii")


def _csv(output: EstarAlOutput) -> bytes:
    return (CSV_HEADER + "\n" + "\n".join(
        f"{r.energy_mev},{r.stopping_total_mev_cm2_g},{r.range_csda_g_cm2},"
        f"{r.range_csda_g_cm2 * 1000},e_Al" for r in output.rows
    ) + "\n").encode("ascii")


def _csv_change(raw: bytes, row: int, column: int, value: str) -> bytes:
    lines = raw.decode("ascii").splitlines()
    cells = lines[row].split(",")
    cells[column] = value
    lines[row] = ",".join(cells)
    return ("\n".join(lines) + "\n").encode("ascii")


def test_real_retained_output_and_production_exact_decimal_match() -> None:
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA
    result = parse_estar_output(raw)
    assert result.raw_sha256 == FIXTURE_SHA
    assert tuple(r.energy_mev for r in result.rows) == ENERGIES == DEFAULT_ENERGIES_MEV
    first = result.rows[0]
    assert first.tokens == (
        "1.000E-02", "1.649E+01", "6.559E-03", "1.650E+01",
        "3.539E-04", "2.132E-04", "3.534E-04",
    )
    assert first.range_csda_g_cm2 == Decimal(".0003539")
    assert result.density_g_cm3 == Decimal("2.6989")
    assert result.mean_excitation_energy_ev == Decimal("166.0")
    assert result.metadata_printed_tokens == (("2.6989E+00", "166.0"),) * 2
    comparison = compare_electron_al_csv(
        result, (ROOT / "src/radar/data/normative/stopping/electron_al_range.csv").read_bytes(),
    )
    assert comparison.row_count == 81
    assert comparison.exact_primary_match and comparison.exact_range_mg_match
    assert comparison.range_mg_internal_consistent
    assert not comparison.primary_differences
    assert not comparison.range_mg_differences
    assert asdict(comparison)["exact_primary_match"] is True
    assert asdict(comparison)["exact_range_mg_match"] is True


@pytest.mark.parametrize("raw", [_output(), _output().replace(b"\n", b"\r\n"),
                                 _output().replace(b"E+", b"D+").replace(b"E-", b"D-")],
                         ids=("lf", "crlf", "D-exponents"))
def test_independent_hand_constructed_grid_legacy_whitespace_and_d_exponents(raw: bytes) -> None:
    result = parse_estar_output(raw)
    assert len(result.rows) == 81
    assert tuple(r.energy_mev for r in result.rows) == ENERGIES
    assert result.rows[40].energy_mev == Decimal("4.0")
    assert result.rows[-1].range_csda_g_cm2 == 1000
    assert not result.source_authenticity_verified
    assert not result.fortran_execution_verified
    assert not result.physical_accuracy_verified
    assert not result.normative_export_allowed
    assert "Derived from NIST" in result.derivative_notice


@pytest.mark.parametrize("old,new", [
    (b"ALUMINUM", b"SILICON"), (b"Electrons", b"Protons"),
    (b"Density, g/cm3", b"Density, kg/m3"), (b"I, eV", b"I, keV"),
    (b"2.6989E+00", b"2.6990E+00"), (b"166.0", b"166.1"),
    (b"KINETIC", b"POTENTIAL"), (b"COLLISION  RADIATIVE", b"NUCLEAR RADIATIVE"),
    (b"COLLISION RADIATIVE", b"RADIATIVE COLLISION"),
    (b"CSDA", b"PROJECTED"), (b"MeV cm2/g", b"MeV cm2/mg"),
    (b"g/cm2", b"mg/cm2"), (b"DELTA", b"DET0UR"),
])
def test_wrong_material_metadata_headers_and_units_rejected(old: bytes, new: bytes) -> None:
    raw = _output()
    if old not in raw:
        raw = FIXTURE.read_bytes()
    assert old in raw
    with pytest.raises(ValueError):
        parse_estar_output(raw.replace(old, new, 1))


@pytest.mark.parametrize("raw", [b"", b"\xff", b"\x00" + _output(), b"\x7f" + _output(),
                                 _output().replace(b"\f", b""), _output() + b"\f",
                                 b"garbage\n" + _output(), _output() + b"extra\n"],
                         ids=("empty", "nonascii", "nul", "del", "no-page", "extra-page",
                              "prefix", "suffix"))
def test_bad_structure_or_encoding_rejected(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_estar_output(raw)


@pytest.mark.parametrize("value", [None, "text", bytearray(b"x"), True, b" " * 100001],
                         ids=("none", "str", "bytearray", "bool", "too-large"))
def test_wrong_input_types_or_unbounded_size(value: Any) -> None:
    with pytest.raises(ValueError):
        parse_estar_output(value)


@pytest.mark.parametrize("token", ["NaN", "Inf", "1.000E+999", "-1.000E+00", "-0.000E+00",
                                   "1.00E+00", "1.0000E+00", "0.100E+01", "True", "1_000"])
def test_bad_numeric_tokens_rejected(token: str) -> None:
    rows = _rows()
    rows[10][1] = token
    with pytest.raises(ValueError):
        parse_estar_output(_output(rows))


@pytest.mark.parametrize("column", range(5))
def test_positive_columns_reject_zero(column: int) -> None:
    rows = _rows()
    rows[0][column] = "0.000E+00"
    with pytest.raises(ValueError):
        parse_estar_output(_output(rows))


def test_yield_and_delta_can_be_zero_but_yield_cannot_exceed_one() -> None:
    rows = _rows()
    rows[0][5:] = ["0.000E+00", "0.000E+00"]
    assert parse_estar_output(_output(rows)).rows[0].radiation_yield == 0
    rows[0][5] = "1.001E+00"
    with pytest.raises(ValueError, match="yield"):
        parse_estar_output(_output(rows))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra_column", "missing_column",
                                      "wrong_grid", "unordered", "range_plateau", "range_falling"])
def test_exact_row_grid_and_range_order(mutation: str) -> None:
    rows = _rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[-2].copy()
    elif mutation == "extra_column":
        rows[0].append("1.000E+00")
    elif mutation == "missing_column":
        rows[0].pop()
    elif mutation == "wrong_grid":
        rows[0][0] = "1.000E-03"
    elif mutation == "unordered":
        rows[10], rows[11] = rows[11], rows[10]
    elif mutation == "range_plateau":
        rows[1][4] = rows[0][4]
    else:
        rows[1][4] = "9.000E-03"
    with pytest.raises(ValueError):
        parse_estar_output(_output(rows))


@pytest.mark.parametrize("token,half", [("1.234E+01", ".005"), ("1.000D-03", ".0000005"),
                                      ("0.000E+00", ".0005")])
def test_half_printed_decimal_unit(token: str, half: str) -> None:
    assert half_printed_unit(token) == Decimal(half)


def test_rounded_stopping_cells_not_exact_printed_sum() -> None:
    rows = _rows()
    rows[0][1:4] = ["1.234E+00", "1.234E-03", "1.236E+00"]
    row = parse_estar_output(_output(rows)).rows[0]
    assert row.stopping_total_mev_cm2_g != (
        row.stopping_collision_mev_cm2_g + row.stopping_radiative_mev_cm2_g
    )
    rows[0][3] = "1.237E+00"
    with pytest.raises(ValueError, match="rounded"):
        parse_estar_output(_output(rows))


def test_decimal_context_cannot_change_acceptance() -> None:
    raw = FIXTURE.read_bytes()
    csv_bytes = (ROOT / "src/radar/data/normative/stopping/electron_al_range.csv").read_bytes()
    with localcontext() as context:
        context.prec = 2
        result = parse_estar_output(raw)
        comparison = compare_electron_al_csv(result, csv_bytes)
    assert comparison.exact_primary_match and comparison.exact_range_mg_match


@pytest.mark.parametrize("field,value", [
    ("density_g_cm3", Decimal("NaN")), ("density_g_cm3", 2.6989),
    ("mean_excitation_energy_ev", Decimal("sNaN")), ("raw_sha256", "bad"),
    ("rows", ()), ("metadata_printed_tokens", ()),
])
def test_metadata_dataclass_invariants(field: str, value: Any) -> None:
    result = parse_estar_output(_output())
    with pytest.raises(ValueError):
        replace(result, **{field: value})


def test_frozen_records_and_nonrelease_flags() -> None:
    result = parse_estar_output(_output())
    with pytest.raises(FrozenInstanceError):
        setattr(result.rows[0], "energy_mev", Decimal(100))
    with pytest.raises(FrozenInstanceError):
        setattr(result, "normative_export_allowed", True)
    with pytest.raises(TypeError):
        replace(result, source_authenticity_verified=True)
    with pytest.raises(ValueError):
        replace(result.rows[0], energy_mev=Decimal(1))
    with pytest.raises(ValueError):
        replace(result.rows[0], printed_tokens=list(result.rows[0].tokens))  # type: ignore[arg-type]


@pytest.mark.parametrize("column,value,name", [
    (0, "0.0100000000000000000000001", "energy_mev"),
    (1, "2.000000000000000000000001", "stopping_mev_cm2_g"),
    (2, "0.0100000000000000000000001", "range_g_cm2"),
])
def test_primary_decimal_differences_never_tolerated(column: int, value: str, name: str) -> None:
    output = parse_estar_output(_output())
    comparison = compare_electron_al_csv(output, _csv_change(_csv(output), 1, column, value))
    assert not comparison.exact_primary_match
    assert len(comparison.primary_differences) == 1
    difference = comparison.primary_differences[0]
    assert difference.row_number == 1 and difference.field_name == name
    assert difference.observed == Decimal(value)
    assert difference.signed_difference != 0
    assert not comparison.normative_export_allowed


def test_mg_discrepancy_does_not_contaminate_primary_comparison() -> None:
    output = parse_estar_output(_output())
    comparison = compare_electron_al_csv(output, _csv_change(_csv(output), 1, 3, "10.001"))
    assert comparison.exact_primary_match
    assert not comparison.exact_range_mg_match
    assert not comparison.range_mg_internal_consistent
    assert len(comparison.range_mg_differences) == 1
    assert not comparison.range_mg_binary64_text_only_rows


def test_binary64_auxiliary_rendering_is_flagged_not_silently_accepted() -> None:
    rows = _rows()
    rows[0][4] = "1.006E-04"
    output = parse_estar_output(_output(rows))
    legacy = str(float(output.rows[0].range_csda_g_cm2) * 1000)
    assert Decimal(legacy) != output.rows[0].range_csda_g_cm2 * 1000
    comparison = compare_electron_al_csv(output, _csv_change(_csv(output), 1, 3, legacy))
    assert comparison.exact_primary_match
    assert not comparison.exact_range_mg_match
    assert comparison.range_mg_binary64_text_only_rows == (1,)


@pytest.mark.parametrize("column,value", [(0, "NaN"), (1, "Inf"), (2, "0"), (3, "-1"),
                                         (0, "0.02"), (2, "0.02"), (4, "p_Al"),
                                         (1, " 2"), (3, "1e10000"), (3, "9" * 129)])
def test_invalid_csv_values_or_order_rejected(column: int, value: str) -> None:
    output = parse_estar_output(_output())
    with pytest.raises(ValueError):
        compare_electron_al_csv(output, _csv_change(_csv(output), 1, column, value))


@pytest.mark.parametrize("mutation", ["header", "missing_row", "duplicate_row", "blank_row",
                                     "extra_column", "bad_quote", "encoding"])
def test_bad_csv_structure_rejected(mutation: str) -> None:
    output = parse_estar_output(_output())
    raw = _csv(output)
    if mutation == "header":
        raw = raw.replace(b"range_g_cm2", b"range_kg_cm2", 1)
    elif mutation == "missing_row":
        raw = b"\n".join(raw.splitlines()[:-1]) + b"\n"
    elif mutation == "duplicate_row":
        raw += raw.splitlines()[-1] + b"\n"
    elif mutation == "blank_row":
        raw += b"\n"
    elif mutation == "extra_column":
        raw = raw.replace(b",e_Al", b",e_Al,unexpected", 1)
    elif mutation == "bad_quote":
        raw += b'"unterminated'
    else:
        raw = b"\xef\xbb\xbf" + raw
    with pytest.raises(ValueError):
        compare_electron_al_csv(output, raw)


def test_comparison_requires_parsed_output() -> None:
    with pytest.raises(ValueError):
        compare_electron_al_csv(None, b"anything")  # type: ignore[arg-type]
