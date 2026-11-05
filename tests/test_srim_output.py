"""Strict full-range SRIM reader controls; no native program execution in tests."""

from dataclasses import FrozenInstanceError, asdict, replace
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pytest

from radar.validation.srim_output import (
    SrimConversionFactor, SrimRangeOutput, SrimRangeRow, parse_srim_range_output,
)


FIXTURE = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
FIXTURE_SHA = "cea7b04a00688166d7e51443d4be338d6193aab4a990d1da75352f34ca641592"


def _raw() -> bytes:
    return FIXTURE.read_bytes()


def _replace(old: str, new: str, count: int = -1) -> bytes:
    text = _raw().decode("cp1251")
    assert old in text
    return text.replace(old, new, count).encode("cp1251")


def _row_change(index: int, value: str, *, row_number: int = 0) -> bytes:
    lines = _raw().decode("cp1251").splitlines()
    positions = [i for i, line in enumerate(lines) if len(line.split()) == 10
                 and line.split()[1] in {"eV", "keV", "MeV", "GeV"}]
    selected = positions[row_number]
    tokens = lines[selected].split()
    tokens[index] = value
    lines[selected] = " ".join(tokens)
    return ("\n".join(lines) + "\n").encode("cp1251")


def test_native_fixture_preserves_all_fields_and_units() -> None:
    raw = _raw()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA
    result = parse_srim_range_output(raw)
    assert result.raw_sha256 == FIXTURE_SHA
    assert result.engine_label == "SRIM-2006"
    assert result.version_label == "SRIM-2012.01"
    assert result.output_name == "probe_h_al.txt"
    assert result.calculation_date == "Ноябрь 01, 2026"
    assert result.ion_name == "Hydrogen"
    assert result.ion_z == 1 and result.ion_mass_u == Decimal("1.008")
    assert result.target_z == 13 and result.target_symbol == "Al"
    assert result.density_g_cm3 == Decimal("2.6989")
    assert result.atomic_density_cm3 == Decimal("6.0237e22")
    assert result.target_atomic_percent == result.target_mass_percent == 100
    assert result.bragg_correction_percent == 0
    assert len(result.rows) == 27 and len(result.conversion_factors) == 8
    first = result.rows[0]
    assert first.energy_total_mev == 1
    assert first.electronic_mev_cm2_mg == Decimal(".1749")
    assert first.nuclear_mev_cm2_mg == Decimal(".0001256")
    assert first.stopping_total_mev_cm2_mg == Decimal(".1750256")
    assert first.projected_range_cm == Decimal(".001440")
    assert first.longitudinal_straggling_cm == Decimal(".00006823")
    assert first.lateral_straggling_cm == Decimal(".00008600")
    assert first.printed_tokens == (
        "1,00", "MeV", "1,749E-01", "1,256E-04", "14,40", "um", "6823", "A", "8600", "A",
    )
    assert first.energy_unit == "MeV" and first.range_unit == "um"
    assert result.rows[-1].energy_total_mev == 10
    assert result.rows[-1].projected_range_cm == Decimal(".062341")
    assert dict(result.metadata_printed_tokens)["density_g_cm3"] == "2,6989E+00"
    assert result.conversion_factors[0].printed_token == "2,6988E+01"
    assert result.conversion_factors[-1].unit == "L.S.S. reduced units"
    assert result.range_kind == "projected"
    assert result.energy_basis == "total_ion_kinetic_energy"
    assert not result.normative_export_allowed
    assert not result.physical_accuracy_verified
    assert not result.source_authenticity_verified
    assert asdict(result)["normative_export_allowed"] is False


def test_reproducibility_fingerprint_excludes_date_filename_and_decimal_locale() -> None:
    raw = _raw()
    original = parse_srim_range_output(raw)
    name_date = _replace("probe_h_al.txt", "another-output.txt").replace(
        "Ноябрь 01, 2026".encode("cp1251"), b"January 12, 2030",
    )
    point = b"\n".join(
        re.sub(rb"(?<=[0-9]),(?=[0-9])", b".", line) if b"(C)" not in line else line
        for line in name_date.splitlines()
    )
    result = parse_srim_range_output(point)
    assert result.raw_sha256 != original.raw_sha256
    assert result.semantic_payload() == original.semantic_payload()
    assert result.semantic_sha256 == original.semantic_sha256
    assert result.calculation_date == "January 12, 2030"
    assert json.loads(json.dumps(result.semantic_payload())) == result.semantic_payload()


@pytest.mark.parametrize("old,new", [
    ("SRIM-2006", "SRIM-2008"), ("SRIM-2012.01", "SRIM-2013.00"),
    ("1,008 amu", "1,009 amu"), ("2,6989E+00", "2,7000E+00"),
    ("6,0237E+22", "6,0240E+22"), ("0,00%", "1,00%"),
    ("2,8898E+01", "2,8899E+01"), ("4,4803E+01", "4,4804E+01"),
])
def test_physics_and_metadata_changes_are_not_erased_from_semantics(old: str, new: str) -> None:
    original = parse_srim_range_output(_raw())
    changed = parse_srim_range_output(_replace(old, new))
    assert changed.semantic_sha256 != original.semantic_sha256


@pytest.mark.parametrize("column,value", [
    (2, "1,750E-01"), (3, "1,257E-04"), (4, "14,41"), (6, "6824"), (8, "8601"),
])
def test_every_stopping_range_and_straggling_column_affects_identity(
    column: int, value: str,
) -> None:
    assert (parse_srim_range_output(_row_change(column, value)).semantic_sha256
            != parse_srim_range_output(_raw()).semantic_sha256)


@pytest.mark.parametrize("value,unit", [
    ("1000000", "eV"), ("1000", "keV"), ("1,00", "MeV"), ("0,001", "GeV"),
])
def test_exact_energy_unit_conversions(value: str, unit: str) -> None:
    changed = _row_change(0, value).replace(
        f"{value} MeV".encode(), f"{value} {unit}".encode(), 1,
    )
    result = parse_srim_range_output(changed)
    assert result.rows[0].energy_total_mev == 1
    assert result.rows[0].energy_unit == unit
    assert result.semantic_sha256 == parse_srim_range_output(_raw()).semantic_sha256


@pytest.mark.parametrize("value,unit", [
    ("144000", "A"), ("14,40", "um"), ("0,01440", "mm"),
    ("0,001440", "cm"), ("0,00001440", "m"),
])
def test_exact_distance_unit_conversions(value: str, unit: str) -> None:
    changed = _row_change(4, value).replace(
        f"{value} um".encode(), f"{value} {unit}".encode(), 1,
    )
    result = parse_srim_range_output(changed)
    assert result.rows[0].projected_range_cm == Decimal(".001440")
    assert result.rows[0].range_unit == unit
    assert result.semantic_sha256 == parse_srim_range_output(_raw()).semantic_sha256


@pytest.mark.parametrize("old,new", [
    ("Calculation using SRIM-2006", "Calculation using NIST"),
    ("SRIM version ---> SRIM-2012.01", "version unknown"),
    ("Calc. date", "Run date"),
    ("Disk File Name = probe_h_al.txt", "Disk File Name = ../elsewhere.txt"),
    ("g/cm3", "kg/m3"), ("atoms/cm3", "atoms/m3"),
    ("Target  Composition", "Target  Material"),
    ("Atom   Atom   Atomic    Mass", "Atom   Atom   Mass    Atomic"),
    ("Name   Numb   Percent   Percent", "Name   Numb   Fraction   Fraction"),
    ("Al     13", "Si     13"),
    ("Bragg Correction", "Unknown Correction"),
    ("Stopping Units =  MeV/(mg/cm2)", "Stopping Units =  MeV/(g/cm2)"),
    ("Projected", "CSDA"), ("Longitudinal   Lateral", "Lateral   Longitudinal"),
    ("Elec.      Nuclear", "Nuclear    Elec."),
    ("Multiply Stopping by", "Multiply Range by"),
    ("MeV/mm", "MeV/cm"), ("keV/(ug/cm2)", "keV/(mg/cm2)"),
    ("1,0000E+03", "1,0000E+02"),
    ("2,6988E+01", "2,6989E+01"),
    ("J.P. Biersack and J.F. Ziegler", "unknown"),
])
def test_header_column_units_and_footer_corruption_rejected(old: str, new: str) -> None:
    with pytest.raises(ValueError):
        parse_srim_range_output(_replace(old, new))


@pytest.mark.parametrize("column,value", [
    (0, "NaN"), (2, "Inf"), (3, "-1"), (4, "0"), (6, "-1"),
    (8, "1,234.5"), (2, "1,000E+999"), (0, "1,2,3"), (0, "1.01"),
    (1, "MeV/u"), (5, "nm"), (7, "angstrom"), (9, "km"), (2, "1.0"),
])
def test_bad_rows_never_silently_skipped(column: int, value: str) -> None:
    with pytest.raises(ValueError):
        parse_srim_range_output(_row_change(column, value))


@pytest.mark.parametrize("column,value", [(0, "1,00"), (0, "0,90"), (4, "14,39")])
def test_duplicate_or_decreasing_energy_and_projected_range_rejected(
    column: int, value: str,
) -> None:
    with pytest.raises(ValueError, match="strictly increase|must not decrease"):
        parse_srim_range_output(_row_change(column, value, row_number=1))


def test_equal_printed_projected_ranges_are_retained_not_inverted() -> None:
    raw = _row_change(4, "14,40", row_number=1)
    result = parse_srim_range_output(raw)
    assert result.rows[0].energy_total_mev < result.rows[1].energy_total_mev
    assert result.rows[0].projected_range_cm == result.rows[1].projected_range_cm
    assert result.rows[1].printed_tokens[4] == "14,40"
    assert not result.physical_accuracy_verified


@pytest.mark.parametrize("raw", [b"", b"X" * 1_000_001, b"\x98", b"\x00", b"\x1b", b"\x7f"],
                         ids=("empty", "oversize", "bad-cp1251", "nul", "escape", "del"))
def test_bounded_valid_encoded_input_required(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_srim_range_output(raw)


@pytest.mark.parametrize("raw", [None, "text", bytearray(b"text")])
def test_nonbytes_rejected(raw: Any) -> None:
    with pytest.raises(ValueError):
        parse_srim_range_output(raw)


@pytest.mark.parametrize("position", [0, 1, 4, 7, 11, 15, 19, -1, -2, -5])
def test_missing_full_header_or_footer_line_rejected(position: int) -> None:
    lines = [line for line in _raw().decode("cp1251").splitlines() if line.strip()]
    lines.pop(position)
    with pytest.raises(ValueError):
        parse_srim_range_output("\n".join(lines).encode("cp1251"))


def test_missing_or_extra_row_fields_rejected() -> None:
    for value in ("", "1,00 extra"):
        with pytest.raises(ValueError):
            parse_srim_range_output(_row_change(3, value))


def test_extra_table_footer_or_unknown_lines_rejected() -> None:
    for raw in (
        _raw() + b"extra\r\n",
        _replace(" Bragg Correction", " UNKNOWN LINE\r\n Bragg Correction"),
        _replace("   1,10 MeV", " unknown record\r\n   1,10 MeV"),
    ):
        with pytest.raises(ValueError):
            parse_srim_range_output(raw)


def test_repeated_ion_and_compound_target_rejected() -> None:
    with pytest.raises(ValueError, match="Repeated"):
        parse_srim_range_output(_replace("1,008 amu", "2,000 amu", count=1))
    with pytest.raises(ValueError):
        parse_srim_range_output(_replace("100,00", "50,00"))
    with pytest.raises(ValueError):
        parse_srim_range_output(_replace("     Al     13", " C 6 50,00 50,00\r\n     Al     13"))


def test_zero_nuclear_and_straggling_valid_but_not_negative() -> None:
    for column in (3, 6, 8):
        result = parse_srim_range_output(_row_change(column, "0"))
        assert len(result.rows) == 27


def test_special_energy_list_output_rejected() -> None:
    raw = b"Ion = Hydrogen, Mass = 1.008\nEnergy(keV) S(Elec) S(Nuc)\n1 2 3\n"
    with pytest.raises(ValueError):
        parse_srim_range_output(raw)


def test_decimal_context_does_not_change_readback() -> None:
    expected = parse_srim_range_output(_raw())
    with localcontext() as context:
        context.prec = 3
        found = parse_srim_range_output(_raw())
        assert found.semantic_payload() == expected.semantic_payload()
        assert found.rows[0].stopping_total_mev_cm2_mg == Decimal(".1750256")


def test_immutable_rows_and_output_and_unpromotable_flags() -> None:
    output = parse_srim_range_output(_raw())
    with pytest.raises(FrozenInstanceError):
        setattr(output, "range_kind", "csda")
    with pytest.raises(FrozenInstanceError):
        setattr(output.rows[0], "projected_range_cm", Decimal(10))
    for field in ("physical_accuracy_verified", "normative_export_allowed", "range_kind"):
        changes: dict[str, Any] = {field: True}
        with pytest.raises((ValueError, TypeError)):
            replace(output, **changes)


def test_direct_row_constructor_cannot_desynchronize_values_from_tokens() -> None:
    row = parse_srim_range_output(_raw()).rows[0]
    with pytest.raises(ValueError):
        replace(row, energy_total_mev=Decimal(2))
    with pytest.raises(ValueError):
        replace(row, printed_tokens=row.printed_tokens[:-1])
    with pytest.raises(ValueError):
        SrimConversionFactor("unknown", Decimal(1), "1")
    with pytest.raises(ValueError):
        SrimConversionFactor("MeV/mm", Decimal(2), "1")


def test_direct_output_constructor_requires_row_shape_and_single_target() -> None:
    output = parse_srim_range_output(_raw())
    for kwargs in (
        {"ion_z": True}, {"target_z": 93}, {"target_symbol": "Si"},
        {"rows": output.rows[:1]}, {"raw_sha256": "bad"},
        {"target_atomic_percent": Decimal(50)}, {"density_g_cm3": Decimal(0)},
        {"rows": list(output.rows)}, {"conversion_factors": output.conversion_factors[:-1]},
    ):
        with pytest.raises(ValueError):
            replace(output, **kwargs)


def test_public_class_names_and_types() -> None:
    output = parse_srim_range_output(_raw())
    assert isinstance(output, SrimRangeOutput)
    assert isinstance(output.rows[0], SrimRangeRow)


@pytest.mark.parametrize("field,value", [
    ("ion_z", 2), ("ion_mass_u", Decimal("1.009")),
    ("density_g_cm3", Decimal("999")), ("atomic_density_cm3", Decimal("1e22")),
    ("target_atomic_percent", Decimal(50)), ("target_mass_percent", Decimal(50)),
    ("bragg_correction_percent", Decimal(1)),
    ("engine_label", "SRIM-2008"), ("version_label", "SRIM-2013.00"),
    ("output_name", "changed.txt"), ("calculation_date", "different date"),
    ("ion_name", "Helium"), ("density_g_cm3", Decimal("sNaN")),
])
def test_header_fields_cannot_be_desynchronized_from_retained_tokens(
    field: str, value: Any,
) -> None:
    output = parse_srim_range_output(_raw())
    changes: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match="header metadata"):
        replace(output, **changes)


def test_target_z_cannot_be_relabelled_with_consistent_symbol_only() -> None:
    output = parse_srim_range_output(_raw())
    with pytest.raises(ValueError, match="header metadata"):
        replace(output, target_z=14, target_symbol="Si")


def test_changing_header_tokens_and_value_without_header_is_rejected() -> None:
    output = parse_srim_range_output(_raw())
    tokens = tuple((key, "999" if key == "density_g_cm3" else value)
                   for key, value in output.metadata_printed_tokens)
    with pytest.raises(ValueError, match="tokens differ"):
        replace(output, density_g_cm3=Decimal(999), metadata_printed_tokens=tokens)


def test_header_lines_and_metadata_tokens_must_stay_immutable_complete_and_ordered() -> None:
    output = parse_srim_range_output(_raw())
    changes: list[dict[str, Any]] = [
        {"metadata_lines": list(output.metadata_lines)},
        {"metadata_lines": output.metadata_lines[:-1]},
        {"metadata_printed_tokens": list(output.metadata_printed_tokens)},
        {"metadata_printed_tokens": output.metadata_printed_tokens[::-1]},
        {"metadata_printed_tokens": output.metadata_printed_tokens[:-1]},
    ]
    for values in changes:
        with pytest.raises(ValueError, match="header metadata"):
            replace(output, **values)


def test_forged_header_is_not_ignored_even_with_unchanged_parsed_fields() -> None:
    output = parse_srim_range_output(_raw())
    for index, value in ((1, "Calculation using SRIM-2008"), (15, "Stopping Units = bad"),
                         (7, "Density = 999 g/cm3 = 6,0237E+22 atoms/cm3"),
                         (3, "Calc. date ---> bad\x00date")):
        lines = list(output.metadata_lines)
        lines[index] = value
        with pytest.raises(ValueError):
            replace(output, metadata_lines=tuple(lines))


def test_constructor_consistency_is_explicitly_not_raw_byte_authentication() -> None:
    output = parse_srim_range_output(_raw())
    synthetic = replace(output, raw_sha256="0" * 64)
    assert synthetic.raw_sha256 != hashlib.sha256(_raw()).hexdigest()
    assert synthetic.semantic_sha256 == output.semantic_sha256
    assert not synthetic.source_authenticity_verified
    assert not synthetic.physical_accuracy_verified
    assert not synthetic.normative_export_allowed


def test_stopping_sum_retains_smallest_component_beside_largest_permitted_token() -> None:
    original = parse_srim_range_output(_raw()).rows[0]
    large, tiny = "9" * 59 + "E100", "1E-100"
    tokens = (*original.printed_tokens[:2], large, tiny, *original.printed_tokens[4:])
    row = replace(original, electronic_mev_cm2_mg=Decimal(large),
                  nuclear_mev_cm2_mg=Decimal(tiny), printed_tokens=tokens)
    with localcontext() as exact:
        exact.prec = 1000
        expected = Decimal(large) + Decimal(tiny)
        assert row.stopping_total_mev_cm2_mg - Decimal(large) == Decimal(tiny)
    with localcontext() as caller:
        caller.prec = 2
        assert row.stopping_total_mev_cm2_mg == expected
