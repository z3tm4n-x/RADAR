"""Synthetic direct-node controls; no native program execution or CSV rewriting."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from math import nextafter
from pathlib import Path
import re

import pytest

from radar.validation.srim_comparison import compare_legacy_table
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output


P_AL = "stopping/proton_al_range.csv"
P_SI = "stopping/proton_si_let.csv"
H_AL = "stopping/hze_al_range.csv"
H_SI = "stopping/hze_si_let.csv"
ION_CASES = ((2, "He", 4), (6, "C", 12), (7, "N", 14), (8, "O", 16),
             (10, "Ne", 20), (12, "Mg", 24), (14, "Si", 28), (16, "S", 32),
             (18, "Ar", 40), (20, "Ca", 40), (24, "Cr", 52), (25, "Mn", 55),
             (26, "Fe", 56), (27, "Co", 59), (28, "Ni", 58))


def native(*, z: int = 1, mass: str = "1.008", target: int = 13,
           energies: tuple[str, ...] = ("1", "2"), electronic: str = "2",
           nuclear: str = "1") -> SrimRangeOutput:
    # The public fixture supplies the full unchanged grammar, not numerical answers.
    path = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    text = "\n".join(line if "(C)" in line else re.sub(r"(?<=[0-9]),(?=[0-9])", ".", line)
                     for line in path.read_bytes().decode("cp1251").splitlines())
    text = re.sub(r"Ion = Hydrogen\s+\[1\] , Mass = 1\.008 amu",
                  f"Ion = Synthetic [{z}] , Mass = {mass} amu", text)
    text = re.sub(r"\bAl\s+13\s+100\.00\s+100\.00",
                  f"{'Al' if target == 13 else 'Si'} {target} 100.00 100.00", text)
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("----------- ")) + 1
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "-" * 59)
    rows = [f"{energy} MeV {electronic} {nuclear} {i + 1} cm 0.1 cm 0.1 cm"
            for i, energy in enumerate(energies)]
    return parse_srim_range_output(("\n".join(lines[:start] + rows + lines[end:])
                                    + "\n").encode("cp1251"))


def legacy(path: str) -> bytes:
    if path == P_AL:
        records = ["energy_mev,stopping_mev_cm2_g,range_g_cm2,range_mm,source_sheet"]
        records += [f"{i},3000,{Decimal(i) * Decimal('2.70')},{i * 10},p_Al"
                    for i in range(1, 158)]
    elif path == P_SI:
        records = ["energy_mev,let_electronic_mev_cm2_mg,let_nuclear_mev_cm2_mg,"
                   "let_total_mev_cm2_mg,electronic_column,nuclear_column,source_sheet"]
        records += [f"{i},2,0,2,H_coll,,p_Si" for i in range(1, 106)]
    else:
        prefix = "z,symbol,mass_number,mass_to_charge,energy_mev_per_nucleon,"
        records = [prefix + ("range_g_cm2,range_mm,source_sheet" if path == H_AL else
                             "let_electronic_mev_cm2_mg,let_nuclear_mev_cm2_mg,"
                             "let_total_mev_cm2_mg,electronic_source_sheet,nuclear_source_sheet")]
        for z, symbol, mass in ION_CASES:
            for energy in range(1, 124):
                row = f"{z},{symbol},{mass},{mass / z},{energy / mass},"
                row += (f"{Decimal(energy) * Decimal('2.70')},{energy * 10},i_R_Al"
                        if path == H_AL else "2,1,3,i_Si_e,i_Si_n")
                records.append(row)
    return ("\n".join(records) + "\n").encode("ascii")


def change(raw: bytes, *, row: int = 1, column: int = 0, value: str) -> bytes:
    lines = raw.decode("ascii").splitlines()
    values = lines[row].split(",")
    values[column] = value
    lines[row] = ",".join(values)
    return ("\n".join(lines) + "\n").encode("ascii")


def test_proton_al_exact_values_and_separate_density_assumptions() -> None:
    data = legacy(P_AL)
    result = compare_legacy_table(native(), data, P_AL)
    assert result["legacy_total_rows"] == result["legacy_selected_ion_rows"] == 157
    assert result["counts"] == {
        "matched_exact_decimal_energy": 2, "matched_binary64_roundtrip_energy": 0,
        "unmatched_below_native_grid": 0, "unmatched_above_native_grid": 155,
        "unmatched_inside_native_grid": 0,
    }
    assert len(result["rows"]) == 157
    assert result["legacy_csv_sha256"] == hashlib.sha256(data).hexdigest()
    first = result["rows"][0]
    assert first["legacy_raw_fields"]["range_mm"] == "10"
    comparisons = first["comparisons"]
    assert comparisons["total_stopping_mev_cm2_mg"]["exact_decimal_match"]
    assert comparisons["range_cm_vs_srim_projected_cm"]["exact_decimal_match"]
    assert comparisons["range_g_cm2_vs_projected_at_legacy_density"]["exact_decimal_match"]
    native_density = comparisons["range_g_cm2_vs_projected_at_native_density"]
    assert Decimal(native_density["native_minus_legacy"]) == Decimal("-.0011")
    assert not native_density["exact_decimal_match"]
    assert not result["legacy_range_kind_verified"]
    assert result["native_range_kind"] == "projected"
    assert not result["legacy_ion_mass_recorded"]
    assert result["comparison_status"] == "INCOMPLETE"
    for field in ("legacy_upstream_identity_verified", "physical_accuracy_verified",
                  "normative_export_allowed", "production_data_changed"):
        assert result[field] is False
    json.dumps(result, allow_nan=False)


def test_proton_si_missing_nuclear_is_not_a_measured_zero() -> None:
    result = compare_legacy_table(native(target=14), legacy(P_SI), P_SI)
    first = result["rows"][0]
    assert first["legacy_nuclear_source_missing"] is True
    assert first["legacy_nuclear_placeholder"] == "0"
    assert first["native_nuclear_mev_cm2_mg"] == "1"
    assert "nuclear_stopping_mev_cm2_mg" not in first["comparisons"]
    assert first["comparisons"]["electronic_stopping_mev_cm2_mg"]["exact_decimal_match"]
    total = first["comparisons"]["total_stopping_mev_cm2_mg"]
    assert Decimal(total["relative_difference_percent_of_legacy"]) == 50
    assert "incomplete" in total["comparison_meaning"]


@pytest.mark.parametrize("path,target", ((H_AL, 13), (H_SI, 14)))
@pytest.mark.parametrize("z,mass", ((2, "4"), (26, "56")))
def test_hze_energy_per_nucleon_conversion_retains_binary64_residuals(
    path: str, target: int, z: int, mass: str,
) -> None:
    result = compare_legacy_table(native(z=z, mass=mass, target=target), legacy(path), path)
    assert result["legacy_total_rows"] == 1845
    assert result["legacy_selected_ion_rows"] == 123
    assert sum(result["counts"].values()) == 123
    matched = [r for r in result["rows"] if "native_row" in r]
    assert len(matched) == 2
    if z == 26:
        assert result["counts"]["matched_binary64_roundtrip_energy"] == 2
        assert Decimal(matched[0]["native_minus_legacy_energy_mev"]) != 0
    else:
        assert result["counts"]["matched_exact_decimal_energy"] == 2
    if path == H_SI:
        assert all(m["exact_decimal_match"] for m in matched[0]["comparisons"].values())
    assert result["legacy_ion_mass_recorded"]


def test_coverage_is_retained_both_directions_without_interpolation() -> None:
    result = compare_legacy_table(native(energies=("2", "2.5", "4")), legacy(P_AL), P_AL)
    assert result["counts"]["unmatched_below_native_grid"] == 1
    assert result["counts"]["unmatched_inside_native_grid"] == 1
    assert result["counts"]["unmatched_above_native_grid"] == 153
    assert result["native_rows_without_legacy_match"] == [
        {"native_row": 2, "energy_total_mev": "2.5",
         "native_printed_tokens": ["2.5", "MeV", "2", "1", "2", "cm", "0.1", "cm", "0.1", "cm"]},
    ]
    assert result["rows"][2]["match_status"] == "unmatched_inside_native_grid"
    assert "comparisons" not in result["rows"][2]


def test_zero_legacy_denominator_is_null_not_zero_error() -> None:
    data = change(change(legacy(P_SI), column=1, value="0"), column=3, value="0")
    result = compare_legacy_table(native(target=14), data, P_SI)
    metric = result["rows"][0]["comparisons"]["electronic_stopping_mev_cm2_mg"]
    assert metric["relative_difference_percent_of_legacy"] is None
    assert metric["native_minus_legacy"] == "2"
    assert metric["exact_decimal_match"] is False
    assert result["summary"]["electronic_stopping_mev_cm2_mg"]["zero_legacy_denominator_rows"] == 1


def test_nonzero_value_difference_never_hidden_by_energy_tolerance() -> None:
    data = change(legacy(P_AL), column=1, value="3000.000000000000001")
    result = compare_legacy_table(native(), data, P_AL)
    metric = result["rows"][0]["comparisons"]["total_stopping_mev_cm2_mg"]
    assert not metric["exact_decimal_match"]
    assert Decimal(metric["native_minus_legacy"]) == Decimal("-1e-18")


def test_long_decimal_value_residual_is_not_rounded_to_zero() -> None:
    data = change(legacy(P_AL), column=1, value="3000." + "0" * 100 + "1")
    result = compare_legacy_table(native(), data, P_AL)
    metric = result["rows"][0]["comparisons"]["total_stopping_mev_cm2_mg"]
    assert not metric["exact_decimal_match"]
    assert Decimal(metric["native_minus_legacy"]) == Decimal("-1e-104")
    assert result["relative_percentage_decimal_precision"] == 80


def test_whole_ion_energy_overflow_is_rejected_not_matched_everywhere() -> None:
    data = change(legacy(H_AL), row=123, column=4, value="1e308")
    with pytest.raises(ValueError, match="coordinate"):
        compare_legacy_table(native(z=2, mass="4"), data, H_AL)


def test_too_large_coordinate_difference_is_unmatched_not_interpolated() -> None:
    data = change(legacy(P_AL), value="1.00000001")
    result = compare_legacy_table(native(), data, P_AL)
    assert result["rows"][0]["match_status"] == "unmatched_inside_native_grid"
    assert "comparisons" not in result["rows"][0]


def test_ambiguous_native_neighbors_are_rejected() -> None:
    output = native(energies=("1", str(nextafter(1.0, 2.0))))
    with pytest.raises(ValueError, match="Ambiguous"):
        compare_legacy_table(output, legacy(P_AL), P_AL)


def test_two_legacy_nodes_cannot_reuse_one_native_node() -> None:
    data = change(legacy(P_AL), row=2, value=str(nextafter(1.0, 2.0)))
    with pytest.raises(ValueError, match="Ambiguous"):
        compare_legacy_table(native(), data, P_AL)


def test_signed_percent_uses_legacy_denominator_and_keeps_extrema() -> None:
    data = change(legacy(P_AL), column=1, value="2000")
    data = change(data, row=2, column=1, value="6000")
    result = compare_legacy_table(native(), data, P_AL)
    summary = result["summary"]["total_stopping_mev_cm2_mg"]
    assert Decimal(summary["max_absolute_relative_difference_percent"]) == 50
    assert Decimal(summary["min_signed_relative_difference_percent"]) == -50
    assert Decimal(summary["max_signed_relative_difference_percent"]) == 50
    assert Decimal(summary["max_absolute_difference"]) == 3


@pytest.mark.parametrize("path", (P_AL, P_SI, H_AL, H_SI))
def test_truncated_csv_is_rejected(path: str) -> None:
    data = b"\n".join(legacy(path).splitlines()[:-1]) + b"\n"
    with pytest.raises(ValueError, match="record count"):
        compare_legacy_table(native(), data, path)


@pytest.mark.parametrize("value", ("nan", "inf", "-1", "", "1,0", "1e999", "1e-999"))
def test_malformed_energy_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        compare_legacy_table(native(), change(legacy(P_AL), value=value), P_AL)


@pytest.mark.parametrize("mutation", ("header", "width", "duplicate", "sheet", "control",
                                      "unicode", "blank", "negative_stopping"))
def test_structural_defects_are_rejected(mutation: str) -> None:
    data = legacy(P_AL)
    if mutation == "header":
        data = data.replace(b"energy_mev,", b"energy_kev,", 1)
    elif mutation == "width":
        data = data.replace(b",p_Al\n", b",p_Al,extra\n", 1)
    elif mutation == "duplicate":
        data = change(data, row=2, value="1")
    elif mutation == "sheet":
        data = change(data, column=4, value="other")
    elif mutation == "control":
        data = data.replace(b"p_Al", b"p_Al\x00", 1)
    elif mutation == "unicode":
        data += b"\xff"
    elif mutation == "blank":
        data += b"\n"
    else:
        data = change(data, column=1, value="-1")
    with pytest.raises(ValueError):
        compare_legacy_table(native(), data, P_AL)


@pytest.mark.parametrize("column,value", ((0, "3"), (1, "Fe"), (2, "5"), (3, "99"),
                                          (7, "wrong")))
def test_hze_roster_and_mass_fields_are_checked(column: int, value: str) -> None:
    with pytest.raises(ValueError):
        compare_legacy_table(native(z=2, mass="4"), change(legacy(H_AL), column=column,
                                                           value=value), H_AL)


def test_hze_mass_cannot_be_silently_changed_to_match() -> None:
    with pytest.raises(ValueError, match="mass differs"):
        compare_legacy_table(native(z=2, mass="4.0026"), legacy(H_AL), H_AL)


def test_wrong_target_wrong_ion_and_unknown_resource_fail() -> None:
    with pytest.raises(ValueError, match="target"):
        compare_legacy_table(native(target=14), legacy(P_AL), P_AL)
    with pytest.raises(ValueError, match="ion"):
        compare_legacy_table(native(z=2, mass="4"), legacy(P_AL), P_AL)
    with pytest.raises(ValueError, match="resource"):
        compare_legacy_table(native(), legacy(P_AL), "unregistered.csv")


def test_missing_nuclear_provenance_cannot_be_relabelled() -> None:
    with pytest.raises(ValueError, match="provenance"):
        compare_legacy_table(native(target=14), change(legacy(P_SI), column=5, value="H_n"), P_SI)


def test_no_matches_has_empty_summary_not_zero_error() -> None:
    result = compare_legacy_table(native(energies=("1000", "2000")), legacy(P_AL), P_AL)
    assert result["summary"] == {}
    assert result["counts"]["unmatched_below_native_grid"] == 157
    assert len(result["native_rows_without_legacy_match"]) == 2
    assert result["comparison_status"] == "INCOMPLETE"
