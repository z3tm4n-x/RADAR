"""Source-preserving IAEA archive/selection/unit witnesses; no physics acceptance."""

from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
from decimal import Decimal
import importlib
import io
import json
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any
import zipfile

import pytest


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        return importlib.import_module("iaea_stopping_reference")
    finally:
        sys.path.remove(str(scripts))


def _row(**changes: str) -> dict[str, str]:
    values = {
        "projectile_name": "C", "ion_isotope": "12.0", "target_name": "Al", "phase": "s",
        "energy": "12.0", "energy_unit": "MeV", "stopping_power": "0.25",
        "stopping_unit": "MeV/(mg/cm2)", "percentage_error": "",
        "target_mass_atom_ratio": "26.9815384", "dataset_last_modification": "2026-01-28",
        "publication_year": "2023", "ref_id": "Example2023", "citation_reference": "Example",
        "doi": "10.0/example", "comments": "unaltered, note\nsecond line",
        "facility": "Laboratory", "technique": "measurement",
        "stopping_power_converted": "0.25", "stopping_unit_converted": "MeV/(mg/cm2)",
    }
    values.update(changes)
    return values


def _csv(header: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([row[key] for key in header])
    return stream.getvalue().encode("utf-8")


def _members(runner: ModuleType, rows: list[dict[str, str]]) -> tuple[bytes, bytes]:
    references = [{"ref_id": ref_id, "doi": "10.0/example", "authors": "A. Author",
                   "citation_reference": "Example", "year": "2023"}
                  for ref_id in sorted({row["ref_id"] for row in rows})]
    return _csv(runner.DATA_HEADER, rows), _csv(runner.REFERENCE_HEADER, references)


def _inventory(runner: ModuleType, rows: list[dict[str, str]]) -> dict[str, Any]:
    records, refs = runner._parse_members(*_members(runner, rows))
    archive = runner.IaeaArchive(records, refs, (), ())
    result: dict[str, Any] = runner.build_inventory(archive)
    return result


def test_lossless_fields_and_logical_record_numbers(runner: ModuleType) -> None:
    first, second = _row(), _row(comments="third physical line is not record identity")
    records, refs = runner._parse_members(*_members(runner, [first, second]))
    assert records[0].fields == first
    assert records[1].fields == second
    assert [row.csv_record_number for row in records] == [2, 3]
    assert [row.data_record_number for row in records] == [1, 2]
    assert refs[0].csv_record_number == 2
    copied = records[0].fields
    copied["energy"] = "999"
    assert records[0].fields["energy"] == "12.0"
    with pytest.raises(FrozenInstanceError):
        records[0].csv_record_number = 100


@pytest.mark.parametrize("token", ["nan", "inf", "-1", "1_000", " 1", "1 ", "1e99999", ""])
def test_strict_numeric_token_rejected(runner: ModuleType, token: str) -> None:
    with pytest.raises(ValueError):
        runner._parse_members(*_members(runner, [_row(energy=token)]))


@pytest.mark.parametrize("column", ["energy", "ion_isotope", "target_mass_atom_ratio",
                                     "stopping_power", "stopping_power_converted"])
def test_nonfinite_not_silently_skipped(runner: ModuleType, column: str) -> None:
    with pytest.raises(ValueError):
        runner._parse_members(*_members(runner, [_row(**{column: "NaN"})]))


def test_missing_optional_numeric_is_preserved(runner: ModuleType) -> None:
    data, references = _members(runner, [_row(ion_isotope="", target_mass_atom_ratio="",
                                             stopping_power_converted="")])
    records, _ = runner._parse_members(data, references)
    assert records[0].fields["ion_isotope"] == ""
    report = _inventory(runner, [records[0].fields])
    assert report["selected_record_count"] == 0
    assert report["primary_exclusion_counts"] == {"explicit_positive_isotope_missing": 1}


@pytest.mark.parametrize("header", [(), ("energy",), ("projectile_name", "projectile_name")])
def test_header_exact(runner: ModuleType, header: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        runner._csv_records((",".join(header) + "\n1\n").encode(), runner.DATA_HEADER)


@pytest.mark.parametrize("suffix", [b"\n", b"only,three,fields\n", b'"unterminated\n', b"\x00\n"])
def test_bad_record_boundaries_rejected(runner: ModuleType, suffix: bytes) -> None:
    data, references = _members(runner, [_row()])
    with pytest.raises((ValueError, csv.Error)):
        runner._parse_members(data + suffix, references)


def test_unknown_and_duplicate_reference_rejected(runner: ModuleType) -> None:
    data, references = _members(runner, [_row()])
    with pytest.raises(ValueError):
        runner._parse_members(data.replace(b"Example2023", b"Unknown2023"), references)
    reference_line = references.splitlines(keepends=True)[1]
    with pytest.raises(ValueError):
        runner._parse_members(data, references + reference_line)


@pytest.mark.parametrize("column,value", [
    ("energy_unit", "GeV/u"), ("stopping_unit", "MeV/cm"),
    ("stopping_unit_converted", "MeV/(g/cm2)"), ("target_name", ""),
])
def test_schema_drift_fails_closed(runner: ModuleType, column: str, value: str) -> None:
    with pytest.raises(ValueError):
        runner._parse_members(*_members(runner, [_row(**{column: value})]))


@pytest.mark.parametrize("unit,expected", [("MeV", "1"), ("MeV/u", "12"),
                                           ("keV", ".001"), ("keV/u", ".012")])
def test_energy_axis_no_extra_or_missing_mass(runner: ModuleType, unit: str, expected: str) -> None:
    assert runner.energy_mev_per_nucleon(Decimal("12"), unit, 12) == Decimal(expected)


@pytest.mark.parametrize("isotope", [True, 12.0, 0, -1])
def test_energy_helper_rejects_nonintegral_identity(runner: ModuleType, isotope: Any) -> None:
    with pytest.raises(ValueError):
        runner.energy_mev_per_nucleon(Decimal("12"), "MeV", isotope)


def test_native_mass_and_atomic_unit_conversions(runner: ModuleType) -> None:
    assert runner.native_stopping_mev_cm2_g(Decimal(".25"), "MeV/(mg/cm2)", None) == 250
    # Independent dimensional value: 5e-15 eV cm²/atom × N_A /10g/mol /1e6.
    assert runner.native_stopping_mev_cm2_g(
        Decimal(5), "E-15eV cm2/atom", Decimal(10)) == Decimal("301.107038")
    with pytest.raises(ValueError):
        runner.native_stopping_mev_cm2_g(Decimal(5), "E-15eV cm2/atom", None)
    with pytest.raises(ValueError):
        runner.native_stopping_mev_cm2_g(Decimal(5), "eV/A", Decimal(10))


@pytest.mark.parametrize("changes,reason", [
    ({"ref_id": "ESPNN"}, "known_nonexperimental_model_reference"),
    ({"target_name": "SiO2"}, "target_not_elemental_Al_or_Si"),
    ({"phase": "s "}, "phase_not_literal_solid_s"),
    ({"phase": ""}, "phase_not_literal_solid_s"),
    ({"projectile_name": "He", "ion_isotope": "4"}, "projectile_outside_heavy_Z3_to92"),
    ({"ion_isotope": "12.01"}, "fractional_mass_not_explicit_isotope"),
    ({"ion_isotope": "5"}, "isotope_mass_below_atomic_number"),
    ({"ion_isotope": "0"}, "explicit_positive_isotope_missing"),
    ({"energy": ".119999"}, "energy_outside_0.01_to100000_MeV_per_u"),
    ({"energy": "1200000.01"}, "energy_outside_0.01_to100000_MeV_per_u"),
    ({"energy": "0"}, "energy_not_positive"),
    ({"energy_unit": "Mev"}, "unsupported_energy_unit_spelling"),
    ({"stopping_power": "0"}, "native_stopping_not_positive"),
    ({"stopping_unit": " MeV/(mg/cm2)"}, "unsupported_native_stopping_unit"),
    ({"stopping_unit": "E-15eV cm2/atom", "target_mass_atom_ratio": ""},
     "target_mass_missing_for_atomic_conversion"),
])
def test_explicit_exclusion_ledger(runner: ModuleType, changes: dict[str, str], reason: str) -> None:
    report = _inventory(runner, [_row(**changes)])
    assert report["selected_records"] == []
    assert reason in {entry["reason"] for entry in report["exclusion_groups_nonexclusive"]}
    assert report["exclusion_groups_nonexclusive"][0]["csv_record_numbers"] == [2]


@pytest.mark.parametrize("energy", [".12", "1200000"])
def test_inclusive_energy_boundaries(runner: ModuleType, energy: str) -> None:
    report = _inventory(runner, [_row(energy=energy)])
    assert report["selected_record_count"] == 1


@pytest.mark.parametrize("model", ["SRIM2013", "ICRU49", "MSTAR2003", "ESPNN", "DPASS21.06"])
def test_all_known_models_excluded_even_solid(runner: ModuleType, model: str) -> None:
    report = _inventory(runner, [_row(ref_id=model)])
    assert report["selected_record_count"] == 0
    assert report["known_model_reference_counts"][model] == 1


def test_provider_mismatch_kept_not_fitted_or_excluded(runner: ModuleType) -> None:
    report = _inventory(runner, [_row(stopping_power="5", stopping_unit="E-15eV cm2/atom",
                                     target_mass_atom_ratio="10", stopping_power_converted=".3")])
    row = report["selected_records"][0]
    assert row["stopping_native_mev_cm2_g"] == 301.107038
    assert row["stopping_converted_mev_cm2_g"] == 300
    assert row["provider_conversion_arithmetic_matches"] is False
    assert row["provider_conversion_status"] == "provider_conversion_mismatch_preserved"
    assert report["conversion_policy"]["fitted_conversion_constants"] is False


@pytest.mark.parametrize("error,status,scalar", [
    ("", "missing_not_zero", None), ("3-7", "opaque_reported_text", None),
    ("<5", "opaque_reported_text", None), ("2.5", "reported_scalar_percent_not_verified_confidence_interval", 2.5),
])
def test_uncertainty_text_not_reinterpreted(runner: ModuleType, error: str,
                                         status: str, scalar: float | None) -> None:
    row = _inventory(runner, [_row(percentage_error=error)])["selected_records"][0]
    assert row["reported_percentage_error_text"] == error
    assert row["reported_percentage_error_scalar"] == scalar
    assert row["uncertainty_interpretation"] == status


def test_duplicates_preserved_and_grouped_not_independent(runner: ModuleType) -> None:
    report = _inventory(runner, [_row(), _row(), _row(comments="different observation note")])
    assert report["selected_record_count"] == 3
    assert report["exact_raw_unique_record_count"] == 2
    assert report["exact_duplicate_excess_records"] == 1
    assert report["exact_raw_duplicate_groups"] == [
        {"duplicate_group_id": "csv_record_2", "csv_record_numbers": [2, 3], "multiplicity": 2}]
    assert [row["exact_raw_duplicate_multiplicity"] for row in report["selected_records"]] == [2, 2, 1]
    assert report["coverage_by_reference"][0]["record_count"] == 3
    assert report["coverage_by_reference"][0]["distinct_raw_record_count"] == 2
    assert report["coverage_by_reference"][0]["distinct_energy_count"] == 1


def test_reference_metadata_difference_not_silently_repaired(runner: ModuleType) -> None:
    row = _inventory(runner, [_row(doi="10.0/different")])["selected_records"][0]
    assert row["raw_fields"]["doi"] == "10.0/different"
    assert row["reference"]["doi"] == "10.0/example"
    assert row["reference_metadata_differences"] == ["doi"]


def _zip(member_names: list[str], mode: int = stat.S_IFREG | 0o644) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in member_names:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, b"content")
    return stream.getvalue()


@pytest.mark.parametrize("names", [[], ["../StoppingPower.csv"], ["sub/StoppingPower.csv"],
                                    ["StoppingPower.csv", "other.csv"]])
def test_exact_archive_member_inventory(runner: ModuleType, names: list[str]) -> None:
    with pytest.raises(ValueError):
        runner._archive_member(_zip(names), "StoppingPower.csv")


def test_duplicate_zip_member_rejected(runner: ModuleType) -> None:
    with pytest.warns(UserWarning):
        payload = _zip(["StoppingPower.csv", "StoppingPower.csv"])
    with pytest.raises(ValueError):
        runner._archive_member(payload, "StoppingPower.csv")


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFDIR | 0o755, stat.S_IFIFO | 0o644])
def test_nonregular_zip_members_rejected(runner: ModuleType, mode: int) -> None:
    with pytest.raises(ValueError):
        runner._archive_member(_zip(["StoppingPower.csv"], mode), "StoppingPower.csv")


def test_archive_hash_required_before_parsing(runner: ModuleType, tmp_path: Path) -> None:
    target = tmp_path / "not_official.zip"
    target.write_bytes(_zip(["StoppingPower.csv"]))
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.load_archives(target, target)


def test_report_exclusive_and_no_qualification(runner: ModuleType, tmp_path: Path) -> None:
    report = _inventory(runner, [_row()])
    target = tmp_path / "inventory.json"
    runner.write_report(report, target)
    original = target.read_bytes()
    decoded = json.loads(original)
    assert decoded["status"] == "INCOMPLETE"
    for key in ("normative_export_allowed", "physical_accuracy_verified", "production_tables_replaced"):
        assert decoded[key] is False
    with pytest.raises(FileExistsError):
        runner.write_report(report, target)
    assert target.read_bytes() == original


@pytest.fixture(scope="module")
def actual_report(runner: ModuleType) -> dict[str, Any]:
    result: dict[str, Any] = runner.build_inventory(runner.load_archives())
    return result


def test_actual_archive_inventory_and_models(actual_report: dict[str, Any]) -> None:
    assert actual_report["total_source_record_count"] == 274092
    assert actual_report["reference_count"] == 760
    assert actual_report["exact_raw_unique_record_count"] == 273777
    assert actual_report["exact_duplicate_excess_records"] == 315
    assert len(actual_report["exact_raw_duplicate_groups"]) == 293
    assert actual_report["known_model_reference_counts"] == {
        "SRIM2013": 203025, "ICRU49": 151, "MSTAR2003": 1997, "ESPNN": 1972, "DPASS21.06": 1085}
    assert actual_report["nonmodel_source_record_count_not_independent_measurements"] == 65862
    assert sum(actual_report["primary_exclusion_counts"].values()) + 2612 == 274092


def test_actual_selection_coverage(actual_report: dict[str, Any]) -> None:
    assert actual_report["selected_record_count"] == 2612
    assert len(actual_report["coverage_by_reference"]) == 97
    assert len(actual_report["coverage_by_ion_and_material"]) == 79
    rows = actual_report["selected_records"]
    assert len({(row["z"], row["mass_number"]) for row in rows}) == 52
    assert all(type(row["mass_number"]) is int and row["mass_number"] >= row["z"] for row in rows)
    assert max(row["energy_mev_per_nucleon"] for row in rows if row["material_z"] == 13) == 985
    assert max(row["energy_mev_per_nucleon"] for row in rows if row["material_z"] == 14) == 90.63


def test_actual_atomic_conversion_and_record_identity(actual_report: dict[str, Any]) -> None:
    mismatches = [row for row in actual_report["selected_records"]
                  if not row["provider_conversion_arithmetic_matches"]]
    assert len(mismatches) == 1
    row = mismatches[0]
    assert row["csv_record_number"] == 200868
    assert row["data_record_number"] == 200867
    assert (row["z"], row["mass_number"], row["material_z"]) == (7, 14, 14)
    assert row["energy_mev_per_nucleon"] == 153 / 14000
    assert row["stopping_native_mev_cm2_g"] == pytest.approx(80.8e-15 * 6.02214076e23 / 28.085 / 1e6)
    assert row["provider_conversion_signed_relative_difference"] == pytest.approx(-0.0002775032065945274)
    assert row["raw_fields"]["comments"] == "sample thicknesses 50-200A with random orientation"
    assert row["reported_percentage_error_text"] == "3-7"
