"""Read pinned IAEA compilation CSV archives without cleaning source observations.

This is a diagnostic inventory, not experimental acceptance or production data.
The compilation includes calculated model tables and duplicated observations.
Every original CSV field remains in immutable records; selected JSON records
retain their original fields, while excluded records are indexed back to the
pinned archive. CSV record numbers count the header as record 1 (not physical
lines: quoted fields can contain newlines). No isotope is inferred or rounded.

Atomic stopping cross sections use the exact SI Avogadro constant, independently
of the provider's converted column. A discrepancy is reported, never fitted.
No workbook or source CSV is authored. Reuse/redistribution is not qualified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/iaea_stopping_20261031"
DATA_SHA256 = "0813d4137aa571eb8d8d48e96ed7ca3dd2b7ef10e510d627e7d01029379ba01b"
REFERENCES_SHA256 = "9d5b2bfec9aa4b51ad5a5fc64e1d6b34228d38f1e5c263f23b886159e03faed4"
DATA_HEADER = (
    "projectile_name", "ion_isotope", "target_name", "phase", "energy", "energy_unit",
    "stopping_power", "stopping_unit", "percentage_error", "target_mass_atom_ratio",
    "dataset_last_modification", "publication_year", "ref_id", "citation_reference",
    "doi", "comments", "facility", "technique", "stopping_power_converted",
    "stopping_unit_converted",
)
REFERENCE_HEADER = ("ref_id", "doi", "authors", "citation_reference", "year")
MODEL_REFERENCE_IDS = frozenset({"SRIM2013", "ICRU49", "MSTAR2003", "ESPNN", "DPASS21.06"})
SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At "
    "Rn Fr Ra Ac Th Pa U"
).split()
ATOMIC_NUMBERS = {symbol: index + 1 for index, symbol in enumerate(SYMBOLS)}
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_NUMERIC_FIELDS = (
    "ion_isotope", "energy", "stopping_power", "target_mass_atom_ratio",
    "stopping_power_converted",
)
_ENERGY_UNITS = frozenset({"MeV/u", "keV/u", "MeV", "keV", "Mev"})
_STOPPING_UNITS = frozenset({"MeV/(mg/cm2)", " MeV/(mg/cm2)", "E-15eV cm2/atom", "eV/A"})
AVOGADRO_PER_MOL = Decimal("6.02214076e23")


@dataclass(frozen=True)
class SourceRecord:
    """Unmodified source fields; numbers remain text in this lossless layer."""

    csv_record_number: int
    raw_fields: tuple[str, ...]

    @property
    def data_record_number(self) -> int:
        return self.csv_record_number - 1

    @property
    def fields(self) -> dict[str, str]:
        return dict(zip(DATA_HEADER, self.raw_fields, strict=True))


@dataclass(frozen=True)
class ReferenceRecord:
    csv_record_number: int
    raw_fields: tuple[str, ...]

    @property
    def fields(self) -> dict[str, str]:
        return dict(zip(REFERENCE_HEADER, self.raw_fields, strict=True))


@dataclass(frozen=True)
class IaeaArchive:
    records: tuple[SourceRecord, ...]
    references: tuple[ReferenceRecord, ...]
    archive_sha256: tuple[tuple[str, str], ...]
    member_sha256: tuple[tuple[str, str], ...]


def _decimal(text: str, *, optional: bool = False) -> Decimal | None:
    if text == "" and optional:
        return None
    if _NUMBER.fullmatch(text) is None:
        raise ValueError(f"Invalid numeric source token: {text!r}")
    value = Decimal(text)
    if not value.is_finite() or value < 0 or not math.isfinite(float(value)):
        raise ValueError("Negative or nonfinite source number.")
    return value


def _archive_member(payload: bytes, expected_member: str) -> bytes:
    """Read one regular flat ZIP member; no extraction, links or extra members."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != expected_member:
            raise ValueError("Unexpected, duplicate or missing ZIP members.")
        member = members[0]
        mode = member.external_attr >> 16
        if (member.is_dir() or member.flag_bits & 1 or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                or member.file_size > 100_000_000):
            raise ValueError("Unsafe or oversized ZIP member.")
        return archive.read(member)


def _csv_records(payload: bytes, header: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    text = payload.decode("utf-8-sig", errors="strict")
    if "\x00" in text:
        raise ValueError("NUL in CSV source.")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    if tuple(next(reader, ())) != header:
        raise ValueError("CSV header does not match the retained schema.")
    rows = []
    for row in reader:
        if len(row) != len(header):
            raise ValueError("CSV record field count differs from the header.")
        rows.append(tuple(row))
    if not rows:
        raise ValueError("CSV source must contain data records.")
    return tuple(rows)


def _parse_members(data: bytes, references: bytes) -> tuple[
    tuple[SourceRecord, ...], tuple[ReferenceRecord, ...],
]:
    ref_records = tuple(ReferenceRecord(i, row) for i, row in
                        enumerate(_csv_records(references, REFERENCE_HEADER), 2))
    ref_ids = [record.raw_fields[0] for record in ref_records]
    if any(not ref for ref in ref_ids) or len(set(ref_ids)) != len(ref_ids):
        raise ValueError("Missing or duplicate reference identifiers.")
    known_refs = set(ref_ids)
    result = []
    for number, row in enumerate(_csv_records(data, DATA_HEADER), 2):
        record = SourceRecord(number, row)
        fields = record.fields
        if fields["ref_id"] not in known_refs:
            raise ValueError("Source record has an unresolved bibliography reference.")
        for key in _NUMERIC_FIELDS:
            _decimal(fields[key], optional=key not in ("energy", "stopping_power"))
        if fields["energy_unit"] not in _ENERGY_UNITS:
            raise ValueError("Unknown source energy unit; schema review is required.")
        if fields["stopping_unit"] not in _STOPPING_UNITS:
            raise ValueError("Unknown source stopping unit; schema review is required.")
        if fields["stopping_unit_converted"] != "MeV/(mg/cm2)":
            raise ValueError("Unexpected provider-converted stopping unit.")
        if not fields["projectile_name"] or not fields["target_name"]:
            raise ValueError("Missing projectile or target name.")
        result.append(record)
    return tuple(result), ref_records


def load_archives(
    data_path: Path = SOURCE / "data.zip",
    references_path: Path = SOURCE / "references.zip",
) -> IaeaArchive:
    """Load only the two fixed, retained IAEA archives with complete raw records."""
    payloads = (data_path.read_bytes(), references_path.read_bytes())
    hashes = tuple(hashlib.sha256(data).hexdigest() for data in payloads)
    if hashes != (DATA_SHA256, REFERENCES_SHA256):
        raise ValueError("Pinned IAEA archive hash mismatch.")
    members = (_archive_member(payloads[0], "StoppingPower.csv"),
               _archive_member(payloads[1], "StoppingPower_refs.csv"))
    records, refs = _parse_members(*members)
    if (data_path.read_bytes(), references_path.read_bytes()) != payloads:
        raise ValueError("IAEA archive changed during parsing.")
    return IaeaArchive(
        records, refs, (("data.zip", hashes[0]), ("references.zip", hashes[1])),
        tuple((name, hashlib.sha256(payload).hexdigest()) for name, payload in
              zip(("StoppingPower.csv", "StoppingPower_refs.csv"), members, strict=True)),
    )


def energy_mev_per_nucleon(value: Decimal, unit: str, isotope: int) -> Decimal:
    """Use the literal unit: MeV/keV are whole-ion, /u axes are already per mass."""
    if type(isotope) is not int or isotope <= 0 or not value.is_finite() or value <= 0:
        raise ValueError("Positive explicit isotope and energy required.")
    if unit == "MeV/u":
        return value
    if unit == "keV/u":
        return value / 1000
    if unit == "MeV":
        return value / isotope
    if unit == "keV":
        return value / (1000 * isotope)
    raise ValueError("Unsupported energy spelling; no silent normalization.")


def native_stopping_mev_cm2_g(
    value: Decimal, unit: str, target_mass: Decimal | None,
) -> Decimal:
    if not value.is_finite() or value <= 0:
        raise ValueError("Positive finite stopping required.")
    if unit == "MeV/(mg/cm2)":
        return value * 1000
    if unit == "E-15eV cm2/atom":
        if target_mass is None or not target_mass.is_finite() or target_mass <= 0:
            raise ValueError("Atomic cross-section conversion needs target molar mass.")
        return value * Decimal("1e-15") * AVOGADRO_PER_MOL / target_mass / Decimal("1e6")
    raise ValueError("Unsupported native stopping unit; no guessed density or conversion.")


def _selection(fields: dict[str, str]) -> tuple[list[str], dict[str, Any] | None]:
    reasons = []
    z = ATOMIC_NUMBERS.get(fields["projectile_name"])
    if fields["ref_id"] in MODEL_REFERENCE_IDS:
        reasons.append("known_nonexperimental_model_reference")
    if fields["target_name"] not in ("Al", "Si"):
        reasons.append("target_not_elemental_Al_or_Si")
    if fields["phase"] != "s":
        reasons.append("phase_not_literal_solid_s")
    if z is None or not 3 <= z <= 92:
        reasons.append("projectile_outside_heavy_Z3_to92")
    mass = _decimal(fields["ion_isotope"], optional=True)
    isotope = None
    if mass is None or mass <= 0:
        reasons.append("explicit_positive_isotope_missing")
    elif mass != mass.to_integral_value():
        reasons.append("fractional_mass_not_explicit_isotope")
    elif z is not None and mass < z:
        reasons.append("isotope_mass_below_atomic_number")
    else:
        isotope = int(mass)
    energy = _decimal(fields["energy"])
    normalized_energy = None
    if energy is None or energy <= 0:
        reasons.append("energy_not_positive")
    elif fields["energy_unit"] == "Mev":
        reasons.append("unsupported_energy_unit_spelling")
    elif isotope is not None:
        normalized_energy = energy_mev_per_nucleon(energy, fields["energy_unit"], isotope)
        if not Decimal("0.01") <= normalized_energy <= Decimal("100000"):
            reasons.append("energy_outside_0.01_to100000_MeV_per_u")
    value = _decimal(fields["stopping_power"])
    target_mass = _decimal(fields["target_mass_atom_ratio"], optional=True)
    if value is None or value <= 0:
        reasons.append("native_stopping_not_positive")
    if fields["stopping_unit"] not in ("MeV/(mg/cm2)", "E-15eV cm2/atom"):
        reasons.append("unsupported_native_stopping_unit")
    elif fields["stopping_unit"] == "E-15eV cm2/atom" and (
        target_mass is None or target_mass <= 0
    ):
        reasons.append("target_mass_missing_for_atomic_conversion")
    if reasons:
        return reasons, None
    assert isotope is not None and normalized_energy is not None and value is not None
    native = native_stopping_mev_cm2_g(value, fields["stopping_unit"], target_mass)
    provider = _decimal(fields["stopping_power_converted"], optional=True)
    provider_g = provider * 1000 if provider is not None else None
    relative = (provider_g / native - 1) if provider_g is not None else None
    # This is only arithmetic agreement (1e-12), never a physical tolerance.
    conversion_matches = relative is not None and abs(relative) <= Decimal("1e-12")
    error = fields["percentage_error"]
    error_scalar = _decimal(error) if _NUMBER.fullmatch(error) else None
    return [], {
        "z": z, "projectile_z": z, "mass_number": isotope,
        "mass_parameter_u": float(isotope), "material_z": {"Al": 13, "Si": 14}[fields["target_name"]],
        "energy_mev_per_nucleon": float(normalized_energy),
        "energy_mev_per_u": float(normalized_energy),
        "stopping_native_mev_cm2_g": float(native),
        "stopping_converted_mev_cm2_g": float(provider_g) if provider_g is not None else None,
        "provider_conversion_signed_relative_difference": float(relative) if relative is not None else None,
        "provider_conversion_arithmetic_matches": conversion_matches,
        "provider_conversion_status": ("matches_independent_conversion" if conversion_matches else
                                       "provider_value_missing" if provider_g is None else
                                       "provider_conversion_mismatch_preserved"),
        "reported_percentage_error_text": error,
        "reported_percentage_error_scalar": float(error_scalar) if error_scalar is not None else None,
        "uncertainty_interpretation": ("reported_scalar_percent_not_verified_confidence_interval"
                                       if error_scalar is not None else
                                       "missing_not_zero" if error == "" else "opaque_reported_text"),
        "source_record_status": "compilation_candidate_experimental_not_article_verified",
    }


def _coverage(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, values in sorted(groups.items()):
        energies = [row["energy_mev_per_nucleon"] for row in values]
        result.append({
            **dict(zip(keys, group, strict=True)), "record_count": len(values),
            "distinct_raw_record_count": len({row["duplicate_group_id"] for row in values}),
            "distinct_energy_count": len(set(energies)), "energy_min_mev_per_nucleon": min(energies),
            "energy_max_mev_per_nucleon": max(energies),
            "records_above_250_mev_per_nucleon": sum(energy > 250 for energy in energies),
            "csv_record_numbers": [row["csv_record_number"] for row in values],
        })
    return result


def build_inventory(archive: IaeaArchive) -> dict[str, Any]:
    """Select bounded Al/Si heavy-ion diagnostics, preserving exclusion ledgers."""
    refs = {ref.raw_fields[0]: ref.fields for ref in archive.references}
    multiplicities = Counter(record.raw_fields for record in archive.records)
    first_record: dict[tuple[str, ...], int] = {}
    duplicate_records: dict[int, list[int]] = defaultdict(list)
    exclusions: dict[str, list[int]] = defaultdict(list)
    primary_exclusions: Counter[str] = Counter()
    selected = []
    for record in archive.records:
        first = first_record.setdefault(record.raw_fields, record.csv_record_number)
        if multiplicities[record.raw_fields] > 1:
            duplicate_records[first].append(record.csv_record_number)
        fields = record.fields
        reasons, converted = _selection(fields)
        if reasons:
            primary_exclusions[reasons[0]] += 1
            for reason in reasons:
                exclusions[reason].append(record.csv_record_number)
            continue
        assert converted is not None
        reference = refs[fields["ref_id"]]
        selected.append({
            "csv_record_number": record.csv_record_number,
            "data_record_number": record.data_record_number, "ref_id": fields["ref_id"],
            "raw_fields": fields, "reference": reference,
            "reference_metadata_differences": [key for key, ref_key in
                                               (("doi", "doi"), ("citation_reference", "citation_reference"),
                                                ("publication_year", "year"))
                                               if fields[key] != reference[ref_key]],
            "duplicate_group_id": f"csv_record_{first}",
            "exact_raw_duplicate_multiplicity": multiplicities[record.raw_fields],
            **converted,
        })
    all_refs = Counter(record.raw_fields[12] for record in archive.records)
    return {
        "schema_version": "radar.iaea_stopping_reference_inventory.v1",
        "status": "INCOMPLETE", "normative_export_allowed": False,
        "physical_accuracy_verified": False, "production_tables_replaced": False,
        "archive_sha256": dict(archive.archive_sha256), "member_sha256": dict(archive.member_sha256),
        "source_urls": {
            "data": "https://nds.iaea.org/stopping/download/tuples",
            "references": "https://nds.iaea.org/stopping/download/refs",
        },
        "data_header": DATA_HEADER, "reference_header": REFERENCE_HEADER,
        "csv_record_number_policy": "one_based_logical_CSV_record_with_header_as_1",
        "all_original_records_preserved_in_pinned_archive_and_loader": True,
        "unselected_original_values": "read_via_csv_record_number_from_pinned_StoppingPower.csv",
        "total_source_record_count": len(archive.records), "reference_count": len(archive.references),
        "raw_unit_inventory": [
            {"energy_unit": key[0], "stopping_unit": key[1],
             "stopping_unit_converted": key[2], "record_count": count}
            for key, count in sorted(Counter(
                (record.raw_fields[5], record.raw_fields[7], record.raw_fields[19])
                for record in archive.records).items())
        ],
        "known_model_reference_counts": {key: all_refs[key] for key in sorted(MODEL_REFERENCE_IDS)},
        "nonmodel_source_record_count_not_independent_measurements": sum(
            count for key, count in all_refs.items() if key not in MODEL_REFERENCE_IDS),
        "exact_raw_unique_record_count": len(multiplicities),
        "exact_duplicate_excess_records": sum(count - 1 for count in multiplicities.values()),
        "exact_raw_duplicate_groups": [
            {"duplicate_group_id": f"csv_record_{first}", "csv_record_numbers": numbers,
             "multiplicity": len(numbers)} for first, numbers in sorted(duplicate_records.items())
        ],
        "selected_record_count": len(selected), "selected_records": selected,
        "primary_exclusion_counts": dict(sorted(primary_exclusions.items())),
        "exclusion_groups_nonexclusive": [
            {"reason": reason, "record_count": len(numbers), "csv_record_numbers": numbers}
            for reason, numbers in sorted(exclusions.items())
        ],
        "coverage_by_reference": _coverage(selected, ("ref_id",)),
        "coverage_by_ion_and_material": _coverage(selected, ("z", "mass_number", "material_z")),
        "coverage_by_reference_ion_material": _coverage(selected, ("ref_id", "z", "mass_number", "material_z")),
        "selected_conversion_status_counts": dict(Counter(row["provider_conversion_status"] for row in selected)),
        "conversion_policy": {
            "avogadro_per_mol_exact": str(AVOGADRO_PER_MOL),
            "atomic_cross_section": "raw * 1e-15 eV cm2/atom * N_A / target_molar_mass / 1e6",
            "mass_stopping": "MeV cm2/mg * 1000 = MeV cm2/g",
            "energy": "MeV/u unchanged; keV/u /1000; whole MeV /A; whole keV /(1000*A)",
            "arithmetic_relative_tolerance_not_physical_tolerance": 1e-12,
            "provider_mismatch_excluded": False, "fitted_conversion_constants": False,
        },
        "limitations": [
            "The archive mixes observations and model tables; five identified model references are excluded.",
            "Exact duplicate records are retained, labelled and not independent measurements.",
            "Explicit integral isotope labels are used without rounding fractional or missing masses.",
            "MeV/u is preserved numerically as the candidate coordinate; precise isotope mass is not inferred.",
            "Missing uncertainty is unknown, not zero; scalar/range error confidence meanings are unverified.",
            "Native and provider-converted stopping remain distinct; mismatches are not fitted away.",
            "Electronic versus total experimental stopping must be verified from each original article.",
            "Selected records are a compilation inventory, not approval of individual experiments or CATIMA.",
            "No range measurements or high-energy coverage are inferred from stopping-power observations.",
        ],
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SOURCE / "data.zip")
    parser.add_argument("--references", type=Path, default=SOURCE / "references.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    implementation_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = build_inventory(load_archives(args.data, args.references))
    if (hashlib.sha256(args.data.read_bytes()).hexdigest() != DATA_SHA256
            or hashlib.sha256(args.references.read_bytes()).hexdigest() != REFERENCES_SHA256
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != implementation_before):
        raise ValueError("Archive or implementation changed during inventory.")
    report["implementation_sha256"] = implementation_before
    write_report(report, args.output)
    print(json.dumps({"selected_records": report["selected_record_count"],
                      "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "normative_export_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
