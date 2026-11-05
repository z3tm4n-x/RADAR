"""Synthetic structural checks, not physical golden values from COSRAD."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from radar.validation.cosrad_gcr_text import load_cosrad_gcr_text, parse_cosrad_gcr_text


def _lines(
    temporal: str = "FLUENCE [particles/(MeV/nuclon*cm2)]",
    energy_label: str = "1 col. - Energy [MeV/nuclon]",
) -> list[str]:
    labels = [f"{z}X*****" for z in range(1, 93)]
    labels[0], labels[1], labels[33] = "1H*******", "2He******", "Se34*****"
    header = [
        "Start year (from begining of solar cycle) =1 year",
        "Flight time=10 year(s)",
        "Solar  cycle number - even",
        "Solar cycle - mean",
        "Apogee = 36000 km.",
        "Perigee = 36000 km.",
        "Inclination = 0.0 deg.",
        "Argument of perigee = 0.0 deg.",
        "GCR PARTICLE SPECTRUM in orbit",
        temporal,
        energy_label,
        " 2-93 col. " + "  ".join(labels),
    ]
    # Deliberately synthetic and simple, not exported COSRAD model predictions.
    return header + [
        " ".join([str(energy), *(f"{z / energy:.8E}" for z in range(1, 93))])
        for energy in range(1, 51)
    ]


def _bytes(lines: list[str]) -> bytes:
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


@pytest.mark.parametrize(
    ("temporal", "statistic", "units"),
    [
        ("FLUENCE [particles/(MeV/nuclon*cm2)]", "fluence", "particles/(MeV/nuclon*cm2)"),
        (
            "Maximal orbit FLUX [particles/(MeV/nuclon*cm2*day)]",
            "maximal_orbit_flux_per_day", "particles/(MeV/nuclon*cm2*day)",
        ),
        (
            "Maximal peak FLUX [particles/(MeV/nuclon*cm2*s)]",
            "maximal_peak_flux_per_second", "particles/(MeV/nuclon*cm2*s)",
        ),
    ],
)
@pytest.mark.parametrize("label", ["Energy [MeV/nuclon]", "1 col. - Energy [MeV/nuclon]"])
def test_native_statistics_units_and_values_are_not_converted(
    temporal: str, statistic: str, units: str, label: str,
) -> None:
    lines = _lines(temporal, label)
    data = _bytes(lines)
    table = parse_cosrad_gcr_text(data)
    assert table.statistic == statistic
    assert table.native_units == units
    assert table.raw_header == "\r\n".join(lines[:12]) + "\r\n"
    assert table.raw_data_lines == tuple(lines[12:])
    assert table.source_sha256 == hashlib.sha256(data).hexdigest()
    assert table.energies_mev_per_nucleon == tuple(float(i) for i in range(1, 51))
    assert table.values_for_atomic_number(2)[0] == 2.0
    assert table.values_for_atomic_number(92)[0] == 92.0
    assert table.native_ion_labels[33] == "Se34*****"
    assert table.scenario.start_year_from_solar_cycle_beginning == 1
    assert table.scenario.start_year_basis == "relative_to_solar_cycle_start"
    assert table.scenario.flight_years == 10
    assert table.scenario.solar_cycle_parity == "even"
    assert table.scenario.solar_cycle_level == "mean"


def test_file_load_uses_content_not_filename_and_does_not_modify_file(tmp_path: Path) -> None:
    # An intentionally misleading name must not override the fluence header.
    path = tmp_path / "gfxm.txt"
    data = _bytes(_lines())
    path.write_bytes(data)
    before = path.stat().st_mtime_ns
    table = load_cosrad_gcr_text(path)
    assert table.statistic == "fluence"
    assert table.source_name == str(path)
    assert path.read_bytes() == data
    assert path.stat().st_mtime_ns == before


def test_structural_success_cannot_authorize_normative_use_or_isolated_nek() -> None:
    table = parse_cosrad_gcr_text(_bytes(_lines()))
    serialized = asdict(table)
    assert serialized["evidence_role"] == "independent_model_comparison_only"
    assert serialized["component"] == "orbital_total_gcr_by_atomic_number"
    assert serialized["helium_component"] == "total_gcr_he_not_isolated_nek"
    assert serialized["normative_release_eligible"] is False
    for field in ("run_freshness", "executable_version", "active_parameter_table"):
        assert serialized[field] == "not_established"
    with pytest.raises(FrozenInstanceError):
        setattr(table, "normative_release_eligible", True)


@pytest.mark.parametrize(
    ("index", "replacement"),
    [
        (0, "Start year =2026 year"),
        (0, "Start year (from begining of solar cycle) =0 year"),
        (1, "Flight time=0 year(s)"),
        (1, "Flight time=1.5 year(s)"),
        (2, "Solar cycle number - unknown"),
        (3, "Solar cycle - "),
        (4, "Apogee = 35000 km."),
        (5, "Perigee = -1 km."),
        (6, "Inclination = 181 deg."),
        (7, "Argument of perigee = 361 deg."),
        (8, "SEP PARTICLE SPECTRUM in orbit"),
        (9, "FLUENCE [particles/(MeV/nuclon*cm2*s)]"),
        (9, "Maximal orbit FLUX [particles/(MeV/nuclon*cm2*s)]"),
        (9, "Maximal peak FLUX [particles/(MeV/nuclon*cm2*day)]"),
        (10, "Energy [MeV]"),
        (11, "2-92 col. 1H 2He"),
    ],
)
def test_rejects_wrong_or_inconsistent_headers(index: int, replacement: str) -> None:
    lines = _lines()
    lines[index] = replacement
    with pytest.raises(ValueError):
        parse_cosrad_gcr_text(_bytes(lines))


@pytest.mark.parametrize("index", range(12))
def test_rejects_missing_or_duplicate_header_lines(index: int) -> None:
    lines = _lines()
    missing = lines[:index] + lines[index + 1:]
    duplicate = lines[:index] + [lines[index]] + lines[index:]
    with pytest.raises(ValueError):
        parse_cosrad_gcr_text(_bytes(missing))
    with pytest.raises(ValueError):
        parse_cosrad_gcr_text(_bytes(duplicate))


@pytest.mark.parametrize("mutation", ["blank", "missing_row", "extra_row", "missing_value", "extra_value"])
def test_rejects_partial_or_overfull_tables(mutation: str) -> None:
    lines = _lines()
    if mutation == "blank":
        lines[-1] = " "
    elif mutation == "missing_row":
        lines.pop()
    elif mutation == "extra_row":
        lines.append(lines[-1])
    elif mutation == "missing_value":
        lines[-1] = " ".join(lines[-1].split()[:-1])
    else:
        lines[-1] += " 0"
    with pytest.raises(ValueError):
        parse_cosrad_gcr_text(_bytes(lines))


@pytest.mark.parametrize(
    "token", ["NaN", "Inf", "-Inf", "1E999", "-1", "1_0", "1D+02", "1E-9999", "-1E-9999"],
)
def test_rejects_invalid_flux_values(token: str) -> None:
    lines = _lines()
    row = lines[12].split()
    row[2] = token
    lines[12] = " ".join(row)
    with pytest.raises(ValueError):
        parse_cosrad_gcr_text(_bytes(lines))


@pytest.mark.parametrize("token", ["0", "-1", "NaN", "1E999", "1", "0.5"])
def test_rejects_nonpositive_nonfinite_or_nonincreasing_energy(token: str) -> None:
    lines = _lines()
    row = lines[13].split()
    row[0] = token
    lines[13] = " ".join(row)
    with pytest.raises(ValueError):
        parse_cosrad_gcr_text(_bytes(lines))


@pytest.mark.parametrize("replacement", ["2He******", "99X*****", "X*****", "3X", "-3X*****"])
def test_rejects_duplicate_missing_or_unsupported_charge_identifiers(replacement: str) -> None:
    lines = _lines()
    lines[11] = lines[11].replace("3X*****", replacement)
    with pytest.raises(ValueError):
        parse_cosrad_gcr_text(_bytes(lines))


def test_accepts_finite_zero_preserving_native_printed_precision() -> None:
    lines = _lines()
    row = lines[12].split()
    row[1:4] = ["0.00E+0000", "1.23E-0004", "-0.0"]
    lines[12] = " ".join(row)
    table = parse_cosrad_gcr_text(_bytes(lines))
    assert table.values_by_energy_z1_to_92[0][:3] == (0.0, 0.000123, -0.0)


def test_raw_data_lines_preserve_lexical_precision_and_whitespace() -> None:
    lines = _lines()
    row = lines[12].split()
    row[1:5] = ["1.0", "1.00", "1.2300E-0004", "0.0000"]
    lines[12] = "  " + "   ".join(row) + " "
    table = parse_cosrad_gcr_text(_bytes(lines))
    assert table.raw_data_lines[0] == lines[12]
    assert table.raw_data_lines[0].split()[1:5] == row[1:5]
    assert table.values_by_energy_z1_to_92[0][:4] == (1.0, 1.0, 0.000123, 0.0)


@pytest.mark.parametrize("token", ["1E-9999", "-1E-9999"])
def test_rejects_underflow_in_numeric_header_fields(token: str) -> None:
    lines = _lines()
    lines[6] = f"Inclination = {token} deg."
    with pytest.raises(ValueError, match="underflows"):
        parse_cosrad_gcr_text(_bytes(lines))


@pytest.mark.parametrize("z", [0, -1, 93, True, 1.5, "2"])
def test_column_selection_rejects_non_atomic_numbers(z: object) -> None:
    table = parse_cosrad_gcr_text(_bytes(_lines()))
    with pytest.raises(ValueError):
        table.values_for_atomic_number(z)  # type: ignore[arg-type]


def test_hash_distinguishes_line_endings_without_changing_values() -> None:
    data = _bytes(_lines())
    crlf = parse_cosrad_gcr_text(data)
    lf = parse_cosrad_gcr_text(data.replace(b"\r\n", b"\n"))
    assert crlf.source_sha256 != lf.source_sha256
    assert crlf.raw_header != lf.raw_header
    assert crlf.values_by_energy_z1_to_92 == lf.values_by_energy_z1_to_92


def test_rejects_non_ascii_and_empty_source_name() -> None:
    with pytest.raises(ValueError, match="ASCII"):
        parse_cosrad_gcr_text(b"\xff")
    with pytest.raises(ValueError, match="source name"):
        parse_cosrad_gcr_text(_bytes(_lines()), source_name=" ")
