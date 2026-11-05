"""Strict reader for unmodified NIST ESTAR 1.2 MOUT=1 aluminium output.

Derived from the documented output format of NIST STAR 1.2. This is a new
reader, not an implementation of ESTAR physics or proof that bytes came from
NIST. Archive/build provenance must be checked separately. No interpolation,
fitting, extrapolation or production-table replacement is performed.
NIST permits redistribution/modification with a derivative/modified notice:
https://physics.nist.gov/PhysRefData/Star/Text/download.html
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
import hashlib
import io
from math import isfinite
import re


DEFAULT_ENERGIES_MEV = tuple(
    Decimal(f"{mantissa}E{exponent}")
    for exponent in range(-2, 3)
    for mantissa in ("1", "1.25", "1.5", "1.75", "2", "2.5", "3", "3.5",
                     "4", "4.5", "5", "5.5", "6", "7", "8", "9")
) + (Decimal("1000"),)
DERIVATIVE_NOTICE = (
    "Derived from NIST STAR 1.2 output by a new strict Python reader; "
    "parsing alone does not verify the producing executable or physical accuracy."
)
_PRINTED = re.compile(r"([0-9])\.([0-9]{3})[EeDd]([+-][0-9]{2})")
_CSV_NUMBER = re.compile(r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]{1,3})?")
_HEADERS = (
    "KINETIC STOPPING POWER CSDA RADIATION DENSITY",
    "ENERGY COLLISION RADIATIVE TOTAL RANGE YIELD EFFECT",
    "DELTA",
    "MeV MeV cm2/g MeV cm2/g MeV cm2/g g/cm2",
)
_CSV_HEADERS = (
    "energy_mev", "stopping_mev_cm2_g", "range_g_cm2", "range_mg_cm2", "source_sheet",
)


def _ascii(raw: bytes, name: str) -> str:
    if not isinstance(raw, bytes) or not raw or len(raw) > 100_000:
        raise ValueError(f"{name} must be nonempty bounded ASCII bytes.")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} must contain ASCII only.") from exc
    if any(ord(char) < 32 and char not in "\t\r\n\f" or ord(char) == 127 for char in text):
        raise ValueError(f"Unexpected control character in {name}.")
    return text


def _printed_decimal(token: str) -> Decimal:
    if not isinstance(token, str) or _PRINTED.fullmatch(token) is None:
        raise ValueError("ESTAR numeric rows require four-significant-figure E/D tokens.")
    value = Decimal(token.upper().replace("D", "E"))
    if token[0] == "0" and value != 0:
        raise ValueError("ESTAR scientific tokens must be normalized.")
    return value


def half_printed_unit(token: str) -> Decimal:
    """Half one last printed decimal unit, not a physical error estimate."""
    _printed_decimal(token)
    match = _PRINTED.fullmatch(token)
    assert match is not None
    return Decimal(f"5E{int(match.group(3)) - 4}")


@dataclass(frozen=True)
class EstarRow:
    energy_mev: Decimal
    stopping_collision_mev_cm2_g: Decimal
    stopping_radiative_mev_cm2_g: Decimal
    stopping_total_mev_cm2_g: Decimal
    range_csda_g_cm2: Decimal
    radiation_yield: Decimal
    density_effect: Decimal
    printed_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        values = (
            self.energy_mev, self.stopping_collision_mev_cm2_g,
            self.stopping_radiative_mev_cm2_g, self.stopping_total_mev_cm2_g,
            self.range_csda_g_cm2, self.radiation_yield, self.density_effect,
        )
        if not isinstance(self.printed_tokens, tuple) or len(self.printed_tokens) != 7:
            raise ValueError("Exactly seven immutable ESTAR printed tokens are required.")
        for i, (value, token) in enumerate(zip(values, self.printed_tokens, strict=True)):
            if (not isinstance(value, Decimal) or not value.is_finite()
                    or value < 0 or (i < 5 and value == 0)):
                raise ValueError("ESTAR values must be finite nonnegative Decimal numbers.")
            if value != _printed_decimal(token):
                raise ValueError("ESTAR value does not match its retained printed token.")
        if self.radiation_yield > 1:
            raise ValueError("Radiation yield cannot exceed one.")
        with localcontext() as context:
            context.prec = 4096
            error = abs(self.stopping_total_mev_cm2_g
                        - self.stopping_collision_mev_cm2_g
                        - self.stopping_radiative_mev_cm2_g)
            radius = sum((half_printed_unit(self.printed_tokens[i]) for i in (1, 2, 3)),
                         Decimal(0))
            if error > radius:
                raise ValueError("Total stopping is inconsistent with rounded collision+radiative.")

    @property
    def tokens(self) -> tuple[str, ...]:
        return self.printed_tokens


@dataclass(frozen=True)
class EstarAlOutput:
    rows: tuple[EstarRow, ...]
    raw_sha256: str
    density_g_cm3: Decimal
    mean_excitation_energy_ev: Decimal
    metadata_printed_tokens: tuple[tuple[str, str], ...]
    version: str = field(default="1.2", init=False)
    material_id: int = field(default=13, init=False)
    material_name: str = field(default="ALUMINUM", init=False)
    derivative_notice: str = field(default=DERIVATIVE_NOTICE, init=False)
    source_authenticity_verified: bool = field(default=False, init=False)
    fortran_execution_verified: bool = field(default=False, init=False)
    physical_accuracy_verified: bool = field(default=False, init=False)
    normative_export_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.rows, tuple) or len(self.rows) != 81
                or any(not isinstance(row, EstarRow) for row in self.rows)):
            raise ValueError("Exactly 81 immutable ESTAR rows are required.")
        if tuple(row.energy_mev for row in self.rows) != DEFAULT_ENERGIES_MEV:
            raise ValueError("ESTAR energies must match the ordered 81-node 0.01..1000 MeV grid.")
        if any(a.range_csda_g_cm2 >= b.range_csda_g_cm2
               for a, b in zip(self.rows, self.rows[1:])):
            raise ValueError("ESTAR CSDA ranges must increase strictly.")
        if (not isinstance(self.raw_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.raw_sha256) is None):
            raise ValueError("A raw-output SHA256 is required.")
        if (type(self.density_g_cm3) is not Decimal or not self.density_g_cm3.is_finite()
                or self.density_g_cm3 != Decimal("2.6989")
                or type(self.mean_excitation_energy_ev) is not Decimal
                or not self.mean_excitation_energy_ev.is_finite()
                or self.mean_excitation_energy_ev != Decimal("166.0")):
            raise ValueError("ESTAR 1.2 aluminium composition metadata differs.")
        if (not isinstance(self.metadata_printed_tokens, tuple)
                or len(self.metadata_printed_tokens) != 2):
            raise ValueError("Both repeated metadata records are required.")
        for tokens in self.metadata_printed_tokens:
            _metadata(tokens)


def _metadata(tokens: tuple[str, ...]) -> tuple[Decimal, Decimal]:
    if (not isinstance(tokens, tuple) or len(tokens) != 2
            or not isinstance(tokens[0], str) or not isinstance(tokens[1], str)
            or re.fullmatch(r"2\.6989[EeDd]\+00", tokens[0]) is None
            or tokens[1] != "166.0"):
        raise ValueError("Expected printed aluminium density 2.6989E+00 and I=166.0 eV.")
    return Decimal("2.6989"), Decimal("166.0")


def parse_estar_output(raw: bytes) -> EstarAlOutput:
    """Read both full MOUT=1 pages; no arbitrary text or missing headers accepted.

    Whitespace and CRLF/LF are immaterial; the original seven row tokens and
    two metadata records are retained. Four-digit rounding cells may intersect
    at a tie: the reader does not guess the producing runtime's tie rule.
    """
    text = _ascii(raw, "ESTAR output")
    pages = text.split("\f")
    if len(pages) != 2:
        raise ValueError("Expected exactly two ESTAR pages separated by one form-feed.")
    rows: list[EstarRow] = []
    metadata: list[tuple[str, str]] = []
    for page, expected_rows in zip(pages, (40, 41), strict=True):
        lines = tuple(" ".join(line.split()) for line in page.splitlines() if line.strip())
        if len(lines) != expected_rows + 7:
            raise ValueError("Wrong ESTAR page row/header count.")
        if lines[0] != "Electrons in ALUMINUM":
            raise ValueError("Expected Electrons in ALUMINUM material header.")
        if lines[1] != "Density, g/cm3 I, eV":
            raise ValueError("Wrong ESTAR composition units/header.")
        tokens = tuple(lines[2].split())
        _metadata(tokens)
        metadata.append((tokens[0], tokens[1]))
        if lines[3:7] != _HEADERS:
            raise ValueError("Wrong ESTAR columns, order or physical units.")
        for line in lines[7:]:
            row_tokens = tuple(line.split())
            if len(row_tokens) != 7:
                raise ValueError("Exactly seven numerical ESTAR columns are required.")
            values = tuple(_printed_decimal(token) for token in row_tokens)
            rows.append(EstarRow(values[0], values[1], values[2], values[3], values[4],
                                 values[5], values[6], row_tokens))
    return EstarAlOutput(tuple(rows), hashlib.sha256(raw).hexdigest(),
                         Decimal("2.6989"), Decimal("166.0"), tuple(metadata))


@dataclass(frozen=True)
class EstarCsvDifference:
    row_number: int
    field_name: str
    expected: Decimal
    observed: Decimal

    @property
    def signed_difference(self) -> Decimal:
        with localcontext() as context:
            context.prec = 4096
            return self.observed - self.expected


@dataclass(frozen=True)
class EstarCsvComparison:
    estar_raw_sha256: str
    csv_raw_sha256: str
    row_count: int
    primary_differences: tuple[EstarCsvDifference, ...]
    range_mg_differences: tuple[EstarCsvDifference, ...]
    range_mg_internal_differences: tuple[EstarCsvDifference, ...]
    range_mg_binary64_text_only_rows: tuple[int, ...]
    exact_primary_match: bool = field(init=False)
    exact_range_mg_match: bool = field(init=False)
    range_mg_internal_consistent: bool = field(init=False)
    normative_export_allowed: bool = field(default=False, init=False)
    physical_accuracy_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exact_primary_match", not self.primary_differences)
        object.__setattr__(self, "exact_range_mg_match", not self.range_mg_differences)
        object.__setattr__(self, "range_mg_internal_consistent",
                           not self.range_mg_internal_differences)


def _csv_decimal(token: str) -> Decimal:
    if len(token) > 128 or _CSV_NUMBER.fullmatch(token) is None:
        raise ValueError("CSV fields must contain finite positive decimal numbers.")
    value = Decimal(token)
    if not value.is_finite() or value <= 0:
        raise ValueError("CSV fields must contain finite positive decimal numbers.")
    return value


def compare_electron_al_csv(output: EstarAlOutput, csv_bytes: bytes) -> EstarCsvComparison:
    """Compare primary fields exactly; report redundant mg fields separately.

    No tolerance can turn a primary-field difference into a match. For the
    redundant mg column only, a separate ledger identifies exact Python
    binary64 multiplication/rendering artifacts, without erasing differences.
    Row numbers count data rows from one, not the CSV header.
    """
    if not isinstance(output, EstarAlOutput):
        raise ValueError("A parsed ESTAR aluminium output is required.")
    text = _ascii(csv_bytes, "electron Al CSV")
    try:
        records = tuple(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise ValueError("Malformed electron Al CSV.") from exc
    if len(records) != 82 or tuple(records[0]) != _CSV_HEADERS:
        raise ValueError("Expected exact electron Al CSV header and 81 data records.")
    primary: list[EstarCsvDifference] = []
    mg: list[EstarCsvDifference] = []
    mg_internal: list[EstarCsvDifference] = []
    binary_rows: list[int] = []
    previous_energy = previous_range = Decimal(0)
    with localcontext() as context:
        context.prec = 4096
        for index, (record, row) in enumerate(zip(records[1:], output.rows, strict=True), 1):
            if len(record) != 5 or record[4] != "e_Al":
                raise ValueError("Wrong electron Al CSV columns or source-sheet label.")
            energy, stopping, csda, range_mg = tuple(_csv_decimal(v) for v in record[:4])
            if energy <= previous_energy or csda <= previous_range:
                raise ValueError("CSV energies and ranges must be strictly increasing.")
            previous_energy, previous_range = energy, csda
            expected = (row.energy_mev, row.stopping_total_mev_cm2_g, row.range_csda_g_cm2)
            for name, left, right in zip(_CSV_HEADERS[:3], expected,
                                         (energy, stopping, csda), strict=True):
                if left != right:
                    primary.append(EstarCsvDifference(index, name, left, right))
            if range_mg != row.range_csda_g_cm2 * 1000:
                mg.append(EstarCsvDifference(index, "range_mg_cm2",
                                             row.range_csda_g_cm2 * 1000, range_mg))
            if range_mg != csda * 1000:
                mg_internal.append(EstarCsvDifference(index, "range_mg_cm2", csda * 1000,
                                                      range_mg))
                binary_product = float(csda) * 1000
                if isfinite(binary_product) and range_mg == Decimal(str(binary_product)):
                    binary_rows.append(index)
    return EstarCsvComparison(output.raw_sha256, hashlib.sha256(csv_bytes).hexdigest(), 81,
                              tuple(primary), tuple(mg), tuple(mg_internal), tuple(binary_rows))
