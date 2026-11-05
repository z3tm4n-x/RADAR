"""Strict reader for the retained SR Module's full stopping/range text format.

This new reader is not SRIM physics and does not authenticate a native program.
The special energy-list mode is intentionally unsupported. Projected range is
never labelled CSDA range. Original tokens and all native table columns survive.
Constructors check internal token/header consistency, not byte authenticity:
only parsing retained raw bytes establishes their hash, and a caller can build
an internally consistent synthetic object with a different claimed raw hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
import hashlib
import json
import re
from typing import Any


_NUMBER = re.compile(r"[+-]?[0-9]+(?:[.,][0-9]+)?(?:[Ee][+-]?[0-9]{1,3})?")
_ENERGY = {"eV": Decimal(".000001"), "keV": Decimal(".001"),
           "MeV": Decimal(1), "GeV": Decimal(1000)}
_DISTANCE = {"A": Decimal("1e-8"), "um": Decimal(".0001"),
             "mm": Decimal(".1"), "cm": Decimal(1), "m": Decimal(100)}
_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Po At Rn Fr Ra Ac Th Pa U"
).split()
_FOOTER_UNITS = (
    "eV/Angstrom", "keV/micron", "MeV/mm", "keV/(ug/cm2)",
    "MeV/(mg/cm2)", "keV/(mg/cm2)", "eV/(1E15 atoms/cm2)", "L.S.S. reduced units",
)
_FULL_RULE = "=" * 66
_RANGE_RULE = "-" * 59
_COPYRIGHT = "(C) 1984,1989,1992,1998,2006 by J.P. Biersack and J.F. Ziegler"
_METADATA_NUMBERS = (
    "ion_z", "ion_mass_u", "target_z", "density_g_cm3", "atomic_density_cm3",
    "target_atomic_percent", "target_mass_percent", "bragg_correction_percent",
)


def _header_fields(original_lines: tuple[str, ...]) -> dict[str, str]:
    """Validate the complete header and retain its exact numerical tokens."""
    if (not isinstance(original_lines, tuple) or len(original_lines) != 21
            or any(not isinstance(line, str) for line in original_lines)):
        raise ValueError("SRIM header metadata must retain all 21 immutable lines.")
    if any(any(ord(c) < 32 and c != "\t" or ord(c) == 127 for c in line)
           for line in original_lines):
        raise ValueError("Unexpected control character in SRIM header metadata.")
    lines = tuple(" ".join(line.split()) for line in original_lines)
    if any(not line.isascii() for i, line in enumerate(lines) if i != 3):
        raise ValueError("Non-ASCII structural or numerical SRIM header metadata.")
    literal_headers = {
        0: _FULL_RULE, 4: _FULL_RULE, 8: "======= Target Composition ========",
        9: "Atom Atom Atomic Mass", 10: "Name Numb Percent Percent",
        11: "---- ---- ------- -------", 13: "=" * 36,
        15: "Stopping Units = MeV/(mg/cm2)",
        16: "See bottom of Table for other Stopping units",
        18: "Ion dE/dx dE/dx Projected Longitudinal Lateral",
        19: "Energy Elec. Nuclear Range Straggling Straggling",
        20: "----------- ---------- ---------- ---------- ---------- ----------",
    }
    if any(lines[index] != expected for index, expected in literal_headers.items()):
        raise ValueError("Wrong SRIM full-range header, columns, units or target structure.")

    def capture(index: int, pattern: str) -> tuple[str, ...]:
        match = re.fullmatch(pattern, lines[index])
        if match is None:
            raise ValueError(f"Invalid SRIM header at nonblank line {index + 1}.")
        return match.groups()

    fields: dict[str, str] = {}
    fields["engine_label"], = capture(1, r"Calculation using (SRIM-[0-9]{4}(?:\.[0-9]+)?)")
    fields["version_label"], = capture(2, r"SRIM version ---> (SRIM-[0-9]{4}(?:\.[0-9]+)?)")
    fields["calculation_date"], = capture(3, r"Calc\. date ---> (.{1,100})")
    fields["output_name"], = capture(5, r"Disk File Name = ([A-Za-z0-9_. -]{1,100})")
    fields["ion_name"], fields["ion_z"], fields["ion_mass_u"] = capture(
        6, r"Ion = ([A-Za-z]+) \[([0-9]{1,2})\] , Mass = (\S+) amu",
    )
    if lines[6] != lines[17]:
        raise ValueError("Repeated SRIM ion metadata differs.")
    fields["density_g_cm3"], fields["atomic_density_cm3"] = capture(
        7, r"Density = (\S+) g/cm3 = (\S+) atoms/cm3",
    )
    (fields["target_symbol"], fields["target_z"], fields["target_atomic_percent"],
     fields["target_mass_percent"]) = capture(12, r"([A-Z][a-z]?) ([0-9]{1,2}) (\S+) (\S+)")
    fields["bragg_correction_percent"], = capture(14, r"Bragg Correction = (\S+)%")
    return fields


def _decimal(token: str) -> Decimal:
    if (not isinstance(token, str) or len(token) > 64
            or _NUMBER.fullmatch(token) is None):
        raise ValueError("Invalid SRIM decimal token; mixed/grouped separators are unsupported.")
    value = Decimal(token.replace(",", "."))
    exponent = value.as_tuple().exponent
    if not value.is_finite() or not isinstance(exponent, int) or abs(exponent) > 100:
        raise ValueError("Nonfinite or excessive SRIM numeric exponent.")
    return value


def _canonical(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _validate_value(value: Decimal, *, positive: bool = False) -> None:
    if (not isinstance(value, Decimal) or not value.is_finite()
            or value < 0 or (positive and value == 0)):
        raise ValueError("SRIM physical fields must be finite and nonnegative (positive if required).")


@dataclass(frozen=True)
class SrimRangeRow:
    energy_total_mev: Decimal
    electronic_mev_cm2_mg: Decimal
    nuclear_mev_cm2_mg: Decimal
    projected_range_cm: Decimal
    longitudinal_straggling_cm: Decimal
    lateral_straggling_cm: Decimal
    printed_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.printed_tokens, tuple) or len(self.printed_tokens) != 10:
            raise ValueError("Six numeric and four unit SRIM row tokens must be retained.")
        tokens = self.printed_tokens
        if tokens[1] not in _ENERGY or any(tokens[i] not in _DISTANCE for i in (5, 7, 9)):
            raise ValueError("Unsupported SRIM energy or distance unit.")
        with localcontext() as context:
            context.prec = 256
            expected = (
                _decimal(tokens[0]) * _ENERGY[tokens[1]], _decimal(tokens[2]),
                _decimal(tokens[3]), _decimal(tokens[4]) * _DISTANCE[tokens[5]],
                _decimal(tokens[6]) * _DISTANCE[tokens[7]],
                _decimal(tokens[8]) * _DISTANCE[tokens[9]],
            )
        values = (
            self.energy_total_mev, self.electronic_mev_cm2_mg, self.nuclear_mev_cm2_mg,
            self.projected_range_cm, self.longitudinal_straggling_cm,
            self.lateral_straggling_cm,
        )
        for index, (value, wanted) in enumerate(zip(values, expected, strict=True)):
            _validate_value(value, positive=index in (0, 3))
            if value != wanted:
                raise ValueError("SRIM converted field differs from its original token/unit.")

    @property
    def energy_unit(self) -> str:
        return self.printed_tokens[1]

    @property
    def range_unit(self) -> str:
        return self.printed_tokens[5]

    @property
    def stopping_total_mev_cm2_mg(self) -> Decimal:
        with localcontext() as context:
            # Each token has <=64 characters and a stored exponent in [-100, 100].
            # 1000 digits safely retain even the smallest allowed component
            # alongside the largest allowed component, independently of callers.
            context.prec = 1000
            return self.electronic_mev_cm2_mg + self.nuclear_mev_cm2_mg


@dataclass(frozen=True)
class SrimConversionFactor:
    unit: str
    factor: Decimal
    printed_token: str

    def __post_init__(self) -> None:
        _validate_value(self.factor, positive=True)
        if self.unit not in _FOOTER_UNITS or self.factor != _decimal(self.printed_token):
            raise ValueError("Invalid SRIM footer factor or unit.")


@dataclass(frozen=True)
class SrimRangeOutput:
    raw_sha256: str
    engine_label: str
    version_label: str
    output_name: str
    calculation_date: str
    ion_name: str
    ion_z: int
    ion_mass_u: Decimal
    target_symbol: str
    target_z: int
    density_g_cm3: Decimal
    atomic_density_cm3: Decimal
    target_atomic_percent: Decimal
    target_mass_percent: Decimal
    bragg_correction_percent: Decimal
    metadata_printed_tokens: tuple[tuple[str, str], ...]
    metadata_lines: tuple[str, ...]
    rows: tuple[SrimRangeRow, ...]
    conversion_factors: tuple[SrimConversionFactor, ...]
    range_kind: str = field(default="projected", init=False)
    energy_basis: str = field(default="total_ion_kinetic_energy", init=False)
    stopping_unit: str = field(default="MeV/(mg/cm2)", init=False)
    encoding: str = field(default="cp1251", init=False)
    physical_accuracy_verified: bool = field(default=False, init=False)
    source_authenticity_verified: bool = field(default=False, init=False)
    normative_export_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.raw_sha256) is None:
            raise ValueError("SRIM output requires its raw SHA256.")
        if (type(self.ion_z) is not int or not 1 <= self.ion_z <= 92
                or type(self.target_z) is not int or not 1 <= self.target_z <= 92
                or self.target_symbol != _ELEMENTS[self.target_z - 1]):
            raise ValueError("Invalid ion Z or single-element target identification.")
        fields = _header_fields(self.metadata_lines)
        expected_tokens = tuple((key, fields[key]) for key in _METADATA_NUMBERS)
        if (not isinstance(self.metadata_printed_tokens, tuple)
                or self.metadata_printed_tokens != expected_tokens):
            raise ValueError("SRIM header metadata tokens differ from the retained header.")
        for key, token in expected_tokens:
            value = getattr(self, key)
            if key not in ("ion_z", "target_z") and (
                    not isinstance(value, Decimal) or not value.is_finite()):
                raise ValueError("SRIM header metadata numeric fields must be finite Decimal values.")
            if value != _decimal(token):
                raise ValueError("SRIM header metadata field differs from its original token.")
        for key in ("engine_label", "version_label", "output_name", "calculation_date",
                    "ion_name", "target_symbol"):
            if getattr(self, key) != fields[key]:
                raise ValueError("SRIM header metadata label differs from the retained header.")
        for value in (self.ion_mass_u, self.density_g_cm3, self.atomic_density_cm3):
            _validate_value(value, positive=True)
        if self.target_atomic_percent != 100 or self.target_mass_percent != 100:
            raise ValueError("Only a single-element 100% target is supported.")
        if not self.bragg_correction_percent.is_finite():
            raise ValueError("Nonfinite Bragg correction.")
        if (not isinstance(self.rows, tuple) or not 2 <= len(self.rows) <= 1000
                or any(not isinstance(row, SrimRangeRow) for row in self.rows)):
            raise ValueError("Expected 2..1000 immutable full-range rows.")
        for left, right in zip(self.rows, self.rows[1:]):
            if left.energy_total_mev >= right.energy_total_mev:
                raise ValueError("SRIM energies must strictly increase.")
            # Native low-energy output rounds ranges to integer Angstroms.
            # Equal PRINTED values are valid observations, not zero CSDA loss.
            # We preserve their tokens and never invert these projected ranges.
            if left.projected_range_cm > right.projected_range_cm:
                raise ValueError("SRIM printed projected ranges must not decrease.")
        if (not isinstance(self.conversion_factors, tuple)
                or tuple(x.unit for x in self.conversion_factors) != _FOOTER_UNITS):
            raise ValueError("All eight ordered footer conversion factors are required.")
        factors = tuple(x.factor for x in self.conversion_factors)
        if factors[3:6] != (Decimal(1), Decimal(1), Decimal(1000)):
            raise ValueError("Footer is not expressed relative to MeV/(mg/cm2).")
        with localcontext() as context:
            context.prec = 256
            if factors[1] != 10 * factors[0] or factors[2] != factors[1]:
                raise ValueError("Inconsistent printed dimensional footer conversion factors.")

    def semantic_payload(self) -> dict[str, Any]:
        """Parsed physics/metadata identity; excludes date, filename and text formatting.

        No physical correctness is implied. Normalization removes decimal-locale
        rendering, while all stopping, projected-range and straggling values and
        every footer factor remain represented.
        """
        return {
            "schema": "radar.srim_range_semantics.v1",
            "engine_label": self.engine_label, "version_label": self.version_label,
            "ion_name": self.ion_name, "ion_z": self.ion_z,
            "ion_mass_u": _canonical(self.ion_mass_u),
            "target_symbol": self.target_symbol, "target_z": self.target_z,
            "density_g_cm3": _canonical(self.density_g_cm3),
            "atomic_density_cm3": _canonical(self.atomic_density_cm3),
            "target_atomic_percent": _canonical(self.target_atomic_percent),
            "target_mass_percent": _canonical(self.target_mass_percent),
            "bragg_correction_percent": _canonical(self.bragg_correction_percent),
            "range_kind": self.range_kind, "energy_basis": self.energy_basis,
            "stopping_unit": self.stopping_unit,
            "rows": [
                {name: _canonical(getattr(row, name)) for name in (
                    "energy_total_mev", "electronic_mev_cm2_mg", "nuclear_mev_cm2_mg",
                    "projected_range_cm", "longitudinal_straggling_cm", "lateral_straggling_cm",
                )} for row in self.rows
            ],
            "conversion_factors": [
                {"unit": factor.unit, "factor": _canonical(factor.factor)}
                for factor in self.conversion_factors
            ],
        }

    @property
    def semantic_sha256(self) -> str:
        encoded = json.dumps(self.semantic_payload(), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


def parse_srim_range_output(raw: bytes) -> SrimRangeOutput:
    """Read the complete retained full-range format or raise ValueError.

    No unknown line, incomplete header/footer, unsupported unit, malformed row
    or special-list output is skipped. cp1251 supports the observed Russian
    date; all numerical/structural fields remain ASCII.
    """
    if not isinstance(raw, bytes) or not raw or len(raw) > 1_000_000:
        raise ValueError("SRIM output must be bounded nonempty bytes.")
    try:
        text = raw.decode("cp1251")
    except UnicodeDecodeError as error:
        raise ValueError("SRIM output is not valid cp1251 text.") from error
    if any(ord(c) < 32 and c not in "\r\n\t" or ord(c) == 127 for c in text):
        raise ValueError("Unexpected control character in SRIM output.")
    original_lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    lines = tuple(" ".join(line.split()) for line in original_lines)
    if len(lines) < 36:
        raise ValueError("Incomplete SRIM full-range output.")
    if any(not line.isascii() for i, line in enumerate(lines) if i != 3):
        raise ValueError("Non-ASCII structural or numerical SRIM fields.")
    fields = _header_fields(original_lines[:21])
    metadata_tokens = tuple((key, fields[key]) for key in _METADATA_NUMBERS)
    try:
        footer_start = lines.index(_RANGE_RULE, 21)
    except ValueError as error:
        raise ValueError("Missing SRIM range-table terminator.") from error
    if len(lines) - footer_start != 13:
        raise ValueError("Wrong SRIM footer length or trailing content.")
    if (lines[footer_start + 1] != "Multiply Stopping by for Stopping Units"
            or lines[footer_start + 2] != "------------------- ------------------"
            or lines[-2] != _FULL_RULE or lines[-1] != _COPYRIGHT):
        raise ValueError("Missing or malformed complete SRIM conversion/copyright footer.")
    numbers = [token for _, token in metadata_tokens]
    rows: list[SrimRangeRow] = []
    with localcontext() as context:
        context.prec = 256
        for line in lines[21:footer_start]:
            tokens = tuple(line.split())
            if len(tokens) != 10:
                raise ValueError("Expected all six numeric and four unit SRIM row fields.")
            if tokens[1] not in _ENERGY or any(tokens[i] not in _DISTANCE for i in (5, 7, 9)):
                raise ValueError("Unsupported SRIM energy or distance unit.")
            numbers.extend(tokens[i] for i in (0, 2, 3, 4, 6, 8))
            rows.append(SrimRangeRow(
                _decimal(tokens[0]) * _ENERGY[tokens[1]], _decimal(tokens[2]),
                _decimal(tokens[3]), _decimal(tokens[4]) * _DISTANCE[tokens[5]],
                _decimal(tokens[6]) * _DISTANCE[tokens[7]],
                _decimal(tokens[8]) * _DISTANCE[tokens[9]], tokens,
            ))
    conversions = []
    for line, unit in zip(lines[footer_start + 3:footer_start + 11], _FOOTER_UNITS, strict=True):
        token, _, found_unit = line.partition(" ")
        if found_unit != unit:
            raise ValueError("Missing, reordered or unknown SRIM footer conversion unit.")
        numbers.append(token)
        conversions.append(SrimConversionFactor(unit, _decimal(token), token))
    separators = {char for token in numbers for char in token if char in ",."}
    if len(separators) > 1:
        raise ValueError("Mixed decimal separator conventions in SRIM numeric fields.")
    return SrimRangeOutput(
        hashlib.sha256(raw).hexdigest(), fields["engine_label"], fields["version_label"],
        fields["output_name"], fields["calculation_date"], fields["ion_name"],
        int(fields["ion_z"]), _decimal(fields["ion_mass_u"]), fields["target_symbol"],
        int(fields["target_z"]), _decimal(fields["density_g_cm3"]),
        _decimal(fields["atomic_density_cm3"]), _decimal(fields["target_atomic_percent"]),
        _decimal(fields["target_mass_percent"]), _decimal(fields["bragg_correction_percent"]),
        metadata_tokens, original_lines[:21], tuple(rows),
        tuple(conversions),
    )
