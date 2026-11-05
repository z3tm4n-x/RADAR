"""Strict diagnostic reader for the observed COSRAD GCR text-output format.

This is an independent-model comparison input, not a normative reference.
The reader recognizes the observed 50-energy, Z=1..92 orbital GCR tables.
It preserves native units, temporal statistics, printed values and labels;
it never converts orbital total helium into an isolated directional NEK flux.
Successful parsing establishes only structural validity of these bytes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Final, Literal


CosradGcrStatistic = Literal[
    "fluence", "maximal_orbit_flux_per_day", "maximal_peak_flux_per_second",
]

_HEADER_LINES: Final = 12
_ENERGY_ROWS: Final = 50
_ION_COUNT: Final = 92
_NUMBER: Final = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?"
_TEMPORAL_HEADERS: Final[dict[str, tuple[CosradGcrStatistic, str]]] = {
    "FLUENCE [particles/(MeV/nuclon*cm2)]": (
        "fluence", "particles/(MeV/nuclon*cm2)",
    ),
    "Maximal orbit FLUX [particles/(MeV/nuclon*cm2*day)]": (
        "maximal_orbit_flux_per_day", "particles/(MeV/nuclon*cm2*day)",
    ),
    "Maximal peak FLUX [particles/(MeV/nuclon*cm2*s)]": (
        "maximal_peak_flux_per_second", "particles/(MeV/nuclon*cm2*s)",
    ),
}


@dataclass(frozen=True)
class CosradGcrScenario:
    """Native scenario declarations, without an inferred calendar or Wolf series."""

    start_year_from_solar_cycle_beginning: int
    flight_years: int
    solar_cycle_parity: str
    solar_cycle_level: str
    apogee_km: float
    perigee_km: float
    inclination_deg: float
    argument_of_perigee_deg: float
    start_year_basis: str = field(default="relative_to_solar_cycle_start", init=False)


@dataclass(frozen=True)
class CosradGcrTextTable:
    """Parsed native total-GCR columns; no physical acceptance follows from parsing."""

    source_name: str
    source_sha256: str
    raw_header: str
    scenario: CosradGcrScenario
    statistic: CosradGcrStatistic
    native_units: str
    native_ion_labels: tuple[str, ...]
    raw_data_lines: tuple[str, ...]
    energies_mev_per_nucleon: tuple[float, ...]
    values_by_energy_z1_to_92: tuple[tuple[float, ...], ...]
    component: str = field(default="orbital_total_gcr_by_atomic_number", init=False)
    helium_component: str = field(default="total_gcr_he_not_isolated_nek", init=False)
    evidence_role: str = field(default="independent_model_comparison_only", init=False)
    normative_release_eligible: bool = field(default=False, init=False)
    run_freshness: str = field(default="not_established", init=False)
    executable_version: str = field(default="not_established", init=False)
    active_parameter_table: str = field(default="not_established", init=False)
    label_policy: str = field(default="native_labels_preserved_without_correction", init=False)

    def values_for_atomic_number(self, z: int) -> tuple[float, ...]:
        """Select one TOTAL-GCR column without temporal, angular or component conversion."""
        if isinstance(z, bool) or not isinstance(z, int) or not 1 <= z <= _ION_COUNT:
            raise ValueError("COSRAD atomic number must be an integer in 1..92.")
        return tuple(row[z - 1] for row in self.values_by_energy_z1_to_92)


def _match(line: str, pattern: str, name: str) -> str:
    result = re.fullmatch(pattern, line.strip())
    if result is None:
        raise ValueError(f"Unrecognized or misplaced COSRAD {name} header.")
    return result.group(1)


def _finite_number(token: str, name: str) -> float:
    if re.fullmatch(_NUMBER, token) is None:
        raise ValueError(f"COSRAD {name} must be a finite decimal number.")
    value = float(token)
    if not isfinite(value):
        raise ValueError(f"COSRAD {name} must be finite.")
    significand = token.lower().partition("e")[0]
    if value == 0.0 and any(character in "123456789" for character in significand):
        raise ValueError(f"COSRAD {name} has nonzero decimal data that underflows to float zero.")
    return value


def _scenario(lines: list[str]) -> CosradGcrScenario:
    start = int(_match(
        lines[0], r"Start year \(from begining of solar cycle\)\s*=\s*([0-9]+) year",
        "cycle-relative start year",
    ))
    duration = int(_match(lines[1], r"Flight time\s*=\s*([0-9]+) year\(s\)", "flight time"))
    parity = _match(lines[2], r"Solar\s+cycle number\s*-\s*(even|odd)", "cycle parity")
    # Preserve this label, but do not infer a Wolf-number history from its wording.
    level = _match(lines[3], r"Solar cycle\s*-\s*([A-Za-z]+(?: [A-Za-z]+)*)", "cycle level")
    apogee = _finite_number(
        _match(lines[4], rf"Apogee\s*=\s*({_NUMBER}) km\.", "apogee"), "apogee",
    )
    perigee = _finite_number(
        _match(lines[5], rf"Perigee\s*=\s*({_NUMBER}) km\.", "perigee"), "perigee",
    )
    inclination = _finite_number(
        _match(lines[6], rf"Inclination\s*=\s*({_NUMBER}) deg\.", "inclination"),
        "inclination",
    )
    argument = _finite_number(
        _match(lines[7], rf"Argument of perigee\s*=\s*({_NUMBER}) deg\.", "argument"),
        "argument of perigee",
    )
    if start <= 0 or duration <= 0:
        raise ValueError("COSRAD cycle-relative start year and flight years must be positive.")
    if not 0.0 <= perigee <= apogee:
        raise ValueError("COSRAD orbit must satisfy 0 <= perigee <= apogee.")
    if not 0.0 <= inclination <= 180.0 or not 0.0 <= argument <= 360.0:
        raise ValueError("COSRAD orbital angles are outside their supported ranges.")
    return CosradGcrScenario(start, duration, parity, level, apogee, perigee, inclination, argument)


def _ion_labels(line: str) -> tuple[str, ...]:
    labels = tuple(_match(line, r"2-93 col\.\s+(.+)", "ion columns").split())
    if len(labels) != _ION_COUNT:
        raise ValueError("COSRAD column declaration must contain all 92 ion labels.")
    for expected_z, label in enumerate(labels, start=1):
        # The observed native declaration includes `Se34`, unlike `2He`.
        match = re.fullmatch(r"(?:([0-9]+)[A-Za-z]+|[A-Za-z]+([0-9]+))\*+", label)
        if match is None or int(match.group(1) or match.group(2)) != expected_z:
            raise ValueError("COSRAD ion column identifiers must be ordered and unique Z=1..92.")
    return labels


def parse_cosrad_gcr_text(data: bytes, *, source_name: str = "<bytes>") -> CosradGcrTextTable:
    """Parse the observed ASCII 50-row format, failing closed on unsupported variants.

    Full-row truncation is detectable because this format has exactly 50 rows.
    No inferred file freshness, completion status or model provenance is attached.
    SHA-256 identifies the exact supplied bytes, including line endings.
    """
    if not source_name.strip():
        raise ValueError("COSRAD source name must not be blank.")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Observed COSRAD GCR text format must be ASCII.") from exc
    lines = text.splitlines()
    if len(lines) != _HEADER_LINES + _ENERGY_ROWS:
        raise ValueError("Observed COSRAD GCR format requires 12 header lines and 50 data rows.")
    if any(not line.strip() for line in lines):
        raise ValueError("COSRAD GCR output must not contain blank header or data lines.")
    scenario = _scenario(lines)
    if lines[8].strip() != "GCR PARTICLE SPECTRUM in orbit":
        raise ValueError("COSRAD output must identify an orbital GCR particle spectrum.")
    temporal = _TEMPORAL_HEADERS.get(lines[9].strip())
    if temporal is None:
        raise ValueError("Unrecognized COSRAD temporal statistic or inconsistent native units.")
    if lines[10].strip() not in {"Energy [MeV/nuclon]", "1 col. - Energy [MeV/nuclon]"}:
        raise ValueError("COSRAD energy column must be in MeV/nuclon.")
    labels = _ion_labels(lines[11])
    energies: list[float] = []
    rows: list[tuple[float, ...]] = []
    for line_number, line in enumerate(lines[_HEADER_LINES:], start=_HEADER_LINES + 1):
        tokens = line.split()
        if len(tokens) != 1 + _ION_COUNT:
            raise ValueError(f"COSRAD data line {line_number} must contain exactly 93 columns.")
        energy = _finite_number(tokens[0], f"energy at line {line_number}")
        if energy <= 0.0 or (energies and energy <= energies[-1]):
            raise ValueError("COSRAD energies must be positive and strictly increasing.")
        values = tuple(_finite_number(token, f"flux at line {line_number}") for token in tokens[1:])
        if any(value < 0.0 for value in values):
            raise ValueError(f"COSRAD flux values at line {line_number} must be non-negative.")
        energies.append(energy)
        rows.append(values)
    statistic, units = temporal
    return CosradGcrTextTable(
        source_name=source_name,
        source_sha256=hashlib.sha256(data).hexdigest(),
        raw_header="".join(text.splitlines(keepends=True)[:_HEADER_LINES]),
        scenario=scenario,
        statistic=statistic,
        native_units=units,
        native_ion_labels=labels,
        raw_data_lines=tuple(lines[_HEADER_LINES:]),
        energies_mev_per_nucleon=tuple(energies),
        values_by_energy_z1_to_92=tuple(rows),
    )


def load_cosrad_gcr_text(path: Path) -> CosradGcrTextTable:
    """Read one file without modifying it or relying on its filename for semantics."""
    return parse_cosrad_gcr_text(path.read_bytes(), source_name=str(path))
