"""SEP proton coefficient tables and spectrum formula."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

PROTON_REST_ENERGY_MEV = 938.0

SEP_PROTON_EVENT_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
SEP_PROTON_PROBABILITIES = (0.9, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001)


class SepProtonSpectrumProduct(StrEnum):
    """SEP proton spectrum products represented by coefficient tables."""

    FLUENCE = "fluence"
    PEAK_FLUX = "peak_flux"


class SepProtonCoefficientName(StrEnum):
    """Coefficient names for the SEP proton spectrum formula."""

    LOG10_C = "log10_c"
    BREAK_ENERGY_MEV = "break_energy_mev"
    GAMMA1 = "gamma1"
    GAMMA2 = "gamma2"


@dataclass(frozen=True)
class SepProtonCoefficientRecord:
    """One transferred coefficient table cell."""

    model: str
    product: SepProtonSpectrumProduct
    parameter: SepProtonCoefficientName
    event_count: int
    probability: float
    value: float | None
    source_table: str


@dataclass(frozen=True)
class SepProtonSpectrumCoefficients:
    """Coefficients defining one SEP proton spectrum."""

    log10_c: float
    break_energy_mev: float
    gamma1: float
    gamma2: float

    def __post_init__(self) -> None:
        if self.break_energy_mev <= 0.0:
            msg = "SEP proton break energy must be positive."
            raise ValueError(msg)

        if self.gamma1 <= 0.0 or self.gamma2 <= 0.0:
            msg = "SEP proton spectral indices must be positive."
            raise ValueError(msg)


_REQUIRED_COLUMNS = {
    "model",
    "product",
    "parameter",
    "n",
    "probability",
    "value",
    "source_table",
}

_REQUIRED_PARAMETERS = (
    SepProtonCoefficientName.LOG10_C,
    SepProtonCoefficientName.BREAK_ENERGY_MEV,
    SepProtonCoefficientName.GAMMA1,
    SepProtonCoefficientName.GAMMA2,
)


def _required_text(
    row: Mapping[str, str | None],
    column: str,
    line_number: int,
) -> str:
    value = row.get(column)

    if value is None or not value.strip():
        msg = f"Missing required column {column!r} at coefficient CSV line {line_number}."
        raise ValueError(msg)

    return value.strip()


def _optional_float(text: str) -> float | None:
    stripped = text.strip()

    if not stripped:
        return None

    return float(stripped.replace(",", "."))


def _parse_record(
    row: Mapping[str, str | None],
    line_number: int,
) -> SepProtonCoefficientRecord:
    return SepProtonCoefficientRecord(
        model=_required_text(row, "model", line_number),
        product=SepProtonSpectrumProduct(_required_text(row, "product", line_number)),
        parameter=SepProtonCoefficientName(_required_text(row, "parameter", line_number)),
        event_count=int(_required_text(row, "n", line_number)),
        probability=float(_required_text(row, "probability", line_number)),
        value=_optional_float(row.get("value") or ""),
        source_table=_required_text(row, "source_table", line_number),
    )


def load_sep_proton_coefficient_records(
    path: str | Path,
) -> tuple[SepProtonCoefficientRecord, ...]:
    """Load SEP proton coefficient records from a CSV file."""

    records: list[SepProtonCoefficientRecord] = []
    csv_path = Path(path)

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = _REQUIRED_COLUMNS - fieldnames

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            msg = f"SEP proton coefficient CSV is missing columns: {missing}."
            raise ValueError(msg)

        for line_number, row in enumerate(reader, start=2):
            records.append(_parse_record(row, line_number))

    return tuple(records)


def _records_for_exact_point(
    records: tuple[SepProtonCoefficientRecord, ...],
    *,
    model: str,
    product: SepProtonSpectrumProduct,
    event_count: int,
    probability: float,
) -> tuple[SepProtonCoefficientRecord, ...]:
    return tuple(
        record
        for record in records
        if record.model == model
        and record.product is product
        and record.event_count == event_count
        and math.isclose(record.probability, probability, rel_tol=0.0, abs_tol=1e-12)
    )


def _required_parameter_value(
    point_records: tuple[SepProtonCoefficientRecord, ...],
    parameter: SepProtonCoefficientName,
) -> float:
    parameter_records = tuple(record for record in point_records if record.parameter is parameter)

    if not parameter_records:
        msg = f"Missing SEP proton coefficient row for parameter {parameter.value}."
        raise ValueError(msg)

    if len(parameter_records) > 1:
        msg = f"Duplicate SEP proton coefficient row for parameter {parameter.value}."
        raise ValueError(msg)

    record = parameter_records[0]

    if record.value is None:
        msg = f"SEP proton coefficient {parameter.value} is unavailable at this table point."
        raise ValueError(msg)

    return record.value


def lookup_sep_proton_coefficients_exact(
    records: tuple[SepProtonCoefficientRecord, ...],
    *,
    model: str,
    product: SepProtonSpectrumProduct,
    event_count: int,
    probability: float,
) -> SepProtonSpectrumCoefficients:
    """Return coefficients for an exact tabular SEP proton model point."""

    point_records = _records_for_exact_point(
        records,
        model=model,
        product=product,
        event_count=event_count,
        probability=probability,
    )

    return SepProtonSpectrumCoefficients(
        log10_c=_required_parameter_value(point_records, SepProtonCoefficientName.LOG10_C),
        break_energy_mev=_required_parameter_value(
            point_records,
            SepProtonCoefficientName.BREAK_ENERGY_MEV,
        ),
        gamma1=_required_parameter_value(point_records, SepProtonCoefficientName.GAMMA1),
        gamma2=_required_parameter_value(point_records, SepProtonCoefficientName.GAMMA2),
    )


def proton_momentum_mev(energy_mev: float) -> float:
    """Return proton momentum term p(E) in MeV for the SEP spectrum formula."""

    if energy_mev <= 0.0:
        msg = "SEP proton kinetic energy must be positive."
        raise ValueError(msg)

    return math.sqrt(energy_mev * (energy_mev + 2.0 * PROTON_REST_ENERGY_MEV))


def evaluate_sep_proton_spectrum_value(
    energy_mev: float,
    coefficients: SepProtonSpectrumCoefficients,
) -> float:
    """Evaluate SEP proton differential spectrum at one energy."""

    if energy_mev <= 0.0:
        msg = "SEP proton kinetic energy must be positive."
        raise ValueError(msg)

    c_value = float(10.0**coefficients.log10_c)

    if energy_mev < coefficients.break_energy_mev:
        energy_ratio = energy_mev / coefficients.break_energy_mev
        return float(c_value * energy_ratio ** (-coefficients.gamma1))

    momentum = proton_momentum_mev(energy_mev)
    break_momentum = proton_momentum_mev(coefficients.break_energy_mev)
    momentum_ratio = momentum / break_momentum

    return float(c_value * momentum_ratio ** (-2.0 * coefficients.gamma2))


def evaluate_sep_proton_spectrum(
    energy_grid_mev: tuple[float, ...],
    coefficients: SepProtonSpectrumCoefficients,
) -> tuple[float, ...]:
    """Evaluate SEP proton differential spectrum on an energy grid."""

    if not energy_grid_mev:
        msg = "SEP proton energy grid must not be empty."
        raise ValueError(msg)

    return tuple(
        evaluate_sep_proton_spectrum_value(
            energy_mev=energy,
            coefficients=coefficients,
        )
        for energy in energy_grid_mev
    )
