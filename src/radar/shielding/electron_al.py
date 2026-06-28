"""ESTAR electron range-energy tables for aluminium CSDA shielding."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import exp, isfinite, log
from pathlib import Path
from typing import Mapping

_REQUIRED_COLUMNS = {
    "energy_mev",
    "stopping_mev_cm2_g",
    "range_g_cm2",
}


@dataclass(frozen=True)
class ElectronAlRangeEnergyTable:
    """Range-energy and stopping table for electrons in aluminium."""

    energy_mev: tuple[float, ...]
    range_g_cm2: tuple[float, ...]
    stopping_mev_cm2_g: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_equal_length(self.energy_mev, self.range_g_cm2, "energy/range")
        _validate_equal_length(self.energy_mev, self.stopping_mev_cm2_g, "energy/stopping")

        if len(self.energy_mev) < 2:
            msg = "Electron Al table must contain at least two energy points."
            raise ValueError(msg)

        _validate_positive_increasing(self.energy_mev, "energy")
        _validate_positive_increasing(self.range_g_cm2, "range")
        _validate_positive(self.stopping_mev_cm2_g, "stopping")

    @property
    def max_range_g_cm2(self) -> float:
        return self.range_g_cm2[-1]

    def range_at_energy(self, energy_mev: float) -> float:
        """Return aluminium CSDA mass range at electron kinetic energy."""

        return _interpolate_loglog_strict(
            x=energy_mev,
            xp=self.energy_mev,
            fp=self.range_g_cm2,
            quantity="electron Al energy",
        )

    def energy_at_range(self, range_g_cm2: float) -> float:
        """Return electron kinetic energy corresponding to aluminium CSDA mass range."""

        return _interpolate_loglog_strict(
            x=range_g_cm2,
            xp=self.range_g_cm2,
            fp=self.energy_mev,
            quantity="electron Al range",
        )

    def stopping_at_energy(self, energy_mev: float) -> float:
        """Return electron mass stopping power in aluminium."""

        return _interpolate_loglog_strict(
            x=energy_mev,
            xp=self.energy_mev,
            fp=self.stopping_mev_cm2_g,
            quantity="electron Al stopping",
        )


def _validate_equal_length(
    left: tuple[float, ...],
    right: tuple[float, ...],
    name: str,
) -> None:
    if len(left) != len(right):
        msg = f"Electron Al table {name} arrays must have equal length."
        raise ValueError(msg)


def _validate_finite(values: tuple[float, ...], name: str) -> None:
    if any(not isfinite(value) for value in values):
        msg = f"Electron Al table {name} values must be finite."
        raise ValueError(msg)


def _validate_positive(values: tuple[float, ...], name: str) -> None:
    _validate_finite(values, name)

    if any(value <= 0.0 for value in values):
        msg = f"Electron Al table {name} values must be positive."
        raise ValueError(msg)


def _validate_positive_increasing(values: tuple[float, ...], name: str) -> None:
    _validate_positive(values, name)

    if any(right <= left for left, right in zip(values, values[1:])):
        msg = f"Electron Al table {name} values must be strictly increasing."
        raise ValueError(msg)


def _interpolate_loglog_strict(
    *,
    x: float,
    xp: tuple[float, ...],
    fp: tuple[float, ...],
    quantity: str,
) -> float:
    if not isfinite(x):
        msg = f"{quantity} interpolation coordinate must be finite."
        raise ValueError(msg)

    if x < xp[0] or x > xp[-1]:
        msg = f"{quantity} interpolation coordinate is outside the tabulated range."
        raise ValueError(msg)

    if x == xp[0]:
        return fp[0]

    if x == xp[-1]:
        return fp[-1]

    log_x = log(x)
    log_xp = tuple(log(value) for value in xp)
    log_fp = tuple(log(value) for value in fp)

    for index in range(len(log_xp) - 1):
        left_x = log_xp[index]
        right_x = log_xp[index + 1]

        if left_x <= log_x <= right_x:
            fraction = (log_x - left_x) / (right_x - left_x)
            return exp(log_fp[index] + fraction * (log_fp[index + 1] - log_fp[index]))

    msg = f"Could not interpolate {quantity}."
    raise ValueError(msg)


def _required_text(
    row: Mapping[str, str | None],
    column: str,
    line_number: int,
) -> str:
    value = row.get(column)

    if value is None or not value.strip():
        msg = f"Missing required column {column!r} at electron Al CSV line {line_number}."
        raise ValueError(msg)

    return value.strip()


def load_electron_al_range_energy_table(path: str | Path) -> ElectronAlRangeEnergyTable:
    """Load normalized ESTAR electron aluminium range-energy CSV table."""

    csv_path = Path(path)
    rows: list[tuple[float, float, float]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = _REQUIRED_COLUMNS - fieldnames

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            msg = f"Electron Al CSV is missing columns: {missing}."
            raise ValueError(msg)

        for line_number, row in enumerate(reader, start=2):
            rows.append(
                (
                    float(_required_text(row, "energy_mev", line_number)),
                    float(_required_text(row, "range_g_cm2", line_number)),
                    float(_required_text(row, "stopping_mev_cm2_g", line_number)),
                )
            )

    rows.sort(key=lambda row: row[0])

    return ElectronAlRangeEnergyTable(
        energy_mev=tuple(row[0] for row in rows),
        range_g_cm2=tuple(row[1] for row in rows),
        stopping_mev_cm2_g=tuple(row[2] for row in rows),
    )
