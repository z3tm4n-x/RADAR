"""HZE LET spectrum construction in silicon."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import exp, isfinite, log
from pathlib import Path
from typing import Mapping

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit

HZE_SI_LET_MODEL = "hze_si_let_histogram"

_REQUIRED_COLUMNS = {
    "z",
    "symbol",
    "mass_number",
    "mass_to_charge",
    "energy_mev_per_nucleon",
    "let_electronic_mev_cm2_mg",
    "let_nuclear_mev_cm2_mg",
    "let_total_mev_cm2_mg",
}

_FLUENCE_QUANTITIES = (SpectrumQuantity.DIFFERENTIAL_FLUENCE,)
_FLUX_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
)


@dataclass(frozen=True)
class HzeSiLetTable:
    """Silicon LET table for one HZE ion."""

    z: int
    symbol: str
    mass_number: float
    mass_to_charge: float
    energy_mev_per_nucleon: tuple[float, ...]
    let_total_mev_cm2_mg: tuple[float, ...]
    let_electronic_mev_cm2_mg: tuple[float, ...]
    let_nuclear_mev_cm2_mg: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.energy_mev_per_nucleon)

        if self.z < 2:
            msg = "HZE Si LET table ion charge Z must be at least 2."
            raise ValueError(msg)

        if not self.symbol:
            msg = "HZE Si LET table ion symbol must not be empty."
            raise ValueError(msg)

        if self.mass_number <= 0.0:
            msg = "HZE Si LET table mass number must be positive."
            raise ValueError(msg)

        if self.mass_to_charge <= 0.0:
            msg = "HZE Si LET table A/Z must be positive."
            raise ValueError(msg)

        if n < 2:
            msg = "HZE Si LET table must contain at least two points."
            raise ValueError(msg)

        for values, name in (
            (self.let_total_mev_cm2_mg, "total LET"),
            (self.let_electronic_mev_cm2_mg, "electronic LET"),
            (self.let_nuclear_mev_cm2_mg, "nuclear LET"),
        ):
            if len(values) != n:
                msg = f"HZE Si {name} array must match energy array length."
                raise ValueError(msg)

        _validate_positive_increasing(self.energy_mev_per_nucleon, "energy")
        _validate_positive(self.let_total_mev_cm2_mg, "total LET")
        _validate_nonnegative(self.let_electronic_mev_cm2_mg, "electronic LET")
        _validate_nonnegative(self.let_nuclear_mev_cm2_mg, "nuclear LET")

    def total_let_at_energy(self, energy_mev_per_nucleon: float) -> float:
        """Return total LET in MeV cm²/mg at ion kinetic energy per nucleon."""

        return _interpolate_strict(
            x=energy_mev_per_nucleon,
            xp=self.energy_mev_per_nucleon,
            fp=self.let_total_mev_cm2_mg,
            quantity="HZE Si total LET",
        )


def _validate_finite(values: tuple[float, ...], name: str) -> None:
    if any(not isfinite(value) for value in values):
        msg = f"HZE Si LET table {name} values must be finite."
        raise ValueError(msg)


def _validate_positive(values: tuple[float, ...], name: str) -> None:
    _validate_finite(values, name)

    if any(value <= 0.0 for value in values):
        msg = f"HZE Si LET table {name} values must be positive."
        raise ValueError(msg)


def _validate_nonnegative(values: tuple[float, ...], name: str) -> None:
    _validate_finite(values, name)

    if any(value < 0.0 for value in values):
        msg = f"HZE Si LET table {name} values must be non-negative."
        raise ValueError(msg)


def _validate_positive_increasing(values: tuple[float, ...], name: str) -> None:
    _validate_positive(values, name)

    if any(right <= left for left, right in zip(values, values[1:])):
        msg = f"HZE Si LET table {name} values must be strictly increasing."
        raise ValueError(msg)


def _interpolate_strict(
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

    for index in range(len(xp) - 1):
        left_x = xp[index]
        right_x = xp[index + 1]

        if left_x <= x <= right_x:
            fraction = (x - left_x) / (right_x - left_x)
            return fp[index] + fraction * (fp[index + 1] - fp[index])

    msg = f"Could not interpolate {quantity}."
    raise ValueError(msg)


def _log_edges_from_centers(centers: tuple[float, ...]) -> tuple[float, ...]:
    if len(centers) < 2:
        msg = "At least two centers are required to build logarithmic edges."
        raise ValueError(msg)

    if any(center <= 0.0 for center in centers):
        msg = "Logarithmic centers must be positive."
        raise ValueError(msg)

    if any(right <= left for left, right in zip(centers, centers[1:])):
        msg = "Logarithmic centers must be strictly increasing."
        raise ValueError(msg)

    log_centers = tuple(log(center) for center in centers)
    edges: list[float] = [0.0] * (len(centers) + 1)

    for index in range(1, len(centers)):
        edges[index] = exp(0.5 * (log_centers[index - 1] + log_centers[index]))

    edges[0] = exp(log_centers[0] - 0.5 * (log_centers[1] - log_centers[0]))
    edges[-1] = exp(log_centers[-1] + 0.5 * (log_centers[-1] - log_centers[-2]))

    return tuple(edges)


def _default_let_centers(let_values: tuple[float, ...]) -> tuple[float, ...]:
    min_let = min(let_values)
    max_let = max(let_values)

    if min_let <= 0.0:
        msg = "LET values must be positive."
        raise ValueError(msg)

    if min_let == max_let:
        return (min_let / 2.0, min_let, min_let * 2.0)

    n = len(let_values)
    log_min = log(min_let)
    log_max = log(max_let)

    return tuple(exp(log_min + (log_max - log_min) * index / (n - 1)) for index in range(n))


def _locate_bin(value: float, edges: tuple[float, ...]) -> int | None:
    if value < edges[0] or value > edges[-1]:
        return None

    if value == edges[-1]:
        return len(edges) - 2

    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index

    return None


def _output_quantity_and_unit(quantity: SpectrumQuantity) -> tuple[SpectrumQuantity, Unit]:
    if quantity in _FLUENCE_QUANTITIES:
        return SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE, Unit.DIFFERENTIAL_LET_FLUENCE

    if quantity in _FLUX_QUANTITIES:
        return SpectrumQuantity.LET_DIFFERENTIAL_FLUX, Unit.DIFFERENTIAL_LET_FLUX

    msg = "HZE LET conversion requires an energy fluence or flux spectrum."
    raise ValueError(msg)


def hze_energy_spectrum_to_let_spectrum(
    *,
    spectrum: Spectrum1D,
    table: HzeSiLetTable,
    let_centers: tuple[float, ...] | None = None,
    model: str | None = None,
) -> Spectrum1D:
    """Convert shielded HZE energy spectrum to a LET spectrum in silicon."""

    if spectrum.source is not RadiationSource.SEP:
        msg = "HZE LET conversion currently supports SEP spectra only."
        raise ValueError(msg)

    if spectrum.particle is not Particle.HZE:
        msg = "HZE LET conversion requires an HZE spectrum."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "HZE LET conversion requires MeV/nucleon encoded as Unit.MEV."
        raise ValueError(msg)

    energy_edges = _log_edges_from_centers(spectrum.x)
    let_values = tuple(table.total_let_at_energy(energy) for energy in spectrum.x)

    output_centers = let_centers
    if output_centers is None:
        output_centers = _default_let_centers(let_values)

    output_edges = _log_edges_from_centers(output_centers)
    output_counts = [0.0 for _ in output_centers]

    for index, let_value in enumerate(let_values):
        output_index = _locate_bin(let_value, output_edges)

        if output_index is None:
            continue

        energy_width = energy_edges[index + 1] - energy_edges[index]
        output_counts[output_index] += spectrum.y[index] * energy_width

    output_values = tuple(
        count / (output_edges[index + 1] - output_edges[index])
        for index, count in enumerate(output_counts)
    )
    output_quantity, output_unit = _output_quantity_and_unit(spectrum.quantity)

    return Spectrum1D(
        x=output_centers,
        y=output_values,
        x_unit=Unit.LET,
        y_unit=output_unit,
        quantity=output_quantity,
        particle=Particle.HZE,
        source=spectrum.source,
        model=model or f"{spectrum.model}+Z{table.z}_{table.symbol}+{HZE_SI_LET_MODEL}",
    )


def _required_text(row: Mapping[str, str | None], column: str, line_number: int) -> str:
    value = row.get(column)

    if value is None or not value.strip():
        msg = f"Missing required column {column!r} at HZE Si LET CSV line {line_number}."
        raise ValueError(msg)

    return value.strip()


def load_hze_si_let_tables(path: str | Path) -> dict[int, HzeSiLetTable]:
    """Load normalized HZE silicon LET CSV tables."""

    rows_by_z: dict[int, list[tuple[float, float, float, float, str, float, float]]] = {}

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = _REQUIRED_COLUMNS - fieldnames

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            msg = f"HZE Si LET CSV is missing columns: {missing}."
            raise ValueError(msg)

        for line_number, row in enumerate(reader, start=2):
            z = int(_required_text(row, "z", line_number))
            rows_by_z.setdefault(z, []).append(
                (
                    float(_required_text(row, "energy_mev_per_nucleon", line_number)),
                    float(_required_text(row, "let_total_mev_cm2_mg", line_number)),
                    float(_required_text(row, "let_electronic_mev_cm2_mg", line_number)),
                    float(_required_text(row, "let_nuclear_mev_cm2_mg", line_number)),
                    _required_text(row, "symbol", line_number),
                    float(_required_text(row, "mass_number", line_number)),
                    float(_required_text(row, "mass_to_charge", line_number)),
                )
            )

    tables: dict[int, HzeSiLetTable] = {}

    for z, rows in rows_by_z.items():
        rows.sort(key=lambda row: row[0])
        first = rows[0]
        tables[z] = HzeSiLetTable(
            z=z,
            symbol=first[4],
            mass_number=first[5],
            mass_to_charge=first[6],
            energy_mev_per_nucleon=tuple(row[0] for row in rows),
            let_total_mev_cm2_mg=tuple(row[1] for row in rows),
            let_electronic_mev_cm2_mg=tuple(row[2] for row in rows),
            let_nuclear_mev_cm2_mg=tuple(row[3] for row in rows),
        )

    return tables
