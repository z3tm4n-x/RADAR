"""Primary HZE CSDA transport through spherical aluminium shielding."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import exp, isfinite, log
from pathlib import Path
from typing import Mapping

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit

HZE_AL_PRIMARY_CSDA_MODEL = "hze_primary_al_spherical_csda"
HZE_AL_NO_FRAGMENTATION_NOTE = "hze_nuclear_attenuation_and_fragmentation_not_included"

_ALLOWED_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
)

_REQUIRED_COLUMNS = {
    "z",
    "symbol",
    "mass_number",
    "mass_to_charge",
    "energy_mev_per_nucleon",
    "range_g_cm2",
}


@dataclass(frozen=True)
class HzeAlRangeTable:
    """Range-energy table for one HZE ion in aluminium."""

    z: int
    symbol: str
    mass_number: float
    mass_to_charge: float
    energy_mev_per_nucleon: tuple[float, ...]
    range_g_cm2: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.z < 2:
            msg = "HZE Al table ion charge Z must be at least 2."
            raise ValueError(msg)

        if not self.symbol:
            msg = "HZE Al table ion symbol must not be empty."
            raise ValueError(msg)

        if self.mass_number <= 0.0:
            msg = "HZE Al table mass number must be positive."
            raise ValueError(msg)

        if self.mass_to_charge <= 0.0:
            msg = "HZE Al table A/Z must be positive."
            raise ValueError(msg)

        if len(self.energy_mev_per_nucleon) != len(self.range_g_cm2):
            msg = "HZE Al table energy and range arrays must have equal length."
            raise ValueError(msg)

        if len(self.energy_mev_per_nucleon) < 2:
            msg = "HZE Al table must contain at least two energy points."
            raise ValueError(msg)

        _validate_positive_increasing(self.energy_mev_per_nucleon, "energy")
        _validate_nonnegative_increasing(self.range_g_cm2, "range")

    @property
    def max_range_g_cm2(self) -> float:
        return self.range_g_cm2[-1]

    def range_at_energy(self, energy_mev_per_nucleon: float) -> float:
        """Return aluminium mass range at ion kinetic energy per nucleon."""

        return _interpolate_strict(
            x=energy_mev_per_nucleon,
            xp=self.energy_mev_per_nucleon,
            fp=self.range_g_cm2,
            quantity="HZE Al energy",
        )

    def energy_at_range(self, range_g_cm2: float) -> float:
        """Return ion kinetic energy per nucleon at aluminium mass range."""

        return _interpolate_strict(
            x=range_g_cm2,
            xp=self.range_g_cm2,
            fp=self.energy_mev_per_nucleon,
            quantity="HZE Al range",
        )


def _validate_finite(values: tuple[float, ...], name: str) -> None:
    if any(not isfinite(value) for value in values):
        msg = f"HZE Al table {name} values must be finite."
        raise ValueError(msg)


def _validate_positive_increasing(values: tuple[float, ...], name: str) -> None:
    _validate_finite(values, name)

    if any(value <= 0.0 for value in values):
        msg = f"HZE Al table {name} values must be positive."
        raise ValueError(msg)

    if any(right <= left for left, right in zip(values, values[1:])):
        msg = f"HZE Al table {name} values must be strictly increasing."
        raise ValueError(msg)


def _validate_nonnegative_increasing(values: tuple[float, ...], name: str) -> None:
    _validate_finite(values, name)

    if any(value < 0.0 for value in values):
        msg = f"HZE Al table {name} values must be non-negative."
        raise ValueError(msg)

    if any(right <= left for left, right in zip(values, values[1:])):
        msg = f"HZE Al table {name} values must be strictly increasing."
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


def _interpolate_zero_outside(
    *,
    x: float,
    xp: tuple[float, ...],
    fp: tuple[float, ...],
) -> float:
    if x < xp[0] or x > xp[-1]:
        return 0.0

    return _interpolate_strict(x=x, xp=xp, fp=fp, quantity="HZE spectrum")


def log_bin_edges_from_centers(centers: tuple[float, ...]) -> tuple[float, ...]:
    """Return logarithmic bin edges for strictly positive bin centers."""

    if len(centers) < 2:
        msg = "At least two energy centers are required to build log bin edges."
        raise ValueError(msg)

    if any(center <= 0.0 for center in centers):
        msg = "Logarithmic bin centers must be positive."
        raise ValueError(msg)

    if any(right <= left for left, right in zip(centers, centers[1:])):
        msg = "Logarithmic bin centers must be strictly increasing."
        raise ValueError(msg)

    log_centers = tuple(log(center) for center in centers)
    edges: list[float] = [0.0] * (len(centers) + 1)

    for index in range(1, len(centers)):
        edges[index] = exp(0.5 * (log_centers[index - 1] + log_centers[index]))

    edges[0] = exp(log_centers[0] - 0.5 * (log_centers[1] - log_centers[0]))
    edges[-1] = exp(log_centers[-1] + 0.5 * (log_centers[-1] - log_centers[-2]))

    return tuple(edges)


def _validate_spectrum(spectrum: Spectrum1D, table: HzeAlRangeTable) -> None:
    if spectrum.source not in (RadiationSource.SEP, RadiationSource.GCR):
        msg = "HZE Al shielding supports SEP and GCR spectra only."
        raise ValueError(msg)

    if spectrum.particle is not Particle.HZE:
        msg = "HZE Al shielding requires an HZE spectrum."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "HZE Al shielding energy grid must be in MeV per nucleon encoded as Unit.MEV."
        raise ValueError(msg)

    if spectrum.quantity not in _ALLOWED_QUANTITIES:
        msg = "HZE Al shielding requires an energy fluence or flux spectrum."
        raise ValueError(msg)

    if len(spectrum.x) != len(spectrum.y):
        msg = "HZE Al shielding spectrum x/y arrays must have equal length."
        raise ValueError(msg)

    if len(spectrum.x) < 2:
        msg = "HZE Al shielding spectrum must contain at least two energy points."
        raise ValueError(msg)

    if any(value < 0.0 for value in spectrum.y):
        msg = "HZE Al shielding spectrum values must be non-negative."
        raise ValueError(msg)

    edges = log_bin_edges_from_centers(spectrum.x)
    _ = table.range_at_energy(edges[0])
    _ = table.range_at_energy(edges[-1])


def shield_hze_spectrum_primary_through_al(
    *,
    spectrum: Spectrum1D,
    table: HzeAlRangeTable,
    thickness_g_cm2: float,
    model: str | None = None,
) -> Spectrum1D:
    """Return primary HZE spectrum behind spherical aluminium shielding.

    The calculation is primary-ion CSDA only. HZE nuclear attenuation and
    fragmentation are intentionally not included in this model.
    """

    if not isfinite(thickness_g_cm2):
        msg = "Al shielding thickness must be finite."
        raise ValueError(msg)

    if thickness_g_cm2 < 0.0:
        msg = "Al shielding thickness must be non-negative."
        raise ValueError(msg)

    _validate_spectrum(spectrum, table)

    edges = log_bin_edges_from_centers(spectrum.x)
    shielded_values: list[float] = []

    for index, energy_center in enumerate(spectrum.x):
        energy_lo = edges[index]
        energy_hi = edges[index + 1]
        output_width = energy_hi - energy_lo

        range_lo = table.range_at_energy(energy_lo)
        range_hi = table.range_at_energy(energy_hi)
        range_center = table.range_at_energy(energy_center)

        input_range_lo = range_lo + thickness_g_cm2
        input_range_hi = range_hi + thickness_g_cm2
        input_range_center = range_center + thickness_g_cm2

        if input_range_hi > table.max_range_g_cm2:
            shielded_values.append(0.0)
            continue

        input_energy_lo = table.energy_at_range(input_range_lo)
        input_energy_hi = table.energy_at_range(input_range_hi)
        input_energy_center = table.energy_at_range(input_range_center)
        input_width = input_energy_hi - input_energy_lo

        source_value = _interpolate_zero_outside(
            x=input_energy_center,
            xp=spectrum.x,
            fp=spectrum.y,
        )
        shielded_values.append(source_value * input_width / output_width)

    output_model = model
    if output_model is None:
        output_model = (
            f"{spectrum.model}+Z{table.z}_{table.symbol}+"
            f"{HZE_AL_PRIMARY_CSDA_MODEL}+{HZE_AL_NO_FRAGMENTATION_NOTE}"
        )

    return Spectrum1D(
        x=spectrum.x,
        y=tuple(shielded_values),
        x_unit=spectrum.x_unit,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=spectrum.source,
        model=output_model,
    )


def _required_text(row: Mapping[str, str | None], column: str, line_number: int) -> str:
    value = row.get(column)

    if value is None or not value.strip():
        msg = f"Missing required column {column!r} at HZE Al CSV line {line_number}."
        raise ValueError(msg)

    return value.strip()


def load_hze_al_range_tables(path: str | Path) -> dict[int, HzeAlRangeTable]:
    """Load normalized HZE aluminium range-energy CSV tables."""

    rows_by_z: dict[int, list[tuple[float, float, str, float, float]]] = {}

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = _REQUIRED_COLUMNS - fieldnames

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            msg = f"HZE Al CSV is missing columns: {missing}."
            raise ValueError(msg)

        for line_number, row in enumerate(reader, start=2):
            z = int(_required_text(row, "z", line_number))
            rows_by_z.setdefault(z, []).append(
                (
                    float(_required_text(row, "energy_mev_per_nucleon", line_number)),
                    float(_required_text(row, "range_g_cm2", line_number)),
                    _required_text(row, "symbol", line_number),
                    float(_required_text(row, "mass_number", line_number)),
                    float(_required_text(row, "mass_to_charge", line_number)),
                )
            )

    tables: dict[int, HzeAlRangeTable] = {}

    for z, rows in rows_by_z.items():
        rows.sort(key=lambda row: row[0])
        first = rows[0]
        tables[z] = HzeAlRangeTable(
            z=z,
            symbol=first[2],
            mass_number=first[3],
            mass_to_charge=first[4],
            energy_mev_per_nucleon=tuple(row[0] for row in rows),
            range_g_cm2=tuple(row[1] for row in rows),
        )

    return tables
