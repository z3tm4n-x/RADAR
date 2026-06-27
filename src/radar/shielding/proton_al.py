"""Primary proton CSDA transport through spherical aluminium shielding."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import exp, isfinite, log
from pathlib import Path
from typing import Mapping

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit

PROTON_AL_PRIMARY_CSDA_MODEL = "proton_primary_al_spherical_csda"

_ENERGY_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
)

_REQUIRED_COLUMNS = {
    "energy_mev",
    "stopping_mev_cm2_g",
    "range_g_cm2",
}


@dataclass(frozen=True)
class ProtonAlRangeEnergyTable:
    """Range-energy and stopping table for protons in aluminium."""

    energy_mev: tuple[float, ...]
    range_g_cm2: tuple[float, ...]
    stopping_mev_cm2_g: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_equal_length(self.energy_mev, self.range_g_cm2, "energy/range")
        _validate_equal_length(self.energy_mev, self.stopping_mev_cm2_g, "energy/stopping")

        if len(self.energy_mev) < 2:
            msg = "Proton Al table must contain at least two energy points."
            raise ValueError(msg)

        _validate_positive_increasing(self.energy_mev, "energy")
        _validate_nonnegative_increasing(self.range_g_cm2, "range")
        _validate_positive(self.stopping_mev_cm2_g, "stopping")

    @property
    def max_range_g_cm2(self) -> float:
        return self.range_g_cm2[-1]

    def range_at_energy(self, energy_mev: float) -> float:
        """Return aluminium mass range at proton kinetic energy."""

        return _interpolate_strict(
            x=energy_mev,
            xp=self.energy_mev,
            fp=self.range_g_cm2,
            quantity="proton Al energy",
        )

    def energy_at_range(self, range_g_cm2: float) -> float:
        """Return proton kinetic energy corresponding to aluminium mass range."""

        return _interpolate_strict(
            x=range_g_cm2,
            xp=self.range_g_cm2,
            fp=self.energy_mev,
            quantity="proton Al range",
        )

    def stopping_at_energy(self, energy_mev: float) -> float:
        """Return proton mass stopping power in aluminium."""

        return _interpolate_strict(
            x=energy_mev,
            xp=self.energy_mev,
            fp=self.stopping_mev_cm2_g,
            quantity="proton Al stopping",
        )


def _validate_equal_length(
    left: tuple[float, ...],
    right: tuple[float, ...],
    name: str,
) -> None:
    if len(left) != len(right):
        msg = f"Proton Al table {name} arrays must have equal length."
        raise ValueError(msg)


def _validate_finite(values: tuple[float, ...], name: str) -> None:
    if any(not isfinite(value) for value in values):
        msg = f"Proton Al table {name} values must be finite."
        raise ValueError(msg)


def _validate_positive(values: tuple[float, ...], name: str) -> None:
    _validate_finite(values, name)

    if any(value <= 0.0 for value in values):
        msg = f"Proton Al table {name} values must be positive."
        raise ValueError(msg)


def _validate_positive_increasing(values: tuple[float, ...], name: str) -> None:
    _validate_positive(values, name)

    if any(right <= left for left, right in zip(values, values[1:])):
        msg = f"Proton Al table {name} values must be strictly increasing."
        raise ValueError(msg)


def _validate_nonnegative_increasing(values: tuple[float, ...], name: str) -> None:
    _validate_finite(values, name)

    if any(value < 0.0 for value in values):
        msg = f"Proton Al table {name} values must be non-negative."
        raise ValueError(msg)

    if any(right <= left for left, right in zip(values, values[1:])):
        msg = f"Proton Al table {name} values must be strictly increasing."
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

    return _interpolate_strict(x=x, xp=xp, fp=fp, quantity="proton spectrum")


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


def _validate_spectrum(spectrum: Spectrum1D, table: ProtonAlRangeEnergyTable) -> None:
    if spectrum.source not in (
        RadiationSource.SEP,
        RadiationSource.GCR,
        RadiationSource.ERB,
    ):
        msg = "Proton Al shielding supports SEP, GCR and ERB spectra only."
        raise ValueError(msg)

    if spectrum.particle is not Particle.PROTON:
        msg = "Proton Al shielding requires a proton spectrum."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "Proton Al shielding energy grid must be in MeV."
        raise ValueError(msg)

    if spectrum.quantity not in _ENERGY_QUANTITIES:
        msg = "Proton Al shielding requires an energy fluence or flux spectrum."
        raise ValueError(msg)

    if len(spectrum.x) != len(spectrum.y):
        msg = "Proton Al shielding spectrum x/y arrays must have equal length."
        raise ValueError(msg)

    if len(spectrum.x) < 2:
        msg = "Proton Al shielding spectrum must contain at least two energy points."
        raise ValueError(msg)

    if any(value < 0.0 for value in spectrum.y):
        msg = "Proton Al shielding spectrum values must be non-negative."
        raise ValueError(msg)

    edges = log_bin_edges_from_centers(spectrum.x)
    _ = table.range_at_energy(edges[0])
    _ = table.range_at_energy(edges[-1])


def shield_proton_spectrum_primary_through_al(
    *,
    spectrum: Spectrum1D,
    table: ProtonAlRangeEnergyTable,
    thickness_g_cm2: float,
    model: str | None = None,
) -> Spectrum1D:
    """Return primary proton spectrum behind spherical aluminium shielding."""

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
        output_model = f"{spectrum.model}+{PROTON_AL_PRIMARY_CSDA_MODEL}"

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


def _required_text(
    row: Mapping[str, str | None],
    column: str,
    line_number: int,
) -> str:
    value = row.get(column)

    if value is None or not value.strip():
        msg = f"Missing required column {column!r} at proton Al CSV line {line_number}."
        raise ValueError(msg)

    return value.strip()


def load_proton_al_range_energy_table(path: str | Path) -> ProtonAlRangeEnergyTable:
    """Load normalized proton aluminium range-energy CSV table."""

    csv_path = Path(path)
    rows: list[tuple[float, float, float]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = _REQUIRED_COLUMNS - fieldnames

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            msg = f"Proton Al CSV is missing columns: {missing}."
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

    return ProtonAlRangeEnergyTable(
        energy_mev=tuple(row[0] for row in rows),
        range_g_cm2=tuple(row[1] for row in rows),
        stopping_mev_cm2_g=tuple(row[2] for row in rows),
    )
