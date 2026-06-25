"""Nonelastic survival correction for primary protons in aluminium."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import exp, isfinite
from pathlib import Path
from typing import Mapping

from radar.core.spectra import Spectrum1D
from radar.shielding.proton_al import (
    PROTON_AL_PRIMARY_CSDA_MODEL,
    ProtonAlRangeEnergyTable,
    shield_proton_spectrum_primary_through_al,
)

AVOGADRO_CONSTANT_MOL_INV = 6.02214076e23
BARN_CM2 = 1.0e-24
AL27_MOLAR_MASS_G_MOL = 26.9815385
DEFAULT_AL27_NONELASTIC_MT = 5
PROTON_AL_NONELASTIC_SURVIVAL_MODEL = "proton_primary_al_nonelastic_survival"

_REQUIRED_XS_COLUMNS = {
    "mt",
    "energy_mev",
    "sigma_barn",
}


@dataclass(frozen=True)
class Al27NonelasticCrossSectionTable:
    """Selected p+Al27 reaction cross-section table."""

    mt: int
    energy_mev: tuple[float, ...]
    sigma_barn: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.mt <= 0:
            msg = "Al27 cross-section MT must be positive."
            raise ValueError(msg)

        if len(self.energy_mev) != len(self.sigma_barn):
            msg = "Al27 cross-section energy and sigma arrays must have equal length."
            raise ValueError(msg)

        if len(self.energy_mev) < 2:
            msg = "Al27 cross-section table must contain at least two points."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.energy_mev):
            msg = "Al27 cross-section energies must be finite."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.sigma_barn):
            msg = "Al27 cross-section values must be finite."
            raise ValueError(msg)

        if any(value < 0.0 for value in self.energy_mev):
            msg = "Al27 cross-section energies must be non-negative."
            raise ValueError(msg)

        if any(value < 0.0 for value in self.sigma_barn):
            msg = "Al27 cross-section values must be non-negative."
            raise ValueError(msg)

        if any(right <= left for left, right in zip(self.energy_mev, self.energy_mev[1:])):
            msg = "Al27 cross-section energies must be strictly increasing."
            raise ValueError(msg)

    def sigma_at_energy(self, energy_mev: float) -> float:
        """Return cross section in barns at proton kinetic energy in MeV."""

        return _interpolate_strict(
            x=energy_mev,
            xp=self.energy_mev,
            fp=self.sigma_barn,
            quantity="Al27 cross section",
        )


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


def _required_text(
    row: Mapping[str, str | None],
    column: str,
    line_number: int,
) -> str:
    value = row.get(column)

    if value is None or not value.strip():
        msg = f"Missing required column {column!r} at p+Al27 CSV line {line_number}."
        raise ValueError(msg)

    return value.strip()


def _merge_duplicate_energy_rows_keep_max_sigma(
    rows: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: dict[float, float] = {}

    for energy, sigma in rows:
        previous = merged.get(energy)
        if previous is None:
            merged[energy] = sigma
        else:
            merged[energy] = max(previous, sigma)

    return sorted(merged.items(), key=lambda row: row[0])


def load_al27_nonelastic_cross_section_table(
    path: str | Path,
    *,
    mt: int = DEFAULT_AL27_NONELASTIC_MT,
) -> Al27NonelasticCrossSectionTable:
    """Load selected p+Al27 MF=3 cross section from compact CSV."""

    csv_path = Path(path)
    rows: list[tuple[float, float]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = _REQUIRED_XS_COLUMNS - fieldnames

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            msg = f"p+Al27 cross-section CSV is missing columns: {missing}."
            raise ValueError(msg)

        for line_number, row in enumerate(reader, start=2):
            row_mt = int(_required_text(row, "mt", line_number))

            if row_mt != mt:
                continue

            rows.append(
                (
                    float(_required_text(row, "energy_mev", line_number)),
                    float(_required_text(row, "sigma_barn", line_number)),
                )
            )

    if not rows:
        msg = f"p+Al27 cross-section CSV contains no rows for MT={mt}."
        raise ValueError(msg)

    rows = _merge_duplicate_energy_rows_keep_max_sigma(rows)

    return Al27NonelasticCrossSectionTable(
        mt=mt,
        energy_mev=tuple(row[0] for row in rows),
        sigma_barn=tuple(row[1] for row in rows),
    )


def proton_nonelastic_survival_factor(
    *,
    output_energy_mev: float,
    input_energy_mev: float,
    stopping_table: ProtonAlRangeEnergyTable,
    cross_section_table: Al27NonelasticCrossSectionTable,
    integration_steps: int = 64,
    target_molar_mass_g_mol: float = AL27_MOLAR_MASS_G_MOL,
) -> float:
    """Return nonelastic survival factor along a CSDA proton path in Al."""

    if integration_steps < 1:
        msg = "Survival integration must use at least one step."
        raise ValueError(msg)

    if target_molar_mass_g_mol <= 0.0:
        msg = "Target molar mass must be positive."
        raise ValueError(msg)

    if input_energy_mev < output_energy_mev:
        msg = "Input energy must be greater than or equal to output energy."
        raise ValueError(msg)

    if input_energy_mev == output_energy_mev:
        return 1.0

    number_per_gram = AVOGADRO_CONSTANT_MOL_INV / target_molar_mass_g_mol
    step = (input_energy_mev - output_energy_mev) / integration_steps
    integral = 0.0

    for index in range(integration_steps + 1):
        energy = output_energy_mev + step * index
        sigma_cm2 = cross_section_table.sigma_at_energy(energy) * BARN_CM2
        stopping = stopping_table.stopping_at_energy(energy)
        weight = 0.5 if index in (0, integration_steps) else 1.0
        integral += weight * number_per_gram * sigma_cm2 / stopping

    return exp(-integral * step)


def apply_proton_nonelastic_survival_to_primary_spectrum(
    *,
    spectrum: Spectrum1D,
    stopping_table: ProtonAlRangeEnergyTable,
    cross_section_table: Al27NonelasticCrossSectionTable,
    thickness_g_cm2: float,
    integration_steps: int = 64,
    model: str | None = None,
) -> Spectrum1D:
    """Return primary proton CSDA spectrum with nonelastic survival applied."""

    primary = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=stopping_table,
        thickness_g_cm2=thickness_g_cm2,
        model=f"{spectrum.model}+{PROTON_AL_PRIMARY_CSDA_MODEL}",
    )

    if thickness_g_cm2 == 0.0:
        return primary

    survived_values: list[float] = []

    for index, output_energy in enumerate(spectrum.x):
        if primary.y[index] == 0.0:
            survived_values.append(0.0)
            continue

        output_range = stopping_table.range_at_energy(output_energy)
        input_range = output_range + thickness_g_cm2

        if input_range > stopping_table.max_range_g_cm2:
            survived_values.append(0.0)
            continue

        input_energy = stopping_table.energy_at_range(input_range)
        survival = proton_nonelastic_survival_factor(
            output_energy_mev=output_energy,
            input_energy_mev=input_energy,
            stopping_table=stopping_table,
            cross_section_table=cross_section_table,
            integration_steps=integration_steps,
        )

        survived_values.append(primary.y[index] * survival)

    output_model = model
    if output_model is None:
        output_model = f"{primary.model}+{PROTON_AL_NONELASTIC_SURVIVAL_MODEL}"

    return Spectrum1D(
        x=primary.x,
        y=tuple(survived_values),
        x_unit=primary.x_unit,
        y_unit=primary.y_unit,
        quantity=primary.quantity,
        particle=primary.particle,
        source=primary.source,
        model=output_model,
    )
