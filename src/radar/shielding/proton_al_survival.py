"""Nonelastic survival correction for primary protons in aluminium."""

from __future__ import annotations

import csv
from bisect import bisect_left
from dataclasses import dataclass
from enum import StrEnum
from math import exp, isfinite
from pathlib import Path
from typing import Mapping

from radar.core.spectra import Spectrum1D, TransformationStep
from radar.core.spectrum_ops import derive_spectrum, mark_spectrum_incomplete
from radar.nuclear.tendl import (
    TendlIncidentEnergySupport,
    assess_tendl_incident_energy_coverage,
)
from radar.shielding.proton_al import (
    PROTON_AL_PRIMARY_CSDA_MODEL,
    ProtonAlRangeEnergyTable,
    shield_proton_spectrum_primary_through_al,
)

AVOGADRO_CONSTANT_MOL_INV = 6.02214076e23
BARN_CM2 = 1.0e-24
AL27_MOLAR_MASS_G_MOL = 26.9815385
DEFAULT_AL27_NONELASTIC_MT = 3
PROTON_AL_NONELASTIC_SURVIVAL_MODEL = "proton_primary_al_nonelastic_survival"
PROTON_AL_NONELASTIC_SUPPORT_OPERATOR = "TENDL p+Al-27 nonelastic survival"

_REQUIRED_XS_COLUMNS = {
    "mt",
    "energy_mev",
    "sigma_barn",
}


class IncidentEnergyLimit(StrEnum):
    """One-sided value to use at an ENDF incident-energy discontinuity."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class Al27CrossSectionDiscontinuity:
    """One source-authentic jump represented by duplicate ENDF energy rows."""

    energy_mev: float
    left_sigma_barn: float
    right_sigma_barn: float

    def __post_init__(self) -> None:
        values = (self.energy_mev, self.left_sigma_barn, self.right_sigma_barn)
        if any(not isfinite(value) for value in values):
            msg = "Al27 cross-section discontinuity values must be finite."
            raise ValueError(msg)
        if self.energy_mev < 0.0:
            msg = "Al27 cross-section discontinuity energy must be non-negative."
            raise ValueError(msg)
        if self.left_sigma_barn < 0.0 or self.right_sigma_barn < 0.0:
            msg = "Al27 cross-section discontinuity values must be non-negative."
            raise ValueError(msg)
        if self.left_sigma_barn == self.right_sigma_barn:
            msg = "Al27 cross-section discontinuity must contain distinct limits."
            raise ValueError(msg)


@dataclass(frozen=True)
class CanonicalAl27CrossSectionRows:
    """Strictly increasing MF=3 knots plus explicit one-sided jumps."""

    energy_mev: tuple[float, ...]
    sigma_barn: tuple[float, ...]
    discontinuities: tuple[Al27CrossSectionDiscontinuity, ...]


def canonicalize_al27_cross_section_rows(
    rows: list[tuple[float, float]],
    *,
    context: str,
) -> CanonicalAl27CrossSectionRows:
    """Preserve ENDF duplicate-energy jumps without inventing a threshold ramp.

    The compact CSV retains source order. At a duplicate coordinate, the first
    distinct ordinate is the left limit and the second is the right limit.
    Equal duplicates are harmless and collapse to one knot. More than one
    change at the same coordinate is ambiguous and therefore rejected.
    """

    if not rows:
        msg = f"{context} contains no cross-section rows."
        raise ValueError(msg)

    if any(right[0] < left[0] for left, right in zip(rows, rows[1:])):
        msg = f"{context} incident energies must be non-decreasing in source order."
        raise ValueError(msg)

    energies: list[float] = []
    sigmas: list[float] = []
    discontinuities: list[Al27CrossSectionDiscontinuity] = []
    index = 0

    while index < len(rows):
        energy = rows[index][0]
        values: list[float] = []
        while index < len(rows) and rows[index][0] == energy:
            value = rows[index][1]
            if not values or value != values[-1]:
                values.append(value)
            index += 1

        if len(values) > 2:
            msg = (
                f"{context} has an ambiguous duplicate-energy sequence at "
                f"{energy!r} MeV."
            )
            raise ValueError(msg)

        right_value = values[-1]
        energies.append(energy)
        sigmas.append(right_value)
        if len(values) == 2:
            discontinuities.append(
                Al27CrossSectionDiscontinuity(
                    energy_mev=energy,
                    left_sigma_barn=values[0],
                    right_sigma_barn=right_value,
                )
            )

    return CanonicalAl27CrossSectionRows(
        energy_mev=tuple(energies),
        sigma_barn=tuple(sigmas),
        discontinuities=tuple(discontinuities),
    )


def validate_al27_cross_section_discontinuities(
    *,
    energy_mev: tuple[float, ...],
    sigma_barn: tuple[float, ...],
    discontinuities: tuple[Al27CrossSectionDiscontinuity, ...],
) -> None:
    """Validate that explicit jumps agree with the canonical right-limit knots."""

    by_energy: dict[float, Al27CrossSectionDiscontinuity] = {}
    sigma_by_energy = dict(zip(energy_mev, sigma_barn, strict=True))
    for discontinuity in discontinuities:
        if discontinuity.energy_mev in by_energy:
            msg = "Al27 cross-section discontinuity energies must be unique."
            raise ValueError(msg)
        if discontinuity.energy_mev not in sigma_by_energy:
            msg = "Al27 cross-section discontinuity energy must be a tabulated knot."
            raise ValueError(msg)
        if sigma_by_energy[discontinuity.energy_mev] != discontinuity.right_sigma_barn:
            msg = "Al27 cross-section knot must store the discontinuity right limit."
            raise ValueError(msg)
        by_energy[discontinuity.energy_mev] = discontinuity


def interpolate_al27_cross_section(
    *,
    energy_mev: float,
    xp: tuple[float, ...],
    fp: tuple[float, ...],
    discontinuities: tuple[Al27CrossSectionDiscontinuity, ...],
    limit: IncidentEnergyLimit,
    quantity: str,
) -> float:
    """Interpolate MF=3 data while retaining one-sided threshold values."""

    if not isfinite(energy_mev):
        msg = f"{quantity} interpolation coordinate must be finite."
        raise ValueError(msg)
    if energy_mev < xp[0] or energy_mev > xp[-1]:
        msg = f"{quantity} interpolation coordinate is outside the tabulated range."
        raise ValueError(msg)

    right_index = bisect_left(xp, energy_mev)
    if right_index < len(xp) and xp[right_index] == energy_mev:
        discontinuity = next(
            (
                item
                for item in discontinuities
                if item.energy_mev == energy_mev
            ),
            None,
        )
        if discontinuity is not None and limit is IncidentEnergyLimit.LEFT:
            return discontinuity.left_sigma_barn
        return fp[right_index]

    left_index = right_index - 1
    left_energy = xp[left_index]
    right_energy = xp[right_index]
    left_value = fp[left_index]
    right_discontinuity = next(
        (
            item
            for item in discontinuities
            if item.energy_mev == right_energy
        ),
        None,
    )
    right_value = (
        right_discontinuity.left_sigma_barn
        if right_discontinuity is not None
        else fp[right_index]
    )
    fraction = (energy_mev - left_energy) / (right_energy - left_energy)
    return left_value + fraction * (right_value - left_value)


@dataclass(frozen=True)
class Al27NonelasticCrossSectionTable:
    """Selected p+Al27 reaction cross-section table."""

    mt: int
    energy_mev: tuple[float, ...]
    sigma_barn: tuple[float, ...]
    discontinuities: tuple[Al27CrossSectionDiscontinuity, ...] = ()
    incident_energy_support: TendlIncidentEnergySupport | None = None

    def __post_init__(self) -> None:
        if self.mt != DEFAULT_AL27_NONELASTIC_MT:
            msg = "Al27 nonelastic survival requires the TENDL MF=3/MT=3 table."
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

        validate_al27_cross_section_discontinuities(
            energy_mev=self.energy_mev,
            sigma_barn=self.sigma_barn,
            discontinuities=self.discontinuities,
        )

        if self.incident_energy_support is None:
            object.__setattr__(
                self,
                "incident_energy_support",
                TendlIncidentEnergySupport(
                    minimum_mev=0.0,
                    maximum_mev=self.energy_mev[-1],
                    dataset_ref="tendl/p_al27_mf3_xs.csv",
                ),
            )
        else:
            support = self.incident_energy_support
            assert support is not None
            if support.maximum_mev > self.energy_mev[-1]:
                msg = "Al27 support maximum exceeds the tabulated cross-section domain."
                raise ValueError(msg)

    def sigma_at_energy(
        self,
        energy_mev: float,
        *,
        limit: IncidentEnergyLimit = IncidentEnergyLimit.RIGHT,
    ) -> float:
        """Return cross section in barns at proton kinetic energy in MeV."""

        support = self.incident_energy_support
        assert support is not None
        support.require_contains(
            energy_mev,
            operator=PROTON_AL_NONELASTIC_SUPPORT_OPERATOR,
        )

        if energy_mev < self.energy_mev[0]:
            return 0.0

        return interpolate_al27_cross_section(
            energy_mev=energy_mev,
            xp=self.energy_mev,
            fp=self.sigma_barn,
            discontinuities=self.discontinuities,
            limit=limit,
            quantity="Al27 cross section",
        )


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


def load_al27_nonelastic_cross_section_table(
    path: str | Path,
    *,
    mt: int = DEFAULT_AL27_NONELASTIC_MT,
) -> Al27NonelasticCrossSectionTable:
    """Load selected p+Al27 MF=3 cross section from compact CSV."""

    if mt != DEFAULT_AL27_NONELASTIC_MT:
        msg = "Al27 nonelastic survival must load TENDL MF=3/MT=3."
        raise ValueError(msg)

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

    canonical = canonicalize_al27_cross_section_rows(
        rows,
        context=f"p+Al27 MF=3/MT={mt}",
    )

    return Al27NonelasticCrossSectionTable(
        mt=mt,
        energy_mev=canonical.energy_mev,
        sigma_barn=canonical.sigma_barn,
        discontinuities=canonical.discontinuities,
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
    energy_knots = {
        output_energy_mev + step * index for index in range(integration_steps + 1)
    }
    energy_knots.update(
        discontinuity.energy_mev
        for discontinuity in cross_section_table.discontinuities
        if output_energy_mev < discontinuity.energy_mev < input_energy_mev
    )
    ordered_knots = tuple(sorted(energy_knots))
    integral = 0.0

    for left_energy, right_energy in zip(ordered_knots, ordered_knots[1:]):
        # At an ENDF jump the interval below must approach the left limit,
        # while the interval above must depart from the right limit. A single
        # point at the jump has zero measure and must never receive trapezoid
        # weight from the wrong one-sided branch.
        left_sigma_cm2 = cross_section_table.sigma_at_energy(
            left_energy,
            limit=IncidentEnergyLimit.RIGHT,
        ) * BARN_CM2
        right_sigma_cm2 = cross_section_table.sigma_at_energy(
            right_energy,
            limit=IncidentEnergyLimit.LEFT,
        ) * BARN_CM2
        left_integrand = (
            number_per_gram
            * left_sigma_cm2
            / stopping_table.stopping_at_energy(left_energy)
        )
        right_integrand = (
            number_per_gram
            * right_sigma_cm2
            / stopping_table.stopping_at_energy(right_energy)
        )
        integral += (
            0.5
            * (left_integrand + right_integrand)
            * (right_energy - left_energy)
        )

    return exp(-integral)


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
    support = cross_section_table.incident_energy_support
    assert support is not None
    if primary.bin_edges is None:
        msg = "Positive-thickness proton Al primary transport must declare bin edges."
        raise ValueError(msg)

    relevant_energies: list[float] = []
    source_edges = (
        spectrum.bin_edges
        if spectrum.bin_edges is not None
        else primary.bin_edges
    )
    for index, (energy, value) in enumerate(
        zip(spectrum.x, spectrum.y, strict=True),
    ):
        if value <= 0.0:
            continue
        relevant_energies.extend(
            (source_edges[index], energy, source_edges[index + 1])
        )

    for index, output_energy in enumerate(spectrum.x):
        if primary.y[index] == 0.0:
            survived_values.append(0.0)
            continue

        declared_output_lo = primary.bin_edges[index]
        declared_output_hi = primary.bin_edges[index + 1]
        output_lo = max(declared_output_lo, stopping_table.energy_mev[0])
        output_hi = min(declared_output_hi, stopping_table.energy_mev[-1])
        if output_hi <= output_lo:
            survived_values.append(primary.y[index])
            continue
        mapped_range_lo = (
            stopping_table.range_at_energy(output_lo) + thickness_g_cm2
        )
        mapped_range_hi = (
            stopping_table.range_at_energy(output_hi) + thickness_g_cm2
        )
        supported_range_lo = max(mapped_range_lo, stopping_table.range_g_cm2[0])
        supported_range_hi = min(mapped_range_hi, stopping_table.max_range_g_cm2)
        if supported_range_hi <= supported_range_lo:
            survived_values.append(primary.y[index])
            continue
        input_lo = max(
            stopping_table.energy_at_range(supported_range_lo),
            spectrum.x[0],
        )
        input_hi = min(
            stopping_table.energy_at_range(supported_range_hi),
            spectrum.x[-1],
        )
        if input_hi <= input_lo:
            survived_values.append(primary.y[index])
            continue
        # Primary CSDA bins retain their supported fraction even when the
        # nominal centre maps beyond the table/source domain.  Select the
        # attenuation path inside that same fraction; discarding the entire
        # bin based on its centre would lose valid particles even for sigma=0.
        valid_output_range_lo = max(
            stopping_table.range_g_cm2[0],
            stopping_table.range_at_energy(input_lo) - thickness_g_cm2,
        )
        valid_output_range_hi = min(
            stopping_table.max_range_g_cm2,
            stopping_table.range_at_energy(input_hi) - thickness_g_cm2,
        )
        valid_output_lo = stopping_table.energy_at_range(valid_output_range_lo)
        valid_output_hi = stopping_table.energy_at_range(valid_output_range_hi)
        output_energy = min(max(output_energy, valid_output_lo), valid_output_hi)
        input_range = min(
            max(
                stopping_table.range_at_energy(output_energy) + thickness_g_cm2,
                supported_range_lo,
            ),
            supported_range_hi,
        )
        input_energy = min(
            max(stopping_table.energy_at_range(input_range), input_lo),
            input_hi,
        )
        # The survived ordinate describes the whole transported bin, not only
        # its centre.  Release support therefore has to cover both output-bin
        # edges and both CSDA-mapped entrance edges.
        relevant_energies.extend(
            (output_lo, output_energy, output_hi, input_lo, input_energy, input_hi)
        )
        if not support.contains(output_energy) or not support.contains(input_energy):
            # Preserve a diagnostic primary estimate without inventing a
            # nonelastic correction outside the evaluated TENDL domain.
            survived_values.append(primary.y[index])
            continue
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

    output = derive_spectrum(
        primary,
        y=tuple(survived_values),
        model=output_model,
        transformation=TransformationStep(
            name=PROTON_AL_NONELASTIC_SURVIVAL_MODEL,
            version="2",
            parameters=(
                ("integration_steps", str(integration_steps)),
                ("bin_attenuation", "representative_path_clipped_to_supported_bin"),
                ("nonelastic_mf_mt", f"3/{cross_section_table.mt}"),
                ("thickness_g_cm2", f"{thickness_g_cm2:.17g}"),
                (
                    "tendl_incident_energy_support_mev",
                    f"{support.minimum_mev:g}..{support.maximum_mev:g}",
                ),
            ),
        ),
        additional_dataset_refs=("tendl/p_al27_mf3_xs.csv",),
    )
    coverage = assess_tendl_incident_energy_coverage(
        relevant_energies_mev=relevant_energies,
        support=support,
    )
    reason = coverage.incomplete_reason(
        operator=PROTON_AL_NONELASTIC_SUPPORT_OPERATOR,
    )
    if reason is not None:
        output = mark_spectrum_incomplete(output, reason)
    return output
