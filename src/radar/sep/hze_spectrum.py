"""SEP HZE/TZCh source spectrum core for OST 134-1044-2007 Appendix B."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from radar.sep.proton_spectrum import (
    SepProtonSpectrumCoefficients,
    proton_momentum_mev,
)

OST_SEP_HZE_COEFFICIENT_MODEL = "ost_134_1044_2007"
OST_SEP_HZE_B9_RESOURCE = "sep_hze/ost_134_1044_2007_b9.csv"

# Appendix B.5 uses 1.34. Appendix B.6 is treated with the known printed
# OST correction: gamma2(z) = 1.68 * gamma2(p)**0.62.
OST_SEP_HZE_GAMMA1_MULTIPLIER = 1.34
OST_SEP_HZE_GAMMA1_EXPONENT = 0.57
OST_SEP_HZE_GAMMA2_MULTIPLIER = 1.68
OST_SEP_HZE_GAMMA2_EXPONENT = 0.62
OST_SEP_HZE_REFERENCE_ENERGY_MEV_PER_NUCLEON = 30.0

_REQUIRED_B9_COLUMNS = {
    "model",
    "z",
    "symbol",
    "mass_number",
    "mass_to_charge",
    "relative_abundance",
    "source_table",
}


@dataclass(frozen=True)
class OstSepHzeIonRecord:
    """One OST Appendix B.9 ion abundance row."""

    model: str
    z: int
    symbol: str
    mass_number: float
    mass_to_charge: float
    relative_abundance: float
    source_table: str

    def __post_init__(self) -> None:
        if self.model != OST_SEP_HZE_COEFFICIENT_MODEL:
            msg = "OST SEP HZE ion record has unsupported model."
            raise ValueError(msg)

        if self.z < 2:
            msg = "OST SEP HZE ion charge Z must be at least 2."
            raise ValueError(msg)

        if not self.symbol:
            msg = "OST SEP HZE ion symbol must not be empty."
            raise ValueError(msg)

        if self.mass_number <= 0.0:
            msg = "OST SEP HZE ion mass number must be positive."
            raise ValueError(msg)

        if self.mass_to_charge <= 0.0:
            msg = "OST SEP HZE ion A/Q must be positive."
            raise ValueError(msg)

        if self.relative_abundance < 0.0:
            msg = "OST SEP HZE ion relative abundance must be non-negative."
            raise ValueError(msg)

        if not self.source_table:
            msg = "OST SEP HZE ion source table must not be empty."
            raise ValueError(msg)


@dataclass(frozen=True)
class OstSepHzeSpectrumCoefficients:
    """Coefficients defining one SEP HZE differential source spectrum."""

    c: float
    break_energy_mev_per_nucleon: float
    gamma1: float
    gamma2: float
    ion: OstSepHzeIonRecord

    def __post_init__(self) -> None:
        if self.c < 0.0:
            msg = "OST SEP HZE coefficient C must be non-negative."
            raise ValueError(msg)

        if self.break_energy_mev_per_nucleon <= 0.0:
            msg = "OST SEP HZE break energy must be positive."
            raise ValueError(msg)

        if self.gamma1 <= 0.0 or self.gamma2 <= 0.0:
            msg = "OST SEP HZE spectral indices must be positive."
            raise ValueError(msg)


def _required_text(
    row: Mapping[str, str | None],
    column: str,
    line_number: int,
) -> str:
    value = row.get(column)

    if value is None or not value.strip():
        msg = f"Missing required column {column!r} at HZE B.9 CSV line {line_number}."
        raise ValueError(msg)

    return value.strip()


def _parse_ion_record(
    row: Mapping[str, str | None],
    line_number: int,
) -> OstSepHzeIonRecord:
    return OstSepHzeIonRecord(
        model=_required_text(row, "model", line_number),
        z=int(_required_text(row, "z", line_number)),
        symbol=_required_text(row, "symbol", line_number),
        mass_number=float(_required_text(row, "mass_number", line_number)),
        mass_to_charge=float(_required_text(row, "mass_to_charge", line_number)),
        relative_abundance=float(
            _required_text(row, "relative_abundance", line_number),
        ),
        source_table=_required_text(row, "source_table", line_number),
    )


def load_ost_sep_hze_ion_records(
    path: str | Path,
) -> tuple[OstSepHzeIonRecord, ...]:
    """Load OST Appendix B.9 ion records from CSV."""

    csv_path = Path(path)
    records: list[OstSepHzeIonRecord] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = _REQUIRED_B9_COLUMNS - fieldnames

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            msg = f"OST SEP HZE B.9 CSV is missing columns: {missing}."
            raise ValueError(msg)

        for line_number, row in enumerate(reader, start=2):
            records.append(_parse_ion_record(row, line_number))

    return tuple(records)


def ost_sep_hze_coefficients_from_proton(
    *,
    proton_coefficients: SepProtonSpectrumCoefficients,
    ion: OstSepHzeIonRecord,
) -> OstSepHzeSpectrumCoefficients:
    """Return HZE spectrum coefficients from proton coefficients.

    Implements Appendix B formulas B.3--B.6, with the corrected B.6 multiplier
    for gamma2(z).
    """

    proton_c = 10.0**proton_coefficients.log10_c
    proton_break_energy = proton_coefficients.break_energy_mev

    gamma1 = OST_SEP_HZE_GAMMA1_MULTIPLIER * (
        proton_coefficients.gamma1**OST_SEP_HZE_GAMMA1_EXPONENT
    )
    gamma2 = OST_SEP_HZE_GAMMA2_MULTIPLIER * (
        proton_coefficients.gamma2**OST_SEP_HZE_GAMMA2_EXPONENT
    )
    break_energy = proton_break_energy / ion.mass_to_charge

    c_value = (
        ion.relative_abundance
        * proton_c
        * (
            proton_momentum_mev(OST_SEP_HZE_REFERENCE_ENERGY_MEV_PER_NUCLEON)
            / proton_momentum_mev(proton_break_energy)
        )
        ** (-2.0 * proton_coefficients.gamma2)
        * (
            proton_momentum_mev(break_energy)
            / proton_momentum_mev(OST_SEP_HZE_REFERENCE_ENERGY_MEV_PER_NUCLEON)
        )
        ** (-2.0 * gamma2)
    )

    return OstSepHzeSpectrumCoefficients(
        c=float(c_value),
        break_energy_mev_per_nucleon=float(break_energy),
        gamma1=float(gamma1),
        gamma2=float(gamma2),
        ion=ion,
    )


def evaluate_ost_sep_hze_spectrum_value(
    energy_mev_per_nucleon: float,
    coefficients: OstSepHzeSpectrumCoefficients,
) -> float:
    """Evaluate one OST SEP HZE differential spectrum value."""

    if energy_mev_per_nucleon <= 0.0:
        msg = "OST SEP HZE kinetic energy per nucleon must be positive."
        raise ValueError(msg)

    if coefficients.c == 0.0:
        return 0.0

    if energy_mev_per_nucleon < coefficients.break_energy_mev_per_nucleon:
        energy_ratio = energy_mev_per_nucleon / coefficients.break_energy_mev_per_nucleon
        return float(coefficients.c * energy_ratio ** (-coefficients.gamma1))

    momentum = proton_momentum_mev(energy_mev_per_nucleon)
    break_momentum = proton_momentum_mev(coefficients.break_energy_mev_per_nucleon)
    momentum_ratio = momentum / break_momentum

    return float(coefficients.c * momentum_ratio ** (-2.0 * coefficients.gamma2))


def evaluate_ost_sep_hze_spectrum(
    energy_grid_mev_per_nucleon: tuple[float, ...],
    coefficients: OstSepHzeSpectrumCoefficients,
) -> tuple[float, ...]:
    """Evaluate an OST SEP HZE differential spectrum on an energy grid."""

    if not energy_grid_mev_per_nucleon:
        msg = "OST SEP HZE energy grid must not be empty."
        raise ValueError(msg)

    return tuple(
        evaluate_ost_sep_hze_spectrum_value(
            energy_mev_per_nucleon=energy,
            coefficients=coefficients,
        )
        for energy in energy_grid_mev_per_nucleon
    )
