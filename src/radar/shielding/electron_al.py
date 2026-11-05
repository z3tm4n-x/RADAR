"""ESTAR electron range-energy tables and CSDA transport in aluminium."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import exp, isfinite, log
from pathlib import Path
from typing import Mapping

from radar.core.spectra import ShieldingDescriptor, Spectrum1D, TransformationStep
from radar.core.spectrum_ops import derive_spectrum, mark_spectrum_incomplete
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
)
from radar.core.units import Unit

ELECTRON_AL_PRIMARY_CSDA_MODEL = "electron_primary_al_spherical_csda"
ELECTRON_AL_OUTPUT_ENERGY_SUPPORT_INCOMPLETE = (
    "electron_al_output_energy_interval_outside_stopping_table_support"
)
ELECTRON_AL_MAPPED_RANGE_SUPPORT_INCOMPLETE = (
    "electron_al_mapped_range_interval_outside_stopping_table_support"
)
ELECTRON_AL_SOURCE_ENERGY_SUPPORT_INCOMPLETE = (
    "electron_al_mapped_source_interval_outside_spectrum_support"
)

_REQUIRED_COLUMNS = {
    "energy_mev",
    "stopping_mev_cm2_g",
    "range_g_cm2",
}

_ENERGY_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
)


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


def log_bin_edges_from_centers(centers: tuple[float, ...]) -> tuple[float, ...]:
    """Return logarithmic bin edges from strictly increasing positive bin centers."""

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


def integrate_spectrum_loglog(
    spectrum: Spectrum1D,
    lo_mev: tuple[float, ...],
    hi_mev: tuple[float, ...],
) -> tuple[float, ...]:
    """Integrate point samples, using log-log or linear-through-zero segments."""

    if len(lo_mev) != len(hi_mev):
        msg = "Integration interval lower/upper arrays must have equal length."
        raise ValueError(msg)

    if spectrum.ordinate_basis is not SpectrumOrdinateBasis.POINT_SAMPLE:
        msg = (
            "Log-log point interpolation requires point-sample source ordinates; "
            "conservative remapping of bin-average input is not implemented."
        )
        raise ValueError(msg)

    values: list[float] = []

    for lo, hi in zip(lo_mev, hi_mev):
        if not (isfinite(lo) and isfinite(hi)) or hi <= lo:
            values.append(0.0)
            continue

        if lo < spectrum.x[0] or hi > spectrum.x[-1]:
            msg = "Integration interval is outside the source spectrum support."
            raise ValueError(msg)

        interval_lo = lo
        interval_hi = hi

        total = 0.0

        for index in range(len(spectrum.x) - 1):
            segment_lo = max(interval_lo, spectrum.x[index])
            segment_hi = min(interval_hi, spectrum.x[index + 1])

            if segment_hi <= segment_lo:
                continue

            total += _integral_powerlaw_segment(
                e1=spectrum.x[index],
                y1=spectrum.y[index],
                e2=spectrum.x[index + 1],
                y2=spectrum.y[index + 1],
                a=segment_lo,
                b=segment_hi,
            )

        values.append(total)

    return tuple(values)


def shield_electron_spectrum_primary_through_al(
    *,
    spectrum: Spectrum1D,
    table: ElectronAlRangeEnergyTable | None,
    thickness_g_cm2: float,
    output_energy_mev: tuple[float, ...] | None = None,
    model: str | None = None,
) -> Spectrum1D:
    """Return primary electron spectrum behind straight-ahead CSDA aluminium shielding."""

    if not isfinite(thickness_g_cm2):
        msg = "Al shielding thickness must be finite."
        raise ValueError(msg)

    if thickness_g_cm2 < 0.0:
        msg = "Al shielding thickness must be non-negative."
        raise ValueError(msg)

    _validate_spectrum(spectrum)

    if output_energy_mev is None:
        output_energy = spectrum.x
    else:
        output_energy = output_energy_mev

    _validate_output_energy_grid(output_energy)

    output_model = model
    if output_model is None:
        output_model = f"{spectrum.model}+{ELECTRON_AL_PRIMARY_CSDA_MODEL}"

    if thickness_g_cm2 == 0.0 and output_energy == spectrum.x:
        return derive_spectrum(
            spectrum,
            y=spectrum.y,
            model=output_model,
            location=SpectrumLocation.BEHIND_SHIELD,
            shielding=ShieldingDescriptor(
                material="aluminium",
                areal_density_g_cm2=thickness_g_cm2,
            ),
            transformation=TransformationStep(
                name=ELECTRON_AL_PRIMARY_CSDA_MODEL,
                version="1",
                parameters=(
                    ("geometry", "centered_spherical_shell"),
                    ("thickness_g_cm2", f"{thickness_g_cm2:.17g}"),
                ),
            ),
        )

    if table is None:
        msg = "Positive Al shielding requires an electron Al range table."
        raise ValueError(msg)

    if spectrum.ordinate_basis is not SpectrumOrdinateBasis.POINT_SAMPLE:
        msg = (
            "Electron Al CSDA remapping requires point-sample source ordinates; "
            "conservative remapping of bin-average input is not implemented."
        )
        raise ValueError(msg)

    edges = log_bin_edges_from_centers(output_energy)
    widths = tuple(right - left for left, right in zip(edges, edges[1:]))

    input_energy_lo: list[float] = []
    input_energy_hi: list[float] = []
    valid: list[bool] = []

    incomplete_reasons: list[str] = []

    for output_lo, output_hi in zip(edges, edges[1:]):
        # CSDA is monotonic, so each supported part remains one contiguous
        # interval as it is intersected with the table and source domains.
        # The final integral is nevertheless divided by the original, full
        # output-bin width below: the published bin edges are never clipped.
        supported_output_lo = max(output_lo, table.energy_mev[0])
        supported_output_hi = min(output_hi, table.energy_mev[-1])
        if supported_output_lo != output_lo or supported_output_hi != output_hi:
            incomplete_reasons.append(ELECTRON_AL_OUTPUT_ENERGY_SUPPORT_INCOMPLETE)

        if supported_output_hi <= supported_output_lo:
            input_energy_lo.append(0.0)
            input_energy_hi.append(0.0)
            valid.append(False)
            continue

        mapped_range_lo = table.range_at_energy(supported_output_lo) + thickness_g_cm2
        mapped_range_hi = table.range_at_energy(supported_output_hi) + thickness_g_cm2
        supported_range_lo = max(mapped_range_lo, table.range_g_cm2[0])
        supported_range_hi = min(mapped_range_hi, table.max_range_g_cm2)
        if supported_range_lo != mapped_range_lo or supported_range_hi != mapped_range_hi:
            incomplete_reasons.append(ELECTRON_AL_MAPPED_RANGE_SUPPORT_INCOMPLETE)

        if supported_range_hi <= supported_range_lo:
            input_energy_lo.append(0.0)
            input_energy_hi.append(0.0)
            valid.append(False)
            continue

        mapped_source_lo = table.energy_at_range(supported_range_lo)
        mapped_source_hi = table.energy_at_range(supported_range_hi)
        supported_source_lo = max(mapped_source_lo, spectrum.x[0])
        supported_source_hi = min(mapped_source_hi, spectrum.x[-1])
        if supported_source_lo != mapped_source_lo or supported_source_hi != mapped_source_hi:
            incomplete_reasons.append(ELECTRON_AL_SOURCE_ENERGY_SUPPORT_INCOMPLETE)

        if supported_source_hi <= supported_source_lo:
            input_energy_lo.append(0.0)
            input_energy_hi.append(0.0)
            valid.append(False)
            continue

        input_energy_lo.append(supported_source_lo)
        input_energy_hi.append(supported_source_hi)
        valid.append(True)

    integrals = integrate_spectrum_loglog(
        spectrum,
        tuple(input_energy_lo),
        tuple(input_energy_hi),
    )

    output_values = tuple(
        integral / width if is_valid and width > 0.0 else 0.0
        for integral, width, is_valid in zip(integrals, widths, valid)
    )

    output = derive_spectrum(
        spectrum,
        x=output_energy,
        y=output_values,
        model=output_model,
        location=SpectrumLocation.BEHIND_SHIELD,
        shielding=ShieldingDescriptor(
            material="aluminium",
            areal_density_g_cm2=thickness_g_cm2,
        ),
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=edges,
        transformation=TransformationStep(
            name=ELECTRON_AL_PRIMARY_CSDA_MODEL,
            version="1",
            parameters=(
                ("geometry", "centered_spherical_shell"),
                ("thickness_g_cm2", f"{thickness_g_cm2:.17g}"),
            ),
        ),
        additional_dataset_refs=("stopping/electron_al_range.csv",),
    )

    if incomplete_reasons:
        output = mark_spectrum_incomplete(
            output,
            *tuple(dict.fromkeys(incomplete_reasons)),
        )

    return output


def _integral_powerlaw_segment(
    *,
    e1: float,
    y1: float,
    e2: float,
    y2: float,
    a: float,
    b: float,
) -> float:
    if b <= a:
        return 0.0

    if y1 == 0.0 or y2 == 0.0:
        # A POINT_SAMPLE spectrum still defines the segment when one endpoint
        # is zero.  Log-log interpolation is undefined there, so use the
        # unique straight line through the two samples and integrate only the
        # requested sub-interval.  Treating the whole segment as zero loses a
        # finite triangular contribution present in OST electron spectra.
        slope = (y2 - y1) / (e2 - e1)
        value_a = y1 + slope * (a - e1)
        value_b = y1 + slope * (b - e1)
        return 0.5 * (value_a + value_b) * (b - a)

    exponent = log(y2 / y1) / log(e2 / e1)
    coefficient = y1 / (e1**exponent)

    if abs(exponent + 1.0) < 1.0e-12:
        return float(coefficient * log(b / a))

    return float(coefficient * (b ** (exponent + 1.0) - a ** (exponent + 1.0)) / (exponent + 1.0))


def _validate_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.source is not RadiationSource.ERB:
        msg = "Electron Al shielding supports ERB spectra only."
        raise ValueError(msg)

    if spectrum.particle is not Particle.ELECTRON:
        msg = "Electron Al shielding requires an electron spectrum."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "Electron Al shielding energy grid must be in MeV."
        raise ValueError(msg)

    if spectrum.quantity not in _ENERGY_QUANTITIES:
        msg = "Electron Al shielding requires an energy fluence or flux spectrum."
        raise ValueError(msg)

    if len(spectrum.x) < 2:
        msg = "Electron Al shielding spectrum must contain at least two energy points."
        raise ValueError(msg)


def _validate_output_energy_grid(output_energy_mev: tuple[float, ...]) -> None:
    if len(output_energy_mev) < 2:
        msg = "Output electron energy grid must contain at least two points."
        raise ValueError(msg)

    if any(energy <= 0.0 for energy in output_energy_mev):
        msg = "Output electron energy grid must be positive."
        raise ValueError(msg)

    if any(right <= left for left, right in zip(output_energy_mev, output_energy_mev[1:])):
        msg = "Output electron energy grid must be strictly increasing."
        raise ValueError(msg)


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


__all__ = [
    "ELECTRON_AL_MAPPED_RANGE_SUPPORT_INCOMPLETE",
    "ELECTRON_AL_OUTPUT_ENERGY_SUPPORT_INCOMPLETE",
    "ELECTRON_AL_PRIMARY_CSDA_MODEL",
    "ELECTRON_AL_SOURCE_ENERGY_SUPPORT_INCOMPLETE",
    "ElectronAlRangeEnergyTable",
    "integrate_spectrum_loglog",
    "load_electron_al_range_energy_table",
    "log_bin_edges_from_centers",
    "shield_electron_spectrum_primary_through_al",
]
