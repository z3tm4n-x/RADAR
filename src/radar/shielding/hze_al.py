"""Primary HZE CSDA transport through spherical aluminium shielding."""

from __future__ import annotations

import csv
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from math import exp, isclose, isfinite, log, sqrt
from pathlib import Path
from typing import Mapping

from radar.core.spectra import ShieldingDescriptor, Spectrum1D, TransformationStep
from radar.core.spectrum_ops import derive_spectrum, mark_spectrum_incomplete
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
)
from radar.core.units import Unit

HZE_AL_PRIMARY_CSDA_MODEL = "hze_primary_al_spherical_csda"
HZE_AL_NO_FRAGMENTATION_NOTE = "hze_nuclear_attenuation_and_fragmentation_not_included"
HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX = "HZE Al transport domain incomplete"
HZE_AL_MASS_NUMBER_MISMATCH_REASON_PREFIX = "HZE Al ion mass mismatch"
HZE_AL_LOWER_OUTPUT_WINDOW_REASON_PREFIX = "HZE Al supported exits below output window"

_SUPPORT_REL_TOL = 1.0e-12
_SUPPORT_ABS_TOL = 1.0e-14

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


def _integrate_piecewise_linear_strict(
    *,
    lower: float,
    upper: float,
    xp: tuple[float, ...],
    fp: tuple[float, ...],
) -> float:
    """Integrate a point-sample spectrum without extrapolation."""

    if lower < xp[0] or upper > xp[-1]:
        msg = "HZE source integration interval is outside the sampled support."
        raise ValueError(msg)
    if upper <= lower:
        msg = "HZE source integration interval must have positive width."
        raise ValueError(msg)

    first_internal = bisect_right(xp, lower)
    last_internal = bisect_left(xp, upper)
    knots = (lower, *xp[first_internal:last_internal], upper)
    values = tuple(
        _interpolate_strict(
            x=energy,
            xp=xp,
            fp=fp,
            quantity="HZE spectrum",
        )
        for energy in knots
    )

    return sum(
        0.5 * (left_value + right_value) * (right_energy - left_energy)
        for left_energy, right_energy, left_value, right_value in zip(
            knots,
            knots[1:],
            values,
            values[1:],
        )
    )


def _close(left: float, right: float) -> bool:
    return isclose(
        left,
        right,
        rel_tol=_SUPPORT_REL_TOL,
        abs_tol=_SUPPORT_ABS_TOL,
    )


@dataclass(frozen=True)
class _DeclaredSourceTransportLedger:
    """Partition only the known point-sample source, not unknown source tails.

    The unresolved part includes incident energies outside the range table and
    residual ranges below its minimum.  In particular, it is not a count of
    physically stopped ions: an absolute range origin is not established here.
    """

    source_integral: float
    supported_exit_in_window: float
    supported_exit_below_window: float
    unresolved_transport: float


def _declared_source_transport_ledger(
    *,
    spectrum: Spectrum1D,
    table: HzeAlRangeTable,
    thickness_g_cm2: float,
    output_lower_energy: float,
) -> _DeclaredSourceTransportLedger:
    """Account for supported exits omitted by the inherited output low edge.

    Both source and range interpolation remain inside their declared support.
    The output upper edge inherited from the source log grid exceeds every
    source energy; nonnegative CSDA energy loss therefore cannot cross it.
    A common additive range offset cancels from all classification boundaries.
    """

    def source_integral(lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.0
        return _integrate_piecewise_linear_strict(
            lower=lower,
            upper=upper,
            xp=spectrum.x,
            fp=spectrum.y,
        )

    total = source_integral(spectrum.x[0], spectrum.x[-1])
    input_lower = max(spectrum.x[0], table.energy_mev_per_nucleon[0])
    input_upper = min(spectrum.x[-1], table.energy_mev_per_nucleon[-1])
    minimum_supported_input_range = table.range_g_cm2[0] + thickness_g_cm2
    if (
        input_upper <= input_lower
        or minimum_supported_input_range >= table.range_at_energy(input_upper)
    ):
        return _DeclaredSourceTransportLedger(total, 0.0, 0.0, total)

    supported_lower = max(
        input_lower,
        table.energy_at_range(minimum_supported_input_range),
    )
    split_energy = supported_lower
    if output_lower_energy >= table.energy_mev_per_nucleon[-1]:
        split_energy = input_upper
    elif output_lower_energy > table.energy_mev_per_nucleon[0]:
        required_input_range = (
            table.range_at_energy(output_lower_energy) + thickness_g_cm2
        )
        if required_input_range >= table.range_at_energy(input_upper):
            split_energy = input_upper
        elif required_input_range > table.range_at_energy(supported_lower):
            split_energy = table.energy_at_range(required_input_range)

    return _DeclaredSourceTransportLedger(
        source_integral=total,
        supported_exit_in_window=source_integral(split_energy, input_upper),
        supported_exit_below_window=source_integral(supported_lower, split_energy),
        unresolved_transport=(
            source_integral(spectrum.x[0], supported_lower)
            + source_integral(input_upper, spectrum.x[-1])
        ),
    )


def _split_edges_at_support_boundaries(
    *,
    edges: tuple[float, ...],
    boundaries: tuple[float, ...],
) -> tuple[float, ...]:
    """Insert physical support boundaries without producing near-zero bins."""

    result = list(edges)
    for boundary in boundaries:
        if boundary <= edges[0] or boundary >= edges[-1]:
            continue

        matching_index = next(
            (
                index
                for index, existing in enumerate(result)
                if _close(existing, boundary)
            ),
            None,
        )
        if matching_index is None:
            result.append(boundary)
        else:
            result[matching_index] = boundary

    result.sort()
    if any(right <= left for left, right in zip(result, result[1:])):
        msg = "HZE support-aligned output edges must be strictly increasing."
        raise RuntimeError(msg)
    return tuple(result)


def _mapped_output_boundary(
    *,
    input_energy: float,
    table: HzeAlRangeTable,
    thickness_g_cm2: float,
) -> float | None:
    input_range = table.range_at_energy(input_energy)
    output_range = input_range - thickness_g_cm2
    if output_range < table.range_g_cm2[0] or output_range > table.max_range_g_cm2:
        return None
    return table.energy_at_range(output_range)


def _output_support_boundaries(
    *,
    spectrum: Spectrum1D,
    table: HzeAlRangeTable,
    thickness_g_cm2: float,
) -> tuple[float, ...]:
    """Return every active table, range and source support boundary."""

    boundaries = [
        table.energy_mev_per_nucleon[0],
        table.energy_mev_per_nucleon[-1],
    ]

    input_lower = max(spectrum.x[0], table.energy_mev_per_nucleon[0])
    input_upper = min(spectrum.x[-1], table.energy_mev_per_nucleon[-1])
    if input_lower <= input_upper:
        for input_energy in (input_lower, input_upper):
            boundary = _mapped_output_boundary(
                input_energy=input_energy,
                table=table,
                thickness_g_cm2=thickness_g_cm2,
            )
            if boundary is not None:
                boundaries.append(boundary)

    range_limited_output_range = table.max_range_g_cm2 - thickness_g_cm2
    if table.range_g_cm2[0] <= range_limited_output_range <= table.max_range_g_cm2:
        boundaries.append(table.energy_at_range(range_limited_output_range))

    return tuple(boundaries)


def _clamp_roundoff_to_support(
    value: float,
    *,
    lower: float,
    upper: float,
) -> float:
    if _close(value, lower):
        return lower
    if _close(value, upper):
        return upper
    return value


def _interval_support_relation(
    *,
    lower: float,
    upper: float,
    support_lower: float,
    support_upper: float,
) -> str:
    """Classify an interval after all physical boundaries have been inserted."""

    if upper < support_lower or _close(upper, support_lower):
        return "below"
    if lower > support_upper or _close(lower, support_upper):
        return "above"

    clamped_lower = _clamp_roundoff_to_support(
        lower,
        lower=support_lower,
        upper=support_upper,
    )
    clamped_upper = _clamp_roundoff_to_support(
        upper,
        lower=support_lower,
        upper=support_upper,
    )
    if clamped_lower >= support_lower and clamped_upper <= support_upper:
        return "inside"
    return "partial"


def _require_aligned_interval(relation: str, *, domain: str) -> None:
    if relation != "partial":
        return
    msg = (
        "HZE support-aligned transport produced a partially supported "
        f"{domain} interval."
    )
    raise RuntimeError(msg)


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


def _validate_spectrum(spectrum: Spectrum1D) -> None:
    if spectrum.source not in (RadiationSource.SEP, RadiationSource.GCR):
        msg = "HZE Al shielding supports SEP and GCR spectra only."
        raise ValueError(msg)

    if spectrum.particle is not Particle.HZE:
        msg = "HZE Al shielding requires an HZE spectrum."
        raise ValueError(msg)

    allowed_x_units: tuple[Unit, ...]
    if spectrum.axis_kind is None:
        allowed_x_units = (Unit.MEV, Unit.MEV_PER_NUCLEON)
    else:
        allowed_x_units = (Unit.MEV_PER_NUCLEON,)
        if spectrum.axis_kind is not SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON:
            msg = "Typed HZE Al shielding requires a kinetic-energy-per-nucleon axis."
            raise ValueError(msg)

    if spectrum.x_unit not in allowed_x_units:
        msg = "HZE Al shielding energy grid must be in MeV/nucleon."
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


def _validate_spectrum_species_for_table(
    spectrum: Spectrum1D,
    table: HzeAlRangeTable,
) -> str | None:
    species = spectrum.species
    if species is None:
        if spectrum.axis_kind is not None:
            msg = "Typed HZE Al shielding requires explicit ion species metadata."
            raise ValueError(msg)
        return None

    expected_mass_number = int(round(table.mass_number))
    if (
        species.is_aggregate
        or species.atomic_number != table.z
        or species.symbol.casefold() != table.symbol.casefold()
    ):
        msg = (
            "HZE spectrum species does not match the selected aluminium range "
            f"table Z={table.z}, A={expected_mass_number}, {table.symbol}."
        )
        raise ValueError(msg)
    if species.mass_number != expected_mass_number:
        return (
            f"{HZE_AL_MASS_NUMBER_MISMATCH_REASON_PREFIX} for Z={table.z} "
            f"{table.symbol}: spectrum A={species.mass_number}, aluminium "
            f"range table A={expected_mass_number}"
        )
    return None

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

    _validate_spectrum(spectrum)
    mass_number_mismatch_reason = _validate_spectrum_species_for_table(
        spectrum,
        table,
    )

    output_model = model
    if output_model is None:
        output_model = (
            f"{spectrum.model}+Z{table.z}_{table.symbol}+"
            f"{HZE_AL_PRIMARY_CSDA_MODEL}+{HZE_AL_NO_FRAGMENTATION_NOTE}"
        )

    transformation = TransformationStep(
        name=HZE_AL_PRIMARY_CSDA_MODEL,
        version="3",
        parameters=(
            ("geometry", "centered_spherical_shell"),
            ("thickness_g_cm2", f"{thickness_g_cm2:.17g}"),
            ("z", str(table.z)),
            ("fragmentation", "not_included"),
            ("nuclear_attenuation", "not_included"),
            (
                "domain_policy",
                "support_aligned_explicit_incomplete_no_extrapolation",
            ),
            ("remap", "piecewise_linear_source_integral"),
        ),
    )

    def derived_bin_average_output(
        centers: tuple[float, ...],
        values: tuple[float, ...],
        *,
        bin_edges: tuple[float, ...],
    ) -> Spectrum1D:
        return derive_spectrum(
            spectrum,
            x=centers,
            y=values,
            model=output_model,
            location=SpectrumLocation.BEHIND_SHIELD,
            shielding=ShieldingDescriptor(
                material="aluminium",
                areal_density_g_cm2=thickness_g_cm2,
            ),
            transformation=transformation,
            additional_dataset_refs=("stopping/hze_al_range.csv",),
            ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
            bin_edges=bin_edges,
        )

    # No material is traversed at zero areal density.  Avoid a numerically
    # lossy range -> energy round trip and do not require a stopping table to
    # cover an operation that is exactly the identity.
    if thickness_g_cm2 == 0.0:
        return derive_spectrum(
            spectrum,
            model=output_model,
            location=SpectrumLocation.BEHIND_SHIELD,
            shielding=ShieldingDescriptor(
                material="aluminium",
                areal_density_g_cm2=thickness_g_cm2,
            ),
            transformation=transformation,
        )

    if spectrum.ordinate_basis is not SpectrumOrdinateBasis.POINT_SAMPLE:
        msg = (
            "Positive-thickness HZE Al CSDA transport requires point-sample "
            "source ordinates; conservative remapping of bin-average input is "
            "not implemented."
        )
        raise ValueError(msg)

    source_edges = log_bin_edges_from_centers(spectrum.x)
    edges = _split_edges_at_support_boundaries(
        edges=source_edges,
        boundaries=_output_support_boundaries(
            spectrum=spectrum,
            table=table,
            thickness_g_cm2=thickness_g_cm2,
        ),
    )
    source_ledger = _declared_source_transport_ledger(
        spectrum=spectrum,
        table=table,
        thickness_g_cm2=thickness_g_cm2,
        output_lower_energy=edges[0],
    )
    integral_unit = (
        Unit.INTEGRAL_FLUENCE
        if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
        else Unit.INTEGRAL_FLUX
    )
    transformation = replace(
        transformation,
        version="4",
        parameters=(
            *transformation.parameters,
            ("source_ledger_scope", "declared_point_sample_support_only"),
            ("source_ledger_interpolation", "piecewise_linear_no_extrapolation"),
            ("source_ledger_energy_lower_mev_per_nucleon", f"{spectrum.x[0]:.17g}"),
            ("source_ledger_energy_upper_mev_per_nucleon", f"{spectrum.x[-1]:.17g}"),
            ("output_window_energy_lower_mev_per_nucleon", f"{edges[0]:.17g}"),
            ("source_ledger_integral_unit", integral_unit.value),
            ("source_ledger_integral", f"{source_ledger.source_integral:.17g}"),
            (
                "supported_exit_integral_in_output_window",
                f"{source_ledger.supported_exit_in_window:.17g}",
            ),
            (
                "supported_exit_integral_below_output_window",
                f"{source_ledger.supported_exit_below_window:.17g}",
            ),
            (
                "unresolved_transport_integral_on_declared_source",
                f"{source_ledger.unresolved_transport:.17g}",
            ),
            ("physical_stopping_classification", "not_established"),
        ),
    )
    centers = tuple(
        sqrt(lower * upper)
        for lower, upper in zip(edges, edges[1:])
    )
    shielded_values: list[float] = []
    table_energy_domain_bins = 0
    table_range_domain_bins = 0
    source_lower_domain_bins = 0
    source_upper_domain_bins = 0
    minimum_required_source_energy = spectrum.x[0]
    maximum_required_source_energy = spectrum.x[-1]

    table_energy_lower = table.energy_mev_per_nucleon[0]
    table_energy_upper = table.energy_mev_per_nucleon[-1]
    table_range_lower = table.range_g_cm2[0]
    table_range_upper = table.max_range_g_cm2

    for energy_lo, energy_hi in zip(edges, edges[1:]):
        table_energy_relation = _interval_support_relation(
            lower=energy_lo,
            upper=energy_hi,
            support_lower=table_energy_lower,
            support_upper=table_energy_upper,
        )
        _require_aligned_interval(
            table_energy_relation,
            domain="range-table energy",
        )
        if table_energy_relation != "inside":
            table_energy_domain_bins += 1
            shielded_values.append(0.0)
            continue

        energy_lo = _clamp_roundoff_to_support(
            energy_lo,
            lower=table_energy_lower,
            upper=table_energy_upper,
        )
        energy_hi = _clamp_roundoff_to_support(
            energy_hi,
            lower=table_energy_lower,
            upper=table_energy_upper,
        )
        output_width = energy_hi - energy_lo

        range_lo = table.range_at_energy(energy_lo)
        range_hi = table.range_at_energy(energy_hi)

        input_range_lo = range_lo + thickness_g_cm2
        input_range_hi = range_hi + thickness_g_cm2
        table_range_relation = _interval_support_relation(
            lower=input_range_lo,
            upper=input_range_hi,
            support_lower=table_range_lower,
            support_upper=table_range_upper,
        )
        _require_aligned_interval(
            table_range_relation,
            domain="range-table mass-range",
        )
        if table_range_relation != "inside":
            table_range_domain_bins += 1
            shielded_values.append(0.0)
            continue

        input_range_lo = _clamp_roundoff_to_support(
            input_range_lo,
            lower=table_range_lower,
            upper=table_range_upper,
        )
        input_range_hi = _clamp_roundoff_to_support(
            input_range_hi,
            lower=table_range_lower,
            upper=table_range_upper,
        )
        input_energy_lo = table.energy_at_range(input_range_lo)
        input_energy_hi = table.energy_at_range(input_range_hi)
        source_relation = _interval_support_relation(
            lower=input_energy_lo,
            upper=input_energy_hi,
            support_lower=spectrum.x[0],
            support_upper=spectrum.x[-1],
        )
        _require_aligned_interval(source_relation, domain="source-energy")
        if source_relation == "below":
            source_lower_domain_bins += 1
            minimum_required_source_energy = min(
                minimum_required_source_energy,
                input_energy_lo,
            )
            shielded_values.append(0.0)
            continue
        if source_relation == "above":
            source_upper_domain_bins += 1
            maximum_required_source_energy = max(
                maximum_required_source_energy,
                input_energy_hi,
            )
            shielded_values.append(0.0)
            continue

        input_energy_lo = _clamp_roundoff_to_support(
            input_energy_lo,
            lower=spectrum.x[0],
            upper=spectrum.x[-1],
        )
        input_energy_hi = _clamp_roundoff_to_support(
            input_energy_hi,
            lower=spectrum.x[0],
            upper=spectrum.x[-1],
        )
        source_count = _integrate_piecewise_linear_strict(
            lower=input_energy_lo,
            upper=input_energy_hi,
            xp=spectrum.x,
            fp=spectrum.y,
        )
        shielded_values.append(source_count / output_width)

    output = derived_bin_average_output(
        centers,
        tuple(shielded_values),
        bin_edges=edges,
    )
    domain_reasons: list[str] = []
    if source_ledger.supported_exit_below_window > 0.0:
        domain_reasons.append(
            f"{HZE_AL_LOWER_OUTPUT_WINDOW_REASON_PREFIX} for Z={table.z}: "
            f"{source_ledger.supported_exit_below_window:.17g} "
            f"{integral_unit.value} from declared source support maps below "
            f"output lower edge {edges[0]:.17g} MeV/nucleon at "
            f"{thickness_g_cm2:.17g} g/cm^2 shielding; "
            "these table-supported exits are omitted, not physically stopped"
        )
    if table_energy_domain_bins:
        domain_reasons.append(
            f"{HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX} for Z={table.z}: "
            f"{table_energy_domain_bins} output bin(s) exceed aluminium "
            "range-table energy support "
            f"{table.energy_mev_per_nucleon[0]:.17g}.."
            f"{table.energy_mev_per_nucleon[-1]:.17g} MeV/nucleon"
        )
    if table_range_domain_bins:
        domain_reasons.append(
            f"{HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX} for Z={table.z}: "
            f"{table_range_domain_bins} output bin(s) require aluminium range "
            f"above {table.max_range_g_cm2:.17g} g/cm^2 at "
            f"{thickness_g_cm2:.17g} g/cm^2 shielding"
        )
    if source_lower_domain_bins:
        domain_reasons.append(
            f"{HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX} for Z={table.z}: source "
            f"lower support {spectrum.x[0]:.17g} MeV/nucleon is above the "
            f"required mapped energy {minimum_required_source_energy:.17g} "
            f"MeV/nucleon at {thickness_g_cm2:.17g} g/cm^2 shielding "
            f"({source_lower_domain_bins} bin(s))"
        )
    if source_upper_domain_bins:
        domain_reasons.append(
            f"{HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX} for Z={table.z}: source "
            f"upper support {spectrum.x[-1]:.17g} MeV/nucleon is below the "
            f"required mapped energy {maximum_required_source_energy:.17g} "
            f"MeV/nucleon at {thickness_g_cm2:.17g} g/cm^2 shielding "
            f"({source_upper_domain_bins} bin(s))"
        )
    if domain_reasons:
        output = mark_spectrum_incomplete(output, *domain_reasons)
    if mass_number_mismatch_reason is not None:
        output = mark_spectrum_incomplete(output, mass_number_mismatch_reason)
    return output


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
