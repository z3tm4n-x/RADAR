"""Conservative electronic-LET pushforward for a supported primary measure.

The original transported source measure is integrated over every preimage of
each LET bin. An intermediate energy histogram is deliberately not re-used as
a sub-bin density model. Piecewise-linear Si electronic stopping may have any
number of turning points or plateaus; no inverse derivative is divided by.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import fsum, isfinite
from numbers import Real
from typing import Protocol, runtime_checkable

from radar.core.spectra import Spectrum1D, TransformationStep
from radar.core.spectrum_ops import derive_spectrum, mark_spectrum_incomplete
from radar.core.types import (
    LetSpectrumRole,
    Particle,
    SpectrumAxisKind,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
)
from radar.core.units import Unit


def _nonnegative_integral(value: float) -> None:
    try:
        invalid = (isinstance(value, bool) or not isinstance(value, Real)
                   or not isfinite(value) or value < 0)
    except OverflowError:
        invalid = True
    if invalid:
        raise ValueError("LET integral components must be finite and nonnegative, not bool.")


def _sum_integrals(values: Sequence[float]) -> float:
    for value in values:
        _nonnegative_integral(value)
    try:
        result = fsum(values)
    except OverflowError as exc:
        raise ValueError("LET integral sum exceeds finite representation.") from exc
    _nonnegative_integral(result)
    return result


@dataclass(frozen=True)
class ElectronicLetLedger:
    """Integral counts (or rates), never reconstructed by time multiplication."""

    supported_energy_integral: float
    mapped_integral: float
    below_let_window_integral: float
    above_let_window_integral: float
    outside_si_domain_integral: float
    balance_residual: float


@dataclass(frozen=True)
class ElectronicLetResult:
    spectrum: Spectrum1D
    ledger: ElectronicLetLedger
    normative_export_allowed: bool = field(default=False, init=False)
    physical_table_accuracy_verified: bool = field(default=False, init=False)


def _edges(values: tuple[float, ...]) -> tuple[float, ...]:
    edges = tuple(values)
    if len(edges) < 2:
        raise ValueError("At least two LET edges are required.")
    try:
        invalid = any(
            isinstance(value, bool) or not isinstance(value, Real)
            or not isfinite(value) or value < 0
            for value in edges
        )
    except OverflowError:
        invalid = True
    if invalid:
        raise ValueError("LET edges must be finite nonnegative real numbers, not bool.")
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError("LET edges must be strictly increasing.")
    return edges


def _output_quantity(quantity: SpectrumQuantity) -> tuple[SpectrumQuantity, Unit]:
    if quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        return SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE, Unit.DIFFERENTIAL_LET_FLUENCE
    if quantity in (
        SpectrumQuantity.DIFFERENTIAL_FLUX,
        SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    ):
        return SpectrumQuantity.LET_DIFFERENTIAL_FLUX, Unit.DIFFERENTIAL_LET_FLUX
    raise ValueError("Electronic LET requires a differential energy flux or fluence.")


class ElectronicLetTable(Protocol):
    """Read-only electronic stopping coordinates; energies are MeV/source A."""
    @property
    def si_energy(self) -> tuple[float, ...]: ...
    @property
    def si_electronic_let(self) -> tuple[float, ...]: ...
    @property
    def mass_policy(self) -> str: ...
    @property
    def mass_parameter_u(self) -> float: ...
    @property
    def dataset_ref(self) -> str: ...


class SupportedPrimaryLedger(Protocol):
    @property
    def supported_exit_integral(self) -> float: ...


@runtime_checkable
class ElectronicLetMeasure(Protocol):
    """A source measure with supported exit preimages, not a histogram."""
    @property
    def source(self) -> Spectrum1D: ...
    @property
    def spectrum(self) -> Spectrum1D: ...
    @property
    def table(self) -> ElectronicLetTable: ...
    @property
    def ledger(self) -> SupportedPrimaryLedger: ...
    @property
    def exit_support(self) -> tuple[float, float] | None: ...

    def integral_on_exit(self, lower: float, upper: float) -> float: ...


@dataclass(frozen=True)
class ElectronicLetProvenance:
    model_suffix: str
    transformation_name: str
    diagnostic_reason: str
    outside_domain_reason: str

    def __post_init__(self) -> None:
        for value in (self.model_suffix, self.transformation_name,
                      self.diagnostic_reason, self.outside_domain_reason):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Explicit nonempty electronic-LET provenance is required.")


SRIM_ELECTRONIC_LET_PROVENANCE = ElectronicLetProvenance(
    model_suffix="srim_electronic_let",
    transformation_name="srim_electronic_let_pushforward",
    diagnostic_reason="SRIM electronic LET: supported partial measure; release qualification pending",
    outside_domain_reason="Supported primary exits outside SRIM Si electronic-LET domain",
)


def _validate_measure(transport: ElectronicLetMeasure) -> None:
    if not isinstance(transport, ElectronicLetMeasure):
        raise ValueError("A supported primary source measure with exit preimages is required.")
    if not isinstance(transport.source, Spectrum1D) or not isinstance(transport.spectrum, Spectrum1D):
        raise ValueError("Primary source and energy output must be typed spectra.")
    source = transport.source
    if (source.particle is not Particle.HZE or source.species is None or source.species.is_aggregate
            or source.axis_kind is not SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
            or source.x_unit is not Unit.MEV_PER_NUCLEON):
        raise ValueError("Electronic LET needs an individual HZE source on a per-nucleon energy axis.")
    for key in ("particle", "species", "source", "assessment_mode", "quantity", "x_unit", "y_unit",
                "temporal_basis", "mission_duration_s", "angular_basis"):
        if getattr(source, key) != getattr(transport.spectrum, key):
            raise ValueError("Primary output changed its source identity, mode or units.")
    table = transport.table
    try:
        energies, lets = table.si_energy, table.si_electronic_let
        mass, policy, dataset = table.mass_parameter_u, table.mass_policy, table.dataset_ref
    except AttributeError as exc:
        raise ValueError("Incomplete electronic stopping table interface.") from exc
    if (not isinstance(energies, tuple) or not isinstance(lets, tuple)
            or len(energies) < 2 or len(energies) != len(lets)):
        raise ValueError("Matched immutable Si energy/LET arrays are required.")
    _edges(energies)
    if energies[0] <= 0:
        raise ValueError("Si energies must be strictly positive.")
    for value in lets:
        _nonnegative_integral(value)
    _nonnegative_integral(mass)
    if mass <= 0 or not isinstance(policy, str) or not policy.strip():
        raise ValueError("Explicit positive source mass and mass policy are required.")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("An explicit stopping dataset reference is required.")
    support = transport.exit_support
    if support is not None:
        if not isinstance(support, tuple) or len(support) != 2:
            raise ValueError("Exit support must be an ordered energy pair.")
        _edges(support)
        if support[0] <= 0:
            raise ValueError("Supported exit energies must be positive.")


def convert_primary_electronic_let(
    transport: ElectronicLetMeasure,
    *,
    let_edges: tuple[float, ...],
    provenance: ElectronicLetProvenance,
) -> ElectronicLetResult:
    """Push the supported primary measure into an explicit electronic-LET grid.

    LET edges use MeV cm^2/mg, already converted from the table's electronic
    stopping. Every Si segment is treated separately, including all branches
    around a stopping maximum. For a plateau, the containing interval is
    left-closed/right-open, except that the final upper edge is included.

    The ledger covers only the supported primary exit measure. The transport
    object's unresolved incident counts remain unresolved and are not silently
    assigned a zero LET or called stopped. All results remain INCOMPLETE.
    """
    _validate_measure(transport)
    if not isinstance(provenance, ElectronicLetProvenance):
        raise ValueError("Explicit electronic-LET provenance is required.")
    edges = _edges(let_edges)
    quantity, y_unit = _output_quantity(transport.source.quantity)
    table = transport.table
    pieces: list[list[float]] = [[] for _ in edges[:-1]]
    below: list[float] = []
    above: list[float] = []
    si_outside: list[float] = []
    support = transport.exit_support
    if support is not None:
        low, high = support
        if low < table.si_energy[0]:
            si_outside.append(transport.integral_on_exit(low, min(high, table.si_energy[0])))
        if high > table.si_energy[-1]:
            si_outside.append(transport.integral_on_exit(max(low, table.si_energy[-1]), high))
        for e0, e1, l0, l1 in zip(
            table.si_energy, table.si_energy[1:],
            table.si_electronic_let, table.si_electronic_let[1:],
        ):
            if e1 <= low or e0 >= high:
                continue
            if l0 == l1:
                count = transport.integral_on_exit(e0, e1)
                if l0 < edges[0]:
                    below.append(count)
                elif l0 > edges[-1]:
                    above.append(count)
                else:
                    index = min(bisect_right(edges, l0) - 1, len(pieces) - 1)
                    pieces[index].append(count)
                continue

            lmin, lmax = min(l0, l1), max(l0, l1)

            def count_between(left: float, right: float) -> float:
                lower, upper = max(lmin, left), min(lmax, right)
                if upper <= lower:
                    return 0.0
                # Preserve exact endpoint energies. The two roots can reverse
                # order on a decreasing branch, so integrate their sorted pair.
                def root(value: float) -> float:
                    if value == l0:
                        return e0
                    if value == l1:
                        return e1
                    return e0 + (value - l0) / (l1 - l0) * (e1 - e0)

                a, b = root(lower), root(upper)
                return transport.integral_on_exit(min(a, b), max(a, b))

            if lmin < edges[0]:
                below.append(count_between(lmin, edges[0]))
            if lmax > edges[-1]:
                above.append(count_between(edges[-1], lmax))
            start = max(0, bisect_right(edges, lmin) - 1)
            stop = min(len(pieces), bisect_left(edges, lmax))
            for index in range(start, stop):
                pieces[index].append(count_between(edges[index], edges[index + 1]))

    counts = tuple(_sum_integrals(values) for values in pieces)
    mapped = _sum_integrals(counts)
    below_count = _sum_integrals(below)
    above_count = _sum_integrals(above)
    outside_count = _sum_integrals(si_outside)
    supported = transport.ledger.supported_exit_integral
    _nonnegative_integral(supported)
    try:
        balance = fsum((supported, -mapped, -below_count, -above_count, -outside_count))
    except OverflowError as exc:
        raise ValueError("LET integral balance exceeds finite representation.") from exc
    if not isfinite(balance) or abs(balance) > 1e-10 * supported:
        raise ValueError("Primary electronic-LET integral balance failed.")
    ledger = ElectronicLetLedger(
        supported, mapped, below_count, above_count, outside_count, balance,
    )
    densities = tuple(count / (right - left)
                      for count, left, right in zip(counts, edges, edges[1:]))
    for value in densities:
        _nonnegative_integral(value)
    spectrum = derive_spectrum(
        transport.spectrum,
        x=tuple(left + (right - left) / 2 for left, right in zip(edges, edges[1:])),
        y=densities,
        x_unit=Unit.LET,
        y_unit=y_unit,
        quantity=quantity,
        model=f"{transport.spectrum.model}+{provenance.model_suffix}",
        axis_kind=SpectrumAxisKind.LINEAR_ENERGY_TRANSFER,
        let_role=LetSpectrumRole.HZE_SPECIES,
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=edges,
        transformation=TransformationStep(
            name=provenance.transformation_name, version="1",
            parameters=(
                ("target", "silicon"),
                ("stopping_component_used", "electronic_collisional_only"),
                ("si_interpolation", "piecewise_linear_energy_electronic_let"),
                ("measure", "original_source_preimage_not_intermediate_energy_histogram"),
                ("plateau_bin_ownership", "left_closed_right_open_final_upper_edge_inclusive"),
                ("mass_policy", table.mass_policy),
                ("mass_parameter_u", repr(table.mass_parameter_u)),
                ("dataset_ref", table.dataset_ref),
                ("normative_export_allowed", "false"),
            ),
        ),
        additional_dataset_refs=(table.dataset_ref,),
    )
    reasons = [provenance.diagnostic_reason]
    if outside_count > 0:
        reasons.append(provenance.outside_domain_reason)
    if below_count > 0 or above_count > 0:
        reasons.append("Supported primary exits outside requested electronic-LET window")
    spectrum = mark_spectrum_incomplete(spectrum, *reasons)
    return ElectronicLetResult(spectrum, ledger)


