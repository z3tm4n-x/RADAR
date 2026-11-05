"""Exact source-grid/native-table interval diagnostics, not radiation transport.

The native total-energy interval is divided by the literal source A only after
exact numerical mass agreement. Runtime float grid values are recorded using
Decimal(str(value)); decisions use exact rational arithmetic on those records,
not rounded display quotients. A covered interval is table support, not a
physical-error bound, a source truncation rule, or a qualified mass convention.
Raw-file, executable and source-code authentication remains the caller's duty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from math import isfinite
from typing import Literal

from radar.core.constants import (
    DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON,
)
from radar.validation.srim_native_coverage import (
    NativeCoverageObservation, SourceNativeCoverageRow, audit_srim_native_coverage,
)
from radar.validation.srim_output import SrimRangeOutput


GridNumber = int | float | Decimal
_CONTEXT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
IntervalMeaning = Literal[
    "declared_evaluation_span", "declared_finite_integral_domain", "native_table_support",
    "below_native_support", "above_native_support", "native_support_unknown",
]


def _number(value: GridNumber) -> Decimal:
    if type(value) not in (int, float, Decimal):
        raise ValueError("Energy values must be int, float or Decimal; bool is not energy.")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("Energy values must be finite and positive.")
    converted = value if isinstance(value, Decimal) else Decimal(str(value))
    if (not converted.is_finite() or converted <= 0
            or len(converted.as_tuple().digits) > 80 or abs(converted.adjusted()) > 1000):
        raise ValueError("Energy values must be positive finite bounded Decimal coordinates.")
    return converted


@dataclass(frozen=True)
class ExactEnergyBoundary:
    """An exact positive rational MeV/u coordinate plus a rounded display value."""

    numerator: Decimal
    denominator: Decimal
    decimal_mev_per_u: Decimal = field(init=False)

    def __post_init__(self) -> None:
        if type(self.numerator) is not Decimal or type(self.denominator) is not Decimal:
            raise ValueError("Boundary numerator and denominator must be Decimal.")
        _number(self.numerator)
        _number(self.denominator)
        with localcontext(_CONTEXT):
            value = self.numerator / self.denominator
        object.__setattr__(self, "decimal_mev_per_u", value)

    @property
    def exact(self) -> Fraction:
        return Fraction(self.numerator) / Fraction(self.denominator)


@dataclass(frozen=True)
class EnergyInterval:
    lower: ExactEnergyBoundary
    upper: ExactEnergyBoundary
    lower_closed: bool
    upper_closed: bool
    meaning: IntervalMeaning

    def __post_init__(self) -> None:
        if (type(self.lower) is not ExactEnergyBoundary or type(self.upper) is not ExactEnergyBoundary
                or type(self.lower_closed) is not bool or type(self.upper_closed) is not bool
                or self.lower.exact >= self.upper.exact):
            raise ValueError("An energy interval requires ordered bounds and explicit closure.")


@dataclass(frozen=True)
class SourceEnergyDomain:
    source: str
    evaluation_centers_mev_per_u: tuple[Decimal, ...]
    evaluation_span: EnergyInterval
    finite_integral_domain: EnergyInterval | None
    finite_integral_upper_semantics: str
    grid_conversion: str = field(default="Decimal(str(runtime_number)); Decimal retained", init=False)
    coordinate_semantics: str = field(default="energy_per_literal_source_mass_parameter", init=False)


@dataclass(frozen=True)
class EnergyCenterPartition:
    total: int
    supported: int
    below_native_support: int
    above_native_support: int
    native_support_unknown: int

    def __post_init__(self) -> None:
        parts = (self.supported, self.below_native_support,
                 self.above_native_support, self.native_support_unknown)
        if (type(self.total) is not int or self.total < 1
                or any(type(n) is not int or n < 0 for n in parts) or sum(parts) != self.total):
            raise ValueError("Center partition must conserve the complete declared grid count.")


@dataclass(frozen=True)
class DeclaredIntervalCoverage:
    domain: EnergyInterval
    uncovered_intervals: tuple[EnergyInterval, ...]
    status: Literal["COMPLETE_DECLARED_INTERVAL", "PARTIAL", "UNKNOWN"]
    declared_interval_supported: bool


@dataclass(frozen=True)
class SourceEnergySupportRow:
    source: str
    z: int
    symbol: str
    source_tabulated_mass_parameter: Decimal
    source_mass_to_charge: Decimal | None
    source_weight_parameter_name: str
    source_weight_parameter: Decimal
    nonzero_source_components: tuple[str, ...]
    material_z: int
    classification: Literal["declared_zero", "missing_native", "observed_interval_only"]
    native_observation: NativeCoverageObservation | None
    native_support: EnergyInterval | None
    center_partition: EnergyCenterPartition | None
    evaluation_span_coverage: DeclaredIntervalCoverage | None
    finite_integral_domain_coverage: DeclaredIntervalCoverage | None
    residual_energy_below_native_interval: str = field(default="UNKNOWN_not_physical_stopping",
                                                     init=False)
    physical_mass_policy_qualified: bool = field(default=False, init=False)
    normative_export_allowed: bool = field(default=False, init=False)


@dataclass(frozen=True)
class EnergySupportSummary:
    source: str
    material_z: int
    source_species_count: int
    positive_species_count: int
    declared_zero_species_count: int
    positive_species_with_native_table: int
    positive_species_without_native_table: int
    positive_evaluation_spans_supported: int
    positive_finite_integral_domains_supported: int | None
    center_partition: EnergyCenterPartition


@dataclass(frozen=True)
class SrimEnergySupportAudit:
    source_domains: tuple[SourceEnergyDomain, ...]
    rows: tuple[SourceEnergySupportRow, ...]
    summaries: tuple[EnergySupportSummary, ...]
    native_table_count: int
    positive_source_material_rows: int
    declared_zero_source_material_rows: int
    schema_version: str = field(default="radar.srim_energy_support.v1", init=False)
    diagnostic_status: str = field(default="INCOMPLETE", init=False)
    division_display_precision_decimal_digits: int = field(default=80, init=False)
    interval_comparison_arithmetic: str = field(default="exact_rational_from_retained_Decimal",
                                              init=False)
    source_energy_domain_covered: bool = field(default=False, init=False)
    shielded_transport_domain_covered: bool = field(default=False, init=False)
    physical_accuracy_verified: bool = field(default=False, init=False)
    isotope_policy_qualified: bool = field(default=False, init=False)
    normative_export_allowed: bool = field(default=False, init=False)
    source_spectrum_convolution_performed: bool = field(default=False, init=False)
    production_tables_replaced: bool = field(default=False, init=False)
    native_authenticity_verified: bool = field(default=False, init=False)
    limitations: tuple[str, ...] = field(default=(
        "Support is the closed interval of one native table, not physical validity between nodes.",
        "Exact numeric source A/native mass agreement is not an isotope-distribution claim.",
        "Different mass parameters and duplicate tables are rejected, not reconciled or merged.",
        "Decimal quotients are display only; exact rational bounds determine classifications.",
        "Counts are grid samples, not fluence fractions, missing doses or SEE contributions.",
        "GCR finite integral closure is retained separately from its final evaluated center.",
        "SEP last evaluated center is not asserted to be a physical or infinite-source boundary.",
        "Declared-zero source rows do not require native data and are not missing species.",
        "Missing native data are UNKNOWN and never zero-valued physical contributions.",
        "Residual energies below observed support and physical stopping at zero remain unresolved.",
        "No extrapolation, source truncation, isotope selection or stopping-physics change occurs.",
        "Caller must authenticate all native evidence and pin the current source definitions.",
    ), init=False)


def _boundary(value: Decimal) -> ExactEnergyBoundary:
    return ExactEnergyBoundary(value, Decimal(1))


def _domain(source: str, grid: tuple[GridNumber, ...],
            upper: GridNumber | None) -> SourceEnergyDomain:
    if type(grid) is not tuple or len(grid) < 2:
        raise ValueError("Source grids must be immutable tuples with at least two centers.")
    converted = tuple(_number(value) for value in grid)
    if any(b <= a for a, b in zip(converted, converted[1:])):
        raise ValueError("Source grid centers must be strictly increasing without duplicates.")
    first, last = _boundary(converted[0]), _boundary(converted[-1])
    span = EnergyInterval(first, last, True, True, "declared_evaluation_span")
    integral = None
    semantics = "not_declared; last_center_is_evaluation_limit_only"
    if upper is not None:
        end = _boundary(_number(upper))
        if end.exact < last.exact:
            raise ValueError("Finite GCR integral closure cannot precede the final center.")
        integral = EnergyInterval(first, end, True, True, "declared_finite_integral_domain")
        semantics = "explicit_finite_integral_closure_not_infinite_source"
    return SourceEnergyDomain(source, converted, span, integral, semantics)


def _coverage(domain: EnergyInterval, native: EnergyInterval | None) -> DeclaredIntervalCoverage:
    if native is None:
        unknown = EnergyInterval(domain.lower, domain.upper, True, True, "native_support_unknown")
        return DeclaredIntervalCoverage(domain, (unknown,), "UNKNOWN", False)
    gaps: list[EnergyInterval] = []
    if domain.lower.exact < native.lower.exact:
        stop = domain.upper if domain.upper.exact < native.lower.exact else native.lower
        gaps.append(EnergyInterval(domain.lower, stop, True,
                                   domain.upper.exact < native.lower.exact, "below_native_support"))
    if domain.upper.exact > native.upper.exact:
        start = domain.lower if domain.lower.exact > native.upper.exact else native.upper
        gaps.append(EnergyInterval(start, domain.upper,
                                   domain.lower.exact > native.upper.exact, True,
                                   "above_native_support"))
    return DeclaredIntervalCoverage(domain, tuple(gaps),
                                    "PARTIAL" if gaps else "COMPLETE_DECLARED_INTERVAL", not gaps)


def _row(source: SourceNativeCoverageRow, domain: SourceEnergyDomain) -> SourceEnergySupportRow:
    native = source.observation
    support = None
    if native is not None:
        if native.native_ion_mass_u != source.source_tabulated_mass_parameter:
            raise ValueError("Native numeric mass must equal literal source A exactly; no rounding.")
        low, high = native.energy_total_mev_bounds
        support = EnergyInterval(ExactEnergyBoundary(low, source.source_tabulated_mass_parameter),
                                 ExactEnergyBoundary(high, source.source_tabulated_mass_parameter),
                                 True, True, "native_table_support")
    counts = None
    evaluated = None
    integral = None
    if not source.nonzero_source_components:
        classification: Literal["declared_zero", "missing_native", "observed_interval_only"] = (
            "declared_zero")
    else:
        centers = tuple(Fraction(value) for value in domain.evaluation_centers_mev_per_u)
        if support is None:
            classification = "missing_native"
            counts = EnergyCenterPartition(len(centers), 0, 0, 0, len(centers))
        else:
            classification = "observed_interval_only"
            lo, hi = support.lower.exact, support.upper.exact
            counts = EnergyCenterPartition(len(centers), sum(lo <= c <= hi for c in centers),
                                           sum(c < lo for c in centers), sum(c > hi for c in centers), 0)
        evaluated = _coverage(domain.evaluation_span, support)
        if domain.finite_integral_domain is not None:
            integral = _coverage(domain.finite_integral_domain, support)
    return SourceEnergySupportRow(
        source.source, source.z, source.symbol, source.source_tabulated_mass_parameter,
        source.source_mass_to_charge, source.source_weight_parameter_name,
        source.source_weight_parameter, source.nonzero_source_components, source.material_z,
        classification, native, support, counts, evaluated, integral,
    )


def audit_srim_energy_support(
    native_outputs: tuple[SrimRangeOutput, ...], *,
    gcr_grid: tuple[GridNumber, ...] = DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    sep_grid: tuple[GridNumber, ...] = DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    gcr_upper_bound: GridNumber = GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON,
) -> SrimEnergySupportAudit:
    """Partition fixed OST source grids against exact, single-table HZE support.

    Empty observations yield an explicit all-missing positive roster. GCR and
    SEP retain all source rows including coefficient-declared zeros. No positive
    source contribution is inferred from a sample count or a mass comparison.
    """
    if gcr_upper_bound is None:
        raise ValueError("GCR requires an explicit finite integral closure.")
    domains = (_domain("GCR", gcr_grid, gcr_upper_bound), _domain("SEP", sep_grid, None))
    audit = audit_srim_native_coverage(native_outputs)
    if audit.proton_observations:
        raise ValueError("This ledger covers HZE only; proton observations require a separate audit.")
    by_source = {domain.source: domain for domain in domains}
    rows = tuple(_row(source, by_source[source.source]) for source in audit.rows)
    summaries = []
    for source in ("GCR", "SEP"):
        for material in (13, 14):
            selected = tuple(row for row in rows if row.source == source and row.material_z == material)
            positive = tuple(row for row in selected if row.nonzero_source_components)
            counts = tuple(row.center_partition for row in positive if row.center_partition is not None)
            summary_partition = EnergyCenterPartition(
                sum(c.total for c in counts), sum(c.supported for c in counts),
                sum(c.below_native_support for c in counts), sum(c.above_native_support for c in counts),
                sum(c.native_support_unknown for c in counts),
            )
            summaries.append(EnergySupportSummary(
                source, material, len(selected), len(positive), len(selected) - len(positive),
                sum(row.native_observation is not None for row in positive),
                sum(row.native_observation is None for row in positive),
                sum(row.evaluation_span_coverage is not None and
                    row.evaluation_span_coverage.declared_interval_supported for row in positive),
                None if source == "SEP" else sum(row.finite_integral_domain_coverage is not None and
                    row.finite_integral_domain_coverage.declared_interval_supported for row in positive),
                summary_partition,
            ))
    return SrimEnergySupportAudit(domains, rows, tuple(summaries), len(native_outputs),
                                  sum(bool(row.nonzero_source_components) for row in rows),
                                  sum(not row.nonzero_source_components for row in rows))
