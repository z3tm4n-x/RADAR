"""Opt-in HZE primary transport from paired, bounded native SRIM stopping.

The Al path coordinate integrates electronic plus elastic-nuclear stopping.
Its zero at the first observed energy is NOT a physical stopping range. Native
projected ranges are never used. File authentication belongs to the loader;
this operator preserves identities and keeps every result explicitly incomplete.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from math import fsum, isfinite

from radar.core.spectra import ShieldingDescriptor, Spectrum1D, TransformationStep
from radar.core.spectrum_ops import derive_spectrum
from radar.core.types import (
    AssessmentMode, Particle, RadiationSource, SpectrumAxisKind, SpectrumCompleteness,
    SpectrumLocation, SpectrumOrdinateBasis, SpectrumQuantity,
)
from radar.core.units import Unit
from radar.validation.candidate_transport import (
    PrimaryTransportLedger, _finite, _grid, _source_integral, _source_support,
)
from radar.validation.srim_anchored_range import AnchoredSrimRangeTable
from radar.validation.srim_output import SrimRangeOutput

_WORK = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
_REASON = (
    "SRIM primary transport is incomplete: bounded native stopping, unresolved low-energy origin "
    "and upper support; no nuclear-reaction attenuation or fragmentation."
)


@dataclass(frozen=True)
class SrimIonStoppingTable:
    """An exact native Al/Si pair; derived arrays cannot be injected by callers."""

    native_al: SrimRangeOutput = field(repr=False)
    native_si: SrimRangeOutput = field(repr=False)
    z: int = field(init=False)
    mass_parameter_u: float = field(init=False)
    mass_parameter_u_decimal: Decimal = field(init=False)
    dataset_ref: str = field(init=False)
    al_energy: tuple[float, ...] = field(init=False)
    si_energy: tuple[float, ...] = field(init=False)
    si_electronic_let: tuple[float, ...] = field(init=False)
    al_range_table: AnchoredSrimRangeTable = field(init=False, repr=False)
    mass_policy: str = field(default="literal_source_A_numeric_transport_mass", init=False)
    normative_export_allowed: bool = field(default=False, init=False)
    physical_accuracy_verified: bool = field(default=False, init=False)
    native_source_authenticity_verified: bool = field(default=False, init=False)
    _range_cache: dict[float, Decimal] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        al, si = self.native_al, self.native_si
        if type(al) is not SrimRangeOutput or type(si) is not SrimRangeOutput:
            raise ValueError("Structurally validated native SRIM Al/Si objects are required.")
        with localcontext(_WORK):
            al.__post_init__()
            si.__post_init__()
        if (al.target_z != 13 or si.target_z != 14 or al.ion_z < 2
                or al.ion_z != si.ion_z or al.ion_mass_u != si.ion_mass_u):
            raise ValueError("Native pair must contain Al and Si for exactly the same HZE Z and source A.")
        if al.ion_mass_u < al.ion_z:
            raise ValueError("Native source mass parameter must be at least Z.")
        for native in (al, si):
            if (native.stopping_unit != "MeV/(mg/cm2)"
                    or native.energy_basis != "total_ion_kinetic_energy"):
                raise ValueError("Unsupported native stopping units or energy coordinate.")
        mass = float(al.ion_mass_u)
        _finite(mass, "native source mass", positive=True)
        al_energy = tuple(float(Fraction(row.energy_total_mev) / Fraction(al.ion_mass_u))
                          for row in al.rows)
        si_energy = tuple(float(Fraction(row.energy_total_mev) / Fraction(si.ion_mass_u))
                          for row in si.rows)
        _grid(al_energy, "native Al energy per source A")
        _grid(si_energy, "native Si energy per source A")
        si_let = tuple(float(row.electronic_mev_cm2_mg) for row in si.rows)
        for value in si_let:
            _finite(value, "native electronic LET")
        anchor = AnchoredSrimRangeTable.from_native(al, interpolation="loglog_stopping")
        for name, attribute in (
            ("z", al.ion_z), ("mass_parameter_u", mass),
            ("mass_parameter_u_decimal", al.ion_mass_u), ("al_energy", al_energy),
            ("si_energy", si_energy), ("si_electronic_let", si_let),
            ("al_range_table", anchor),
            ("dataset_ref", f"srim-native:Al@sha256:{al.raw_sha256};Si@sha256:{si.raw_sha256}"),
        ):
            object.__setattr__(self, name, attribute)

    @classmethod
    def from_native(cls, al: SrimRangeOutput, si: SrimRangeOutput) -> SrimIonStoppingTable:
        return cls(al, si)

    def _total_energy(self, energy: float) -> Decimal:
        """Preserve native endpoints exactly, otherwise use the binary64 coordinate."""
        _finite(energy, "incident/exit energy", positive=True)
        if not self.al_energy[0] <= energy <= self.al_energy[-1]:
            raise ValueError("Energy is outside native Al support; no extrapolation.")
        index = bisect_left(self.al_energy, energy)
        if index < len(self.al_energy) and self.al_energy[index] == energy:
            return self.al_range_table.energy_total_mev[index]
        with localcontext(_WORK):
            return +(Decimal.from_float(float(energy)) * self.mass_parameter_u_decimal)

    def range_at_energy(self, energy_mev_per_nucleon: float) -> Decimal:
        """Return anchored path coordinate, in g/cm2, with exact total-energy model."""
        _finite(energy_mev_per_nucleon, "incident/exit energy", positive=True)
        if energy_mev_per_nucleon not in self._range_cache:
            self._range_cache[energy_mev_per_nucleon] = self.al_range_table.range_at_energy(
                self._total_energy(energy_mev_per_nucleon))
        return self._range_cache[energy_mev_per_nucleon]

    def energy_at_range(self, anchored_range_g_cm2: Decimal) -> float:
        """Invert the same stopping interpolant; never extrapolate to rest."""
        total = self.al_range_table.energy_at_range(anchored_range_g_cm2)
        with localcontext(_WORK):
            energy = float(total / self.mass_parameter_u_decimal)
        _finite(energy, "inverse per-source-A energy", positive=True)
        if not self.al_energy[0] <= energy <= self.al_energy[-1]:
            raise ValueError("Inverse energy is outside supported binary64 representation.")
        return energy

    @property
    def max_anchored_range_g_cm2(self) -> Decimal:
        return self.al_range_table.anchored_range_g_cm2[-1]


def _thickness_decimal(thickness: float) -> Decimal:
    _finite(thickness, "shield thickness")
    with localcontext(_WORK):
        return +Decimal.from_float(float(thickness))


@dataclass(frozen=True)
class SrimPrimaryTransport:
    source: Spectrum1D
    table: SrimIonStoppingTable
    thickness_g_cm2: float
    spectrum: Spectrum1D
    ledger: PrimaryTransportLedger
    exit_support: tuple[float, float] | None
    _incident_support: tuple[float, float] | None = field(repr=False)
    normative_export_allowed: bool = field(default=False, init=False)
    _preimage_cache: dict[float, float] = field(default_factory=dict, init=False, repr=False, compare=False)

    def integral_on_exit(self, lower: float, upper: float) -> float:
        """Count/rate in supported exit interval, integrated in ORIGINAL source energy."""
        _finite(lower, "exit lower bound")
        _finite(upper, "exit upper bound")
        if upper < lower:
            raise ValueError("Exit bounds must be ordered.")
        if self.exit_support is None:
            return 0.
        lo, hi = max(lower, self.exit_support[0]), min(upper, self.exit_support[1])
        if hi <= lo:
            return 0.
        if self.thickness_g_cm2 == 0:
            return _source_integral(self.source, lo, hi)
        assert self._incident_support is not None
        thickness = _thickness_decimal(self.thickness_g_cm2)

        def incident(exit_energy: float) -> float:
            # Preserve established exact interval endpoints after float conversion.
            assert self.exit_support is not None and self._incident_support is not None
            if exit_energy == self.exit_support[0]:
                return self._incident_support[0]
            if exit_energy == self.exit_support[1]:
                return self._incident_support[1]
            if exit_energy in self._preimage_cache:
                return self._preimage_cache[exit_energy]
            with localcontext(_WORK):
                coordinate = self.table.range_at_energy(exit_energy) + thickness
            if coordinate > self.table.max_anchored_range_g_cm2:
                raise ValueError("Interior preimage exceeds native Al support.")
            result = self.table.energy_at_range(coordinate)
            if not self._incident_support[0] <= result <= self._incident_support[1]:
                raise ValueError("Interior preimage escaped established incident support.")
            self._preimage_cache[exit_energy] = result
            return result

        return _source_integral(self.source, incident(lo), incident(hi))


def _validate_source(source: Spectrum1D, table: SrimIonStoppingTable, mass: float) -> None:
    _finite(mass, "declared source mass", positive=True)
    if mass != table.mass_parameter_u:
        raise ValueError("Exact literal source/table mass parameters must match; no isotope rounding.")
    if (source.particle is not Particle.HZE or source.species is None or source.species.is_aggregate
            or source.species.atomic_number != table.z):
        raise ValueError("An individual HZE source with matching Z is required.")
    if source.source not in (RadiationSource.GCR, RadiationSource.SEP):
        raise ValueError("Only GCR/SEP HZE sources are supported.")
    if (source.axis_kind is not SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
            or source.x_unit is not Unit.MEV_PER_NUCLEON):
        raise ValueError("An explicit MeV/nucleon source axis is required.")
    if source.location not in (SpectrumLocation.ON_ORBIT, SpectrumLocation.OUTSIDE_MAGNETOSPHERE):
        raise ValueError("Source must be unshielded; repeated shielding is not implicit.")
    if source.assessment_mode is AssessmentMode.PEAK:
        if source.quantity not in (SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
                                   SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX):
            raise ValueError("PEAK source must be a peak/maximum differential flux.")
    elif source.assessment_mode is AssessmentMode.CUMULATIVE:
        if source.quantity is not SpectrumQuantity.DIFFERENTIAL_FLUENCE:
            raise ValueError("CUMULATIVE source must be a differential fluence.")
    else:
        raise ValueError("Explicit PEAK or CUMULATIVE mode is required.")
    violations = tuple(v for v in source.production_contract_violations()
                       if v != "incomplete spectra cannot be released as production results")
    if violations:
        raise ValueError(f"Invalid typed source contract: {violations}")
    if source.ordinate_basis not in (SpectrumOrdinateBasis.POINT_SAMPLE, SpectrumOrdinateBasis.BIN_AVERAGE):
        raise ValueError("Unsupported source ordinate convention.")
    nodes = source.bin_edges if source.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE else source.x
    assert nodes is not None
    _grid(nodes, "source support")
    for value in source.y:
        _finite(value, "source ordinate")
    for step in source.transformation_chain:
        for key, text_value in step.parameters:
            if key == "source_tabulated_mass_parameter":
                try:
                    declared = float(text_value)
                except (ValueError, OverflowError) as exc:
                    raise ValueError("Invalid source mass trace.") from exc
                _finite(declared, "source mass trace", positive=True)
                # Existing source traces serialize the numeric float with .17g
                # (e.g. 55.799999999999997). This is not a different source A.
                # Native Al/Si identity above remains an exact Decimal check.
                if declared != table.mass_parameter_u:
                    raise ValueError("Numeric source mass conflicts with transformation trace.")


def transport_srim_primary(
    source: Spectrum1D, table: SrimIonStoppingTable, thickness_g_cm2: float, *,
    source_mass_parameter_u: float, output_edges: tuple[float, ...] | None = None,
) -> SrimPrimaryTransport:
    """Supported primary CSDA pushforward through a centered spherical Al shell.

    This operator applies neither mission time, density, nor angular multipliers.
    An unsupported source/low exit contributes to an explicit unresolved ledger,
    not to a claim that the ion has stopped. No high-energy continuation is used.
    """
    if type(table) is not SrimIonStoppingTable:
        raise ValueError("A paired native SRIM stopping table is required.")
    _validate_source(source, table, source_mass_parameter_u)
    thickness = _thickness_decimal(thickness_g_cm2)
    if thickness == 0 and output_edges is not None:
        raise ValueError("Zero shielding preserves source identity; output rebinning is not allowed.")
    edges = table.al_energy if output_edges is None else output_edges
    _grid(edges, "output energy edges")
    s0, s1 = _source_support(source)
    total = _source_integral(source, s0, s1)
    incident_support: tuple[float, float] | None = None
    exit_support: tuple[float, float] | None = None
    outside = unresolved = supported = 0.
    if thickness == 0:
        incident_support = exit_support = (s0, s1)
        supported = total
    else:
        e0, e1 = table.al_energy[0], table.al_energy[-1]
        lo, hi = max(s0, e0), min(s1, e1)
        outside = fsum((_source_integral(source, s0, min(s1, e0)),
                        _source_integral(source, max(s0, e1), s1)))
        if lo < hi:
            if thickness >= table.max_anchored_range_g_cm2:
                unresolved = _source_integral(source, lo, hi)
            else:
                threshold = table.energy_at_range(thickness)
                start = max(lo, threshold)
                unresolved = _source_integral(source, lo, min(hi, start))
                if start < hi:
                    incident_support = (start, hi)
                    with localcontext(_WORK):
                        remaining = table.range_at_energy(hi) - thickness
                        start_remaining = table.range_at_energy(start) - thickness
                    # The inverse threshold was rounded to binary64. At that
                    # exact threshold its true exit is the original positive anchor.
                    lower_exit = e0 if start == threshold else table.energy_at_range(start_remaining)
                    upper_exit = table.energy_at_range(remaining)
                    if not lower_exit < upper_exit:
                        raise ValueError("Supported exit interval collapsed at floating precision.")
                    if upper_exit >= hi:
                        raise ValueError("Positive shield loss is below binary64 energy resolution.")
                    exit_support = (lower_exit, upper_exit)
                    supported = _source_integral(source, start, hi)
    provisional = PrimaryTransportLedger(total, supported, 0., 0., 0., outside, unresolved, 0.)
    measure = SrimPrimaryTransport(source, table, thickness_g_cm2, source, provisional,
                                   exit_support, incident_support)
    if thickness == 0:
        represented, below, above = total, 0., 0.
        x, y, bin_edges, basis = source.x, source.y, source.bin_edges, source.ordinate_basis
    else:
        counts = tuple(measure.integral_on_exit(a, b) for a, b in zip(edges, edges[1:]))
        represented = fsum(counts)
        below = measure.integral_on_exit(0., edges[0])
        above = (measure.integral_on_exit(edges[-1], max(edges[-1], exit_support[1]))
                 if exit_support is not None else 0.)
        x = tuple(a + (b - a) / 2 for a, b in zip(edges, edges[1:]))
        y = tuple(c / (b - a) for c, a, b in zip(counts, edges, edges[1:]))
        bin_edges, basis = edges, SpectrumOrdinateBasis.BIN_AVERAGE
    for value in y:
        _finite(value, "output density")
    balance = fsum((represented, below, above, outside, unresolved, -total))
    if not isfinite(balance) or abs(balance) > 1e-10 * total:
        raise ValueError("SRIM primary source-integral ledger failed closure.")
    ledger = PrimaryTransportLedger(total, supported, represented, below, above, outside, unresolved, balance)
    reasons = [*source.incomplete_reasons, _REASON]
    if outside > 0:
        reasons.append("Source integral outside observed SRIM Al support remains unresolved.")
    if unresolved > 0:
        reasons.append("Residual energy below SRIM positive-energy anchor is unresolved, not stopped.")
    if below + above > 0:
        reasons.append("Supported SRIM exits lie outside the requested energy display window.")
    spectrum = derive_spectrum(
        source, x=x, y=y, bin_edges=bin_edges, ordinate_basis=basis,
        model="srim_primary_al_anchored_csda", location=SpectrumLocation.BEHIND_SHIELD,
        shielding=ShieldingDescriptor("Al", thickness_g_cm2),
        additional_dataset_refs=(table.dataset_ref,), completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=tuple(dict.fromkeys(reasons)),
        transformation=TransformationStep("srim_primary_al_anchored_csda", "1", (
            ("dataset_ref", table.dataset_ref), ("mass_policy", table.mass_policy),
            ("mass_parameter_u", str(table.mass_parameter_u_decimal)),
            ("thickness_g_cm2", repr(thickness_g_cm2)),
            ("range_interpolation", "analytic_integral_of_piecewise_loglog_total_stopping"),
            ("range_origin", "R_at_first_observed_positive_energy_zero_offset_only"),
            ("projected_ranges_used", "false"), ("al_stopping", "electronic_plus_elastic_nuclear"),
            ("source_point_policy", "piecewise_linear_density"),
            ("source_bin_average_policy", "piecewise_constant_density"),
            ("time_factor", "none"), ("angular_factor", "none"), ("density_factor", "none"),
            ("nuclear_reaction_attenuation", "not_included"), ("fragmentation", "not_included"),
            ("normative_export_allowed", "false"),
        )),
    )
    return SrimPrimaryTransport(source, table, thickness_g_cm2, spectrum, ledger,
                                exit_support, incident_support)
