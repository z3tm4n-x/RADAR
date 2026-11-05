"""Opt-in primary CSDA pushforward for mass-parameter stopping candidates.

This is not a replacement for production shielding. The source measure is
piecewise linear for POINT_SAMPLE, piecewise constant for BIN_AVERAGE, and
has no inferred tails. R(E) is linear between supported Al nodes. Integrals
over exit-energy intervals are evaluated in incident energy, without dividing
by a numerical derivative or reconstructing a histogram's sub-bin shape.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import fsum, isfinite
from numbers import Real

from radar.core.spectra import ShieldingDescriptor, Spectrum1D, TransformationStep
from radar.core.spectrum_ops import derive_spectrum
from radar.core.types import (
    AssessmentMode, Particle, RadiationSource, SpectrumAxisKind, SpectrumCompleteness, SpectrumLocation,
    SpectrumOrdinateBasis, SpectrumQuantity,
)
from radar.core.units import Unit
from radar.validation.parameterized_stopping import ParameterizedStoppingRow

MASS_POLICIES = (
    "source_tabulated_parameter_diagnostic", "explicit_pilot_isotope_diagnostic",
)
DIAGNOSTIC_REASON = (
    "Candidate primary transport: mass policy, stopping data and physical model unqualified; "
    "no nuclear attenuation/fragmentation or absolute low-energy stopping classification."
)


def _finite(value: float, name: str, *, positive: bool = False) -> None:
    try:
        invalid = (isinstance(value, bool) or not isinstance(value, Real)
                   or not isfinite(value) or (value <= 0 if positive else value < 0))
    except OverflowError:
        invalid = True
    if invalid:
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")


def _grid(values: tuple[float, ...], name: str) -> None:
    if not isinstance(values, tuple) or len(values) < 2:
        raise ValueError(f"{name} must be an immutable tuple with at least two nodes.")
    for value in values:
        _finite(value, name, positive=True)
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must increase strictly.")


def _linear(x: float, xp: tuple[float, ...], yp: tuple[float, ...]) -> float:
    """Bounded interpolation; endpoints are returned bit-for-bit, no extrapolation."""
    if not isfinite(x) or x < xp[0] or x > xp[-1]:
        raise ValueError("Interpolation outside candidate support.")
    i = min(bisect_right(xp, x) - 1, len(xp) - 2)
    if x == xp[i]:
        return yp[i]
    if x == xp[i + 1]:
        return yp[i + 1]
    f = (x - xp[i]) / (xp[i + 1] - xp[i])
    return (1 - f) * yp[i] + f * yp[i + 1]


@dataclass(frozen=True)
class CandidateIonTable:
    """One exact (Z, mass parameter, policy); species A is not an isotope proof."""

    z: int
    mass_parameter_u: float
    mass_policy: str
    dataset_ref: str
    al_energy: tuple[float, ...]
    al_range: tuple[float, ...]
    si_energy: tuple[float, ...]
    si_electronic_let: tuple[float, ...]
    relative_al_range: tuple[float, ...] = field(init=False, repr=False)
    normative_export_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.z, bool) or not isinstance(self.z, int) or not 2 <= self.z <= 92:
            raise ValueError("Candidate Z must be an integer in 2..92.")
        _finite(self.mass_parameter_u, "mass parameter", positive=True)
        if self.mass_parameter_u < self.z or self.mass_policy not in MASS_POLICIES:
            raise ValueError("Invalid mass parameter or explicit diagnostic mass policy.")
        if not isinstance(self.dataset_ref, str) or not self.dataset_ref.strip():
            raise ValueError("An explicit candidate dataset reference is required.")
        _grid(self.al_energy, "Al energy")
        _grid(self.al_range, "Al relative range")
        _grid(self.si_energy, "Si energy")
        if (len(self.al_range) != len(self.al_energy)
                or not isinstance(self.si_electronic_let, tuple)
                or len(self.si_electronic_let) != len(self.si_energy)):
            raise ValueError("Candidate arrays must have matching lengths.")
        for value in self.si_electronic_let:
            _finite(value, "electronic LET", positive=True)
        # Cancel an arbitrary range origin *before* interpolating or adding a
        # thin shield. Otherwise a large common offset can erase the shield.
        object.__setattr__(self, "relative_al_range",
                           tuple(r - self.al_range[0] for r in self.al_range))

    @classmethod
    def from_rows(
        cls, rows: Sequence[ParameterizedStoppingRow], *, mass_policy: str, dataset_ref: str,
    ) -> CandidateIonTable:
        if not rows or any(not isinstance(row, ParameterizedStoppingRow) for row in rows):
            raise ValueError("A nonempty sequence of parameterized rows is required.")
        identities = {(row.z, row.mass_parameter_u) for row in rows}
        if len(identities) != 1 or {row.material_z for row in rows} != {13, 14}:
            raise ValueError("Exactly one projectile and both Al/Si materials are required.")
        al = tuple(row for row in rows if row.material_z == 13)
        si = tuple(row for row in rows if row.material_z == 14)
        z, mass = next(iter(identities))
        return cls(z, mass, mass_policy, dataset_ref,
                   tuple(r.energy_mev_per_nucleon for r in al), tuple(r.range_g_cm2 for r in al),
                   tuple(r.energy_mev_per_nucleon for r in si),
                   tuple(r.let_electronic_mev_cm2_mg for r in si))


def _source_support(source: Spectrum1D) -> tuple[float, float]:
    nodes = source.bin_edges if source.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE else source.x
    assert nodes is not None
    return nodes[0], nodes[-1]


def _source_integral(source: Spectrum1D, lower: float, upper: float) -> float:
    """Local sums, not differences of large cumulative integrals in tiny tails."""
    a, b = _source_support(source)
    lower, upper = max(a, lower), min(b, upper)
    if lower >= upper:
        return 0.0
    binned = source.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE
    nodes = source.bin_edges if binned else source.x
    assert nodes is not None
    start = min(bisect_right(nodes, lower) - 1, len(nodes) - 2)
    terms = []
    for i in range(start, len(nodes) - 1):
        lo, hi = max(lower, nodes[i]), min(upper, nodes[i + 1])
        if hi > lo:
            if binned:
                value = source.y[i] * (hi - lo)
            else:
                yl = _linear(lo, nodes, source.y)
                yr = _linear(hi, nodes, source.y)
                value = (0.5 * yl + 0.5 * yr) * (hi - lo)
            _finite(value, "integral term")
            terms.append(value)
        if nodes[i + 1] >= upper:
            break
    result = fsum(terms)
    _finite(result, "source integral")
    return result


@dataclass(frozen=True)
class PrimaryTransportLedger:
    source_integral: float
    supported_exit_integral: float
    represented_integral: float
    below_output_window_integral: float
    above_output_window_integral: float
    outside_al_domain_integral: float
    unresolved_low_residual_integral: float
    balance_residual: float
    # This ledger covers only the source's declared finite interval.
    source_tails_known: bool = field(default=False, init=False)
    stopped_particle_classification_established: bool = field(default=False, init=False)


@dataclass(frozen=True)
class DiagnosticPrimaryTransport:
    source: Spectrum1D
    table: CandidateIonTable
    thickness_g_cm2: float
    spectrum: Spectrum1D
    ledger: PrimaryTransportLedger
    exit_support: tuple[float, float] | None
    _incident_support: tuple[float, float] | None = field(repr=False)
    normative_export_allowed: bool = field(default=False, init=False)

    def integral_on_exit(self, lower: float, upper: float) -> float:
        """Integral of the original source over supported exit interval preimages."""
        _finite(lower, "exit lower bound")
        _finite(upper, "exit upper bound")
        if upper < lower:
            raise ValueError("Exit bounds must be ordered.")
        if self.exit_support is None:
            return 0.0
        lo = max(lower, self.exit_support[0])
        hi = min(upper, self.exit_support[1])
        if hi <= lo:
            return 0.0
        if self.thickness_g_cm2 == 0:
            return _source_integral(self.source, lo, hi)
        assert self._incident_support is not None
        table = self.table
        exit_support = self.exit_support
        incident_support = self._incident_support

        def incident(exit_energy: float) -> float:
            if exit_energy == exit_support[0]:
                return incident_support[0]
            if exit_energy == exit_support[1]:
                return incident_support[1]
            ranges = table.relative_al_range
            value = _linear(exit_energy, table.al_energy, ranges) + self.thickness_g_cm2
            # Restrict to the previously established supported interval, not
            # to an extrapolated table or a physical zero-energy range.
            value = min(max(value, ranges[0]), ranges[-1])
            return _linear(value, ranges, table.al_energy)

        return _source_integral(self.source, incident(lo), incident(hi))


def _validate_source(source: Spectrum1D, table: CandidateIonTable, mass: float) -> None:
    _finite(mass, "declared source mass parameter", positive=True)
    if mass != table.mass_parameter_u:
        raise ValueError("Exact declared source/table mass parameters must match; no rounding.")
    if (source.particle is not Particle.HZE or source.species is None
            or source.species.is_aggregate or source.species.atomic_number != table.z):
        raise ValueError("An individual HZE spectrum with matching Z is required.")
    if source.source not in (RadiationSource.GCR, RadiationSource.SEP):
        raise ValueError("Only GCR/SEP HZE sources are in this candidate scope.")
    if (source.axis_kind is not SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
            or source.x_unit is not Unit.MEV_PER_NUCLEON):
        raise ValueError("Source must have an explicit MeV/nucleon energy axis.")
    if source.location not in (SpectrumLocation.ON_ORBIT, SpectrumLocation.OUTSIDE_MAGNETOSPHERE):
        raise ValueError("Source must be unshielded; repeated shielding is not implicit.")
    if source.assessment_mode not in (AssessmentMode.PEAK, AssessmentMode.CUMULATIVE):
        raise ValueError("Only explicit PEAK or CUMULATIVE inputs are supported.")
    if source.assessment_mode is AssessmentMode.PEAK:
        if source.quantity not in (SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
                                   SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX):
            raise ValueError("Peak inputs require a peak/maximum differential flux.")
    elif source.quantity is not SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        raise ValueError("Cumulative inputs require a differential fluence.")
    violations = tuple(v for v in source.production_contract_violations()
                       if v != "incomplete spectra cannot be released as production results")
    if violations:
        raise ValueError(f"Invalid typed source contract: {violations}")
    if source.ordinate_basis not in (SpectrumOrdinateBasis.POINT_SAMPLE,
                                     SpectrumOrdinateBasis.BIN_AVERAGE):
        raise ValueError("Unsupported source ordinate convention.")
    nodes = source.bin_edges if source.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE else source.x
    assert nodes is not None
    _grid(nodes, "source support")
    for value in source.y:
        _finite(value, "source ordinate")
    for step in source.transformation_chain:
        for key, text_value in step.parameters:
            if key == "source_tabulated_mass_parameter" and float(text_value) != mass:
                raise ValueError("Explicit source mass conflicts with source transformation trace.")


def transport_candidate_primary(
    source: Spectrum1D, table: CandidateIonTable, thickness_g_cm2: float, *,
    source_mass_parameter_u: float, output_edges: tuple[float, ...] | None = None,
) -> DiagnosticPrimaryTransport:
    """Conserve supported primary counts and expose unsupported mass separately.

    Spherical-shell-at-centre, straight-path CSDA only. No angular multiplier,
    density factor or mission time multiplier is applied to the source.
    At zero thickness the numerical energy spectrum is exactly the input;
    custom rebinning is rejected to make that identity unambiguous.
    """
    _validate_source(source, table, source_mass_parameter_u)
    _finite(thickness_g_cm2, "shield thickness")
    if thickness_g_cm2 == 0 and output_edges is not None:
        raise ValueError("Zero thickness preserves identity; do not request output rebinning.")
    edges = table.al_energy if output_edges is None else output_edges
    _grid(edges, "output energy edges")
    s0, s1 = _source_support(source)
    total = _source_integral(source, s0, s1)
    incident_support: tuple[float, float] | None = None
    exit_support: tuple[float, float] | None = None
    outside = unresolved = supported = 0.0
    if thickness_g_cm2 == 0:
        incident_support = exit_support = (s0, s1)
        supported = total
    else:
        e, r = table.al_energy, table.relative_al_range
        lo, hi = max(s0, e[0]), min(s1, e[-1])
        outside = fsum((_source_integral(source, s0, min(s1, e[0])),
                        _source_integral(source, max(s0, e[-1]), s1)))
        if lo < hi:
            if thickness_g_cm2 >= r[-1] - r[0]:
                unresolved = _source_integral(source, lo, hi)
            else:
                threshold = _linear(r[0] + thickness_g_cm2, r, e)
                start = max(lo, threshold)
                unresolved = _source_integral(source, lo, min(hi, start))
                if start < hi:
                    incident_support = (start, hi)
                    ranges = tuple(max(r[0], _linear(x, e, r) - thickness_g_cm2)
                                   for x in (start, hi))
                    exit_support = (_linear(ranges[0], r, e), _linear(ranges[1], r, e))
                    if exit_support[0] >= exit_support[1]:
                        raise ValueError("Supported exit interval collapsed at floating precision.")
                    supported = _source_integral(source, start, hi)
    empty_ledger = PrimaryTransportLedger(total, supported, 0, 0, 0, outside, unresolved, 0)
    measure = DiagnosticPrimaryTransport(source, table, thickness_g_cm2, source,
                                         empty_ledger, exit_support, incident_support)
    if thickness_g_cm2 == 0:
        represented, below, above = total, 0.0, 0.0
        x, y, bin_edges, basis = source.x, source.y, source.bin_edges, source.ordinate_basis
    else:
        counts = tuple(measure.integral_on_exit(a, b) for a, b in zip(edges, edges[1:]))
        represented = fsum(counts)
        below = measure.integral_on_exit(0, edges[0])
        above = (measure.integral_on_exit(edges[-1], max(edges[-1], exit_support[1]))
                 if exit_support is not None else 0.0)
        x = tuple(a + (b - a) / 2 for a, b in zip(edges, edges[1:]))
        y = tuple(c / (b - a) for c, a, b in zip(counts, edges, edges[1:]))
        bin_edges, basis = edges, SpectrumOrdinateBasis.BIN_AVERAGE
    for ordinate in y:
        _finite(ordinate, "output density")
    residual = fsum((represented, below, above, outside, unresolved, -total))
    if abs(residual) > 1e-10 * total:
        raise ValueError("Primary candidate count ledger failed closure.")
    ledger = PrimaryTransportLedger(total, supported, represented, below, above,
                                     outside, unresolved, residual)
    reasons = [*source.incomplete_reasons, DIAGNOSTIC_REASON]
    if outside > 0:
        reasons.append("Candidate source integral outside Al table support is unresolved.")
    if unresolved > 0:
        reasons.append("Candidate low residual range is unresolved, not classified as stopped.")
    if below + above > 0:
        reasons.append("Supported candidate exits lie outside the requested energy display window.")
    spectrum = derive_spectrum(
        source, x=x, y=y, bin_edges=bin_edges, ordinate_basis=basis,
        model="candidate_primary_al_csda_diagnostic", location=SpectrumLocation.BEHIND_SHIELD,
        shielding=ShieldingDescriptor("Al", thickness_g_cm2),
        additional_dataset_refs=(table.dataset_ref,), completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=tuple(dict.fromkeys(reasons)),
        transformation=TransformationStep("candidate_primary_al_csda", "1", (
            ("candidate_dataset", table.dataset_ref), ("mass_policy", table.mass_policy),
            ("mass_parameter_u", repr(table.mass_parameter_u)),
            ("thickness_g_cm2", repr(thickness_g_cm2)),
            ("interpolation", "piecewise_linear_range_and_source_point_density"),
            ("source_bin_average_policy", "piecewise_constant_density"),
            ("time_factor", "none"), ("angular_factor", "none"),
            ("range_origin", "not_qualified_only_supported_differences"),
        )),
    )
    return DiagnosticPrimaryTransport(source, table, thickness_g_cm2, spectrum, ledger,
                                      exit_support, incident_support)
