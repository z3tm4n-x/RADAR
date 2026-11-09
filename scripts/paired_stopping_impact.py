"""Diagnostic finite-source factorial transport; not a joined SRIM/CATIMA model.

SS/CS/SC/CC vary Al path loss and Si electronic LET independently on exactly
the common *incident* interval. A fifth, pure CATIMA calculation follows
only particles starting above that interval. They may exit below the seam.
Nothing here qualifies a model, absolute range, stopped count or extrapolation.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from math import expm1, fsum, inf, isfinite, log1p, nextafter
from typing import Any

from radar.core.spectra import Spectrum1D
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationSource, SpectrumAxisKind, SpectrumLocation,
    SpectrumOrdinateBasis, SpectrumQuantity,
)
from radar.core.units import Unit
from radar.validation.candidate_transport import _finite, _grid, _source_integral, _source_support

ENERGY_THRESHOLDS = (1., 10., 100., 1000., 5000., 10000., 50000.)
LET_THRESHOLDS = (.005, .0075, .01, .02, .05, .1, .2, .5, 1., 1.4, 1.53, 2., 5., 10., 17.5, 20., 50.)
ENERGY_EDGES = tuple(sorted({0., *ENERGY_THRESHOLDS, *(10 ** (i/10) for i in range(-30, 51))}))
LET_EDGES = tuple(sorted({0., *LET_THRESHOLDS, *(10 ** (i/10) for i in range(-40, 31))}))
BOUNDARY = 50000.


def _stopping(grid: tuple[float, ...], values: tuple[float, ...]) -> None:
    _grid(grid, "paired energy grid")
    if not isinstance(values, tuple) or len(values) != len(grid):
        raise ValueError("Exactly one value per energy node is required.")
    for value in values:
        _finite(value, "stopping", positive=True)


@dataclass(frozen=True)
class Segments:
    energy: tuple[float, ...]
    left: tuple[float, ...]  # at left edge, approached from inside segment
    right: tuple[float, ...]

    @classmethod
    def build(cls, grid: tuple[float, ...], values: tuple[float, ...]) -> Segments:
        _stopping(grid, values)
        guard = nextafter(BOUNDARY, inf)
        if grid[0] < BOUNDARY < grid[-1]:
            if BOUNDARY not in grid or guard not in grid:
                raise ValueError("An extended grid must retain both sides of the radiative threshold.")
            i = grid.index(BOUNDARY)
            if grid[i + 1] != guard or i + 2 >= len(grid):
                raise ValueError("The one-sided guard must precede an actual upper interval.")
            # The T+1ULP sample represents the right-hand limit at T. The
            # tiny gap is not an independent segment with a fabricated range
            # increment. Below T use the measured left value; above use right.
            keep = [k for k, e in enumerate(grid) if e not in (nextafter(BOUNDARY, -inf), guard)]
            if not keep or keep[0] != 0 or keep[-1] != len(grid)-1:
                raise ValueError("One-sided guards may not replace window endpoints.")
            return cls(tuple(grid[k] for k in keep),
                       tuple(values[i+1] if k == i else values[k] for k in keep[:-1]),
                       tuple(values[k] for k in keep[1:]))
        return cls(grid, values[:-1], values[1:])


@dataclass(frozen=True)
class PathRange:
    """Anchored integral of A/S over energy per source A; no projected range."""
    curve: Segments
    mass: float
    one_minus_power: tuple[float, ...]
    ranges: tuple[float, ...]

    @classmethod
    def build(cls, grid: tuple[float, ...], values: tuple[float, ...], mass: float) -> PathRange:
        _finite(mass, "source A", positive=True)
        curve = Segments.build(grid, values)
        powers, increments, ranges = [], [], [0.]
        for x, y, s, t in zip(curve.energy, curve.energy[1:], curve.left, curve.right):
            h = log1p((y - x) / x)
            k = 1 - log1p((t - s) / s) / h
            u = k*h
            inc = mass * x / s * h * (expm1(u)/u if u else 1.)
            powers.append(k)
            increments.append(inc)
            r = fsum(increments)
            if not isfinite(r) or r <= ranges[-1]:
                raise ValueError("Anchored range increment is not representable; no epsilon repair.")
            ranges.append(r)
        return cls(curve, mass, tuple(powers), tuple(ranges))

    def at(self, energy: float) -> float:
        grid = self.curve.energy
        if not grid[0] <= energy <= grid[-1]:
            raise ValueError("Energy outside Al stopping support.")
        i = bisect_right(grid, energy) - 1
        if energy == grid[i]:
            return self.ranges[i]
        h = log1p((energy - grid[i]) / grid[i])
        u = self.one_minus_power[i] * h
        return self.ranges[i] + self.mass * grid[i] / self.curve.left[i] * h * (expm1(u)/u if u else 1.)

    def invert(self, coordinate: float) -> float:
        if not self.ranges[0] <= coordinate <= self.ranges[-1]:
            raise ValueError("Path coordinate outside Al stopping support.")
        i = bisect_right(self.ranges, coordinate) - 1
        x = self.curve.energy[i]
        if coordinate == self.ranges[i]:
            return x
        scaled = (coordinate-self.ranges[i]) * self.curve.left[i] / (self.mass*x)
        u = self.one_minus_power[i] * scaled
        result = x + x * expm1(scaled * (log1p(u)/u if u else 1.))
        if not x <= result <= self.curve.energy[i+1]:
            raise ValueError("Represented inverse escaped its stopping segment.")
        return result


@dataclass(frozen=True)
class PreparedPair:
    z: int
    mass: float
    srim_al: PathRange
    catima_al: PathRange
    srim_si: Segments
    catima_si: Segments
    extended_al: PathRange
    extended_si: Segments


def prepare_pair(z: int, mass: float, common_energy: tuple[float, ...], srim_al_total: tuple[float, ...],
                 catima_al_total: tuple[float, ...], srim_si_electronic: tuple[float, ...],
                 catima_si_electronic: tuple[float, ...], extended_energy: tuple[float, ...],
                 extended_catima_al_total: tuple[float, ...], extended_catima_si_electronic: tuple[float, ...],
                 ) -> PreparedPair:
    """All stopping input arrays are whole-projectile MeV cm2/g. Si is electronic only."""
    _finite(mass, "literal source A", positive=True)
    if type(z) is not int or not 2 <= z <= 92 or mass < z:
        raise ValueError("Invalid ion or literal source mass.")
    n = len(common_energy)
    if (len(extended_energy) <= n or extended_energy[:n] != common_energy
            or extended_catima_al_total[:n] != catima_al_total
            or extended_catima_si_electronic[:n] != catima_si_electronic):
        raise ValueError("CATIMA upper-only calculation must retain its unchanged CATIMA common prefix.")

    def si(grid: tuple[float, ...], values: tuple[float, ...]) -> Segments:
        _stopping(grid, values)
        return Segments.build(grid, tuple(value/1000 for value in values))

    if extended_energy[-1] > ENERGY_EDGES[-1]:
        raise ValueError("The declared output energy bins do not cover the stopping domain.")
    result = PreparedPair(z, mass, PathRange.build(common_energy, srim_al_total, mass),
        PathRange.build(common_energy, catima_al_total, mass), si(common_energy, srim_si_electronic),
        si(common_energy, catima_si_electronic), PathRange.build(extended_energy, extended_catima_al_total, mass),
        si(extended_energy, extended_catima_si_electronic))
    if any(max((*s.left, *s.right)) > LET_EDGES[-1] for s in (result.srim_si, result.extended_si)):
        raise ValueError("The declared output LET bins do not cover the electronic stopping domain.")
    return result


def _source(source: Spectrum1D, pair: PreparedPair) -> None:
    if (not isinstance(source, Spectrum1D) or source.particle is not Particle.HZE or source.species is None
            or source.species.is_aggregate or source.species.atomic_number != pair.z
            or source.source not in (RadiationSource.GCR, RadiationSource.SEP)
            or source.axis_kind is not SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
            or source.x_unit is not Unit.MEV_PER_NUCLEON or source.angular_basis is not AngularBasis.OMNIDIRECTIONAL_4PI
            or source.location not in (SpectrumLocation.ON_ORBIT, SpectrumLocation.OUTSIDE_MAGNETOSPHERE)
            or source.shielding is not None
            or source.ordinate_basis not in (SpectrumOrdinateBasis.POINT_SAMPLE, SpectrumOrdinateBasis.BIN_AVERAGE)):
        raise ValueError("An individual unshielded omni GCR/SEP source with matching Z and energy axis is required.")
    if source.assessment_mode is AssessmentMode.PEAK:
        if source.quantity not in (SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX, SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX):
            raise ValueError("Peak calculation requires an actual peak/maximum flux.")
    elif source.assessment_mode is not AssessmentMode.CUMULATIVE or source.quantity is not SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        raise ValueError("Only PEAK flux and CUMULATIVE fluence are supported.")
    if any(v != "incomplete spectra cannot be released as production results" for v in source.production_contract_violations()):
        raise ValueError("Invalid source spectrum contract.")
    masses = [float(v) for step in source.transformation_chain for key, v in step.parameters
              if key == "source_tabulated_mass_parameter"]
    if not masses or any(m != pair.mass for m in masses):
        raise ValueError("Missing or conflicting literal source mass trace; do not infer A from species label.")


def _close(part: float, total: float, name: str) -> None:
    if not isfinite(part) or not isfinite(total) or abs(part-total) > 1e-10*total:
        raise ValueError(f"{name} count ledger failed: {part} versus {total}")


def _variant(source: Spectrum1D, al: PathRange, si: Segments, thickness: float,
             lower: float, upper: float, seam: float) -> dict[str, Any]:
    s0, s1 = _source_support(source)
    lo, hi = max(lower, s0), min(upper, s1)
    count = _source_integral(source, lo, hi)
    unresolved, supported = count, 0.
    exit_lo = exit_hi = 0.
    start = lo
    if hi > lo and thickness < al.at(hi):
        rlo = al.at(lo)
        start = lo if thickness <= rlo else al.invert(thickness)
        unresolved = _source_integral(source, lo, start)
        supported = _source_integral(source, start, hi)
        exit_lo = (lo if thickness == 0 else al.curve.energy[0] if thickness >= rlo
                   else al.invert(rlo-thickness))
        exit_hi = hi if thickness == 0 else al.invert(al.at(hi)-thickness)
        if start >= hi or exit_lo >= exit_hi:
            raise ValueError("Supported incident/exit interval collapsed at floating precision.")
        if thickness > 0 and (exit_hi >= hi or (thickness < rlo and exit_lo >= lo)):
            raise ValueError("Positive shield is below represented energy/range resolution.")
    _close(fsum((supported, unresolved)), count, "partition")

    @lru_cache(maxsize=None)
    def inverse(exit_energy: float) -> float:
        if exit_energy == exit_lo:
            return start
        if exit_energy == exit_hi:
            return hi
        if thickness == 0:
            return exit_energy
        result = al.invert(al.at(exit_energy) + thickness)
        if not start <= result <= hi or result <= exit_energy:
            raise ValueError("Interior preimage escaped supported interval or erased positive thickness.")
        return result

    def integrate(a: float, b: float) -> float:
        a, b = max(a, exit_lo), min(b, exit_hi)
        if a >= b:
            return 0.
        return _source_integral(source, inverse(a), inverse(b))

    energy_bins = [integrate(a, b) for a, b in zip(ENERGY_EDGES, ENERGY_EDGES[1:])]
    _close(fsum(energy_bins), supported, "energy histogram")
    pieces: list[list[float]] = [[] for _ in LET_EDGES[:-1]]
    for e0, e1, l0, l1 in zip(si.energy, si.energy[1:], si.left, si.right):
        if exit_hi <= e0 or exit_lo >= e1:
            continue
        if l0 == l1:
            i = min(len(pieces)-1, bisect_right(LET_EDGES, l0)-1)
            pieces[i].append(integrate(e0, e1))
            continue
        lmin, lmax = min(l0, l1), max(l0, l1)
        first, end = bisect_right(LET_EDGES, lmin)-1, bisect_left(LET_EDGES, lmax)

        def root(value: float) -> float:
            if value == l0:
                return e0
            if value == l1:
                return e1
            return e0 + (value-l0)/(l1-l0)*(e1-e0)

        for i in range(first, end):
            a, b = root(max(LET_EDGES[i], lmin)), root(min(LET_EDGES[i+1], lmax))
            pieces[i].append(integrate(min(a, b), max(a, b)))
    let_bins = [fsum(values) for values in pieces]
    _close(fsum(let_bins), supported, "electronic LET histogram")
    return {"source_partition_integral": count, "supported_exit_integral": supported,
            "unresolved_low_residual_integral": unresolved, "exit_support": [exit_lo, exit_hi] if exit_hi else None,
            "below_seam_exit_integral": integrate(0., seam), "energy_bin_integrals": energy_bins,
            "let_bin_integrals": let_bins,
            "energy_bin_average_density": [v/(b-a) for v, a, b in zip(energy_bins, ENERGY_EDGES, ENERGY_EDGES[1:])],
            "let_bin_average_density": [v/(b-a) for v, a, b in zip(let_bins, LET_EDGES, LET_EDGES[1:])],
            "energy_histogram_balance_residual": fsum(energy_bins)-supported,
            "let_histogram_balance_residual": fsum(let_bins)-supported,
            "energy_ge": {str(t): fsum(energy_bins[ENERGY_EDGES.index(t):]) for t in ENERGY_THRESHOLDS},
            "let_ge": {str(t): fsum(let_bins[LET_EDGES.index(t):]) for t in LET_THRESHOLDS}}


def evaluate_case(source: Spectrum1D, pair: PreparedPair, thickness: float) -> dict[str, Any]:
    """Report finite-source counts per cm² (per s for peak); no multiplication by SAS."""
    _source(source, pair)
    _finite(thickness, "Al shield g/cm2")
    lo, seam = pair.srim_al.curve.energy[0], pair.srim_al.curve.energy[-1]
    top = pair.extended_al.curve.energy[-1]
    start, end = _source_support(source)
    total = _source_integral(source, start, end)
    parts = {"below_common": _source_integral(source, start, lo),
             "common": _source_integral(source, lo, seam), "upper_catima_only": _source_integral(source, seam, top),
             "above_catima": _source_integral(source, top, end)}
    _close(fsum(parts.values()), total, "finite source")
    variants = {a+s: _variant(source, al, si, thickness, lo, seam, seam) for a, al in
                (("S", pair.srim_al), ("C", pair.catima_al)) for s, si in (("S", pair.srim_si), ("C", pair.catima_si))}
    upper = _variant(source, pair.extended_al, pair.extended_si, thickness, seam, top, seam)
    return {"status": "INCOMPLETE", "normative_export_allowed": False, "physical_accuracy_verified": False,
            "absolute_range_origin_established": False, "source_outer_tails_known": False,
            "time_factor": "none", "angular_multiplier": "none", "source_integral": total,
            "integral_unit": Unit.INTEGRAL_FLUX.value if source.assessment_mode is AssessmentMode.PEAK
            else Unit.INTEGRAL_FLUENCE.value, "energy_axis_unit": Unit.MEV_PER_NUCLEON.value,
            "let_axis_unit": Unit.LET.value, "input_partition": parts,
            "zero_shield_identity_unmapped_energy_integral": (parts["below_common"]+parts["above_catima"]
                                                          if thickness == 0 else None),
            "zero_shield_energy_identity": True if thickness == 0 else None,
            "variants": variants, "upper_catima_only": upper}
