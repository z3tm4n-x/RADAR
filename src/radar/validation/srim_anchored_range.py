"""Diagnostic path-range coordinates derived from retained SRIM stopping nodes.

These are not native SRIM ranges. The zero at the first supplied kinetic energy
is only an anchor. No continuation to zero energy, isotope policy, material
qualification or physical stopping classification follows from this object.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from decimal import Context, Decimal, DecimalException, ROUND_HALF_EVEN, localcontext
from typing import Literal

from radar.validation.range_stopping_consistency import (
    _expm1_over_x, _integrals, _log1p_over_x, _log_ratio,
)
from radar.validation.srim_output import SrimRangeOutput


StoppingInterpolation = Literal["linear_stopping", "loglog_stopping"]
_CHOICES = ("linear_stopping", "loglog_stopping")
_WORK = Context(prec=110, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
_RESULT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
_ZERO = Decimal(0)
_ONE = Decimal(1)


def _scalar(value: Decimal, name: str, *, positive: bool = False) -> None:
    if (type(value) is not Decimal or not value.is_finite() or value < 0
            or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} Decimal.")
    if len(value.as_tuple().digits) > 80 or (value != 0 and abs(value.adjusted()) > 1000):
        raise ValueError(f"{name} exceeds the supported 80-digit/exponent-1000 arithmetic domain.")


def _rounded(value: Decimal) -> Decimal:
    if not value.is_finite():
        raise ValueError("Nonfinite anchored-range arithmetic result.")
    with localcontext(_RESULT):
        return +value


@dataclass(frozen=True)
class AnchoredSrimRangeTable:
    """A declared inter-node stopping model, independent of projected range.

    Constructors take only an already structurally validated native object and
    an explicit interpolation choice. Derived arrays cannot be injected through
    the constructor. Native byte authenticity remains the caller's pinned-reader
    responsibility; this class does not authenticate an executable or its files.
    """

    native: SrimRangeOutput = field(repr=False)
    interpolation: StoppingInterpolation
    ion_z: int = field(init=False)
    ion_mass_u: Decimal = field(init=False)
    native_raw_sha256: str = field(init=False)
    native_semantic_sha256: str = field(init=False)
    energy_total_mev: tuple[Decimal, ...] = field(init=False)
    total_stopping_mev_cm2_g: tuple[Decimal, ...] = field(init=False)
    anchored_range_g_cm2: tuple[Decimal, ...] = field(init=False)
    projected_range_cm: tuple[Decimal, ...] = field(init=False)
    native_projected_range_g_cm2: tuple[Decimal, ...] = field(init=False)
    _working_range_g_cm2: tuple[Decimal, ...] = field(init=False, repr=False)
    range_kind: str = field(default="derived_anchored_csda_path", init=False)
    native_range_kind: str = field(default="projected", init=False)
    energy_coordinate: str = field(default="total_ion_kinetic_energy_MeV", init=False)
    stopping_definition: str = field(default="electronic_plus_elastic_nuclear_MeV_cm2_g", init=False)
    range_unit: str = field(default="g/cm2", init=False)
    anchor_definition: str = field(default="R(first_observed_energy)=0_coordinate_offset_only",
                                   init=False)
    below_anchor_result: str = field(default="None_means_unknown_not_physically_stopped", init=False)
    precision_decimal_digits: int = field(default=80, init=False)
    diagnostic_status: str = field(default="INCOMPLETE", init=False)
    physical_accuracy_verified: bool = field(default=False, init=False)
    native_source_authenticity_verified: bool = field(default=False, init=False)
    isotope_selection_policy_established: bool = field(default=False, init=False)
    absolute_range_origin_established: bool = field(default=False, init=False)
    normative_export_allowed: bool = field(default=False, init=False)
    production_tables_replaced: bool = field(default=False, init=False)
    limitations: tuple[str, ...] = field(default=(
        "This is a derived integral of a declared stopping interpolant, not SRIM native range.",
        "Only observed Al energy support; no extrapolation or low-energy stopping origin.",
        "Native projected range is retained separately and never used to construct CSDA range.",
        "Native numerical mass in amu is retained without rounding or isotope qualification.",
        "Nuclear stopping means elastic recoil loss, not nuclear-reaction attenuation.",
        "Interpolation agreement and precision do not establish physical uncertainty bounds.",
        "A positive thickness smaller than range-coordinate resolution is rejected, not erased.",
    ), init=False)

    def __post_init__(self) -> None:
        if type(self.native) is not SrimRangeOutput:
            raise ValueError("A structurally validated SrimRangeOutput is required.")
        with localcontext(_WORK):
            self.native.__post_init__()
            semantic_sha256 = self.native.semantic_sha256
        if self.native.target_z != 13 or self.native.target_symbol != "Al":
            raise ValueError("Only single-element Al native controls are supported.")
        if self.native.ion_mass_u < self.native.ion_z:
            raise ValueError("Native mass parameter must be at least its nuclear charge Z.")
        if type(self.interpolation) is not str or self.interpolation not in _CHOICES:
            raise ValueError("Explicit linear_stopping or loglog_stopping is required.")
        energy = tuple(row.energy_total_mev for row in self.native.rows)
        try:
            with localcontext(_WORK):
                stopping = tuple(row.stopping_total_mev_cm2_mg * 1000 for row in self.native.rows)
                projected = tuple(row.projected_range_cm for row in self.native.rows)
                projected_mass = tuple(r * self.native.density_g_cm3 for r in projected)
                for value in (*energy, *stopping):
                    _scalar(value, "native energy/stopping", positive=True)
                working = [_ZERO]
                choice = 0 if self.interpolation == "linear_stopping" else 1
                for e0, e1, s0, s1 in zip(energy, energy[1:], stopping, stopping[1:]):
                    working.append(working[-1] + _integrals(e0, e1, s0, s1)[choice])
                ranges = tuple(_rounded(value) for value in working)
        except DecimalException as exc:
            raise ValueError("Unsupported anchored-range construction arithmetic.") from exc
        if any(b <= a for a, b in zip(ranges, ranges[1:])):
            raise ValueError("Anchored increments collapse at the declared 80-digit precision.")
        for key, value in (
            ("ion_z", self.native.ion_z), ("ion_mass_u", self.native.ion_mass_u),
            ("native_raw_sha256", self.native.raw_sha256),
            ("native_semantic_sha256", semantic_sha256),
            ("energy_total_mev", energy), ("total_stopping_mev_cm2_g", stopping),
            ("projected_range_cm", projected), ("native_projected_range_g_cm2", projected_mass),
            ("anchored_range_g_cm2", ranges), ("_working_range_g_cm2", tuple(working)),
        ):
            object.__setattr__(self, key, value)

    @classmethod
    def from_native(cls, native: SrimRangeOutput, *,
                    interpolation: StoppingInterpolation) -> AnchoredSrimRangeTable:
        return cls(native=native, interpolation=interpolation)

    def _energy_interval(self, energy: Decimal) -> int:
        _scalar(energy, "total energy", positive=True)
        if energy < self.energy_total_mev[0] or energy > self.energy_total_mev[-1]:
            raise ValueError("Energy is outside observed native support; no extrapolation.")
        return min(bisect_right(self.energy_total_mev, energy) - 1, len(self.energy_total_mev) - 2)

    def range_at_energy(self, energy_total_mev: Decimal) -> Decimal:
        """Return the anchored integral at supported TOTAL ion energy only."""
        index = self._energy_interval(energy_total_mev)
        exact = bisect_left(self.energy_total_mev, energy_total_mev)
        if self.energy_total_mev[exact] == energy_total_mev:
            return self.anchored_range_g_cm2[exact]
        e0, e1 = self.energy_total_mev[index:index + 2]
        s0, s1 = self.total_stopping_mev_cm2_g[index:index + 2]
        try:
            with localcontext(_WORK):
                if self.interpolation == "linear_stopping":
                    relative = (s1 - s0) / s0 * (energy_total_mev - e0) / (e1 - e0)
                    increment = (energy_total_mev - e0) / s0 * _log1p_over_x(relative)
                else:
                    p = _log_ratio(s0, s1) / _log_ratio(e0, e1)
                    log_energy = _log_ratio(e0, energy_total_mev)
                    increment = e0 / s0 * log_energy * _expm1_over_x((1 - p) * log_energy)
                result = _rounded(self._working_range_g_cm2[index] + increment)
        except DecimalException as exc:
            raise ValueError("Unsupported range evaluation arithmetic.") from exc
        if not self.anchored_range_g_cm2[index] < result < self.anchored_range_g_cm2[index + 1]:
            raise ValueError("Range evaluation cannot be represented within its supported interval.")
        return result

    def energy_at_range(self, anchored_range_g_cm2: Decimal) -> Decimal:
        """Invert the SAME analytic inter-node model, not a linear range table."""
        _scalar(anchored_range_g_cm2, "anchored range")
        if anchored_range_g_cm2 > self.anchored_range_g_cm2[-1]:
            raise ValueError("Range is outside observed native support; no extrapolation.")
        exact = bisect_left(self.anchored_range_g_cm2, anchored_range_g_cm2)
        if self.anchored_range_g_cm2[exact] == anchored_range_g_cm2:
            return self.energy_total_mev[exact]
        index = exact - 1
        e0, e1 = self.energy_total_mev[index:index + 2]
        s0, s1 = self.total_stopping_mev_cm2_g[index:index + 2]
        try:
            with localcontext(_WORK):
                delta = anchored_range_g_cm2 - self._working_range_g_cm2[index]
                if delta <= 0:
                    raise ValueError("Interior range increment is below representable support.")
                if self.interpolation == "linear_stopping":
                    slope = (s1 - s0) / (e1 - e0)
                    energy = e0 + s0 * delta * _expm1_over_x(slope * delta)
                else:
                    p = _log_ratio(s0, s1) / _log_ratio(e0, e1)
                    scaled = delta * s0 / e0
                    argument = (1 - p) * scaled
                    if argument <= -1:
                        raise ValueError("Inverse power-law argument is outside supported domain.")
                    energy = e0 * (scaled * _log1p_over_x(argument)).exp()
                result = _rounded(energy)
        except DecimalException as exc:
            raise ValueError("Unsupported inverse-range arithmetic.") from exc
        if not e0 < result < e1:
            raise ValueError("Inverse range cannot be represented within its supported interval.")
        return result

    def residual_energy(self, energy_total_mev: Decimal,
                        thickness_g_cm2: Decimal) -> Decimal | None:
        """Return supported exit energy, or None for unresolved below-anchor exit.

        A None result does not mean zero exit energy or that an ion has stopped.
        It is not a zero-filled spectrum. Equality with available anchored range
        returns the first OBSERVED energy, which is strictly positive.
        """
        self._energy_interval(energy_total_mev)
        _scalar(thickness_g_cm2, "thickness")
        if thickness_g_cm2 == 0:
            return energy_total_mev
        available = self.range_at_energy(energy_total_mev)
        if thickness_g_cm2 > available:
            return None
        if thickness_g_cm2 == available:
            return self.energy_total_mev[0]
        with localcontext(_WORK):
            remaining = _rounded(available - thickness_g_cm2)
        if remaining == available:
            raise ValueError("Positive thickness is below range-coordinate resolution.")
        result = self.energy_at_range(remaining)
        if result >= energy_total_mev:
            raise ValueError("Positive energy loss is below energy-coordinate resolution.")
        return result
