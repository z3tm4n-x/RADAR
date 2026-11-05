"""Opt-in SRIM HZE energy/LET products with explicit unsupported contributions.

Consumes real upstream GCR/SEP products, not a new source or orbit model. It
does not replace the default normative profile or fall back to legacy tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

from radar.core.products import SpectrumProduct
from radar.core.spectra import ShieldingDescriptor, Spectrum1D, TransformationStep
from radar.core.spectrum_ops import derive_spectrum, mark_spectrum_incomplete
from radar.core.types import (
    AssessmentMode, Particle, RadiationProductKind, RadiationSource,
    SpectrumAxisKind, SpectrumCompleteness, SpectrumLocation, SpectrumQuantity,
)
from radar.core.units import Unit
from radar.shielding.primary_let import (
    ElectronicLetResult, SRIM_ELECTRONIC_LET_PROVENANCE, convert_primary_electronic_let,
)
from radar.shielding.srim_hze import (
    SrimIonStoppingTable, SrimPrimaryTransport, transport_srim_primary,
)

_LET_KINDS = {
    RadiationProductKind.MAXIMUM_FLUX: RadiationProductKind.MAXIMUM_LET_FLUX,
    RadiationProductKind.PEAK_FLUX: RadiationProductKind.PEAK_LET_FLUX,
    RadiationProductKind.MISSION_FLUENCE: RadiationProductKind.MISSION_LET_FLUENCE,
}


@dataclass(frozen=True)
class SrimHzeProductCase:
    source_kind: RadiationProductKind
    source: RadiationSource
    z: int
    thickness_g_cm2: float
    energy_product: SpectrumProduct | None
    let_product: SpectrumProduct | None
    transport: SrimPrimaryTransport | None
    let_result: ElectronicLetResult | None
    incomplete_reasons: tuple[str, ...]
    completeness: SpectrumCompleteness = field(default=SpectrumCompleteness.INCOMPLETE, init=False)
    normative_export_allowed: bool = field(default=False, init=False)


@dataclass(frozen=True)
class SrimHzePipelineResult:
    assessment_mode: AssessmentMode
    cases: tuple[SrimHzeProductCase, ...]
    normative_export_allowed: bool = field(default=False, init=False)
    completeness: SpectrumCompleteness = field(default=SpectrumCompleteness.INCOMPLETE, init=False)

    @property
    def products(self) -> tuple[SpectrumProduct, ...]:
        return tuple(product for case in self.cases for product in
                     (case.energy_product, case.let_product) if product is not None)

    @property
    def incomplete_reasons(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(reason for case in self.cases for reason in case.incomplete_reasons))


def _mass(source: Spectrum1D) -> float:
    values = []
    for step in source.transformation_chain:
        for key, text in step.parameters:
            if key == "source_tabulated_mass_parameter":
                value = float(text)
                if not isfinite(value) or value <= 0:
                    raise ValueError("Source mass trace must be positive and finite.")
                values.append(value)
    if not values or len(set(values)) != 1:
        raise ValueError("A consistent explicit source_tabulated_mass_parameter is required.")
    # The trace uses .17g float formatting. Species.mass_number is only the
    # old integer label and must not silently replace the literal source A.
    return values[0]


def _zero_identity(product: SpectrumProduct, reason: str) -> SpectrumProduct:
    spectrum = derive_spectrum(
        product.spectrum, location=SpectrumLocation.BEHIND_SHIELD,
        shielding=ShieldingDescriptor("Al", 0.0),
        transformation=TransformationStep("srim_zero_al_identity_without_table", "1",
                                         (("thickness_g_cm2", "0"),)),
    )
    return SpectrumProduct(product.kind, mark_spectrum_incomplete(spectrum, reason),
                           f"{product.label or product.kind.value}; Al 0 g/cm2; SRIM incomplete")


def calculate_srim_hze_products(
    *, products: tuple[SpectrumProduct, ...], tables: Mapping[int, SrimIonStoppingTable],
    thicknesses_g_cm2: tuple[float, ...], assessment_mode: AssessmentMode,
    let_edges: tuple[float, ...], output_energy_edges: tuple[float, ...] | None = None,
) -> SrimHzePipelineResult:
    """Return partial primary energy and electronic-LET products for one mode.

    Missing ion -> explicit case with no fabricated spectrum. At zero shield
    the energy identity needs no stopping table; LET still does. Integral
    ledgers belong to the original finite source interval, not unknown tails
    beyond it. Fluxes are never multiplied by the mission duration here.
    """
    if not isinstance(assessment_mode, AssessmentMode):
        raise ValueError("An explicit PEAK/CUMULATIVE mode is required.")
    if not isinstance(thicknesses_g_cm2, tuple) or not thicknesses_g_cm2:
        raise ValueError("A nonempty immutable shield list is required.")
    for thickness in thicknesses_g_cm2:
        if (isinstance(thickness, bool) or not isinstance(thickness, (int, float))
                or not isfinite(thickness) or thickness < 0):
            raise ValueError("Shield thickness must be finite and nonnegative.")
    if len(set(thicknesses_g_cm2)) != len(thicknesses_g_cm2):
        raise ValueError("Duplicate shield thicknesses are not permitted.")
    for z, supplied_table in tables.items():
        if (type(z) is not int or not isinstance(supplied_table, SrimIonStoppingTable)
                or supplied_table.z != z):
            raise ValueError("SRIM table mapping key must match its ion.")
    # Validate even if every requested table is absent: malformed output grids
    # must not be hidden behind an INCOMPLETE result.
    for grid, zero in ((let_edges, True), (output_energy_edges, False)):
        if grid is None and not zero:
            continue
        if not isinstance(grid, tuple) or len(grid) < 2:
            raise ValueError("Output grids need at least two immutable edges.")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x)
               or (x < 0 if zero else x <= 0) for x in grid):
            raise ValueError("Invalid output grid edges.")
        if any(a >= b for a, b in zip(grid, grid[1:])):
            raise ValueError("Output grid edges must strictly increase.")
    selected = []
    seen = set()
    for product in products:
        if not isinstance(product, SpectrumProduct):
            raise ValueError("Typed SpectrumProduct inputs are required.")
        source = product.spectrum
        if source.particle is not Particle.HZE or source.assessment_mode is not assessment_mode:
            continue
        # Real upstream sets also contain matching integral products. Those are
        # alternate presentations, not additional differential particle sources.
        if source.quantity not in (SpectrumQuantity.DIFFERENTIAL_FLUENCE,
                                   SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
                                   SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX):
            continue
        if (source.source not in (RadiationSource.GCR, RadiationSource.SEP)
                or source.species is None or source.species.is_aggregate
                or source.species.atomic_number is None or product.kind not in _LET_KINDS
                or source.axis_kind is not SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
                or source.x_unit is not Unit.MEV_PER_NUCLEON
                or source.location not in (SpectrumLocation.ON_ORBIT,
                                           SpectrumLocation.OUTSIDE_MAGNETOSPHERE)):
            raise ValueError("Individual unshielded GCR/SEP energy products are required.")
        violations = tuple(v for v in source.production_contract_violations()
                           if v != "incomplete spectra cannot be released as production results")
        if violations:
            raise ValueError(f"Invalid SRIM upstream spectrum: {violations}")
        key = source.source, source.species.atomic_number, product.kind
        if key in seen:
            raise ValueError("Duplicate upstream source/ion/product.")
        seen.add(key)
        selected.append((product, _mass(source)))
    if not selected:
        raise ValueError("No individual HZE source products match the selected mode.")
    cases = []
    for product, mass in selected:
        source = product.spectrum
        assert source.species is not None and source.species.atomic_number is not None
        z = source.species.atomic_number
        table = tables.get(z)
        if table is not None and mass != table.mass_parameter_u:
            raise ValueError("Source/table mass mismatch; isotope rounding is forbidden.")
        for thickness in thicknesses_g_cm2:
            if table is None:
                reason = f"SRIM HZE table pair is unavailable for Z={z}; no legacy fallback or zero fill."
                energy = _zero_identity(product, reason) if thickness == 0 else None
                cases.append(SrimHzeProductCase(
                    product.kind, source.source, z, thickness, energy, None, None, None,
                    tuple(dict.fromkeys((*source.incomplete_reasons, reason))),
                ))
                continue
            transport = transport_srim_primary(
                source, table, thickness, source_mass_parameter_u=mass,
                output_edges=None if thickness == 0 else output_energy_edges,
            )
            let = convert_primary_electronic_let(
                transport, let_edges=let_edges, provenance=SRIM_ELECTRONIC_LET_PROVENANCE,
            )
            energy_product = SpectrumProduct(product.kind, transport.spectrum,
                                             f"{source.species.symbol}; SRIM Al {thickness:g} g/cm2")
            let_product = SpectrumProduct(_LET_KINDS[product.kind], let.spectrum,
                                          f"{source.species.symbol}; SRIM electronic LET; Al {thickness:g} g/cm2")
            cases.append(SrimHzeProductCase(
                product.kind, source.source, z, thickness, energy_product, let_product, transport, let,
                tuple(dict.fromkeys((*transport.spectrum.incomplete_reasons, *let.spectrum.incomplete_reasons))),
            ))
    return SrimHzePipelineResult(assessment_mode, tuple(cases))
