"""OST source/native SRIM inventory diagnostics, without selecting isotopes.

Observed intervals describe calculated table support only. They are not source
or shielded-spectrum coverage, and neither equal numerical masses nor a native
projected-range column qualifies a transport table. Raw-file and source-code
pins must be verified by the caller; a parsed object alone is not authenticity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from importlib import resources
from math import isfinite

from radar.data.integrity import validate_normative_resource_integrity
from radar.gcr.parameters import (
    get_ost_akl_parameters, get_ost_gcr_ion_parameters, get_ost_nek_parameter,
)
from radar.sep.hze_spectrum import OST_SEP_HZE_B9_RESOURCE, load_ost_sep_hze_ion_records
from radar.validation.srim_output import SrimRangeOutput


_CONTEXT = Context(prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
_MATERIALS = (13, 14)


@dataclass(frozen=True)
class NativeCoverageObservation:
    ion_z: int
    ion_name: str
    native_ion_mass_u: Decimal
    material_z: int
    native_raw_sha256: str
    native_semantic_sha256: str
    native_node_count: int
    energy_total_mev_bounds: tuple[Decimal, Decimal]
    total_energy_divided_by_native_mass: tuple[Decimal, Decimal]
    native_range_kind: str = field(default="projected", init=False)
    mass_policy: str = field(default="native_numeric_mass_parameter_no_isotope_selection", init=False)
    observed_interval_only: bool = field(default=True, init=False)
    normative_export_allowed: bool = field(default=False, init=False)


@dataclass(frozen=True)
class SourceNativeCoverageRow:
    source: str
    z: int
    symbol: str
    source_tabulated_mass_parameter: Decimal
    source_mass_to_charge: Decimal | None
    source_weight_parameter_name: str
    source_weight_parameter: Decimal
    nonzero_source_components: tuple[str, ...]
    material_z: int
    classification: tuple[str, ...]
    mass_comparison: str
    observation: NativeCoverageObservation | None
    total_energy_divided_by_source_mass_parameter: tuple[Decimal, Decimal] | None
    physical_mass_policy_established: bool = field(default=False, init=False)
    source_energy_domain_covered: bool = field(default=False, init=False)
    normative_export_allowed: bool = field(default=False, init=False)


@dataclass(frozen=True)
class NativeSourceCoverageSummary:
    source: str
    source_hze_species: int
    positive_hze_species: int
    declared_zero_species: int
    positive_species_with_any_material: int
    positive_species_with_both_materials: int
    missing_positive_species: int
    positive_species_missing_one_material: int
    positive_species_numeric_mass_equal_both_materials: int
    positive_species_with_any_numeric_mass_difference: int
    mass_policy_qualified_species_count: int = field(default=0, init=False)


@dataclass(frozen=True)
class NativeMaterialCoverageSummary:
    source: str
    material_z: int
    positive_species_observed: int
    positive_species_without_material: int
    positive_species_numeric_mass_equal: int
    positive_species_numeric_mass_different: int


@dataclass(frozen=True)
class SrimNativeCoverageAudit:
    native_observations: tuple[NativeCoverageObservation, ...]
    proton_observations: tuple[NativeCoverageObservation, ...]
    rows: tuple[SourceNativeCoverageRow, ...]
    summaries: tuple[NativeSourceCoverageSummary, ...]
    material_summaries: tuple[NativeMaterialCoverageSummary, ...]
    division_precision_decimal_digits: int = field(default=80, init=False)
    diagnostic_status: str = field(default="INCOMPLETE", init=False)
    native_authenticity_verified: bool = field(default=False, init=False)
    isotope_selection_policy_established: bool = field(default=False, init=False)
    source_energy_domain_covered: bool = field(default=False, init=False)
    shielded_transport_domain_covered: bool = field(default=False, init=False)
    absolute_range_origin_established: bool = field(default=False, init=False)
    normative_export_allowed: bool = field(default=False, init=False)
    production_tables_replaced: bool = field(default=False, init=False)
    limitations: tuple[str, ...] = field(default=(
        "Observed native intervals are not full source or shielded-transport coverage.",
        "T/native_mass and T/source_A are separate numerical coordinate comparisons only.",
        "Coordinate quotients are rounded to 80 Decimal digits; original bounds/masses are retained.",
        "Neither mass equality nor an integer native mass establishes an isotope distribution.",
        "Source A/Q is retained separately and is not replaced by A/Z.",
        "Source-declared zero means zero model coefficients, not an absent native calculation.",
        "Absent native data or failed calculations are never zero-valued physical contributions.",
        "Failed native-run details must remain in the caller's parent run ledger.",
        "Native projected range is not imported as a CSDA path range.",
        "Parsed objects do not authenticate raw bytes, executables or source documents.",
        "Caller must pin native reports and the current source definitions.",
    ), init=False)


@dataclass(frozen=True)
class _SourceIon:
    source: str
    z: int
    symbol: str
    mass: Decimal
    mass_to_charge: Decimal | None
    weight_name: str
    weight: Decimal
    components: tuple[str, ...]


def _number(value: float, label: str, *, positive: bool = False) -> Decimal:
    if (type(value) not in (int, float) or not isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f"Invalid literal OST {label} parameter.")
    return Decimal(str(value))


def _source_ions() -> tuple[_SourceIon, ...]:
    """Load literal current source parameters; never construct CandidateIsotope."""
    result = []
    akl = get_ost_akl_parameters()
    for ion in get_ost_gcr_ion_parameters().values():
        if ion.z == 1:
            continue
        nek = get_ost_nek_parameter(ion.z)
        # Validate all components before their signs are used as zero evidence.
        weight = _number(ion.c, "main C")
        nek_weight = Decimal(0) if nek is None else _number(nek.ac, "NEK ac")
        akl_weight = Decimal(0) if ion.symbol not in akl else _number(akl[ion.symbol].c, "AKL c")
        components = tuple(name for name, value in (
            ("main", weight), ("NEK", nek_weight), ("AKL", akl_weight),
        ) if value > 0)
        result.append(_SourceIon(
            "GCR", ion.z, ion.symbol, _number(ion.mass_number, "A", positive=True),
            None, "main_component_C", weight, components,
        ))
    validate_normative_resource_integrity(OST_SEP_HZE_B9_RESOURCE)
    resource = resources.files("radar.data.normative").joinpath(OST_SEP_HZE_B9_RESOURCE)
    with resources.as_file(resource) as path:
        sep = load_ost_sep_hze_ion_records(path)
    for sep_ion in sep:
        weight = _number(sep_ion.relative_abundance, "SEP K")
        result.append(_SourceIon(
            "SEP", sep_ion.z, sep_ion.symbol, _number(sep_ion.mass_number, "A", positive=True),
            _number(sep_ion.mass_to_charge, "A/Q", positive=True), "relative_abundance_K",
            weight, ("SEP",) if weight > 0 else (),
        ))
    for source, expected_z, expected_positive in (("GCR", set(range(2, 93)), 84),
                                                ("SEP", set(range(2, 29)), 24)):
        selected = [row for row in result if row.source == source]
        if (len(selected) != len(expected_z) or {row.z for row in selected} != expected_z
                or any(type(row.z) is not int or row.mass < row.z for row in selected)
                or sum(bool(row.components) for row in selected) != expected_positive):
            raise ValueError("OST source roster changed; explicit source re-audit required.")
    gcr = {row.z: row for row in result if row.source == "GCR"}
    if any(row.mass != gcr[row.z].mass or row.symbol != gcr[row.z].symbol
           for row in result if row.source == "SEP"):
        raise ValueError("Conflicting literal GCR/SEP mass or symbol parameters.")
    if any(row.components and not gcr[row.z].components for row in result if row.source == "SEP"):
        raise ValueError("Positive SEP species is missing from the positive GCR source roster.")
    return tuple(sorted(result, key=lambda row: (row.source, row.z)))


def _observation(native: SrimRangeOutput) -> NativeCoverageObservation:
    bounds = (native.rows[0].energy_total_mev, native.rows[-1].energy_total_mev)
    with localcontext(_CONTEXT):
        converted = (bounds[0] / native.ion_mass_u, bounds[1] / native.ion_mass_u)
    return NativeCoverageObservation(
        native.ion_z, native.ion_name, native.ion_mass_u, native.target_z,
        native.raw_sha256, native.semantic_sha256, len(native.rows), bounds, converted,
    )


def audit_srim_native_coverage(
    native_outputs: tuple[SrimRangeOutput, ...],
) -> SrimNativeCoverageAudit:
    """Observe one native mass/table per Z/material without choosing transport mass.

    Empty input is a valid all-missing audit. Duplicate material tables or
    contradictory Al/Si native masses are rejected instead of silently chosen.
    Caller retains failed-run reasons separately; no failure is parsed as a table.
    """
    if type(native_outputs) is not tuple or any(type(n) is not SrimRangeOutput for n in native_outputs):
        raise ValueError("Native outputs must be an immutable tuple of SrimRangeOutput objects.")
    observations = []
    for native in native_outputs:
        with localcontext(_CONTEXT):
            native.__post_init__()
        if native.target_z not in _MATERIALS:
            raise ValueError("Only Al and Si native materials are in scope.")
        if native.ion_mass_u < native.ion_z:
            raise ValueError("Native numeric mass must be at least ion Z.")
        observations.append(_observation(native))
    observations.sort(key=lambda row: (row.ion_z, row.material_z))
    inventory = {(row.ion_z, row.material_z): row for row in observations}
    if len(inventory) != len(observations):
        raise ValueError("Duplicate native Z/material; choose the control explicitly.")
    for z in {row.ion_z for row in observations}:
        if len({row.native_ion_mass_u for row in observations if row.ion_z == z}) != 1:
            raise ValueError("Conflicting native Al/Si masses for the same Z.")
    ions = _source_ions()
    rows = []
    present_z = {row.ion_z for row in observations}
    for ion in ions:
        for material in _MATERIALS:
            observation = inventory.get((ion.z, material))
            equal = observation is not None and ion.mass == observation.native_ion_mass_u
            mass_comparison = ("not_observed" if observation is None else
                               "numeric_equal_not_isotope_proof" if equal else
                               "numeric_difference_policy_unresolved")
            classification: tuple[str, ...]
            if not ion.components:
                classification = ("declared_zero",)
            elif ion.z not in present_z:
                classification = ("missing_positive_species",)
            elif observation is None:
                classification = ("missing_material",)
            else:
                classification = ("observed_energy_interval_only", "mass_policy_unresolved")
            converted = None
            if observation is not None:
                with localcontext(_CONTEXT):
                    lo, hi = observation.energy_total_mev_bounds
                    converted = (lo / ion.mass, hi / ion.mass)
            rows.append(SourceNativeCoverageRow(
                ion.source, ion.z, ion.symbol, ion.mass, ion.mass_to_charge, ion.weight_name,
                ion.weight, ion.components, material, classification, mass_comparison,
                observation, converted,
            ))
    summaries = []
    materials = []
    for source in ("GCR", "SEP"):
        selected_ions = [ion for ion in ions if ion.source == source]
        positive = [ion for ion in selected_ions if ion.components]
        observed = {ion.z: [inventory[(ion.z, m)] for m in _MATERIALS if (ion.z, m) in inventory]
                    for ion in positive}
        summaries.append(NativeSourceCoverageSummary(
            source, len(selected_ions), len(positive), len(selected_ions) - len(positive),
            sum(bool(observed[ion.z]) for ion in positive),
            sum(len(observed[ion.z]) == 2 for ion in positive),
            sum(not observed[ion.z] for ion in positive),
            sum(len(observed[ion.z]) == 1 for ion in positive),
            sum(len(observed[ion.z]) == 2 and all(o.native_ion_mass_u == ion.mass
                                                for o in observed[ion.z]) for ion in positive),
            sum(any(o.native_ion_mass_u != ion.mass for o in observed[ion.z]) for ion in positive),
        ))
        for material in _MATERIALS:
            positive_rows = [row for row in rows if row.source == source
                             and row.material_z == material and row.nonzero_source_components]
            materials.append(NativeMaterialCoverageSummary(
                source, material, sum(row.observation is not None for row in positive_rows),
                sum(row.observation is None for row in positive_rows),
                sum(row.mass_comparison == "numeric_equal_not_isotope_proof" for row in positive_rows),
                sum(row.mass_comparison == "numeric_difference_policy_unresolved"
                    for row in positive_rows),
            ))
    return SrimNativeCoverageAudit(
        tuple(row for row in observations if row.ion_z != 1),
        tuple(row for row in observations if row.ion_z == 1), tuple(rows),
        tuple(summaries), tuple(materials),
    )
