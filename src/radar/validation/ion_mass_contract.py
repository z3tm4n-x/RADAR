"""Diagnostic OST source/transport mass inventory, without choosing isotopes.

The fractional A printed in OST tables B.9/V.1 is preserved as a source-model
parameter. An explicit integer candidate A describes a declared isotope; even
numeric equality does not establish an isotope distribution in the source model.
This audit does not change a spectrum, production table, or release gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Literal

from radar.data.integrity import validate_normative_resource_integrity
from radar.gcr.parameters import (
    get_ost_akl_parameters,
    get_ost_gcr_ion_parameters,
    get_ost_nek_parameter,
)
from radar.sep.hze_spectrum import OST_SEP_HZE_B9_RESOURCE, load_ost_sep_hze_ion_records


MassClassification = Literal[
    "missing",
    "exact_numeric_mass_not_isotope_proof",
    "different_mass_parameter_policy_unresolved",
    "declared_zero",
]

SOURCE_DEFINITION_REFERENCES = {
    "GCR": {
        "document": "OST 134-1044-2007, Appendix V, tables V.1/V.2/V.3",
        "parameters": "radar.gcr.parameters.get_ost_gcr_ion_parameters",
        "metadata": "radar.gcr.source_spectra._make_spectrum",
        "zero_rule": "main C=0 and no nonzero NEK ac or AKL c component",
    },
    "SEP": {
        "document": "OST 134-1044-2007, Appendix B, table B.9",
        "parameters": "radar.sep.hze_spectrum.load_ost_sep_hze_ion_records",
        "resource": OST_SEP_HZE_B9_RESOURCE,
        "metadata": "radar.sep.model._make_sep_hze_spectrum and _hze_scaling_step",
        "zero_rule": "relative_abundance K=0 in table B.9",
    },
}


@dataclass(frozen=True)
class CandidateIsotope:
    """One explicitly supplied isotope per element; no rounded-A selection."""

    z: int
    mass_number: int
    symbol: str

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.z, self.mass_number)):
            raise ValueError("Candidate Z and isotope A must be integers, not bool or float.")
        if self.z < 1 or self.mass_number < self.z:
            raise ValueError("Candidate isotope requires Z >= 1 and A >= Z.")
        known = {ion.z: ion.symbol for ion in get_ost_gcr_ion_parameters().values()}
        if self.z not in known or self.symbol != known[self.z]:
            raise ValueError("Candidate symbol must exactly match the declared Z (H through U).")


@dataclass(frozen=True)
class IonMassAuditRow:
    source: str
    z: int
    symbol: str
    source_tabulated_mass_parameter: float
    source_mass_to_charge: float | None
    source_weight_parameter_name: str
    source_weight_parameter: float
    nonzero_source_components: tuple[str, ...]
    current_rounded_metadata_mass_number: int
    candidate_isotope_mass_number: int | None
    classification: MassClassification
    rounded_metadata_hides_mass_difference: bool


@dataclass(frozen=True)
class SourceMassSummary:
    source: str
    table_hze_row_count: int
    positive_hze_count: int
    declared_zero_count: int
    candidate_present_positive_count: int
    missing_positive_count: int
    exact_numeric_mass_not_isotope_proof_count: int
    different_mass_parameter_policy_unresolved_count: int
    rounded_metadata_hides_mass_difference_count: int


@dataclass(frozen=True)
class IonMassContractAudit:
    candidate_inventory: tuple[CandidateIsotope, ...]
    rows: tuple[IonMassAuditRow, ...]
    summaries: tuple[SourceMassSummary, ...]
    candidate_z_outside_hze_rosters: tuple[int, ...]
    normative_export_allowed: bool = field(default=False, init=False)
    isotope_selection_policy_established: bool = field(default=False, init=False)
    production_tables_replaced: bool = field(default=False, init=False)
    scope: str = field(default="OST_HZE_source_mass_and_declared_candidate_inventory_only", init=False)


def _row(
    *, source: str, z: int, symbol: str, mass: float, mass_to_charge: float | None,
    weight_name: str, weight: float, components: tuple[str, ...],
    inventory: dict[int, CandidateIsotope],
) -> IonMassAuditRow:
    candidate = inventory.get(z)
    candidate_a = None if candidate is None else candidate.mass_number
    # Report current production metadata behavior, never select a transport isotope by it.
    metadata_a = int(round(mass))
    classification: MassClassification
    if not components:
        classification = "declared_zero"
    elif candidate is None:
        classification = "missing"
    elif mass == candidate_a:
        classification = "exact_numeric_mass_not_isotope_proof"
    else:
        classification = "different_mass_parameter_policy_unresolved"
    return IonMassAuditRow(
        source, z, symbol, mass, mass_to_charge, weight_name, weight, components,
        metadata_a, candidate_a, classification,
        classification == "different_mass_parameter_policy_unresolved" and metadata_a == candidate_a,
    )


def audit_ost_ion_mass_contract(
    candidate_inventory: Sequence[CandidateIsotope],
) -> IonMassContractAudit:
    """Compare actual OST source rosters with a supplied isotope inventory.

    Declared-zero rows are retained and proved from model coefficients, not from
    sampled fluxes. Exact numeric masses are not a physical isotope assignment.
    Candidate presence checks only inventory, not material/energy table coverage.
    """
    candidates = tuple(candidate_inventory)
    if not candidates or any(not isinstance(ion, CandidateIsotope) for ion in candidates):
        raise ValueError("Candidate inventory must contain explicit CandidateIsotope records.")
    if len({ion.z for ion in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate Z: explicitly supply one isotope per element.")
    candidates = tuple(sorted(candidates, key=lambda ion: ion.z))
    inventory = {ion.z: ion for ion in candidates}
    rows: list[IonMassAuditRow] = []
    akl = get_ost_akl_parameters()
    for ion in get_ost_gcr_ion_parameters().values():
        if ion.z == 1:
            continue
        nek = get_ost_nek_parameter(ion.z)
        components = tuple(name for name, positive in (
            ("main", ion.c > 0.0),
            ("NEK", nek is not None and nek.ac > 0.0),
            ("AKL", ion.symbol in akl and akl[ion.symbol].c > 0.0),
        ) if positive)
        rows.append(_row(
            source="GCR", z=ion.z, symbol=ion.symbol, mass=ion.mass_number,
            mass_to_charge=None, weight_name="main_component_C", weight=ion.c,
            components=components, inventory=inventory,
        ))
    validate_normative_resource_integrity(OST_SEP_HZE_B9_RESOURCE)
    resource = resources.files("radar.data.normative").joinpath(OST_SEP_HZE_B9_RESOURCE)
    with resources.as_file(resource) as path:
        sep_records = load_ost_sep_hze_ion_records(path)
    for sep_ion in sep_records:
        rows.append(_row(
            source="SEP", z=sep_ion.z, symbol=sep_ion.symbol, mass=sep_ion.mass_number,
            mass_to_charge=sep_ion.mass_to_charge, weight_name="relative_abundance_K",
            weight=sep_ion.relative_abundance,
            components=("SEP",) if sep_ion.relative_abundance > 0.0 else (),
            inventory=inventory,
        ))
    summaries = []
    for source in ("GCR", "SEP"):
        selected = tuple(row for row in rows if row.source == source)
        positive = tuple(row for row in selected if row.nonzero_source_components)
        summaries.append(SourceMassSummary(
            source, len(selected), len(positive), len(selected) - len(positive),
            sum(row.candidate_isotope_mass_number is not None for row in positive),
            sum(row.classification == "missing" for row in positive),
            sum(row.classification == "exact_numeric_mass_not_isotope_proof" for row in positive),
            sum(row.classification == "different_mass_parameter_policy_unresolved"
                for row in positive),
            sum(row.rounded_metadata_hides_mass_difference for row in positive),
        ))
    source_z = {row.z for row in rows}
    return IonMassContractAudit(
        candidates, tuple(rows), tuple(summaries),
        tuple(ion.z for ion in candidates if ion.z not in source_z),
    )
