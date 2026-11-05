"""Synthetic native inventory controls; never run SRIM or rewrite data."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from decimal import Decimal, ROUND_DOWN, localcontext
from pathlib import Path
import re
from typing import Any

import pytest

from radar.validation import srim_native_coverage as coverage
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output
from radar.gcr.parameters import get_ost_gcr_ion_parameters
from radar.validation.ion_mass_contract import CandidateIsotope, audit_ost_ion_mass_contract


D = Decimal
LEGACY = ((2, 4), (6, 12), (7, 14), (8, 16), (10, 20), (12, 24), (14, 28),
          (16, 32), (18, 40), (20, 40), (24, 52), (25, 55), (26, 56), (27, 59), (28, 58))


def native(z: int = 2, mass: str = "4", target: int = 13,
           energies: tuple[str, str] = ("1", "45000")) -> SrimRangeOutput:
    fixture = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    lines = [line.strip() for line in fixture.read_bytes().decode("cp1251").splitlines()
             if line.strip()]
    lines = [line if "(C)" in line else re.sub(r"(?<=[0-9]),(?=[0-9])", ".", line)
             for line in lines]
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC NOT PHYSICAL DATA"
    header[6] = header[17] = f"Ion = Synthetic [{z}] , Mass = {mass} amu"
    header[7] = "Density = 2 g/cm3 = 6E22 atoms/cm3"
    header[12] = f"{ {13: 'Al', 14: 'Si', 6: 'C'}[target]} {target} 100 100"
    rows = [f"{e} MeV 0.001 0.001 {i+1} cm 1 A 1 A" for i, e in enumerate(energies)]
    footer = lines[lines.index("-" * 59):]
    return parse_srim_range_output(("\n".join(header + rows + footer) + "\n").encode("ascii"))


def row(audit: coverage.SrimNativeCoverageAudit, source: str, z: int,
        material: int = 13) -> coverage.SourceNativeCoverageRow:
    return next(r for r in audit.rows if (r.source, r.z, r.material_z) == (source, z, material))


def test_complete_legacy_roster_is_not_complete_source_coverage() -> None:
    audit = coverage.audit_srim_native_coverage(tuple(native(z, str(mass), target)
        for z, mass in LEGACY for target in (13, 14)))
    assert len(audit.native_observations) == 30
    assert audit.proton_observations == ()
    assert len(audit.rows) == (91 + 27) * 2
    gcr, sep = audit.summaries
    assert (gcr.positive_hze_species, gcr.declared_zero_species) == (84, 7)
    assert (sep.positive_hze_species, sep.declared_zero_species) == (24, 3)
    assert (gcr.missing_positive_species, sep.missing_positive_species) == (69, 9)
    for summary in audit.summaries:
        assert summary.positive_species_with_both_materials == 15
        assert summary.positive_species_with_any_material == 15
        assert summary.positive_species_missing_one_material == 0
        assert summary.positive_species_numeric_mass_equal_both_materials == 5
        assert summary.positive_species_with_any_numeric_mass_difference == 10
        assert summary.mass_policy_qualified_species_count == 0
    for material_summary in audit.material_summaries:
        assert material_summary.positive_species_observed == 15
        assert material_summary.positive_species_numeric_mass_equal == 5
        assert material_summary.positive_species_numeric_mass_different == 10


@pytest.mark.parametrize("source", ["GCR", "SEP"])
@pytest.mark.parametrize("z,mass,source_mass", [(26, "56", "55.8"), (28, "58", "58.7")])
def test_mass_differences_are_not_hidden_by_rounding(source: str, z: int, mass: str,
                                                   source_mass: str) -> None:
    audit = coverage.audit_srim_native_coverage((native(z, mass),))
    result = row(audit, source, z)
    assert result.source_tabulated_mass_parameter == D(source_mass)
    assert result.observation is not None and result.observation.native_ion_mass_u == D(mass)
    assert result.mass_comparison == "numeric_difference_policy_unresolved"
    assert result.classification == ("observed_energy_interval_only", "mass_policy_unresolved")
    assert not result.physical_mass_policy_established


def test_fractional_native_mass_is_retained_not_cast_to_isotope() -> None:
    audit = coverage.audit_srim_native_coverage((native(26, "55.8"), native(26, "55.8", 14)))
    result = row(audit, "GCR", 26)
    assert result.mass_comparison == "numeric_equal_not_isotope_proof"
    assert result.observation is not None and result.observation.native_ion_mass_u == D("55.8")
    assert result.observation.mass_policy == "native_numeric_mass_parameter_no_isotope_selection"
    assert result.classification == ("observed_energy_interval_only", "mass_policy_unresolved")
    assert not result.physical_mass_policy_established


def test_total_energy_conversions_are_separate_and_explicit() -> None:
    audit = coverage.audit_srim_native_coverage((native(26, "56", energies=("56", "5600")),))
    result = row(audit, "GCR", 26)
    assert result.observation is not None
    assert result.observation.energy_total_mev_bounds == (D(56), D(5600))
    assert result.observation.total_energy_divided_by_native_mass == (D(1), D(100))
    with localcontext() as context:
        context.prec = 80
        assert result.total_energy_divided_by_source_mass_parameter == (
            D(56)/D("55.8"), D(5600)/D("55.8"))
    assert result.observation.native_range_kind == "projected"
    assert not result.source_energy_domain_covered


def test_material_specific_bounds_and_missing_material_are_retained() -> None:
    audit = coverage.audit_srim_native_coverage((native(6, "12", energies=("1", "12000")),
                                               native(6, "12", 14, ("1", "45000")),
                                               native(2, "4")))
    al, si = row(audit, "GCR", 6), row(audit, "GCR", 6, 14)
    assert al.observation is not None and si.observation is not None
    assert al.observation.energy_total_mev_bounds[-1] == D(12000)
    assert si.observation.energy_total_mev_bounds[-1] == D(45000)
    absent = row(audit, "GCR", 2, 14)
    assert absent.classification == ("missing_material",)
    assert absent.observation is None and absent.total_energy_divided_by_source_mass_parameter is None
    assert audit.summaries[0].positive_species_missing_one_material == 1


def test_declared_zero_is_not_missing_positive_data() -> None:
    audit = coverage.audit_srim_native_coverage(())
    assert row(audit, "GCR", 84).classification == ("declared_zero",)
    assert row(audit, "SEP", 3).classification == ("declared_zero",)
    assert row(audit, "GCR", 3).classification == ("missing_positive_species",)
    assert row(audit, "SEP", 9).classification == ("missing_positive_species",)
    assert (audit.summaries[0].missing_positive_species,
            audit.summaries[1].missing_positive_species) == (84, 24)


def test_observing_a_declared_zero_does_not_make_it_a_positive_contribution() -> None:
    audit = coverage.audit_srim_native_coverage((native(3, "6.9"),))
    result = row(audit, "SEP", 3)
    assert result.observation is not None and result.classification == ("declared_zero",)
    assert not result.nonzero_source_components
    assert audit.summaries[1].positive_species_with_any_material == 0
    assert audit.summaries[0].positive_species_with_any_material == 1


def test_sep_mass_to_charge_remains_distinct_from_native_a_over_z() -> None:
    audit = coverage.audit_srim_native_coverage((native(26, "56"),))
    result = row(audit, "SEP", 26)
    assert result.source_mass_to_charge is not None
    assert result.source_mass_to_charge != D(56)/26
    assert row(audit, "GCR", 26).source_mass_to_charge is None


def test_hydrogen_is_separate_from_hze_source_coverage() -> None:
    audit = coverage.audit_srim_native_coverage((native(1, "1.008"), native(1, "1.008", 14)))
    assert len(audit.proton_observations) == 2 and audit.native_observations == ()
    assert all(r.z >= 2 for r in audit.rows)
    assert all(s.positive_species_with_any_material == 0 for s in audit.summaries)


def test_input_order_does_not_change_report() -> None:
    inputs = (native(26, "56"), native(2, "4", 14), native(2, "4"))
    assert coverage.audit_srim_native_coverage(inputs) == coverage.audit_srim_native_coverage(
        tuple(reversed(inputs)))


def test_private_decimal_conversion_context() -> None:
    inputs = (native(26, "56"),)
    expected = coverage.audit_srim_native_coverage(inputs)
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_DOWN
        context.Emax = 3
        context.Emin = -3
        for signal in context.traps:
            context.traps[signal] = True
        assert coverage.audit_srim_native_coverage(inputs) == expected


@pytest.mark.parametrize("inputs", [None, [], [True], (True,), (object(),)])
def test_malformed_input_is_rejected(inputs: Any) -> None:
    with pytest.raises(ValueError, match="immutable tuple"):
        coverage.audit_srim_native_coverage(inputs)


def test_duplicate_table_rejected_even_when_identical() -> None:
    table = native()
    with pytest.raises(ValueError, match="Duplicate native"):
        coverage.audit_srim_native_coverage((table, table))


def test_conflicting_native_material_masses_are_not_chosen_silently() -> None:
    with pytest.raises(ValueError, match="Conflicting native Al/Si masses"):
        coverage.audit_srim_native_coverage((native(26, "56"), native(26, "55.8", 14)))


def test_other_material_rejected() -> None:
    with pytest.raises(ValueError, match="Only Al and Si"):
        coverage.audit_srim_native_coverage((native(target=6),))


def test_invalid_native_mass_rejected() -> None:
    with pytest.raises(ValueError, match="at least ion Z"):
        coverage.audit_srim_native_coverage((native(26, "2"),))


def test_incomplete_source_roster_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = get_ost_gcr_ion_parameters()
    monkeypatch.setattr(coverage, "get_ost_gcr_ion_parameters",
                        lambda: {key: value for key, value in original.items() if value.z != 2})
    with pytest.raises(ValueError, match="source roster changed"):
        coverage.audit_srim_native_coverage(())


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1.])
def test_bad_source_number_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="literal OST"):
        coverage._number(value, "synthetic")


def test_status_flags_are_fixed_and_frozen() -> None:
    audit = coverage.audit_srim_native_coverage((native(),))
    assert audit.diagnostic_status == "INCOMPLETE"
    flags = {key: value for key, value in asdict(audit).items() if isinstance(value, bool)}
    assert flags and not any(flags.values())
    with pytest.raises(FrozenInstanceError):
        audit.normative_export_allowed = True  # type: ignore[misc]


def test_source_fields_match_existing_independent_mass_inventory() -> None:
    # The explicit test isotope is unrelated to native mass conversion. It lets
    # the older inventory expose its full source roster for an independent check.
    previous = audit_ost_ion_mass_contract((CandidateIsotope(2, 4, "He"),))
    audit = coverage.audit_srim_native_coverage(())
    for expected in previous.rows:
        actual = row(audit, expected.source, expected.z)
        assert actual.symbol == expected.symbol
        assert actual.source_tabulated_mass_parameter == D(str(expected.source_tabulated_mass_parameter))
        assert actual.source_mass_to_charge == (None if expected.source_mass_to_charge is None else
                                               D(str(expected.source_mass_to_charge)))
        assert actual.source_weight_parameter_name == expected.source_weight_parameter_name
        assert actual.source_weight_parameter == D(str(expected.source_weight_parameter))
        assert actual.nonzero_source_components == expected.nonzero_source_components
