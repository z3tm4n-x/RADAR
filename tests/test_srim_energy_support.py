"""Synthetic source-energy support checks; no native launch or physics changes."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from decimal import Decimal, ROUND_DOWN, localcontext
from fractions import Fraction
from pathlib import Path
import re
from typing import Any

import pytest

from radar.validation.srim_energy_support import (
    EnergyCenterPartition, EnergyInterval, ExactEnergyBoundary, SourceEnergySupportRow,
    SrimEnergySupportAudit, audit_srim_energy_support,
)
from radar.validation.srim_native_coverage import audit_srim_native_coverage
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output


D = Decimal


def native(z: int = 2, mass: str = "4", target: int = 13,
           energies: tuple[str, str] = ("1", "12000")) -> SrimRangeOutput:
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


def row(audit: SrimEnergySupportAudit, source: str, z: int,
        material: int = 13) -> SourceEnergySupportRow:
    return next(r for r in audit.rows if (r.source, r.z, r.material_z) == (source, z, material))


def test_empty_native_input_retains_all_sources_and_declared_zeros() -> None:
    result = audit_srim_energy_support(())
    assert len(result.rows) == 236
    assert result.positive_source_material_rows == 216
    assert result.declared_zero_source_material_rows == 20
    assert result.native_table_count == 0
    for item in result.rows:
        if item.nonzero_source_components:
            assert item.classification == "missing_native"
            assert item.center_partition is not None
            assert item.center_partition.supported == 0
            assert item.center_partition.native_support_unknown == item.center_partition.total
            assert item.evaluation_span_coverage is not None
            assert item.evaluation_span_coverage.status == "UNKNOWN"
            assert item.evaluation_span_coverage.uncovered_intervals[0].meaning == "native_support_unknown"
        else:
            assert item.classification == "declared_zero"
            assert item.center_partition is None
            assert item.evaluation_span_coverage is None
            assert item.finite_integral_domain_coverage is None
    for summary in result.summaries:
        assert summary.positive_species_with_native_table == 0
        assert summary.positive_species_without_native_table == summary.positive_species_count
        assert summary.center_partition.native_support_unknown == summary.center_partition.total


@pytest.fixture(scope="module")
def complete() -> SrimEnergySupportAudit:
    sources = audit_srim_native_coverage(())
    outputs = tuple(native(item.z, str(item.source_tabulated_mass_parameter), item.material_z)
                    for item in sources.rows if item.source == "GCR" and item.nonzero_source_components)
    return audit_srim_energy_support(outputs)


def test_full_positive_roster_is_not_full_source_energy_domain(complete: SrimEnergySupportAudit) -> None:
    assert complete.native_table_count == 168
    assert complete.positive_source_material_rows == 216
    for summary in complete.summaries:
        assert summary.positive_species_without_native_table == 0
        assert summary.positive_species_with_native_table == summary.positive_species_count
        assert summary.positive_evaluation_spans_supported == 0
        assert summary.positive_finite_integral_domains_supported == (0 if summary.source == "GCR" else None)
        assert summary.center_partition.below_native_support == 0
        assert summary.center_partition.native_support_unknown == 0
        assert summary.center_partition.supported + summary.center_partition.above_native_support == (
            summary.center_partition.total)
    assert not complete.source_energy_domain_covered
    assert not complete.normative_export_allowed


@pytest.mark.parametrize("source,z,supported,total", [
    ("GCR", 2, 57, 81), ("GCR", 26, 38, 81), ("GCR", 92, 28, 81),
    ("SEP", 2, 140, 165), ("SEP", 26, 94, 165),
])
def test_representative_counts(complete: SrimEnergySupportAudit, source: str, z: int,
                               supported: int, total: int) -> None:
    for material in (13, 14):
        result = row(complete, source, z, material)
        assert result.center_partition == EnergyCenterPartition(total, supported, 0, total-supported, 0)
        assert result.native_support is not None
        assert result.native_support.upper.exact == Fraction(12000) / Fraction(result.source_tabulated_mass_parameter)
        assert result.native_support.lower.exact == Fraction(1) / Fraction(result.source_tabulated_mass_parameter)


def test_separate_source_endpoints_and_finite_gcr_closure(complete: SrimEnergySupportAudit) -> None:
    gcr, sep = complete.source_domains
    assert len(gcr.evaluation_centers_mev_per_u) == 81
    assert len(sep.evaluation_centers_mev_per_u) == 165
    assert gcr.finite_integral_domain is not None
    assert gcr.finite_integral_domain.upper.exact == 100000
    assert gcr.evaluation_span.upper.exact < 100000
    assert sep.finite_integral_domain is None
    assert "evaluation_limit_only" in sep.finite_integral_upper_semantics
    gcr_he, sep_he = row(complete, "GCR", 2), row(complete, "SEP", 2)
    assert gcr_he.finite_integral_domain_coverage is not None
    assert gcr_he.finite_integral_domain_coverage.uncovered_intervals[0].upper.exact == 100000
    assert sep_he.finite_integral_domain_coverage is None


def test_both_gaps_and_closed_native_endpoints() -> None:
    result = audit_srim_energy_support((native(energies=("12", "28")),),
                                      gcr_grid=(1, 3, 5, 7, 9), sep_grid=(1, 3, 5, 7, 9),
                                      gcr_upper_bound=10)
    item = row(result, "GCR", 2)
    assert item.center_partition == EnergyCenterPartition(5, 3, 1, 1, 0)
    assert item.evaluation_span_coverage is not None
    below, above = item.evaluation_span_coverage.uncovered_intervals
    assert (below.lower.exact, below.upper.exact, below.lower_closed, below.upper_closed) == (
        Fraction(1), Fraction(3), True, False)
    assert (above.lower.exact, above.upper.exact, above.lower_closed, above.upper_closed) == (
        Fraction(7), Fraction(9), False, True)
    assert item.finite_integral_domain_coverage is not None
    assert item.finite_integral_domain_coverage.uncovered_intervals[-1].upper.exact == 10


@pytest.mark.parametrize("energies,reason,closures,supported", [
    (("40", "60"), "below_native_support", (True, True), 0),
    (("1", "2"), "above_native_support", (True, True), 0),
    (("36", "60"), "below_native_support", (True, False), 1),
    (("1", "4"), "above_native_support", (False, True), 1),
])
def test_disjoint_and_single_endpoint_intersections(energies: tuple[str, str], reason: str,
                                                  closures: tuple[bool, bool], supported: int) -> None:
    result = audit_srim_energy_support((native(energies=energies),),
                                      gcr_grid=(1, 3, 9), sep_grid=(1, 3, 9), gcr_upper_bound=9)
    item = row(result, "GCR", 2)
    assert item.center_partition is not None and item.center_partition.supported == supported
    assert item.evaluation_span_coverage is not None
    gap, = item.evaluation_span_coverage.uncovered_intervals
    assert gap.meaning == reason
    assert (gap.lower_closed, gap.upper_closed) == closures


def test_evaluation_span_complete_does_not_hide_gcr_closure_gap() -> None:
    result = audit_srim_energy_support((native(energies=("4", "36")),),
                                      gcr_grid=(1, 3, 9), sep_grid=(1, 3, 9), gcr_upper_bound=10)
    item = row(result, "GCR", 2)
    assert item.evaluation_span_coverage is not None
    assert item.evaluation_span_coverage.status == "COMPLETE_DECLARED_INTERVAL"
    assert item.evaluation_span_coverage.uncovered_intervals == ()
    assert item.finite_integral_domain_coverage is not None
    assert item.finite_integral_domain_coverage.status == "PARTIAL"
    assert not result.source_energy_domain_covered
    assert not result.normative_export_allowed


def test_exact_rational_comparison_does_not_use_rounded_boundary() -> None:
    boundary = ExactEnergyBoundary(D("4.6"), D("6.9"))
    assert boundary.exact == Fraction(2, 3)
    assert Fraction(boundary.decimal_mev_per_u) > boundary.exact
    result = audit_srim_energy_support((native(3, "6.9", energies=("1", "4.6")),),
        gcr_grid=(D(".2"), boundary.decimal_mev_per_u, D(1)),
        sep_grid=(D(".2"), boundary.decimal_mev_per_u, D(1)), gcr_upper_bound=1)
    item = row(result, "GCR", 3)
    assert item.center_partition == EnergyCenterPartition(3, 1, 0, 2, 0)


def test_input_float_records_are_decimal_str_not_binary_expansion() -> None:
    value = 0.1 + 0.2
    result = audit_srim_energy_support((), gcr_grid=(0.1, value, 1), sep_grid=(1, 2), gcr_upper_bound=2)
    recorded = result.source_domains[0].evaluation_centers_mev_per_u[1]
    assert recorded == D(str(value))
    assert recorded != D.from_float(value)


def test_results_do_not_depend_on_callers_decimal_context() -> None:
    outputs = (native(26, "55.8"),)
    expected = asdict(audit_srim_energy_support(outputs))
    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_DOWN
        actual = asdict(audit_srim_energy_support(outputs))
    assert actual == expected


def test_material_bounds_and_missing_material_remain_separate() -> None:
    result = audit_srim_energy_support((native(energies=("1", "12000")),
                                       native(target=14, energies=("1", "24000")),
                                       native(26, "55.8")))
    al, si = row(result, "GCR", 2), row(result, "GCR", 2, 14)
    assert al.center_partition != si.center_partition
    assert row(result, "GCR", 26, 14).classification == "missing_native"
    assert row(result, "GCR", 26).classification == "observed_interval_only"


@pytest.mark.parametrize("mass", ["56", "55", "55.80000000000001"])
def test_nonmatching_mass_is_rejected_without_rounding(mass: str) -> None:
    with pytest.raises(ValueError, match="mass must equal"):
        audit_srim_energy_support((native(26, mass),))


@pytest.mark.parametrize("outputs", [(native(), native()),
    (native(), native(energies=("12000", "24000"))),
    (native(26, "55.8"), native(26, "56", 14)),
    (native(1, "1"),), (native(target=6),)])
def test_duplicates_disjoint_pieces_conflicts_and_out_of_scope_rejected(outputs: tuple[SrimRangeOutput, ...]) -> None:
    with pytest.raises(ValueError):
        audit_srim_energy_support(outputs)


@pytest.mark.parametrize("grid", [[], (), (1,), (1, 1), (2, 1), (0, 1), (-1, 1),
    (True, 2), ("1", 2), (float("nan"), 2), (1, float("inf")),
    (D("NaN"), D(2)), (D("Infinity"), D(2)), (D("1e-1001"), D(2))])
@pytest.mark.parametrize("source", ["gcr_grid", "sep_grid"])
def test_invalid_grids_rejected(source: str, grid: Any) -> None:
    with pytest.raises(ValueError):
        audit_srim_energy_support((), **{source: grid})


@pytest.mark.parametrize("upper", [None, False, "2", 0, -1, 1.5, float("nan"), float("inf")])
def test_invalid_gcr_closure_rejected(upper: Any) -> None:
    with pytest.raises(ValueError):
        audit_srim_energy_support((), gcr_grid=(1, 2), gcr_upper_bound=upper)


def test_source_aq_and_unusual_literal_masses_preserved(complete: SrimEnergySupportAudit) -> None:
    for z, mass in ((81, "200.6"), (82, "200.4"), (83, "207.2")):
        item = row(complete, "GCR", z)
        assert item.source_tabulated_mass_parameter == D(mass)
        assert item.native_observation is not None
        assert item.native_observation.native_ion_mass_u == D(mass)
    source = audit_srim_native_coverage(())
    original = next(r for r in source.rows if r.source == "SEP" and r.z == 26 and r.material_z == 13)
    assert row(complete, "SEP", 26).source_mass_to_charge == original.source_mass_to_charge


def test_optional_native_table_for_zero_source_is_not_counted_as_positive() -> None:
    source = next(r for r in audit_srim_native_coverage(()).rows if r.source == "GCR"
                  and not r.nonzero_source_components)
    result = audit_srim_energy_support((native(source.z, str(source.source_tabulated_mass_parameter)),))
    item = row(result, "GCR", source.z)
    assert item.classification == "declared_zero"
    assert item.native_observation is not None and item.native_support is not None
    assert item.center_partition is None and item.evaluation_span_coverage is None
    assert result.summaries[0].positive_species_with_native_table == 0


def test_flags_remain_conservative_and_result_frozen(complete: SrimEnergySupportAudit) -> None:
    for name in ("source_energy_domain_covered", "shielded_transport_domain_covered",
                 "physical_accuracy_verified", "isotope_policy_qualified", "normative_export_allowed",
                 "source_spectrum_convolution_performed", "production_tables_replaced",
                 "native_authenticity_verified"):
        assert getattr(complete, name) is False
    assert all(item.residual_energy_below_native_interval == "UNKNOWN_not_physical_stopping"
               and not item.normative_export_allowed for item in complete.rows)
    with pytest.raises(FrozenInstanceError):
        complete.native_table_count = 0  # type: ignore[misc]


@pytest.mark.parametrize("values", [(3, 1, 1, 0, 0), (2, 1, -1, 2, 0), (2, True, 1, 0, 0)])
def test_partition_rejects_nonconservation_and_invalid_counts(values: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        EnergyCenterPartition(*values)


def test_intervals_require_ordered_positive_bounds_and_boolean_closure() -> None:
    a, b = ExactEnergyBoundary(D(1), D(1)), ExactEnergyBoundary(D(2), D(1))
    with pytest.raises(ValueError):
        ExactEnergyBoundary(D(1), D(0))
    with pytest.raises(ValueError):
        EnergyInterval(b, a, True, True, "native_table_support")
    with pytest.raises(ValueError):
        EnergyInterval(a, b, 1, True, "native_table_support")  # type: ignore[arg-type]
