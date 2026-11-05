"""Analytic lower-support extension invariants; no physical range-origin claim."""
from __future__ import annotations

from decimal import Decimal, localcontext
from math import exp, log

import pytest

from radar.core.types import SpectrumCompleteness
from radar.shielding.primary_let import (
    SRIM_ELECTRONIC_LET_PROVENANCE, convert_primary_electronic_let,
)
from radar.shielding.srim_hze import SrimIonStoppingTable, transport_srim_primary
from test_srim_hze_transport import native, source


def pairs(power: bool = False) -> tuple[SrimIonStoppingTable, SrimIonStoppingTable]:
    energies = ("3", "6", "12", "24", "48")
    stopping = energies if power else ("12",) * 5
    # Si is deliberately nonmonotone, with overlapping LET images. It need not
    # equal the Al stopping curve or use the same total/electronic fraction.
    si_stopping = ("0.3", "6", "12", "6", "24")
    new = SrimIonStoppingTable.from_native(
        native(energies=energies, stopping=stopping),
        native(target=14, energies=energies, stopping=si_stopping, fraction="1"),
    )
    old = SrimIonStoppingTable.from_native(
        native(energies=energies[2:], stopping=stopping[2:]),
        native(target=14, energies=energies[2:], stopping=si_stopping[2:], fraction="1"),
    )
    return old, new


@pytest.mark.parametrize("power", [False, True])
def test_range_coordinate_shift_is_constant_over_shared_support(power: bool) -> None:
    old, new = pairs(power)
    offset = new.range_at_energy(1.)
    assert float(offset) == pytest.approx(log(4) if power else .75, rel=1e-14)
    for energy in (1., 1.1, 1.5, 2., 2.4, 3.5, 4.):
        with localcontext() as context:
            context.prec = 80
            delta = new.range_at_energy(energy) - old.range_at_energy(energy)
        assert abs(delta - offset) < Decimal("1e-76")


@pytest.mark.parametrize("peak", [False, True])
@pytest.mark.parametrize("power", [False, True])
@pytest.mark.parametrize("thickness", [.01, .1, 1., 3., 10.])
def test_shared_exit_measure_unchanged_and_added_measure_accounted(
    peak: bool, power: bool, thickness: float,
) -> None:
    old_table, new_table = pairs(power)
    src = source(peak=peak)
    # Deliberately keep the old display lower edge. New supported low exits
    # must be reported below the display, never lost or merged into its bins.
    edges = (1., 1.5, 2., 3., 4.)
    old, new = tuple(transport_srim_primary(
        src, table, thickness, source_mass_parameter_u=12., output_edges=edges,
    ) for table in (old_table, new_table))
    for lo, hi in zip(edges, edges[1:]):
        assert new.integral_on_exit(lo, hi) == pytest.approx(old.integral_on_exit(lo, hi), abs=1e-13)
    assert new.spectrum.y == pytest.approx(old.spectrum.y, abs=1e-13)
    extra = new.integral_on_exit(.25, 1.)
    assert extra >= 0
    assert new.ledger.below_output_window_integral == pytest.approx(extra, abs=1e-13)
    assert new.ledger.supported_exit_integral - old.ledger.supported_exit_integral == pytest.approx(
        extra, abs=1e-13)
    assert old.ledger.unresolved_low_residual_integral - new.ledger.unresolved_low_residual_integral == pytest.approx(
        extra, abs=1e-13)
    # An independent analytic threshold for the full supported source measure.
    threshold = .25 * exp(thickness) if power else .25 + thickness
    expected = 2 * max(0., 4. - max(1., threshold))
    assert new.ledger.supported_exit_integral == pytest.approx(expected, abs=1e-13)
    assert new.ledger.outside_al_domain_integral == old.ledger.outside_al_domain_integral == 0
    assert new.spectrum.assessment_mode is old.spectrum.assessment_mode is src.assessment_mode
    assert new.spectrum.y_unit is old.spectrum.y_unit is src.y_unit
    assert new.spectrum.temporal_basis is src.temporal_basis
    assert new.spectrum.mission_duration_s == src.mission_duration_s
    assert new.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
    assert not new.ledger.stopped_particle_classification_established
    assert not new.normative_export_allowed


@pytest.mark.parametrize("peak", [False, True])
@pytest.mark.parametrize("power", [False, True])
def test_high_source_tail_does_not_disappear_with_lower_extension(peak: bool, power: bool) -> None:
    src = source(peak=peak, x=(1., 3., 6.), y=(2., 2., 2.))
    old, new = tuple(transport_srim_primary(src, table, .5, source_mass_parameter_u=12.)
                     for table in pairs(power))
    assert old.ledger.outside_al_domain_integral == new.ledger.outside_al_domain_integral == 4.
    assert old.ledger.source_integral == new.ledger.source_integral == 10.
    assert not new.ledger.source_tails_known


@pytest.mark.parametrize("peak", [False, True])
def test_let_added_low_energy_branches_not_renormalized_or_subtracted(peak: bool) -> None:
    src = source(peak=peak)
    measures = tuple(transport_srim_primary(src, table, 1., source_mass_parameter_u=12.)
                     for table in pairs())
    let_edges = (0., .001, .004, .007, .010, .013, .020, .03)
    old, new = tuple(convert_primary_electronic_let(
        item, let_edges=let_edges, provenance=SRIM_ELECTRONIC_LET_PROVENANCE,
    ) for item in measures)
    expected_added = []
    # Constant Al stopping implies Eout=Ein-1 MeV/u and constant source density
    # 2. Low supported branch consists of two native Si segments only.
    for left, right in zip(let_edges, let_edges[1:]):
        count = 0.
        for e0, e1, l0, l1 in ((.25, .5, .0003, .006), (.5, 1., .006, .012)):
            lo, hi = max(left, l0), min(right, l1)
            if hi > lo:
                count += 2 * (hi - lo) / (l1 - l0) * (e1 - e0)
        expected_added.append(count)
    for index, (left, right) in enumerate(zip(let_edges, let_edges[1:])):
        actual = (new.spectrum.y[index] - old.spectrum.y[index]) * (right - left)
        assert actual == pytest.approx(expected_added[index], abs=1e-13)
        assert actual >= -1e-14
    assert sum(expected_added) == pytest.approx(1.5)
    assert new.ledger.mapped_integral - old.ledger.mapped_integral == pytest.approx(1.5)
    assert new.ledger.outside_si_domain_integral == old.ledger.outside_si_domain_integral == 0


@pytest.mark.parametrize("peak", [False, True])
@pytest.mark.parametrize("power", [False, True])
def test_zero_shield_has_no_change_for_source_above_old_cutoff(peak: bool, power: bool) -> None:
    src = source(peak=peak)
    old, new = tuple(transport_srim_primary(src, table, 0., source_mass_parameter_u=12.)
                     for table in pairs(power))
    assert old.spectrum.x is new.spectrum.x is src.x
    assert old.spectrum.y is new.spectrum.y is src.y
    old_let, new_let = tuple(convert_primary_electronic_let(
        measure, let_edges=(0., .004, .007, .01, .013, .03),
        provenance=SRIM_ELECTRONIC_LET_PROVENANCE,
    ) for measure in (old, new))
    assert new_let.spectrum.y == old_let.spectrum.y
    assert new_let.ledger == old_let.ledger


@pytest.mark.parametrize("peak", [False, True])
def test_lower_extension_does_not_turn_below_anchor_into_stopped(peak: bool) -> None:
    _, lower = pairs()
    result = transport_srim_primary(source(peak=peak), lower, 100., source_mass_parameter_u=12.)
    assert result.ledger.unresolved_low_residual_integral == 6.
    assert result.ledger.supported_exit_integral == 0
    assert result.exit_support is None
    assert not result.ledger.stopped_particle_classification_established
    assert not result.normative_export_allowed
