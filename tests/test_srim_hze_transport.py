"""Analytic witnesses for the actual bounded SRIM spectral operator; no native launch."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from math import exp, inf, nan, nextafter
from pathlib import Path
import re
from typing import Any

import pytest

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationSource, SpectrumAxisKind,
    SpectrumCompleteness, SpectrumLocation, SpectrumOrdinateBasis, SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit
from radar.shielding.srim_hze import SrimIonStoppingTable, transport_srim_primary
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output


def native(*, target: int = 13, mass: str = "12", z: int = 6,
           energies: tuple[str, ...] = ("12", "24", "48"),
           stopping: tuple[str, ...] = ("12", "12", "12"),
           fraction: str = ".75", density: str = "2.70",
           projected_scale: str = "1") -> SrimRangeOutput:
    raw = (Path(__file__).parent / "fixtures/srim_h_al_smoke.txt").read_bytes()
    lines = [line.strip() if "(C)" in line else re.sub(r"(?<=[0-9]),(?=[0-9])", ".", line.strip())
             for line in raw.decode("cp1251").splitlines() if line.strip()]
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC NOT PHYSICAL DATA"
    header[6] = header[17] = f"Ion = Synthetic [{z}] , Mass = {mass} amu"
    header[7] = f"Density = {density} g/cm3 = 6E22 atoms/cm3"
    header[12] = f"{'Al' if target == 13 else 'Si'} {target} 100 100"
    footer = lines[lines.index("-" * 59):]
    with localcontext() as context:
        context.prec = 100
        rows = [f"{e} MeV {Decimal(s)*Decimal(fraction)/1000} "
                f"{Decimal(s)*(1-Decimal(fraction))/1000} "
                f"{(i+1)*Decimal(projected_scale)} cm 0.1 cm 0.1 cm"
                for i, (e, s) in enumerate(zip(energies, stopping, strict=True))]
    return parse_srim_range_output(("\r\n".join(header + rows + footer) + "\r\n").encode("ascii"))


def table(**kwargs: Any) -> SrimIonStoppingTable:
    return SrimIonStoppingTable.from_native(native(**kwargs), native(target=14, **kwargs))


def source(*, peak: bool = False, **changes: Any) -> Spectrum1D:
    values: dict[str, Any] = {
        "x": (1., 2., 4.), "y": (2., 2., 2.), "x_unit": Unit.MEV_PER_NUCLEON,
        "y_unit": (Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY if peak
                   else Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY),
        "quantity": SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX if peak else SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        "particle": Particle.HZE, "source": RadiationSource.SEP, "model": "synthetic_srim_source",
        "axis_kind": SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        "angular_basis": AngularBasis.OMNIDIRECTIONAL_4PI,
        "temporal_basis": TemporalBasis.SEP_PEAK if peak else TemporalBasis.MISSION_TOTAL,
        "assessment_mode": AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE,
        "mission_duration_s": None if peak else 1234567., "location": SpectrumLocation.ON_ORBIT,
        "species": ParticleSpecies("C", 6, 12),
        "model_reference": ModelReference("synthetic_srim_source", "1", ("analytic test",), ("synthetic",)),
        "transformation_chain": (TransformationStep("synthetic_source", "1", (
            ("source_tabulated_mass_parameter", "12"),)),),
        "completeness": SpectrumCompleteness.COMPLETE,
    }
    values.update(changes)
    return Spectrum1D(**values)


def transport(*, peak: bool = False, thickness: float = 1., **kwargs: Any) -> Any:
    return transport_srim_primary(source(peak=peak), table(**kwargs), thickness,
                                  source_mass_parameter_u=12.)


def test_pair_uses_native_energy_and_stopping_not_projected_range() -> None:
    result = table()
    assert result.z == 6 and result.mass_parameter_u == 12
    assert result.al_energy == result.si_energy == (1., 2., 4.)
    assert result.si_electronic_let == (.009, .009, .009)
    assert result.al_range_table.anchored_range_g_cm2 == (Decimal(0), Decimal(1), Decimal(3))
    assert result.al_range_table.interpolation == "loglog_stopping"
    changed = table(projected_scale="1000", density="9.9")
    assert changed.al_range_table.anchored_range_g_cm2 == result.al_range_table.anchored_range_g_cm2
    assert changed.dataset_ref != result.dataset_ref
    assert not result.normative_export_allowed
    assert not result.physical_accuracy_verified
    assert not result.native_source_authenticity_verified
    with pytest.raises(FrozenInstanceError):
        result.mass_parameter_u = 14.  # type: ignore[misc]


@pytest.mark.parametrize("pair", [(native(), native()), (native(target=14), native()),
                                  (native(), native(target=14, mass="12.1")),
                                  (native(), native(target=14, z=7))])
def test_reject_material_or_exact_mass_identity_mismatch(pair: tuple[SrimRangeOutput, ...]) -> None:
    with pytest.raises(ValueError):
        SrimIonStoppingTable.from_native(*pair)


@pytest.mark.parametrize("bad", [None, object(), True])
def test_reject_unparsed_native_objects(bad: Any) -> None:
    with pytest.raises(ValueError):
        SrimIonStoppingTable.from_native(bad, native(target=14))


@pytest.mark.parametrize("peak", [False, True])
def test_constant_stopping_analytic_shift_preserves_time_units_and_balance(peak: bool) -> None:
    result = transport(peak=peak)
    assert result.exit_support == (1., 3.)
    assert result._incident_support == (2., 4.)
    assert result.ledger.source_integral == 6
    assert result.ledger.supported_exit_integral == 4
    assert result.ledger.represented_integral == 4
    assert result.ledger.unresolved_low_residual_integral == 2
    assert result.ledger.outside_al_domain_integral == 0
    assert result.ledger.balance_residual == 0
    assert result.spectrum.y == (2., 1.)
    assert result.spectrum.integral_conserved() == 4
    assert result.spectrum.y_unit is result.source.y_unit
    assert result.spectrum.quantity is result.source.quantity
    assert result.spectrum.assessment_mode is result.source.assessment_mode
    assert result.spectrum.temporal_basis is result.source.temporal_basis
    assert result.spectrum.mission_duration_s == result.source.mission_duration_s
    assert result.spectrum.angular_basis is result.source.angular_basis
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
    assert not result.normative_export_allowed
    assert not result.ledger.stopped_particle_classification_established
    assert not result.ledger.source_tails_known
    assert result.spectrum.production_contract_violations() == (
        "incomplete spectra cannot be released as production results",)
    parameters = dict(result.spectrum.transformation_chain[-1].parameters)
    assert parameters["time_factor"] == parameters["angular_factor"] == parameters["density_factor"] == "none"
    assert parameters["projected_ranges_used"] == "false"


@pytest.mark.parametrize("peak", [False, True])
def test_zero_shield_exact_identity_even_outside_native_al_domain(peak: bool) -> None:
    original = source(peak=peak, x=(.001, 2., 1000.))
    result = transport_srim_primary(original, table(), 0, source_mass_parameter_u=12)
    assert result.spectrum.x is original.x and result.spectrum.y is original.y
    assert result.spectrum.bin_edges is original.bin_edges
    assert result.spectrum.ordinate_basis is original.ordinate_basis
    assert result.exit_support == (.001, 1000.)
    assert result.ledger.outside_al_domain_integral == 0
    assert result.integral_on_exit(.001, 1000.) == original.integral_conserved()
    with pytest.raises(ValueError, match="identity"):
        transport_srim_primary(original, table(), 0, source_mass_parameter_u=12, output_edges=(1., 2.))


def test_fractional_literal_mass_and_nonmatching_isotope_label_remain_separate() -> None:
    paired = table(mass="12.3", energies=("12.3", "24.6", "49.2"), stopping=("12.3",)*3)
    original = source(transformation_chain=(TransformationStep("source", "1", (
        ("source_tabulated_mass_parameter", "12.3"),)),))
    result = transport_srim_primary(original, paired, 1, source_mass_parameter_u=12.3)
    assert result.exit_support == (1., 3.)
    assert result.spectrum.y == (2., 1.)
    assert paired.mass_parameter_u_decimal == Decimal("12.3")
    with pytest.raises(ValueError, match="mass"):
        transport_srim_primary(original, paired, 1, source_mass_parameter_u=12)
    with pytest.raises(ValueError, match="trace"):
        transport_srim_primary(source(), paired, 1, source_mass_parameter_u=12.3)


def test_real_source_17_digit_float_trace_preserves_literal_native_fe_mass() -> None:
    paired = table(z=26, mass="55.8", energies=("55.8", "111.6", "223.2"), stopping=("55.8",)*3)
    original = source(species=ParticleSpecies("Fe", 26, 56),
                      transformation_chain=(TransformationStep("source", "1", (
                          ("source_tabulated_mass_parameter", format(55.8, ".17g")),)),))
    result = transport_srim_primary(original, paired, 1, source_mass_parameter_u=55.8)
    assert format(55.8, ".17g") == "55.799999999999997"
    assert paired.mass_parameter_u_decimal == Decimal("55.8")
    assert result.exit_support == (1., 3.)


def test_power_law_stopping_uses_analytic_integral_not_linear_range_interpolation() -> None:
    # Total S(E_total)=E_total, so range difference=ln(Ein/Eout), eout=ein*exp(-d).
    paired = table(stopping=("12", "24", "48"))
    result = transport_srim_primary(source(), paired, .25, source_mass_parameter_u=12)
    assert result.exit_support == pytest.approx((1, 4*exp(-.25)))
    assert result.integral_on_exit(1.2, 2.) == pytest.approx(2*(2.-1.2)*exp(.25), rel=1e-13)
    assert result.ledger.supported_exit_integral == pytest.approx(2*(4-exp(.25)), rel=1e-13)
    assert float(paired.range_at_energy(1.5)) == pytest.approx(__import__("math").log(1.5))
    assert paired.energy_at_range(paired.range_at_energy(1.5)) == 1.5


@pytest.mark.parametrize("fraction", ["0", ".25", "1"])
def test_each_native_loss_component_enters_al_range_once(fraction: str) -> None:
    result = transport(fraction=fraction)
    assert result.exit_support == (1., 3.) and result.spectrum.y == (2., 1.)


def test_partition_all_source_below_above_and_unresolved_low_exit() -> None:
    original = source(x=(.5, 2., 6.))
    result = transport_srim_primary(original, table(), .5, source_mass_parameter_u=12)
    assert result.ledger.source_integral == 11
    assert result.ledger.outside_al_domain_integral == 5
    assert result.ledger.unresolved_low_residual_integral == 1
    assert result.ledger.supported_exit_integral == 5
    assert result.ledger.balance_residual == 0
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
    assert any("outside observed" in x for x in result.spectrum.incomplete_reasons)


@pytest.mark.parametrize("thickness", [3., 3.001, 100.])
def test_thick_shield_does_not_classify_all_ions_as_stopped(thickness: float) -> None:
    result = transport(thickness=thickness)
    assert result.exit_support is None
    assert result.spectrum.y == (0., 0.)
    assert result.ledger.unresolved_low_residual_integral == 6
    assert result.ledger.supported_exit_integral == 0
    assert any("not stopped" in x for x in result.spectrum.incomplete_reasons)


def test_source_entirely_outside_al_domain_is_explicit_unresolved() -> None:
    result = transport_srim_primary(source(x=(10., 20., 40.)), table(), 1, source_mass_parameter_u=12)
    assert result.ledger.outside_al_domain_integral == 60
    assert result.ledger.unresolved_low_residual_integral == 0
    assert result.ledger.represented_integral == 0
    assert result.exit_support is None
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE


def test_display_window_does_not_discard_supported_integral() -> None:
    result = transport_srim_primary(source(), table(), .5, source_mass_parameter_u=12,
                                    output_edges=(1.5, 2.5, 3.))
    assert result.ledger.represented_integral == 3
    assert result.ledger.below_output_window_integral == 1
    assert result.ledger.above_output_window_integral == 1
    assert result.ledger.supported_exit_integral == 5
    assert result.ledger.balance_residual == 0


def test_bin_average_source_uses_declared_piecewise_constant_measure() -> None:
    original = source(x=(1.5, 3.), y=(2., 5.), ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
                      bin_edges=(1., 2., 4.))
    result = transport_srim_primary(original, table(), .5, source_mass_parameter_u=12)
    assert result.ledger.source_integral == 12
    assert result.ledger.unresolved_low_residual_integral == 1
    assert result.ledger.represented_integral == 11
    assert result.integral_on_exit(1, 2) == 3.5


def test_point_source_integrates_linear_density_in_original_coordinate() -> None:
    original = source(y=(1., 2., 4.))  # j(e)=e
    result = transport_srim_primary(original, table(), 1, source_mass_parameter_u=12)
    assert result.integral_on_exit(1, 2) == 2.5
    assert result.integral_on_exit(1.25, 1.75) == 1.25


@pytest.mark.parametrize("factor", [.01, 3., 1e9])
def test_source_scaling_has_no_hidden_normalization(factor: float) -> None:
    a = transport()
    b = transport_srim_primary(source(y=tuple(v*factor for v in source().y)), table(), 1,
                               source_mass_parameter_u=12)
    assert b.spectrum.y == pytest.approx(tuple(v*factor for v in a.spectrum.y))
    assert b.ledger.source_integral == pytest.approx(factor*a.ledger.source_integral)


def test_ambient_decimal_context_does_not_change_transport() -> None:
    before = transport(stopping=("12", "24", "48"), thickness=.25)
    with localcontext() as context:
        context.prec = 2
        after = transport(stopping=("12", "24", "48"), thickness=.25)
    assert before.spectrum.y == after.spectrum.y and before.ledger == after.ledger


@pytest.mark.parametrize("value", [-1., nan, inf, True, "1"])
def test_bad_thickness_rejected(value: Any) -> None:
    with pytest.raises(ValueError):
        transport(thickness=value)


@pytest.mark.parametrize("edges", [(1.,), (2., 1.), (0., 2.), (1., nan), [1., 2.]])
def test_invalid_output_edges_rejected(edges: Any) -> None:
    with pytest.raises(ValueError):
        transport_srim_primary(source(), table(), 1, source_mass_parameter_u=12, output_edges=edges)


@pytest.mark.parametrize("bounds", [(-1., 2.), (2., 1.), (nan, 1.), (0., inf), (False, 1.)])
def test_bad_integral_bounds_rejected(bounds: Any) -> None:
    with pytest.raises(ValueError):
        transport().integral_on_exit(*bounds)


@pytest.mark.parametrize("e", [.999, 4.001, 0, nan, True])
def test_table_does_not_extrapolate(e: Any) -> None:
    with pytest.raises(ValueError):
        table().range_at_energy(e)


@pytest.mark.parametrize("r", [Decimal(-1), Decimal("3.01"), Decimal("NaN"), 1.])
def test_inverse_does_not_extrapolate(r: Any) -> None:
    with pytest.raises(ValueError):
        table().energy_at_range(r)


def test_endpoint_native_coordinate_is_exact_for_fractional_mass() -> None:
    paired = table(mass="12.3")
    for e, r in zip(paired.al_energy, paired.al_range_table.anchored_range_g_cm2):
        assert paired.range_at_energy(e) == r
        assert paired.energy_at_range(r) == e


def test_unresolvably_small_positive_loss_is_rejected_not_erased() -> None:
    with pytest.raises(ValueError, match="resolution"):
        transport(thickness=1e-40)


def test_cached_preimages_do_not_change_results() -> None:
    result = transport(stopping=("12", "24", "48"), thickness=.25)
    first = result.integral_on_exit(1.25, 2.5)
    second = result.integral_on_exit(1.25, 2.5)
    assert first == second
    assert 1.25 in result._preimage_cache and 2.5 in result._preimage_cache


def test_prior_incomplete_reasons_are_preserved() -> None:
    original = source(completeness=SpectrumCompleteness.INCOMPLETE, incomplete_reasons=("source limitation",))
    result = transport_srim_primary(original, table(), 1, source_mass_parameter_u=12)
    assert "source limitation" in result.spectrum.incomplete_reasons


@pytest.mark.parametrize("text", ["13", "NaN", "nonsense"])
def test_malformed_or_conflicting_literal_mass_trace_rejected(text: str) -> None:
    original = source(transformation_chain=(TransformationStep("source", "1", (
        ("source_tabulated_mass_parameter", text),)),))
    with pytest.raises(ValueError):
        transport_srim_primary(original, table(), 1, source_mass_parameter_u=12)


def test_different_source_z_rejected() -> None:
    with pytest.raises(ValueError, match="matching Z"):
        transport_srim_primary(source(species=ParticleSpecies("N", 7, 14)), table(), 1,
                               source_mass_parameter_u=12)


def test_untyped_wrong_mode_or_already_shielded_source_rejected() -> None:
    original = source()
    # Valid Spectrum1D constructors, then reject these operator inputs.
    with pytest.raises(ValueError, match="unshielded"):
        transport_srim_primary(transport().spectrum, table(), 1, source_mass_parameter_u=12)
    with pytest.raises(ValueError, match="mass"):
        transport_srim_primary(original, table(), 1, source_mass_parameter_u=True)


def test_maximum_flux_gcr_mode_preserved() -> None:
    original = source(peak=True, source=RadiationSource.GCR, quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
                      temporal_basis=TemporalBasis.SOURCE_MAXIMUM)
    result = transport_srim_primary(original, table(), .1, source_mass_parameter_u=12)
    assert result.spectrum.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX


def test_thickness_just_below_available_anchor_has_positive_supported_measure() -> None:
    value = nextafter(3., 0.)
    result = transport(thickness=value)
    assert result.ledger.supported_exit_integral > 0
    assert result.ledger.balance_residual == 0
