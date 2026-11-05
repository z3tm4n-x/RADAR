"""Actual paired-SRIM product integration; synthetic physics, no native executable."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from functools import lru_cache
from math import inf, nan
from pathlib import Path
import re
from typing import Any

import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationProductKind, RadiationSource,
    SpectrumAxisKind, SpectrumCompleteness, SpectrumLocation, SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit
from radar.pipelines.srim_hze import SrimHzePipelineResult, calculate_srim_hze_products
from radar.shielding.srim_hze import SrimIonStoppingTable
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output

SHIELDS = (0., .01, .1, 1., 3., 10.)
LET_EDGES = (0., .001, .002, .003)


def _native(target: int, *, mass: str = "55.8", z: int = 26) -> SrimRangeOutput:
    fixture = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    lines = [line.strip() if "(C)" in line else re.sub(r"(?<=[0-9]),(?=[0-9])", ".", line.strip())
             for line in fixture.read_bytes().decode("cp1251").splitlines() if line.strip()]
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC PIPELINE WITNESS ONLY"
    header[6] = header[17] = f"Ion = Synthetic [{z}] , Mass = {mass} amu"
    header[7] = "Density = 2.70 g/cm3 = 6E22 atoms/cm3"
    header[12] = f"{'Al' if target == 13 else 'Si'} {target} 100 100"
    footer = lines[lines.index("-" * 59):]
    with localcontext() as context:
        context.prec = 60
        # Constant whole-ion total stopping2 MeVcm2/g. Si electronic LET.0015.
        rows = [f"{Decimal(mass)*e} MeV 0.0015 0.0005 {i+1} cm 0.1 cm 0.1 cm"
                for i, e in enumerate((1, 10, 100))]
    return parse_srim_range_output(("\r\n".join(header + rows + footer) + "\r\n").encode("ascii"))


@lru_cache(maxsize=4)
def _table(mass: str = "55.8", z: int = 26) -> SrimIonStoppingTable:
    return SrimIonStoppingTable.from_native(_native(13, mass=mass, z=z), _native(14, mass=mass, z=z))


def _product(*, source: RadiationSource = RadiationSource.SEP, peak: bool = False,
             trace: tuple[str, ...] = ("55.8",), **changes: Any) -> SpectrumProduct:
    kind = (RadiationProductKind.MISSION_FLUENCE if not peak else
            RadiationProductKind.MAXIMUM_FLUX if source is RadiationSource.GCR else
            RadiationProductKind.PEAK_FLUX)
    quantity = (SpectrumQuantity.DIFFERENTIAL_FLUENCE if not peak else
                SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX if source is RadiationSource.GCR else
                SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX)
    temporal = (TemporalBasis.MISSION_TOTAL if not peak else
                TemporalBasis.SOURCE_MAXIMUM if source is RadiationSource.GCR else
                TemporalBasis.SEP_PEAK)
    values: dict[str, Any] = {
        "x": (10., 30., 80.), "y": (2., 2., 2.),
        "x_unit": Unit.MEV_PER_NUCLEON,
        "y_unit": (Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY if peak else
                   Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY),
        "quantity": quantity, "particle": Particle.HZE, "source": source,
        "model": "synthetic_pipeline_source", "axis_kind": SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        "angular_basis": AngularBasis.OMNIDIRECTIONAL_4PI, "temporal_basis": temporal,
        "assessment_mode": AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE,
        "mission_duration_s": None if peak else 123456789.,
        "location": SpectrumLocation.ON_ORBIT, "species": ParticleSpecies("Fe", 26, 56),
        "model_reference": ModelReference("synthetic_pipeline_source", "1", ("synthetic",), ("synthetic",)),
        "transformation_chain": tuple(TransformationStep(f"source_mass_{i}", "1", (
            ("source_tabulated_mass_parameter", value),
        )) for i, value in enumerate(trace)), "completeness": SpectrumCompleteness.COMPLETE,
    }
    values.update(changes)
    return SpectrumProduct(kind, Spectrum1D(**values), "Synthetic source")


def _run(**changes: Any) -> SrimHzePipelineResult:
    values: dict[str, Any] = {
        "products": (_product(),), "tables": {26: _table()},
        "thicknesses_g_cm2": (1.,), "assessment_mode": AssessmentMode.CUMULATIVE,
        "let_edges": LET_EDGES,
    }
    values.update(changes)
    return calculate_srim_hze_products(**values)


@pytest.mark.parametrize("source", [RadiationSource.GCR, RadiationSource.SEP])
@pytest.mark.parametrize("peak", [False, True])
def test_both_modes_all_six_shields_return_real_typed_energy_and_let_products(
    source: RadiationSource, peak: bool,
) -> None:
    upstream = _product(source=source, peak=peak)
    mode = AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE
    result = _run(products=(upstream,), assessment_mode=mode, thicknesses_g_cm2=SHIELDS)
    assert result.assessment_mode is mode and len(result.cases) == 6
    assert len(result.products) == 12
    expected_let_kind = {
        RadiationProductKind.MISSION_FLUENCE: RadiationProductKind.MISSION_LET_FLUENCE,
        RadiationProductKind.MAXIMUM_FLUX: RadiationProductKind.MAXIMUM_LET_FLUX,
        RadiationProductKind.PEAK_FLUX: RadiationProductKind.PEAK_LET_FLUX,
    }[upstream.kind]
    for thickness, case in zip(SHIELDS, result.cases, strict=True):
        assert case.thickness_g_cm2 == thickness and case.z == 26 and case.source is source
        assert case.source_kind is upstream.kind
        assert case.energy_product is not None and case.let_product is not None
        assert case.transport is not None and case.let_result is not None
        assert case.energy_product.kind is upstream.kind
        assert case.let_product.kind is expected_let_kind
        assert case.transport.ledger.source_integral == pytest.approx(140.)
        assert case.transport.ledger.supported_exit_integral == pytest.approx(140.)
        assert case.transport.ledger.outside_al_domain_integral == 0
        assert case.transport.ledger.unresolved_low_residual_integral == 0
        assert case.let_result.ledger.mapped_integral == pytest.approx(140.)
        assert case.let_product.spectrum.integral_conserved() == pytest.approx(140.)
        assert case.let_product.spectrum.y == pytest.approx((0., 140000., 0.))
        for product in (case.energy_product, case.let_product):
            assert product.spectrum.assessment_mode is mode
            assert product.spectrum.temporal_basis is upstream.spectrum.temporal_basis
            assert product.spectrum.mission_duration_s == upstream.spectrum.mission_duration_s
            assert product.spectrum.angular_basis is upstream.spectrum.angular_basis
            assert product.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
            assert product.spectrum.production_contract_violations() == (
                "incomplete spectra cannot be released as production results",
            )
        assert case.completeness is SpectrumCompleteness.INCOMPLETE
        assert not case.normative_export_allowed
        assert case.incomplete_reasons
    assert result.completeness is SpectrumCompleteness.INCOMPLETE
    assert not result.normative_export_allowed and result.incomplete_reasons


@pytest.mark.parametrize("peak", [False, True])
def test_zero_shield_preserves_all_source_nodes_even_beyond_both_native_tables(peak: bool) -> None:
    source = _product(peak=peak, x=(.1, 1., 10., 100., 200.), y=(2.,) * 5)
    result = _run(products=(source,), assessment_mode=source.spectrum.assessment_mode,
                  thicknesses_g_cm2=(0.,), output_energy_edges=(2., 5., 20.))
    case = result.cases[0]
    assert case.energy_product is not None and case.transport is not None and case.let_result is not None
    spectrum = case.energy_product.spectrum
    assert spectrum.x is source.spectrum.x and spectrum.y is source.spectrum.y
    assert spectrum.bin_edges == source.spectrum.bin_edges
    assert spectrum.ordinate_basis is source.spectrum.ordinate_basis
    assert case.transport.ledger.outside_al_domain_integral == 0
    assert case.transport.ledger.supported_exit_integral == pytest.approx(399.8)
    assert case.let_result.ledger.mapped_integral == pytest.approx(198.)
    assert case.let_result.ledger.outside_si_domain_integral == pytest.approx(201.8)
    assert abs(case.let_result.ledger.balance_residual) < 1e-10


def test_positive_shield_keeps_unsupported_incident_contribution_explicit() -> None:
    source = _product(x=(.1, 1., 10., 100., 200.), y=(2.,) * 5)
    case = _run(products=(source,)).cases[0]
    assert case.transport is not None and case.let_result is not None
    ledger = case.transport.ledger
    assert ledger.source_integral == pytest.approx(399.8)
    assert ledger.outside_al_domain_integral == pytest.approx(201.8)
    assert ledger.unresolved_low_residual_integral == pytest.approx(4. / 55.8)
    assert ledger.supported_exit_integral == pytest.approx(198. - 4. / 55.8)
    assert ledger.represented_integral == pytest.approx(ledger.supported_exit_integral)
    assert case.let_result.ledger.mapped_integral == pytest.approx(ledger.supported_exit_integral)
    assert not ledger.stopped_particle_classification_established
    assert not ledger.source_tails_known
    assert abs(ledger.balance_residual) < 1e-10


def test_all_sources_and_modes_are_filtered_without_merging_gcr_and_sep() -> None:
    products = tuple(_product(source=source, peak=peak)
                     for source in (RadiationSource.GCR, RadiationSource.SEP) for peak in (False, True))
    for mode in AssessmentMode:
        result = _run(products=products, assessment_mode=mode)
        assert len(result.cases) == 2 and len(result.products) == 4
        assert {case.source for case in result.cases} == {RadiationSource.GCR, RadiationSource.SEP}
        assert all(product.spectrum.assessment_mode is mode for product in result.products)


def test_non_hze_products_are_ignored_and_unsupported_mode_is_not_computed() -> None:
    hze = _product()
    proton = SpectrumProduct(hze.kind, replace(
        hze.spectrum, particle=Particle.PROTON, species=ParticleSpecies("p", 1, 1),
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY, x_unit=Unit.MEV, y_unit=Unit.DIFFERENTIAL_FLUENCE,
    ))
    result = _run(products=(proton, _product(peak=True), hze))
    assert len(result.cases) == 1 and result.cases[0].z == 26


@pytest.mark.parametrize("peak", [False, True])
def test_missing_ion_does_not_create_zero_spectrum_or_legacy_fallback(peak: bool) -> None:
    source = _product(peak=peak)
    result = _run(products=(source,), tables={}, assessment_mode=source.spectrum.assessment_mode,
                  thicknesses_g_cm2=SHIELDS)
    assert len(result.cases) == 6 and len(result.products) == 1
    first = result.cases[0]
    assert first.energy_product is not None
    assert first.energy_product.spectrum.x is source.spectrum.x
    assert first.energy_product.spectrum.y is source.spectrum.y
    assert first.energy_product.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
    assert first.let_product is None and first.transport is None and first.let_result is None
    for case in result.cases[1:]:
        assert case.energy_product is None and case.let_product is None
        assert case.transport is None and case.let_result is None
        assert case.completeness is SpectrumCompleteness.INCOMPLETE
    assert any("Z=26" in reason for reason in result.incomplete_reasons)


def test_partial_missing_roster_keeps_supported_ions_and_explicit_missing_cases() -> None:
    absent = _product(species=ParticleSpecies("Ni", 28, 59), trace=("58.7",))
    result = _run(products=(_product(), absent))
    assert len(result.cases) == 2
    assert result.cases[0].energy_product is not None
    assert result.cases[1].z == 28 and result.cases[1].energy_product is None
    assert any("Z=28" in reason for reason in result.incomplete_reasons)


def test_seventeen_digit_mass_trace_matches_literal_table_without_isotope_rounding() -> None:
    source = _product(trace=(format(55.8, ".17g"),))
    assert source.spectrum.species is not None and source.spectrum.species.mass_number == 56
    case = _run(products=(source,)).cases[0]
    assert case.transport is not None
    assert case.transport.table.mass_parameter_u_decimal == Decimal("55.8")
    assert case.transport.table.native_al.ion_mass_u != 56


@pytest.mark.parametrize("trace", [(), ("55.8", "56"), ("nan",), ("inf",),
                                    ("-1",), ("0",), ("not-a-mass",)])
@pytest.mark.parametrize("missing", [False, True])
def test_missing_conflicting_or_nonfinite_mass_trace_is_never_inferred(
    trace: tuple[str, ...], missing: bool,
) -> None:
    with pytest.raises(ValueError):
        _run(products=(_product(trace=trace),), tables={} if missing else {26: _table()},
             thicknesses_g_cm2=(0.,))


def test_matching_repeated_mass_trace_is_allowed_but_integer_table_substitution_is_not() -> None:
    assert len(_run(products=(_product(trace=("55.8", "55.799999999999997")),)).cases) == 1
    with pytest.raises(ValueError, match="mass"):
        _run(tables={26: _table("56")})


def test_duration_and_flux_fluence_semantics_do_not_apply_hidden_time_factor() -> None:
    a = _run(products=(_product(mission_duration_s=1.),)).cases[0]
    b = _run(products=(_product(mission_duration_s=1e12),)).cases[0]
    c = _run(products=(_product(peak=True),), assessment_mode=AssessmentMode.PEAK).cases[0]
    assert a.transport is not None and b.transport is not None and c.transport is not None
    assert a.let_result is not None and b.let_result is not None and c.let_result is not None
    assert a.transport.ledger == b.transport.ledger == c.transport.ledger
    assert a.let_result.ledger == b.let_result.ledger == c.let_result.ledger
    assert a.let_result.spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUENCE
    assert c.let_result.spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUX


def test_source_scaling_is_linear_and_input_incomplete_reason_survives() -> None:
    source = _product(completeness=SpectrumCompleteness.INCOMPLETE,
                      incomplete_reasons=("upstream source incomplete witness",))
    first = _run(products=(source,)).cases[0]
    scaled = replace(source, spectrum=replace(source.spectrum, y=(14., 14., 14.)))
    second = _run(products=(scaled,)).cases[0]
    assert first.let_result is not None and second.let_result is not None
    assert second.let_result.spectrum.y == pytest.approx(tuple(7*x for x in first.let_result.spectrum.y))
    assert "upstream source incomplete witness" in first.incomplete_reasons


@pytest.mark.parametrize("changes", [
    {"products": ()}, {"products": (_product(peak=True),)},
    {"products": (None,)}, {"products": (_product(), _product())},
    {"assessment_mode": "cumulative"}, {"assessment_mode": True},
    {"thicknesses_g_cm2": ()}, {"thicknesses_g_cm2": [1.]},
    {"thicknesses_g_cm2": (1., 1.)}, {"thicknesses_g_cm2": (0., -0.)},
    {"thicknesses_g_cm2": (-1.,)}, {"thicknesses_g_cm2": (nan,)},
    {"thicknesses_g_cm2": (inf,)}, {"thicknesses_g_cm2": (True,)},
    {"thicknesses_g_cm2": ("1",)}, {"tables": {25: _table()}},
    {"tables": {"26": _table()}}, {"tables": {26: None}},
])
def test_invalid_batch_requests_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _run(**changes)


@pytest.mark.parametrize("field,bad", [
    ("let_edges", ()), ("let_edges", (1.,)), ("let_edges", (1., 1.)),
    ("let_edges", (-1., 1.)), ("let_edges", (0., nan)), ("let_edges", [0., 1.]),
    ("let_edges", (False, 1.)), ("output_energy_edges", (0., 1.)),
    ("output_energy_edges", (1., 1.)), ("output_energy_edges", (1., inf)),
])
def test_bad_output_grid_is_rejected_even_when_every_ion_is_missing(field: str, bad: Any) -> None:
    with pytest.raises(ValueError):
        _run(tables={}, **{field: bad})


def test_sources_must_be_individual_and_unshielded() -> None:
    for spectrum in (
        replace(_product().spectrum, species=ParticleSpecies("all", is_aggregate=True)),
        replace(_product().spectrum, location=SpectrumLocation.BEHIND_SHIELD),
    ):
        with pytest.raises(ValueError):
            _run(products=(SpectrumProduct(RadiationProductKind.MISSION_FLUENCE, spectrum),))


@pytest.mark.parametrize("peak", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_integral_only_presentations_are_not_treated_as_differential_inputs(
    peak: bool, missing: bool,
) -> None:
    source = _product(peak=peak)
    spectrum = replace(source.spectrum,
                       quantity=(SpectrumQuantity.PEAK_INTEGRAL_FLUX if peak else
                                 SpectrumQuantity.INTEGRAL_FLUENCE),
                       y_unit=Unit.INTEGRAL_FLUX if peak else Unit.INTEGRAL_FLUENCE)
    with pytest.raises(ValueError, match="No individual HZE"):
        _run(products=(SpectrumProduct(source.kind, spectrum),),
             tables={} if missing else {26: _table()},
             assessment_mode=AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE)


@pytest.mark.parametrize("missing", [False, True])
def test_total_energy_axis_is_not_misread_as_per_source_mass_even_without_table(missing: bool) -> None:
    spectrum = replace(_product().spectrum, axis_kind=SpectrumAxisKind.KINETIC_ENERGY,
                       x_unit=Unit.MEV, y_unit=Unit.DIFFERENTIAL_FLUENCE)
    with pytest.raises(ValueError):
        _run(products=(SpectrumProduct(RadiationProductKind.MISSION_FLUENCE, spectrum),),
             tables={} if missing else {26: _table()})


def test_no_legacy_resource_is_attached_to_new_srim_products() -> None:
    case = _run().cases[0]
    assert case.energy_product is not None and case.let_product is not None
    for product in (case.energy_product, case.let_product):
        ref = product.spectrum.model_reference
        assert ref is not None and _table().dataset_ref in ref.dataset_refs
        assert not any("hze_si_let.csv" in value for value in ref.dataset_refs)
        assert not any("quarantined" in reason.lower() for reason in product.spectrum.incomplete_reasons)


@pytest.mark.parametrize("source", [RadiationSource.GCR, RadiationSource.SEP])
def test_real_upstream_source_products_can_be_consumed_without_manual_relabelling(
    source: RadiationSource,
) -> None:
    from radar.core.project import MissionConfig
    from radar.gcr.components import OstGcrSolarState
    from radar.gcr.products import GcrFluxTimeBin, calculate_gcr_mission_products
    from radar.gcr.source_spectra import calculate_ost_gcr_source_spectra_outside_magnetosphere
    from radar.sep.model import OstSepModel, SepModelInput

    grid = (10., 30., 80.)
    products: tuple[SpectrumProduct, ...]
    if source is RadiationSource.GCR:
        state = OstGcrSolarState(wolf_current=60., wolf_lagged=50., wolf_min=10., wolf_max=100.,
                                cycle_number=24, after_polarity_reversal=True)
        external = calculate_ost_gcr_source_spectra_outside_magnetosphere(
            energy_grid_mev_per_nucleon=grid, solar_state=state, nek_wolf_number=60.,
            akl_years_from_cycle_start=2., symbols=("Fe",),
        )
        # Includes both differential and integral variants with the same product kind.
        products = calculate_gcr_mission_products((GcrFluxTimeBin(external.ion("Fe").total,
                                                                  5 * 365.25 * 86400),)).products
    else:
        model = OstSepModel(energy_grid_mev=grid, hze_energy_grid_mev_per_nucleon=grid,
                            monthly_smoothed_wolf_numbers=(60.,) * 60)
        result = model.calculate(SepModelInput(MissionConfig(
            launch_year=2026, lifetime_years=5, sep_exceedance_probability=.1,
        )))
        products = tuple(p for p in result.products if p.spectrum.particle is Particle.HZE
                         and p.spectrum.species is not None and p.spectrum.species.atomic_number == 26)
    for mode in AssessmentMode:
        batch = _run(products=products, assessment_mode=mode, thicknesses_g_cm2=(0., .1))
        assert len(batch.cases) == 2
        for case in batch.cases:
            assert case.source is source and case.z == 26
            assert case.energy_product is not None and case.let_product is not None
            assert case.transport is not None and case.let_result is not None
            assert case.transport.ledger.source_integral > 0
            assert case.transport.ledger.supported_exit_integral == pytest.approx(
                case.transport.ledger.source_integral,
            )
            assert case.let_result.ledger.mapped_integral == pytest.approx(
                case.transport.ledger.supported_exit_integral,
            )
            assert all(product.spectrum.assessment_mode is mode
                       for product in (case.energy_product, case.let_product))
