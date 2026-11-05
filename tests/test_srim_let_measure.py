"""Shared LET operator witnesses, independent of native calculators and histograms."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf, nan
from typing import Any

import pytest

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.spectrum_ops import derive_spectrum
from radar.core.types import (
    AngularBasis, AssessmentMode, LetSpectrumRole, Particle, RadiationSource,
    SpectrumAxisKind, SpectrumCompleteness, SpectrumLocation, SpectrumOrdinateBasis,
    SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit
from radar.shielding.primary_let import (
    ElectronicLetProvenance, SRIM_ELECTRONIC_LET_PROVENANCE, convert_primary_electronic_let,
)
from radar.shielding.srim_hze import SrimIonStoppingTable, transport_srim_primary

from test_srim_hze_transport import native as srim_native


def source(*, cumulative: bool = False, slope: float = 0., shift: float = 0.,
           density: float = 2., bounds: tuple[float, float] = (1., 3.)) -> Spectrum1D:
    x = tuple(value + shift for value in bounds)
    return Spectrum1D(
        x=x, y=tuple(density + slope * e for e in x),
        x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=(Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY if cumulative
                else Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY),
        quantity=(SpectrumQuantity.DIFFERENTIAL_FLUENCE if cumulative
                  else SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX),
        particle=Particle.HZE, source=RadiationSource.GCR, model="analytic-source",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=TemporalBasis.MISSION_TOTAL if cumulative else TemporalBasis.SOURCE_MAXIMUM,
        location=SpectrumLocation.ON_ORBIT,
        species=ParticleSpecies("He", atomic_number=2, mass_number=4),
        assessment_mode=AssessmentMode.CUMULATIVE if cumulative else AssessmentMode.PEAK,
        mission_duration_s=9999999. if cumulative else None,
        model_reference=ModelReference("analytic-source", "1", ("analytic witness",), ("synthetic",)),
        transformation_chain=(TransformationStep("analytic_source", "1"),),
        completeness=SpectrumCompleteness.COMPLETE,
    )


@dataclass(frozen=True)
class Table:
    si_energy: tuple[float, ...] = (1., 2., 3.)
    si_electronic_let: tuple[float, ...] = (1., 3., 1.)
    mass_policy: str = "literal_source_A_numeric_transport_mass"
    mass_parameter_u: float = 4.
    dataset_ref: str = "srim:test:al-and-si-raw-sha256"


@dataclass(frozen=True)
class Ledger:
    supported_exit_integral: float


@dataclass(frozen=True)
class Measure:
    source: Spectrum1D
    spectrum: Spectrum1D
    table: Table
    ledger: Ledger
    exit_support: tuple[float, float] | None = (1., 3.)
    density: float = 2.
    slope: float = 0.
    shift: float = 0.

    def integral_on_exit(self, lower: float, upper: float) -> float:
        if self.exit_support is None:
            return 0.
        lo, hi = max(lower, self.exit_support[0]), min(upper, self.exit_support[1])
        if hi <= lo:
            return 0.
        left = self.density + self.slope * (lo + self.shift)
        right = self.density + self.slope * (hi + self.shift)
        return (hi - lo) * (.5 * left + .5 * right)


def measure(*, cumulative: bool = False, slope: float = 0., shift: float = 0.,
            density: float = 2., bounds: tuple[float, float] = (1., 3.),
            table: Table = Table()) -> Measure:
    original = source(cumulative=cumulative, slope=slope, shift=shift,
                      density=density, bounds=bounds)
    total = (bounds[1] - bounds[0]) * (density + slope * (.5 * sum(bounds) + shift))
    # Intentionally one coarse display bin. Its sub-bin shape must not be reused.
    display = derive_spectrum(
        original, x=(.5 * sum(bounds),), y=(total / (bounds[1] - bounds[0]),),
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE, bin_edges=bounds,
        location=SpectrumLocation.BEHIND_SHIELD, completeness=SpectrumCompleteness.INCOMPLETE,
        incomplete_reasons=("unresolved original source tail",),
    )
    return Measure(original, display, table, Ledger(total), bounds, density, slope, shift)


def convert(value: Any, edges: tuple[float, ...] = (1., 2., 3.)) -> Any:
    return convert_primary_electronic_let(
        value, let_edges=edges, provenance=SRIM_ELECTRONIC_LET_PROVENANCE,
    )


@pytest.mark.parametrize("cumulative", [False, True])
def test_structural_measure_nonmonotone_branches_and_typed_contract(cumulative: bool) -> None:
    incoming = measure(cumulative=cumulative)
    result = convert(incoming)
    assert result.spectrum.y == (2., 2.)
    assert result.ledger.mapped_integral == 4.
    assert result.ledger.balance_residual == 0.
    assert result.spectrum.quantity is (SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE if cumulative
                                         else SpectrumQuantity.LET_DIFFERENTIAL_FLUX)
    assert result.spectrum.y_unit is (Unit.DIFFERENTIAL_LET_FLUENCE if cumulative
                                       else Unit.DIFFERENTIAL_LET_FLUX)
    assert result.spectrum.assessment_mode is incoming.source.assessment_mode
    assert result.spectrum.temporal_basis is incoming.source.temporal_basis
    assert result.spectrum.mission_duration_s == incoming.source.mission_duration_s
    assert result.spectrum.let_role is LetSpectrumRole.HZE_SPECIES
    assert result.spectrum.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
    assert not result.normative_export_allowed
    assert not result.physical_table_accuracy_verified
    assert "unresolved original source tail" in result.spectrum.incomplete_reasons
    assert result.spectrum.transformation_chain[-1].name == "srim_electronic_let_pushforward"
    params = dict(result.spectrum.transformation_chain[-1].parameters)
    assert params["stopping_component_used"] == "electronic_collisional_only"
    assert params["dataset_ref"] == incoming.table.dataset_ref
    assert params["mass_parameter_u"] == "4.0"
    assert result.spectrum.model.endswith("+srim_electronic_let")
    assert result.spectrum.model_reference is not None
    assert incoming.table.dataset_ref in result.spectrum.model_reference.dataset_refs
    assert "stopping/hze_si_let.csv" not in result.spectrum.model_reference.dataset_refs
    assert not any("quarantined" in reason for reason in result.spectrum.incomplete_reasons)


def test_original_preimages_not_coarse_energy_histogram() -> None:
    incoming = measure(density=0., slope=1., shift=.5,
                       table=Table(si_electronic_let=(1., 2., 3.)))
    assert incoming.spectrum.y == (2.5,)
    result = convert(incoming)
    assert result.spectrum.y == pytest.approx((2., 3.))
    assert result.ledger.balance_residual == 0.


@pytest.mark.parametrize("plateau, expected", [(0., (4., 0.)), (1., (0., 4.)), (2., (0., 4.))])
def test_plateaus_include_zero_and_final_upper_edge(
    plateau: float, expected: tuple[float, float],
) -> None:
    result = convert(measure(table=Table(si_electronic_let=(plateau,) * 3)), (0., 1., 2.))
    assert result.spectrum.y == expected
    assert result.ledger.balance_residual == 0.


def test_si_and_display_window_losses_remain_separate() -> None:
    incoming = measure(bounds=(.5, 4.), table=Table(si_electronic_let=(1., 2., 3.)))
    result = convert(incoming, (1.5, 2., 2.5))
    assert result.ledger.supported_energy_integral == 7.
    assert result.ledger.mapped_integral == 2.
    assert result.ledger.below_let_window_integral == 1.
    assert result.ledger.above_let_window_integral == 1.
    assert result.ledger.outside_si_domain_integral == 3.
    assert result.ledger.balance_residual == 0.
    assert "Supported primary exits outside SRIM Si electronic-LET domain" in (
        result.spectrum.incomplete_reasons
    )


def test_no_supported_measure_is_not_complete_zero() -> None:
    incoming = replace(measure(), exit_support=None, ledger=Ledger(0.))
    result = convert(incoming)
    assert result.spectrum.y == (0., 0.)
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
    assert not result.normative_export_allowed
    assert result.ledger.mapped_integral == 0.


def test_doubling_let_grid_preserves_original_coarse_integrals() -> None:
    incoming = measure(density=1., slope=1.)
    coarse = convert(incoming)
    fine = convert(incoming, (1., 1.5, 2., 2.5, 3.))
    assert coarse.spectrum.y == pytest.approx((.5 * sum(fine.spectrum.y[:2]),
                                               .5 * sum(fine.spectrum.y[2:])))
    assert coarse.ledger.mapped_integral == fine.ledger.mapped_integral


def test_source_scaling_once_no_hidden_time_or_solid_angle() -> None:
    peak = convert(measure(density=2.))
    cum = convert(measure(density=6., cumulative=True))
    assert cum.spectrum.y == tuple(3 * v for v in peak.spectrum.y)
    assert cum.ledger.mapped_integral == 3 * peak.ledger.mapped_integral


@pytest.mark.parametrize("value", [None, object(), {}, 1, True])
def test_missing_measure_interface_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="source measure"):
        convert(value)


@pytest.mark.parametrize("field,value", [
    ("si_energy", (1.,)), ("si_energy", (2., 1., 3.)), ("si_energy", (0., 2., 3.)),
    ("si_energy", (1., inf, 3.)), ("si_energy", (True, 2., 3.)),
    ("si_energy", [1., 2., 3.]), ("si_electronic_let", (1., 2.)),
    ("si_electronic_let", (1., -2., 3.)), ("si_electronic_let", (1., nan, 3.)),
    ("si_electronic_let", (1., False, 3.)), ("si_electronic_let", [1., 2., 3.]),
    ("mass_parameter_u", 0.), ("mass_parameter_u", True), ("mass_parameter_u", inf),
    ("mass_policy", ""), ("dataset_ref", " "),
])
def test_invalid_structural_table_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        convert(measure(table=replace(Table(), **{field: value})))


@pytest.mark.parametrize("support", [(1., 1.), (2., 1.), (0., 3.), (1., inf), (1.,), [1., 3.]])
def test_invalid_exit_support_rejected(support: Any) -> None:
    with pytest.raises(ValueError):
        convert(replace(measure(), exit_support=support))


@pytest.mark.parametrize("count", [-1., nan, inf, True, 3.9, 4.1])
def test_invalid_or_false_ledger_cannot_promote_result(count: Any) -> None:
    with pytest.raises(ValueError):
        convert(replace(measure(), ledger=Ledger(count)))


@pytest.mark.parametrize("value", [-1., nan, inf, True, 1e308])
def test_invalid_integral_components_rejected(value: float, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Measure, "integral_on_exit", lambda *_: value)
    with pytest.raises(ValueError):
        convert(measure())


@pytest.mark.parametrize("field", ["model_suffix", "transformation_name", "diagnostic_reason",
                                   "outside_domain_reason"])
def test_provenance_must_be_explicit(field: str) -> None:
    with pytest.raises(ValueError, match="provenance"):
        replace(SRIM_ELECTRONIC_LET_PROVENANCE, **{field: ""})


def test_wrong_provenance_type_rejected() -> None:
    wrong: Any = None
    with pytest.raises(ValueError, match="provenance"):
        convert_primary_electronic_let(measure(), let_edges=(1., 2., 3.), provenance=wrong)


def test_caller_specific_provenance_retained_without_changing_arithmetic() -> None:
    provenance = ElectronicLetProvenance("test-let", "test-pushforward", "test pending", "test gap")
    result = convert_primary_electronic_let(measure(), let_edges=(1., 2., 3.), provenance=provenance)
    assert result.spectrum.y == (2., 2.)
    assert result.spectrum.model.endswith("+test-let")
    assert result.spectrum.transformation_chain[-1].name == "test-pushforward"
    assert "test pending" in result.spectrum.incomplete_reasons


def test_mutated_transport_time_metadata_rejected() -> None:
    incoming = measure(cumulative=True)
    changed = replace(incoming.spectrum, mission_duration_s=1.)
    with pytest.raises(ValueError, match="identity, mode or units"):
        convert(replace(incoming, spectrum=changed))


def srim_pair() -> SrimIonStoppingTable:
    al = srim_native(target=13, mass="4", z=2, energies=("4", "8", "16"),
                     stopping=("4", "4", "4"))
    si = srim_native(target=14, mass="4", z=2, energies=("4", "8", "16"),
                     stopping=("1000", "3000", "1000"), fraction="1", density="2.3290")
    return SrimIonStoppingTable.from_native(al, si)


@pytest.mark.parametrize("thickness", [0., .01, .1, 1., 3., 10.])
@pytest.mark.parametrize("cumulative", [False, True])
def test_actual_srim_transport_to_shared_let_all_shields_and_modes(
    thickness: float, cumulative: bool,
) -> None:
    original = source(cumulative=cumulative, bounds=(1., 4.))
    table = srim_pair()
    transport = transport_srim_primary(
        original, table, thickness, source_mass_parameter_u=4.,
        output_edges=None if thickness == 0 else (1., 4.),
    )
    result = convert(transport)
    supported = 2. * max(0., 3. - thickness)
    assert result.ledger.supported_energy_integral == pytest.approx(supported)
    assert result.ledger.mapped_integral == pytest.approx(supported)
    assert result.ledger.outside_si_domain_integral == 0.
    assert result.ledger.balance_residual == pytest.approx(0., abs=1e-12)
    assert transport.ledger.unresolved_low_residual_integral == pytest.approx(6. - supported)
    assert not transport.ledger.stopped_particle_classification_established
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
    assert result.spectrum.assessment_mode is original.assessment_mode
    assert result.spectrum.mission_duration_s == original.mission_duration_s
    assert result.spectrum.y_unit is (Unit.DIFFERENTIAL_LET_FLUENCE if cumulative
                                       else Unit.DIFFERENTIAL_LET_FLUX)
    assert result.spectrum.transformation_chain[-1].name == "srim_electronic_let_pushforward"
    assert dict(result.spectrum.transformation_chain[-1].parameters)["dataset_ref"] == table.dataset_ref
    if thickness == 0:
        assert transport.spectrum.x == original.x and transport.spectrum.y == original.y
    if thickness == 1.:
        # On E_exit=1..3, the nonmonotone LET curve has two branches.
        # The coarse one-bin energy display cannot supply these sub-bin counts.
        assert len(transport.spectrum.y) == 1
        assert result.spectrum.y == pytest.approx((1., 3.))


def test_actual_zero_srim_identity_and_si_domain_ledger_are_distinct() -> None:
    original = source(bounds=(.5, 5.))
    transport = transport_srim_primary(original, srim_pair(), 0., source_mass_parameter_u=4.)
    assert transport.spectrum.x == original.x and transport.spectrum.y == original.y
    result = convert(transport)
    assert transport.ledger.outside_al_domain_integral == 0.
    assert result.ledger.supported_energy_integral == 9.
    assert result.ledger.outside_si_domain_integral == 3.
    assert result.ledger.mapped_integral == 6.
    assert result.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
