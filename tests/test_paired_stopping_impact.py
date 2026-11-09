"""Independent analytic cases; no scientific archive or native runtime needed."""
from dataclasses import replace
from math import exp, inf, log, nextafter
from pathlib import Path
import sys
from typing import Any

import pytest

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationSource, SpectrumAxisKind,
    SpectrumCompleteness, SpectrumLocation, SpectrumOrdinateBasis, SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paired_stopping_impact as impact  # noqa: E402


def source(peak: bool = False) -> Spectrum1D:
    return Spectrum1D(x=(.5, 5., 12.), y=(1., 1., 1.), x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY if peak else Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
        quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX if peak else SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.HZE, source=RadiationSource.SEP, model="test",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON, angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=TemporalBasis.SEP_PEAK if peak else TemporalBasis.MISSION_TOTAL,
        assessment_mode=AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE,
        mission_duration_s=None if peak else 731234., location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
        species=ParticleSpecies("Fe", 26, 56), model_reference=ModelReference("test", "1", ("test",), ("test",)),
        transformation_chain=(TransformationStep("test", "1", (("source_tabulated_mass_parameter", "55.8"),)),),
        completeness=SpectrumCompleteness.COMPLETE)


def pair(**changes: Any) -> impact.PreparedPair:
    values: dict[str, Any] = dict(z=26, mass=55.8, common_energy=(1., 2., 3., 4., 5.),
        srim_al_total=(55.8,)*5, catima_al_total=(111.6,)*5,
        srim_si_electronic=(1000., 3000., 1000., 3000., 1000.), catima_si_electronic=(2000.,)*5,
        extended_energy=(1., 2., 3., 4., 5., 7., 10.), extended_catima_al_total=(111.6,)*7,
        extended_catima_si_electronic=(2000.,)*7)
    values.update(changes)
    return impact.prepare_pair(**values)


@pytest.mark.parametrize("peak", [True, False])
def test_independent_factorial_constant_stopping_witness(peak: bool) -> None:
    # S_Al=A or 2A implies E_out = E_in - t or E_in - 2t respectively.
    result = impact.evaluate_case(source(peak), pair(), 1.)
    assert result["integral_unit"] == ("cm^-2 s^-1" if peak else "cm^-2")
    assert result["source_integral"] == 11.5
    assert result["input_partition"] == {"below_common": .5, "common": 4., "upper_catima_only": 5., "above_catima": 2.}
    for label, expected in {"SS": (3., 1., 1.5), "SC": (3., 1., 3.), "CS": (2., 2., 1.), "CC": (2., 2., 2.)}.items():
        row = result["variants"][label]
        assert (row["supported_exit_integral"], row["unresolved_low_residual_integral"], row["let_ge"]["2.0"]) == pytest.approx(expected)
        assert sum(row["let_bin_integrals"]) == pytest.approx(expected[0])
        assert sum(row["energy_bin_integrals"]) == pytest.approx(expected[0])
        assert row["energy_histogram_balance_residual"] == pytest.approx(0., abs=1e-14)
        assert row["let_histogram_balance_residual"] == pytest.approx(0., abs=1e-14)
    upper = result["upper_catima_only"]
    assert upper["source_partition_integral"] == 5.
    assert upper["supported_exit_integral"] == pytest.approx(5.)
    assert upper["below_seam_exit_integral"] == pytest.approx(2.)
    assert upper["exit_support"] == pytest.approx([3., 8.])
    assert upper["let_ge"]["2.0"] == pytest.approx(5.)
    assert not result["normative_export_allowed"] and not result["source_outer_tails_known"]


def test_zero_shield_identity_and_unmapped_outside_counts() -> None:
    result = impact.evaluate_case(source(), pair(), 0.)
    assert result["zero_shield_energy_identity"]
    assert result["zero_shield_identity_unmapped_energy_integral"] == 2.5
    rows = result["variants"]
    assert all(r["source_partition_integral"] == r["supported_exit_integral"] for r in rows.values())
    assert all(r["energy_bin_integrals"] == rows["SS"]["energy_bin_integrals"] for r in rows.values())
    assert rows["CS"]["let_bin_integrals"] == rows["SS"]["let_bin_integrals"]
    assert rows["SC"]["let_bin_integrals"] == rows["CC"]["let_bin_integrals"]
    assert result["upper_catima_only"]["below_seam_exit_integral"] == 0


def test_source_cut_inside_nonuniform_linear_measure() -> None:
    # j(E)=2E+1. Exact primitive E²+E, not a histogram regenerated on SRIM nodes.
    original = replace(source(), x=(.5, 3., 12.), y=(2., 7., 25.))
    result = impact.evaluate_case(original, pair(), 1.)
    assert result["source_integral"] == pytest.approx(156.-.75)
    assert result["input_partition"]["common"] == pytest.approx(30.-2.)
    upper = result["upper_catima_only"]
    assert upper["supported_exit_integral"] == pytest.approx(110.-30.)
    assert upper["below_seam_exit_integral"] == pytest.approx(56.-30.)


def test_binned_source_is_not_linear_between_display_centers() -> None:
    binned = replace(source(), x=(1.75, 7.5), bin_edges=(.5, 3., 12.), y=(1., 4.),
                     ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE)
    result = impact.evaluate_case(binned, pair(), 1.)
    assert result["input_partition"] == {"below_common": .5, "common": 10., "upper_catima_only": 20., "above_catima": 8.}
    assert result["variants"]["CS"]["supported_exit_integral"] == pytest.approx(8.)


@pytest.mark.parametrize("power", [-2., -1., 0., 1., 1.+1e-9, 2., 3.5])
def test_path_range_matches_power_integral_and_inverse(power: float) -> None:
    mass, nodes = 4., (1., 2., 8.)
    curve = impact.PathRange.build(nodes, tuple(n**power for n in nodes), mass)
    for x in (1., 1.25, 2., 3.4, 6.125, 8.):
        from math import expm1
        expected = mass*(log(x) if power == 1 else expm1((1-power)*log(x))/(1-power))
        assert curve.at(x) == pytest.approx(expected, rel=2e-14, abs=0.)
        assert curve.invert(curve.at(x)) == pytest.approx(x, rel=2e-14)


def test_lognormal_slope_one_loss_not_linear_in_energy() -> None:
    model = impact.PathRange.build((1., 3., 10.), (1., 3., 10.), 4.)
    expected = 3. * exp(1./4.)
    assert model.invert(model.at(3.)+1.) == pytest.approx(expected)


def test_plateau_at_threshold_included_and_integrals_recover_densities() -> None:
    result = impact.evaluate_case(source(), pair(), 1.)["variants"]["CC"]
    assert result["let_ge"]["2.0"] == pytest.approx(2.)
    assert result["let_ge"]["5.0"] == 0
    for edges, prefix in ((impact.ENERGY_EDGES, "energy"), (impact.LET_EDGES, "let")):
        recovered = [v*(b-a) for v, a, b in zip(result[prefix+"_bin_average_density"], edges, edges[1:])]
        assert recovered == pytest.approx(result[prefix+"_bin_integrals"])


def test_one_sided_jump_not_smeared_in_neighboring_cells() -> None:
    b = 50000.
    grid = (40000., nextafter(b, -inf), b, nextafter(b, inf), 60000.)
    path = impact.PathRange.build(grid, (2., 2., 2., 4., 4.), 4.)
    assert path.curve.energy == (40000., b, 60000.)
    assert path.at(b) == pytest.approx(20000.)
    assert path.at(60000.) == pytest.approx(30000.)
    assert path.invert(20001.) == pytest.approx(50001.)
    assert path.curve.left == (2., 4.) and path.curve.right == (2., 4.)


@pytest.mark.parametrize("grid", [(40000., 50000., 51000.), (40000., 49000., 51000.), (49999., 50000., nextafter(50000., inf))])
def test_inadequate_transition_sampling_rejected(grid: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        impact.Segments.build(grid, tuple(1. for _ in grid))


@pytest.mark.parametrize("x", [-1., float("nan"), float("inf"), True])
def test_invalid_shield(x: float) -> None:
    with pytest.raises(ValueError):
        impact.evaluate_case(source(), pair(), x)


@pytest.mark.parametrize("x", [9., 10., 1000.])
def test_all_low_residual_counts_are_unknown_not_zero(x: float) -> None:
    result = impact.evaluate_case(source(), pair(), x)
    for row in (*result["variants"].values(), result["upper_catima_only"]):
        assert row["supported_exit_integral"] == 0.
        assert row["unresolved_low_residual_integral"] == row["source_partition_integral"]
        assert row["exit_support"] is None


def test_upper_partition_empty_is_not_a_missing_ion_or_an_unbounded_domain() -> None:
    s = replace(source(), x=(1., 2., 4.5))
    result = impact.evaluate_case(s, pair(), 1.)
    assert result["input_partition"]["upper_catima_only"] == 0.
    assert result["upper_catima_only"]["source_partition_integral"] == 0.
    assert not result["source_outer_tails_known"]


@pytest.mark.parametrize("changes", [dict(z=True), dict(mass=25.), dict(common_energy=(2., 1.)),
    dict(catima_al_total=(1.,)*5), dict(extended_energy=(1., 2., 3., 4., 5.)),
    dict(srim_al_total=(0.,)*5), dict(srim_si_electronic=(float("nan"),)*5),
    dict(extended_catima_si_electronic=(1.,)*7)])
def test_invalid_table_identity_or_stopping_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        pair(**changes)


@pytest.mark.parametrize("changes", [dict(transformation_chain=()),
    dict(transformation_chain=(TransformationStep("test", "1", (("source_tabulated_mass_parameter", "56"),)),)),
    dict(species=ParticleSpecies("Cr", 24, 52)), dict(location=SpectrumLocation.BEHIND_SHIELD)])
def test_missing_mass_mismatch_or_repeated_shield_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        impact.evaluate_case(replace(source(), **changes), pair(), 0.)
