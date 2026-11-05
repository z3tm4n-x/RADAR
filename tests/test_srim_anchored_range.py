"""Synthetic analytic controls; never run SRIM or qualify physical data."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
from decimal import Decimal, ROUND_DOWN, localcontext
from pathlib import Path
import re
from typing import Any

import pytest

from radar.validation.range_stopping_consistency import audit_range_stopping_consistency
from radar.validation.srim_anchored_range import AnchoredSrimRangeTable
from radar.validation.srim_output import SrimRangeOutput, parse_srim_range_output


D = Decimal
CHOICES = ("linear_stopping", "loglog_stopping")


def native(*, energies: tuple[str, ...] = ("1", "2", "4"),
           stopping: tuple[str, ...] = ("2", "2", "2"), mass: str = "12",
           z: int = 6, density: str = "2.70", target: int = 13,
           electronic_fraction: str = ".75", projected_scale: str = "1") -> SrimRangeOutput:
    fixture = Path(__file__).parent / "fixtures/srim_h_al_smoke.txt"
    lines = [line.strip() if "(C)" in line else re.sub(r"(?<=[0-9]),(?=[0-9])", ".", line.strip())
             for line in fixture.read_bytes().decode("cp1251").splitlines() if line.strip()]
    header = lines[:21]
    header[3] = "Calc. date ---> SYNTHETIC NOT PHYSICAL DATA"
    header[6] = header[17] = f"Ion = Synthetic [{z}] , Mass = {mass} amu"
    header[7] = f"Density = {density} g/cm3 = 6E22 atoms/cm3"
    header[12] = f"{'Al' if target == 13 else 'Si'} {target} 100 100"
    footer = lines[lines.index("-" * 59):]
    with localcontext() as context:
        context.prec = 100
        rows = [
            f"{energy} MeV {D(s)*D(electronic_fraction)/1000} "
            f"{D(s)*(1-D(electronic_fraction))/1000} "
            f"{(i+1)*D(projected_scale)} cm 0.1 cm 0.1 cm"
            for i, (energy, s) in enumerate(zip(energies, stopping, strict=True))
        ]
    return parse_srim_range_output(("\r\n".join(header + rows + footer) + "\r\n").encode("ascii"))


def table(interpolation: Any, **kwargs: Any) -> AnchoredSrimRangeTable:
    return AnchoredSrimRangeTable.from_native(native(**kwargs), interpolation=interpolation)


def close(actual: Decimal, expected: Decimal, tolerance: str = "1e-76") -> None:
    with localcontext() as context:
        context.prec = 150
        assert abs(actual - expected) <= D(tolerance) * max(abs(expected), D(1))


@pytest.mark.parametrize("choice", CHOICES)
def test_constant_stopping_analytic_range_inverse_and_shielding(choice: Any) -> None:
    result = table(choice)
    assert result.anchored_range_g_cm2 == (D(0), D('.5'), D('1.5'))
    close(result.range_at_energy(D(3)), D(1))
    close(result.energy_at_range(D(1)), D(3))
    close(result.residual_energy(D(4), D('.5')), D(3))  # type: ignore[arg-type]
    assert result.residual_energy(D(4), D(0)) == D(4)
    assert result.residual_energy(D(4), D('1.5')) == D(1)
    assert result.residual_energy(D(4), D('1.50001')) is None
    assert result.residual_energy(D(1), D('.01')) is None


@pytest.mark.parametrize("choice", CHOICES)
def test_construction_matches_existing_interval_diagnostic(choice: Any) -> None:
    result = table(choice, stopping=("3", "5", "9"))
    diagnostic = audit_range_stopping_consistency(
        energy_total_mev=result.energy_total_mev,
        total_stopping_mev_cm2_g=result.total_stopping_mev_cm2_g,
        range_g_cm2=result.native_projected_range_g_cm2, range_kind="projected")
    wanted = (D(0), *(getattr(row, choice).anchored_integral_g_cm2
                      for row in diagnostic.intervals))
    assert result.anchored_range_g_cm2 == wanted


def test_linear_stopping_analytic_partial_integral_and_inverse() -> None:
    result = table("linear_stopping", stopping=("3", "5", "9"))
    with localcontext() as context:
        context.prec = 80
        wanted = (D(7)/3).ln()/2
    close(result.range_at_energy(D(3)), wanted)
    close(result.energy_at_range(wanted), D(3))


@pytest.mark.parametrize("power", [-3, -1, 0, 1, 2, 4])
def test_loglog_stopping_analytic_partial_integral_and_inverse(power: int) -> None:
    with localcontext() as context:
        context.prec = 100
        stopping = tuple(str(3*D(e)**power) for e in (1, 2, 4))
        context.prec = 80
        wanted = (D(3).ln()/3 if power == 1 else (D(3)**(1-power)-1)/(3*(1-power)))
    result = table("loglog_stopping", stopping=stopping)
    close(result.range_at_energy(D(3)), wanted)
    close(result.energy_at_range(wanted), D(3))


@pytest.mark.parametrize("choice", CHOICES)
@pytest.mark.parametrize("stopping", [("2", "2", "2"), ("3", "5", "9"),
                                     ("32", "2", ".125"), (".125", "2", "32")])
def test_roundtrips_monotonicity_and_exact_nodes(choice: Any, stopping: tuple[str, ...]) -> None:
    result = table(choice, stopping=stopping)
    previous = D(-1)
    for raw in ("1", "1.1", "1.5", "1.99", "2", "2.01", "2.5", "3.9", "4"):
        energy = D(raw)
        r = result.range_at_energy(energy)
        assert r > previous
        close(result.energy_at_range(r), energy)
        previous = r
    for energy, r in zip(result.energy_total_mev, result.anchored_range_g_cm2, strict=True):
        assert result.range_at_energy(energy) == r
        assert result.energy_at_range(r) == energy


@pytest.mark.parametrize("choice", CHOICES)
def test_close_to_constant_and_close_to_power_one(choice: Any) -> None:
    almost_constant = table(choice, stopping=("2", "2.0000000000000000000000000000000000000001",
                                              "2.0000000000000000000000000000000000000002"))
    almost_power_one = table(choice, stopping=("1", "2.0000000000000000000000000000000000000001",
                                               "4.0000000000000000000000000000000000000002"))
    for result in (almost_constant, almost_power_one):
        for energy in (D('1.5'), D('2.5'), D('3.5')):
            close(result.energy_at_range(result.range_at_energy(energy)), energy)


@pytest.mark.parametrize("choice", CHOICES)
def test_native_mass_density_and_projected_range_do_not_rescale_fixed_stopping(choice: Any) -> None:
    # Fixing S is intentional: this does not claim physical S is density-independent.
    first = table(choice)
    changed = table(choice, z=26, mass="55.8", density="9.9", projected_scale="100")
    assert first.anchored_range_g_cm2 == changed.anchored_range_g_cm2
    assert changed.ion_mass_u == D('55.8') and changed.ion_z == 26
    assert changed.native_projected_range_g_cm2 != first.native_projected_range_g_cm2
    assert changed.native_raw_sha256 != first.native_raw_sha256
    assert changed.native_semantic_sha256 != first.native_semantic_sha256
    assert not changed.isotope_selection_policy_established


@pytest.mark.parametrize("choice", CHOICES)
def test_total_energy_scaling_needs_mass_factor_not_stopping_division(choice: Any) -> None:
    first = table(choice)
    changed = table(choice, energies=("12", "24", "48"))
    for before, after in zip(first.anchored_range_g_cm2, changed.anchored_range_g_cm2, strict=True):
        close(after, 12*before)


@pytest.mark.parametrize("choice", CHOICES)
@pytest.mark.parametrize("fraction", ["0", ".25", "1"])
def test_both_native_stopping_components_enter_once(choice: Any, fraction: str) -> None:
    result = table(choice, electronic_fraction=fraction)
    assert result.total_stopping_mev_cm2_g == (D(2), D(2), D(2))
    assert result.anchored_range_g_cm2 == (D(0), D('.5'), D('1.5'))


@pytest.mark.parametrize("choice", CHOICES)
def test_no_extrapolation_or_zero_filling(choice: Any) -> None:
    result = table(choice)
    for energy in (D('.999999'), D('4.00001')):
        with pytest.raises(ValueError, match="outside observed"):
            result.range_at_energy(energy)
        with pytest.raises(ValueError, match="outside observed"):
            result.residual_energy(energy, D(0))
    with pytest.raises(ValueError, match="outside observed"):
        result.energy_at_range(D('1.5000001'))
    assert result.residual_energy(D(2), D('1')) is None


@pytest.mark.parametrize("method", ["range_at_energy", "energy_at_range", "residual_energy"])
@pytest.mark.parametrize("bad", [True, 1, 1.0, "1", None, D('NaN'), D('Infinity'), D('-1'),
                                D('1e1001'), D('1.' + '1234567890'*9)])
def test_invalid_scalar_inputs_fail(method: str, bad: Any) -> None:
    result = table("linear_stopping")
    with pytest.raises(ValueError):
        if method == "residual_energy":
            result.residual_energy(D(2), bad)
        else:
            getattr(result, method)(bad)


@pytest.mark.parametrize("choice", [None, True, "linear", "", 1])
def test_interpolation_must_be_explicit_and_supported(choice: Any) -> None:
    with pytest.raises(ValueError, match="Explicit"):
        table(choice)


@pytest.mark.parametrize("bad", [None, {}, (), True])
def test_native_type_is_required(bad: Any) -> None:
    with pytest.raises(ValueError, match="SrimRangeOutput"):
        AnchoredSrimRangeTable.from_native(bad, interpolation="linear_stopping")


def test_invalid_material_mass_and_zero_total_stopping_rejected() -> None:
    for changes, reason in (({"target": 14}, "Al"), ({"z": 26, "mass": "12"}, "mass"),
                            ({"stopping": ("0", "0", "0")}, "positive")):
        with pytest.raises(ValueError, match=reason):
            table("linear_stopping", **changes)


@pytest.mark.parametrize("choice", CHOICES)
def test_derived_arrays_and_flags_cannot_be_constructor_inputs(choice: Any) -> None:
    result = table(choice)
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(result, anchored_range_g_cm2=(D(0), D(1), D(2)))
    with pytest.raises(FrozenInstanceError):
        result.ion_mass_u = D(13)  # type: ignore[misc]
    with pytest.raises(TypeError):
        AnchoredSrimRangeTable(native(), choice, normative_export_allowed=True)  # type: ignore[call-arg]
    data = asdict(result)
    for key in ("normative_export_allowed", "physical_accuracy_verified",
                "native_source_authenticity_verified", "absolute_range_origin_established",
                "isotope_selection_policy_established", "production_tables_replaced"):
        assert data[key] is False
    assert result.native.range_kind == "projected"
    assert result.range_kind == "derived_anchored_csda_path"
    assert result.anchored_range_g_cm2[0] == 0 and result.energy_total_mev[0] > 0


@pytest.mark.parametrize("choice", CHOICES)
def test_private_context_covers_constructor_and_all_methods(choice: Any) -> None:
    raw = native(stopping=("3", "5", "9"))
    expected = AnchoredSrimRangeTable.from_native(raw, interpolation=choice)
    expected_values = (expected.range_at_energy(D(3)), expected.energy_at_range(D('.1')),
                       expected.residual_energy(D(3), D('.01')))
    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_DOWN
        context.Emax = 9
        context.Emin = -9
        for signal in context.traps:
            context.traps[signal] = True
        actual = AnchoredSrimRangeTable.from_native(raw, interpolation=choice)
        values = (actual.range_at_energy(D(3)), actual.energy_at_range(D('.1')),
                  actual.residual_energy(D(3), D('.01')))
        assert actual == expected
        assert values == expected_values


@pytest.mark.parametrize("choice", CHOICES)
def test_unrepresentable_cumulative_intervals_rejected(choice: Any) -> None:
    with pytest.raises(ValueError, match="collapse"):
        table(choice, energies=("1", "2", "3", "4"),
              stopping=("1e-70", "1e-70", "1e70", "1e70"), electronic_fraction="1")


@pytest.mark.parametrize("choice", CHOICES)
def test_tiny_interior_range_must_not_become_endpoint_energy(choice: Any) -> None:
    result = table(choice)
    with pytest.raises(ValueError, match="represented"):
        result.energy_at_range(D('1e-1000'))


@pytest.mark.parametrize("choice", CHOICES)
def test_interior_energy_must_not_become_endpoint_range(choice: Any) -> None:
    # Full node increments remain distinct, but the tiny final portion does not
    # survive 80-digit cumulative-range representation. Do not return a knot.
    result = table(choice, energies=("1", "2", "3"),
                   stopping=("1e-60", "1e-60", "1e20"))
    with pytest.raises(ValueError, match="represented"):
        result.range_at_energy(D('2.999'))


@pytest.mark.parametrize("choice", CHOICES)
def test_positive_subresolution_shield_cannot_disappear(choice: Any) -> None:
    result = table(choice)
    with pytest.raises(ValueError, match="resolution"):
        result.residual_energy(D(4), D('1e-1000'))
    # R changes at its own precision while the energy change rounds to its knot.
    with pytest.raises(ValueError, match="represented|resolution"):
        result.residual_energy(D(2), D('1e-80'))
