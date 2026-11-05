"""Bounded diagnostic wiring tests, not normative orbit/spectrum acceptance."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from radar.core.products import SpectrumProduct
from radar.core.types import AssessmentMode, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.gcr.penetration import gcr_product_mass_to_charge
from radar.geomagnetic.rigidity import RigidityGrid
from radar.physics.rigidity import ion_kinetic_energy_per_nucleon_to_rigidity_gv


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/validate_heo_orbit_sampling.py"
    spec = importlib.util.spec_from_file_location("heo_orbit_sampling_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sources(runner: ModuleType) -> tuple[SpectrumProduct, ...]:
    products, _ = runner._source_products()
    return tuple(products)


def test_fixed_matrix_separates_density_windows_mlt_and_initial_raan(runner: ModuleType) -> None:
    all_specs = runner._specifications("all")
    assert len(all_specs) == 18
    assert len({spec.name for spec in all_specs}) == 18
    assert len(runner._specifications("heo")) == 13
    assert len(runner._specifications("critical")) == 3
    assert len(runner._specifications("smoke")) == 2
    primary = {spec.name: spec for spec in runner._specifications("heo")}
    for steps in (60, 120, 240, 480):
        spec = primary[f"heo_16d_n{steps}_m24"]
        assert (spec.min_days, spec.min_orbits, spec.mlt_sample_count) == (16.0, 10, 24)
    assert primary["heo_32d_n480_m24"].min_days == 32.0
    assert {spec.raan_deg for spec in all_specs} == {0.0, 90.0, 180.0, 270.0}
    inclinations = [spec.orbit.inclination_deg for spec in runner._specifications("critical")]
    assert inclinations[0] < runner._CRITICAL_INCLINATION_DEG < inclinations[2]
    assert inclinations[1] == runner._CRITICAL_INCLINATION_DEG
    with pytest.raises(ValueError, match="Unknown"):
        runner._specifications("unbounded")


def test_preflight_has_no_legacy_allocation_and_enforces_explicit_budgets(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = runner._preflight(runner._specifications("all"))
    assert report["variant_count"] == 18
    assert 0 < report["total_orbit_mlt_states"] < report["orbit_mlt_state_limit"]
    assert report["legacy_orbit_traces_allocated"] == 0
    assert max(row["samples"] for row in report["variants"]) < 40000
    monkeypatch.setattr(runner, "_MAX_ORBIT_MLT_STATES", 1)
    with pytest.raises(ValueError, match="state budget exceeded"):
        runner._preflight(runner._specifications("heo"))
    monkeypatch.setattr(runner, "_MAX_SAMPLES", 1)
    with pytest.raises(ValueError, match="orbit budget exceeded"):
        runner._preflight(runner._specifications("heo"))


def test_window_extension_is_separate_fixed_density_bounded_scope(runner: ModuleType) -> None:
    specs = runner._specifications("window_extension")
    assert len(specs) == 10
    assert {spec.min_days for spec in specs} == {16.0, 32.0, 64.0, 128.0, 256.0}
    assert {spec.raan_deg for spec in specs} == {0.0, 90.0}
    assert all(spec.steps_per_orbit == 240 and spec.mlt_sample_count == 24 for spec in specs)
    report = runner._preflight(specs)
    assert 11000000 < report["total_orbit_mlt_states"] < 12000000
    assert 120000 < max(row["samples"] for row in report["variants"]) < 125000
    assert report["legacy_orbit_traces_allocated"] == 0
    assert len(runner._specifications("all")) == 18


def test_actual_source_selects_twelve_differential_products_with_original_modes_and_reasons(
    runner: ModuleType, sources: tuple[SpectrumProduct, ...],
) -> None:
    assert len(sources) == 12
    assert {runner._product_key(product) for product in sources} == (
        {f"gcr:{symbol}:{kind}" for symbol in ("H", "He", "Fe")
         for kind in ("mission_fluence", "maximum_flux")}
        | {f"sep:{symbol}:{kind}" for symbol in ("p", "He", "Fe")
           for kind in ("mission_fluence", "peak_flux")}
    )
    for product in sources:
        spectrum = product.spectrum
        assert spectrum.quantity in (
            SpectrumQuantity.DIFFERENTIAL_FLUENCE, SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        )
        if spectrum.source is RadiationSource.GCR:
            assert any("ion roster incomplete" in reason for reason in spectrum.incomplete_reasons)
        if product.kind is RadiationProductKind.MISSION_FLUENCE:
            assert spectrum.assessment_mode is AssessmentMode.CUMULATIVE
            assert spectrum.mission_duration_s == pytest.approx(5.0 * 365.25 * 86400.0)
        else:
            assert spectrum.assessment_mode is AssessmentMode.PEAK
            assert spectrum.mission_duration_s is None
    helium = [p for p in sources if p.spectrum.species and p.spectrum.species.symbol == "He"
              and p.spectrum.source is RadiationSource.GCR]
    assert all(any("b_c" in reason for reason in p.spectrum.incomplete_reasons) for p in helium)


def test_rigidity_grid_contains_exact_source_nodes_and_uses_production_mass_ratio(
    runner: ModuleType, sources: tuple[SpectrumProduct, ...],
) -> None:
    grid = runner._rigidity_grid(sources)
    for product in sources:
        values = runner._source_rigidities(product)
        assert set(values).issubset(set(grid.values_gv))
        assert values[0] == ion_kinetic_energy_per_nucleon_to_rigidity_gv(
            product.spectrum.x[0], runner._mass_to_charge(product),
        )
    iron = next(p for p in sources if p.spectrum.species and p.spectrum.species.symbol == "Fe"
                and p.spectrum.source is RadiationSource.GCR)
    assert gcr_product_mass_to_charge(iron) == pytest.approx(55.8 / 26.0)
    sep_iron = next(p for p in sources if p.spectrum.species and p.spectrum.species.symbol == "Fe"
                    and p.spectrum.source is RadiationSource.SEP)
    # SEP production uses B.9 A/Q, not rounded A divided by atomic number.
    assert runner._mass_to_charge(sep_iron) == pytest.approx(4.12)


def test_source_scenario_is_actual_configured_2026_mission_not_fixed_wolf_snapshot(
    runner: ModuleType,
) -> None:
    products, scenario = runner._source_products()
    assert len(products) == 12
    mission = scenario["calculation_config"]["mission"]
    assert (mission["launch_year"], mission["lifetime_years"]) == (2026, 5)
    assert scenario["sep_exceedance_probability"] == 0.1
    assert scenario["sep_full_source_products_calculated"] > 6
    assert scenario["source_products_evaluated_once"] is True
    assert scenario["orbit_window_changes_do_not_recompute_source_time_products"] is True


def test_finite_integral_has_no_tail_or_hidden_time_factor(runner: ModuleType) -> None:
    assert runner._finite_trapezoid((1.0, 2.0, 4.0), (3.0, 3.0, 3.0)) == 9.0
    assert runner._finite_trapezoid((1.0, 2.0), (0.0, 0.0)) == 0.0
    for x, y in (((1.0,), (1.0,)), ((2.0, 1.0), (1.0, 1.0)),
                 ((1.0, 2.0), (float("nan"), 1.0)), ((1.0, 2.0), (-1.0, 1.0))):
        with pytest.raises(ValueError):
            runner._finite_trapezoid(x, y)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_zero_reference_relative_change_is_unknown_not_zero(runner: ModuleType, value: float) -> None:
    result = runner._relative_difference(value, 0.0)
    assert result["absolute_change"] == value
    assert result["relative_change"] is None
    assert result["relative_change_status"] == "zero_reference"
    assert result["below_one_percent_for_this_fixed_metric_only"] is None


def test_relative_change_uses_declared_reference_and_strict_one_percent(runner: ModuleType) -> None:
    result = runner._relative_difference(101.0, 100.0)
    assert result["relative_change"] == 0.01
    assert result["below_one_percent_for_this_fixed_metric_only"] is False
    with pytest.raises(ValueError):
        runner._relative_difference(-1.0, 0.0)
    with pytest.raises(ValueError, match="overflowed"):
        runner._relative_difference(1e300, 1e-300)


def test_legacy_window_is_only_analytical_and_handles_zero_drift(runner: ModuleType) -> None:
    trace = SimpleNamespace(perigee_precession_rad_s=2.138545554e-11,
                            eccentricity=0.74, orbital_period_s=42861.90056655)
    estimate = runner._legacy_window_estimate(trace)
    assert estimate["role"] == "analytical_counterfactual_not_sampled"
    assert estimate["legacy_window_s"] / (365.25 * 86400.0) == pytest.approx(23275.4, rel=1e-5)
    assert estimate["legacy_step_over_orbital_period"] > 40.0
    assert estimate["legacy_sample_cap_would_coarsen"] is True
    trace.perigee_precession_rad_s = 0.0
    estimate = runner._legacy_window_estimate(trace)
    assert estimate["perigee_drift_period_s"] is None
    assert estimate["legacy_window_s"] == 16.0 * 86400.0


def test_variant_preserves_peak_and_fluence_metadata_without_time_rescaling(
    runner: ModuleType, sources: tuple[SpectrumProduct, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = runner.VariantSpec("tiny", runner._orbit(500.0, 39700.0, 63.4),
                              min_days=1e-9, min_orbits=1, steps_per_orbit=4, raan_deg=90.0)
    grid = runner._rigidity_grid(sources)

    def constant_operator(**kwargs: Any) -> Any:
        assert kwargs["rigidity_grid"] is grid
        return SimpleNamespace(values=(0.25,) * len(grid.values_gv), model_version="test",
                               parameters=(), incomplete_reasons=("synthetic diagnostic",))

    monkeypatch.setattr(runner, "build_ost_penetration_function_from_samples", constant_operator)
    result = runner._variant(spec, grid, sources)
    assert result["sample_count"] == 4
    assert result["step_over_orbital_period"] == pytest.approx(0.25)
    for product in sources:
        metric = result["source_weighted_integrals"][runner._product_key(product)]
        assert metric["value"] == pytest.approx(metric["source_integral_same_grid"] * 0.25)
        assert metric["time_multiplier"] == "none"
        assert metric["mission_duration_s"] == product.spectrum.mission_duration_s
        assert metric["source_incomplete_reasons"] == product.spectrum.incomplete_reasons
        assert metric["source_differential_unit"] == product.spectrum.y_unit.value
        assert product.spectrum.angular_basis is not None
        assert metric["source_angular_basis"] == product.spectrum.angular_basis.value
        assert metric["unit"] == ("cm^-2" if product.kind is RadiationProductKind.MISSION_FLUENCE
                                  else "cm^-2 s^-1")
    longer_metadata = tuple(replace(
        p, spectrum=replace(p.spectrum, mission_duration_s=2.0 * p.spectrum.mission_duration_s)
    ) if p.spectrum.mission_duration_s else p for p in sources)
    longer = runner._variant(spec, grid, longer_metadata)
    assert {key: row["value"] for key, row in longer["source_weighted_integrals"].items()} == {
        key: row["value"] for key, row in result["source_weighted_integrals"].items()
    }


def _fake_variant(spec: Any, grid: RigidityGrid, products: tuple[SpectrumProduct, ...]) -> Any:
    return {
        "name": spec.name, "penetration": (0.5,) * len(grid.values_gv),
        "requested_window_s": spec.min_days * 86400.0,
        "window_s": spec.min_days * 86400.0 + 1.0 / spec.steps_per_orbit,
        "source_weighted_integrals": {f"source{n}": {"value": 1.0} for n in range(len(products))},
    }


def test_report_pins_actual_sources_and_retains_explicit_non_acceptance(
    runner: ModuleType, sources: tuple[SpectrumProduct, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_source_products", lambda: (sources, {"fixture": "actual_sources"}))
    monkeypatch.setattr(runner, "_variant", _fake_variant)
    report = runner.build_report("all")
    assert len(report["variants"]) == 18
    assert len(report["source_products"]) == 12
    assert report["model_sha256_before"] == report["model_sha256_after"]
    assert report["input_sha256_before"] == report["input_sha256_after"]
    assert len(report["input_sha256_before"]) == 7
    assert len(report["source_products_sha256"]) == 64
    for key in ("normative_release_eligible", "stage4_spectral_convergence_established",
                "source_roster_complete", "parameters_fitted"):
        assert report[key] is False
    axes = {comparison["axis"] for comparison in report["comparisons"]}
    assert len(axes) == 5
    assert "initial_raan_sensitivity_not_grid_convergence" in axes
    assert all(not row["stage4_acceptance_claim"] for row in report["comparisons"])
    density_pairs = [row for row in report["comparisons"] if row["axis"].startswith("orbital_density")]
    assert len(density_pairs) == 3
    for row in density_pairs:
        assert row["axis"] == "orbital_density_at_fixed_requested_window_and_mlt"
        assert row["left_requested_window_s"] == row["right_requested_window_s"]
        assert row["actual_window_equal"] is False
        assert row["actual_window_change_s"] != 0.0
        assert row["density_comparison_includes_rounding_window_change"] is True
    for row in report["comparisons"]:
        if row["axis"].startswith("mlt_density"):
            assert row["actual_window_equal"] is True
            assert row["density_comparison_includes_rounding_window_change"] is False


@pytest.mark.parametrize("hash_function", ["_model_hashes", "_input_hashes"])
def test_changed_implementation_or_data_prevents_report(
    runner: ModuleType, sources: tuple[SpectrumProduct, ...], monkeypatch: pytest.MonkeyPatch,
    hash_function: str,
) -> None:
    calls = 0

    def changed() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"synthetic": str(calls)}

    monkeypatch.setattr(runner, "_source_products", lambda: (sources, {}))
    monkeypatch.setattr(runner, "_variant", _fake_variant)
    monkeypatch.setattr(runner, hash_function, changed)
    with pytest.raises(ValueError, match="changed during"):
        runner.build_report("smoke")


def test_extension_comparisons_separate_window_and_raan_phase_coverage(
    runner: ModuleType, sources: tuple[SpectrumProduct, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_source_products", lambda: (sources, {}))
    monkeypatch.setattr(runner, "_variant", _fake_variant)
    report = runner.build_report("window_extension")
    assert report["scope"] == "window_extension"
    assert len(report["variants"]) == 10
    assert len(report["comparisons"]) == 13
    window_pairs = [row for row in report["comparisons"]
                    if row["axis"] == "phase_coverage_window_extension_not_mission_qualification"]
    raan_pairs = [row for row in report["comparisons"] if row["axis"] ==
                  "phase_coverage_initial_raan_sensitivity_not_mission_qualification"]
    assert len(window_pairs) == 8 and len(raan_pairs) == 5
    for row in window_pairs:
        assert row["right_requested_window_s"] == 2.0 * row["left_requested_window_s"]
        assert row["actual_window_equal"] is False
        assert row["left"].split("raan")[-1] == row["right"].split("raan")[-1]
    for row in raan_pairs:
        assert row["right_requested_window_s"] == row["left_requested_window_s"]
        assert row["actual_window_equal"] is True
        assert row["left"].endswith("raan0") and row["right"].endswith("raan90")
    for row in report["comparisons"]:
        assert row["stage4_acceptance_claim"] is False
        assert row["density_comparison_includes_rounding_window_change"] is False
    assert report["normative_release_eligible"] is False
    assert report["stage4_spectral_convergence_established"] is False


def test_output_is_exclusive_and_nonfinite_json_never_creates_file(
    runner: ModuleType, tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    with pytest.raises(ValueError):
        runner.write_report_exclusively({"bad": float("nan")}, output)
    assert not output.exists()
    runner.write_report_exclusively({"eligible": False}, output)
    before = output.read_bytes()
    assert json.loads(before)["eligible"] is False
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively({"eligible": True}, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize("scope", ["smoke", "window_extension"])
def test_cli_requires_new_output_and_prints_preflight_before_build(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], scope: str,
) -> None:
    with pytest.raises(SystemExit) as missing:
        runner.main([])
    assert missing.value.code == 2
    output = tmp_path / "report.json"

    def build(requested_scope: str) -> dict[str, Any]:
        assert "preflight" in capsys.readouterr().out
        assert requested_scope == scope
        return {"normative_release_eligible": False}

    monkeypatch.setattr(runner, "build_report", build)
    assert runner.main(["--case", scope, "--output", str(output)]) == 0
    with pytest.raises(SystemExit) as existing:
        runner.main(["--case", scope, "--output", str(output)])
    assert existing.value.code == 2
