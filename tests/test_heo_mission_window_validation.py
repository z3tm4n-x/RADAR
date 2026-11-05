"""Finite-window diagnostic tests; no normative mission qualification."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from radar.core.products import SpectrumProduct
from radar.core.types import RadiationProductKind
from radar.geomagnetic.ost_orbit import sample_ost_orbit
from radar.geomagnetic.ost_penetration import build_ost_penetration_function_from_samples
from radar.geomagnetic.rigidity import RigidityGrid


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/validate_heo_mission_window.py"
    spec = importlib.util.spec_from_file_location("heo_mission_validation_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sources(runner: ModuleType) -> tuple[SpectrumProduct, ...]:
    products, _ = runner._helper._source_products()
    return tuple(products)


def test_exact_production_period_arithmetic(runner: ModuleType) -> None:
    for spec in runner._specifications("n60"):
        trace = sample_ost_orbit(spec.orbit, min_days=1e-9, min_orbits=1,
                                steps_per_orbit=spec.steps_per_orbit)
        assert runner._orbital_period(spec.orbit) == trace.orbital_period_s
        assert runner._orbital_period(spec.orbit) / spec.steps_per_orbit == trace.step_s


@pytest.mark.parametrize("cells", [1, 7, 60, 420, 220800])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_aligned_and_nextafter_boundaries_have_only_positive_intersections(
    runner: ModuleType, cells: int, direction: int,
) -> None:
    spec = runner._specifications("n60")[0]
    step = runner._orbital_period(spec.orbit) / 60
    boundary = cells * step
    duration = (math.nextafter(boundary, math.inf if direction > 0 else -math.inf)
                if direction else boundary)
    expected = cells + int(direction > 0)
    assert runner._cell_count(duration, step) == expected
    plan = runner._block_plan(duration, step, 60, 32)
    assert sum(block.used_cell_count for block in plan) == expected
    assert plan[0].start_time_s == 0.0
    assert plan[-1].end_time_s == duration
    assert all(0.0 < block.last_cell_fraction <= 1.0 for block in plan)
    assert all(block.represented_duration_s > 0.0 for block in plan)
    assert math.fsum(block.duration_fraction for block in plan) == pytest.approx(1.0, abs=4e-16)
    assert abs(math.fsum(block.represented_duration_s for block in plan) - duration) <= (
        4.0 * math.ulp(duration)
    )
    for left, right in zip(plan, plan[1:]):
        assert left.end_time_s == right.start_time_s
        assert left.start_sample_index + left.used_cell_count == right.start_sample_index
    last_weights = runner._cell_weights(plan[-1], step)
    assert all(weight > 0.0 for weight in last_weights)
    if direction == 0:
        assert plan[-1].last_cell_fraction == 1.0
    else:
        assert plan[-1].last_cell_fraction < 1.0


def test_420_aligned_cells_do_not_create_empty_eighth_block(runner: ModuleType) -> None:
    spec = runner._specifications("n60")[0]
    step = runner._orbital_period(spec.orbit) / 60
    assert math.ceil(420 * step / step) == 421  # The actual troublesome float arithmetic.
    plan = runner._block_plan(420 * step, step, 60, 1)
    assert len(plan) == 7
    assert all(block.used_cell_count == block.generated_cell_count == 60 for block in plan)
    assert plan[-1].last_cell_fraction == 1.0


def test_partial_last_cell_does_not_extend_mission_or_multiply_source_time(
    runner: ModuleType,
) -> None:
    plan = runner._block_plan(25.25, 2.0, 4, 2)
    assert len(plan) == 2
    assert [(block.start_sample_index, block.used_cell_count) for block in plan] == [(0, 8), (8, 5)]
    last = plan[-1]
    assert last.generated_cell_count == 8
    assert last.end_time_s == 25.25
    assert last.last_cell_duration_s == 1.25
    assert last.last_cell_fraction == 0.625
    assert runner._cell_weights(last, 2.0) == (2.0, 2.0, 2.0, 2.0, 1.25)
    accumulator = runner._DurationAccumulator(2)
    accumulator.add((0.2, 0.7), plan[0].represented_duration_s)
    accumulator.add((0.2, 0.7), plan[1].represented_duration_s)
    assert accumulator.values(25.25) == pytest.approx((0.2, 0.7), abs=2e-16)


@pytest.mark.parametrize("case,states", [
    ("n60", 10602192), ("n120", 21204384), ("n240", 42408720),
])
def test_mission_preflight_is_bounded_and_uses_both_relative_node_orientations(
    runner: ModuleType, case: str, states: int,
) -> None:
    specs = runner._specifications(case)
    assert {spec.raan_deg for spec in specs} == {0.0, 90.0}
    assert all(spec.duration_s == 157788000.0 for spec in specs)
    preflight = runner._preflight(specs, 32, 50000000)
    assert preflight["variant_count"] == 2
    assert preflight["total_orbit_mlt_states"] == states
    assert preflight["silent_coarsening"] is False
    assert all(row["block_count"] == 116 for row in preflight["variants"])
    assert all(row["max_generated_cells_per_block"] <= 7680 for row in preflight["variants"])


def test_budget_failure_precedes_source_loading_or_sampling(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Expensive work must not start before the budget check")

    monkeypatch.setattr(runner._helper, "_source_products", forbidden)
    monkeypatch.setattr(runner, "sample_ost_orbit", forbidden)
    with pytest.raises(ValueError, match="state budget"):
        runner.build_report("n60", max_states=1)
    with pytest.raises(ValueError, match="Per-block sample budget"):
        runner.build_report("n60", block_orbits=7000)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            runner.build_report("n60", block_orbits=bad)
    with pytest.raises(ValueError, match="Unknown"):
        runner.build_report("all")
    invalid_spec = replace(runner._specifications("smoke")[0], steps_per_orbit=3)
    with pytest.raises(ValueError, match="at least four"):
        runner._preflight((invalid_spec,), 2, 50000000)


@pytest.mark.parametrize("duration,step", [
    (0.0, 1.0), (-1.0, 1.0), (math.inf, 1.0), (1.0, math.nan),
    (True, 1.0), (1.0, 0.0), (1e300, 1e-300),
])
def test_invalid_or_unresolvable_time_grid_rejected(
    runner: ModuleType, duration: float, step: float,
) -> None:
    with pytest.raises(ValueError):
        runner._cell_count(duration, step)


@pytest.mark.parametrize("cells", [7.0, 137.4, 420.0])
def test_chunked_operator_matches_monolithic_weighted_operator_without_phase_restart(
    runner: ModuleType, cells: float,
) -> None:
    original = runner._specifications("n60")[0]
    step = runner._orbital_period(original.orbit) / original.steps_per_orbit
    spec = replace(original, duration_s=cells * step, mlt_sample_count=4)
    grid = RigidityGrid((0.01, 0.1, 1.0, 3.0, 10.0, 100.0))
    result = runner._variant(spec, 1, grid, ())
    plan = runner._block_plan(spec.duration_s, step, 60, 1)
    count = runner._cell_count(spec.duration_s, step)
    monolithic = sample_ost_orbit(spec.orbit, min_days=1e-9, min_orbits=(count + 59) // 60,
                                 steps_per_orbit=60)
    weights = tuple(weight for block in plan for weight in runner._cell_weights(block, step))
    operator = build_ost_penetration_function_from_samples(
        samples=monolithic.samples[:count], rigidity_grid=grid, kp=3,
        apply_disturbance=True, mlt_sample_count=4, sample_weights=weights,
    )
    assert result["penetration"] == pytest.approx(operator.values, abs=7e-16)
    assert result["used_cell_count"] == count
    assert result["coverage_end_time_s"] == spec.duration_s
    assert result["duration_fraction_sum"] == pytest.approx(1.0, abs=3e-16)
    assert result["exact_configured_mission_temporal_coverage"] is False
    for record, block in zip(result["blocks"], plan, strict=True):
        assert record["first_node_time_s"] == monolithic.samples[block.start_sample_index].time_s
        assert record["last_node_time_s"] == monolithic.samples[
            block.start_sample_index + block.used_cell_count - 1
        ].time_s
        assert record["operator_incomplete_reasons"]
    assert all(comparison["contiguous_without_overlap"]
               for comparison in result["consecutive_block_comparisons"])
    assert all(not comparison["physical_source_time_series"]
               for comparison in result["consecutive_block_comparisons"])


def test_compensated_accumulator_preserves_unity_and_rejects_duration_gap(runner: ModuleType) -> None:
    accumulator = runner._DurationAccumulator(3)
    durations = (1e8, 1.0, 1e-8, 3.125)
    for duration in durations:
        accumulator.add((0.0, 0.5, 1.0), duration)
    assert accumulator.values(math.fsum(durations)) == (0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="do not close"):
        accumulator.values(math.fsum(durations) + 1.0)
    with pytest.raises(ValueError):
        accumulator.add((0.5,), 1.0)
    with pytest.raises(ValueError):
        accumulator.add((0.5, 0.5, math.nan), 1.0)
    for invalid in (0.0, -1.0, math.inf, math.nan, True):
        with pytest.raises(ValueError, match="positive and finite"):
            accumulator.values(invalid)
    with pytest.raises(ValueError, match="positive integer"):
        runner._DurationAccumulator(0)


def test_aggregate_eight_ulp_guard_does_not_repair_real_bound_or_monotonicity_errors(
    runner: ModuleType,
) -> None:
    accumulator = runner._DurationAccumulator(2)
    accumulator.add((0.0, 1.0), 1.0)
    accumulator.total[1] = 1.0 + 4.0 * math.ulp(1.0)
    assert accumulator.values(1.0) == (0.0, 1.0)
    accumulator.total[1] = 1.0 + 9.0 * math.ulp(1.0)
    with pytest.raises(ArithmeticError, match="physical bounds"):
        accumulator.values(1.0)
    accumulator.total[:] = [0.9, 0.1]
    with pytest.raises(ArithmeticError, match="nondecreasing"):
        accumulator.values(1.0)
    accumulator.total[:] = [0.0, math.nan]
    with pytest.raises(ArithmeticError, match="physical bounds"):
        accumulator.values(1.0)


def test_all_source_durations_are_checked_not_only_first_source(
    runner: ModuleType, sources: tuple[SpectrumProduct, ...],
) -> None:
    assert runner._validate_source_duration(sources) == 157788000.0
    for index, product in enumerate(sources):
        if product.kind is RadiationProductKind.MISSION_FLUENCE:
            changed = replace(product, spectrum=replace(
                product.spectrum, mission_duration_s=157788001.0,
            ))
            products = (*sources[:index], changed, *sources[index + 1:])
            with pytest.raises(ValueError, match="durations must match"):
                runner._validate_source_duration(products)
    with pytest.raises(ValueError, match="twelve unique"):
        runner._validate_source_duration(sources[:-1])


def test_fixed_source_metrics_preserve_units_reasons_and_have_no_time_multiplier(
    runner: ModuleType, sources: tuple[SpectrumProduct, ...],
) -> None:
    grid = runner._helper._rigidity_grid(sources)
    prepared = runner._prepare_products(sources, grid)
    unit_metrics = runner._integrals((1.0,) * len(grid.values_gv), prepared)
    half_metrics = runner._integrals((0.5,) * len(grid.values_gv), prepared)
    assert len(unit_metrics) == 12
    for product in sources:
        key = runner._helper._product_key(product)
        metric = unit_metrics[key]
        assert metric["value"] == metric["source_integral_same_grid"]
        assert half_metrics[key]["value"] == pytest.approx(0.5 * metric["value"], rel=1e-15)
        assert metric["time_multiplier"] == "none"
        assert metric["metric_semantics"] == runner._METRIC_SEMANTICS
        assert metric["source_incomplete_reasons"] == product.spectrum.incomplete_reasons
        assert metric["source_differential_unit"] == product.spectrum.y_unit.value
        if product.kind is RadiationProductKind.MISSION_FLUENCE:
            assert metric["unit"] == "cm^-2"
            assert metric["mission_duration_s"] == 157788000.0
        else:
            assert metric["unit"] == "cm^-2 s^-1"
            assert metric["mission_duration_s"] is None
    assert unit_metrics["sep:Fe:mission_fluence"]["rigidity_mass_to_charge_used"] == 4.12


def test_comparison_unknown_zero_reference_is_not_fabricated_convergence(runner: ModuleType) -> None:
    left = {"name": "left", "source_weighted_integrals": {"p": {"value": 0.0}}}
    right = {"name": "right", "source_weighted_integrals": {"p": {"value": 0.0}}}
    comparison = runner._comparison(left, right, (0.0,), (0.0,), "diagnostic")
    assert comparison["source_integral_changes"]["p"]["relative_change"] is None
    assert comparison["source_integral_changes"]["p"][
        "below_one_percent_for_this_fixed_metric_only"
    ] is None
    assert comparison["normative_acceptance_established"] is False


def test_model_hashes_include_both_runners_and_production_time_origin(runner: ModuleType) -> None:
    hashes = runner._model_hashes()
    assert "scripts/validate_heo_mission_window.py" in hashes
    assert "scripts/validate_heo_orbit_sampling.py" in hashes
    assert "src/radar/geomagnetic/ost_orbit.py" in hashes
    assert all(len(digest) == 64 for digest in hashes.values())


def test_report_discards_a_run_if_implementation_changes(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashes = iter(({"model": "before"}, {"model": "after"}))
    monkeypatch.setattr(runner, "_model_hashes", lambda: next(hashes))
    monkeypatch.setattr(runner, "_variant", lambda *args: {})
    with pytest.raises(ValueError, match="changed during diagnostics"):
        runner.build_report("smoke")


def test_real_smoke_report_preserves_incomplete_status_and_partial_final_cell(
    runner: ModuleType,
) -> None:
    progress: list[dict[str, Any]] = []
    report = runner.build_report("smoke", progress=progress.append)
    assert len(report["source_products"]) == 12
    assert len(report["rigidity_gv"]) == 1047
    assert report["model_sha256_before"] == report["model_sha256_after"]
    assert report["input_sha256_before"] == report["input_sha256_after"]
    assert report["normative_release_eligible"] is False
    assert report["mission_radiation_history_qualified"] is False
    assert report["stage4_spectral_convergence_established"] is False
    assert report["source_roster_complete"] is False
    assert len(progress) == 4  # first and last of two blocks for each RAAN
    for variant in report["variants"]:
        assert variant["block_count"] == 2
        assert variant["used_cell_count"] == 37
        assert variant["generated_cell_count"] == 48
        assert variant["global_phases_preserved"] is True
        assert variant["exact_configured_mission_temporal_coverage"] is False
        assert variant["mission_radiation_history_qualified"] is False
        assert len(variant["source_weighted_integrals"]) == 12
        assert len(variant["consecutive_block_comparisons"]) == 1
        final = variant["blocks"][-1]
        assert final["last_cell_fraction"] == pytest.approx(0.8, abs=1e-14)
        assert final["discarded_generated_cells"] == 11
        assert all(len(block["source_weighted_integrals"]) == 12 for block in variant["blocks"])
    json.dumps(report, allow_nan=False)


def test_output_exclusive_and_nonfinite_rejected_before_file_creation(
    runner: ModuleType, tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    runner.write_report_exclusively({"valid": True}, path)
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively({"valid": False}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"valid": True}
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        runner.write_report_exclusively({"invalid": math.nan}, invalid)
    assert not invalid.exists()


def test_cli_refuses_existing_output_before_work(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path / "existing.json"
    runner.write_report_exclusively({"keep": True}, path)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Existing output must be refused before calculations")

    monkeypatch.setattr(runner, "build_report", forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main(["--case", "smoke", "--output", str(path)])
    assert error.value.code == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"keep": True}
