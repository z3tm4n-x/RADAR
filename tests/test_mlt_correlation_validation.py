"""Local-time counterfactual wiring and arithmetic, not physical qualification."""

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
from radar.geomagnetic.local_time import LocalTimeHistory
from radar.geomagnetic.ost_orbit import OST_EARTH_ROTATION_RATE_RAD_S, sample_ost_orbit


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/validate_mlt_correlation.py"
    spec = importlib.util.spec_from_file_location("mlt_correlation_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sources(runner: ModuleType) -> tuple[SpectrumProduct, ...]:
    products, _ = runner._source._source_products()
    return tuple(products)


def _tiny_spec(runner: ModuleType) -> Any:
    return replace(runner._specifications("heo")[0], steps_per_orbit=8,
                   min_days=1e-9, min_orbits=1)


def test_matrix_separates_orbital_density_orientation_and_candidate_time_definitions(
    runner: ModuleType,
) -> None:
    specs = runner._specifications("all")
    assert len(specs) == 6
    assert len(runner._specifications("heo")) == 4
    assert len(runner._specifications("smoke")) == 2
    heo = runner._specifications("heo")
    assert {(spec.steps_per_orbit, spec.raan_deg) for spec in heo} == {
        (240, 0.0), (240, 90.0), (480, 0.0), (480, 90.0),
    }
    assert all(spec.min_days == 16.0 for spec in heo)
    smoke = runner._specifications("smoke")
    assert all(spec.min_orbits == 10 and spec.steps_per_orbit == 120 for spec in smoke)
    assert len(runner._OPERATORS) == 7
    assert set(runner._FAMILIES) == {"geographic_solar", "centered_dipole"}
    with pytest.raises(ValueError, match="Unknown"):
        runner._specifications("five_year_static_sun")


def test_full_matrix_is_preflight_bounded_below_five_million_states(runner: ModuleType) -> None:
    budget = runner._preflight(runner._specifications("all"), 5000000)
    assert budget["trace_count"] == 6
    assert budget["operator_count"] == 42
    assert budget["states_per_orbit_node"] == 77
    assert budget["total_orbit_local_time_states"] == 3761142
    assert budget["silent_coarsening"] is False
    assert max(row["sample_count"] for row in budget["variants"]) == 15482


def test_budget_rejection_occurs_before_sources_or_orbit_allocation(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Expensive work started before budget validation")

    monkeypatch.setattr(runner._source, "_source_products", forbidden)
    monkeypatch.setattr(runner, "sample_ost_orbit", forbidden)
    with pytest.raises(ValueError, match="state budget exceeded"):
        runner.build_report("all", max_states=1)
    huge = replace(runner._specifications("heo")[0], min_orbits=2000)
    with pytest.raises(ValueError, match="sample budget exceeded"):
        runner._preflight((huge,), 5000000)
    for invalid in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            runner._preflight(runner._specifications("smoke"), invalid)
    with pytest.raises(ValueError, match="at least four"):
        runner._preflight((replace(_tiny_spec(runner), steps_per_orbit=3),), 5000000)
    for invalid in (True, math.nan, 0.0):
        with pytest.raises(ValueError, match="Minimum days"):
            runner._preflight((replace(_tiny_spec(runner), min_days=invalid),), 5000000)
    for invalid in (True, math.inf):
        with pytest.raises(ValueError, match="node orientation"):
            runner._preflight((replace(_tiny_spec(runner), raan_deg=invalid),), 5000000)


def test_artificial_sun_rotation_uses_global_time_and_same_earth_rotation_sign(
    runner: ModuleType,
) -> None:
    spec = _tiny_spec(runner)
    trace = sample_ost_orbit(spec.orbit, min_days=1e-9, min_orbits=1, steps_per_orbit=8)
    quarter_rotation_s = (math.pi / 2.0) / OST_EARTH_ROTATION_RATE_RAD_S
    samples = (replace(trace.samples[0], time_s=0.0),
               replace(trace.samples[1], time_s=quarter_rotation_s))
    phase0 = runner._sun_vectors(samples, 0.0)
    phase90 = runner._sun_vectors(samples, 90.0)
    assert phase0[0] == pytest.approx((1.0, 0.0, 0.0), abs=2e-16)
    assert phase0[1] == pytest.approx((0.0, -1.0, 0.0), abs=2e-16)
    assert phase90[1] == pytest.approx((1.0, 0.0, 0.0), abs=2e-16)
    assert all(math.hypot(*vector) == pytest.approx(1.0, abs=2e-16)
               for vector in (*phase0, *phase90))
    for invalid in (True, math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            runner._sun_vectors(samples, invalid)


@pytest.mark.parametrize("phase", [0.0, 90.0])
def test_solar_candidate_matches_direct_geographic_longitude_hour_angle(
    runner: ModuleType, phase: float,
) -> None:
    spec = _tiny_spec(runner)
    trace = sample_ost_orbit(spec.orbit, min_days=1e-9, min_orbits=1, steps_per_orbit=8,
                             start_sample_index=24, raan_deg=90.0)
    history = runner._history(trace.samples, "geographic_solar", phase)
    assert isinstance(history, LocalTimeHistory)
    expected = tuple((12.0 + (sample.longitude_deg - phase
                              + math.degrees(OST_EARTH_ROTATION_RATE_RAD_S * sample.time_s))
                      / 15.0) % 24.0 for sample in trace.samples)
    assert history.values_hours == pytest.approx(expected, abs=2e-13)
    history.validate_samples(trace.samples)
    assert any("artificial phase counterfactual" in reason for reason in history.incomplete_reasons)
    assert dict(history.parameters)["candidate_family"] == "geographic_solar"
    assert "not_UTC" in dict(history.parameters)["time_reference"]


def test_time_candidates_have_distinct_models_and_explicit_axis_provenance(runner: ModuleType) -> None:
    spec = _tiny_spec(runner)
    trace = sample_ost_orbit(spec.orbit, min_days=1e-9, min_orbits=1, steps_per_orbit=8)
    solar = runner._history(trace.samples, "geographic_solar", 0.0)
    dipole = runner._history(trace.samples, "centered_dipole", 0.0)
    assert solar.model != dipole.model
    assert any(abs(a - b) > 0.01 for a, b in zip(solar.values_hours, dipole.values_hours, strict=True))
    assert math.hypot(*runner._north_axis()) == pytest.approx(1.0, abs=2e-16)
    assert dict(dipole.parameters)["axis_provenance"] == "borrowed_OST_D24_not_qualified_for_MLT"
    assert any("borrowed from OST D.24 only" in reason for reason in dipole.incomplete_reasons)
    with pytest.raises(ValueError, match="fingerprint"):
        solar.validate_samples(tuple(reversed(trace.samples)))
    with pytest.raises(ValueError, match="does not accept"):
        runner._history(trace.samples, "geographic_solar", 0.0, north_axis=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="Unknown"):
        runner._history(trace.samples, "automatic_normative", 0.0)


@pytest.mark.parametrize("family", ["geographic_solar", "centered_dipole"])
@pytest.mark.parametrize("phase", [0.0, 90.0])
def test_co_rotation_is_coordinate_only_and_preserves_local_hour(
    runner: ModuleType, family: str, phase: float,
) -> None:
    spec = _tiny_spec(runner)
    trace = sample_ost_orbit(spec.orbit, min_days=1e-9, min_orbits=1, steps_per_orbit=8)
    result = runner._co_rotation_sanity(trace.samples, family, phase)
    assert result["scope"] == "coordinate_geometry_only_not_cutoff_or_physical_acceptance"
    assert result["coordinate_invariance_check_passed"] is True
    assert result["max_circular_hour_difference"] < 1e-12
    assert result["original_subset_fingerprint"] != result["rotated_subset_fingerprint"]


def test_histogram_treats_undefined_circular_mean_as_unknown_not_zero(runner: ModuleType) -> None:
    result = runner._histogram(tuple(float(value) for value in range(24)))
    assert result["counts"] == (1,) * 24
    assert math.fsum(result["fractions"]) == 1.0
    assert result["occupied_bins"] == 24
    assert result["circular_mean_hours"] is None
    assert result["circular_mean_status"] == "unresolved_near_zero_resultant"
    same = runner._histogram((12.0, 12.0, 12.0))
    assert same["circular_mean_hours"] == 12.0
    assert same["circular_resultant_length"] == 1.0
    for invalid in ((), (24.0,), (-0.1,), (math.nan,), (math.inf,)):
        with pytest.raises(ValueError):
            runner._histogram(invalid)


def test_tiny_variant_passes_trace_bound_histories_and_preserves_source_units(
    runner: ModuleType, sources: tuple[SpectrumProduct, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_builder = runner.build_ost_penetration_function_from_samples
    histories = []

    def tracked_builder(**kwargs: Any) -> Any:
        assert "magnetic_local_times_hours" not in kwargs
        history = kwargs["local_time_history"]
        if history is not None:
            history.validate_samples(kwargs["samples"])
            histories.append(history)
        return real_builder(**kwargs)

    monkeypatch.setattr(runner, "build_ost_penetration_function_from_samples", tracked_builder)
    grid = runner._source._rigidity_grid(sources)
    prepared = runner._integration._prepare_products(sources, grid)
    result = runner._variant(_tiny_spec(runner), grid, prepared)
    assert len(result["operators"]) == 7
    assert len(histories) == len(result["histories"]) == 4
    for report in result["histories"].values():
        assert report["payload"]["sample_fingerprint"] == result["sample_fingerprint"]
        assert report["sha256"] == runner._canonical_sha(report["payload"])
    for operator in result["operators"]:
        metrics = operator["source_weighted_integrals"]
        assert len(metrics) == 12
        assert operator["normative_release_eligible"] is False
        assert operator["operator_incomplete_reasons"]
        if operator["time_case"] in result["histories"]:
            history = result["histories"][operator["time_case"]]
            assert operator["local_time_history_sha256"] == history["sha256"]
            assert set(history["payload"]["incomplete_reasons"]).issubset(
                operator["operator_incomplete_reasons"],
            )
        for product in sources:
            key = runner._source._product_key(product)
            metric = metrics[key]
            assert metric["time_multiplier"] == "none"
            assert metric["source_incomplete_reasons"] == product.spectrum.incomplete_reasons
            assert metric["source_differential_unit"] == product.spectrum.y_unit.value
            if product.kind is RadiationProductKind.MISSION_FLUENCE:
                assert metric["unit"] == "cm^-2"
                assert metric["mission_duration_s"] == 157788000.0
            else:
                assert metric["unit"] == "cm^-2 s^-1"
                assert metric["mission_duration_s"] is None


def test_comparison_axes_do_not_mix_candidate_definition_with_density(runner: ModuleType) -> None:
    variants = []
    for spec in runner._specifications("all"):
        actual = 1000.2 if spec.steps_per_orbit == 240 else 1000.1
        rows = [{
            "name": f"{spec.name}:{name}", "time_case": name,
            "local_time_history_model": name, "requested_window_s": 1000.0,
            "actual_window_s": actual, "penetration": (0.2, 0.8),
            "source_weighted_integrals": {"p": {"value": 0.0}},
        } for name in runner._OPERATORS]
        variants.append({"name": spec.name, "operators": rows})
    comparisons = runner._comparisons(variants)
    assert len(comparisons) == 88
    density = [item for item in comparisons if item["axis"].startswith("orbital_density_")]
    assert len(density) == 14
    assert all(item["density_comparison_includes_rounding_window_change"] for item in density)
    assert all(item["left_local_time_model"] == item["right_local_time_model"] for item in density)
    definition = [item for item in comparisons if item["axis"].startswith("candidate_time_definition")]
    assert len(definition) == 12
    assert all(item["actual_window_equal"] for item in definition)
    assert all(item["source_integral_changes"]["p"]["relative_change"] is None
               for item in comparisons)
    assert all(item["candidate_selection_performed"] is False for item in comparisons)
    assert all(item["normative_acceptance_established"] is False for item in comparisons)


def test_hash_scope_pins_new_production_module_and_both_imported_helpers(runner: ModuleType) -> None:
    hashes = runner._model_hashes()
    assert {"src/radar/geomagnetic/local_time.py", "src/radar/geomagnetic/ost_penetration.py",
            "scripts/validate_heo_orbit_sampling.py", "scripts/validate_heo_mission_window.py",
            "scripts/validate_mlt_correlation.py"}.issubset(hashes)
    assert all(len(digest) == 64 for digest in hashes.values())


def test_relative_comparison_uses_left_denominator_and_reports_right_value(runner: ModuleType) -> None:
    left = {"name": "reference", "local_time_history_model": "left_model",
            "requested_window_s": 1000.0, "actual_window_s": 1000.1,
            "penetration": (0.25, 0.5), "source_weighted_integrals": {"p": {"value": 100.0}}}
    right = {**left, "name": "candidate", "local_time_history_model": "right_model",
             "penetration": (0.375, 0.5), "source_weighted_integrals": {"p": {"value": 125.0}}}
    comparison = runner._pair(left, right, "candidate_comparison")
    metric = comparison["source_integral_changes"]["p"]
    assert metric["reference"] == 100.0
    assert metric["value"] == 125.0
    assert metric["absolute_change"] == 25.0
    assert metric["relative_change"] == 0.25
    assert comparison["max_absolute_psi_change"] == 0.125
    assert comparison["psi_change_is_dimensionless_not_relative_flux_error"] is True


def test_changed_implementation_invalidates_report(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashes = iter(({"implementation": "before"}, {"implementation": "after"}))
    monkeypatch.setattr(runner, "_model_hashes", lambda: next(hashes))
    monkeypatch.setattr(runner, "_variant", lambda *args: {})
    with pytest.raises(ValueError, match="changed during diagnostics"):
        runner.build_report("smoke")


def test_report_does_not_promote_candidate_to_normative_or_mission_history(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real source models and operators, but a deliberately tiny orbit trace.
    spec = _tiny_spec(runner)
    monkeypatch.setattr(runner, "_specifications", lambda case: (spec,))
    progress: list[dict[str, Any]] = []
    report = runner.build_report("smoke", progress=progress.append)
    assert report["schema"] == "radar_local_time_correlation_diagnostic_v1"
    assert report["normative_release_eligible"] is False
    assert report["mission_radiation_history_qualified"] is False
    assert report["stage4_spectral_convergence_established"] is False
    assert report["candidate_selection_performed"] is False
    assert report["source_roster_complete"] is False
    assert len(report["source_products"]) == 12
    assert len(report["rigidity_gv"]) == 1047
    assert len(progress) == 1
    assert report["model_sha256_before"] == report["model_sha256_after"]
    assert report["input_sha256_before"] == report["input_sha256_after"]
    assert report["conventions"]["sun_declination_deg"] == 0.0
    assert "not_UTC" in report["conventions"]["time_reference"]
    assert "no_extra_time_factor" in report["conventions"]["source_operator"]
    json.dumps(report, allow_nan=False)


def test_output_is_exclusive_and_nonfinite_payload_is_rejected_before_creation(
    runner: ModuleType, tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    runner.write_report_exclusively({"keep": True}, path)
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively({"keep": False}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"keep": True}
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        runner.write_report_exclusively({"value": math.nan}, invalid)
    assert not invalid.exists()


def test_cli_refuses_existing_output_before_calculation(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path / "existing.json"
    runner.write_report_exclusively({"keep": True}, path)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Existing outputs must be refused before calculations")

    monkeypatch.setattr(runner, "build_report", forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main(["--case", "smoke", "--output", str(path)])
    assert error.value.code == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"keep": True}
