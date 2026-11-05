"""Check that work forecasts are not presented as evaluated transport."""

import importlib.util
from pathlib import Path
import sys

import pytest

from radar.core.types import AssessmentMode, TemporalBasis
from radar.core.units import Unit
import radar.shielding.proton_al_secondary as secondary
from test_proton_secondary_bin_zero_proof import _stopping
from test_proton_secondary_active_cell_boundaries import _source


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_secondary_active_cells.py"
    spec = importlib.util.spec_from_file_location("radar_active_preflight_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_work_forecast_keeps_positive_to_zero_segments_and_ignores_exact_zero_gap() -> None:
    runner = _runner()
    source = _source((1.0, 2.0, 3.0, 4.0), (1.0, 0.0, 0.0, 1.0))
    assert runner._work_counts(source, source.x, _stopping()) == (2, 8, 24)


def test_work_forecast_does_not_count_source_intervals_outside_range_support() -> None:
    runner = _runner()
    source = _source((0.01, 0.1, 1.0, 20.0, 30.0), (1.0,) * 5)
    assert runner._work_counts(source, source.x, _stopping()) == (2, 8, 24)


def _mixed_screening_plan():
    """A declared classification fixture, not a test of the physical proofs."""
    boundaries = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    classes = ("evaluate", "proven_zero", "omitted_unsupported", "evaluate", "source_zero")
    weights = (1.0, 1.0, 1.0, 0.5, 0.0)
    intervals = tuple(
        secondary._ScreenedIncidentInterval(lo, hi, category, category, weight)
        for lo, hi, category, weight in zip(boundaries, boundaries[1:], classes, weights)
    )
    return secondary._SecondaryScreeningPlan(boundaries, intervals, ((1.0, 2.0), (4.0, 5.0)))


def test_screened_forecast_counts_only_evaluate_panels_without_responses(monkeypatch) -> None:
    def forbidden_response(*args, **kwargs):
        pytest.fail("A work forecast must not evaluate a secondary response")

    monkeypatch.setattr(secondary, "_single_incident_secondary_response", forbidden_response)
    source = _source((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
    runner = _runner()
    assert runner._work_counts(source, source.x, _stopping()) == (4, 16, 48)
    assert runner._work_counts(
        source, source.x, _stopping(), screening=_mixed_screening_plan()
    ) == (2, 8, 24)


def test_screened_forecast_keeps_a_panel_crossing_proof_boundaries() -> None:
    source = _source((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (1.0,) * 6)
    # [2,4] crosses a proven-zero cell and an unknown cell: no single proof
    # applies to the combined interval, so conservatively retain its work.
    assert _runner()._work_counts(
        source, (2.0, 4.0), _stopping(), screening=_mixed_screening_plan()
    ) == (1, 4, 12)


def test_screening_report_keeps_zero_and_unknown_weights_separate() -> None:
    source = _source((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
    plan = _mixed_screening_plan()
    boundaries = tuple(1.0 + index / 2 for index in range(11))
    partition = secondary._IncidentOutputPartition(
        (0,), (1.0,), (0.5, 1.5), secondary._ExitDepthPartition((), False),
        boundaries, False, plan,
    )
    record = _runner()._screening_record(source, partition, include_intervals=True)
    assert record["base_interval_counts"] == {
        "source_zero": 1, "proven_zero": 1, "omitted_unsupported": 1, "evaluate": 2,
    }
    assert record["final_interval_counts"] == {
        "source_zero": 2, "proven_zero": 2, "omitted_unsupported": 2, "evaluate": 4,
    }
    assert record["base_source_integral_by_classification"] == {
        "source_zero": 0.0, "proven_zero": 1.0, "omitted_unsupported": 1.0, "evaluate": 1.5,
    }
    assert record["omitted_physical_contribution_error"] == "unknown"
    assert "must not be summed across output bins" in record["source_weight_scope"]
    assert record["source_integral_dimension"] == "(cm^-2 MeV^-1) * (MeV)"
    assert len(record["base_intervals"]) == 5
    assert record["base_intervals"][2]["classification"] == "omitted_unsupported"
    assert _runner()._screening_record(
        source, partition, include_intervals=False
    )["base_intervals"] is None


def test_preflight_cli_requires_screening_for_interval_detail(monkeypatch) -> None:
    runner = _runner()
    monkeypatch.setattr(sys, "argv", ["preflight", "--include-screening-intervals"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_preflight_sources_use_actual_mission_timeline_and_preserve_time_units() -> None:
    sources, record = _runner()._sources(81)
    peak, cumulative = sources["peak"], sources["cumulative"]
    assert record["solar_months"] == 60
    assert record["positive_support_identical_for_both_modes"]
    assert peak.assessment_mode is AssessmentMode.PEAK
    assert cumulative.assessment_mode is AssessmentMode.CUMULATIVE
    assert peak.y_unit is Unit.DIFFERENTIAL_FLUX
    assert cumulative.y_unit is Unit.DIFFERENTIAL_FLUENCE
    assert peak.mission_duration_s is None
    assert cumulative.mission_duration_s == 157788000.0
    assert cumulative.temporal_basis is TemporalBasis.MISSION_TOTAL
    assert peak.incomplete_reasons and cumulative.incomplete_reasons
    assert len(peak.x) == len(cumulative.x) == 81


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_preflight_cli_rejects_nonpositive_or_nonfinite_depth(value, monkeypatch) -> None:
    runner = _runner()
    monkeypatch.setattr(sys, "argv", ["preflight", "--thickness", value])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_preflight_cli_rejects_response_budget_that_cannot_hold_one_rule(monkeypatch) -> None:
    runner = _runner()
    monkeypatch.setattr(sys, "argv", ["preflight", "--max-response-evaluations", "3"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
