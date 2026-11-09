"""Bounded source refinement contracts with no private native evidence required."""
from pathlib import Path
import sys
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_paired_source_refinement as probe  # noqa: E402
from report_paired_stopping_impact import index_report  # noqa: E402


def model(value: float) -> dict[tuple[Any, ...], Any]:
    return {("gcr", "peak", 26, .1): {"source_integral": 10000., "variants": {
        v: {"let_ge": {"20.0": value}, "energy_ge": {"1.0": 99.}} for v in ("SS", "SC", "CS", "CC")}}}


def test_levels_preserve_all_previous_nodes_and_do_not_overshoot() -> None:
    report = probe.grids((1., 4., 16.))
    assert tuple(report) == (1, 2, 4, 8, 16, 32)
    assert report[2] == (1., 2., 4., 8., 16.)
    for n, row in report.items():
        assert len(row) == 2*n+1 and row[::n] == report[1]
        if n > 1:
            assert row[::2] == report[n//2]


@pytest.mark.parametrize("nodes", [(), (1.,), (1., 1.), (1., 0.), (1., float("nan"))])
def test_invalid_nodes_rejected(nodes: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        probe.grids(nodes)


def test_distinct_denominators_and_zero_policy() -> None:
    assert probe.changes(10, 12, 200) == {"difference": 2., "percent_of_previous": 20.,
                                           "percent_of_previous_full_source": 1.}
    assert probe.changes(0., 0., 1.)["percent_of_previous"] is None
    assert probe.changes(0., 2., 10.)["percent_of_previous_full_source"] == 20.


@pytest.mark.parametrize("a,b,s", [(-1., 0., 1.), (0., -1., 1.), (0., 0., 0.), (1., 1., float("inf"))])
def test_bad_denominators_or_counts_fail(a: float, b: float, s: float) -> None:
    with pytest.raises(ValueError):
        probe.changes(a, b, s)


@pytest.mark.parametrize("new,expected", [(100., 0), (101., 0), (101.1, 4), (98.9, 4)])
def test_strict_magnitude_threshold_frozen(new: float, expected: int) -> None:
    result = probe.exceptional(model(100.), model(new))
    assert len(result) == expected and all(k[-2:] == ("let_ge", "20.0") for k in result)


def test_newly_nonzero_not_silently_dropped() -> None:
    with pytest.raises(ValueError, match="zero-base"):
        probe.exceptional(model(0.), model(1.))
    assert probe.exceptional(model(0.), model(0.)) == ()


def test_mixed_case_or_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="inventories"):
        probe.exceptional({}, model(2.))
    dense = model(2.)
    dense["gcr", "peak", 26, .1]["variants"]["CC"]["let_ge"] = {"10.0": 2.}
    with pytest.raises(ValueError, match="Threshold"):
        probe.exceptional(model(1.), dense)


def test_small_real_model_evaluates_midpoints_rather_than_linear_source_interpolation() -> None:
    initial = (100., 200., 400.)
    g = probe.grids(initial)
    old, new = probe.generated(g[1]), probe.generated(g[2])
    assert len(old) == len(new) == 6
    for key in new:
        assert new[key].x[::2] == old[key].x and new[key].y[::2] == old[key].y
    # Compare against energy-linear prediction for an actual GCR He source.
    s0, s1 = old["peak", 2], new["peak", 2]
    interp = s0.y[0]+(s0.y[1]-s0.y[0])*(s1.x[1]-s0.x[0])/(s0.x[1]-s0.x[0])
    assert s1.y[1] != pytest.approx(interp, rel=1e-5)


def test_existing_output_refused_before_source_or_native_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path/"existing.json"
    path.write_bytes(b"existing")
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(path)])
    monkeypatch.setattr(probe, "run", lambda: pytest.fail("pytest read/compute before overwrite check"))
    with pytest.raises(FileExistsError):
        probe.main()
    assert path.read_bytes() == b"existing"


@pytest.mark.external_data
def test_retained_subset_rows_and_both_replays() -> None:
    # These local scientific artifacts are intentionally absent in a public clone.
    import json
    folder = probe.ROOT / "outputs"
    a, b, primary, repeat = [folder/p for p in (probe.FILES[0][0], probe.FILES[1][0],
        "paired_refinement_probe_final_20261109.json", "paired_refinement_probe_final_repeat_20261109.json")]
    missing = [p.name for p in (a, b, primary, repeat) if not p.is_file()]
    if missing:
        pytest.skip("Probe evidence not retained in checkout: " + ", ".join(missing))
    assert primary.read_bytes() == repeat.read_bytes()
    assert probe.digest(a) == probe.BASE and probe.digest(b) == probe.DENSE
    result = json.loads(primary.read_text(encoding="utf-8"))
    archives = [index_report(json.loads(path.read_text(encoding="utf-8")), level, 80)
                for path, level in ((a, 1), (b, 2))]
    expected = set(probe.exceptional(*archives))
    assert len(expected) == 118 and result["row_count"] == len(result["rows"]) == 118
    assert set(tuple(r["key"]) for r in result["rows"]) == expected
    assert result["node_counts"] == [81, 161, 321, 641, 1281, 2561]
    for row in result["rows"]:
        k = row["key"]
        counts = row["values"]
        assert counts[:2] == [ar[tuple(k[:4])]["variants"][k[4]][k[5]][k[6]] for ar in archives]
        assert len(counts) == 6 and row["steps"] == [probe.changes(x, y, s) for x, y, s in
            zip(counts, counts[1:], row["full_source_integrals"])]
    assert [r["above_one_percent"] for r in result["summary"]] == [118, 16, 8, 4, 0]
    assert result["global_convergence_established"] is False and result["normative_export_allowed"] is False
