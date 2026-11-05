"""Synthetic structural/reproducibility checks, not proprietary golden spectra."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/compare_cosrad_cutoff.py"
    spec = importlib.util.spec_from_file_location("cosrad_cutoff_comparison_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _lines() -> list[str]:
    return [
        "Apogee = 35786 km.", "Perigee = 35786 km.",
        "Inclination = 0.0 deg.", "Argument of perigee = 0.0 deg.",
        "1 col.- rigidity [GV]", "2 col.- penitration function",
    ] + [f"{(i + 1) / 1000:.6e} {i / 54:.9e}" for i in range(55)]


def _bytes(lines: list[str]) -> bytes:
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def test_parser_infers_geometry_and_preserves_nodes(runner: ModuleType) -> None:
    table = runner.parse_cutoff(_bytes(_lines()), expected_inclination_deg=0.0)
    assert table.orbit.apogee_altitude_km == 35786.0
    assert table.orbit.perigee_altitude_km == 35786.0
    assert table.orbit.inclination_deg == 0.0
    assert table.orbit.argument_of_perigee_deg == 0.0
    assert table.rigidity_grid.values_gv == tuple((i + 1) / 1000 for i in range(55))
    assert table.values[0] == 0.0 and table.values[-1] == 1.0
    assert table.source_sha256 == hashlib.sha256(_bytes(_lines())).hexdigest()


@pytest.mark.parametrize(
    ("inclination_header", "argument_header", "expected"),
    [
        ("Inclination =51.6 deg.", "Argument of perigee = 0.0 deg.", 51.6),
        ("Inclination = 0.0 deg.", "Argument of perigee =0.0 deg.", 0.0),
        ("Inclination\t=\t51.6\tdeg.", "Argument of perigee\t=\t0.0 deg.", 51.6),
    ],
)
def test_observed_fortran_padding_preserves_raw_header(
    runner: ModuleType, inclination_header: str, argument_header: str, expected: float,
) -> None:
    lines = _lines()
    lines[2:4] = [inclination_header, argument_header]
    table = runner.parse_cutoff(_bytes(lines), expected_inclination_deg=expected)
    assert table.orbit.inclination_deg == expected
    assert table.orbit.argument_of_perigee_deg == 0.0
    assert table.raw_header == tuple(lines[:6])


@pytest.mark.parametrize(
    "header", [
        "Inclination ==51.6 deg.", "Inclination =51.6 deg. garbage",
        "Inclination =51.6deg.", "Inclination =51.6 123 deg.",
        "Inclination =\n51.6 deg.", "Inclination =\v51.6 deg.",
    ],
)
def test_padding_support_does_not_accept_header_garbage(runner: ModuleType, header: str) -> None:
    lines = _lines()
    lines[2] = header
    with pytest.raises(ValueError):
        runner.parse_cutoff(_bytes(lines))


@pytest.mark.parametrize("expected", [51.6, float("nan"), float("inf")])
def test_expected_inclination_is_assertion_not_override(runner: ModuleType, expected: float) -> None:
    with pytest.raises(ValueError):
        runner.parse_cutoff(_bytes(_lines()), expected_inclination_deg=expected)


@pytest.mark.parametrize("token", ["1e-9999", "-1e-9999"])
@pytest.mark.parametrize("target", ["rigidity", "penetration", "apogee", "perigee", "i", "arg"])
def test_nonzero_decimal_underflow_is_never_silently_zero(
    runner: ModuleType, token: str, target: str,
) -> None:
    lines = _lines()
    # A zero-altitude orbit is valid, so geometry checks cannot hide underflow.
    lines[0:4] = [
        "Apogee = 0 km.", "Perigee = 0 km.",
        "Inclination = 0 deg.", "Argument of perigee = 0 deg.",
    ]
    if target == "rigidity":
        lines[6] = f"{token} 0"
    elif target == "penetration":
        lines[6] = f"0.001 {token}"
    else:
        index = {"apogee": 0, "perigee": 1, "i": 2, "arg": 3}[target]
        lines[index] = lines[index].replace("= 0 ", f"= {token} ")
    with pytest.raises(ValueError, match="underflows to zero"):
        runner.parse_cutoff(_bytes(lines))


@pytest.mark.parametrize("token", ["0e-9999", "-0.000e-9999", "+0.0e+9999"])
def test_exact_decimal_zero_is_not_mistaken_for_underflow(runner: ModuleType, token: str) -> None:
    lines = _lines()
    lines[6] = f"0.001 {token}"
    assert runner.parse_cutoff(_bytes(lines)).values[0] == 0.0


@pytest.mark.parametrize(
    ("index", "replacement"),
    [
        (0, "Apogee = 35000 km."), (1, "Perigee = -1 km."),
        (0, "Apogee = inf km."), (0, "Apogee = 1e309 km."),
        (2, "Inclination = 181 deg."), (3, "Argument of perigee = 360 deg."),
        (4, "1 col.- rigidity [MV]"), (5, "2 col.- something else"),
        (6, "0 0"), (6, "-0.001 0"), (7, "0.001 0.1"), (7, "0.0009 0.1"),
        (6, "nan 0"), (6, "1e309 0"), (6, "0.001 inf"),
        (6, "0.001 -0.1"), (6, "0.001 1.1"), (8, "0.003 0"),
        (6, "0.001"), (6, "0.001 0 1"), (6, ""),
    ],
)
def test_rejects_malformed_or_nonphysical_rows(
    runner: ModuleType, index: int, replacement: str,
) -> None:
    lines = _lines()
    lines[index] = replacement
    with pytest.raises(ValueError):
        runner.parse_cutoff(_bytes(lines))


@pytest.mark.parametrize("mutation", ["missing_header", "duplicate_header", "short", "long", "blank"])
def test_rejects_wrong_table_shape(runner: ModuleType, mutation: str) -> None:
    lines = _lines()
    if mutation == "missing_header":
        del lines[0]
    elif mutation == "duplicate_header":
        lines.insert(0, lines[0])
    elif mutation == "short":
        lines.pop()
    elif mutation == "long":
        lines.append("0.056 1")
    else:
        lines.append("")
    with pytest.raises(ValueError):
        runner.parse_cutoff(_bytes(lines))


def test_report_explicit_scope_defaults_hashes_and_exclusive_json(
    runner: ModuleType, tmp_path: Path,
) -> None:
    raw = _bytes(_lines())
    source = tmp_path / "cutoff.txt"
    source.write_bytes(raw)
    (tmp_path / "orbit.txt").write_bytes(_bytes(_lines()[:4] + ["synthetic orbit body"]))
    report = runner.build_comparison(tmp_path, expected_inclination_deg=0.0)
    assert source.read_bytes() == raw
    assert report["normative_release_eligible"] is False
    assert report["schema"] == "radar_cosrad_cutoff_comparison_v3"
    assert report["orbit_sampling_model_version"] == "bounded_orbit_cells_v1"
    assert report["radar_conventions"]["budget_policy"] == (
        "fail_before_sampling_never_coarsen_or_truncate"
    )
    assert report["parameters_fitted"] is False
    assert report["evidence_role"] == "independent_model_comparison_only"
    assert report["orbit_header_crosscheck"] == "matched"
    assert report["input_sha256_before"] == report["input_sha256_after"]
    assert report["model_sha256_before"] == report["model_sha256_after"]
    assert "src/radar/geomagnetic/ost.py" in report["model_sha256_before"]
    assert "Kp" in report["unestablished_cosrad_conditions"]
    assert report["radar_conventions"]["kp"] == 3
    assert report["radar_conventions"]["apply_disturbance"] is True
    assert report["radar_conventions"]["mlt_sample_count"] == 24
    assert report["radar_conventions"]["cutoff_condition"] == "R>Rc_star"
    assert report["radar_conventions"]["averaging"] == (
        "psi=mean_time(mean_MLT(mu_i*I(R>Rc_i/delta_i(tMLT))))"
    )
    assert "mean_MLT(delta)" not in json.dumps(report)
    assert "Rc_star_i<=R" not in json.dumps(report)
    assert report["operator_qualification"]["status"] == "INCOMPLETE"
    assert len(report["operator_qualification"]["incomplete_reasons"]) >= 2
    assert [v["steps_per_orbit"] for v in report["variants"]] == [60, 120, 240]
    for variant in report["variants"]:
        assert variant["operator_version"] == "appendix_zh_pointwise_access_v2"
        assert variant["operator_parameters"]["cutoff_condition"] == "R>Rc_star"
        assert variant["operator_parameters"]["mlt_policy"] == "uniform_ensemble"
        assert variant["operator_incomplete_reasons"]
        assert variant["disturbed_cutoff_range_definition"] == (
            "all_orbit_x_uniform_24_MLT_pointwise_states"
        )
        differences = variant["radar_minus_cosrad"]
        assert len(differences) == 55
        maximum = variant["largest_absolute_discrepancy"]
        assert maximum["absolute_difference"] == max(abs(v) for v in differences)
    output = tmp_path / "report.json"
    runner.write_report_exclusively(report, output)
    assert json.loads(output.read_text(encoding="utf-8"))["normative_release_eligible"] is False
    original_output = output.read_bytes()
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively(report, output)
    assert output.read_bytes() == original_output


def test_disturbed_extrema_use_all_pointwise_mlt_states_not_mean_delta(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    visits: list[tuple[float, float]] = []

    def pointwise(
        *, cutoff_rigidity_gv: float, kp: int, magnetic_local_time_hours: float,
    ) -> float:
        assert kp == 3
        visits.append((cutoff_rigidity_gv, magnetic_local_time_hours))
        return cutoff_rigidity_gv / (1.0 + magnetic_local_time_hours)

    monkeypatch.setattr(runner, "ost_disturbed_cutoff_rigidity_gv", pointwise)
    assert runner._pointwise_cutoff_range((1.0, 2.0)) == pytest.approx((1.0 / 24.0, 2.0))
    assert len(visits) == 48
    assert set(time for _, time in visits) == set(range(24))


def test_rejects_orbit_header_mismatch(runner: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "cutoff.txt").write_bytes(_bytes(_lines()))
    lines = _lines()[:4]
    lines[2] = "Inclination = 51.6 deg."
    (tmp_path / "orbit.txt").write_bytes(_bytes(lines))
    with pytest.raises(ValueError, match="headers disagree"):
        runner.build_comparison(tmp_path)


def test_source_mutation_fails_before_a_report_can_be_written(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "cutoff.txt"
    source.write_bytes(_bytes(_lines()))
    original = runner._variant

    def mutate(table: object, steps: int) -> object:
        result: object = original(table, steps)
        source.write_bytes(source.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(runner, "_variant", mutate)
    with pytest.raises(ValueError, match="changed during"):
        runner.build_comparison(tmp_path)


def test_strict_json_does_not_leave_nonfinite_output(runner: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        runner.write_report_exclusively({"bad": float("nan")}, output)
    assert not output.exists()


def test_model_mutation_is_not_silently_accepted(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "cutoff.txt").write_bytes(_bytes(_lines()))
    hash_calls = 0

    def changed_hash() -> dict[str, str]:
        nonlocal hash_calls
        hash_calls += 1
        return {"synthetic_model.py": str(hash_calls)}

    monkeypatch.setattr(runner, "_model_hashes", changed_hash)
    with pytest.raises(ValueError, match="changed during"):
        runner.build_comparison(tmp_path)


def test_cli_requires_explicit_output_and_preserves_existing_output(
    runner: ModuleType, tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as missing:
        runner.main(["--capture-dir", str(tmp_path)])
    assert missing.value.code == 2
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="ascii")
    with pytest.raises(SystemExit) as existing:
        runner.main(["--capture-dir", str(tmp_path), "--output", str(output)])
    assert existing.value.code == 2
    assert output.read_text(encoding="ascii") == "preserve"


@pytest.mark.parametrize("token", ["1e-9999", "-1e-9999"])
def test_cli_inclination_underflow_is_rejected_before_running(
    runner: ModuleType, tmp_path: Path, token: str,
) -> None:
    output = tmp_path / "underflow.json"
    with pytest.raises(SystemExit) as error:
        runner.main([
            "--capture-dir", str(tmp_path), f"--inclination-deg={token}",
            "--output", str(output),
        ])
    assert error.value.code == 2
    assert not output.exists()
