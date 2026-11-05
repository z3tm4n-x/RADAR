"""Fast diagnostic-runner checks; these do not certify physical spectral convergence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from radar.geomagnetic.ost_orbit import sample_ost_orbit
from radar.geomagnetic.rigidity import RigidityGrid


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/validate_geomagnetic_operator.py"
    spec = importlib.util.spec_from_file_location("geomagnetic_operator_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "perigee", "apogee", "inclination", "cap"),
    [("leo", 400, 400, 51.6, 400000), ("geo", 35786, 35786, 0, 400000),
     ("heo", 500, 39700, 63.4, 30000)],
)
def test_fixed_cases_and_explicit_limits(
    runner: ModuleType, name: str, perigee: int, apogee: int, inclination: float, cap: int,
    tmp_path: Path,
) -> None:
    case = runner._case(name, tmp_path)
    assert case.orbit.perigee_altitude_km == perigee
    assert case.orbit.apogee_altitude_km == apogee
    assert case.orbit.inclination_deg == inclination
    assert case.max_samples == cap
    assert len(case.rigidity_grid.values_gv) == 401
    assert case.rigidity_grid.values_gv[0] == 1e-4
    assert case.rigidity_grid.values_gv[-1] == 100.0


def test_unknown_case_fails(runner: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        runner._case("unspecified", tmp_path)


def _capture(tmp_path: Path, *, inclination: float = 0.0) -> Path:
    directory = tmp_path / "capture_002_equatorial"
    directory.mkdir()
    header = [
        "Apogee = 35786 km.", "Perigee = 35786 km.",
        f"Inclination ={inclination} deg.", "Argument of perigee =0.0 deg.",
    ]
    cutoff = header + ["1 col.- rigidity [GV]", "2 col.- penitration function"]
    cutoff += [f"{(i + 1) / 1000:.6e} {i / 54:.9e}" for i in range(55)]
    (directory / "cutoff.txt").write_text("\n".join(cutoff) + "\n", encoding="ascii")
    (directory / "orbit.txt").write_text("\n".join(header) + "\n", encoding="ascii")
    return directory


def test_capture_reader_retains_nodes_and_verifies_all_orbital_headers(
    runner: ModuleType, tmp_path: Path,
) -> None:
    directory = _capture(tmp_path)
    case = runner._case("cosrad_eq", tmp_path)
    assert case.capture_directory == directory.resolve()
    assert len(case.input_files) == 2
    assert case.cosrad_values[0] == 0 and case.cosrad_values[-1] == 1
    assert case.raw_cutoff_header[2] == "Inclination =0.0 deg."
    path = directory / "orbit.txt"
    path.write_text(path.read_text().replace("35786", "35787"), encoding="ascii")
    with pytest.raises(ValueError, match="headers disagree"):
        runner._case("cosrad_eq", tmp_path)


def test_wrong_capture_geometry_fails(runner: ModuleType, tmp_path: Path) -> None:
    _capture(tmp_path, inclination=51.6)
    with pytest.raises(ValueError, match="inclination"):
        runner._case("cosrad_eq", tmp_path)


def test_legacy_inclusive_boundary_is_separate_from_new_kernel(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = sample_ost_orbit(runner._orbit(400.0, 400.0, 0.0), min_days=0.0001)
    monkeypatch.setattr(runner, "ost_sample_cutoff_rigidity_gv", lambda sample: 2.5)
    monkeypatch.setattr(runner, "ost_sample_shadow_weight", lambda sample: 0.8)
    monkeypatch.setattr(
        runner, "ost_disturbed_cutoff_rigidity_mlt_average_gv", lambda **kwargs: 1.0,
    )
    grid = RigidityGrid((0.999, 1.0, 1.001))
    assert runner._legacy_values(trace, grid, 24) == pytest.approx((0.0, 0.8, 0.8))


def test_log_r_area_is_dimensionless_diagnostic(runner: ModuleType) -> None:
    grid = RigidityGrid((0.01, 1.0, 100.0))
    assert runner._log_r_area(grid, (0.5, 0.5, 0.5)) == pytest.approx(0.5)
    result = runner._difference((0.5, 0.5, 0.5), (0.0, 0.1, 0.2), grid)
    assert result["max_absolute_change"] == 0.5
    assert result["log_r_area_is_not_a_physical_spectrum_integral"] is True


def _fake_variant(case: Any, steps: int, mlt: int) -> dict[str, Any]:
    # All orbital refinements deliberately hit the same cap.
    values = tuple(0.5 + 0.01 / mlt for _ in case.rigidity_grid.values_gv)
    return {
        "requested_steps_per_orbit": steps, "mlt_sample_count": mlt,
        "sample_count": 30000, "actual_step_s": 1000.0,
        "sampling_capped": True, "new_pointwise_penetration": values,
    }


def test_capped_identical_grid_does_not_claim_density_convergence(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_variant", _fake_variant)
    report = runner._case_report(runner._case("heo", tmp_path))
    pairs = report["density_comparisons"]
    assert all(not pair["density_comparison_informative"] for pair in pairs[:2])
    assert all(pair["limitation"] == "sample_cap_prevents_density_refinement" for pair in pairs[:2])
    assert all(pair["density_comparison_informative"] for pair in pairs[2:])


def test_report_pins_inputs_and_keeps_all_acceptance_flags_false(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _capture(tmp_path)
    monkeypatch.setattr(runner, "_variant", _fake_variant)
    report = runner.build_report("cosrad_eq", capture_root=tmp_path)
    assert report["input_sha256_before"] == report["input_sha256_after"]
    assert report["model_sha256_before"] == report["model_sha256_after"]
    assert len(report["input_sha256_before"]) == 2
    for key in (
        "normative_release_eligible", "parameters_fitted", "peak_operator_qualified",
        "cumulative_operator_qualified", "stage4_spectral_convergence_established",
    ):
        assert report[key] is False
    assert report["conventions"]["kp"] == 3
    assert report["production_operator_version"] == "appendix_zh_pointwise_access_v2"
    assert report["schema"] == "radar_geomagnetic_operator_diagnostic_v2"
    assert report["orbit_sampling_model_version"] == "bounded_orbit_cells_v1"
    assert "not_historical_sampler_replay" in report["conventions"]["legacy_implementation"]
    assert report["conventions"]["ladder"] == ((60, 24), (120, 24), (240, 24), (240, 48), (240, 96))


def test_changed_input_fails(runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    directory = _capture(tmp_path)

    def mutate(case: Any) -> dict[str, Any]:
        path = directory / "cutoff.txt"
        path.write_bytes(path.read_bytes() + b"\n")
        return {}

    monkeypatch.setattr(runner, "_case_report", mutate)
    with pytest.raises(ValueError, match="changed during"):
        runner.build_report("cosrad_eq", capture_root=tmp_path)


def test_mutation_between_parsing_and_initial_hash_is_rejected(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    directory = _capture(tmp_path)
    original = runner._case

    def mutate(name: str, capture_root: Path) -> Any:
        result = original(name, capture_root)
        path = directory / "cutoff.txt"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(runner, "_case", mutate)
    with pytest.raises(ValueError, match="changed after parsing"):
        runner.build_report("cosrad_eq", capture_root=tmp_path)


def test_changed_model_fails(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def hashes() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"test": str(calls)}

    monkeypatch.setattr(runner, "_model_hashes", hashes)
    monkeypatch.setattr(runner, "_case_report", lambda case: {})
    with pytest.raises(ValueError, match="changed during"):
        runner.build_report("geo")


def test_exclusive_output_and_strict_serialization(runner: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    runner.write_report_exclusively({"normative_release_eligible": False}, path)
    original = path.read_bytes()
    assert json.loads(original)["normative_release_eligible"] is False
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively({}, path)
    assert path.read_bytes() == original
    missing = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError):
        runner.write_report_exclusively({"bad": float("nan")}, missing)
    assert not missing.exists()


def test_cli_output_required_and_no_overwrite(runner: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as missing:
        runner.main(["--case", "geo"])
    assert missing.value.code == 2
    output = tmp_path / "keep.json"
    output.write_text("keep", encoding="ascii")
    with pytest.raises(SystemExit) as existing:
        runner.main(["--case", "geo", "--output", str(output)])
    assert existing.value.code == 2
    assert output.read_text() == "keep"
