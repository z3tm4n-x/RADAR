"""Independent comparison guards; synthetic controls never claim native execution."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from math import inf, nan
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from radar.validation.nist_astar import AstarRow, load_astar_archive
from radar.validation.parameterized_stopping import ParameterizedStoppingRow


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "research/nist_star_20261031/STAR.tar.gz"


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    scripts = ROOT / "scripts"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(scripts))
        spec = importlib.util.spec_from_file_location("astar_helium_comparison_test",
                                                     scripts / "compare_astar_helium.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _energies() -> tuple[float, ...]:
    return tuple(row.energy_mev_total for row in load_astar_archive(ARCHIVE).material(13).rows
                 if row.energy_mev_total >= .04)


@pytest.fixture
def controls(runner: ModuleType, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    archive = load_astar_archive(ARCHIVE)
    energies = _energies()
    production = runner.native.production_hashes()
    implementation_paths = (
        "scripts/generate_astar_helium_controls.py", "scripts/generate_catima_candidate.py",
        "src/radar/validation/nist_astar.py",
        "research/ion_stopping_20261031/runtime/include/catima/data_ionisation_potential.h",
    )
    implementation = {p: _sha(ROOT / p) for p in implementation_paths}
    rows = [{"material_z": z, "energy_mev_total": e, "energy_mev_per_nucleon": e / 4,
             "stopping_electronic_mev_cm2_g": 100 + e / 4,
             "stopping_nuclear_mev_cm2_g": 1., "stopping_radiative_mev_cm2_g": 0.,
             "range_g_cm2": e + 3.} for z in (13, 14) for e in energies]
    raw: dict[str, Any] = {
        "schema_version": "radar.astar_helium_native_controls.v1",
        "archive_path": str(ARCHIVE.resolve()), "archive_sha256": archive.archive_sha256,
        "archive_member_sha256": dict(archive.member_sha256), "archive_version": "1.2",
        "implementation_sha256_before": implementation, "implementation_sha256_after": implementation,
        "native_source_pins_before": dict(runner.native.PINS),
        "native_source_pins_after": dict(runner.native.PINS),
        "production_resources_sha256_before": production, "production_resources_sha256_after": production,
        "python": "3.12 synthetic fixture", "backend_info_literal": "synthetic_not_native_execution",
        "projectile": {"z": 2, "mass_parameter_u": 4.0, "charge_input": 2},
        "energies_mev_total": list(energies), "material_atomic_numbers": [13, 14],
        "range_origin_energy_mev_per_nucleon": .001,
        "energy_coordinate_compatibility": "source_MeV_n_numerically_equal_backend_MeV_u_assumed",
        "energy_conversion": "backend_energy_mev_per_u=ASTAR_whole_alpha_energy_mev/4.0",
        "native_sampling_policy": "exact_ASTAR_source_energy_nodes_not_candidate_table_interpolation",
        "normative_export_allowed": False, "physical_accuracy_verified": False,
        "physical_mass_policy_qualified": False, "production_tables_replaced": False,
        "fortran_executed": False, "absolute_range_origin_verified": False,
        "limitations": ["Synthetic fixture; not native execution."], "rows": rows,
        "materials": [{"z": z, "atomic_mass_u": 27.0 if z == 13 else 28.0,
            "density_g_cm3": 2.7 if z == 13 else 2.33, "ionization_potential_override_ev": 0.,
            "ionization_potential_policy": "unchanged_backend_internal_elemental_table",
            "elemental_ionization_potential_ev": 166.0 if z == 13 else 173.0}
            for z in (13, 14)],
        "backend_configuration": dict(runner.CONFIG),
        "nonradiative_configuration": {**runner.CONFIG, "corrections": 8},
        "low_energy_electronic_closure_count": 2 * sum(e / 4 <= 10 for e in energies),
        "low_energy_electronic_closure_max_relative_error": 0.,
    }
    path = tmp_path / "synthetic_controls.json"
    _write(path, raw)
    return path, raw


def _load(runner: ModuleType, controls: tuple[Path, dict[str, Any]]) -> dict[str, Any]:
    path, _ = controls
    return dict(runner._native_controls(path, _sha(path), load_astar_archive(ARCHIVE).archive_sha256,
                                        _energies()))


def test_control_contract_preserves_whole_energy_mass_stopping_and_false_flags(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]],
) -> None:
    result = _load(runner, controls)
    assert len(result["rows"]) == 202
    for row in result["rows"]:
        assert row["energy_mev_per_nucleon"] == row["energy_mev_total"] / 4
        assert row["stopping_electronic_mev_cm2_g"] == 100 + row["energy_mev_total"] / 4
        assert row["range_g_cm2"] == row["energy_mev_total"] + 3
    assert result["physical_accuracy_verified"] is False
    assert result["normative_export_allowed"] is False


@pytest.mark.parametrize("field,value", [
    ("schema_version", "bad"), ("archive_sha256", "0" * 64), ("archive_version", "online"),
    ("projectile", {"z": 3, "mass_parameter_u": 4., "charge_input": 2}),
    ("projectile", {"z": 2, "mass_parameter_u": 4.001, "charge_input": 2}),
    ("projectile", {"z": 2, "mass_parameter_u": 4., "charge_input": 1}),
    ("energy_conversion", "energy_per_nucleon=ASTAR_total_energy"),
    ("energy_coordinate_compatibility", "precision_isotope_mass_verified"),
    ("range_origin_energy_mev_per_nucleon", 0.),
    ("material_atomic_numbers", [13, 15]), ("energies_mev_total", [0.04, 1000.]),
])
def test_control_physical_identity_and_units_guards(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], field: str, value: Any,
) -> None:
    path, raw = controls
    raw[field] = value
    _write(path, raw)
    with pytest.raises(ValueError):
        _load(runner, controls)


@pytest.mark.parametrize("flag", [
    "normative_export_allowed", "physical_accuracy_verified", "physical_mass_policy_qualified",
    "production_tables_replaced", "fortran_executed", "absolute_range_origin_verified",
])
def test_control_cannot_claim_qualification(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], flag: str,
) -> None:
    path, raw = controls
    raw[flag] = True
    _write(path, raw)
    with pytest.raises(ValueError, match="false"):
        _load(runner, controls)


@pytest.mark.parametrize("config,key,value", [
    ("backend_configuration", "z_effective", True),
    ("backend_configuration", "z_effective", 1.0),
    ("backend_configuration", "corrections", 8),
    ("nonradiative_configuration", "corrections", 0),
    ("backend_configuration", "scattering_factor", True),
    ("backend_configuration", "scattering_factor", 0.0),
])
def test_control_configuration_guards(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], config: str, key: str, value: Any,
) -> None:
    path, raw = controls
    raw[config][key] = value
    _write(path, raw)
    with pytest.raises(ValueError):
        _load(runner, controls)


@pytest.mark.parametrize("prefix", [
    "native_source_pins", "production_resources_sha256", "implementation_sha256",
])
def test_changed_control_generation_inputs_are_rejected(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], prefix: str,
) -> None:
    path, raw = controls
    raw[prefix + "_after"] = {**raw[prefix + "_after"], "not_an_input": "0" * 64}
    _write(path, raw)
    with pytest.raises(ValueError):
        _load(runner, controls)


@pytest.mark.parametrize("prefix", [
    "native_source_pins", "production_resources_sha256", "implementation_sha256",
])
def test_wrong_current_source_resource_or_implementation_hash_is_rejected(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], prefix: str,
) -> None:
    path, raw = controls
    broken = dict(raw[prefix + "_before"])
    broken[next(iter(broken))] = "0" * 64
    raw[prefix + "_before"] = broken
    raw[prefix + "_after"] = broken
    _write(path, raw)
    with pytest.raises(ValueError):
        _load(runner, controls)


@pytest.mark.parametrize("field,value", [
    ("material_z", 13.0), ("material_z", True), ("material_z", 14),
    ("energy_mev_total", 0.05), ("energy_mev_per_nucleon", 0.04),
    ("energy_mev_per_nucleon", True), ("stopping_electronic_mev_cm2_g", 0.),
    ("stopping_nuclear_mev_cm2_g", -1.), ("stopping_radiative_mev_cm2_g", 1.),
    ("range_g_cm2", 0.), ("range_g_cm2", "1"),
])
def test_native_row_identity_and_numeric_guards(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], field: str, value: Any,
) -> None:
    path, raw = controls
    raw["rows"][0][field] = value
    _write(path, raw)
    with pytest.raises(ValueError):
        _load(runner, controls)


@pytest.mark.parametrize("operation", ["missing", "extra", "order", "fields"])
def test_native_row_coverage(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], operation: str,
) -> None:
    path, raw = controls
    if operation == "missing":
        raw["rows"].pop()
    elif operation == "extra":
        raw["rows"].append(raw["rows"][0])
    elif operation == "order":
        raw["rows"].reverse()
    else:
        raw["rows"][0]["total_unknown_unit"] = 1
    _write(path, raw)
    with pytest.raises(ValueError):
        _load(runner, controls)


@pytest.mark.parametrize("digest", ["", "bad", "0" * 64, "A" * 64])
def test_native_file_requires_explicit_exact_lowercase_sha(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], digest: str,
) -> None:
    with pytest.raises(ValueError):
        runner._native_controls(controls[0], digest, load_astar_archive(ARCHIVE).archive_sha256, _energies())


@pytest.mark.parametrize("suffix", [",\"extra\":NaN}", ",\"extra\":1e999}",
                                    ",\"schema_version\":\"duplicate\"}"])
def test_json_nonfinite_and_duplicate_keys(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], suffix: str,
) -> None:
    path, raw = controls
    path.write_text(json.dumps(raw)[:-1] + suffix, encoding="utf-8")
    with pytest.raises(ValueError):
        _load(runner, controls)


def _candidate_rows(offset: float = 0.) -> tuple[ParameterizedStoppingRow, ...]:
    return tuple(ParameterizedStoppingRow(2, 4., 13, e, 100 + e, 1., 4 * e + 3 + offset)
                 for e in (.01, 1., 10., 250.))


def test_piecewise_linear_interpolation_is_not_log_interpolation_or_extra_mass_factor(
    runner: ModuleType,
) -> None:
    rows = _candidate_rows()
    for e in (.01, .125, 1., 4., 100., 250.):
        sampled = runner._sample_candidate(rows, e)
        assert sampled["stopping_electronic_mev_cm2_g"] == pytest.approx(100 + e, rel=1e-14)
        assert sampled["stopping_nuclear_mev_cm2_g"] == pytest.approx(1., rel=1e-14)
        assert sampled["stopping_total_mev_cm2_g"] == pytest.approx(101 + e, rel=1e-14)
        assert sampled["range_g_cm2"] == pytest.approx(4 * e + 3, rel=1e-14)


@pytest.mark.parametrize("rows,energy", [
    ((), 1.), ((_candidate_rows()[0],), .01), (_candidate_rows(), .009), (_candidate_rows(), 251.),
])
def test_no_extrapolation_or_single_node_interpolation(
    runner: ModuleType, rows: tuple[ParameterizedStoppingRow, ...], energy: float,
) -> None:
    with pytest.raises(ValueError):
        runner._sample_candidate(rows, energy)


def test_anchored_range_difference_removes_offset_not_absolute_comparison(runner: ModuleType) -> None:
    candidate = {"stopping_electronic_mev_cm2_g": 10., "stopping_nuclear_mev_cm2_g": 2.,
                 "stopping_total_mev_cm2_g": 12., "range_g_cm2": 20.}
    reference = {**candidate, "range_g_cm2": 10.}
    plain = runner._relative_metrics(candidate, reference, 4., 2.)
    shifted = runner._relative_metrics({**candidate, "range_g_cm2": 157.}, reference, 141., 2.)
    assert plain["range_g_cm2"] == 1.
    assert shifted["range_g_cm2"] == pytest.approx(14.7)
    assert plain["range_increment_from_0.04_mev"] == shifted["range_increment_from_0.04_mev"] == 1.
    anchor = runner._relative_metrics(candidate, reference, 20., 10.)
    assert anchor["range_increment_from_0.04_mev"] is None


def test_reference_total_and_csda_not_projected(runner: ModuleType) -> None:
    row = AstarRow(.04, 100., 2., .005, .5)
    result = runner._reference(row)
    assert result["stopping_total_mev_cm2_g"] == 102.
    assert result["range_g_cm2"] == .005
    assert result["range_g_cm2"] != row.range_projected_g_cm2


@pytest.mark.parametrize("material_id", [13, 14])
def test_live_and_archive_are_separate_and_missing_22_5_is_explicit(
    runner: ModuleType, material_id: int,
) -> None:
    archive = load_astar_archive(ARCHIVE)
    text = (ROOT / f"research/nist_star_20261031/live_astar_{material_id:03d}.html").read_text(
        encoding="utf-8",
    )
    result = runner._live_check(archive.material(material_id), text)
    assert result["archive_node_count"] == 122
    assert result["live_node_count"] == result["shared_node_count"] == 121
    assert result["archive_nodes_absent_from_live"] == [22.5]
    assert result["live_nodes_absent_from_archive"] == []
    assert result["comparison_count"] == 121 * 7
    assert result["online_version_equivalence_verified"] is False
    for check in result["outside_printed_rounding_cell"]:
        token = check["online_printed_token"]
        if "E" in token:
            precision = 0.5 * 10 ** (int(token.split("E")[1]) - 3)
        else:
            precision = .00005
        assert check["printed_half_unit"] == pytest.approx(precision, rel=1e-15)
        assert check["difference_over_printed_half_unit"] > 1


@pytest.fixture
def comparison_context(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    controls_path, control = controls
    paths = tmp_path / "coarse.json", tmp_path / "dense.json"
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"synthetic_fixture": True})
    for path, density in zip(paths, (100, 200), strict=True):
        grid = (.01, 1., 10., 250.) if density == 100 else (.01, .5, 1., 5., 10., 100., 250.)
        raw = {"manifest": {"synthetic_fixture": True}, "manifest_sha256": _sha(manifest),
               "manifest_path": str(manifest),
               "points_per_decade": density, "datasets": {runner.POLICY: {
                   "backend_configuration": control["backend_configuration"],
                   "nonradiative_configuration": control["nonradiative_configuration"],
                   "materials": [{k: m[k] for k in ("z", "atomic_mass_u", "density_g_cm3")}
                                 for m in control["materials"]],
                   "rows": [asdict(ParameterizedStoppingRow(2, 4., z, e, 100 + e, 1., 4 * e + 3))
                            for z in (13, 14) for e in grid],
               }}}
        _write(path, raw)

    def validation(path: Path) -> dict[str, Any]:
        return {"input_sha256_after": _sha(path),
                "production_resources_sha256_after": control["production_resources_sha256_after"]}

    monkeypatch.setattr(runner, "run_validator", validation)
    return *paths, controls_path


def _compare(runner: ModuleType, paths: tuple[Path, Path, Path]) -> dict[str, Any]:
    return dict(runner.compare(*paths, controls_sha256=_sha(paths[2])))


def test_full_synthetic_comparison_uses_total_energy_divided_by_four_and_no_density_multiplier(
    runner: ModuleType, comparison_context: tuple[Path, Path, Path],
) -> None:
    report = _compare(runner, comparison_context)
    assert len(report["comparisons"]) == 2
    for comparison in report["comparisons"]:
        assert comparison["node_count"] == 101
        for row in comparison["rows"]:
            total = row["energy_mev_total"]
            assert row["energy_mev_per_nucleon"] == total / 4
            for series in ("coarse", "dense", "native_exact"):
                assert row[series]["stopping_electronic_mev_cm2_g"] == pytest.approx(100 + total / 4)
                assert row[series]["stopping_total_mev_cm2_g"] == pytest.approx(101 + total / 4)
                assert row[series]["range_g_cm2"] == pytest.approx(total + 3)
            for value in row["dense_over_native_minus_one"].values():
                assert value is None or abs(value) < 1e-12
        assert comparison["rows"][0]["native_exact_over_astar_minus_one"]["range_increment_from_0.04_mev"] is None
        assert comparison["online_archive_comparison"]["archive_nodes_absent_from_live"] == [22.5]
    for flag in ("physical_accuracy_verified", "physical_mass_policy_qualified", "normative_export_allowed",
                 "production_tables_replaced", "fortran_execution_replayed", "physical_acceptance_tolerance_established"):
        assert report[flag] is False
    assert report["raw_absolute_range_origin_confounded"] is True
    assert report["implementation_sha256_before"] == report["implementation_sha256_after"]
    assert report["evidence_sha256_before"] == report["evidence_sha256_after"]


@pytest.mark.parametrize("operation", ["manifest", "density", "config", "material", "missing_He"])
def test_unmatched_candidates_rejected(
    runner: ModuleType, comparison_context: tuple[Path, Path, Path], operation: str,
) -> None:
    path = comparison_context[1]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if operation == "manifest":
        raw["manifest"] = {"different": True}
    elif operation == "density":
        raw["points_per_decade"] = 100
    elif operation == "config":
        raw["datasets"][runner.POLICY]["backend_configuration"]["low_energy"] = 1
    elif operation == "material":
        raw["datasets"][runner.POLICY]["materials"][0]["density_g_cm3"] = 100.
    else:
        for row in raw["datasets"][runner.POLICY]["rows"]:
            row["z"] = 3
    _write(path, raw)
    with pytest.raises(ValueError):
        _compare(runner, comparison_context)


def test_existing_output_refused_before_calculation(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")

    def not_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Calculation must not begin for an existing output.")

    monkeypatch.setattr(runner, "compare", not_called)
    monkeypatch.setattr(sys, "argv", ["compare_astar_helium", "coarse", "dense", "native",
                                      "--native-sha256", "0" * 64, "--output", str(output)])
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("value", [True, False, "1", 0., -1., inf, nan])
def test_number_contract(runner: ModuleType, value: Any) -> None:
    with pytest.raises(ValueError):
        runner._number(value)


def test_only_explicit_zero_allowed(runner: ModuleType) -> None:
    assert runner._number(0., zero_allowed=True) == 0.
    with pytest.raises(ValueError):
        runner._number(False, zero_allowed=True)


@pytest.mark.parametrize("value", [True, False, "1", 0., -1., inf, nan, 10**400])
def test_interpolation_energy_requires_finite_positive_number(runner: ModuleType, value: Any) -> None:
    with pytest.raises(ValueError):
        runner._sample_candidate(_candidate_rows(), value)


@pytest.mark.parametrize("operation", ["reverse", "duplicate", "different_z", "different_mass", "different_material"])
def test_interpolation_requires_one_exact_identity_and_ordered_nodes(
    runner: ModuleType, operation: str,
) -> None:
    rows = list(_candidate_rows())
    if operation == "reverse":
        rows.reverse()
    elif operation == "duplicate":
        rows[1] = rows[0]
    else:
        data = asdict(rows[1])
        data[{"different_z": "z", "different_mass": "mass_parameter_u",
              "different_material": "material_z"}[operation]] = {
            "different_z": 3, "different_mass": 4.1, "different_material": 14,
        }[operation]
        rows[1] = ParameterizedStoppingRow(**data)
    with pytest.raises(ValueError):
        runner._sample_candidate(tuple(rows), 1.)


@pytest.mark.parametrize("field", ["z", "charge_input"])
def test_native_projectile_integer_identity_not_float_equivalence(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], field: str,
) -> None:
    path, raw = controls
    raw["projectile"][field] = 2.0
    _write(path, raw)
    with pytest.raises(ValueError, match="integers"):
        _load(runner, controls)


@pytest.mark.parametrize("prefix", ["production_resources_sha256", "implementation_sha256"])
def test_native_pin_set_cannot_be_shortened_even_when_remaining_hashes_match(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], prefix: str,
) -> None:
    path, raw = controls
    shortened = dict(raw[prefix + "_before"])
    shortened.pop(next(iter(shortened)))
    raw[prefix + "_before"] = raw[prefix + "_after"] = shortened
    _write(path, raw)
    with pytest.raises(ValueError, match="coverage"):
        _load(runner, controls)


@pytest.mark.parametrize("field,value", [
    ("z", 13.0), ("z", 14), ("density_g_cm3", 0.), ("atomic_mass_u", "27"),
    ("elemental_ionization_potential_ev", 173.), ("ionization_potential_override_ev", True),
    ("ionization_potential_override_ev", 1.), ("ionization_potential_policy", "adjusted_to_NIST"),
])
def test_native_material_identity_and_no_silent_retuning(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]], field: str, value: Any,
) -> None:
    path, raw = controls
    raw["materials"][0][field] = value
    _write(path, raw)
    with pytest.raises(ValueError):
        _load(runner, controls)


def test_native_ranges_cannot_be_constant_or_decreasing(
    runner: ModuleType, controls: tuple[Path, dict[str, Any]],
) -> None:
    path, raw = controls
    raw["rows"][1]["range_g_cm2"] = raw["rows"][0]["range_g_cm2"]
    _write(path, raw)
    with pytest.raises(ValueError, match="increase strictly"):
        _load(runner, controls)


@pytest.mark.parametrize("target", ["coarse", "controls", "manifest"])
def test_input_changes_during_full_comparison_are_rejected(
    runner: ModuleType, comparison_context: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    if target == "manifest":
        changed = Path(json.loads(comparison_context[0].read_text(encoding="utf-8"))["manifest_path"])
    else:
        changed = comparison_context[0 if target == "coarse" else 2]
    original_live = runner._live_check
    altered = False

    def live_check(*args: Any, **kwargs: Any) -> Any:
        nonlocal altered
        result = original_live(*args, **kwargs)
        if not altered:
            changed.write_bytes(changed.read_bytes() + b" ")
            altered = True
        return result

    monkeypatch.setattr(runner, "_live_check", live_check)
    with pytest.raises(ValueError, match="changed"):
        _compare(runner, comparison_context)


def test_missing_archive_member_provenance_is_not_a_valid_comparison(
    runner: ModuleType, comparison_context: tuple[Path, Path, Path],
) -> None:
    path = comparison_context[2]
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["archive_member_sha256"].pop("ASTAR/FALPH")
    _write(path, raw)
    with pytest.raises(ValueError, match="member"):
        _compare(runner, comparison_context)


def test_exact_decimal_rounding_tie_is_not_an_online_discrepancy(runner: ModuleType) -> None:
    archive = load_astar_archive(ARCHIVE)
    response = (ROOT / "research/nist_star_20261031/live_astar_013.html").read_text(encoding="utf-8")
    result = runner._live_check(archive.material(13), response)
    assert result["outside_printed_rounding_cell"] == []
    assert result["maximum_difference_over_printed_half_unit"]["difference_over_printed_half_unit"] <= 1
