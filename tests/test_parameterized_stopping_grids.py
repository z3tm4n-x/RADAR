"""Grid-only and historical-control comparisons with independent analytic tables."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from radar.validation.parameterized_stopping import (
    ParameterizedStoppingRow, validate_parameterized_rows,
)


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/compare_parameterized_stopping_grids.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("parameterized_grid_runner", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _change(path: Path, operation: Any) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    operation(raw)
    _write(path, raw)


@pytest.fixture
def tables(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    paths = (tmp_path / "left.json", tmp_path / "right.json", tmp_path / "legacy.json")
    materials = [{"z": z, "atomic_mass_u": 27.0 if z == 13 else 28.0, "density_g_cm3": 2.0}
                 for z in (13, 14)]
    for path, energies in zip(paths[:2], ((1.0, 4.0), (1.0, 2.0, 4.0)), strict=True):
        raw: dict[str, Any] = {
            "manifest": {"fixture": True}, "manifest_sha256": "manifest_hash",
            "material_atomic_numbers": [13, 14], "backend_info_literal": "test_backend",
            "range_origin_energy_mev_per_nucleon": 0.001,
            "energy_coordinate_compatibility": "test_only", "projectile_charge_input": "Z",
            "let_convention": "electronic", "python": "test", "source_pins": {"source": "hash"},
            "energies_mev_per_nucleon": energies, "datasets": {},
        }
        for policy, mass in zip(runner.POLICIES, (4.2, 4.0), strict=True):
            raw["datasets"][policy] = {
                "mass_policy": policy,
                "projectiles": [{"z": 2, "mass_parameter_u": mass, "symbol": "He"}],
                "materials": materials, "backend_configuration": {"mode": 1},
                "nonradiative_configuration": {"mode": 2}, "mass_diagnostics": [],
                "rows": [asdict(ParameterizedStoppingRow(2, mass, m, e, mass, 0, e))
                         for m in (13, 14) for e in energies],
            }
        _write(path, raw)
    right = json.loads(paths[1].read_text(encoding="utf-8"))
    pilot = right["datasets"][runner.POLICIES[1]]
    legacy = {
        "schema_version": "radar.ion_stopping_candidate.raw.v1",
        "ions": [{"z": 2, "mass_number": 4, "symbol": "He"}],
        "material_atomic_numbers": [13, 14], "energies_mev_per_nucleon": [1.0, 2.0, 4.0],
        "backend_info_literal": "test_backend", "backend_configuration": {"mode": 1},
        "nonradiative_configuration": {"mode": 2}, "source_provenance": {"hashes": {"source": "hash"}},
        "production_resources_sha256": {"deliberately_historical": "not_checked_against_current"},
        "materials": [{**m, "ionization_potential_override_ev": 0.0,
                       "ionization_potential_policy": "pinned_internal_element_table"} for m in materials],
        "rows": [{"mass_number": row["mass_parameter_u"],
                  **{key: value for key, value in row.items() if key != "mass_parameter_u"}}
                 for row in pilot["rows"]],
    }
    _write(paths[2], legacy)

    def analytic_validator(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        validations = {}
        for policy in runner.POLICIES:
            dataset = data["datasets"][policy]
            validations[policy] = asdict(validate_parameterized_rows(
                [ParameterizedStoppingRow(**row) for row in dataset["rows"]],
                required_projectiles=[(ion["z"], ion["mass_parameter_u"])
                                      for ion in dataset["projectiles"]],
                material_atomic_numbers=(13, 14),
                energy_nodes_mev_per_nucleon=data["energies_mev_per_nucleon"],
            ))
        return {
            "input_sha256_before": _sha(path), "input_sha256_after": _sha(path),
            "production_resources_sha256_after": {"fixture": "same"},
            "generator_hashes_after": {"generator": "same"}, "source_pins_after": {"source": "hash"},
            "manifest_sha256": data["manifest_sha256"], "dataset_validations": validations,
        }

    monkeypatch.setattr(runner, "run_validator", analytic_validator)
    return paths


def _compare(runner: ModuleType, tables: tuple[Path, Path, Path]) -> dict[str, Any]:
    return dict(runner.compare_grids(*tables[:2], legacy_pilot=tables[2],
                                    legacy_pilot_sha256=_sha(tables[2])))


def test_nested_exact_control_and_historical_hash_scope(
    runner: ModuleType, tables: tuple[Path, Path, Path],
) -> None:
    report = _compare(runner, tables)
    for comparison in report["dataset_comparisons"].values():
        assert comparison["shared_row_count"] == 4
        assert comparison["shared_rows_exactly_equal"]
        assert comparison["max_absolute_relative_change"] < 1e-15
        assert comparison["table_quadrature_grid_check_passed"]
    assert [r["shared_row_count"] for r in report["historical_pilot_controls"]] == [4, 6]
    assert all(not r["historical_production_hashes_checked_against_current"]
               for r in report["historical_pilot_controls"])
    assert report["left_input_sha256_before"] == report["left_input_sha256_after"] == _sha(tables[0])
    assert report["historical_pilot_sha256_after"] == _sha(tables[2])
    assert not report["normative_export_allowed"]
    assert not report["uniform_local_convergence_established"]
    assert not report["spectral_grid_convergence_established"]


def test_relative_change_has_left_denominator_and_no_local_override(
    runner: ModuleType, tables: tuple[Path, Path, Path],
) -> None:
    policy = runner.POLICIES[0]

    def peak(raw: dict[str, Any]) -> None:
        for row in raw["datasets"][policy]["rows"]:
            if row["energy_mev_per_nucleon"] == 2:
                row["stopping_electronic_mev_cm2_g"] *= 10

    _change(tables[1], peak)
    result = _compare(runner, tables)["dataset_comparisons"][policy]
    assert not result["right_local_consistency_passed"]
    assert not result["table_quadrature_grid_check_passed"]
    for group in result["groups"]:
        assert group["signed_relative_change"] == pytest.approx(
            group["right_integrated_delta_range_g_cm2"] / group["left_integrated_delta_range_g_cm2"] - 1
        )
        assert group["signed_relative_change"] < 0


@pytest.mark.parametrize("field", ["backend_configuration", "materials", "mass_diagnostics"])
def test_different_non_grid_settings_rejected(
    runner: ModuleType, tables: tuple[Path, Path, Path], field: str,
) -> None:
    _change(tables[1], lambda raw: raw["datasets"][runner.POLICIES[0]].update({field: "different"}))
    with pytest.raises(ValueError, match="Different"):
        _compare(runner, tables)


def test_shared_node_mismatch_rejected(runner: ModuleType, tables: tuple[Path, Path, Path]) -> None:
    _change(tables[1], lambda raw: raw["datasets"][runner.POLICIES[0]]["rows"][0].update(
        {"range_g_cm2": 0.5}))
    with pytest.raises(ValueError, match="Shared-node"):
        _compare(runner, tables)


@pytest.mark.parametrize("reverse", [False, True])
def test_identical_or_reverse_grids_rejected(
    runner: ModuleType, tables: tuple[Path, Path, Path], reverse: bool,
) -> None:
    pair = (tables[1], tables[0]) if reverse else (tables[0], tables[0])
    with pytest.raises(ValueError, match="strictly refine"):
        runner.compare_grids(*pair, legacy_pilot=tables[2], legacy_pilot_sha256=_sha(tables[2]))


@pytest.mark.parametrize("field,value", [
    ("stopping_electronic_mev_cm2_g", 99.0), ("range_g_cm2", 0.9),
])
def test_changed_legacy_physical_control_rejected(
    runner: ModuleType, tables: tuple[Path, Path, Path], field: str, value: float,
) -> None:
    _change(tables[2], lambda raw: raw["rows"][0].update({field: value}))
    with pytest.raises(ValueError, match="Historical pilot physical"):
        _compare(runner, tables)


@pytest.mark.parametrize("pin", ["0" * 64, "ABC", None])
def test_wrong_historical_pin_rejected(
    runner: ModuleType, tables: tuple[Path, Path, Path], pin: Any,
) -> None:
    with pytest.raises(ValueError):
        runner.compare_grids(*tables[:2], legacy_pilot=tables[2], legacy_pilot_sha256=pin)


def test_post_validation_input_mutation_rejected(
    runner: ModuleType, tables: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runner.run_validator

    def mutate(path: Path) -> Any:
        report = original(path)
        path.write_bytes(path.read_bytes() + b" ")
        return report

    monkeypatch.setattr(runner, "run_validator", mutate)
    with pytest.raises(ValueError, match="changed"):
        _compare(runner, tables)


def test_cli_refuses_overwrite(runner: ModuleType, tables: tuple[Path, Path, Path]) -> None:
    output = tables[0].parent / "comparison.json"
    args = [str(tables[0]), str(tables[1]), "--legacy-pilot", str(tables[2]),
            "--legacy-pilot-sha256", _sha(tables[2]), "--output", str(output)]
    assert runner.main(args) == 0
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        runner.main(args)
    assert output.read_bytes() == before
