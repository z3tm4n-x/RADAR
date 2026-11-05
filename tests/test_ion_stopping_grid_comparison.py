"""Matched-grid requirements and analytic candidate quadrature comparisons."""

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

from radar.validation.ion_stopping_candidate import IonStoppingRow


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/compare_ion_stopping_grids.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("ion_grid_comparison_runner", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "research/ion_stopping_20261031/source.txt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    resource = tmp_path / "resource.txt"
    resource.write_bytes(b"resource")
    paths = (tmp_path / "coarse.json", tmp_path / "fine.json")
    for path, energies in zip(paths, ((1, 4), (1, 2, 4)), strict=True):
        raw = {
            "schema_version": "radar.ion_stopping_candidate.raw.v1",
            "ions": [{"z": 2, "mass_number": 4, "symbol": "He"}],
            "material_atomic_numbers": [13, 14], "energies_mev_per_nucleon": energies,
            "rows": [asdict(IonStoppingRow(2, 4, m, e, 4, 0, e))
                     for m in (13, 14) for e in energies],
            "source_provenance": {
                "source_root": "research/ion_stopping_20261031",
                "hashes": {"source.txt": hashlib.sha256(source.read_bytes()).hexdigest()},
            },
            "production_resources_sha256": {
                "resource.txt": hashlib.sha256(resource.read_bytes()).hexdigest(),
            },
            "generation_checks": {"fixture": True}, "backend_configuration": {"mode": 1},
        }
        path.write_text(json.dumps(raw), encoding="utf-8")
    return paths


def _change(path: Path, transform: Any) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    transform(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_constant_stopping_grid_refinement_is_exact_but_not_normative(
    runner: ModuleType, pair: tuple[Path, Path],
) -> None:
    report = runner.compare_grids(*pair, repository_root=pair[0].parent)
    assert report["max_relative_change"] < 1e-15
    assert report["table_quadrature_grid_check_passed"]
    assert report["shared_row_count"] == 4
    assert report["coarse_local_consistency_passed"]
    assert report["fine_local_consistency_passed"]
    assert not report["normative_release_eligible"]
    assert not report["spectral_grid_convergence_established"]


def test_dense_grid_exposes_an_unresolved_stopping_peak(
    runner: ModuleType, pair: tuple[Path, Path],
) -> None:
    def peak(raw: dict[str, Any]) -> None:
        for row in raw["rows"]:
            if row["energy_mev_per_nucleon"] == 2:
                row["stopping_electronic_mev_cm2_g"] = 40
    _change(pair[1], peak)
    report = runner.compare_grids(*pair, repository_root=pair[0].parent)
    assert not report["table_quadrature_grid_check_passed"]
    assert not report["fine_local_consistency_passed"]
    assert report["max_relative_change"] > 0.5


def test_changed_shared_node_is_rejected(runner: ModuleType, pair: tuple[Path, Path]) -> None:
    _change(pair[1], lambda raw: raw["rows"][0].update({"range_g_cm2": 0.9}))
    with pytest.raises(ValueError, match="Shared-node"):
        runner.compare_grids(*pair, repository_root=pair[0].parent)


@pytest.mark.parametrize("field,value", [
    ("backend_configuration", {"mode": 2}),
    ("nonradiative_configuration", {"mode": 1}),
    ("backend_info_literal", "different binary"),
])
def test_changed_backend_configuration_is_rejected(
    runner: ModuleType, pair: tuple[Path, Path], field: str, value: Any,
) -> None:
    _change(pair[1], lambda raw: raw.update({field: value}))
    with pytest.raises(ValueError, match="Different"):
        runner.compare_grids(*pair, repository_root=pair[0].parent)


def test_changed_mass_is_not_a_grid_comparison(runner: ModuleType, pair: tuple[Path, Path]) -> None:
    def mass(raw: dict[str, Any]) -> None:
        raw["ions"][0]["mass_number"] = 5
        for row in raw["rows"]:
            row["mass_number"] = 5
    _change(pair[1], mass)
    with pytest.raises(ValueError, match="Different ions"):
        runner.compare_grids(*pair, repository_root=pair[0].parent)


def test_nonnested_grid_and_reversed_pair_are_rejected(
    runner: ModuleType, pair: tuple[Path, Path],
) -> None:
    with pytest.raises(ValueError, match="strictly refine"):
        runner.compare_grids(pair[1], pair[0], repository_root=pair[0].parent)
    def shift_endpoint(raw: dict[str, Any]) -> None:
        raw["energies_mev_per_nucleon"][-1] = 5
        for row in raw["rows"]:
            if row["energy_mev_per_nucleon"] == 4:
                row["energy_mev_per_nucleon"] = row["range_g_cm2"] = 5
    _change(pair[1], shift_endpoint)
    with pytest.raises(ValueError, match="strictly refine"):
        runner.compare_grids(*pair, repository_root=pair[0].parent)


def test_equal_grids_are_not_refinement(runner: ModuleType, pair: tuple[Path, Path]) -> None:
    with pytest.raises(ValueError, match="strictly refine"):
        runner.compare_grids(pair[0], pair[0], repository_root=pair[0].parent)


def test_mutated_file_after_replay_is_rejected(
    runner: ModuleType, pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runner.run_validator
    def replay(path: Path, **kwargs: Any) -> Any:
        report = original(path, **kwargs)
        path.write_bytes(path.read_bytes() + b" ")
        return report
    monkeypatch.setattr(runner, "run_validator", replay)
    with pytest.raises(ValueError, match="changed"):
        runner.compare_grids(*pair, repository_root=pair[0].parent)


def test_cli_output_is_exclusive_and_does_not_run_on_existing_file(
    runner: ModuleType, pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = pair[0].parent / "comparison.json"
    original = runner.compare_grids
    monkeypatch.setattr(runner, "compare_grids",
                        lambda a, b: original(a, b, repository_root=pair[0].parent))
    assert runner.main([str(pair[0]), str(pair[1]), "--output", str(output)]) == 0
    monkeypatch.setattr(runner, "compare_grids", lambda *args: pytest.fail("should not run"))
    with pytest.raises(SystemExit):
        runner.main([str(pair[0]), str(pair[1]), "--output", str(output)])
