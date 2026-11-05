"""Tests do not require installing or loading the optional native backend."""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import zipfile

import pytest


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/generate_catima_candidate.py"
    spec = importlib.util.spec_from_file_location("catima_candidate_generator_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("density", [50, 100, 200, 400])
def test_grid_domain_transitions_and_order(generator: ModuleType, density: int) -> None:
    energies = generator.energy_grid(density)
    assert energies[0] == 0.01 and energies[-1] == 100000
    assert {10, 30, 50000}.issubset(energies)
    assert all(a < b for a, b in zip(energies, energies[1:]))


@pytest.mark.parametrize("bad", [True, 0, -1, 100.0, 10000])
def test_grid_rejects_invalid_density(generator: ModuleType, bad: Any) -> None:
    with pytest.raises(ValueError):
        generator.energy_grid(bad)


def test_grid_refinement_preserves_nodes(generator: ModuleType) -> None:
    assert set(generator.energy_grid(100)) <= set(generator.energy_grid(200))


@pytest.mark.parametrize("total_energy,expected,tolerance", [
    (0.1, 14.27, 1), (1, 2.161, 0.1), (10, 0.2874, 0.01), (100, 0.03455, 0.001),
])
def test_author_nuclear_regression_points(
    generator: ModuleType, total_energy: float, expected: float, tolerance: float,
) -> None:
    # CATIMA v1.91 tests/test_calculations.cpp: helium in elemental carbon.
    actual = generator.nuclear_stopping(2, 4.00151, 6, 12.0107, total_energy / 4.00151)
    assert actual == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("bad", [True, -1, 0, float("nan"), float("inf"), "1"])
@pytest.mark.parametrize("index", range(5))
def test_nuclear_input_boundaries(generator: ModuleType, bad: Any, index: int) -> None:
    args: list[Any] = [26, 56.0, 13, 26.9815, 100.0]
    args[index] = bad
    with pytest.raises(ValueError):
        generator.nuclear_stopping(*args)


def test_mass_policy_is_explicit_not_source_rounding(generator: ModuleType) -> None:
    assert len(generator.IONS) == 15
    assert (28, 58, "Ni") in generator.IONS
    assert len({ion[0] for ion in generator.IONS}) == 15
    assert all(type(a) is int for _, a, _ in generator.IONS)


def test_configuration_separates_only_radiative_flag(generator: ModuleType) -> None:
    backend = SimpleNamespace(Config=SimpleNamespace)
    full = vars(generator.configuration(backend))
    electronic = vars(generator.configuration(backend, no_radiative=True))
    assert full["z_effective"] == 1
    assert full["corrections"] == 0 and electronic["corrections"] == 8
    assert {k: v for k, v in full.items() if k != "corrections"} == {
        k: v for k, v in electronic.items() if k != "corrections"
    }


def test_retained_source_and_runtime_hashes(generator: ModuleType) -> None:
    assert generator.verify_sources() == generator.PINS


def test_changed_source_is_rejected(generator: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator, "sha256", lambda _path: "0" * 64)
    with pytest.raises(ValueError, match="hash mismatch"):
        generator.verify_sources()


@pytest.mark.parametrize("unexpected", ["_ext/__init__.py", "__pycache__/__init__.pyc"])
def test_extra_runtime_file_rejected(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unexpected: str,
) -> None:
    wheel = tmp_path / generator.WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pycatima/__init__.py", "# fixture")
    package = tmp_path / "runtime/pycatima"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# fixture", encoding="utf-8")
    monkeypatch.setattr(generator, "SOURCE", tmp_path)
    monkeypatch.setattr(generator, "PINS", {
        generator.WHEEL: hashlib.sha256(wheel.read_bytes()).hexdigest(),
    })
    generator.verify_sources()
    (package / unexpected).parent.mkdir()
    (package / unexpected).write_text("# shadow", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected files|Bytecode caches"):
        generator.verify_sources()


def test_preloaded_extension_rejected(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generator.sys, "version_info", (3, 12))
    monkeypatch.setitem(generator.sys.modules, "pycatima._ext", ModuleType("pycatima._ext"))
    with pytest.raises(ValueError, match="previously imported"):
        generator.load_backend()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
@pytest.mark.parametrize("method", ["dedx_from_range", "sezi_dedx_e"])
def test_nonfinite_check_cannot_disappear_in_maximum(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, value: float, method: str,
) -> None:
    monkeypatch.setattr(generator, "energy_grid", lambda _n: (1.0, 2.0))
    monkeypatch.setattr(generator, "IONS", ((2, 4, "He"),))
    monkeypatch.setattr(generator, "nuclear_stopping", lambda *_args: 1.0)
    backend = SimpleNamespace(
        Config=lambda: SimpleNamespace(get=lambda: {}),
        Projectile=lambda *_args: SimpleNamespace(T=lambda _e: None),
        get_material=lambda z: SimpleNamespace(
            get_element=lambda _i: SimpleNamespace(Z=z, A=27, stn=1),
            ncomponents=lambda: 1, density=lambda: 2.7, I=lambda: 0,
        ),
        dedx=lambda *_args: 10.0,
        range=lambda *_args: 1.0,
        sezi_dedx_e=lambda *_args: 9.0,
        dedx_from_range=lambda *_args: 10.0,
    )
    setattr(backend, method, lambda *_args: value)
    with pytest.raises(ValueError, match="check"):
        generator.generate(backend, 100)
