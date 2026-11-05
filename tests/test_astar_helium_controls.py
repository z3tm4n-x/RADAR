"""Exact-energy control generator tests without loading the native extension."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(scripts))
        spec = importlib.util.spec_from_file_location(
            "astar_helium_controls_test", scripts / "generate_astar_helium_controls.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class FakeConfig:
    def get(self) -> dict[str, Any]:
        return dict(vars(self))


class FakeProjectile:
    def __init__(self, a: float, z: int, q: int) -> None:
        self.a, self.z, self.q = a, z, q
        self.energy = 0.0
        self.energies: list[float] = []

    def T(self, energy: float) -> None:
        self.energy = energy
        self.energies.append(energy)


class FakeMaterial:
    def __init__(self, z: int) -> None:
        self.target = SimpleNamespace(A=27.0 if z == 13 else 28.0, Z=z, stn=1)
        self.override: Any = 0.0
        self.rho: Any = 2.7 if z == 13 else 2.33

    def get_element(self, index: int) -> Any:
        assert index == 0
        return self.target

    def ncomponents(self) -> int:
        return 1

    def density(self) -> Any:
        return self.rho

    def I(self) -> Any:  # noqa: E743 - mirrors the native material API
        return self.override


class FakeBackend:
    Config = FakeConfig

    def __init__(self, native: ModuleType) -> None:
        self.native = native
        self.projectiles: list[FakeProjectile] = []
        self.materials = {z: FakeMaterial(z) for z in (13, 14)}
        self.mode = "normal"

    def Projectile(self, a: float, z: int, q: int) -> FakeProjectile:
        projectile = FakeProjectile(a, z, q)
        self.projectiles.append(projectile)
        return projectile

    def get_material(self, z: int) -> FakeMaterial:
        return self.materials[z]

    def dedx(self, projectile: FakeProjectile, material: FakeMaterial, config: Any) -> Any:
        if self.mode == "nan_total":
            return float("nan")
        if self.mode == "negative_total":
            return -1.0
        if self.mode == "bool_total":
            return True
        if self.mode == "configuration_mutation":
            config.low_energy = 2
        nuclear = self.native.nuclear_stopping(
            2, 4.0, material.target.Z, material.target.A, projectile.energy,
        )
        if self.mode == "negative_electronic":
            return nuclear / 2
        extra = 1.0 if self.mode == "radiative" and config.corrections == 0 else 0.0
        return nuclear + 100 + projectile.energy + extra

    def range(self, projectile: FakeProjectile, material: FakeMaterial, config: Any) -> Any:
        if self.mode == "range_nan":
            return float("nan")
        if self.mode == "range_zero":
            return 0.0
        if self.mode == "range_constant":
            return 1.0
        if self.mode == "material_mutation":
            material.rho += 0.01
        return 4 * projectile.energy + 3

    def sezi_dedx_e(self, projectile: FakeProjectile, material: FakeMaterial, config: Any) -> Any:
        if self.mode == "closure_nan":
            return float("nan")
        return 100 + projectile.energy + (1 if self.mode == "closure_error" else 0)

    def catima_info(self) -> str:
        return "synthetic_backend_not_native_execution"


def _samples(generator: ModuleType, backend: FakeBackend) -> dict[str, Any]:
    return dict(generator.generate_samples(
        backend, (0.04, 4.0, 40.0, 1000.0), elemental_i_ev={13: 166.0, 14: 173.0},
    ))


def test_exact_total_to_per_mass_mapping_no_stopping_multiplier(generator: ModuleType) -> None:
    backend = FakeBackend(generator.native)
    result = _samples(generator, backend)
    assert len(result["rows"]) == 8
    assert len(backend.projectiles) == 2
    for projectile in backend.projectiles:
        assert (projectile.a, projectile.z, projectile.q) == (4.0, 2, 2)
        assert projectile.energies == [0.01, 1.0, 10.0, 250.0]
    for row in result["rows"]:
        energy = row["energy_mev_total"] / 4
        assert row["energy_mev_per_nucleon"] == energy
        assert row["stopping_electronic_mev_cm2_g"] == pytest.approx(100 + energy)
        assert row["stopping_radiative_mev_cm2_g"] == 0.0
        assert row["range_g_cm2"] == 4 * energy + 3
    assert result["low_energy_electronic_closure_count"] == 6
    assert result["low_energy_electronic_closure_max_relative_error"] < 1e-14
    assert result["backend_configuration"]["corrections"] == 0
    assert result["nonradiative_configuration"]["corrections"] == 8
    assert all(m["ionization_potential_override_ev"] == 0 for m in result["materials"])


@pytest.mark.parametrize("energies", [
    (), (0.04, 0.04), (1.0, 0.04), (0.039,), (1000.01,), (True,),
    (float("inf"),), (float("nan"),), ("1",), (1.0, "2"), [0.04, 4.0],
])
def test_invalid_energy_grid(generator: ModuleType, energies: Any) -> None:
    with pytest.raises(ValueError):
        generator.generate_samples(FakeBackend(generator.native), energies,
                                   elemental_i_ev={13: 166.0, 14: 173.0})


@pytest.mark.parametrize("mode", [
    "nan_total", "negative_total", "bool_total", "negative_electronic", "radiative",
    "range_nan", "range_zero", "range_constant", "closure_nan", "closure_error",
    "configuration_mutation", "material_mutation",
])
def test_invalid_backend_numbers_and_mutations(generator: ModuleType, mode: str) -> None:
    backend = FakeBackend(generator.native)
    backend.mode = mode
    with pytest.raises(ValueError):
        _samples(generator, backend)


@pytest.mark.parametrize("override", [1.0, True, float("nan"), -1.0, "0"])
def test_material_i_cannot_be_overridden(generator: ModuleType, override: Any) -> None:
    backend = FakeBackend(generator.native)
    backend.materials[13].override = override
    with pytest.raises(ValueError):
        _samples(generator, backend)


@pytest.mark.parametrize("values", [{13: 166.0}, {13: 166.0, 14: 0.0}, {13: True, 14: 173.0}])
def test_required_elemental_i_values(generator: ModuleType, values: Any) -> None:
    with pytest.raises(ValueError):
        generator.generate_samples(FakeBackend(generator.native), (0.04,), elemental_i_ev=values)


def test_wheel_header_values_are_read_without_setting_i(generator: ModuleType) -> None:
    assert generator._elemental_i_values() == {13: 166.0, 14: 173.0}


@pytest.fixture
def control_context(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, Path]:
    monkeypatch.setattr(generator.native, "load_backend", lambda: FakeBackend(generator.native))
    monkeypatch.setattr(generator.native, "verify_sources", lambda: {"native_fixture": "pin"})
    monkeypatch.setattr(generator.native, "production_hashes", lambda: {"production_fixture": "pin"})
    return generator, generator.ROOT / "research/nist_star_20261031/STAR.tar.gz"


def test_real_archive_nodes_with_fake_backend_and_qualification_false(
    control_context: tuple[ModuleType, Path],
) -> None:
    generator, archive = control_context
    report = generator.generate_controls(archive)
    assert len(report["energies_mev_total"]) == 101
    assert len(report["rows"]) == 202
    assert report["energies_mev_total"][0] == 0.04
    assert report["energies_mev_total"][-1] == 1000.0
    assert report["implementation_sha256_before"] == report["implementation_sha256_after"]
    assert report["archive_sha256"] == generator.native.sha256(archive)
    for flag in ("normative_export_allowed", "physical_accuracy_verified", "fortran_executed",
                 "physical_mass_policy_qualified", "production_tables_replaced",
                 "absolute_range_origin_verified"):
        assert report[flag] is False
    assert report["range_origin_energy_mev_per_nucleon"] == 0.001


@pytest.mark.parametrize("which", ["implementation", "production", "native", "archive"])
def test_input_changes_detected(
    control_context: tuple[ModuleType, Path], monkeypatch: pytest.MonkeyPatch, which: str,
) -> None:
    generator, archive = control_context
    states = iter(({"fixture": "before"}, {"fixture": "after"}))
    if which == "implementation":
        monkeypatch.setattr(generator, "_implementation_hashes", lambda: next(states))
    elif which == "production":
        monkeypatch.setattr(generator.native, "production_hashes", lambda: next(states))
    elif which == "native":
        monkeypatch.setattr(generator.native, "verify_sources", lambda: next(states))
    else:
        original = generator.native.sha256
        calls = 0

        def changed_sha(path: Path) -> str:
            nonlocal calls
            if path == archive:
                calls += 1
                if calls >= 2:
                    return "0" * 64
            return str(original(path))

        monkeypatch.setattr(generator.native, "sha256", changed_sha)
    with pytest.raises(ValueError, match="changed"):
        generator.generate_controls(archive)


def test_backend_verification_failure_propagates_before_load(
    control_context: tuple[ModuleType, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, archive = control_context

    def cache_error() -> None:
        raise ValueError("Bytecode caches are not allowed in the pinned backend runtime.")

    monkeypatch.setattr(generator.native, "verify_sources", cache_error)
    monkeypatch.setattr(generator.native, "load_backend", lambda: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="Bytecode caches"):
        generator.generate_controls(archive)


def test_bad_archive_rejected_before_native_load(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path / "wrong.tar.gz"
    path.write_bytes(b"not the pinned archive")
    monkeypatch.setattr(generator.native, "load_backend", lambda: pytest.fail("must not load"))
    with pytest.raises(ValueError):
        generator.generate_controls(path)


def test_report_exclusive_and_nonfinite_rejected(generator: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    generator.write_report({"normative_export_allowed": False}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"normative_export_allowed": False}
    with pytest.raises(FileExistsError):
        generator.write_report({}, path)
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        generator.write_report({"x": float("nan")}, invalid)
    assert not invalid.exists()


def test_main_checks_output_before_computation(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "exists.json"
    output.write_text("original", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["controls", "unused.tar.gz", "--output", str(output)])
    monkeypatch.setattr(generator, "generate_controls", lambda _p: pytest.fail("must not calculate"))
    with pytest.raises(FileExistsError):
        generator.main()
    assert output.read_text(encoding="utf-8") == "original"
