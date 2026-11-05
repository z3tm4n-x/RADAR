"""Isolated generator contract checks; no native extension is imported here."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(scripts))
        return _load("parameterized_generator_test", scripts / "generate_parameterized_stopping.py")


@pytest.fixture
def manifest(generator: ModuleType, tmp_path: Path) -> dict[str, Any]:
    builder = _load("mass_manifest_test_helper", generator.ROOT / "scripts/build_stopping_mass_manifest.py")
    payload = json.dumps({
        "schema_version": "radar.ion_stopping_candidate.raw.v1",
        "ions": [{"z": z, "mass_number": a, "symbol": symbol}
                 for z, a, symbol in generator.native.IONS],
    }).encode()
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(payload)
    result: dict[str, Any] = builder.build_manifest(
        candidate, expected_candidate_sha256=hashlib.sha256(payload).hexdigest(),
    )
    return result


def _write(path: Path, value: Any) -> str:
    payload = json.dumps(value).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_pinned_manifest_preserves_literal_mass_and_separate_source_charge(
    generator: ModuleType, manifest: dict[str, Any], tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    result = generator.read_pinned_manifest(path, _write(path, manifest))
    assert len(result["projectiles"]) == 84
    fe = next(x for x in result["projectiles"] if x["z"] == 26)
    assert fe["mass_parameter_u"] == 55.8
    sep_fe = next(x for x in result["source_rows"] if x["source"] == "SEP" and x["z"] == 26)
    assert sep_fe["source_mass_to_charge"] == 4.12
    assert not result["normative_export_allowed"]


@pytest.mark.parametrize("pin", ["", "0" * 63, "A" * 64, "g" * 64, "0" * 64])
def test_manifest_pin_is_required_and_checked(
    generator: ModuleType, manifest: dict[str, Any], tmp_path: Path, pin: str,
) -> None:
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    with pytest.raises(ValueError):
        generator.read_pinned_manifest(path, pin)


@pytest.mark.parametrize("payload", [
    b'{"schema_version":"x","schema_version":"y"}',
    b'{"value":NaN}', b'{"value":Infinity}', b'{"value":1e999}', b'[]',
])
def test_malformed_json_is_rejected_even_with_matching_hash(
    generator: ModuleType, tmp_path: Path, payload: bytes,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        generator.read_pinned_manifest(path, hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("field,value", [
    ("z", True), ("z", 2.0), ("z", 1), ("z", 93),
    ("mass_parameter_u", True), ("mass_parameter_u", "4"),
    ("mass_parameter_u", 0), ("mass_parameter_u", float("nan")),
    ("mass_parameter_u", float("inf")), ("symbol", ""), ("symbol", 4),
    ("source_membership", ["SEP"]), ("source_membership", ["SEP", "GCR"]),
])
def test_projectile_identity_rejects_invalid_values(
    generator: ModuleType, manifest: dict[str, Any], tmp_path: Path, field: str, value: Any,
) -> None:
    manifest["projectiles"][0][field] = value
    path = tmp_path / "manifest.json"
    with pytest.raises(ValueError):
        generator.read_pinned_manifest(path, _write(path, manifest))


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra_field", "pilot_fractional",
                                      "pilot_symbol", "claimed_release"])
def test_inventory_shape_and_release_guards(
    generator: ModuleType, manifest: dict[str, Any], tmp_path: Path, mutation: str,
) -> None:
    if mutation == "duplicate":
        manifest["projectiles"][1] = copy.deepcopy(manifest["projectiles"][0])
    elif mutation == "missing":
        manifest["projectiles"].pop()
    elif mutation == "extra_field":
        manifest["projectiles"][0]["charge"] = 2
    elif mutation == "pilot_fractional":
        manifest["pilot_isotopes"][0]["mass_parameter_u"] = 4.1
    elif mutation == "pilot_symbol":
        manifest["pilot_isotopes"][0]["symbol"] = "Fe"
    else:
        manifest["normative_export_allowed"] = True
    path = tmp_path / "manifest.json"
    with pytest.raises(ValueError):
        generator.read_pinned_manifest(path, _write(path, manifest))


@pytest.mark.parametrize("mutation", ["mismatch", "missing_required", "before_after", "empty",
                                      "traversal", "absolute", "internal_alias"])
def test_source_hash_inventory_and_path_boundaries(
    generator: ModuleType, manifest: dict[str, Any], tmp_path: Path, mutation: str,
) -> None:
    before = manifest["source_files_sha256_before"]
    first = next(iter(before))
    if mutation == "mismatch":
        before[first] = "0" * 64
    elif mutation == "missing_required":
        before.pop(first)
    elif mutation == "before_after":
        manifest["source_files_sha256_after"] = {}
    elif mutation == "empty":
        before.clear()
    else:
        alias = {
            "traversal": "../escape.txt",
            "absolute": str((tmp_path / "outside.txt").resolve()),
            "internal_alias": "src/radar/gcr/../gcr/parameters.py",
        }[mutation]
        before[alias] = before.pop(first)
    if mutation != "before_after":
        manifest["source_files_sha256_after"] = before.copy()
    path = tmp_path / "manifest.json"
    with pytest.raises(ValueError):
        generator.read_pinned_manifest(path, _write(path, manifest))


@pytest.mark.parametrize("density", [50, 100, 200, 400])
def test_output_grid_is_strict_and_retains_transition_nodes(generator: ModuleType, density: int) -> None:
    grid = generator.experiment_grid(density)
    assert grid[0] == 0.01 and grid[-1] == 100000.0
    assert {10.0, 30.0, 300.0, 50000.0}.issubset(grid)
    assert all(a < b for a, b in zip(grid, grid[1:]))


def test_grid_refinement_is_nested_without_claiming_native_grid_refinement(generator: ModuleType) -> None:
    assert set(generator.experiment_grid(100)) <= set(generator.experiment_grid(200))
    assert set(generator.experiment_grid(200)) <= set(generator.experiment_grid(400))


@pytest.mark.parametrize("density", [True, 0, -1, 100.0, 1000])
def test_output_grid_rejects_invalid_density(generator: ModuleType, density: Any) -> None:
    with pytest.raises(ValueError):
        generator.experiment_grid(density)


class FakeConfig:
    def __init__(self) -> None:
        self.z_effective = 0
        self.corrections = 0
        self.calculation = 0
        self.low_energy = 0
        self.scattering = 0
        self.scattering_factor = 0.0

    def get(self) -> dict[str, Any]:
        return vars(self).copy()


class FakeProjectile:
    def __init__(self, a: float, z: int, q: int) -> None:
        self.a, self.z, self.q, self.energy = a, z, q, 0.0

    def T(self, energy: float) -> None:
        self.energy = energy


class FakeMaterial:
    def __init__(self, z: int) -> None:
        self.z = z

    def get_element(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(A=55.845 if self.z == 26 else float(self.z * 2), Z=self.z, stn=1)

    def ncomponents(self) -> int:
        return 1

    def density(self) -> float:
        return 2.0


class FakeBackend:
    Config = FakeConfig

    def __init__(self, fault: str = "") -> None:
        self.fault = fault
        self.projectiles: list[FakeProjectile] = []

    def Projectile(self, a: float, z: int, q: int) -> FakeProjectile:
        projectile = FakeProjectile(a, z, q)
        self.projectiles.append(projectile)
        return projectile

    def get_material(self, z: int) -> FakeMaterial:
        return FakeMaterial(z)

    def dedx(self, p: FakeProjectile, _m: FakeMaterial, c: FakeConfig) -> float:
        if self.fault == "total_nan":
            return float("nan")
        if self.fault == "negative_electronic":
            return 1.0
        electronic = 100.0 + p.energy / 1000.0
        radiation = 5.0 if p.energy > 50000.0 and c.corrections == 0 else 0.0
        if self.fault == "negative_radiation" and c.corrections == 0:
            radiation = -1.0
        return electronic + 2.0 + radiation

    def range(self, p: FakeProjectile, _m: FakeMaterial, _c: FakeConfig) -> float:
        return 0.0 if self.fault == "zero_range" else p.a * p.energy / 100.0

    def sezi_dedx_e(self, p: FakeProjectile, _m: FakeMaterial, _c: FakeConfig) -> float:
        if self.fault == "zero_check":
            return 0.0
        return 101.0 if self.fault == "failed_closure" else 100.0 + p.energy / 1000.0

    def dedx_from_range(self, p: FakeProjectile, m: FakeMaterial, c: FakeConfig) -> float:
        return -1.0 if self.fault == "negative_derivative" else self.dedx(p, m, c)

    def catima_info(self) -> str:
        return "TEST_BACKEND_NOT_NATIVE"


@pytest.fixture
def backend(generator: ModuleType, monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    monkeypatch.setattr(generator.native, "nuclear_stopping", lambda *_args: 2.0)
    return FakeBackend()


def test_generation_preserves_fractional_mass_and_separates_stopping_terms(
    generator: ModuleType, backend: FakeBackend,
) -> None:
    report = generator.generate_dataset(
        backend, [{"z": 26, "symbol": "Fe", "mass_parameter_u": 55.8}],
        (1.0, 10.0, 50000.0, 100000.0), generator.POLICIES[0],
    )
    assert all(p.a == 55.8 and p.q == p.z == 26 for p in backend.projectiles)
    assert len(report["rows"]) == 8
    for row in report["rows"]:
        e = row["energy_mev_per_nucleon"]
        assert row["mass_parameter_u"] == 55.8
        assert row["stopping_electronic_mev_cm2_g"] == pytest.approx(100.0 + e / 1000.0)
        assert row["stopping_nuclear_mev_cm2_g"] == 2.0
        assert row["stopping_radiative_mev_cm2_g"] == (5.0 if e > 50000.0 else 0.0)
    diagnostic = report["mass_diagnostics"][0]
    assert diagnostic["backend_reference_mass_u"] == 55.845
    assert diagnostic["ls_mass_linear_weight"] == pytest.approx((55.8 / 55.845 - 1) / 0.05)
    assert diagnostic["ls_mass_extrapolation"]
    assert not diagnostic["physical_isotope_assignment_verified"]
    assert not report["normative_export_allowed"]
    assert report["generation_checks"]["low_energy_closure_count"] == 4
    assert report["generation_checks"]["range_derivative_worst"]["relative_error"] == 0.0
    full, nonrad = report["backend_configuration"], report["nonradiative_configuration"]
    assert full["corrections"] == 0 and nonrad["corrections"] == 8
    assert {k: v for k, v in full.items() if k != "corrections"} == {
        k: v for k, v in nonrad.items() if k != "corrections"
    }


@pytest.mark.parametrize("fault", ["total_nan", "negative_electronic", "negative_radiation",
                                  "zero_range", "zero_check", "failed_closure", "negative_derivative"])
def test_bad_backend_outputs_are_rejected_without_clipping(
    generator: ModuleType, backend: FakeBackend, fault: str,
) -> None:
    backend.fault = fault
    with pytest.raises(ValueError):
        generator.generate_dataset(
            backend, [{"z": 26, "symbol": "Fe", "mass_parameter_u": 55.8}],
            (1.0, 100000.0), generator.POLICIES[0],
        )


def test_unknown_mass_policy_rejected(generator: ModuleType, backend: FakeBackend) -> None:
    with pytest.raises(ValueError, match="Unknown mass"):
        generator.generate_dataset(backend, [], (), "infer_isotope_by_rounding")


def test_cli_uses_explicit_policies_and_preserves_nonqualification(
    generator: ModuleType, manifest: dict[str, Any], backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path, output = tmp_path / "manifest.json", tmp_path / "generated.json"
    pin = _write(path, manifest)
    monkeypatch.setattr(sys, "argv", ["generator", "--manifest", str(path),
                                    "--manifest-sha256", pin, "--output", str(output)])
    monkeypatch.setattr(generator, "experiment_grid", lambda _density: (1.0, 100000.0))
    monkeypatch.setattr(generator.native, "verify_sources", lambda: {"test": "pinned"})
    monkeypatch.setattr(generator.native, "production_hashes", lambda: {})
    monkeypatch.setattr(generator.native, "load_backend", lambda: backend)
    generator.main()
    result = json.loads(output.read_bytes())
    assert set(result["datasets"]) == set(generator.POLICIES)
    assert len(result["datasets"][generator.POLICIES[0]]["rows"]) == 84 * 2 * 2
    assert len(result["datasets"][generator.POLICIES[1]]["rows"]) == 15 * 2 * 2
    for field in ("normative_export_allowed", "absolute_range_origin_verified",
                  "physical_mass_policy_qualified", "production_tables_replaced"):
        assert result[field] is False
    assert result["range_origin_energy_mev_per_nucleon"] == 0.001
    assert "assumed" in result["energy_coordinate_compatibility"]
    assert "not SEP A/Q" in result["projectile_charge_input"]
    assert result["source_inventory_independently_reconstructed_by_generator"] is False
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        generator.main()
    assert output.read_bytes() == before


@pytest.mark.parametrize("changed", ["native_pins", "production_hashes"])
def test_changed_inputs_abort_before_output_write(
    generator: ModuleType, manifest: dict[str, Any], backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str,
) -> None:
    path, output = tmp_path / "manifest.json", tmp_path / "generated.json"
    pin = _write(path, manifest)
    monkeypatch.setattr(sys, "argv", ["generator", "--manifest", str(path),
                                    "--manifest-sha256", pin, "--output", str(output)])
    monkeypatch.setattr(generator, "experiment_grid", lambda _density: (1.0, 100000.0))
    monkeypatch.setattr(generator.native, "load_backend", lambda: backend)
    monkeypatch.setattr(generator.native, "verify_sources", lambda: {"test": "pinned"})
    monkeypatch.setattr(generator.native, "production_hashes", lambda: {"test": "pinned"})
    states = iter(({"test": "pinned"}, {"test": "changed"}))
    method = "verify_sources" if changed == "native_pins" else "production_hashes"
    monkeypatch.setattr(generator.native, method, lambda: next(states))
    with pytest.raises(ValueError, match="changed during generation"):
        generator.main()
    assert not output.exists()


def test_manifest_output_alias_cannot_overwrite_input(
    generator: ModuleType, manifest: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    pin = _write(path, manifest)
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["generator", "--manifest", str(path),
                                    "--manifest-sha256", pin, "--output", str(path)])
    with pytest.raises(FileExistsError):
        generator.main()
    assert path.read_bytes() == before
