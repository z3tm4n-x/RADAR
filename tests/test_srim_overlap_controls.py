"""Synthetic exact-node stopping controls; no native extension is loaded."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
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
            "srim_overlap_generator_test", scripts / "generate_srim_overlap_controls.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def request(z: int = 2, mass: str = "4.0", target: int = 13, total: str = "1",
            index: int = 0) -> dict[str, Any]:
    return {"z": z, "mass_parameter_u_decimal": mass, "material_z": target,
            "energy_total_mev_decimal": total,
            "energy_mev_per_nucleon": float(Fraction(Decimal(total)) / Fraction(Decimal(mass))),
            "native_row_index": index, "raw_sha256": "a" * 64, "case_name": f"z{z}_m{target}"}


REQUESTS = [request(), request(total="40", index=1), request(total="12000", index=2),
            request(target=14), request(26, "55.8", total="1000")]


class FakeConfig:
    def get(self) -> dict[str, Any]:
        return dict(vars(self))


class FakeProjectile:
    def __init__(self, mass: float, z: int, q: int) -> None:
        self.mass, self.z, self.q = mass, z, q
        self.energy = 0.
        self.energies: list[float] = []

    def T(self, energy: float) -> None:
        self.energy = energy
        self.energies.append(energy)


class FakeMaterial:
    def __init__(self, z: int) -> None:
        self.target = SimpleNamespace(Z=z, A=26.9815 if z == 13 else 28.0855, stn=1)
        self.rho, self.override = (2.699 if z == 13 else 2.33), 0.

    def ncomponents(self) -> int:
        return 1

    def get_element(self, index: int) -> Any:
        assert index == 0
        return self.target

    def density(self) -> float:
        return self.rho

    def I(self) -> float:  # noqa: E743 - native method name
        return self.override


class FakeBackend:
    Config = FakeConfig

    def __init__(self, native: ModuleType, mode: str = "normal") -> None:
        self.native, self.mode = native, mode
        self.materials = {z: FakeMaterial(z) for z in (13, 14)}
        self.projectiles: list[FakeProjectile] = []

    def get_material(self, z: int) -> FakeMaterial:
        return self.materials[z]

    def Projectile(self, mass: float, z: int, q: int) -> FakeProjectile:
        projectile = FakeProjectile(mass, z, q)
        self.projectiles.append(projectile)
        return projectile

    def dedx(self, p: FakeProjectile, m: FakeMaterial, config: Any) -> Any:
        if self.mode in ("nan", "inf", "negative", "boolean"):
            return {"nan": float("nan"), "inf": float("inf"), "negative": -1,
                    "boolean": True}[self.mode]
        if self.mode == "config_mutation":
            config.low_energy = 1
        if self.mode == "material_mutation":
            m.rho += .001
        nuclear = self.native.nuclear_stopping(p.z, p.mass, m.target.Z, m.target.A, p.energy)
        if self.mode == "negative_electronic":
            return nuclear / 2
        extra = 1. if self.mode == "radiative" and config.corrections == 0 else 0.
        return nuclear + 100. + p.energy + extra

    def sezi_dedx_e(self, p: FakeProjectile, m: FakeMaterial, config: Any) -> float:
        if self.mode == "bad_closure":
            return 100. + p.energy + 1.
        return 100. + p.energy

    def range(self, *args: Any) -> None:
        pytest.fail("Projected range and CSDA range must not be conflated; no range calls")

    def catima_info(self) -> str:
        return "fake backend; no native physics executed"


def test_exact_fractional_source_mass_helium_and_units(generator: ModuleType) -> None:
    backend = FakeBackend(generator.native)
    report = generator.generate_samples(backend, REQUESTS, elemental_i_ev={13: 166., 14: 173.})
    assert [(p.z, p.mass, p.q) for p in backend.projectiles] == [(2, 4., 2), (26, 55.8, 26)]
    assert backend.projectiles[0].energies == [.25, 10., 3000., .25]
    assert backend.projectiles[1].energies == [float(Fraction(5000, 279))]
    assert report["low_energy_electronic_closure_count"] == 3
    assert report["low_energy_electronic_closure_max_relative_error"] < 1e-10
    for row, req in zip(report["rows"], REQUESTS, strict=True):
        assert {key: row[key] for key in req} == req
        assert row["stopping_electronic_mev_cm2_g"] == pytest.approx(
            100. + req["energy_mev_per_nucleon"], rel=1e-14,
        )
        assert row["stopping_total_mev_cm2_g"] == row["stopping_nonradiative_mev_cm2_g"]
        assert row["stopping_radiative_mev_cm2_g"] == 0
        assert set(row) == generator.REQUEST_FIELDS | generator.STOPPING_FIELDS
    assert report["materials"][0]["atomic_mass_u"] == 26.9815
    assert report["materials"][0]["density_g_cm3"] == 2.699
    assert report["materials"][1]["elemental_ionization_potential_ev"] == 173.
    assert report["backend_configuration"]["corrections"] == 0
    assert report["nonradiative_configuration"]["corrections"] == 8


@pytest.mark.parametrize("field,value", [
    ("z", True), ("z", 1), ("z", 93), ("z", 2.0),
    ("material_z", True), ("material_z", 12), ("material_z", 13.0),
    ("mass_parameter_u_decimal", "1.9"), ("mass_parameter_u_decimal", "301"),
    ("mass_parameter_u_decimal", "4e0"), ("mass_parameter_u_decimal", "NaN"),
    ("mass_parameter_u_decimal", "-4"), ("mass_parameter_u_decimal", " 4"),
    ("mass_parameter_u_decimal", 4), ("mass_parameter_u_decimal", "4" * 41),
    ("energy_total_mev_decimal", "0"), ("energy_total_mev_decimal", "1e0"),
    ("energy_total_mev_decimal", "Infinity"), ("energy_total_mev_decimal", ".1"),
    ("energy_mev_per_nucleon", .26), ("energy_mev_per_nucleon", float("nan")),
    ("energy_mev_per_nucleon", float("inf")), ("energy_mev_per_nucleon", True),
    ("raw_sha256", "a" * 63), ("raw_sha256", "A" * 64), ("raw_sha256", None),
    ("native_row_index", -1), ("native_row_index", True), ("native_row_index", 100_000),
    ("case_name", " "), ("case_name", None), ("case_name", "a" * 201),
])
def test_invalid_request_value_rejected(generator: ModuleType, field: str, value: Any) -> None:
    req = request()
    req[field] = value
    with pytest.raises(ValueError):
        generator.validate_requests([req])


@pytest.mark.parametrize("value", [None, {}, [], (), [1], [dict(request(), extra=1)]])
def test_invalid_inventory_rejected(generator: ModuleType, value: Any) -> None:
    with pytest.raises(ValueError):
        generator.validate_requests(value)


@pytest.mark.parametrize("total", ["0.039999999999999999999", "12000.000000000001"])
def test_exact_rational_energy_bound_precedes_float_rounding(generator: ModuleType, total: str) -> None:
    with pytest.raises(ValueError, match="bounded"):
        generator.validate_requests([request(total=total)])


def test_bounds_inclusive_and_requests_copy(generator: ModuleType) -> None:
    lower, upper = request(total="0.04"), request(total="12000", index=1)
    result = generator.validate_requests([lower, upper])
    assert result == (lower, upper)
    assert result[0] is not lower
    lower["raw_sha256"] = "b" * 64
    assert result[0]["raw_sha256"] == "a" * 64


@pytest.mark.parametrize("mode", ["duplicate", "reverse", "row_index", "raw_change", "case_change"])
def test_cross_row_identity_and_order_checks(generator: ModuleType, mode: str) -> None:
    first, second = request(), request(total="2", index=1)
    if mode == "duplicate":
        second = dict(first)
    elif mode == "reverse":
        first, second = second, first
    elif mode == "row_index":
        second["native_row_index"] = 0
    elif mode == "raw_change":
        second["raw_sha256"] = "b" * 64
    else:
        second["material_z"] = 14
    with pytest.raises(ValueError):
        generator.validate_requests([first, second])


@pytest.mark.parametrize("mode", ["nan", "inf", "negative", "boolean", "config_mutation",
                                 "material_mutation", "negative_electronic", "radiative", "bad_closure"])
def test_bad_backend_values_fail_closed(generator: ModuleType, mode: str) -> None:
    with pytest.raises(ValueError):
        generator.generate_samples(FakeBackend(generator.native, mode), REQUESTS,
                                   elemental_i_ev={13: 166., 14: 173.})


@pytest.mark.parametrize("values", [{13: 166.}, {13: 166., 14: 0.}, {13: 166., 14: True}, None])
def test_bad_ionization_metadata_rejected(generator: ModuleType, values: Any) -> None:
    with pytest.raises(ValueError):
        generator.generate_samples(FakeBackend(generator.native), REQUESTS, elemental_i_ev=values)


def _payload() -> dict[str, Any]:
    return {"schema_version": "radar.srim_catima_overlap_requests.v1", "status": "INCOMPLETE",
            "normative_export_allowed": False, "requests": deepcopy(REQUESTS),
            "caller_metadata": {"retained": "opaque extra root metadata"}}


@pytest.fixture
def input_file(tmp_path: Path) -> Path:
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    return path


def test_request_hash_and_schema_are_required(generator: ModuleType, input_file: Path) -> None:
    sha = generator.native.sha256(input_file)
    assert generator.read_requests(input_file, sha) == tuple(REQUESTS)
    for wrong in (sha.upper(), "a" * 64, "", None):
        with pytest.raises(ValueError):
            generator.read_requests(input_file, wrong)


@pytest.mark.parametrize("field,value", [("schema_version", "wrong"), ("status", "PASS"),
                                         ("normative_export_allowed", True),
                                         ("normative_export_allowed", 0),
                                         ("physical_accuracy_verified", True)])
def test_request_file_metadata_fail_closed(
    generator: ModuleType, input_file: Path, field: str, value: Any,
) -> None:
    payload = _payload()
    payload[field] = value
    input_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        generator.read_requests(input_file, generator.native.sha256(input_file))


@pytest.mark.parametrize("raw", ['{"requests":[],"requests":[]}', '{"extra":NaN}',
                               '{"extra":Infinity}', '{"extra":-Infinity}'])
def test_duplicate_or_nonfinite_json_rejected(generator: ModuleType, input_file: Path, raw: str) -> None:
    input_file.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        generator.read_requests(input_file, generator.native.sha256(input_file))


@pytest.fixture
def context(generator: ModuleType, monkeypatch: pytest.MonkeyPatch, input_file: Path
            ) -> tuple[ModuleType, Path, str]:
    monkeypatch.setattr(generator, "_implementation_hashes", lambda: {"implementation": "pin"})
    monkeypatch.setattr(generator.native, "verify_sources", lambda: {"native": "pin"})
    monkeypatch.setattr(generator.native, "production_hashes", lambda: {"production": "pin"})
    monkeypatch.setattr(generator.native, "load_backend", lambda: FakeBackend(generator.native))
    monkeypatch.setattr(generator, "_elemental_i_values", lambda: {13: 166., 14: 173.})
    return generator, input_file, str(generator.native.sha256(input_file))


def test_report_identity_and_qualification_boundaries(context: tuple[ModuleType, Path, str]) -> None:
    generator, path, sha = context
    report = generator.generate_controls(path, sha)
    assert report["schema_version"] == "radar.srim_catima_overlap_native.v1"
    assert report["status"] == "INCOMPLETE"
    assert report["request_path"] == str(path.resolve())
    assert report["requests_sha256"] == sha
    assert report["request_count"] == len(REQUESTS)
    assert all(report[flag] is False for flag in generator.FLAGS)
    assert all("range_g_cm2" not in row for row in report["rows"])
    for prefix in ("implementation_sha256", "native_source_pins", "production_resources_sha256"):
        assert report[prefix + "_before"] == report[prefix + "_after"]


@pytest.mark.parametrize("which", ["implementation", "native", "production", "request"])
def test_mutated_inputs_fail_closed(
    context: tuple[ModuleType, Path, str], monkeypatch: pytest.MonkeyPatch, which: str,
) -> None:
    generator, path, sha = context
    states = iter(({"fixture": "before"}, {"fixture": "after"}))
    if which == "implementation":
        monkeypatch.setattr(generator, "_implementation_hashes", lambda: next(states))
    elif which == "native":
        monkeypatch.setattr(generator.native, "verify_sources", lambda: next(states))
    elif which == "production":
        monkeypatch.setattr(generator.native, "production_hashes", lambda: next(states))
    else:
        def altered_backend() -> FakeBackend:
            path.write_text("changed", encoding="utf-8")
            return FakeBackend(generator.native)
        monkeypatch.setattr(generator.native, "load_backend", altered_backend)
    with pytest.raises(ValueError, match="changed|mismatch"):
        generator.generate_controls(path, sha)


def test_bad_request_stops_before_backend_load(
    context: tuple[ModuleType, Path, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, path, _ = context
    monkeypatch.setattr(generator.native, "load_backend", lambda: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="SHA256"):
        generator.generate_controls(path, "b" * 64)


def test_native_verification_stops_before_backend_load(
    context: tuple[ModuleType, Path, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, path, sha = context
    def invalid_source() -> None:
        raise ValueError("Native source mismatch")
    monkeypatch.setattr(generator.native, "verify_sources", invalid_source)
    monkeypatch.setattr(generator.native, "load_backend", lambda: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="source mismatch"):
        generator.generate_controls(path, sha)


def test_cli_and_exclusive_deterministic_output(
    context: tuple[ModuleType, Path, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    generator, path, sha = context
    output, replay = tmp_path / "native.json", tmp_path / "replay.json"
    monkeypatch.setattr(sys, "argv", ["worker", str(path), "--request-sha256", sha,
                                     "--output", str(output)])
    assert generator.main() == 0
    report = generator.generate_controls(path, sha)
    generator.write_report(report, replay)
    assert output.read_bytes() == replay.read_bytes()
    original = output.read_bytes()
    monkeypatch.setattr(generator, "generate_controls", lambda *_: pytest.fail("must not run"))
    with pytest.raises(FileExistsError):
        generator.main()
    with pytest.raises(FileExistsError):
        generator.write_report(report, output)
    assert output.read_bytes() == original


def test_readonly_validator_never_loads_backend(
    context: tuple[ModuleType, Path, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, path, sha = context
    report = generator.generate_controls(path, sha)
    monkeypatch.setattr(generator.native, "load_backend", lambda: pytest.fail("must not load native"))
    generator.validate_report(report, path, sha, _payload())


@pytest.mark.parametrize("field,value", [
    ("schema_version", "wrong"), ("status", "COMPLETE"), ("request_count", True),
    ("request_count", 4), ("requests_sha256", "b" * 64), ("request_path", "relative"),
    ("normative_export_allowed", True), ("ranges_computed", None), ("physical_accuracy_verified", 0),
    ("mass_policy", "rounded_isotopes"), ("stopping_units", "MeV/u"),
    ("energy_policy", "interpolated"), ("native_sampling_policy", "fit"),
    ("projectile_charge_input", "SEP A/Q"), ("native_source_pins_before", {}),
    ("native_source_pins_after", {}), ("production_resources_sha256_before", {}),
    ("production_resources_sha256_after", {}), ("implementation_sha256_before", {}),
    ("implementation_sha256_after", {}), ("backend_configuration", None),
    ("nonradiative_configuration", {}), ("materials", None), ("materials", []),
    ("python", ""), ("backend_info_literal", None),
    ("low_energy_electronic_closure_count", True), ("low_energy_electronic_closure_count", 2),
    ("low_energy_electronic_closure_max_relative_error", float("nan")),
    ("low_energy_electronic_closure_max_relative_error", True),
    ("low_energy_electronic_closure_max_relative_error", 1e-9), ("rows", []), ("rows", None),
])
def test_validator_rejects_bad_metadata(
    context: tuple[ModuleType, Path, str], field: str, value: Any,
) -> None:
    generator, path, sha = context
    report = generator.generate_controls(path, sha)
    report[field] = value
    with pytest.raises(ValueError):
        generator.validate_report(report, path, sha, _payload())


@pytest.mark.parametrize("which", ["configuration_bool", "I_override", "density", "target_mass",
                                  "material_bool", "extra_row", "row_order", "row_bool",
                                  "row_energy", "missing_field", "extra_field", "negative",
                                  "nan", "bool", "decomposition", "radiative", "nuclear_formula"])
def test_validator_rejects_malformed_nested_metadata_and_samples(
    context: tuple[ModuleType, Path, str], which: str,
) -> None:
    generator, path, sha = context
    report = generator.generate_controls(path, sha)
    if which == "configuration_bool":
        report["backend_configuration"]["calculation"] = True
    elif which == "I_override":
        report["materials"][0]["ionization_potential_override_ev"] = 166.
    elif which == "density":
        report["materials"][0]["density_g_cm3"] = 2.7
    elif which == "target_mass":
        report["materials"][0]["atomic_mass_u"] = 26.981539
    elif which == "material_bool":
        report["materials"][0]["ionization_potential_override_ev"] = False
    elif which == "extra_row":
        report["rows"].append(report["rows"][0])
    elif which == "row_order":
        report["rows"].reverse()
    elif which == "row_bool":
        report["rows"][0]["native_row_index"] = False
    elif which == "row_energy":
        report["rows"][0]["energy_mev_per_nucleon"] += .1
    elif which == "missing_field":
        del report["rows"][0]["raw_sha256"]
    elif which == "extra_field":
        report["rows"][0]["extra"] = 1
    elif which in ("negative", "nan", "bool"):
        report["rows"][0]["stopping_electronic_mev_cm2_g"] = {
            "negative": -1., "nan": float("nan"), "bool": True,
        }[which]
    elif which == "decomposition":
        report["rows"][0]["stopping_electronic_mev_cm2_g"] *= 1.1
    elif which == "radiative":
        report["rows"][0]["stopping_radiative_mev_cm2_g"] = 1.
    else:
        # Preserve total decomposition while corrupting its nuclear component.
        report["rows"][0]["stopping_nuclear_mev_cm2_g"] += .01
        report["rows"][0]["stopping_electronic_mev_cm2_g"] -= .01
    with pytest.raises(ValueError):
        generator.validate_report(report, path, sha, _payload())


@pytest.mark.parametrize("plan", [None, {}, {"requests": []}, {"requests": [request(total="2")]}])
def test_validator_rejects_changed_caller_inventory(
    context: tuple[ModuleType, Path, str], plan: Any,
) -> None:
    generator, path, sha = context
    with pytest.raises(ValueError):
        generator.validate_report(generator.generate_controls(path, sha), path, sha, plan)
