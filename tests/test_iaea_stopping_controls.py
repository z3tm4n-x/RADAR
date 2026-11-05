"""Native experimental controls: pure fake backend; no extension import in tests."""

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
            "iaea_native_controls_test", scripts / "generate_iaea_stopping_controls.py",
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

    def T(self, value: float) -> None:
        self.energy = value
        self.energies.append(value)


class FakeMaterial:
    def __init__(self, z: int) -> None:
        self.target = SimpleNamespace(A=27. if z == 13 else 28., Z=z, stn=1.)
        self.override: Any = 0.
        self.rho: Any = 2.7 if z == 13 else 2.33

    def get_element(self, index: int) -> Any:
        assert index == 0
        return self.target

    def ncomponents(self) -> int:
        return 1

    def density(self) -> Any:
        return self.rho

    def I(self) -> Any:  # noqa: E743 - the native API's field name
        return self.override


class FakeBackend:
    Config = FakeConfig

    def __init__(self, native: ModuleType) -> None:
        self.native = native
        self.projectiles: list[FakeProjectile] = []
        self.materials = {z: FakeMaterial(z) for z in (13, 14)}
        self.mode = "normal"

    def Projectile(self, a: float, z: int, q: int) -> FakeProjectile:
        result = FakeProjectile(a, z, q)
        self.projectiles.append(result)
        return result

    def get_material(self, z: int) -> FakeMaterial:
        return self.materials[z]

    def dedx(self, p: FakeProjectile, m: FakeMaterial, config: Any) -> Any:
        if self.mode == "nan_total":
            return float("nan")
        if self.mode == "negative_total":
            return -1.
        if self.mode == "bool_total":
            return True
        if self.mode == "configuration_mutation":
            config.low_energy = 1
        nuclear = self.native.nuclear_stopping(p.z, p.a, m.target.Z, m.target.A, p.energy)
        if self.mode == "negative_electronic":
            return nuclear / 2.
        extra = 2. if config.corrections == 0 and p.energy > 50_000. else 0.
        if self.mode == "unexpected_radiative" and config.corrections == 0:
            extra = 1.
        if self.mode == "negative_radiative" and config.corrections == 8:
            extra = 1.
        return nuclear + 100. + p.energy + extra

    def range(self, p: FakeProjectile, m: FakeMaterial, config: Any) -> Any:
        if self.mode == "range_nan":
            return float("nan")
        if self.mode == "range_zero":
            return 0.
        if self.mode == "range_constant":
            return 1.
        if self.mode == "material_mutation":
            m.rho += .01
        return p.a * p.energy + 3.

    def sezi_dedx_e(self, p: FakeProjectile, m: FakeMaterial, config: Any) -> Any:
        if self.mode == "closure_nan":
            return float("nan")
        return 100. + p.energy + (1. if self.mode == "closure_error" else 0.)

    def catima_info(self) -> str:
        return "synthetic_backend_not_native_execution"


REQUESTS = (
    (6, 12, 13, .01), (6, 12, 13, 10.), (6, 12, 13, 50_000.),
    (6, 12, 13, 100_000.), (6, 12, 14, 9.3), (14, 30, 14, 9.3),
)


def _samples(generator: ModuleType, backend: FakeBackend) -> dict[str, Any]:
    return dict(generator.generate_samples(backend, REQUESTS, elemental_i_ev={13: 166., 14: 173.}))


def test_exact_isotope_energy_and_whole_ion_stopping_units(generator: ModuleType) -> None:
    backend = FakeBackend(generator.native)
    result = _samples(generator, backend)
    assert len(result["rows"]) == 6
    assert [(p.z, p.a, p.q) for p in backend.projectiles] == [(6, 12., 6), (14, 30., 14)]
    assert backend.projectiles[0].energies == [.01, 10., 50_000., 100_000., 9.3]
    for row, request in zip(result["rows"], REQUESTS, strict=True):
        z, a, material_z, energy = request
        assert (row["z"], row["mass_number"], row["material_z"], row["energy_mev_per_nucleon"]) == request
        assert row["mass_parameter_u"] == a
        assert row["energy_mev_total_assuming_numeric_mass_number"] == a * energy
        assert row["stopping_electronic_mev_cm2_g"] == pytest.approx(100 + energy)
        assert row["range_g_cm2"] == a * energy + 3
        assert row["stopping_radiative_mev_cm2_g"] == (2. if energy > 50_000. else 0.)
        assert row["stopping_total_mev_cm2_g"] == pytest.approx(sum(row[k] for k in (
            "stopping_electronic_mev_cm2_g", "stopping_nuclear_mev_cm2_g", "stopping_radiative_mev_cm2_g",
        )))
        assert 0 < row["nuclear_fraction_of_nonradiative_stopping"] < 1
    assert result["low_energy_electronic_closure_count"] == 4
    assert result["low_energy_electronic_closure_max_relative_error"] < 1e-14
    assert result["backend_configuration"]["z_effective"] == 1
    assert result["backend_configuration"]["low_energy"] == 0
    assert result["nonradiative_configuration"]["corrections"] == 8
    assert all(r["ionization_potential_override_ev"] == 0 for r in result["materials"])


@pytest.mark.parametrize("requests", [
    (), [], ((6, 12, 13, .009),), ((6, 12, 13, 100_001.),),
    ((True, 12, 13, 1.),), ((6, True, 13, 1.),), ((6, 12, True, 1.),),
    ((6, 12, 13, True),), ((6, 12, 13, "1"),), ((6, 12., 13, 1.),),
    ((2, 4, 13, 1.),), ((93, 240, 13, 1.),), ((6, 5, 13, 1.),),
    ((6, 301, 13, 1.),), ((6, 12, 12, 1.),), ((6, 12, 13, float("inf")),),
    ((6, 12, 13, float("nan")),), ((6, 12, 13),), ([6, 12, 13, 1.],),
    ((6, 12, 13, 1.), (6, 12, 13, 1.)), ((6, 12, 14, 1.), (6, 12, 13, 1.)),
])
def test_invalid_or_ambiguous_requests(generator: ModuleType, requests: Any) -> None:
    with pytest.raises(ValueError):
        generator.generate_samples(FakeBackend(generator.native), requests,
                                   elemental_i_ev={13: 166., 14: 173.})


@pytest.mark.parametrize("mode", [
    "nan_total", "negative_total", "bool_total", "negative_electronic",
    "unexpected_radiative", "negative_radiative", "range_nan", "range_zero",
    "range_constant", "closure_nan", "closure_error", "configuration_mutation", "material_mutation",
])
def test_backend_invalid_values_or_configuration_changes(generator: ModuleType, mode: str) -> None:
    backend = FakeBackend(generator.native)
    backend.mode = mode
    with pytest.raises(ValueError):
        _samples(generator, backend)


@pytest.mark.parametrize("override", [1., True, float("nan"), -1., "0"])
def test_material_i_is_never_refitted(generator: ModuleType, override: Any) -> None:
    backend = FakeBackend(generator.native)
    backend.materials[13].override = override
    with pytest.raises(ValueError):
        _samples(generator, backend)


@pytest.mark.parametrize("values", [{13: 166.}, {13: 166., 14: 0.}, {13: True, 14: 173.}])
def test_internal_material_constants_required(generator: ModuleType, values: Any) -> None:
    with pytest.raises(ValueError):
        generator.generate_samples(FakeBackend(generator.native), REQUESTS, elemental_i_ev=values)


def test_native_elemental_i_header_is_unchanged(generator: ModuleType) -> None:
    assert generator._elemental_i_values() == {13: 166., 14: 173.}
    assert generator.MASS_POLICY == "experimental_isotope_mass_number_numeric_u_assumption"


def test_exclusive_report_and_nonfinite_output_rejected(generator: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    generator.write_report({"normative_export_allowed": False}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"normative_export_allowed": False}
    with pytest.raises(FileExistsError):
        generator.write_report({}, path)
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        generator.write_report({"bad": float("nan")}, invalid)
    assert not invalid.exists()


def _inventory() -> dict[str, Any]:
    return {
        "selected_records": [
            {"csv_record_number": 5, "z": 6, "mass_number": 12, "material_z": 13,
             "energy_mev_per_nucleon": 10., "raw_fields": {"source": "first"}},
            {"csv_record_number": 7, "z": 6, "mass_number": 12, "material_z": 13,
             "energy_mev_per_nucleon": 10., "raw_fields": {"source": "duplicate independent record"}},
            {"csv_record_number": 12, "z": 26, "mass_number": 56, "material_z": 14,
             "energy_mev_per_nucleon": 9.3, "raw_fields": {"source": "same request as RADEF"}},
        ],
        "provenance": "synthetic_inventory_for_native_control_tests",
    }


def test_duplicate_records_share_one_native_key_without_losing_identity(generator: ModuleType) -> None:
    requests, links = generator._requests_from_inventory(_inventory())
    assert len(requests) == 8
    assert links[(6, 12, 13, 10.)] == [5, 7]
    assert links[(26, 56, 14, 9.3)] == [12]
    assert (14, 30, 14, 9.3) in requests
    assert (14, 28, 14, 9.3) not in requests
    assert (36, 82, 14, 9.3) in requests
    assert (54, 131, 14, 9.3) in requests


@pytest.mark.parametrize("records", [None, [], "records", [None], [{}]])
def test_missing_selected_record_inventory_rejected(generator: ModuleType, records: Any) -> None:
    with pytest.raises(ValueError):
        generator._requests_from_inventory({"selected_records": records})


@pytest.mark.parametrize("number", [True, 1, -1, 5., "5", 7])
def test_bad_or_repeated_csv_identity_rejected(generator: ModuleType, number: Any) -> None:
    inventory = _inventory()
    inventory["selected_records"][0]["csv_record_number"] = number
    with pytest.raises(ValueError):
        generator._requests_from_inventory(inventory)


@pytest.fixture
def control_context(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, Path, Path]:
    monkeypatch.setattr(generator, "_load_inventory", lambda _data, _refs: _inventory())
    monkeypatch.setattr(generator, "_implementation_hashes", lambda: {"implementation_fixture": "pin"})
    monkeypatch.setattr(generator, "_source_input_hashes", lambda _data, _refs: {"source_fixture": "pin"})
    monkeypatch.setattr(generator.native, "load_backend", lambda: FakeBackend(generator.native))
    monkeypatch.setattr(generator.native, "verify_sources", lambda: {"native_fixture": "pin"})
    monkeypatch.setattr(generator.native, "production_hashes", lambda: {"production_fixture": "pin"})
    return generator, Path("testdata.zip"), Path("testreferences.zip")


def test_generator_records_full_selection_and_all_qualification_boundaries(
    control_context: tuple[ModuleType, Path, Path],
) -> None:
    generator, data, references = control_context
    report = generator.generate_controls(data, references)
    assert report["schema_version"] == "radar.iaea_experimental_stopping_native_controls.v1"
    assert report["status"] == "INCOMPLETE"
    assert report["iaea_selected_record_count"] == 3
    assert report["iaea_unique_request_count"] == 2
    assert report["radef_request_count"] == 7
    assert report["unique_native_request_count"] == 8
    carbon = next(r for r in report["rows"] if r["z"] == 6)
    iron = next(r for r in report["rows"] if r["z"] == 26)
    assert carbon["iaea_csv_record_numbers"] == [5, 7]
    assert iron["iaea_csv_record_numbers"] == [12]
    assert iron["request_origins"] == ["iaea_selected_experimental_records", "radef_author_report_slide_6"]
    assert report["selection_canonical_sha256_before"] == generator._canonical_sha256(_inventory()["selected_records"])
    for prefix in ("source_inputs_sha256", "implementation_sha256", "native_source_pins",
                   "production_resources_sha256", "inventory_canonical_sha256", "selection_canonical_sha256"):
        assert report[prefix + "_before"] == report[prefix + "_after"]
    for flag in ("normative_export_allowed", "physical_accuracy_verified", "physical_mass_policy_qualified",
                 "production_tables_replaced", "source_to_binary_build_verified", "absolute_range_origin_verified",
                 "source_measurement_classification_verified", "reference_data_reuse_rights_qualified",
                 "stage4_acceptance_matrix_completed", "fortran_executed"):
        assert report[flag] is False
    assert report["range_origin_energy_mev_per_nucleon"] == .001
    assert report["mass_policy"] == "experimental_isotope_mass_number_numeric_u_assumption"
    assert report["radef_source"]["comparison_policy"] == "compare_electronic_and_nonradiative_total_separately"


@pytest.mark.parametrize("which", ["implementation", "source", "native", "production", "selection", "inventory"])
def test_changed_pins_or_reference_inventory_rejected(
    control_context: tuple[ModuleType, Path, Path], monkeypatch: pytest.MonkeyPatch, which: str,
) -> None:
    generator, data, references = control_context
    states = iter(({"fixture": "before"}, {"fixture": "after"}))
    if which == "implementation":
        monkeypatch.setattr(generator, "_implementation_hashes", lambda: next(states))
    elif which == "source":
        monkeypatch.setattr(generator, "_source_input_hashes", lambda _d, _r: next(states))
    elif which == "native":
        monkeypatch.setattr(generator.native, "verify_sources", lambda: next(states))
    elif which == "production":
        monkeypatch.setattr(generator.native, "production_hashes", lambda: next(states))
    else:
        before = _inventory()
        after = _inventory()
        if which == "selection":
            after["selected_records"][0]["raw_fields"]["source"] = "changed measured source"
        else:
            after["provenance"] = "changed source description"
        inventories = iter((before, after))
        monkeypatch.setattr(generator, "_load_inventory", lambda _d, _r: next(inventories))
    with pytest.raises(ValueError, match="changed"):
        generator.generate_controls(data, references)


def test_source_and_bytecode_verification_failure_precedes_native_load(
    control_context: tuple[ModuleType, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, data, references = control_context

    def cache_error() -> None:
        raise ValueError("Bytecode caches are not allowed in the pinned backend runtime.")

    monkeypatch.setattr(generator.native, "verify_sources", cache_error)
    monkeypatch.setattr(generator.native, "load_backend", lambda: pytest.fail("unverified native load"))
    with pytest.raises(ValueError, match="Bytecode caches"):
        generator.generate_controls(data, references)


def test_radef_document_identity_is_verified(generator: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator.native, "sha256", lambda _p: "0" * 64)
    with pytest.raises(ValueError, match="RADEF"):
        generator._source_input_hashes(Path("unused.zip"), Path("unused_refs.zip"))


def test_main_preserves_existing_output_before_native_work(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "exists.json"
    output.write_text("original", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["controls", "unused.zip", "unused_refs.zip", "--output", str(output)])
    monkeypatch.setattr(generator, "generate_controls", lambda _d, _r: pytest.fail("must not calculate"))
    with pytest.raises(FileExistsError):
        generator.main()
    assert output.read_text(encoding="utf-8") == "original"


def test_bad_archives_fail_before_native_loading(
    generator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    data, refs = tmp_path / "bad_data.zip", tmp_path / "bad_refs.zip"
    data.write_bytes(b"not the pinned data archive")
    refs.write_bytes(b"not the pinned reference archive")
    monkeypatch.setattr(generator.native, "load_backend", lambda: pytest.fail("unverified native load"))
    with pytest.raises(ValueError, match="IAEA archive hash mismatch"):
        generator.generate_controls(data, refs)


def test_real_pinned_inventory_request_mapping(generator: ModuleType) -> None:
    source = generator.ROOT / "research/iaea_stopping_20261031"
    inventory = generator._load_inventory(source / "data.zip", source / "references.zip")
    requests, links = generator._requests_from_inventory(inventory)
    assert inventory["selected_record_count"] == 2612
    assert sum(len(values) for values in links.values()) == 2612
    assert requests == tuple(sorted(set(requests)))
    assert all(request in requests for request in generator._RADEF_REQUESTS)
    assert all(.01 <= request[3] <= 100_000 for request in requests)
    assert generator._source_input_hashes(source / "data.zip", source / "references.zip")
