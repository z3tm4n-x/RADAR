"""Independent arithmetic and fail-closed IAEA comparison witnesses.

Synthetic controls exercise report validation without loading a native backend;
their values are not physical measurements or qualification evidence.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        return importlib.import_module("compare_iaea_stopping")
    finally:
        sys.path.remove(str(scripts))


def _source_row(energy: str = "1", stopping: str = "0.1", *,
                ref_id: str = "SyntheticA", error: str = "3") -> dict[str, str]:
    return {
        "projectile_name": "C", "ion_isotope": "12.0", "target_name": "Al", "phase": "s",
        "energy": energy, "energy_unit": "MeV/u", "stopping_power": stopping,
        "stopping_unit": "MeV/(mg/cm2)", "percentage_error": error,
        "target_mass_atom_ratio": "26.9815384", "dataset_last_modification": "2026-01-01",
        "publication_year": "2026", "ref_id": ref_id, "citation_reference": "Synthetic",
        "doi": "10.0/synthetic", "comments": "verbatim, multiline\nsource note",
        "facility": "Synthetic only", "technique": "not physical data",
        "stopping_power_converted": stopping, "stopping_unit_converted": "MeV/(mg/cm2)",
    }


def _inventory() -> dict[str, Any]:
    source = importlib.import_module("iaea_stopping_reference")
    raw = [_source_row(), _source_row(), _source_row("2", "0.2", error=""),
           _source_row("3", "0.4", ref_id="SyntheticB", error="3-5")]
    records = tuple(source.SourceRecord(i, tuple(row[k] for k in source.DATA_HEADER))
                    for i, row in enumerate(raw, 2))
    references = tuple(source.ReferenceRecord(i, (ref, "10.0/synthetic", "Author", "Synthetic", "2026"))
                       for i, ref in enumerate(("SyntheticA", "SyntheticB"), 2))
    archive = source.IaeaArchive(records, references, (), ())
    result: dict[str, Any] = source.build_inventory(archive)
    return result


def _native_row(key: tuple[int, int, int, float], *, prediction: float = 110.) -> dict[str, Any]:
    z, a, target, energy = key
    return {
        "z": z, "mass_number": a, "material_z": target,
        "mass_parameter_u": float(a), "charge_input": z,
        "energy_mev_per_nucleon": energy,
        "energy_mev_total_assuming_numeric_mass_number": a * energy,
        "stopping_total_mev_cm2_g": prediction + 5.,
        "stopping_electronic_mev_cm2_g": prediction,
        "stopping_nuclear_mev_cm2_g": 5., "stopping_radiative_mev_cm2_g": 0.,
        "range_g_cm2": energy,
        "nuclear_fraction_of_nonradiative_stopping": 5. / (prediction + 5.),
    }


@pytest.fixture
def case(runner: ModuleType, monkeypatch: pytest.MonkeyPatch,
         tmp_path: Path) -> dict[str, Any]:
    data = tmp_path / "synthetic-data.zip"
    refs = tmp_path / "synthetic-refs.zip"
    data.write_bytes(b"synthetic source; archive parser tested separately")
    refs.write_bytes(b"synthetic reference bytes")
    inventory = _inventory()
    generator = runner.controls
    source_pins = {"native-source": "a" * 64}
    production = {"production/table": "b" * 64}
    implementation = {"native-generator": "c" * 64}
    monkeypatch.setattr(runner.native, "verify_sources", lambda: dict(source_pins))
    monkeypatch.setattr(runner.native, "production_hashes", lambda: dict(production))
    monkeypatch.setattr(generator, "_implementation_hashes", lambda: dict(implementation))

    def input_hashes(data_path: Path, refs_path: Path) -> dict[str, str]:
        return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (data_path, refs_path)}

    monkeypatch.setattr(generator, "_source_input_hashes", input_hashes)
    requests, links = generator._requests_from_inventory(inventory)
    rows = []
    for key in requests:
        row = _native_row(key, prediction=110. * key[3])
        row["iaea_csv_record_numbers"] = links.get(key, [])
        row["request_origins"] = ["iaea_selected_experimental_records"] if key in links else []
        if key in generator._RADEF_REQUESTS:
            row["request_origins"].append("radef_author_report_slide_6")
        rows.append(row)
    payload = {
        "schema_version": "radar.iaea_experimental_stopping_native_controls.v1", "status": "INCOMPLETE",
        "data_archive_path": str(data.resolve()), "references_archive_path": str(refs.resolve()),
        "mass_policy": generator.MASS_POLICY,
        "energy_coordinate_compatibility": "source_MeV_n_numerically_equal_backend_MeV_u_assumed",
        "stopping_units": "whole_projectile_MeV_cm2_g_no_mass_number_multiplier",
        "native_sampling_policy": "exact_selected_experimental_energies_not_candidate_table_interpolation",
        "range_origin_energy_mev_per_nucleon": .001, "native_request_key_fields": list(runner.KEYS),
        **dict.fromkeys(runner.FLAGS, False),
        "backend_configuration": dict(runner.CONFIG),
        "nonradiative_configuration": {**runner.CONFIG, "corrections": 8},
        "iaea_selected_record_count": 4, "iaea_unique_request_count": 3,
        "radef_request_count": 7, "unique_native_request_count": 10,
        "radef_source": {
            "path": generator._RADEF_SOURCE_PATH, "sha256": generator._RADEF_SOURCE_SHA256,
            "page_one_based": 6, "request_keys": [list(k) for k in generator._RADEF_REQUESTS],
            "measurement_convention": "slide_alone_does_not_establish_explicit_nuclear_subtraction",
            "comparison_policy": "compare_electronic_and_nonradiative_total_separately",
        },
        "materials": [
            {"z": z, "atomic_mass_u": a, "density_g_cm3": density,
             "ionization_potential_override_ev": 0., "elemental_ionization_potential_ev": potential,
             "ionization_potential_policy": "unchanged_backend_internal_elemental_table"}
            for z, a, density, potential in ((13, 26.9815, 2.699, 166.), (14, 28.0855, 2.33, 173.))
        ],
        "rows": rows, "low_energy_electronic_closure_count": 10,
        "low_energy_electronic_closure_max_relative_error": 0.,
    }
    for name, value in {
        "source_inputs_sha256": input_hashes(data, refs),
        "implementation_sha256": implementation, "native_source_pins": source_pins,
        "production_resources_sha256": production,
        "inventory_canonical_sha256": generator._canonical_sha256(inventory),
        "selection_canonical_sha256": generator._canonical_sha256(inventory["selected_records"]),
    }.items():
        payload[name + "_before"] = deepcopy(value)
        payload[name + "_after"] = deepcopy(value)
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    monkeypatch.setattr(runner, "load_archives", lambda *_: deepcopy(inventory))
    monkeypatch.setattr(runner, "build_inventory", lambda archive: archive)
    return {"data": data, "refs": refs, "native": path, "inventory": inventory, "payload": payload,
            "production": production, "source_pins": source_pins, "implementation": implementation}


def _validate(runner: ModuleType, case: dict[str, Any]) -> dict[tuple[int, int, int, float], dict[str, Any]]:
    result: dict[tuple[int, int, int, float], dict[str, Any]] = runner._validate_controls(
        case["payload"], case["inventory"], case["data"], case["refs"],
    )
    return result


def _run(runner: ModuleType, case: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = runner.run_comparison(
        case["data"], case["refs"], case["native"],
        native_sha256=hashlib.sha256(case["native"].read_bytes()).hexdigest(),
    )
    return result


def test_whole_ion_stopping_no_extra_mass_or_density(runner: ModuleType) -> None:
    record = _inventory()["selected_records"][0]
    before = deepcopy(record)
    compared = runner._compare_observation(record, _native_row((6, 12, 13, 1.)))
    assert record == before
    assert compared["stopping_native_mev_cm2_g"] == 100.
    assert compared["native_prediction_electronic_mev_cm2_g"] == 110.
    assert compared["signed_relative_difference"] == pytest.approx(.1)
    assert compared["native_prediction_nuclear_mev_cm2_g"] == 5.
    assert compared["raw_fields"] == record["raw_fields"]
    assert compared["difference_denominator"] == "independently_converted_native_experimental_stopping"
    assert compared["uncertainty_used_as_acceptance_tolerance"] is False
    assert "not_individual_article_verification" in compared["experimental_electronic_convention"]


@pytest.mark.parametrize("provider,expected", [(50., 1.2), (None, None)])
def test_provider_conversion_never_replaces_raw_reference(
    runner: ModuleType, provider: float | None, expected: float | None,
) -> None:
    record = _inventory()["selected_records"][0]
    record["stopping_converted_mev_cm2_g"] = provider
    compared = runner._compare_observation(record, _native_row((6, 12, 13, 1.)))
    assert compared["signed_relative_difference"] == pytest.approx(.1)
    actual = compared["signed_relative_difference_using_provider_conversion"]
    assert actual is None if expected is None else actual == pytest.approx(expected)


@pytest.mark.parametrize("key,value", [
    ("z", 7), ("mass_number", 13), ("material_z", 14), ("energy_mev_per_nucleon", 1.000001),
    ("mass_number", True), ("energy_mev_per_nucleon", True),
])
def test_comparison_requires_exact_projectile_target_energy(
    runner: ModuleType, key: str, value: Any,
) -> None:
    row = _native_row((6, 12, 13, 1.))
    row[key] = value
    with pytest.raises(ValueError):
        runner._compare_observation(_inventory()["selected_records"][0], row)


@pytest.mark.parametrize("value", [0., -1., True, float("nan"), float("inf"), "100"])
def test_nonpositive_or_nonnumeric_experiment_rejected(runner: ModuleType, value: Any) -> None:
    record = _inventory()["selected_records"][0]
    record["stopping_native_mev_cm2_g"] = value
    with pytest.raises(ValueError):
        runner._compare_observation(record, _native_row((6, 12, 13, 1.)))


def test_summary_distinguishes_duplicates_and_unknown_uncertainty(runner: ModuleType) -> None:
    rows = []
    for record, ratio in zip(_inventory()["selected_records"], (1.5, 1.5, 1.1, .8), strict=True):
        native_row = _native_row(runner._key(record), prediction=record["stopping_native_mev_cm2_g"] * ratio)
        rows.append(runner._compare_observation(record, native_row))
    result = runner._summarize(rows)
    assert result["all_source_records"]["record_count"] == 4
    assert result["exact_raw_unique_records"]["record_count"] == 3
    assert result["exact_duplicate_excess_records"] == 1
    assert result["all_source_records"]["median_absolute_relative_difference"] == pytest.approx(.35)
    assert result["exact_raw_unique_records"]["median_absolute_relative_difference"] == pytest.approx(.2)
    assert result["all_source_records"]["min_signed_relative_difference"] == pytest.approx(-.2)
    assert result["all_source_records"]["max_signed_relative_difference"] == pytest.approx(.5)
    assert result["reported_uncertainty_missing_count"] == 1
    assert rows[2]["reported_percentage_error_scalar"] is None
    assert rows[3]["reported_percentage_error_text"] == "3-5"
    assert rows[3]["reported_percentage_error_scalar"] is None
    assert result["independent_experiment_count"] is None
    assert result["summary_is_a_physical_acceptance_test"] is False
    with pytest.raises(ValueError):
        runner._summarize([])


def test_radef_measured_column_and_two_stopping_conventions(runner: ModuleType) -> None:
    samples = {}
    for z, a, token in runner.RADEF_MEASURED:
        samples[z, a, 14, 9.3] = _native_row((z, a, 14, 9.3), prediction=float(token) * 1000)
    results = runner._radef_comparisons(samples)
    assert len(results) == 7
    assert [r["measured_let_printed_token"] for r in results] == ["1.88", "3.64", "6.73", "10.1", "18.5", "30.2", "55.3"]
    for row in results:
        assert row["electronic_signed_relative_difference"] == pytest.approx(0.)
        assert row["nonradiative_total_signed_relative_difference"] == pytest.approx(.005 / row["measured_let_mev_cm2_mg"])
        assert row["reported_uncertainty_literal"] == "sigma_LET_MEAS < 3%"
        assert row["nuclear_subtraction_in_measurement_verified"] is False
        assert row["independent_of_IAEA_records_established"] is False
        assert row["uncertainty_used_as_acceptance_tolerance"] is False


def test_complete_exact_native_fixture_validates(runner: ModuleType, case: dict[str, Any]) -> None:
    result = _validate(runner, case)
    assert len(result) == 10
    assert result[6, 12, 13, 1.]["iaea_csv_record_numbers"] == [2, 3]
    assert len(case["inventory"]["selected_records"]) == 4


@pytest.mark.parametrize("field,value", [
    ("schema_version", "legacy"), ("status", "COMPLETE"), ("data_archive_path", "other.zip"),
    ("references_archive_path", "other.zip"), ("mass_policy", "rounded_average_atomic_mass"),
    ("energy_coordinate_compatibility", "whole_MeV"), ("stopping_units", "MeV_cm2_g_per_nucleon"),
    ("native_sampling_policy", "interpolated"), ("range_origin_energy_mev_per_nucleon", 0.),
    ("native_request_key_fields", ["z", "mass_number", "energy"]),
    ("iaea_selected_record_count", 3), ("iaea_unique_request_count", True),
    ("radef_request_count", 8), ("unique_native_request_count", 9),
    ("low_energy_electronic_closure_count", 0),
    ("low_energy_electronic_closure_max_relative_error", 1.1e-10),
    ("low_energy_electronic_closure_max_relative_error", True),
])
def test_control_identity_and_count_guards(
    runner: ModuleType, case: dict[str, Any], field: str, value: Any,
) -> None:
    case["payload"][field] = value
    with pytest.raises(ValueError):
        _validate(runner, case)


@pytest.mark.parametrize("flag", [
    "normative_export_allowed", "physical_accuracy_verified", "physical_mass_policy_qualified",
    "production_tables_replaced", "source_to_binary_build_verified", "absolute_range_origin_verified",
    "source_measurement_classification_verified", "reference_data_reuse_rights_qualified",
    "stage4_acceptance_matrix_completed", "fortran_executed",
])
@pytest.mark.parametrize("value", [True, 0, None])
def test_every_qualification_flag_must_be_literal_false(
    runner: ModuleType, case: dict[str, Any], flag: str, value: Any,
) -> None:
    case["payload"][flag] = value
    with pytest.raises(ValueError):
        _validate(runner, case)


@pytest.mark.parametrize("name", [
    "source_inputs_sha256", "implementation_sha256", "native_source_pins",
    "production_resources_sha256", "inventory_canonical_sha256", "selection_canonical_sha256",
])
@pytest.mark.parametrize("suffix", ["before", "after"])
def test_every_generation_snapshot_rechecked(
    runner: ModuleType, case: dict[str, Any], name: str, suffix: str,
) -> None:
    case["payload"][name + "_" + suffix] = {}
    with pytest.raises(ValueError):
        _validate(runner, case)


@pytest.mark.parametrize("field,value", [
    ("mass_parameter_u", 12.011), ("charge_input", 5), ("charge_input", 6.),
    ("energy_mev_total_assuming_numeric_mass_number", 1.),
    ("stopping_total_mev_cm2_g", 110.), ("stopping_electronic_mev_cm2_g", 0.),
    ("stopping_nuclear_mev_cm2_g", -1.), ("stopping_radiative_mev_cm2_g", 1.),
    ("nuclear_fraction_of_nonradiative_stopping", 0.), ("range_g_cm2", 0.),
    ("iaea_csv_record_numbers", [2]), ("iaea_csv_record_numbers", [2., 3.]),
    ("request_origins", ["new_measurement"]), ("energy_mev_per_nucleon", True),
    ("mass_number", 12.011), ("material_z", 15),
])
def test_row_units_mass_components_and_links_are_not_relabelled(
    runner: ModuleType, case: dict[str, Any], field: str, value: Any,
) -> None:
    case["payload"]["rows"][0][field] = value
    with pytest.raises(ValueError):
        _validate(runner, case)


@pytest.mark.parametrize("kind", ["missing", "extra", "reversed", "extra_field", "decreasing_range"])
def test_native_roster_and_range_order(runner: ModuleType, case: dict[str, Any], kind: str) -> None:
    rows = case["payload"]["rows"]
    if kind == "missing":
        rows.pop()
    elif kind == "extra":
        rows.append(deepcopy(rows[-1]))
    elif kind == "reversed":
        rows.reverse()
    elif kind == "extra_field":
        rows[0]["unknown_column"] = 1
    else:
        rows[1]["range_g_cm2"] = rows[0]["range_g_cm2"]
    with pytest.raises(ValueError):
        _validate(runner, case)


@pytest.mark.parametrize("kind", ["I_fit", "density", "override_bool", "missing_material", "duplicate_material", "radef_page", "config", "config_bool"])
def test_material_configuration_and_author_source_guards(
    runner: ModuleType, case: dict[str, Any], kind: str,
) -> None:
    payload = case["payload"]
    if kind == "I_fit":
        payload["materials"][0]["elemental_ionization_potential_ev"] = 163.
    elif kind == "density":
        payload["materials"][0]["density_g_cm3"] = 1.
    elif kind == "override_bool":
        payload["materials"][0]["ionization_potential_override_ev"] = False
    elif kind == "missing_material":
        payload["materials"].pop()
    elif kind == "duplicate_material":
        payload["materials"][1] = deepcopy(payload["materials"][0])
    elif kind == "radef_page":
        payload["radef_source"]["page_one_based"] = 5
    elif kind == "config":
        payload["backend_configuration"]["corrections"] = 8
    else:
        payload["backend_configuration"]["corrections"] = False
    with pytest.raises(ValueError):
        _validate(runner, case)


def test_run_report_keeps_sources_separate_and_all_gates_closed(runner: ModuleType, case: dict[str, Any]) -> None:
    report = _run(runner, case)
    assert len(report["rows"]) == 4
    assert len(report["radef_checkpoints"]) == 7
    assert report["summary"]["all_source_records"]["record_count"] == 4
    assert report["summary"]["exact_raw_unique_records"]["record_count"] == 3
    assert len(report["by_reference"]) == 2
    assert report["status"] == "INCOMPLETE"
    assert all(report[flag] is False for flag in runner.FLAGS)
    assert report["summary"]["independent_experiment_count"] is None
    assert report["implementation_sha256_before"] == report["implementation_sha256_after"]
    assert report["production_resources_sha256_before"] == report["production_resources_sha256_after"]
    assert report["compilation_catalog_discrepancy"]["same_release_identity_established"] is False
    assert report["source_inventory"] == case["inventory"]
    assert all(row["energy_band"] == "03_[1,10)" for row in report["rows"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("native_sha", [None, True, "", "A" * 64, "a" * 63, "0" * 64])
def test_explicit_native_pin_required(runner: ModuleType, case: dict[str, Any], native_sha: Any) -> None:
    with pytest.raises(ValueError):
        runner.run_comparison(case["data"], case["refs"], case["native"], native_sha256=native_sha)


@pytest.mark.parametrize("text", ['{"status":"INCOMPLETE","status":"COMPLETE"}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
def test_json_duplicates_and_nonfinite_rejected(runner: ModuleType, case: dict[str, Any], text: str) -> None:
    case["native"].write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        _run(runner, case)


@pytest.mark.parametrize("which", ["data", "refs", "native", "production", "source_pins", "implementation", "inventory"])
def test_mutation_during_comparison_fails_closed(
    runner: ModuleType, case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, which: str,
) -> None:
    original = runner._compare_observation
    called = False

    def compare(record: dict[str, Any], native_row: dict[str, Any]) -> dict[str, Any]:
        nonlocal called
        result: dict[str, Any] = original(record, native_row)
        if not called:
            called = True
            if which in ("data", "refs", "native"):
                case[which].write_bytes(case[which].read_bytes() + b" ")
            elif which == "inventory":
                case["inventory"]["selected_records"][0]["raw_fields"]["comments"] += " tampered"
            else:
                case[which]["extra-unexpected-resource"] = "d" * 64
        return result

    monkeypatch.setattr(runner, "_compare_observation", compare)
    with pytest.raises(ValueError):
        _run(runner, case)


def test_cli_refuses_to_overwrite_existing_report(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "existing.json"
    output.write_bytes(b"retain original report")
    monkeypatch.setattr(sys, "argv", ["compare", "data", "refs", "native", "--native-sha256", "a" * 64,
                                     "--output", str(output)])
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == b"retain original report"
