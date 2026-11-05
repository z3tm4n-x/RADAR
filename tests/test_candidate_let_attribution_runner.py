"""Independent replay/aggregation guards for diagnostic LET-band attribution.

The native table validator and source builder are mocked explicitly. Candidate
transport, LET conversion, attribution, baseline replay and file hashes are real.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis, AssessmentMode, Particle, RadiationSource, SpectrumAxisKind,
    SpectrumCompleteness, SpectrumLocation, SpectrumQuantity, TemporalBasis,
)
from radar.core.units import Unit
from radar.validation.parameterized_stopping import ParameterizedStoppingRow


@pytest.fixture(scope="module")
def modules() -> tuple[ModuleType, ModuleType]:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        return (importlib.import_module("attribute_candidate_let"),
                importlib.import_module("validate_candidate_transport"))
    finally:
        sys.path.remove(str(scripts))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _source(z: int, symbol: str, mass: float, source: RadiationSource, *, peak: bool) -> Spectrum1D:
    ordinate = (2. if z == 2 else 3.) if source is RadiationSource.GCR else 5.
    if not peak:
        ordinate *= 7.
    return Spectrum1D(
        x=(1., 2., 4.), y=(ordinate,) * 3, x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=(Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY if peak
                else Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY),
        quantity=((SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX if source is RadiationSource.GCR
                   else SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX) if peak
                  else SpectrumQuantity.DIFFERENTIAL_FLUENCE),
        particle=Particle.HZE, source=source, model="synthetic_attribution",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=((TemporalBasis.SOURCE_MAXIMUM if source is RadiationSource.GCR
                         else TemporalBasis.SEP_PEAK) if peak else TemporalBasis.MISSION_TOTAL),
        location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
        species=ParticleSpecies(symbol, atomic_number=z, mass_number=4 if z == 2 else 197),
        assessment_mode=AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE,
        mission_duration_s=None if peak else 123456789.,
        model_reference=ModelReference("synthetic_attribution", "1", ("analytic fixture",), ("synthetic",)),
        transformation_chain=(TransformationStep("synthetic_source", "1", (
            ("source_tabulated_mass_parameter", str(mass)),
        )),),
        completeness=SpectrumCompleteness.COMPLETE,
    )


@pytest.fixture
def fixture(
    modules: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> dict[str, Any]:
    runner, prior = modules
    for name in ("implementation.py", "production.txt", "generator.py"):
        (tmp_path / name).write_text("synthetic pinned fixture\n", encoding="utf-8")
    source_root = tmp_path / "research/ion_stopping_20261031"
    source_root.mkdir(parents=True)
    (source_root / "source.txt").write_text("synthetic native source\n", encoding="utf-8")
    source_pins = {"source.txt": _sha(source_root / "source.txt")}
    production = {"production.txt": _sha(tmp_path / "production.txt")}
    generators = {"generator.py": _sha(tmp_path / "generator.py")}
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"synthetic": True})
    inventory: list[dict[str, Any]] = [
        {"z": 2, "symbol": "He", "mass_parameter_u": 4., "source_membership": ["GCR", "SEP"]},
        {"z": 79, "symbol": "Au", "mass_parameter_u": 195.1, "source_membership": ["GCR"]},
    ]
    sources = tuple(
        _source(p["z"], p["symbol"], p["mass_parameter_u"], RadiationSource(member.lower()), peak=peak)
        for p in inventory for member in p["source_membership"] for peak in (True, False)
    )
    paths = (tmp_path / "coarse.json", tmp_path / "dense.json")
    policy = runner.MASS_POLICIES[0]
    for path, energies in zip(paths, ((.01, 1., 4.), (.01, .1, 1., 2., 4.)), strict=True):
        raw = {
            "manifest": {"synthetic": True}, "manifest_path": str(manifest),
            "manifest_sha256": _sha(manifest), "source_pins": source_pins,
            "energy_coordinate_compatibility": "synthetic_exact_source_mass",
            "let_convention": "electronic", "projectile_charge_input": "Z",
            "range_origin_energy_mev_per_nucleon": .001,
            "energies_mev_per_nucleon": energies,
            "datasets": {policy: {
                "projectiles": inventory, "backend_configuration": {"synthetic": 1},
                "nonradiative_configuration": {"synthetic": 2}, "materials": [13, 14],
                "rows": [asdict(ParameterizedStoppingRow(
                    p["z"], p["mass_parameter_u"], material, energy,
                    1000. * energy * (1. if p["z"] == 2 else 2.), 0., energy,
                )) for p in inventory for material in (13, 14) for energy in energies],
            }},
        }
        _write(path, raw)

    def validator(path: Path) -> dict[str, Any]:
        return {"input_sha256_after": _sha(path), "numerical_consistency_passed": True,
                "production_resources_sha256_after": production,
                "generator_hashes_after": generators, "source_pins_after": source_pins}

    constants = {
        "SHIELDS": (0., .5), "ENERGY_EDGES": (.01, .1, 1., 2., 4.),
        "LET_EDGES": (.01, .1, 1., 2., 4., 8., 10.), "THRESHOLDS": (.1, 1., 2., 10.),
    }
    for module in (runner, prior):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "IMPLEMENTATION_PATHS", ("implementation.py",))
        monkeypatch.setattr(module, "run_validator", validator)
        monkeypatch.setattr(module, "_source_inputs", lambda _: sources)
        for name, value in constants.items():
            monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(runner, "EXIT_ENERGY_EDGES", (.01, .1, 1., 2., 4.))
    monkeypatch.setattr(runner, "LET_THRESHOLDS", (0., .1, 1., 2., 10.))
    baseline = prior.run_experiment(*paths)
    reference = tmp_path / "baseline.json"
    _write(reference, baseline)
    return {"runner": runner, "prior": prior, "root": tmp_path, "paths": paths,
            "reference": reference, "baseline": json.loads(reference.read_text(encoding="utf-8")),
            "sources": sources, "inventory": inventory, "validator": validator,
            "production": production, "generators": generators, "source_pins": source_pins}


def _run(fixture: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = fixture["runner"].run_attribution(
        fixture["reference"], reference_sha256=_sha(fixture["reference"]),
    )
    return result


def _repin_candidate(fixture: dict[str, Any], grid: int, raw: dict[str, Any]) -> None:
    _write(fixture["paths"][grid], raw)
    fixture["baseline"]["candidate_sha256"][grid] = _sha(fixture["paths"][grid])
    _write(fixture["reference"], fixture["baseline"])


def test_exact_two_species_roster_replay_no_time_or_source_pooling(fixture: dict[str, Any]) -> None:
    result = _run(fixture)
    assert result["source_mode_count"] == 6
    assert result["case_count_per_grid"] == 12
    assert result["full_positive_inventory_selected"] is True
    assert result["reference_sha256"] == _sha(fixture["reference"])
    assert result["candidate_sha256"] == [_sha(p) for p in fixture["paths"]]
    assert result["implementation_sha256_before"] == result["implementation_sha256_after"]
    assert all(result[flag] is False for flag in fixture["runner"].FALSE_FLAGS)
    assert result["error_bound_established"] is False
    assert result["al_path_accuracy_assessed"] is False
    assert "not_error_bound" in result["scope"]
    for grid in result["grids"]:
        groups = {tuple(g["key"]): g for g in grid["aggregates"]}
        assert len(groups) == 8
        gcr = groups["gcr", "peak", .5]
        assert gcr["ion_count"] == 2
        assert gcr["integral_unit"] == "cm^-2 s^-1"
        assert gcr["supported_tail_integrals"] == pytest.approx([15., 15., 14., 10.5, 0.])
        assert gcr["band_tail_integrals"][1] == pytest.approx([2.5, 2.5, 1.5, 0., 0.])
        assert gcr["band_fraction_of_supported_all_species"][1][2] == pytest.approx(1.5 / 14.)
        assert all(row[-1] is None for row in gcr["band_fraction_of_supported_all_species"])
        assert groups["sep", "peak", .5]["ion_count"] == 1
        assert groups["sep", "peak", .5]["supported_tail_integrals"][0] == pytest.approx(15.)
        cumulative = groups["gcr", "cumulative", .5]
        assert cumulative["integral_unit"] == "cm^-2"
        assert cumulative["supported_tail_integrals"] == pytest.approx([7 * x for x in gcr["supported_tail_integrals"]])
        assert [c["mass_parameter_u"] for c in gcr["species_contributions"]] == [4., 195.1]
        assert [c["tail_fraction_of_supported_all_species"][0]
                for c in gcr["species_contributions"]] == pytest.approx([.4, .6])
    assert len(result["species_grid_changes"]) == 12
    assert len(result["aggregate_grid_changes"]) == 8
    assert all(change["absolute_change"] < 1e-12 for group in result["species_grid_changes"]
               for change in group["band_tail_changes"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [None, True, "", "A" * 64, "a" * 63, "0" * 64])
def test_reference_pin_required(fixture: dict[str, Any], value: Any) -> None:
    with pytest.raises(ValueError):
        fixture["runner"].run_attribution(fixture["reference"], reference_sha256=value)


@pytest.mark.parametrize("flag", ["stage4_acceptance_matrix_completed", "physical_accuracy_verified",
                                  "normative_export_allowed", "production_tables_replaced"])
@pytest.mark.parametrize("value", [True, 0, None])
def test_reference_qualification_must_be_literal_false(fixture: dict[str, Any], flag: str, value: Any) -> None:
    fixture["baseline"][flag] = value
    _write(fixture["reference"], fixture["baseline"])
    with pytest.raises(ValueError):
        _run(fixture)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "legacy"), ("full_positive_inventory_selected", False),
    ("full_positive_inventory_selected", 1), ("numerical_table_consistency_passed", False),
    ("all_zero_shield_identities_passed", False), ("implementation_sha256_after", {}),
    ("shield_g_cm2", [0.]), ("energy_edges", [.01, 4.]), ("let_edges", [.01, 10.]),
    ("mass_policy", "explicit_pilot_isotope_diagnostic"), ("source_mode_count", 5),
    ("case_count_per_grid", 11), ("source_fingerprints", []),
])
def test_reference_identity_controls(fixture: dict[str, Any], field: str, value: Any) -> None:
    fixture["baseline"][field] = value
    _write(fixture["reference"], fixture["baseline"])
    with pytest.raises(ValueError):
        _run(fixture)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra", "wrong_mass", "wrong_unit", "wrong_let_unit", "wrong_tail_keys", "wrong_primary", "wrong_let", "wrong_source"])
def test_frozen_reference_case_not_silently_ignored(fixture: dict[str, Any], kind: str) -> None:
    cases = fixture["baseline"]["coarse_cases"]
    if kind == "missing":
        cases.pop()
    elif kind == "duplicate":
        cases[-1] = deepcopy(cases[0])
    elif kind == "extra":
        cases.append(deepcopy(cases[0]))
    elif kind == "wrong_mass":
        cases[0]["mass_parameter_u"] = 4.001
    elif kind == "wrong_unit":
        cases[0]["energy_y_unit"] = Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY.value
    elif kind == "wrong_let_unit":
        cases[0]["let_y_unit"] = Unit.DIFFERENTIAL_LET_FLUENCE.value
    elif kind == "wrong_tail_keys":
        cases[0]["let_tail_integrals"]["999"] = 0.
    elif kind == "wrong_primary":
        cases[0]["primary_ledger"]["source_integral"] += 1.
    elif kind == "wrong_let":
        cases[0]["let_ledger"]["mapped_integral"] += 1.
    else:
        cases[0]["key"][0] = "trapped"
    _write(fixture["reference"], fixture["baseline"])
    with pytest.raises(ValueError):
        _run(fixture)


@pytest.mark.parametrize("what", ["missing_candidate", "missing_pin", "wrong_pin"])
def test_two_candidate_files_and_pins_required(fixture: dict[str, Any], what: str) -> None:
    if what == "missing_candidate":
        fixture["baseline"]["candidate_paths"].pop()
    elif what == "missing_pin":
        fixture["baseline"]["candidate_sha256"].pop()
    else:
        fixture["baseline"]["candidate_sha256"][0] = "0" * 64
    _write(fixture["reference"], fixture["baseline"])
    with pytest.raises(ValueError):
        _run(fixture)


@pytest.mark.parametrize("field,value", [
    ("manifest", {"different": True}), ("manifest_sha256", "0" * 64),
    ("source_pins", {}), ("energy_coordinate_compatibility", "wrong_mass_unit"),
    ("let_convention", "total_not_electronic"), ("projectile_charge_input", "fractional"),
    ("range_origin_energy_mev_per_nucleon", 0.), ("energies_mev_per_nucleon", [.01, 1., 4.]),
])
def test_candidate_semantics_and_grid_not_relabelled(fixture: dict[str, Any], field: str, value: Any) -> None:
    raw = json.loads(fixture["paths"][1].read_text(encoding="utf-8"))
    raw[field] = value
    _repin_candidate(fixture, 1, raw)
    with pytest.raises(ValueError):
        _run(fixture)


@pytest.mark.parametrize("field", ["projectiles", "backend_configuration", "nonradiative_configuration", "materials"])
def test_candidate_dataset_configuration_exact(fixture: dict[str, Any], field: str) -> None:
    raw = json.loads(fixture["paths"][1].read_text(encoding="utf-8"))
    raw["datasets"][fixture["runner"].MASS_POLICIES[0]][field] = []
    _repin_candidate(fixture, 1, raw)
    with pytest.raises(ValueError):
        _run(fixture)


def test_upper_let_window_must_cover_entire_candidate_support(fixture: dict[str, Any]) -> None:
    raw = json.loads(fixture["paths"][0].read_text(encoding="utf-8"))
    rows = raw["datasets"][fixture["runner"].MASS_POLICIES[0]]["rows"]
    for row in rows:
        if row["material_z"] == 14:
            row["stopping_electronic_mev_cm2_g"] = 11000.
    _repin_candidate(fixture, 0, raw)
    with pytest.raises(ValueError, match="window truncates"):
        _run(fixture)


@pytest.mark.parametrize("field,value", [
    ("input_sha256_after", "0" * 64), ("numerical_consistency_passed", False),
    ("numerical_consistency_passed", 1), ("production_resources_sha256_after", {}),
])
def test_prior_validator_result_required(
    fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch, field: str, value: Any,
) -> None:
    def validator(path: Path) -> dict[str, Any]:
        result: dict[str, Any] = fixture["validator"](path)
        return {**result, field: value}
    monkeypatch.setattr(fixture["runner"], "run_validator", validator)
    with pytest.raises(ValueError):
        _run(fixture)


@pytest.mark.parametrize("change", ["warning", "ordinate", "mode", "mass_trace", "missing", "duplicate"])
def test_full_source_fingerprints_replayed(
    fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    sources = list(fixture["sources"])
    if change == "warning":
        sources[0] = replace(sources[0], incomplete_reasons=("new warning",), completeness=SpectrumCompleteness.INCOMPLETE)
    elif change == "ordinate":
        sources[0] = replace(sources[0], y=(2., 2., 2.001))
    elif change == "mode":
        sources.reverse()
    elif change == "mass_trace":
        sources[0] = replace(sources[0], transformation_chain=(TransformationStep("changed_mass", "1"),))
    elif change == "missing":
        sources.pop()
    else:
        sources[-1] = sources[0]
    monkeypatch.setattr(fixture["runner"], "_source_inputs", lambda _: tuple(sources))
    with pytest.raises(ValueError, match="fingerprints"):
        _run(fixture)


@pytest.mark.parametrize("target", ["reference", "coarse", "dense", "manifest", "implementation", "production", "generator", "source"])
def test_concurrent_mutation_is_rejected(
    fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    root = fixture["root"]
    paths = {"reference": fixture["reference"], "coarse": fixture["paths"][0], "dense": fixture["paths"][1],
             "manifest": root / "manifest.json", "implementation": root / "implementation.py",
             "production": root / "production.txt", "generator": root / "generator.py",
             "source": root / "research/ion_stopping_20261031/source.txt"}
    original = fixture["runner"]._case
    done = False

    def changed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal done
        result: dict[str, Any] = original(*args, **kwargs)
        if not done:
            done = True
            path = paths[target]
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(fixture["runner"], "_case", changed)
    with pytest.raises(ValueError):
        _run(fixture)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_mass", "wrong_unit", "wrong_symbol", "thresholds", "edges", "missing_band", "missing_tail", "nan"])
def test_aggregation_refuses_incomplete_or_relabelled_cases(fixture: dict[str, Any], kind: str) -> None:
    result = _run(fixture)
    cases = deepcopy(result["grids"][0]["cases"])
    if kind == "missing":
        cases.pop()
    elif kind == "duplicate":
        cases[-1] = cases[0]
    elif kind == "wrong_mass":
        cases[0]["mass_parameter_u"] = 197.
    elif kind == "wrong_unit":
        cases[0]["integral_unit"] = "cm^-2"
    elif kind == "wrong_symbol":
        cases[0]["symbol"] = "Other"
    elif kind == "thresholds":
        cases[0]["attribution"]["thresholds"] = (.0, .2, 1., 2., 10.)
    elif kind == "edges":
        cases[0]["attribution"]["energy_edges"] = (.01, .2, 1., 2., 4.)
    elif kind == "missing_band":
        cases[0]["attribution"]["band_tail_integrals"] = cases[0]["attribution"]["band_tail_integrals"][:-1]
    elif kind == "missing_tail":
        cases[0]["attribution"]["full_supported_tail_integrals"] = (0.,)
    else:
        cases[0]["attribution"]["full_supported_tail_integrals"] = (float("nan"), .0, .0, .0, .0)
    with pytest.raises(ValueError):
        fixture["runner"]._aggregate(cases, fixture["inventory"])


def test_zero_denominators_and_new_positive_are_not_hidden(modules: tuple[ModuleType, ModuleType]) -> None:
    runner, _ = modules
    left: list[dict[str, Any]] = [{"key": ["gcr", "peak", .1], "integral_unit": "cm^-2 s^-1",
             "band_tail_integrals": [[0.] * len(runner.LET_THRESHOLDS)]}]
    right = deepcopy(left)
    right[0]["band_tail_integrals"][0][0] = 2.
    result = runner._changes(left, right)[0]["band_tail_changes"]
    assert result[0]["absolute_change"] == 2.
    assert result[0]["relative_to_coarse"] is None
    assert result[0]["both_zero"] is False
    assert result[1]["relative_to_coarse"] is None
    assert result[1]["both_zero"] is True
    assert runner._share(0., 0.) is None
    assert runner._share(0., 1.) == 0.


@pytest.mark.parametrize("kind", ["unit", "key", "missing", "shape"])
def test_coarse_dense_groups_are_matched(modules: tuple[ModuleType, ModuleType], kind: str) -> None:
    runner, _ = modules
    left: list[dict[str, Any]] = [{"key": ["gcr", "peak", .1], "integral_unit": "cm^-2 s^-1",
             "band_tail_integrals": [[1.] * len(runner.LET_THRESHOLDS)]}]
    right = deepcopy(left)
    if kind == "unit":
        right[0]["integral_unit"] = "cm^-2"
    elif kind == "key":
        right[0]["key"][0] = "sep"
    elif kind == "missing":
        right.clear()
    else:
        right[0]["band_tail_integrals"][0].pop()
    with pytest.raises(ValueError):
        runner._changes(left, right)


def test_existing_output_never_overwritten(
    modules: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runner, _ = modules
    path = tmp_path / "existing.json"
    path.write_bytes(b"original")
    monkeypatch.setattr(sys, "argv", ["attribution", "reference", "--reference-sha256", "a" * 64,
                                     "--output", str(path)])
    with pytest.raises(FileExistsError):
        runner.main()
    assert path.read_bytes() == b"original"
