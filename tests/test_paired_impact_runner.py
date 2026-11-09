"""Portable public-pilot checks: synthetic source/controls, no native runtime or network."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from radar.core.project import MissionConfig
from radar.core.spectra import ParticleSpecies, TransformationStep
from radar.core.types import RadiationSource, SpectrumLocation, SpectrumQuantity, TemporalBasis
from test_candidate_transport import _source


@pytest.fixture(scope="module")
def modules() -> tuple[ModuleType, ...]:
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return tuple(importlib.import_module(n) for n in ("run_paired_stopping_impact", "report_paired_stopping_impact",
                                                          "build_paired_stopping_controls", "paired_stopping_impact"))


def synthetic_sources(runner: ModuleType) -> dict[str, Any]:
    products, cases = [], []
    for kind, mode, z in sorted(runner.EXPECTED):
        peak = mode == "peak"
        spectrum = _source(peak=peak, source=RadiationSource(kind), location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
            species=ParticleSpecies(f"Z{z}", z, round(runner.IONS[z])), x=(1., 5000., 12000.), y=(3., 2., .5),
            quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX if peak and kind == "gcr" else
                     SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX if peak else SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            temporal_basis=TemporalBasis.SOURCE_MAXIMUM if peak and kind == "gcr" else
                          TemporalBasis.SEP_PEAK if peak else TemporalBasis.MISSION_TOTAL,
            transformation_chain=(TransformationStep("fixture", "1", (("source_tabulated_mass_parameter", str(runner.IONS[z])),)),))
        products.append(SimpleNamespace(spectrum=spectrum))
        cases.append({"source": kind, "mode": mode, "z": z, "thickness_g_cm2": 0.,
                      "energy_product": {"spectrum": asdict(spectrum)}})
    pins = {"synthetic": "0" * 64}
    retained = {"schema_version": "radar.srim_hze_external_products.v1", "source_grid_refinement": 1,
        "orbit_calculation_performed": False, "normative_export_allowed": False,
        "implementation_and_input_sha256_before": dict(pins), "implementation_and_input_sha256_after": dict(pins),
        "mission": asdict(MissionConfig(launch_year=2026, lifetime_years=5, sep_exceedance_probability=.1)), "cases": cases}
    return {"products": products, "retained": retained, "pins": pins}


@pytest.fixture
def harness(modules: tuple[ModuleType, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    runner = modules[0]
    state = synthetic_sources(runner)
    root = tmp_path
    (root / "outputs").mkdir()
    (root / "scripts").mkdir()
    for name in ("run_paired_stopping_impact.py", "build_paired_stopping_controls.py", "paired_stopping_impact.py"):
        (root / "scripts" / name).write_bytes(b"synthetic code pin")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner.sources, "_code_pins", lambda: dict(state["pins"]))
    monkeypatch.setattr(runner.sources, "source_products", lambda mission, refinement: iter(state["products"]))
    return state


def pin(runner: ModuleType, state: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    path = runner.ROOT / "outputs" / "fixture.json"
    runner.io.write_new(path, state["retained"])
    monkeypatch.setattr(runner, "SOURCE_FILES", {1: (path.name, runner._hash(path))})


def test_exact_pilot_catalog_and_literal_source_masses(modules: tuple[ModuleType, ...], harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    run = modules[0]
    pin(run, harness, monkeypatch)
    sources, pins = run.load_sources(1)
    assert pins == harness["pins"] and len(sources) == 14
    assert set(sources) == {(s, m, z) for s in ("gcr", "sep") for m in ("peak", "cumulative")
                            for z in (2, 10, 26, 92) if (s, z) != ("sep", 92)}
    for (source, mode, z), spectrum in sources.items():
        assert spectrum.assessment_mode.value == mode and spectrum.source.value == source
        assert spectrum.transformation_chain[-1].parameters == (("source_tabulated_mass_parameter", str(run.IONS[z])),)


@pytest.mark.parametrize("mutation", ("before", "after", "version", "grid", "orbit", "normative", "mission", "missing_ref",
    "duplicate_ref", "missing_product", "duplicate_product", "sep_u", "source_y", "zero", "mass", "trace", "location", "source_digest"))
def test_source_gates(modules: tuple[ModuleType, ...], harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    run = modules[0]
    retained = harness["retained"]
    if mutation in ("before", "after"):
        retained[f"implementation_and_input_sha256_{mutation}"] = {}
    elif mutation in ("version", "grid", "orbit", "normative"):
        retained[{"version": "schema_version", "grid": "source_grid_refinement", "orbit": "orbit_calculation_performed",
                  "normative": "normative_export_allowed"}[mutation]] = True if mutation in ("orbit", "normative") else 42
    elif mutation == "mission":
        retained["mission"]["lifetime_years"] = 1
    elif mutation == "missing_ref":
        retained["cases"].pop()
    elif mutation == "duplicate_ref":
        retained["cases"].append(deepcopy(retained["cases"][0]))
    elif mutation == "sep_u":
        retained["cases"].append({**retained["cases"][0], "source": "sep", "z": 92})
    elif mutation == "source_y":
        retained["cases"][0]["energy_product"]["spectrum"]["y"] = (.1, .2, .3)
    elif mutation == "missing_product":
        harness["products"].pop()
    elif mutation == "duplicate_product":
        harness["products"].append(harness["products"][0])
    elif mutation in ("zero", "mass", "trace", "location"):
        p = harness["products"][0]
        changes: dict[str, dict[str, Any]] = {
            "zero": {"y": (0., 0., 0.)}, "location": {"location": SpectrumLocation.ON_ORBIT},
            "trace": {"transformation_chain": ()}, "mass": {"transformation_chain": (
                TransformationStep("synthetic", "1", (("source_tabulated_mass_parameter", "100"),)),)}
        }
        p.spectrum = replace(p.spectrum, **changes[mutation])
    pin(run, harness, monkeypatch)
    if mutation == "source_digest":
        (run.ROOT / "outputs" / "fixture.json").write_bytes(b"changed")
    with pytest.raises(ValueError):
        run.load_sources(1)


def test_factorial_and_distinct_denominators(modules: tuple[ModuleType, ...]) -> None:
    run = modules[0]
    value = run.decomposition({"SS": 10., "CS": 12., "SC": 11., "CC": 16.}, 200.)
    assert value["delta"] == {"Al": 2., "Si": 1., "interaction": 3., "combined": 6.}
    assert value["percent_of_full_source"]["combined"] == 3.
    assert value["percent_of_SS"]["combined"] == 60.
    undefined = run.decomposition({"SS": 0., "CS": 1., "SC": 0., "CC": 0.}, 100.)
    assert all(x is None for x in undefined["percent_of_SS"].values())
    assert undefined["delta"]["interaction"] == -1.


def controls(run: ModuleType) -> dict[str, Any]:
    return {"rows": [{"z": z, "material_z": m, "mass_parameter_u_decimal": str(run.IONS[z]),
        "region": "common" if i < 3 else "catima_only", "upper_coarse_node": i != 3,
        "energy_mev_per_nucleon": e, "srim_electronic_mev_cm2_g": 12. if i < 3 else None,
        "srim_nuclear_mev_cm2_g": 3. if i < 3 else None, "catima_total_mev_cm2_g": 20.,
        "catima_electronic_mev_cm2_g": 16.} for z in run.IONS for m in (13, 14)
        for i, e in enumerate((.01, 100., 5000., 8000., 12000.))]}


def test_matched_physical_inputs_and_coarsening(modules: tuple[ModuleType, ...]) -> None:
    run = modules[0]
    fine, coarse = (run.prepare(controls(run), 26, n) for n in (80, 40))
    for p in (fine, coarse):
        assert p.mass == 55.8
        assert p.srim_al.curve.left == (15., 15.) and p.catima_al.curve.left == (20., 20.)
        assert p.srim_si.left == (.012, .012) and p.catima_si.left == (.016, .016)
    for attr in ("srim_al", "srim_si", "catima_al", "catima_si"):
        assert getattr(fine, attr) == getattr(coarse, attr)
    assert fine.extended_al.curve.energy == (.01, 100., 5000., 8000., 12000.)
    assert coarse.extended_al.curve.energy == (.01, 100., 5000., 12000.)


@pytest.mark.parametrize("field,value", (("region", "catima_only"), ("energy_mev_per_nucleon", 10.), ("mass_parameter_u_decimal", "20")))
def test_target_mismatch_rejected(modules: tuple[ModuleType, ...], field: str, value: Any) -> None:
    run = modules[0]
    table = controls(run)
    next(r for r in table["rows"] if r["z"] == 10 and r["material_z"] == 14)[field] = value
    with pytest.raises(ValueError):
        run.prepare(table, 10, 80)


def test_real_factorial_runner_returns_all_six_shields_in_both_modes(modules: tuple[ModuleType, ...], harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    run, writer, worker, _ = modules
    pin(run, harness, monkeypatch)
    path = run.ROOT / "controls.json"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(worker, "read_verified", lambda p, sha: controls(run))
    report = run.run(path, run._hash(path), 1, 80)
    assert len(report["cases"]) == 84 and report["case_count"] == 84 and report["source_count"] == 14
    assert report["source_absences"] == [{"source": "sep", "z": 92, "reason": "not_in_OST_B9_roster_not_a_zero_spectrum"}]
    for row in report["cases"]:
        assert row["source"] != "sep" or row["z"] != 92
        assert set(row["variants"]) == {"SS", "SC", "CS", "CC"}
        assert row["let_density_unit"] == (run.Unit.DIFFERENTIAL_LET_FLUX if row["mode"] == "peak" else run.Unit.DIFFERENTIAL_LET_FLUENCE).value
    assert all(report[k] is False for k in run.FLAGS if k != "status")
    # Exercise the human-readable report on the real numerical operator with
    # portable synthetic inputs. Identical curves are sufficient for this UI check.
    monkeypatch.setattr(writer, "SOURCE_FILES", {1: ("toy", report["source_reference_sha256"]),
                                                2: ("toy", report["source_reference_sha256"])})
    text = writer.render(report, {**report, "source_refinement": 2},
                         {**report, "upper_grid_points_per_decade_setting": 40}, ("0"*64,)*3)
    assert text.count("\n### ") == 84
    assert text.count("\n| ЛПЭ ≥ ") == 84 * len(report["let_thresholds"])
    assert "| U | 20.0 |" in text and "| — |" in text


def report_stub(run: ModuleType, source_grid: int, upper_grid: int) -> dict[str, Any]:
    return {"schema_version": "radar.paired_protected_external_spectra.v1", **run.FLAGS,
        "source_reference_sha256": run.SOURCE_FILES[source_grid][1], "source_refinement": source_grid,
        "case_count": 84, "source_count": 14, "upper_grid_points_per_decade_setting": upper_grid,
        **{k: [] for k in ("implementation_sha256", "production_and_source_sha256", "energy_edges_mev_per_source_A",
                            "let_edges_mev_cm2_mg", "let_thresholds", "energy_thresholds")},
        "controls_sha256": "0"*64, "cases": [{"source": src, "mode": mode, "z": z, "thickness_g_cm2": d}
            for src, mode, z in sorted(run.EXPECTED) for d in run.SHIELDS]}


@pytest.mark.parametrize("mutation", ("flag", "status", "missing", "duplicate", "sep_u", "refinement", "source_sha", "wrong_tail"))
def test_index_requires_full_pinned_individual_case_roster(modules: tuple[ModuleType, ...], mutation: str) -> None:
    run, writer = modules[:2]
    raw = report_stub(run, 1, 80)
    if mutation == "missing":
        raw["cases"].pop()
    elif mutation == "duplicate":
        raw["cases"][0] = raw["cases"][1]
    elif mutation == "sep_u":
        raw["cases"][0]["source"], raw["cases"][0]["z"] = "sep", 92
    else:
        raw[{"flag": "normative_export_allowed", "status": "status", "refinement": "source_refinement",
             "source_sha": "source_reference_sha256", "wrong_tail": "upper_grid_points_per_decade_setting"}[mutation]] = "wrong"
    with pytest.raises(ValueError):
        writer.index_report(raw, 1, 80)


@pytest.mark.parametrize("field", ("controls_sha256", "implementation_sha256", "production_and_source_sha256", "let_thresholds"))
def test_render_rejects_mixed_comparison_pins(modules: tuple[ModuleType, ...], field: str) -> None:
    run, writer = modules[:2]
    base, dense, coarse = report_stub(run, 1, 80), report_stub(run, 2, 80), report_stub(run, 1, 40)
    coarse[field] = "changed"
    with pytest.raises(ValueError):
        writer.render(base, dense, coarse, ("0"*64,)*3)


@pytest.mark.parametrize("idx", (0, 1))
def test_cli_exclusive_output_before_work(modules: tuple[ModuleType, ...], idx: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "exists"
    path.write_bytes(b"original")
    argv = ["worker", "--output", str(path)]
    if idx == 0:
        argv += ["--controls", "unused", "--sha256", "0"*64]
    else:
        for name in ("base", "dense", "coarse"):
            argv.extend(["--"+name, "unused", "--"+name+"-sha256", "0"*64])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        modules[idx].main()
    assert path.read_bytes() == b"original"
