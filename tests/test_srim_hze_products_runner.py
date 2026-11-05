"""Lightweight public-runner checks; mocked source generation, real SRIM operators."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from radar.core.products import SpectrumProduct
from radar.core.project import MissionConfig
from radar.core.spectra import ParticleSpecies
from radar.core.types import AssessmentMode, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from test_srim_hze_pipeline import LET_EDGES, _product, _table


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/run_srim_hze_products.py"
    spec = importlib.util.spec_from_file_location("srim_product_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup_runner(runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    bundle = tmp_path / "fake_bundle.zip"
    bundle.write_bytes(b"not a real SRIM bundle; loader is mocked in this test")
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    state: dict[str, Any] = {"products": (_product(),), "tables": {26: _table()},
                             "loaded": [], "source_calls": [], "pins": {"synthetic.py": "0" * 64}}

    def load(path: Path, *, expected_sha256: str) -> SimpleNamespace:
        assert path == bundle and expected_sha256 == sha
        state["loaded"].append((path, expected_sha256))
        return SimpleNamespace(tables_by_z=state["tables"])

    def sources(mission: MissionConfig, refinement: int) -> tuple[SpectrumProduct, ...]:
        state["source_calls"].append((mission, refinement))
        return tuple(state["products"])

    monkeypatch.setattr(runner, "load_srim_hze_dataset", load)
    monkeypatch.setattr(runner, "source_products", sources)
    monkeypatch.setattr(runner, "_code_pins", lambda: dict(state["pins"]))
    monkeypatch.setattr(runner, "SHIELDS", (0., 1.))
    monkeypatch.setattr(runner, "LET_EDGES", LET_EDGES)
    monkeypatch.setattr(runner, "ENERGY_EDGES", (1., 10., 100.))
    return {"state": state, "bundle": bundle, "sha": sha,
            "mission": MissionConfig(launch_year=2026, lifetime_years=5, sep_exceedance_probability=.1)}


def _run(runner: ModuleType, setup: dict[str, Any], **changes: Any) -> dict[str, Any]:
    kwargs = {"mission": setup["mission"], "refinement": 1, **changes}
    return dict(runner.run(setup["bundle"], setup["sha"], **kwargs))


def test_real_operators_return_json_serializable_products_and_unchanged_flags(
    runner: ModuleType, setup_runner: dict[str, Any],
) -> None:
    report = _run(runner, setup_runner)
    restored = json.loads(json.dumps(report, allow_nan=False))
    assert restored["schema_version"] == "radar.srim_hze_external_products.v1"
    assert restored["status"] == "INCOMPLETE"
    assert restored["source_product_count"] == 1 and restored["case_count"] == 2
    assert restored["source_grid_refinement"] == 1
    assert restored["bundle_sha256"] == setup_runner["sha"]
    assert restored["mission"]["launch_year"] == 2026
    assert restored["implementation_and_input_sha256_before"] == restored["implementation_and_input_sha256_after"]
    for flag in ("normative_export_allowed", "orbit_calculation_performed", "dose_or_see_calculated",
                 "default_profile_replaced"):
        assert restored[flag] is False
    for case in restored["cases"]:
        assert case["energy_product"]["kind"] == "mission_fluence"
        assert case["let_product"]["kind"] == "mission_let_fluence"
        assert case["energy_product"]["spectrum"]["completeness"] == "incomplete"
        assert case["let_product"]["spectrum"]["completeness"] == "incomplete"
        assert case["primary_ledger"]["source_integral"] == pytest.approx(140.)
        assert case["let_ledger"]["mapped_integral"] == pytest.approx(140.)
        assert case["normative_export_allowed"] is False
        assert case["incomplete_reasons"]
    assert setup_runner["state"]["source_calls"] == [(setup_runner["mission"], 1)]


def test_summaries_keep_two_sources_two_modes_and_shields_separate(
    runner: ModuleType, setup_runner: dict[str, Any],
) -> None:
    products = []
    expected = {}
    for i, (source, peak) in enumerate((
        (RadiationSource.GCR, False), (RadiationSource.GCR, True),
        (RadiationSource.SEP, False), (RadiationSource.SEP, True),
    ), 1):
        products.append(_product(source=source, peak=peak, y=(2. * i,) * 3))
        expected[source.value, "peak" if peak else "cumulative"] = 140. * i
    setup_runner["state"]["products"] = tuple(products)
    report = _run(runner, setup_runner)
    assert report["source_product_count"] == 4 and report["case_count"] == 8
    assert len(report["summaries"]) == 8
    for summary in report["summaries"]:
        total = expected[summary["source"], summary["mode"]]
        assert summary["thickness_g_cm2"] in (0., 1.)
        assert summary["case_count"] == 1 and summary["unavailable_cases"] == 0
        assert summary["primary"]["source_integral"] == pytest.approx(total)
        assert summary["primary"]["supported_exit_integral"] == pytest.approx(total)
        assert summary["let"]["mapped_integral"] == pytest.approx(total)
        assert summary["outside_al_fraction_of_declared_source"] == 0
        assert summary["unresolved_low_fraction_of_declared_source"] == 0


def test_known_source_fractions_use_declared_integral_not_grid_counts(
    runner: ModuleType, setup_runner: dict[str, Any],
) -> None:
    setup_runner["state"]["products"] = (
        _product(x=(.1, 1., 10., 100., 200.), y=(2.,) * 5),
    )
    report = _run(runner, setup_runner)
    zero, shielded = report["summaries"]
    assert zero["thickness_g_cm2"] == 0
    assert zero["primary"]["source_integral"] == pytest.approx(399.8)
    assert zero["outside_al_fraction_of_declared_source"] == 0
    assert zero["let"]["outside_si_domain_integral"] == pytest.approx(201.8)
    assert shielded["thickness_g_cm2"] == 1
    assert shielded["primary"]["outside_al_domain_integral"] == pytest.approx(201.8)
    assert shielded["outside_al_fraction_of_declared_source"] == pytest.approx(201.8 / 399.8)
    assert shielded["unresolved_low_fraction_of_declared_source"] == pytest.approx((4. / 55.8) / 399.8)


def test_grouped_integrals_add_ions_without_averaging_fractions(
    runner: ModuleType, setup_runner: dict[str, Any],
) -> None:
    setup_runner["state"]["tables"] = {26: _table(), 28: _table("58.7", 28)}
    setup_runner["state"]["products"] = (
        _product(x=(.1, 1., 10., 100., 200.), y=(2.,) * 5),
        _product(species=ParticleSpecies("Ni", 28, 59), trace=("58.7",), y=(6.,) * 3),
    )
    report = _run(runner, setup_runner)
    assert report["source_product_count"] == 2 and report["case_count"] == 4
    shielded = next(r for r in report["summaries"] if r["thickness_g_cm2"] == 1)
    assert shielded["case_count"] == 2
    assert shielded["primary"]["source_integral"] == pytest.approx(399.8 + 420.)
    assert shielded["outside_al_fraction_of_declared_source"] == pytest.approx(201.8 / 819.8)


def test_missing_tables_are_reported_not_fabricated_and_fraction_is_unknown(
    runner: ModuleType, setup_runner: dict[str, Any],
) -> None:
    setup_runner["state"]["tables"] = {}
    report = _run(runner, setup_runner)
    assert len(report["cases"]) == 2
    assert report["cases"][0]["energy_product"] is not None  # zero-shield identity
    assert report["cases"][1]["energy_product"] is None
    for row in report["cases"]:
        assert row["primary_ledger"] is None and row["let_ledger"] is None
        assert row["let_product"] is None and row["incomplete_reasons"]
    for summary in report["summaries"]:
        assert summary["case_count"] == summary["unavailable_cases"] == 1
        assert summary["outside_al_fraction_of_declared_source"] is None
        assert summary["unresolved_low_fraction_of_declared_source"] is None


def test_exact_zero_sampled_sources_are_explicit_and_never_called_complete_zero(
    runner: ModuleType, setup_runner: dict[str, Any],
) -> None:
    products = tuple(_product(source=source, peak=peak, y=(0.,) * 3)
                     for source in (RadiationSource.GCR, RadiationSource.SEP) for peak in (False, True))
    setup_runner["state"]["products"] = products
    report = _run(runner, setup_runner)
    assert report["source_product_count"] == report["case_count"] == 0
    assert report["cases"] == report["summaries"] == []
    zeros = report["zero_sampled_sources_not_transported"]
    assert len(zeros) == 4
    assert {(row["source"], row["mode"]) for row in zeros} == {
        (s.value, m.value) for s in (RadiationSource.GCR, RadiationSource.SEP) for m in AssessmentMode
    }
    assert all(row["reason"] == "exact_zero_on_the_declared_source_grid_only" for row in zeros)
    assert report["status"] == "INCOMPLETE" and not report["normative_export_allowed"]


@pytest.mark.parametrize("symbols", [(), ("Fe", "Fe"), ("Xe",), (" Fe",), ("Fe", "Xe")])
def test_symbol_selection_rejects_empty_duplicate_missing_or_untrimmed_names(
    runner: ModuleType, setup_runner: dict[str, Any], symbols: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="symbols"):
        _run(runner, setup_runner, symbols=symbols)


def test_symbol_selection_requires_a_positive_sampled_source_and_preserves_requested_subset(
    runner: ModuleType, setup_runner: dict[str, Any],
) -> None:
    zero_ni = _product(species=ParticleSpecies("Ni", 28, 59), trace=("58.7",), y=(0.,) * 3)
    setup_runner["state"]["products"] = (_product(), zero_ni)
    report = _run(runner, setup_runner, symbols=("Fe",))
    assert report["requested_symbols"] == ("Fe",)
    assert report["source_product_count"] == 1
    assert report["zero_sampled_sources_not_transported"] == []
    with pytest.raises(ValueError, match="positive source"):
        _run(runner, setup_runner, symbols=("Ni",))


def test_integral_presentation_is_ignored_instead_of_counted_twice(
    runner: ModuleType, setup_runner: dict[str, Any],
) -> None:
    product = _product()
    integral = replace(product, spectrum=replace(product.spectrum,
                       quantity=SpectrumQuantity.INTEGRAL_FLUENCE, y_unit=Unit.INTEGRAL_FLUENCE))
    setup_runner["state"]["products"] = (product, integral)
    report = _run(runner, setup_runner)
    assert report["source_product_count"] == 1 and report["case_count"] == 2


def test_before_after_source_code_snapshot_mutation_is_rejected(
    runner: ModuleType, setup_runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_sources = runner.source_products

    def mutate(mission: MissionConfig, refinement: int) -> tuple[SpectrumProduct, ...]:
        setup_runner["state"]["pins"] = {"synthetic.py": "1" * 64}
        return tuple(old_sources(mission, refinement))

    monkeypatch.setattr(runner, "source_products", mutate)
    with pytest.raises(ValueError, match="changed during calculation"):
        _run(runner, setup_runner)


def test_bundle_mutation_during_calculation_is_rejected(
    runner: ModuleType, setup_runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_sources = runner.source_products

    def mutate(mission: MissionConfig, refinement: int) -> tuple[SpectrumProduct, ...]:
        setup_runner["bundle"].write_bytes(b"changed synthetic bundle")
        return tuple(old_sources(mission, refinement))

    monkeypatch.setattr(runner, "source_products", mutate)
    with pytest.raises(ValueError, match="changed during calculation"):
        _run(runner, setup_runner)


def test_loader_failure_propagates_before_source_generation(
    runner: ModuleType, setup_runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise ValueError("wrong bundle SHA")
    monkeypatch.setattr(runner, "load_srim_hze_dataset", fail)
    with pytest.raises(ValueError, match="wrong bundle SHA"):
        _run(runner, setup_runner)
    assert not setup_runner["state"]["source_calls"]


@pytest.mark.parametrize("factor", [0, -1, 3, "2", None])
def test_refinement_rejects_unsupported_factor(runner: ModuleType, factor: Any) -> None:
    with pytest.raises(ValueError):
        runner._refine((1., 4., 100.), factor)


def test_refinement_keeps_exact_original_nodes_and_adds_only_geometric_midpoints(runner: ModuleType) -> None:
    coarse = (1., 4., 100.)
    assert runner._refine(coarse, 1) is coarse
    dense = runner._refine(coarse, 2)
    assert dense == (1., 2., 4., 20., 100.)
    assert len(dense) == 2 * len(coarse) - 1 and set(coarse) < set(dense)


@pytest.mark.parametrize("factor", [1, 2])
def test_source_builder_passes_refined_grids_and_same_solar_mission_to_both_models(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, factor: int,
) -> None:
    captures: dict[str, Any] = {}
    mission = MissionConfig(launch_year=2026, lifetime_years=5, sep_exceedance_probability=.1)
    solar = SimpleNamespace(months=(SimpleNamespace(wolf_number=10.), SimpleNamespace(wolf_number=20.)),
                            validate_for_mission=lambda value: captures.update(validated_mission=value))
    monkeypatch.setattr(runner, "build_mission_solar_activity", lambda *a, **k: solar)
    monkeypatch.setattr(runner, "ost_wolf_number_cycle_table", lambda *a: "synthetic-wolf-table")
    monkeypatch.setattr(runner, "DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON", (1., 4., 100.))
    monkeypatch.setattr(runner, "DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON", (1., 9., 100.))
    gcr_product, sep_product = _product(source=RadiationSource.GCR), _product()

    class Gcr:
        def __init__(self, **kwargs: Any) -> None:
            captures["gcr_kwargs"] = kwargs
        def calculate(self, value: Any) -> SimpleNamespace:
            captures["gcr_input"] = value
            return SimpleNamespace(products=(gcr_product,))

    class Sep:
        def __init__(self, **kwargs: Any) -> None:
            captures["sep_kwargs"] = kwargs
        def calculate(self, value: Any) -> SimpleNamespace:
            captures["sep_input"] = value
            return SimpleNamespace(products=(sep_product,))

    monkeypatch.setattr(runner, "OstGcrModel", Gcr)
    monkeypatch.setattr(runner, "OstSepModel", Sep)
    products = runner.source_products(mission, factor)
    assert products == (gcr_product, sep_product)
    assert captures["gcr_input"].mission is captures["sep_input"].mission is mission
    assert captures["gcr_input"].solar_activity is solar
    assert captures["gcr_kwargs"]["energy_grid_mev_per_nucleon"] == runner._refine((1., 4., 100.), factor)
    assert captures["sep_kwargs"]["energy_grid_mev"] == runner._refine((1., 9., 100.), factor)
    assert captures["sep_kwargs"]["hze_energy_grid_mev_per_nucleon"] == captures["sep_kwargs"]["energy_grid_mev"]
    assert captures["sep_kwargs"]["monthly_smoothed_wolf_numbers"] == (10., 20.)


def test_cli_creates_exclusive_json_and_passes_options(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    seen: dict[str, Any] = {}

    def fake_run(bundle: Path, sha: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(bundle=bundle, sha=sha, **kwargs)
        return {"case_count": 12, "status": "INCOMPLETE", "normative_export_allowed": False}

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_srim_hze_products.py", "bundle.zip", "--bundle-sha256", "a" * 64,
                                     "--output", str(output), "--launch-year", "2028", "--lifetime-years", "3",
                                     "--sep-probability", "0.2", "--source-grid-refinement", "2", "--symbols", "He,Fe"])
    assert runner.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["case_count"] == 12 and payload["normative_export_allowed"] is False
    assert seen["bundle"] == Path("bundle.zip") and seen["sha"] == "a" * 64
    assert seen["mission"].launch_year == 2028 and seen["mission"].lifetime_years == 3
    assert seen["mission"].sep_exceedance_probability == .2
    assert seen["refinement"] == 2 and seen["symbols"] == ("He", "Fe")
    before = output.read_bytes()
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("must reject before work"))
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == before


def test_cli_exclusive_write_also_rejects_destination_created_during_run(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "race.json"
    def fake_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        output.write_bytes(b"keep concurrent owner")
        return {"case_count": 0}
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["runner", "bundle.zip", "--bundle-sha256", "a" * 64,
                                     "--output", str(output)])
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == b"keep concurrent owner"
