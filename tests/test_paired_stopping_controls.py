"""Synthetic tests for matched stopping controls. No external artifact or native runtime."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, localcontext
import importlib
from math import nextafter
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from test_srim_overlap_controls import FakeBackend, FakeMaterial, FakeProjectile


@pytest.fixture(scope="module")
def worker() -> ModuleType:
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return importlib.import_module("build_paired_stopping_controls")


def selected() -> dict[str, Any]:
    result = {}
    for z, mass, upper in ((2, "4", "20000"), (10, "20.2", "100000"),
                           (26, "55.8", "275000"), (92, "238", "1100000")):
        for m in (13, 14):
            a = Decimal(mass)
            result[f"z{z}_m{m}"] = SimpleNamespace(ion_z=z, ion_mass_u=a, target_z=m, raw_sha256="a"*64,
                stopping_unit="MeV/(mg/cm2)", energy_basis="total_ion_kinetic_energy",
                rows=[SimpleNamespace(energy_total_mev=e, electronic_mev_cm2_mg=Decimal("1.234"),
                                      nuclear_mev_cm2_mg=Decimal("0.0000067"))
                      for e in (a * Decimal(".0099"), a * Decimal(".01"), a, Decimal(upper))])
    return result


class Backend(FakeBackend):
    def dedx(self, p: FakeProjectile, m: FakeMaterial, config: Any) -> Any:
        value = super().dedx(p, m, config)
        if p.energy > 50000 and config.corrections == 0 and self.mode == "normal":
            return value + 2.
        return value


def test_matched_nodes_preserve_raw_precision_units_and_duplicates(worker: ModuleType) -> None:
    tables = selected()
    requests = worker.build_requests(tables)
    with localcontext() as ctx:
        ctx.prec = 2
        assert worker.build_requests(tables) == requests
    for z, expected in ((2, 5000.), (10, 4950.495049504951), (26, 4928.31541218638), (92, 4621.848739495798)):
        for m in (13, 14):
            group = [r for r in requests if (r["z"], r["material_z"]) == (z, m)]
            assert len(group) == len({r["energy_mev_per_nucleon"] for r in group})
            common = [r for r in group if r["region"] == "common"]
            assert [r["energy_mev_per_nucleon"] for r in common] == [.01, 1., expected]
            assert [r["native_row_index"] for r in common] == [1, 2, 3]
            assert all(r["srim_electronic_mev_cm2_g"] == 1234. and r["srim_nuclear_mev_cm2_g"] == .0067
                       and r["upper_coarse_node"] is True for r in common)
            above = [r for r in group if r["region"] == "catima_only"]
            assert all(r["native_row_index"] is None and r["native_total_energy_mev_decimal"] is None
                       and r["srim_electronic_mev_cm2_g"] is None and r["srim_nuclear_mev_cm2_g"] is None
                       and r["energy_mev_per_nucleon"] > expected for r in above)


def test_upper_grid_is_nested_with_binary64_radiative_guards(worker: ModuleType) -> None:
    nodes = worker.upper_nodes(5000.)
    for e in (10000., nextafter(50000., -float("inf")), 50000., nextafter(50000., float("inf")), 79432.82347242821):
        assert nodes[e] is True
    numbered = [e for e in nodes if e not in worker.anchors.FIXED.values()]
    assert [nodes[e] for e in numbered] == [bool(i % 2) for i in range(len(numbered))]
    assert max(nodes) == 79432.82347242821 and 100000 not in nodes and min(nodes) > 5000


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "source_a", "units", "basis", "hash", "material_grid",
                                      "nonpositive", "negative", "unsorted", "native_upper", "top", "nan"))
def test_bad_inventory_fails_closed(worker: ModuleType, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    tables = selected()
    table = tables["z2_m13"]
    if mutation == "missing":
        tables.pop("z2_m13")
    elif mutation == "duplicate":
        tables["extra"] = table
    elif mutation == "source_a":
        table.ion_mass_u = Decimal("4.1")
    elif mutation == "units":
        table.stopping_unit = "MeV cm2/g"
    elif mutation == "basis":
        table.energy_basis = "per_nucleon"
    elif mutation == "hash":
        table.raw_sha256 = "bad"
    elif mutation == "material_grid":
        table.rows[2].energy_total_mev = Decimal("4.01")
    elif mutation == "nonpositive":
        table.rows[1].electronic_mev_cm2_mg = Decimal(0)
    elif mutation == "negative":
        table.rows[1].nuclear_mev_cm2_mg = Decimal(-1)
    elif mutation == "nan":
        table.rows[1].energy_total_mev = Decimal("NaN")
    elif mutation == "unsorted":
        table.rows[1], table.rows[2] = table.rows[2], table.rows[1]
    elif mutation == "native_upper":
        table.rows[-1].energy_total_mev = Decimal("24000")
    else:
        monkeypatch.setattr(worker.anchors, "TOP", 79432.823472)
    with pytest.raises(ValueError):
        worker.build_requests(tables)


def test_native_sample_components_and_low_energy_direct_closure(worker: ModuleType) -> None:
    requests = worker.build_requests(selected())
    backend = Backend(worker.native)
    report = worker.generate_samples(backend, requests, {13: 166., 14: 173.})
    assert [(p.z, p.mass, p.q) for p in backend.projectiles] == [(2, 4., 2), (10, 20.2, 10), (26, 55.8, 26), (92, 238., 92)]
    for row, req in zip(report["rows"], requests, strict=True):
        assert {k: row[k] for k in req} == req
        e = req["energy_mev_per_nucleon"]
        assert row["catima_electronic_mev_cm2_g"] == pytest.approx(100.+e, rel=1e-14)
        assert row["catima_radiative_mev_cm2_g"] == (2. if e > 50000 else 0.)


@pytest.mark.parametrize("mode", ("nan", "inf", "negative", "boolean", "config_mutation", "material_mutation", "bad_closure"))
def test_bad_backend_rejected(worker: ModuleType, mode: str) -> None:
    with pytest.raises(ValueError):
        worker.generate_samples(Backend(worker.native, mode), worker.build_requests(selected()), {13: 166., 14: 173.})


@pytest.fixture(scope="module")
def example(worker: ModuleType) -> tuple[dict[str, Any], ...]:
    requests = worker.build_requests(selected())
    samples = worker.generate_samples(Backend(worker.native), requests, {13: 166., 14: 173.})
    baseline = {k: samples[k] for k in ("materials", "backend_configuration", "nonradiative_configuration")}
    baseline["rows"] = [{"z": r["z"], "material_z": r["material_z"], "energy_mev_per_nucleon": r["energy_mev_per_nucleon"],
            **{f"stopping_{c}_mev_cm2_g": r[f"catima_{c}_mev_cm2_g"] for c in worker.COMPONENTS}}
        for r in samples["rows"] if (r["region"] == "common" and r["native_row_index"] == 3)
        or r["energy_mev_per_nucleon"] in worker.anchors.FIXED.values()]
    state = {"snapshot": "test"}
    plan = {"requests": requests, "bundle_sha256": "fixture"}
    report = {"schema_version": worker.SCHEMA, "status": "INCOMPLETE", **worker.FLAGS, "stopping_units": "whole_ion_MeV_cm2_g",
        "python": "fixture", "backend_info_literal": "fixture", "snapshot_before": state, "snapshot_after": state,
        "bundle_sha256": "fixture", "row_count": len(requests), **samples}
    worker.validate_report(report, plan, state, baseline)
    return report, plan, state, baseline


@pytest.mark.parametrize("mutation", ("schema", "promote", "units", "digest", "before", "after", "rows", "count", "mass", "region",
    "factor1000", "fake_upper", "nuclear", "electronic", "negative", "inf", "radiative_low", "radiative_high", "anchor", "config", "material"))
def test_validation_rejects_changes(worker: ModuleType, example: tuple[dict[str, Any], ...], mutation: str) -> None:
    report, plan, state, baseline = deepcopy(example)
    if mutation == "schema":
        report["schema_version"] = "v0"
    elif mutation == "promote":
        report["release_ready"] = True
    elif mutation == "units":
        report["stopping_units"] = "per_nucleon"
    elif mutation == "digest":
        report["bundle_sha256"] = "bad"
    elif mutation in ("before", "after"):
        report["snapshot_"+mutation] = {"other": "value"}
    elif mutation == "rows":
        report["rows"].pop()
    elif mutation == "count":
        report["row_count"] -= 1
    elif mutation == "mass":
        report["rows"][0]["mass_parameter_u_decimal"] = "4.0"
    elif mutation == "region":
        report["rows"][0]["region"] = "catima_only"
    elif mutation == "factor1000":
        report["rows"][0]["srim_electronic_mev_cm2_g"] /= 1000
    elif mutation == "fake_upper":
        report["rows"][-1]["srim_electronic_mev_cm2_g"] = 0.0
    elif mutation == "nuclear":
        report["rows"][0]["catima_nuclear_mev_cm2_g"] += 3
    elif mutation in ("electronic", "negative", "inf"):
        report["rows"][0]["catima_electronic_mev_cm2_g"] = {"electronic": 3., "negative": -1., "inf": float("inf")}[mutation]
    elif mutation in ("radiative_low", "radiative_high"):
        row = report["rows"][0 if mutation == "radiative_low" else -1]
        row["catima_total_mev_cm2_g"] = row["catima_nonradiative_mev_cm2_g"] + (1. if mutation == "radiative_low" else 0.)
        row["catima_radiative_mev_cm2_g"] = row["catima_total_mev_cm2_g"] - row["catima_nonradiative_mev_cm2_g"]
    elif mutation == "anchor":
        baseline["rows"][0]["stopping_electronic_mev_cm2_g"] += 1
    elif mutation == "config":
        report["nonradiative_configuration"] = {**report["nonradiative_configuration"], "corrections": 0}
    else:
        report["materials"] = []
    with pytest.raises(ValueError):
        worker.validate_report(report, plan, state, baseline)


def test_exclusive_output_preflight(worker: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_bytes(b"original")
    monkeypatch.setattr(sys, "argv", ["worker", "--output", str(output)])
    monkeypatch.setattr(worker, "generate", lambda: pytest.fail("No native call after existing-file preflight."))
    with pytest.raises(FileExistsError):
        worker.main()
    assert output.read_bytes() == b"original"


def test_read_verified_reconstructs_plan_and_hash(worker: ModuleType, example: tuple[dict[str, Any], ...],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report, plan, state, baseline = example
    path, base = tmp_path / "controls.json", tmp_path / "anchors.json"
    worker.shared.write_new(path, report)
    worker.shared.write_new(base, baseline)
    monkeypatch.setattr(worker, "reconstruct_plan", lambda: plan)
    monkeypatch.setattr(worker, "snapshot", lambda: state)
    monkeypatch.setattr(worker, "ANCHOR_PATH", base)
    monkeypatch.setattr(worker, "ANCHOR_SHA", worker.native.sha256(base))
    sha = worker.native.sha256(path)
    assert worker.read_verified(path, sha) == report
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        worker.read_verified(path, sha)
