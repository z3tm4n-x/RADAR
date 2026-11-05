"""High-band overlap contracts; no native backend is loaded in these tests."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from test_srim_overlap_controls import FakeBackend, request


@pytest.fixture(scope="module")
def worker() -> ModuleType:
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return importlib.import_module("generate_srim_upper_overlap_controls")


def requests() -> list[dict[str, Any]]:
    return [
        request(total="4000"),
        request(total="12000", index=1),
        request(total="20000", index=2),
        request(target=14, total="20000"),
        request(26, "55.8", total="275000"),
    ]


def test_exact_nodes_literal_mass_no_extra_mass_factor(worker: ModuleType) -> None:
    backend = FakeBackend(worker.native)
    output = worker.generate_samples(backend, requests(), elemental_i_ev={13: 166.0, 14: 173.0})
    assert [(p.z, p.mass, p.q) for p in backend.projectiles] == [(2, 4.0, 2), (26, 55.8, 26)]
    assert backend.projectiles[0].energies == [1000.0, 3000.0, 5000.0, 5000.0]
    assert backend.projectiles[1].energies == [float(Fraction(1375000, 279))]
    for row in output["rows"]:
        assert row["stopping_electronic_mev_cm2_g"] == pytest.approx(
            100 + row["energy_mev_per_nucleon"]
        )
        assert row["stopping_radiative_mev_cm2_g"] == 0
        assert row["stopping_total_mev_cm2_g"] == row["stopping_nonradiative_mev_cm2_g"]
    assert output["low_energy_electronic_closure_count"] == 0
    assert output["backend_configuration"]["z_effective"] == 1
    assert output["nonradiative_configuration"]["corrections"] == 8


@pytest.mark.parametrize(
    "field,value",
    [
        ("z", True),
        ("z", 1),
        ("z", 93),
        ("material_z", True),
        ("material_z", 12),
        ("mass_parameter_u_decimal", "301"),
        ("mass_parameter_u_decimal", "1.9"),
        ("mass_parameter_u_decimal", "4e0"),
        ("mass_parameter_u_decimal", "NaN"),
        ("energy_total_mev_decimal", "4000e0"),
        ("energy_total_mev_decimal", "0"),
        ("energy_total_mev_decimal", "3999.999999999999999999999"),
        ("energy_total_mev_decimal", "20000.0000000000000000001"),
        ("energy_mev_per_nucleon", 1000),
        ("energy_mev_per_nucleon", True),
        ("energy_mev_per_nucleon", float("nan")),
        ("energy_mev_per_nucleon", 1001.0),
        ("native_row_index", -1),
        ("native_row_index", True),
        ("raw_sha256", "A" * 64),
        ("case_name", "../case"),
        ("case_name", ""),
    ],
)
def test_malformed_or_out_of_band_request_rejected(
    worker: ModuleType, field: str, value: Any
) -> None:
    row = requests()[0]
    row[field] = value
    with pytest.raises(ValueError):
        worker.validate_requests([row])


@pytest.mark.parametrize("mutation", ("empty", "duplicate", "reversed", "index", "sha", "extra"))
def test_request_inventory_order_and_identity(worker: ModuleType, mutation: str) -> None:
    rows = requests()
    if mutation == "empty":
        rows = []
    elif mutation == "duplicate":
        rows[1] = rows[0].copy()
    elif mutation == "reversed":
        rows.reverse()
    elif mutation == "index":
        rows[1]["native_row_index"] = 0
    elif mutation == "sha":
        rows[1]["raw_sha256"] = "b" * 64
    else:
        rows[0]["extra"] = 1
    with pytest.raises(ValueError):
        worker.validate_requests(rows)


@pytest.mark.parametrize(
    "mode",
    (
        "nan",
        "inf",
        "negative",
        "boolean",
        "config_mutation",
        "material_mutation",
        "negative_electronic",
        "radiative",
    ),
)
def test_changed_backend_or_invalid_stopping_rejected(worker: ModuleType, mode: str) -> None:
    with pytest.raises(ValueError):
        worker.generate_samples(
            FakeBackend(worker.native, mode), requests(), elemental_i_ev={13: 166.0, 14: 173.0}
        )


@pytest.mark.parametrize("values", [None, {13: 166.0}, {13: True, 14: 173.0}, {13: 166.0, 14: 0.0}])
def test_material_potential_metadata_rejected(worker: ModuleType, values: Any) -> None:
    with pytest.raises(ValueError):
        worker.generate_samples(FakeBackend(worker.native), requests(), elemental_i_ev=values)


def inventory() -> tuple[dict[str, Any], tuple[Any, ...]]:
    selected, catalog = {}, []
    with localcontext() as context:
        context.prec = 80
        for z in (*range(2, 85), 92):
            mass = {2: Decimal(4), 26: Decimal("55.8"), 92: Decimal(238)}.get(z, Decimal(z * 2))
            for target in (13, 14):
                name = f"z{z}_m{target}"
                density = Decimal("2.70" if target == 13 else "2.3290")
                catalog.append(
                    SimpleNamespace(
                        ion_z=z, ion_mass_u=mass, target_z=target, density_g_cm3=density
                    )
                )
                rows = tuple(
                    SimpleNamespace(
                        energy_total_mev=mass * e,
                        electronic_mev_cm2_mg=Decimal("0.2"),
                        nuclear_mev_cm2_mg=Decimal("0.001"),
                        printed_tokens=(str(mass * e), "MeV", "0.2", "0.001"),
                    )
                    for e in (999, 1000, 2999, 3000, 4400, 5000)
                )
                selected[name] = SimpleNamespace(
                    ion_z=z,
                    ion_mass_u=mass,
                    target_z=target,
                    density_g_cm3=density,
                    stopping_unit="MeV/(mg/cm2)",
                    energy_basis="total_ion_kinetic_energy",
                    rows=rows,
                    raw_sha256="a" * 64,
                )
    return selected, tuple(catalog)


def test_complete_exact_roster_and_fractional_context(worker: ModuleType) -> None:
    selected, catalog = inventory()
    normal = worker.planner.build_inventory(selected, catalog)
    with localcontext() as context:
        context.prec = 2
        assert worker.planner.build_inventory(selected, catalog) == normal
    assert normal["ion_count"] == 84 and normal["selected_native_table_count"] == 168
    assert normal["matched_row_count"] == 168 * 5 and normal["excluded_row_count"] == 168
    assert all(t["maximum_mev_per_source_a"] == 5000 for t in normal["tables"])
    worker.validate_requests(normal["requests"])


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "missing_catalog",
        "duplicate_catalog",
        "mass",
        "z",
        "density",
        "units",
        "energy_basis",
        "missing_band",
        "above_band",
    ),
)
def test_inventory_fail_closed(worker: ModuleType, mutation: str) -> None:
    selected, catalog = inventory()
    first = next(iter(selected.values()))
    if mutation == "missing":
        selected.pop(next(iter(selected)))
    elif mutation == "extra":
        selected["extra"] = first
    elif mutation == "missing_catalog":
        catalog = catalog[:-1]
    elif mutation == "duplicate_catalog":
        catalog = (*catalog[:-1], catalog[0])
    elif mutation == "mass":
        first.ion_mass_u += 1
    elif mutation == "z":
        first.ion_z = 1
    elif mutation == "density":
        first.density_g_cm3 += 1
    elif mutation == "units":
        first.stopping_unit = "MeV/(g/cm2)"
    elif mutation == "energy_basis":
        first.energy_basis = "per_nucleon"
    elif mutation == "missing_band":
        first.rows = first.rows[:1]
    else:
        first.rows[-1].energy_total_mev += 1
    with pytest.raises(ValueError):
        worker.planner.build_inventory(selected, catalog)


def test_component_comparison_unit_conversion_and_bands(worker: ModuleType) -> None:
    selected, catalog = inventory()
    plan = worker.planner.build_inventory(selected, catalog)
    native = worker.generate_samples(
        FakeBackend(worker.native), plan["requests"], elemental_i_ev={13: 166.0, 14: 173.0}
    )
    rows = worker.planner.compare_rows(selected, plan["requests"], native["rows"])
    assert len(rows) == 840
    assert {row["energy_band"] for row in rows} == {"[1000,3000)", "[3000,4400)", "[4400,5000]"}
    for row in rows:
        electronic = row["components"]["electronic"]
        assert electronic["srim_mev_cm2_g"] == 200
        assert row["components"]["nuclear"]["srim_mev_cm2_g"] == 1
        assert row["components"]["nonradiative"]["srim_mev_cm2_g"] == 201
        assert electronic["catima_over_srim_minus_one"] == pytest.approx(
            (100 + row["energy_mev_per_nucleon"]) / 200 - 1
        )
    summary = worker.shared.summarize(rows)
    assert any(r["group"] == ["band", "13", "[4400,5000]"] for r in summary)
    selected["z2_m13"].rows[1].nuclear_mev_cm2_mg = Decimal(0)
    assert (
        worker.planner.compare_rows(selected, plan["requests"], native["rows"])[0]["components"][
            "nuclear"
        ]["catima_over_srim_minus_one"]
        is None
    )


def report_fixture(
    worker: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    plan = {"schema_version": "radar.srim_catima_upper_overlap_requests.v1", "requests": requests()}
    path = tmp_path / "requests.json"
    worker.shared.write_new(path, plan)
    sha = worker.shared.digest(path.read_bytes())
    monkeypatch.setattr(worker.planner, "reconstruct_plan", lambda: ({}, deepcopy(plan)))
    monkeypatch.setattr(worker, "snapshot", lambda: {"fixed": "snapshot"})
    monkeypatch.setattr(worker.native, "load_backend", lambda: FakeBackend(worker.native))
    monkeypatch.setattr(worker, "_elemental_i_values", lambda: {13: 166.0, 14: 173.0})
    return path, sha, plan, worker.generate_controls(path, sha)


def test_generate_and_read_only_report_replay(
    worker: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha, plan, report = report_fixture(worker, tmp_path, monkeypatch)
    monkeypatch.setattr(
        worker.native,
        "load_backend",
        lambda: pytest.fail("Read-only replay must not load native backend"),
    )
    worker.validate_report(report, path, sha, plan)
    assert report["request_count"] == 5 and report["status"] == "INCOMPLETE"
    assert all(report[flag] is False for flag in worker.FLAGS)
    assert report["snapshot_before"] == report["snapshot_after"]


@pytest.mark.parametrize(
    "mutation",
    (
        "hash",
        "plan",
        "schema",
        "count",
        "flag",
        "policy",
        "bundle",
        "snapshot",
        "config",
        "material",
        "closure",
        "rows",
        "order",
        "nuclear",
        "decomposition",
        "radiative",
        "nonfinite",
    ),
)
def test_retained_report_rejects_changes(
    worker: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path, sha, plan, report = report_fixture(worker, tmp_path, monkeypatch)
    if mutation == "hash":
        sha = "0" * 64
    elif mutation == "plan":
        plan["requests"].pop()
    elif mutation in ("schema", "count", "flag", "policy", "bundle", "snapshot"):
        key, value = {
            "schema": ("schema_version", "other"),
            "count": ("request_count", 4),
            "flag": ("normative_export_allowed", True),
            "policy": ("mass_policy", "rounded"),
            "bundle": ("bundle_sha256", "0" * 64),
            "snapshot": ("snapshot_after", {}),
        }[mutation]
        report[key] = value
    elif mutation == "config":
        report["backend_configuration"]["z_effective"] = 2
    elif mutation == "material":
        report["materials"][0]["density_g_cm3"] = 2.70
    elif mutation == "closure":
        report["low_energy_electronic_closure_count"] = True
    elif mutation == "rows":
        report["rows"].pop()
    elif mutation == "order":
        report["rows"].reverse()
    else:
        field = {
            "nuclear": "nuclear",
            "decomposition": "electronic",
            "radiative": "radiative",
            "nonfinite": "total",
        }[mutation]
        report["rows"][0][f"stopping_{field}_mev_cm2_g"] = (
            float("nan") if mutation == "nonfinite" else 17.0
        )
    with pytest.raises(ValueError):
        worker.validate_report(report, path, sha, plan)


def test_generation_rechecks_request_and_snapshot(
    worker: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha, _, _ = report_fixture(worker, tmp_path, monkeypatch)
    values = iter(({"before": 1}, {"after": 2}))
    monkeypatch.setattr(worker, "snapshot", lambda: next(values))
    with pytest.raises(ValueError, match="snapshot"):
        worker.generate_controls(path, sha)
