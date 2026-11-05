"""Bounded anchor inventory/units/threshold contracts; no native extension loaded."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, localcontext
import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from test_srim_overlap_controls import FakeBackend, FakeMaterial, FakeProjectile


@pytest.fixture(scope="module")
def worker() -> ModuleType:
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        return importlib.import_module("generate_high_energy_anchors")


def selected() -> dict[str, Any]:
    output = {}
    for z, mass, total in ((2, "4", "20000"), (10, "20.2", "100000"),
                           (26, "55.8", "275000"), (92, "238", "1100000")):
        for target in (13, 14):
            output[f"z{z}_m{target}"] = SimpleNamespace(ion_z=z, ion_mass_u=Decimal(mass), target_z=target,
                stopping_unit="MeV/(mg/cm2)", energy_basis="total_ion_kinetic_energy", raw_sha256="a"*64,
                rows=(SimpleNamespace(energy_total_mev=Decimal(total), electronic_mev_cm2_mg=Decimal("0.2"),
                                      nuclear_mev_cm2_mg=Decimal("0.000001")),))
    return output


class AnchorBackend(FakeBackend):
    def dedx(self, p: FakeProjectile, m: FakeMaterial, config: Any) -> Any:
        value = super().dedx(p, m, config)
        if self.mode == "normal" and p.energy > 50000 and config.corrections == 0:
            return value + 2.0
        return value


def test_exact_inventory_and_unrounded_top(worker: ModuleType) -> None:
    requests = worker.build_requests(selected())
    assert len(requests) == 48 and sum(r["core_anchor"] for r in requests) == 32
    assert {r["z"] for r in requests} == {2, 10, 26, 92}
    assert all(r["energy_mev_per_nucleon"] == 79432.82347242821
               for r in requests if r["anchor_kind"] == "gcr_last_center")
    assert requests[2]["energy_mev_per_nucleon"] < requests[3]["energy_mev_per_nucleon"] < requests[4]["energy_mev_per_nucleon"]
    assert requests[2]["energy_mev_per_nucleon"] == worker.nextafter(50000.0, -float("inf"))
    assert requests[4]["energy_mev_per_nucleon"] == worker.nextafter(50000.0, float("inf"))
    with localcontext() as context:
        context.prec = 2
        assert worker.build_requests(selected()) == requests


def test_small_native_nuclear_decimal_not_rejected_as_exponent(worker: ModuleType) -> None:
    inventory = selected()
    inventory["z2_m13"].rows[0].nuclear_mev_cm2_mg = Decimal("1.234E-7")
    rows = worker.build_requests(inventory)
    assert rows[0]["srim_reference"]["nuclear_mev_cm2_mg"] == "0.0000001234"
    worker.validate_requests(rows)


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "mass", "units", "basis", "endpoint", "gridtop"))
def test_native_inventory_fails_closed(worker: ModuleType, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    inventory = selected()
    first = next(iter(inventory.values()))
    if mutation == "missing":
        inventory.pop(next(iter(inventory)))
    elif mutation == "duplicate":
        inventory["extra"] = first
    elif mutation == "mass":
        first.ion_mass_u = Decimal("4.1")
    elif mutation == "units":
        first.stopping_unit = "MeV/cm"
    elif mutation == "basis":
        first.energy_basis = "energy_per_nucleon"
    elif mutation == "endpoint":
        first.rows[0].energy_total_mev = Decimal("24000")
    else:
        monkeypatch.setattr(worker, "TOP", 79432.823472)
    with pytest.raises(ValueError):
        worker.build_requests(inventory)


@pytest.mark.parametrize("field,value", [
    ("z", True), ("z", 4), ("material_z", True), ("material_z", 15),
    ("mass_parameter_u_decimal", "4.0"), ("anchor_kind", "other"), ("core_anchor", 1),
    ("energy_mev_per_nucleon", 5000), ("energy_mev_per_nucleon", float("nan")),
    ("energy_mev_per_nucleon", float("inf")), ("energy_mev_per_nucleon", 5001.0),
    ("energy_binary64_hex", "0x0"), ("energy_exact_fraction", "999"),
    ("srim_reference", None),
])
def test_changed_anchor_rejected(worker: ModuleType, field: str, value: Any) -> None:
    requests = worker.build_requests(selected())
    requests[0][field] = value
    with pytest.raises(ValueError):
        worker.validate_requests(requests)


@pytest.mark.parametrize("mutation", ("empty", "missing", "duplicate", "reverse", "extra", "fake_srim", "fixed_energy"))
def test_request_roster_complete(worker: ModuleType, mutation: str) -> None:
    requests = worker.build_requests(selected())
    if mutation == "empty":
        requests = []
    elif mutation == "missing":
        requests.pop()
    elif mutation == "duplicate":
        requests[1] = requests[0].copy()
    elif mutation == "reverse":
        requests.reverse()
    elif mutation == "extra":
        requests[0]["extra"] = 1
    elif mutation == "fake_srim":
        requests[1]["srim_reference"] = requests[0]["srim_reference"]
    else:
        requests[-1]["energy_mev_per_nucleon"] = 79432.823472
        requests[-1]["energy_binary64_hex"] = float(79432.823472).hex()
    with pytest.raises(ValueError):
        worker.validate_requests(requests)


@pytest.mark.parametrize("field,value", [
    ("raw_sha256", "A"*64), ("native_row_index", True), ("native_row_index", -1),
    ("case_name", "../x"), ("energy_total_mev_decimal", "20000e0"),
    ("electronic_mev_cm2_mg", "NaN"), ("nuclear_mev_cm2_mg", "-1"),
])
def test_native_reference_rejected(worker: ModuleType, field: str, value: Any) -> None:
    requests = worker.build_requests(selected())
    requests[0]["srim_reference"][field] = value
    with pytest.raises(ValueError):
        worker.validate_requests(requests)


def test_stopping_no_mass_multiplier_and_strict_threshold(worker: ModuleType) -> None:
    backend = AnchorBackend(worker.native)
    samples = worker.generate_samples(backend, worker.build_requests(selected()), elemental_i_ev={13: 166., 14: 173.})
    assert [(p.z, p.mass, p.q) for p in backend.projectiles] == [(2, 4., 2), (10, 20.2, 10), (26, 55.8, 26), (92, 238., 92)]
    assert samples["backend_configuration"]["z_effective"] == 1
    assert samples["nonradiative_configuration"]["corrections"] == 8
    for row in samples["rows"]:
        e = row["energy_mev_per_nucleon"]
        assert row["stopping_electronic_mev_cm2_g"] == pytest.approx(100.+e)
        assert row["stopping_radiative_mev_cm2_g"] == (2. if e > 50000 else 0.)
        assert set(row) == worker.REQUEST_FIELDS | worker.previous.STOPPING_FIELDS


@pytest.mark.parametrize("mode", ("nan", "inf", "negative", "boolean", "config_mutation", "material_mutation", "negative_electronic", "radiative"))
def test_invalid_backend(worker: ModuleType, mode: str) -> None:
    with pytest.raises(ValueError):
        worker.generate_samples(AnchorBackend(worker.native, mode), worker.build_requests(selected()), elemental_i_ev={13: 166., 14: 173.})


@pytest.mark.parametrize("values", [None, {13: 166.}, {13: True, 14: 173.}, {13: 166., 14: 0.}])
def test_material_metadata(worker: ModuleType, values: Any) -> None:
    with pytest.raises(ValueError):
        worker.generate_samples(AnchorBackend(worker.native), worker.build_requests(selected()), elemental_i_ev=values)


def synthetic_report(worker: ModuleType) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, state = {"requests": worker.build_requests(selected())}, {"snapshot": "fixture"}
    samples = worker.generate_samples(AnchorBackend(worker.native), plan["requests"], elemental_i_ev={13: 166., 14: 173.})
    report = {"schema_version": "radar.high_energy_anchors.native.v1", "status": "INCOMPLETE", "request_count": 48,
              "plan": deepcopy(plan), "snapshot_before": deepcopy(state), "snapshot_after": deepcopy(state),
              "python": "fixture", "backend_info_literal": "fixture", **worker.FLAGS, **worker.IDENTITY, **samples}
    return report, plan, state


def test_report_validated_without_backend(worker: ModuleType) -> None:
    report, plan, state = synthetic_report(worker)
    worker.validate_report(report, plan, state)


@pytest.mark.parametrize("changed", (None, "plan", "snapshot"))
def test_generate_authenticates_before_and_after(worker: ModuleType, monkeypatch: pytest.MonkeyPatch, changed: str | None) -> None:
    _, plan, state = synthetic_report(worker)
    calls = {"plan": 0, "snapshot": 0}
    def state_value(kind: str, value: dict[str, Any]) -> dict[str, Any]:
        calls[kind] += 1
        result = deepcopy(value)
        if changed == kind and calls[kind] == 2:
            result["changed"] = True
        return result
    monkeypatch.setattr(worker, "reconstruct_plan", lambda: state_value("plan", plan))
    monkeypatch.setattr(worker, "snapshot", lambda: state_value("snapshot", state))
    monkeypatch.setattr(worker.native, "load_backend", lambda: AnchorBackend(worker.native))
    monkeypatch.setattr(worker, "_elemental_i_values", lambda: {13: 166., 14: 173.})
    if changed is None:
        report = worker.generate_controls()
        assert report["request_count"] == 48 and report["normative_export_allowed"] is False
        assert calls == {"plan": 2, "snapshot": 2}
    else:
        with pytest.raises(ValueError):
            worker.generate_controls()


@pytest.mark.parametrize("mutation", ("policy", "count", "snapshot", "plan", "config", "materials", "runtime", "missing", "identity", "fields", "closure", "nuclear", "threshold", "nan"))
def test_report_rejects_mutations(worker: ModuleType, mutation: str) -> None:
    report, plan, state = synthetic_report(worker)
    row = report["rows"][0]
    if mutation == "policy":
        report["normative_export_allowed"] = True
    elif mutation == "count":
        report["request_count"] = 48.0
    elif mutation == "snapshot":
        report["snapshot_after"]["snapshot"] = "changed"
    elif mutation == "plan":
        report["plan"]["requests"].pop()
    elif mutation == "config":
        report["backend_configuration"]["z_effective"] = 0
    elif mutation == "materials":
        report["materials"][0]["density_g_cm3"] = 2.7
    elif mutation == "runtime":
        report["python"] = ""
    elif mutation == "missing":
        report["rows"].pop()
    elif mutation == "identity":
        row["mass_parameter_u_decimal"] = "4.1"
    elif mutation == "fields":
        row["extra"] = 1
    elif mutation == "closure":
        row["stopping_electronic_mev_cm2_g"] += 1
    elif mutation == "nuclear":
        row["stopping_nuclear_mev_cm2_g"] *= 2
    elif mutation == "threshold":
        row["stopping_radiative_mev_cm2_g"] = 1
        row["stopping_total_mev_cm2_g"] += 1
    else:
        row["stopping_total_mev_cm2_g"] = float("nan")
    with pytest.raises(ValueError):
        worker.validate_report(report, plan, state)


def test_cli_existing_output_never_generates(worker: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "exists.json"
    output.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(worker.sys, "argv", ["tool", "--output", str(output)])
    monkeypatch.setattr(worker, "generate_controls", lambda: pytest.fail("Existing output must fail before native load"))
    with pytest.raises(FileExistsError):
        worker.main()
    assert output.read_text(encoding="utf-8") == "keep"


def test_cli_exclusive_output_race(worker: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "race.json"
    monkeypatch.setattr(worker.sys, "argv", ["tool", "--output", str(output)])
    def generate() -> dict[str, str]:
        output.write_text("other writer", encoding="utf-8")
        return {"result": "new"}
    monkeypatch.setattr(worker, "generate_controls", generate)
    with pytest.raises(FileExistsError):
        worker.main()
    assert output.read_text(encoding="utf-8") == "other writer"


def test_plan_mode_never_loads_backend(worker: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    monkeypatch.setattr(worker.sys, "argv", ["tool", "--mode", "plan", "--output", str(output)])
    monkeypatch.setattr(worker, "reconstruct_plan", lambda: {"requests": worker.build_requests(selected())})
    monkeypatch.setattr(worker, "generate_controls", lambda: pytest.fail("Plan must never load backend"))
    assert worker.main() == 0 and output.is_file()
