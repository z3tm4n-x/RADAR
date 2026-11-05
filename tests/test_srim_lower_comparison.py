"""Lower-support comparator tests using real small SRIM transport/LET products."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from radar.core.products import SpectrumProduct
from radar.core.types import AssessmentMode, RadiationProductKind
from radar.pipelines.srim_hze import calculate_srim_hze_products
from radar.shielding.srim_hze import SrimIonStoppingTable
from test_srim_hze_transport import source, table


@pytest.fixture(scope="module")
def comparator() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/compare_srim_lower_support.py"
    spec = importlib.util.spec_from_file_location("srim_lower_comparison_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(stopping: SrimIonStoppingTable, sha: str) -> dict[str, Any]:
    rows = []
    for peak in (False, True):
        kind = RadiationProductKind.PEAK_FLUX if peak else RadiationProductKind.MISSION_FLUENCE
        mode = AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE
        products = calculate_srim_hze_products(
            products=(SpectrumProduct(kind, source(peak=peak)),), tables={6: stopping},
            thicknesses_g_cm2=(0., 1.), assessment_mode=mode,
            let_edges=(0., .005, .01, .02), output_energy_edges=(.1, .5, 1., 2., 3., 4.),
        )
        for case in products.cases:
            assert case.transport is not None and case.let_result is not None
            assert case.energy_product is not None and case.let_product is not None
            rows.append({"source": case.source, "mode": mode, "z": case.z,
                         "thickness_g_cm2": case.thickness_g_cm2,
                         "primary_ledger": asdict(case.transport.ledger),
                         "let_ledger": asdict(case.let_result.ledger),
                         "energy_product": asdict(case.energy_product),
                         "let_product": asdict(case.let_product)})
    result: dict[str, Any] = json.loads(json.dumps({
        "schema_version": "radar.srim_hze_external_products.v1", "status": "INCOMPLETE",
        "normative_export_allowed": False, "mission": {"lifetime_years": 5},
        "source_grid_refinement": 1, "scope": "synthetic_external_source",
        "bundle_sha256": sha, "implementation_and_input_sha256_before": {"test.py": "a" * 64},
        "implementation_and_input_sha256_after": {"test.py": "a" * 64},
        "cases": rows, "case_count": len(rows),
    }))
    return result


def _write(path: Path, data: dict[str, Any]) -> str:
    raw = json.dumps(data).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def setup(comparator: ModuleType, monkeypatch: pytest.MonkeyPatch,
          tmp_path: Path) -> dict[str, Any]:
    parent = table()
    lower = table(energies=("6", "12", "24", "48"), stopping=("12",) * 4)
    parent_bundle, lower_bundle = tmp_path / "parent.zip", tmp_path / "lower.zip"
    parent_bundle.write_bytes(b"synthetic parent bundle; loader mocked")
    lower_bundle.write_bytes(b"synthetic lower bundle; loader mocked")
    parent_sha = hashlib.sha256(parent_bundle.read_bytes()).hexdigest()
    lower_sha = hashlib.sha256(lower_bundle.read_bytes()).hexdigest()
    state: dict[str, Any] = {"parent": parent, "lower": lower, "loaded": []}

    def load(path: Path, *, expected_sha256: str) -> SimpleNamespace:
        expected = parent_sha if path == parent_bundle else lower_sha
        assert expected_sha256 == expected
        state["loaded"].append(path)
        return SimpleNamespace(tables_by_z={6: state["parent" if path == parent_bundle else "lower"]})

    monkeypatch.setattr(comparator, "load_srim_hze_dataset", load)
    a, b = _report(parent, parent_sha), _report(lower, lower_sha)
    return {"state": state, "a": a, "b": b, "kwargs": {
        "parent_bundle": parent_bundle, "parent_sha": parent_sha,
        "lower_bundle": lower_bundle, "lower_sha": lower_sha,
        "parent_products": tmp_path / "parent.json", "lower_products": tmp_path / "lower.json",
    }}


def _compare(comparator: ModuleType, setup: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    options = dict(setup["kwargs"])
    options["parent_products_sha"] = _write(options["parent_products"], setup["a"])
    options["lower_products_sha"] = _write(options["lower_products"], setup["b"])
    options.update(kwargs)
    return dict(comparator.compare(**options))


def test_real_lower_extension_has_conservative_gain_without_stopped_claim(
    comparator: ModuleType, setup: dict[str, Any],
) -> None:
    result = _compare(comparator, setup)
    assert result["case_count"] == 4
    assert result["status"] == "INCOMPLETE"
    assert result["normative_export_allowed"] is False
    assert result["stopped_particle_classification_established"] is False
    assert float(result["tables"][0]["added_path_g_cm2"]) == .5
    assert float(result["tables"][0]["range_translation_max_residual_g_cm2"]) == 0
    assert len(result["input_sha256"]) == 5
    for row in result["cases"]:
        expected = 0 if row["thickness_g_cm2"] == 0 else 1
        assert row["new_supported_integral"] == expected
        assert row["new_mapped_let_integral"] == pytest.approx(expected)
        assert row["new_supported_fraction_of_source"] == expected / 6
        assert row["remaining_below_anchor_integral"] == expected


def test_pilot_cannot_be_presented_as_full(comparator: ModuleType, setup: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Full inventory"):
        _compare(comparator, setup, require_full_inventory=True)


def full_inventory() -> dict[str, Any]:
    cases = [{"source": name, "mode": mode, "z": z, "thickness_g_cm2": shield}
             for name, roster in (("gcr", range(2, 86)), ("sep", range(2, 26)))
             for mode in ("peak", "cumulative") for z in roster
             for shield in (0., .01, .1, 1., 3., 10.)]
    return {"cases": cases, "case_count": len(cases), "source_product_count": 216,
            "requested_symbols": None}


def test_complete_inventory(comparator: ModuleType) -> None:
    a = full_inventory()
    comparator._full_inventory(a, deepcopy(a), set(range(2, 86)), set(range(2, 86)))


@pytest.mark.parametrize("change", ["case", "shield", "mode", "ion", "pair", "count", "symbols"])
def test_incomplete_full_inventory_rejected(comparator: ModuleType, change: str) -> None:
    a, b = full_inventory(), full_inventory()
    ions = set(range(2, 86))
    if change == "case":
        b["cases"].pop()
        b["case_count"] -= 1
    elif change in ("shield", "mode", "ion"):
        field, value = {"shield": ("thickness_g_cm2", 11.), "mode": ("mode", "mean"),
                        "ion": ("z", 93)}[change]
        b["cases"][0][field] = value
    elif change == "pair":
        ions.remove(85)
    elif change == "count":
        b["source_product_count"] = 215
    else:
        b["requested_symbols"] = ["He"]
    with pytest.raises(ValueError, match="Full inventory"):
        comparator._full_inventory(a, b, set(range(2, 86)), ions)


@pytest.mark.parametrize("mode,unit", [("peak", "MeV cm^-2 s^-1"), ("cumulative", "MeV cm^-2")])
@pytest.mark.parametrize("count", [0., 1., 12345.67])
def test_low_residual_uses_total_energy_and_no_extra_factors(
    comparator: ModuleType, mode: str, unit: str, count: float,
) -> None:
    result = comparator._low_residual_statement(count, Decimal(".00000999999"), mode)
    from decimal import localcontext
    with localcontext() as context:
        context.prec = 1200
        expected = Decimal.from_float(count) * Decimal(".00000999999")
    assert Decimal(result["residual_kinetic_energy_upper_bound_decimal"]) == expected
    assert result["residual_kinetic_energy_upper_bound_unit"] == unit
    assert result["classification"] == "stopped_or_exit_below_observed_energy"
    assert not result["stopped_particle_classification_established"]
    assert not result["is_let_dose_or_see_error_bound"]


@pytest.mark.parametrize("count,minimum,mode", [
    (-1., Decimal(1), "peak"), (float("nan"), Decimal(1), "peak"),
    (float("inf"), Decimal(1), "peak"), (True, Decimal(1), "peak"),
    (1., Decimal(0), "peak"), (1., Decimal("NaN"), "peak"), (1., Decimal(1), "mean"),
])
def test_invalid_low_residual_inputs(comparator: ModuleType, count: Any, minimum: Any, mode: str) -> None:
    with pytest.raises(ValueError):
        comparator._low_residual_statement(count, minimum, mode)


@pytest.mark.parametrize("which", ["parent", "lower"])
def test_report_hash_mismatch_fails(comparator: ModuleType, setup: dict[str, Any], which: str) -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        _compare(comparator, setup, **{f"{which}_products_sha": "0" * 64})


@pytest.mark.parametrize("field,value", [
    ("schema_version", "different"), ("status", "COMPLETE"), ("normative_export_allowed", True),
])
def test_report_type_and_release_promotion_rejected(
    comparator: ModuleType, setup: dict[str, Any], field: str, value: Any,
) -> None:
    setup["b"][field] = value
    with pytest.raises(ValueError, match="incomplete external-source"):
        _compare(comparator, setup)


@pytest.mark.parametrize("field,value", [
    ("bundle_sha256", "0" * 64), ("mission", {}), ("source_grid_refinement", 2), ("scope", "orbit"),
])
def test_report_comparability_fields_rejected(
    comparator: ModuleType, setup: dict[str, Any], field: str, value: Any,
) -> None:
    setup["b"][field] = value
    with pytest.raises(ValueError):
        _compare(comparator, setup)


@pytest.mark.parametrize("report,field", [
    ("a", "implementation_and_input_sha256_before"),
    ("a", "implementation_and_input_sha256_after"),
    ("b", "implementation_and_input_sha256_before"),
    ("b", "implementation_and_input_sha256_after"),
])
def test_changed_implementation_pins_rejected(
    comparator: ModuleType, setup: dict[str, Any], report: str, field: str,
) -> None:
    setup[report][field] = {"changed": "b" * 64}
    with pytest.raises(ValueError, match="inputs"):
        _compare(comparator, setup)


@pytest.mark.parametrize("kind", ["duplicate", "case_count", "new_key", "empty"])
def test_case_inventory_errors_rejected(comparator: ModuleType, setup: dict[str, Any], kind: str) -> None:
    if kind == "duplicate":
        setup["b"]["cases"].append(setup["b"]["cases"][0])
    elif kind == "case_count":
        setup["b"]["case_count"] = 99
    elif kind == "new_key":
        setup["b"]["cases"][0]["source"] = "different"
    else:
        setup["b"]["cases"], setup["b"]["case_count"] = [], 0
    with pytest.raises(ValueError):
        _compare(comparator, setup)


@pytest.mark.parametrize("kind", ["same_lower", "stopping", "upper", "density"])
def test_native_extension_must_keep_original_identity_and_stopping_suffix(
    comparator: ModuleType, setup: dict[str, Any], kind: str,
) -> None:
    kwargs: dict[str, Any] = {"energies": ("6", "12", "24", "48"), "stopping": ("12",) * 4}
    if kind == "same_lower":
        setup["state"]["lower"] = table()
    else:
        if kind == "stopping":
            kwargs["stopping"] = ("12", "12", "13", "12")
        elif kind == "upper":
            kwargs["energies"] = ("6", "12", "24", "60")
        else:
            kwargs["density"] = "2.9"
        setup["state"]["lower"] = table(**kwargs)
    with pytest.raises(ValueError):
        _compare(comparator, setup)


@pytest.mark.parametrize("field,value", [
    ("source_integral", 7.), ("outside_al_domain_integral", 1.),
    ("supported_exit_integral", 3.), ("unresolved_low_residual_integral", 2.),
])
def test_changed_source_or_nonconservative_ledgers_rejected(
    comparator: ModuleType, setup: dict[str, Any], field: str, value: float,
) -> None:
    setup["b"]["cases"][1]["primary_ledger"][field] = value
    with pytest.raises(ValueError):
        _compare(comparator, setup)


@pytest.mark.parametrize("product", ["energy_product", "let_product"])
@pytest.mark.parametrize("field,value", [
    ("x_unit", "wrong"), ("y_unit", "wrong"), ("assessment_mode", "peak"),
    ("quantity", "different"), ("temporal_basis", "different"), ("mission_duration_s", 1.),
    ("x", []), ("bin_edges", []),
])
def test_changed_grid_or_unit_mode_time_contract_rejected(
    comparator: ModuleType, setup: dict[str, Any], product: str, field: str, value: Any,
) -> None:
    setup["b"]["cases"][1][product]["spectrum"][field] = value
    with pytest.raises(ValueError, match="grid or source semantics"):
        _compare(comparator, setup)


@pytest.mark.parametrize("product", ["energy_product", "let_product"])
def test_zero_shield_identity_must_remain_exact(
    comparator: ModuleType, setup: dict[str, Any], product: str,
) -> None:
    setup["b"]["cases"][0][product]["spectrum"]["y"][0] += 1
    with pytest.raises(ValueError, match="unaffected"):
        _compare(comparator, setup)


def test_energy_bin_above_old_anchor_must_not_change(comparator: ModuleType, setup: dict[str, Any]) -> None:
    setup["b"]["cases"][1]["energy_product"]["spectrum"]["y"][3] += 1
    with pytest.raises(ValueError, match="unaffected"):
        _compare(comparator, setup)


@pytest.mark.parametrize("product,index", [("energy_product", 3), ("let_product", 1)])
def test_preexisting_bin_contribution_cannot_be_lost(
    comparator: ModuleType, setup: dict[str, Any], product: str, index: int,
) -> None:
    setup["b"]["cases"][1][product]["spectrum"]["y"][index] = -1.
    with pytest.raises(ValueError, match="lost"):
        _compare(comparator, setup)


@pytest.mark.parametrize("value", [3., 7.])
def test_let_gain_must_lie_within_added_supported_count(
    comparator: ModuleType, setup: dict[str, Any], value: float,
) -> None:
    setup["b"]["cases"][1]["let_ledger"]["mapped_integral"] = value
    with pytest.raises(ValueError, match="LET gain"):
        _compare(comparator, setup)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_report_values_fail_closed(
    comparator: ModuleType, setup: dict[str, Any], value: float,
) -> None:
    setup["b"]["cases"][1]["let_product"]["spectrum"]["y"][0] = value
    with pytest.raises(ValueError):
        _compare(comparator, setup)


@pytest.mark.parametrize("raw", [
    b'{"status": "INCOMPLETE", "status": "COMPLETE"}',
    b'{"value": 1e999}', b'{"value": -1e999}', b'[]', b'null', b'1',
])
def test_json_duplicates_overflow_and_wrong_root_rejected(
    comparator: ModuleType, tmp_path: Path, raw: bytes,
) -> None:
    path = tmp_path / "malformed.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        comparator._read(path, hashlib.sha256(raw).hexdigest())
