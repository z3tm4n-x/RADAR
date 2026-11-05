"""Actual synthetic upper transport products; comparator never runs native code."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import compare_srim_upper_support as subject
from radar.core.products import SpectrumProduct
from radar.core.spectra import ParticleSpecies, TransformationStep
from radar.core.types import (
    AssessmentMode,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.pipelines.srim_hze import calculate_srim_hze_products
from test_srim_hze_transport import source, table

MASSES = {2: ("He", "4"), 10: ("Ne", "20.2"), 26: ("Fe", "55.8"), 92: ("U", "238")}


@pytest.fixture(scope="module")
def native_tables() -> tuple[dict[int, Any], dict[int, Any]]:
    pairs = []
    for length in (3, 4):
        pairs.append(
            {
                z: table(
                    z=z,
                    mass=mass,
                    energies=tuple(str(Decimal(mass) * e) for e in (1, 2, 4, 8)[:length]),
                    stopping=(mass,) * length,
                )
                for z, (_, mass) in MASSES.items()
            }
        )
    return pairs[0], pairs[1]


def report(tables: dict[int, Any], sha: str) -> dict[str, Any]:
    rows = []
    for src, roster in ((RadiationSource.GCR, (2, 10, 26, 92)), (RadiationSource.SEP, (2, 10, 26))):
        for peak in (False, True):
            mode = AssessmentMode.PEAK if peak else AssessmentMode.CUMULATIVE
            for z in roster:
                symbol, mass = MASSES[z]
                changes: dict[str, Any] = {
                    "source": src,
                    "x": (1.0, 2.0, 4.0, 8.0),
                    "y": (2.0,) * 4,
                    "species": ParticleSpecies(symbol, z, round(float(mass))),
                    "transformation_chain": (
                        TransformationStep(
                            "synthetic_source", "1", (("source_tabulated_mass_parameter", mass),)
                        ),
                    ),
                }
                kind = (
                    RadiationProductKind.PEAK_FLUX if peak else RadiationProductKind.MISSION_FLUENCE
                )
                if src is RadiationSource.GCR and peak:
                    changes.update(
                        quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
                        temporal_basis=TemporalBasis.SOURCE_MAXIMUM,
                    )
                    kind = RadiationProductKind.MAXIMUM_FLUX
                products = calculate_srim_hze_products(
                    products=(SpectrumProduct(kind, source(peak=peak, **changes)),),
                    tables=tables,
                    thicknesses_g_cm2=subject.SHIELDS,
                    assessment_mode=mode,
                    output_energy_edges=(0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0),
                    let_edges=(0.0, 0.005, 0.01, 0.1, 1.0),
                )
                for c in products.cases:
                    assert c.transport is not None and c.let_result is not None
                    assert c.energy_product is not None and c.let_product is not None
                    rows.append(
                        {
                            "source": src,
                            "mode": mode,
                            "z": z,
                            "thickness_g_cm2": c.thickness_g_cm2,
                            "energy_product": asdict(c.energy_product),
                            "let_product": asdict(c.let_product),
                            "primary_ledger": asdict(c.transport.ledger),
                            "let_ledger": asdict(c.let_result.ledger),
                        }
                    )
    return dict(
        json.loads(
            json.dumps(
                {
                    "schema_version": "radar.srim_hze_external_products.v1",
                    "status": "INCOMPLETE",
                    "normative_export_allowed": False,
                    "bundle_sha256": sha,
                    "source_grid_refinement": 1,
                    "scope": "synthetic_external_source",
                    "mission": {"years": 5},
                    "implementation_and_input_sha256_before": {"fake.py": "a" * 64},
                    "implementation_and_input_sha256_after": {"fake.py": "a" * 64},
                    "source_product_count": 14,
                    "requested_symbols": ["He", "Ne", "Fe", "U"],
                    "case_count": len(rows),
                    "cases": rows,
                }
            )
        )
    )


@pytest.fixture(scope="module")
def reports(native_tables: tuple[Any, Any]) -> tuple[Any, Any]:
    return report(native_tables[0], hashlib.sha256(b"parent").hexdigest()), report(
        native_tables[1], hashlib.sha256(b"upper").hexdigest()
    )


@pytest.fixture
def setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_tables: tuple[Any, Any],
    reports: tuple[Any, Any],
) -> dict[str, Any]:
    old, new = native_tables
    parent, upper = tmp_path / "parent.zip", tmp_path / "upper.zip"
    parent.write_bytes(b"parent")
    upper.write_bytes(b"upper")
    state = {"old": dict(old), "new": dict(new)}

    def load(path: Path, *, expected_sha256: str) -> Any:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
        return SimpleNamespace(tables_by_z=state["old" if path == parent else "new"])

    monkeypatch.setattr(subject, "load_srim_hze_dataset", load)
    return {
        "a": deepcopy(reports[0]),
        "b": deepcopy(reports[1]),
        "state": state,
        "kwargs": {
            "parent_bundle": parent,
            "parent_sha": hashlib.sha256(b"parent").hexdigest(),
            "upper_bundle": upper,
            "upper_sha": hashlib.sha256(b"upper").hexdigest(),
            "parent_products": tmp_path / "parent.json",
            "upper_products": tmp_path / "upper.json",
        },
    }


def compare(setup: dict[str, Any], **changes: Any) -> dict[str, Any]:
    options = dict(setup["kwargs"])
    for prefix, key in (("parent", "a"), ("upper", "b")):
        raw = json.dumps(setup[key]).encode()
        options[f"{prefix}_products"].write_bytes(raw)
        options[f"{prefix}_products_sha"] = hashlib.sha256(raw).hexdigest()
    options.update(changes)
    return subject.compare(**options)


def test_real_all_pilot_products_conserve_new_counts(setup: dict[str, Any]) -> None:
    result = compare(setup)
    assert result["case_count"] == 84 and len(result["tables"]) == 4
    assert result["pilot_inventory_verified"] and not result["normative_export_allowed"]
    for row in result["cases"]:
        assert row["recovered_al_source_integral"] == pytest.approx(
            row["new_supported_integral"] + row["new_below_anchor_integral"]
        )
        assert row["new_mapped_let_integral"] >= 0
        if row["thickness_g_cm2"] == 0:
            assert row["new_supported_integral"] == 0
            assert row["new_mapped_let_integral"] == 8
        if row["thickness_g_cm2"] == 10:
            assert row["new_supported_integral"] == 0
            assert row["new_below_anchor_integral"] == 8
            assert row["mapped_let_relative_gain"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_product_count", 13),
        ("requested_symbols", ["He"]),
        ("bundle_sha256", "b" * 64),
        ("mission", {"years": 6}),
        ("source_grid_refinement", 2),
        ("scope", "orbit"),
        ("status", "COMPLETE"),
        ("normative_export_allowed", True),
        ("implementation_and_input_sha256_before", {}),
        ("implementation_and_input_sha256_after", {}),
    ],
)
def test_report_contract_change_rejected(setup: dict[str, Any], field: str, value: Any) -> None:
    setup["b"][field] = value
    with pytest.raises(ValueError):
        compare(setup)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "shield", "z", "mode"])
def test_inventory_strict(setup: dict[str, Any], change: str) -> None:
    b = setup["b"]
    if change == "missing":
        b["cases"].pop()
    elif change == "duplicate":
        b["cases"][1] = deepcopy(b["cases"][0])
    elif change == "extra":
        b["cases"].append({**b["cases"][0], "z": 6})
    else:
        b["cases"][0][{"shield": "thickness_g_cm2", "z": "z", "mode": "mode"}[change]] = {
            "shield": 2.0,
            "z": 6,
            "mode": "mean",
        }[change]
    b["case_count"] = len(b["cases"])
    with pytest.raises(ValueError):
        compare(setup)


@pytest.mark.parametrize(
    "field",
    [
        "x",
        "bin_edges",
        "x_unit",
        "y_unit",
        "assessment_mode",
        "quantity",
        "temporal_basis",
        "mission_duration_s",
        "angular_basis",
        "source",
        "species",
    ],
)
def test_spectral_semantics_change_rejected(setup: dict[str, Any], field: str) -> None:
    setup["b"]["cases"][1]["energy_product"]["spectrum"][field] = "changed"
    with pytest.raises(ValueError):
        compare(setup)


def test_joint_spectrum_mode_differs_from_case_key(setup: dict[str, Any]) -> None:
    for key in ("a", "b"):
        setup[key]["cases"][0]["energy_product"]["spectrum"]["assessment_mode"] = "peak"
    with pytest.raises(ValueError, match="case key"):
        compare(setup)


@pytest.mark.parametrize(
    "field",
    [
        "source_integral",
        "supported_exit_integral",
        "unresolved_low_residual_integral",
        "outside_al_domain_integral",
        "represented_integral",
        "below_output_window_integral",
        "above_output_window_integral",
    ],
)
def test_primary_ledger_change_rejected(setup: dict[str, Any], field: str) -> None:
    setup["b"]["cases"][1]["primary_ledger"][field] += 1.0
    with pytest.raises(ValueError):
        compare(setup)


@pytest.mark.parametrize("name", ["energy_product", "let_product"])
def test_lost_bins_rejected(setup: dict[str, Any], name: str) -> None:
    setup["b"]["cases"][1][name]["spectrum"]["y"][0] = -1.0e6
    with pytest.raises(ValueError):
        compare(setup)


def test_zero_energy_identity_and_let_excess(setup: dict[str, Any]) -> None:
    setup["b"]["cases"][0]["energy_product"]["spectrum"]["y"][0] += 1
    with pytest.raises(ValueError, match="Zero-Al"):
        compare(setup)


def test_bad_hash(setup: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        compare(setup, upper_products_sha="0" * 64)


@pytest.mark.parametrize("change", ["missing", "wrong_key", "mass", "prefix"])
def test_table_invariants(setup: dict[str, Any], change: str) -> None:
    state = setup["state"]
    if change == "missing":
        state["new"].pop(92)
    elif change == "wrong_key":
        state["new"][2] = state["new"][10]
    elif change == "mass":
        state["new"][2] = table(
            z=2, mass="5", energies=("5", "10", "20", "40"), stopping=("5",) * 4
        )
    else:
        state["new"][2] = table(z=2, mass="4", energies=("4", "8", "16", "32"), stopping=("5",) * 4)
    with pytest.raises(ValueError):
        compare(setup)
