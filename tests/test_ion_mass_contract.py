"""Mass inventory diagnostics preserve OST data and never infer isotope approval."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from radar.validation.ion_mass_contract import CandidateIsotope, audit_ost_ion_mass_contract


PILOT = tuple(CandidateIsotope(z, a, symbol) for z, a, symbol in (
    (2, 4, "He"), (6, 12, "C"), (7, 14, "N"), (8, 16, "O"), (10, 20, "Ne"),
    (12, 24, "Mg"), (14, 28, "Si"), (16, 32, "S"), (18, 40, "Ar"), (20, 40, "Ca"),
    (24, 52, "Cr"), (25, 55, "Mn"), (26, 56, "Fe"), (27, 59, "Co"), (28, 58, "Ni"),
))


def test_pilot_counts_distinguish_positive_sources_from_declared_zero() -> None:
    audit = audit_ost_ion_mass_contract(PILOT)
    gcr, sep = audit.summaries
    assert (gcr.table_hze_row_count, gcr.positive_hze_count, gcr.declared_zero_count) == (91, 84, 7)
    assert (sep.table_hze_row_count, sep.positive_hze_count, sep.declared_zero_count) == (27, 24, 3)
    assert (gcr.missing_positive_count, sep.missing_positive_count) == (69, 9)
    for summary in audit.summaries:
        assert summary.candidate_present_positive_count == 15
        assert summary.exact_numeric_mass_not_isotope_proof_count == 5
        assert summary.different_mass_parameter_policy_unresolved_count == 10
        assert summary.rounded_metadata_hides_mass_difference_count == 9
    assert {r.symbol for r in audit.rows if r.source == "SEP" and r.classification == "missing"} == {
        "F", "Na", "Al", "P", "Cl", "K", "Sc", "Ti", "V",
    }
    assert not audit.normative_export_allowed
    assert not audit.isotope_selection_policy_established
    assert not audit.production_tables_replaced


def test_fractional_mass_rounded_metadata_and_effective_charge_are_separate() -> None:
    rows = {(r.source, r.symbol): r for r in audit_ost_ion_mass_contract(PILOT).rows}
    fe = rows["SEP", "Fe"]
    assert fe.source_tabulated_mass_parameter == 55.8
    assert fe.source_mass_to_charge == 4.12
    assert fe.current_rounded_metadata_mass_number == fe.candidate_isotope_mass_number == 56
    assert fe.rounded_metadata_hides_mass_difference
    assert fe.classification == "different_mass_parameter_policy_unresolved"
    ni = rows["GCR", "Ni"]
    assert (ni.source_tabulated_mass_parameter, ni.current_rounded_metadata_mass_number,
            ni.candidate_isotope_mass_number) == (58.7, 59, 58)
    assert not ni.rounded_metadata_hides_mass_difference
    assert ni.classification == "different_mass_parameter_policy_unresolved"
    assert rows["GCR", "He"].classification == "exact_numeric_mass_not_isotope_proof"
    assert rows["GCR", "Fe"].source_mass_to_charge is None


def test_printed_unusual_masses_and_zero_rows_are_preserved() -> None:
    rows = {(r.source, r.symbol): r for r in audit_ost_ion_mass_contract(PILOT).rows}
    for symbol, mass in (("Sm", 145.0), ("Pb", 200.4), ("Bi", 207.2)):
        assert rows["GCR", symbol].source_tabulated_mass_parameter == mass
    assert {r.symbol for r in rows.values() if r.source == "GCR" and r.classification == "declared_zero"} == {
        "Po", "At", "Rn", "Fr", "Ra", "Ac", "Pa",
    }
    for source, symbol in (("GCR", "Po"), ("SEP", "Li")):
        assert rows[source, symbol].nonzero_source_components == ()
        assert rows[source, symbol].source_weight_parameter == 0


def test_explicit_different_isotope_is_not_replaced_by_rounded_source_mass() -> None:
    for mass in (58, 59, 60):
        audit = audit_ost_ion_mass_contract((CandidateIsotope(28, mass, "Ni"),))
        ni = next(r for r in audit.rows if r.source == "GCR" and r.symbol == "Ni")
        assert ni.candidate_isotope_mass_number == mass
        assert ni.classification == "different_mass_parameter_policy_unresolved"
        assert not audit.normative_export_allowed


def test_zero_candidate_remains_zero_and_proton_is_outside_hze_scope() -> None:
    audit = audit_ost_ion_mass_contract((CandidateIsotope(1, 1, "H"), CandidateIsotope(84, 209, "Po")))
    assert audit.candidate_z_outside_hze_rosters == (1,)
    po = next(r for r in audit.rows if r.source == "GCR" and r.symbol == "Po")
    assert po.classification == "declared_zero"
    assert po.candidate_isotope_mass_number == 209


@pytest.mark.parametrize("z,a,symbol", [
    (True, 4, "He"), (2, True, "He"), (2.0, 4, "He"), (2, 4.0, "He"),
    (2, 4.1, "He"), (0, 4, "He"), (2, 1, "He"), (93, 238, "U"),
    (2, 4, "he"), (2, 4, "Fe"), (2, 4, None), (2, 4, ["He"]),
])
def test_malformed_candidate_identity_is_rejected(z: Any, a: Any, symbol: Any) -> None:
    with pytest.raises(ValueError):
        CandidateIsotope(z, a, symbol)


@pytest.mark.parametrize("inventory", [(), (PILOT[0], PILOT[0]),
                                        (PILOT[0], CandidateIsotope(2, 3, "He")), ({"z": 2},)])
def test_empty_duplicate_or_implicit_inventory_is_rejected(inventory: Any) -> None:
    with pytest.raises(ValueError):
        audit_ost_ion_mass_contract(inventory)


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/audit_ion_mass_contract.py"
    spec = importlib.util.spec_from_file_location("ion_mass_contract_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({"schema_version": "radar.ion_stopping_candidate.raw.v1",
                                "ions": [asdict(ion) for ion in PILOT]}), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runner_pins_input_and_source_definitions(runner: ModuleType, candidate: Path) -> None:
    report = runner.run_audit(candidate, expected_candidate_sha256=_sha(candidate))
    assert report["candidate_sha256_before"] == report["candidate_sha256_after"] == _sha(candidate)
    assert report["source_files_sha256_before"] == report["source_files_sha256_after"]
    assert len(report["source_files_sha256_before"]) == 7
    assert "stopping_rows_not_validated" in report["candidate_payload_scope"]
    assert report["source_document"] is None
    assert not report["audit"]["normative_export_allowed"]
    assert not report["normative_export_allowed"]


@pytest.mark.parametrize("pin", ["0" * 64, "ABC", None])
def test_runner_rejects_bad_or_mismatched_pin(runner: ModuleType, candidate: Path, pin: Any) -> None:
    with pytest.raises(ValueError):
        runner.run_audit(candidate, expected_candidate_sha256=pin)


@pytest.mark.parametrize("payload", [
    '{"schema_version":"radar.ion_stopping_candidate.raw.v1","ions":[],"ions":[]}',
    '{"value":NaN}', '{"value":1e999}', '{}', '[]',
    '{"schema_version":"radar.ion_stopping_candidate.raw.v1","ions":[]}',
    '{"schema_version":"radar.ion_stopping_candidate.raw.v1","ions":[{"z":2}]}',
])
def test_runner_rejects_malformed_json_inventory(
    runner: ModuleType, candidate: Path, payload: str,
) -> None:
    candidate.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        runner.run_audit(candidate, expected_candidate_sha256=_sha(candidate))


def test_source_document_pin_and_exclusive_output(runner: ModuleType, candidate: Path) -> None:
    document = candidate.parent / "source.pdf"
    document.write_bytes(b"synthetic pin fixture, not a real PDF")
    kwargs = {"expected_candidate_sha256": _sha(candidate), "source_document_path": document}
    with pytest.raises(ValueError, match="together"):
        runner.run_audit(candidate, **kwargs)
    with pytest.raises(ValueError, match="does not match"):
        runner.run_audit(candidate, **kwargs, expected_source_document_sha256="0" * 64)
    report = runner.run_audit(candidate, **kwargs, expected_source_document_sha256=_sha(document))
    assert report["source_document"]["sha256"] == _sha(document)
    assert not report["source_document"]["contents_independently_read_by_runner"]
    output = candidate.parent / "report.json"
    runner.write_report_exclusively(report, output)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        runner.write_report_exclusively(report, output)
    assert output.read_bytes() == before


def test_runner_detects_input_and_source_mutation(
    runner: ModuleType, candidate: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runner.audit_ost_ion_mass_contract

    def mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        candidate.write_bytes(candidate.read_bytes() + b" ")
        return result

    monkeypatch.setattr(runner, "audit_ost_ion_mass_contract", mutate)
    with pytest.raises(ValueError, match="changed"):
        runner.run_audit(candidate, expected_candidate_sha256=_sha(candidate))
    monkeypatch.setattr(runner, "audit_ost_ion_mass_contract", original)
    hashes = iter(({"source": "a"}, {"source": "b"}))
    monkeypatch.setattr(runner, "_source_hashes", lambda: next(hashes))
    with pytest.raises(ValueError, match="changed"):
        runner.run_audit(candidate, expected_candidate_sha256=_sha(candidate))


def test_cli_writes_once_and_refuses_overwrite(runner: ModuleType, candidate: Path) -> None:
    output = candidate.parent / "report.json"
    args = [str(candidate), "--candidate-sha256", _sha(candidate), "--output", str(output)]
    assert runner.main(args) == 0
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        runner.main(args)
    assert output.read_bytes() == before
