"""Raw workbook evidence must survive parsing, comparison and exclusive export."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, localcontext
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import pytest

from scripts import audit_legacy_workbook_srim as subject


_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_BOOK = f'<workbook xmlns="{_NS}" xmlns:r="{_REL}"><sheets>'
_RELS = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'


def _xlsx(
    cells: str = '<c r="A1"><v>1</v></c>',
    *,
    book: str | None = None,
    rels: str | None = None,
    extra: list[tuple[str, str]] | None = None,
) -> bytes:
    entries = [
        ("xl/workbook.xml", book or _BOOK + '<sheet name="i_Si_e" sheetId="7" r:id="rId9"/></sheets></workbook>'),
        ("xl/_rels/workbook.xml.rels", rels or _RELS + '<Relationship Id="rId9" Target="worksheets/source.xml"/></Relationships>'),
        ("xl/worksheets/source.xml", f'<worksheet xmlns="{_NS}"><sheetData><row>{cells}</row></sheetData></worksheet>'),
        *(extra or []),
    ]
    raw = io.BytesIO()
    with ZipFile(raw, "w") as archive:
        for name, text in entries:
            archive.writestr(name, text)
    return raw.getvalue()


@pytest.mark.parametrize("index,expected", [(1, "A"), (2, "B"), (26, "Z"), (27, "AA"),
                                            (52, "AZ"), (53, "BA"), (93, "CO"),
                                            (702, "ZZ"), (703, "AAA")])
def test_excel_column_mapping(index: int, expected: str) -> None:
    assert subject.column(index) == expected


@pytest.mark.parametrize("value", [0, -1, 1.0, True, False, "2", None])
def test_column_rejects_nonpositive_noninteger_and_bool(value: Any) -> None:
    with pytest.raises(ValueError):
        subject.column(value)


def test_read_preserves_numeric_lexemes_without_float_normalization() -> None:
    tokens = ["1", "1.0", "1E+0", "-0", "+.125", "123456789012345678901234567890.1", "1.2300e-18", "1."]
    cells = "".join(f'<c r="{subject.column(i)}1" t="n"><v>{token}</v></c>' for i, token in enumerate(tokens, 1))
    assert subject.read_sheets(_xlsx(cells)) == {
        "i_Si_e": {f"{subject.column(i)}1": token for i, token in enumerate(tokens, 1)}
    }


def test_read_ignores_strings_booleans_errors_and_empty_cells() -> None:
    cells = ('<c r="A1" t="s"><v>17</v></c><c r="A2" t="inlineStr"><is><t>123</t></is></c>'
             '<c r="A3" t="str"><v>123</v></c><c r="A4" t="b"><v>1</v></c>'
             '<c r="A5" t="e"><v>#N/A</v></c><c r="A6"/><c r="A7"><v>123</v></c>')
    assert subject.read_sheets(_xlsx(cells)) == {"i_Si_e": {"A7": "123"}}


def test_read_resolves_relationships_not_sheet_id_or_member_order() -> None:
    book = _BOOK + ('<sheet name="Second" sheetId="1" r:id="rId2"/>'
                    '<sheet name="First" sheetId="99" r:id="rId9"/></sheets></workbook>')
    rels = _RELS + ('<Relationship Id="rId9" Target="worksheets/source.xml"/>'
                    '<Relationship Id="rId2" Target="/xl/worksheets/other.xml"/></Relationships>')
    other = f'<worksheet xmlns="{_NS}"><sheetData><row><c r="Z8"><v>2.000</v></c></row></sheetData></worksheet>'
    assert subject.read_sheets(_xlsx(book=book, rels=rels, extra=[("xl/worksheets/other.xml", other)])) == {
        "Second": {"Z8": "2.000"}, "First": {"A1": "1"}
    }


@pytest.mark.parametrize("token", ["", " 1", "1 ", "NaN", "Infinity", "-inf", "1_000", "1,23", "1e", ".", "--1", "0x10"])
def test_read_rejects_non_numeric_or_non_raw_tokens(token: str) -> None:
    with pytest.raises(ValueError, match="numeric XML token"):
        subject.read_sheets(_xlsx(f'<c r="A1"><v>{token}</v></c>'))


@pytest.mark.parametrize("kind", ["", ' t="str"', ' t="s"'])
def test_read_rejects_formula_even_with_cached_numeric_result(kind: str) -> None:
    with pytest.raises(ValueError, match="cached formula"):
        subject.read_sheets(_xlsx(f'<c r="A1"{kind}><f>1+1</f><v>2</v></c>'))


@pytest.mark.parametrize("address", ["a1", "A0", "A01", "1A", "$A$1", "A-1", "", "A 1"])
def test_read_rejects_malformed_cell_addresses(address: str) -> None:
    with pytest.raises(ValueError, match="cell address"):
        subject.read_sheets(_xlsx(f'<c r="{address}"><v>1</v></c>'))


def test_read_rejects_duplicate_cell_including_ignored_text() -> None:
    with pytest.raises(ValueError, match="cell address"):
        subject.read_sheets(_xlsx('<c r="A1" t="s"><v>0</v></c><c r="A1"><v>1</v></c>'))


def test_read_rejects_duplicate_sheet_name() -> None:
    book = _BOOK + ('<sheet name="same" r:id="rId9"/><sheet name="same" r:id="rId9"/></sheets></workbook>')
    with pytest.raises(ValueError, match="worksheet name"):
        subject.read_sheets(_xlsx(book=book))


def test_read_rejects_duplicate_archive_members() -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        raw = _xlsx(extra=[("xl/workbook.xml", "duplicate")])
    with pytest.raises(ValueError, match="ZIP members"):
        subject.read_sheets(raw)


@pytest.mark.parametrize("links", [
    '<Relationship Id="rId9" Target="worksheets/source.xml" TargetMode="External"/>',
    '<Relationship Id="rId9" Target="../outside.xml"/>',
    '<Relationship Id="rId9" Target="/xl/../outside.xml"/>',
    '<Relationship Id="rId9" Target="worksheets/source.xml"/><Relationship Id="rId9" Target="worksheets/source.xml"/>',
])
def test_read_rejects_external_escaping_or_duplicate_relationship(links: str) -> None:
    with pytest.raises(ValueError):
        subject.read_sheets(_xlsx(rels=_RELS + links + "</Relationships>"))


def test_read_does_not_guess_missing_relationship() -> None:
    with pytest.raises(KeyError):
        subject.read_sheets(_xlsx(rels=_RELS + "</Relationships>"))


@pytest.fixture
def tail_sheets() -> dict[str, dict[str, str]]:
    cells = {f"A{row}": str(11000+(row-107)*2000) for row in range(107, 125)}
    cells.update({f"{subject.column(col)}{row}": f"{row}{col}.00" for row in range(107, 125) for col in range(2, 94)})
    return {"i_Si_e": cells, "i_Al_n": deepcopy(cells)}


def test_tail_full_rectangle_includes_first_last_cells(tail_sheets: dict[str, dict[str, str]]) -> None:
    report = subject.compare_tail(tail_sheets)
    assert report["range"] == "B107:CO124"
    assert report["cell_count"] == report["raw_matches"] == report["numeric_matches"] == 1656
    assert report["creation_history_established"] is False
    assert report["cells"][0]["cell"] == "B107"
    assert report["cells"][-1]["cell"] == "CO124"
    assert len({c["cell"] for c in report["cells"]}) == 1656


def test_tail_distinguishes_lexical_and_numeric_identity(tail_sheets: dict[str, dict[str, str]]) -> None:
    tail_sheets["i_Al_n"]["B107"] = "1072"
    tail_sheets["i_Al_n"]["CO124"] = "999"
    report = subject.compare_tail(tail_sheets)
    assert report["raw_matches"] == 1654 and report["numeric_matches"] == 1655
    first, last = report["cells"][0], report["cells"][-1]
    assert first["raw_token_identical"] is False and first["numeric_identical"] is True
    assert last["raw_token_identical"] is last["numeric_identical"] is False


def test_tail_energy_match_is_numeric_not_lexical(tail_sheets: dict[str, dict[str, str]]) -> None:
    tail_sheets["i_Al_n"]["A107"] = "1.1E4"
    assert subject.compare_tail(tail_sheets)["raw_matches"] == 1656


@pytest.mark.parametrize("sheet,address", [("i_Si_e", "B107"), ("i_Al_n", "CO124"), ("i_Si_e", "A107")])
def test_tail_missing_evidence_never_counted_as_match(tail_sheets: dict[str, dict[str, str]], sheet: str, address: str) -> None:
    tail_sheets[sheet].pop(address)
    with pytest.raises(KeyError):
        subject.compare_tail(tail_sheets)


@pytest.mark.parametrize("energy", ["10000", "0", "45001", "99999"])
def test_tail_rejects_outside_declared_energy_band(tail_sheets: dict[str, dict[str, str]], energy: str) -> None:
    tail_sheets["i_Si_e"]["A107"] = tail_sheets["i_Al_n"]["A107"] = energy
    with pytest.raises(ValueError, match="energy coordinates"):
        subject.compare_tail(tail_sheets)


def test_tail_rejects_mismatched_energy_coordinate(tail_sheets: dict[str, dict[str, str]]) -> None:
    tail_sheets["i_Al_n"]["A107"] = "12000"
    with pytest.raises(ValueError, match="energy coordinates"):
        subject.compare_tail(tail_sheets)


def _row(old: str, new: str, energy: str = "10000", metric: str = "Si_electronic", label: str = "a") -> dict[str, Any]:
    return {"workbook_value": old, "native_value": new, "energy_total_mev": energy, "metric": metric, "label": label}


def test_summary_uses_workbook_denominator_and_retains_worst_sign() -> None:
    result = subject.summarize([_row("2", "3", label="positive"), _row("2", "0", label="zero"), _row("4", "2", label="negative")])
    assert len(result) == 1 and result[0]["count"] == 3
    assert Decimal(result[0]["max_absolute_relative_percent"]) == 100
    assert Decimal(result[0]["worst"]["relative_percent_of_workbook"]) == -100
    assert result[0]["worst"]["label"] == "zero"


def test_summary_groups_metrics_and_10000_boundary_without_mutating_rows() -> None:
    rows = [_row("1", "1", "10000"), _row("1", "2", "10000.0001"),
            _row("2", "1", "1", "Al_projected_range_mm_semantics_unqualified")]
    before = deepcopy(rows)
    result = subject.summarize(rows)
    assert rows == before
    assert {(r["metric"], r["band"], r["count"]) for r in result} == {
        ("Si_electronic", "through_10000", 1), ("Si_electronic", "above_10000", 1),
        ("Al_projected_range_mm_semantics_unqualified", "through_10000", 1)}
    assert subject.summarize([]) == []


@pytest.mark.parametrize("old,new", [("0", "0"), ("0", "1"), ("-1", "1"), ("1", "-1"),
                                      ("NaN", "1"), ("Infinity", "1"), ("1", "NaN"), ("1", "Infinity")])
def test_summary_rejects_bad_denominator_and_nonfinite_values(old: str, new: str) -> None:
    with pytest.raises(ValueError):
        subject.summarize([_row(old, new)])


def test_summary_precision_independent_of_caller_context() -> None:
    expected = subject.summarize([_row("7", "8")])
    with localcontext() as context:
        context.prec = 3
        actual = subject.summarize([_row("7", "8")])
        assert context.prec == 3
    assert actual == expected
    assert len(expected[0]["max_absolute_relative_percent"]) > 50


def test_run_rejects_unpinned_workbook_before_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workbook = tmp_path / "different.xlsx"
    workbook.write_bytes(_xlsx())
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No backend access before workbook fingerprint passes")
    monkeypatch.setattr(subject, "load_srim_hze_dataset", forbidden)
    monkeypatch.setattr(subject, "parse_srim_range_output", forbidden)
    with pytest.raises(ValueError, match="Changed pinned evidence"):
        subject.run(workbook)


@pytest.fixture
def pinned_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tail_sheets: dict[str, dict[str, str]]) -> dict[str, Any]:
    workbook, bundle = tmp_path / "workbook.xlsx", tmp_path / "bundle.zip"
    workbook.write_bytes(b"mock pinned workbook")
    bundle.write_bytes(b"mock pinned native bundle")
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "WORKBOOK_SHA", hashlib.sha256(workbook.read_bytes()).hexdigest())
    monkeypatch.setattr(subject, "BUNDLE", bundle)
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    monkeypatch.setattr(subject, "BUNDLE_SHA", bundle_sha)
    sheets = deepcopy(tail_sheets)
    sheets["i_Si_n"] = deepcopy(sheets["i_Si_e"])
    sheets["i_R_Al"] = deepcopy(sheets["i_Si_e"])
    sheets.update({f"unused_{i}": {} for i in range(5)})
    monkeypatch.setattr(subject, "read_sheets", lambda raw: sheets)
    def point() -> SimpleNamespace:
        return SimpleNamespace(energy_total_mev=Decimal(11000), electronic_mev_cm2_mg=Decimal(1),
                               nuclear_mev_cm2_mg=Decimal("0.01"), projected_range_cm=Decimal(2))
    tables = {z: SimpleNamespace(mass_parameter_u_decimal=Decimal(z * 2),
                                native_al=SimpleNamespace(rows=(point(),)),
                                native_si=SimpleNamespace(rows=(point(),))) for z in range(2, 17)}
    def load(path: Path, *, expected_sha256: str) -> SimpleNamespace:
        assert path == bundle and expected_sha256 == bundle_sha
        return SimpleNamespace(tables_by_z=tables)
    monkeypatch.setattr(subject, "load_srim_hze_dataset", load)
    native_paths: list[Path] = []
    report_paths: list[Path] = []
    reports: dict[str, str] = {}
    for kind, roster in (("baseline", [2]), ("upper", list(range(3, 17)))):
        cases = []
        for z in roster:
            for target in [13, 14]:
                raw = json.dumps({"z": z, "target": target, "mass": str(z * 2)}).encode()
                repeats = []
                for repeat in range(2):
                    path = tmp_path / f"raw_{z}_{target}_{repeat}.txt"
                    path.write_bytes(raw)
                    native_paths.append(path)
                    repeats.append({"output": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}})
                cases.append({"case": {"ion_z": z, "target_z": target, "ion_mass_u": str(z * 2)},
                              "status": "CONTROL_COMPLETE", "repeats": repeats})
        report = tmp_path / f"outputs/srim_legacy_{kind}_final_20261101/report.json"
        report.parent.mkdir(parents=True)
        raw = json.dumps({"cases": cases}).encode()
        report.write_bytes(raw)
        report_paths.append(report)
        reports[kind] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(subject, "REPORTS", reports)
    def parse(raw: bytes) -> SimpleNamespace:
        data = json.loads(raw)
        return SimpleNamespace(ion_z=data["z"], target_z=data["target"], ion_mass_u=Decimal(data["mass"]),
                               raw_sha256=hashlib.sha256(raw).hexdigest(), rows=(point(),))
    monkeypatch.setattr(subject, "parse_srim_range_output", parse)
    return {"workbook": workbook, "bundle": bundle, "native": native_paths, "reports": report_paths,
            "sheets": sheets, "parse": parse, "tables": tables}


def test_run_mocked_retained_tables_keeps_diagnostic_scope_and_pins(pinned_run: dict[str, Any]) -> None:
    paths = [pinned_run["workbook"], pinned_run["bundle"], *pinned_run["native"], *pinned_run["reports"]]
    before = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = subject.run(pinned_run["workbook"])
    assert result["schema_version"] == "radar.legacy_workbook_srim_audit.v1"
    assert result["workbook_modified"] is result["normative_export_allowed"] is result["physical_accuracy_verified"] is False
    assert len(result["mass_mapping"]) == 15 and all(row["same_mass"] for row in result["mass_mapping"])
    assert len(result["rows"]) == 45
    assert result["current_same_mass_native_rows_identical"] == 30
    assert result["input_sha256"] == before
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in before.items())


@pytest.mark.parametrize("group", ["bundle", "reports", "native"])
def test_run_rejects_changed_pinned_evidence_before_comparing(pinned_run: dict[str, Any], group: str) -> None:
    path = pinned_run[group] if group == "bundle" else pinned_run[group][0]
    path.write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="Changed pinned evidence"):
        subject.run(pinned_run["workbook"])


@pytest.mark.parametrize("group", ["workbook", "bundle", "reports", "native"])
def test_run_rechecks_evidence_after_comparison(pinned_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch, group: str) -> None:
    path = pinned_run[group] if group in {"workbook", "bundle"} else pinned_run[group][0]
    parser = pinned_run["parse"]
    changed = False
    def mutate_once(raw: bytes) -> SimpleNamespace:
        nonlocal changed
        result = parser(raw)
        assert isinstance(result, SimpleNamespace)
        if not changed:
            path.write_bytes(b"changed after read")
            changed = True
        return result
    monkeypatch.setattr(subject, "parse_srim_range_output", mutate_once)
    with pytest.raises(ValueError, match="evidence changed during calculation"):
        subject.run(pinned_run["workbook"])


def test_run_requires_every_native_energy_match(pinned_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    parser = pinned_run["parse"]
    def unmatched(raw: bytes) -> SimpleNamespace:
        result = parser(raw)
        assert isinstance(result, SimpleNamespace)
        result.rows[0].energy_total_mev = Decimal(11001)
        return result
    monkeypatch.setattr(subject, "parse_srim_range_output", unmatched)
    with pytest.raises(ValueError, match="workbook coordinate"):
        subject.run(pinned_run["workbook"])


def test_run_rejects_ambiguous_roundtrip_energy_match(pinned_run: dict[str, Any]) -> None:
    pinned_run["sheets"]["i_Si_e"]["A106"] = "11000.000000000001"
    with pytest.raises(ValueError, match="workbook coordinate"):
        subject.run(pinned_run["workbook"])


def test_run_rejects_duplicate_numeric_energy_coordinate(pinned_run: dict[str, Any]) -> None:
    pinned_run["sheets"]["i_R_Al"]["A106"] = "11000.0"
    with pytest.raises(ValueError, match="Duplicate workbook energy"):
        subject.run(pinned_run["workbook"])


def test_run_rejects_nuclear_energy_coordinate_shift(pinned_run: dict[str, Any]) -> None:
    pinned_run["sheets"]["i_Si_n"]["A107"] = "12000"
    with pytest.raises(ValueError, match="Electronic and nuclear"):
        subject.run(pinned_run["workbook"])


def test_run_accepts_unique_binary64_roundtrip_but_records_it(pinned_run: dict[str, Any]) -> None:
    for name in ["i_Si_e", "i_Al_n", "i_Si_n", "i_R_Al"]:
        pinned_run["sheets"][name]["A107"] = "11000.000000000001"
    result = subject.run(pinned_run["workbook"])
    assert {row["energy_match"] for row in result["rows"]} == {"binary64_roundtrip_at_most_8_ULP"}
    assert {row["workbook_energy_token"] for row in result["rows"]} == {"11000.000000000001"}
    assert {row["energy_total_mev"] for row in result["rows"]} == {"11000"}


@pytest.mark.parametrize("target", ["native_al", "native_si"])
def test_run_rejects_changed_current_same_mass_row(pinned_run: dict[str, Any], target: str) -> None:
    native = getattr(pinned_run["tables"][2], target)
    native.rows[0].electronic_mev_cm2_mg = Decimal(2)
    with pytest.raises(ValueError, match="same-mass table changed"):
        subject.run(pinned_run["workbook"])


def test_run_distinguishes_different_mass_from_changed_stopping_table(pinned_run: dict[str, Any]) -> None:
    pinned_run["tables"][2].mass_parameter_u_decimal = Decimal("4.1")
    pinned_run["tables"][2].native_al.rows[0].electronic_mev_cm2_mg = Decimal(2)
    result = subject.run(pinned_run["workbook"])
    mass = next(row for row in result["mass_mapping"] if row["z"] == 2)
    assert mass == {"z": 2, "legacy_control_mass": "4", "current_source_mass": "4.1", "same_mass": False}
    assert result["current_same_mass_native_rows_identical"] == 28
    assert result["physical_accuracy_verified"] is False


def _minimal_result() -> dict[str, Any]:
    return {"tail_identity": {"raw_matches": 1656}, "summaries": [], "workbook_modified": False,
            "normative_export_allowed": False, "physical_accuracy_verified": False}


def test_cli_exclusive_output_and_unmodified_workbook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    workbook, output = tmp_path / "input.xlsx", tmp_path / "report.json"
    workbook.write_bytes(b"unchanged input")
    result = _minimal_result()
    def run(path: Path) -> dict[str, Any]:
        assert path == workbook
        return result
    monkeypatch.setattr(subject, "run", run)
    monkeypatch.setattr(sys, "argv", ["audit", "--workbook", str(workbook), "--output", str(output)])
    assert subject.main() == 0
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert workbook.read_bytes() == b"unchanged input"
    announcement = json.loads(capsys.readouterr().out)
    assert announcement["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_cli_existing_output_fails_before_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "report.json"
    output.write_bytes(b"existing evidence")
    def forbidden(path: Path) -> dict[str, Any]:
        raise AssertionError("Audit should not run for an existing output")
    monkeypatch.setattr(subject, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    with pytest.raises(FileExistsError):
        subject.main()
    assert output.read_bytes() == b"existing evidence"


def test_cli_output_created_during_audit_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "report.json"
    def raced(path: Path) -> dict[str, Any]:
        output.write_bytes(b"concurrent evidence")
        return _minimal_result()
    monkeypatch.setattr(subject, "run", raced)
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    with pytest.raises(FileExistsError):
        subject.main()
    assert output.read_bytes() == b"concurrent evidence"
