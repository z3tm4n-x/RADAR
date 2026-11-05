"""Read-only, pinned workbook comparison; never repair cells or qualify physics."""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from radar.validation.srim_output import parse_srim_range_output  # noqa: E402
from radar.validation.srim_comparison import _roundtrip_match  # noqa: E402
from radar.shielding.srim_dataset import load_srim_hze_dataset  # noqa: E402

WORKBOOK_SHA = "d4ed7928331a0b7abcdfe1b585504b136809a3f6823f220edfed8967c28ec684"
REPORTS = {
    "baseline": "8ed401964181371ebb508f16bec41aa5b3600b136f47f763be4c0596f448c9cc",
    "upper": "3bbb3992b432074860b049820b065d8f47dd19d5cfc9cd074d8541f41754c029",
}
BUNDLE = ROOT / "outputs/srim_upper_full_20261105/observed_upper_full.zip"
BUNDLE_SHA = "d4754dce59446a44d8c96ffe9853feb4102a70ade2e3e839737eaff63289434c"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def column(index: int) -> str:
    if type(index) is not int or index < 1:
        raise ValueError("Positive Excel column index required.")
    result = ""
    while index:
        index, remainder = divmod(index-1, 26)
        result = chr(65+remainder)+result
    return result


def read_sheets(raw: bytes) -> dict[str, dict[str, str]]:
    """Retain raw XML numeric tokens, not rounded/cached formula results."""
    result: dict[str, dict[str, str]] = {}
    with ZipFile(io.BytesIO(raw)) as archive:
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError("Duplicate ZIP members.")
        links = {}
        for link in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels")):
            identity = link.attrib["Id"]
            if identity in links or link.get("TargetMode") == "External":
                raise ValueError("Duplicate or external workbook relationship.")
            target = PurePosixPath(link.attrib["Target"])
            path = str(target).lstrip("/") if target.is_absolute() else str(PurePosixPath("xl")/target)
            if ".." in PurePosixPath(path).parts:
                raise ValueError("Workbook relationship escapes its archive location.")
            links[identity] = path
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in book.findall(f"{NS}sheets/{NS}sheet"):
            name = sheet.attrib["name"]
            if name in result:
                raise ValueError("Duplicate worksheet name.")
            cells: dict[str, str] = {}
            seen = set()
            root = ET.fromstring(archive.read(links[sheet.attrib[RID]]))
            for cell in root.iter(f"{NS}c"):
                address = cell.attrib["r"]
                if address in seen or re.fullmatch(r"[A-Z]+[1-9][0-9]*", address) is None:
                    raise ValueError("Duplicate or malformed cell address.")
                seen.add(address)
                if cell.find(f"{NS}f") is not None:
                    raise ValueError("This raw-value audit cannot substitute cached formula results.")
                value = cell.find(f"{NS}v")
                if cell.get("t", "n") != "n" or value is None:
                    continue
                token = value.text or ""
                if re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", token) is None:
                    raise ValueError("Malformed numeric XML token.")
                try:
                    number = Decimal(token)
                except InvalidOperation as exc:
                    raise ValueError("Invalid numeric XML token.") from exc
                if not number.is_finite():
                    raise ValueError("Nonfinite workbook value.")
                cells[address] = token
            result[name] = cells
    return result


def compare_tail(sheets: dict[str, dict[str, str]]) -> dict[str, Any]:
    left, right = sheets["i_Si_e"], sheets["i_Al_n"]
    comparisons = []
    for row in range(107, 125):
        energy = Decimal(left[f"A{row}"])
        if energy != Decimal(right[f"A{row}"]) or not 10000 < energy <= 45000:
            raise ValueError("Tail energy coordinates differ.")
        for col in range(2, 94):
            address = f"{column(col)}{row}"
            a, b = left[address], right[address]
            comparisons.append({"cell": address, "energy_total_mev": str(energy),
                                "si_e_token": a, "al_n_token": b,
                                "raw_token_identical": a == b,
                                "numeric_identical": Decimal(a) == Decimal(b)})
    return {"left_sheet": "i_Si_e", "right_sheet": "i_Al_n", "range": "B107:CO124",
            "cell_count": len(comparisons), "raw_matches": sum(c["raw_token_identical"] for c in comparisons),
            "numeric_matches": sum(c["numeric_identical"] for c in comparisons), "cells": comparisons,
            "creation_history_established": False}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        old, new = Decimal(row["workbook_value"]), Decimal(row["native_value"])
        if not old.is_finite() or not new.is_finite() or old <= 0 or new < 0:
            raise ValueError("Positive finite denominator and nonnegative finite native value required.")
        with localcontext() as context:
            context.prec = 80
            relative = (new/old-1)*100
        groups[row["metric"], "through_10000" if Decimal(row["energy_total_mev"]) <= 10000 else "above_10000"].append(
            {**row, "relative_percent_of_workbook": str(relative)})
    return [{"metric": key[0], "band": key[1], "count": len(values),
             "max_absolute_relative_percent": str(max(Decimal(v["relative_percent_of_workbook"]).copy_abs() for v in values)),
             "worst": max(values, key=lambda v: Decimal(v["relative_percent_of_workbook"]).copy_abs())}
            for key, values in sorted(groups.items())]


def run(workbook: Path) -> dict[str, Any]:
    pins: dict[str, str] = {}
    def read(path: Path, expected: str) -> bytes:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Changed pinned evidence: {path}")
        pins[str(path.resolve())] = expected
        return raw
    sheets = read_sheets(read(workbook, WORKBOOK_SHA))
    tail = compare_tail(sheets)
    if len(sheets) != 9 or tail["cell_count"] != 1656:
        raise ValueError("Unexpected workbook inventory.")
    dataset = load_srim_hze_dataset(BUNDLE, expected_sha256=BUNDLE_SHA)
    read(BUNDLE, BUNDLE_SHA)
    rows, masses, seen = [], {}, set()
    current_same_mass_rows = 0
    for kind, sha in REPORTS.items():
        path = ROOT / f"outputs/srim_legacy_{kind}_final_20261101/report.json"
        report = json.loads(read(path, sha))
        for case in report["cases"]:
            metadata = case["case"]
            z, target = metadata["ion_z"], metadata["target_z"]
            if kind == "baseline" and z != 2:
                continue
            if (z, target) in seen or case["status"] != "CONTROL_COMPLETE":
                raise ValueError("Duplicate or unsuccessful selected control.")
            seen.add((z, target))
            repeats = case["repeats"]
            if len(repeats) != 2:
                raise ValueError("Two retained repeats required.")
            raws = [read(Path(r["output"]["path"]), r["output"]["sha256"]) for r in repeats]
            if raws[0] != raws[1]:
                raise ValueError("Retained native repeats disagree.")
            native = parse_srim_range_output(raws[0])
            if native.ion_z != z or native.target_z != target or str(native.ion_mass_u) != metadata["ion_mass_u"]:
                raise ValueError("Retained native source identity disagrees.")
            current = dataset.tables_by_z[z]
            masses[z] = {"z": z, "legacy_control_mass": str(native.ion_mass_u),
                         "current_source_mass": str(current.mass_parameter_u_decimal),
                         "same_mass": native.ion_mass_u == current.mass_parameter_u_decimal}
            data = sheets["i_Si_e" if target == 14 else "i_R_Al"]
            energy_cells = [(Decimal(value), int(address[1:])) for address, value in data.items()
                            if re.fullmatch(r"A[0-9]+", address)]
            energy_rows = dict(energy_cells)
            if len(energy_rows) != len(energy_cells):
                raise ValueError("Duplicate workbook energy coordinate.")
            current_native = current.native_al if target == 13 else current.native_si
            current_by_energy = {p.energy_total_mev: p for p in current_native.rows}
            for point in native.rows:
                matches = [(energy, row_number) for energy, row_number in energy_rows.items()
                           if _roundtrip_match(energy, point.energy_total_mev)]
                if len(matches) != 1:
                    raise ValueError("Every retained native node needs one exact/8-ULP workbook coordinate.")
                workbook_energy, row_number = matches[0]
                if target == 14 and Decimal(sheets["i_Si_n"][f"A{row_number}"]) != workbook_energy:
                    raise ValueError("Electronic and nuclear workbook energy coordinates disagree.")
                if masses[z]["same_mass"]:
                    if current_by_energy.get(point.energy_total_mev) != point:
                        raise ValueError("Current same-mass table changed a retained native row.")
                    current_same_mass_rows += 1
                address = f"{column(z+1)}{row_number}"
                metrics = (("Si_electronic", "i_Si_e", point.electronic_mev_cm2_mg),
                           ("Si_nuclear", "i_Si_n", point.nuclear_mev_cm2_mg)) if target == 14 else (
                               ("Al_projected_range_mm_semantics_unqualified", "i_R_Al", point.projected_range_cm*10),)
                for metric, sheet_name, value in metrics:
                    rows.append({"z": z, "legacy_mass": str(native.ion_mass_u), "metric": metric,
                                 "sheet": sheet_name, "cell": address,
                                 "energy_total_mev": str(point.energy_total_mev),
                                 "workbook_energy_token": data[f"A{row_number}"],
                                 "energy_match": "exact_decimal" if workbook_energy == point.energy_total_mev
                                     else "binary64_roundtrip_at_most_8_ULP",
                                 "workbook_value": sheets[sheet_name][address], "native_value": str(value),
                                 "native_raw_sha256": native.raw_sha256})
    if len(seen) != 30 or len(masses) != 15:
        raise ValueError("Expected all 15 legacy controls in Al and Si.")
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha for p, sha in pins.items()):
        raise ValueError("Audit evidence changed during calculation.")
    return {"schema_version": "radar.legacy_workbook_srim_audit.v1", "workbook_modified": False,
            "normative_export_allowed": False, "physical_accuracy_verified": False,
            "tail_identity": tail, "mass_mapping": list(masses.values()), "summaries": summarize(rows),
            "current_same_mass_native_rows_identical": current_same_mass_rows,
            "rows": rows, "input_sha256": pins, "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": ["Exact-node comparison of 15 retained legacy-mass controls, not every workbook column.",
                            "Ten old/new source mass changes must not be attributed solely to stopping physics.",
                            "Projected range is not the CSDA range used by the new diagnostic transport.",
                            "Matching cell contents do not prove how the workbook was originally created."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path("C:/Users/Иван/Desktop/ЛПЭ.xlsx"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.workbook)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "raw_tail_matches": result["tail_identity"]["raw_matches"], "summaries": result["summaries"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
