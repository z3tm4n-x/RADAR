"""Direct-node SRIM/legacy CSV diagnostics, without interpolation or promotion.

The legacy range's physical definition and its SRIM ancestry are unresolved.
Matching a projected range is a comparison, not proof that the old column is
projected or that it is appropriate for a CSDA transport operator.
"""

from __future__ import annotations

import csv
from decimal import Decimal, localcontext
import hashlib
import io
from math import isfinite, ulp
import re
from typing import Any

from radar.validation.srim_output import SrimRangeOutput


_HEADERS = {
    "stopping/proton_al_range.csv": (
        "energy_mev", "stopping_mev_cm2_g", "range_g_cm2", "range_mm", "source_sheet",
    ),
    "stopping/proton_si_let.csv": (
        "energy_mev", "let_electronic_mev_cm2_mg", "let_nuclear_mev_cm2_mg",
        "let_total_mev_cm2_mg", "electronic_column", "nuclear_column", "source_sheet",
    ),
    "stopping/hze_al_range.csv": (
        "z", "symbol", "mass_number", "mass_to_charge", "energy_mev_per_nucleon",
        "range_g_cm2", "range_mm", "source_sheet",
    ),
    "stopping/hze_si_let.csv": (
        "z", "symbol", "mass_number", "mass_to_charge", "energy_mev_per_nucleon",
        "let_electronic_mev_cm2_mg", "let_nuclear_mev_cm2_mg",
        "let_total_mev_cm2_mg", "electronic_source_sheet", "nuclear_source_sheet",
    ),
}
_COUNTS = (157, 105, 1845, 1845)
_IONS = {
    2: ("He", 4), 6: ("C", 12), 7: ("N", 14), 8: ("O", 16),
    10: ("Ne", 20), 12: ("Mg", 24), 14: ("Si", 28), 16: ("S", 32),
    18: ("Ar", 40), 20: ("Ca", 40), 24: ("Cr", 52), 25: ("Mn", 55),
    26: ("Fe", 56), 27: ("Co", 59), 28: ("Ni", 58),
}
_DECIMAL = re.compile(r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]{1,3})?")
_LEGACY_DENSITY = Decimal("2.70")


def _number(value: str, *, zero: bool = False) -> Decimal:
    if len(value) > 128 or _DECIMAL.fullmatch(value) is None:
        raise ValueError("Legacy CSV numeric fields require bounded decimal tokens.")
    result = Decimal(value)
    if not result.is_finite() or result < 0 or (not zero and result == 0):
        raise ValueError("Legacy CSV numeric fields must be finite and nonnegative.")
    if not isfinite(float(result)) or (result > 0 and float(result) == 0):
        raise ValueError("Legacy numeric field is outside finite binary64 coordinate support.")
    return result


def _roundtrip_match(left: Decimal, right: Decimal) -> bool:
    """At most eight binary64 ULPs, exclusively for energy/metadata conversion.

    This is neither a stopping/range-value tolerance nor an uncertainty bound.
    It accommodates the retained float E/A workbook conversion and nothing is
    interpolated. The exact energy discrepancy remains in the returned ledger.
    """
    binary = (float(left), float(right))
    if any(not isfinite(value) or value <= 0 for value in binary):
        raise ValueError("Energy/metadata coordinate is outside finite positive binary64 support.")
    if left == right:
        return True
    tolerance = Decimal.from_float(8 * max(ulp(value) for value in binary))
    return abs(left - right) <= tolerance


def _read_legacy(raw: bytes, path: str) -> list[dict[str, Any]]:
    if path not in _HEADERS:
        raise ValueError("Unsupported legacy stopping resource path.")
    if not isinstance(raw, bytes) or not raw or len(raw) > 2_000_000:
        raise ValueError("Expected bounded nonempty CSV bytes.")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Legacy CSV must contain ASCII fields.") from exc
    if any(ord(c) < 32 and c not in "\r\n" for c in text):
        raise ValueError("Unexpected legacy CSV control character.")
    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise ValueError("Malformed legacy CSV.") from exc
    count = dict(zip(_HEADERS, _COUNTS, strict=True))[path]
    headers = _HEADERS[path]
    if len(records) != count + 1 or tuple(records[0]) != headers:
        raise ValueError("Legacy CSV header or complete record count differs.")
    is_hze = "hze_" in path
    is_al = "_al_" in path
    previous: dict[int, Decimal] = {}
    populations: dict[int, int] = {}
    result: list[dict[str, Any]] = []
    for index, cells in enumerate(records[1:], 1):
        if len(cells) != len(headers):
            raise ValueError("Malformed legacy CSV row width.")
        original = dict(zip(headers, cells, strict=True))
        row: dict[str, Any] = {"source_row": index, "raw_fields": original}
        if is_hze:
            if not original["z"].isdigit():
                raise ValueError("Legacy HZE charge must be an integer.")
            z = int(original["z"])
            mass = _number(original["mass_number"])
            if (z not in _IONS or original["symbol"] != _IONS[z][0]
                    or mass != Decimal(_IONS[z][1])):
                raise ValueError("Legacy HZE ion/mass roster differs.")
            ratio = _number(original["mass_to_charge"])
            if not _roundtrip_match(ratio, mass / z):
                raise ValueError("Legacy HZE A/Z is inconsistent with the retained mass.")
            energy = _number(original["energy_mev_per_nucleon"])
            row.update(mass_u=mass, energy_mev_per_nucleon=energy,
                       energy_total_mev=energy * mass)
        else:
            z = 1
            energy = _number(original["energy_mev"])
            row["energy_total_mev"] = energy
        row["z"] = z
        if z in previous and energy <= previous[z]:
            raise ValueError("Duplicate or unordered legacy energy coordinate.")
        previous[z] = energy
        populations[z] = populations.get(z, 0) + 1
        if is_al:
            if original["source_sheet"] != ("i_R_Al" if is_hze else "p_Al"):
                raise ValueError("Unexpected legacy range source sheet.")
            row["range_cm"] = _number(original["range_mm"]) / 10
            row["range_g_cm2"] = _number(original["range_g_cm2"])
            if not is_hze:
                row["stopping_total"] = _number(original["stopping_mev_cm2_g"]) / 1000
        else:
            if is_hze:
                if (original["electronic_source_sheet"], original["nuclear_source_sheet"]) != (
                    "i_Si_e", "i_Si_n",
                ):
                    raise ValueError("Unexpected legacy HZE stopping source sheets.")
            elif (original["electronic_column"], original["nuclear_column"],
                  original["source_sheet"]) != ("H_coll", "", "p_Si"):
                raise ValueError("Unsupported legacy proton stopping-column provenance.")
            row.update(
                electronic=_number(original["let_electronic_mev_cm2_mg"], zero=True),
                nuclear=_number(original["let_nuclear_mev_cm2_mg"], zero=True),
                stopping_total=_number(original["let_total_mev_cm2_mg"], zero=True),
            )
        result.append(row)
    if is_hze and populations != dict.fromkeys(_IONS, 123):
        raise ValueError("Expected all 15 legacy ions with exactly 123 records each.")
    return result


def _metric(legacy: Decimal, native: Decimal, meaning: str) -> dict[str, Any]:
    delta = native - legacy
    with localcontext() as context:
        context.prec = 80
        relative = None if legacy == 0 else 100 * delta / legacy
    return {
        "legacy": legacy, "native_srim": native, "native_minus_legacy": delta,
        "exact_decimal_match": legacy == native,
        "relative_difference_percent_of_legacy": relative,
        "comparison_meaning": meaning,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[Decimal, dict[str, Any]]]] = {}
    for record in records:
        for name, metric in record.get("comparisons", {}).items():
            grouped.setdefault(name, []).append((record["native_energy_total_mev"], metric))
    summary: dict[str, Any] = {}
    for name, values in grouped.items():
        nonzero = [(e, m) for e, m in values
                   if m["relative_difference_percent_of_legacy"] is not None]
        largest_absolute = max(values, key=lambda p: abs(p[1]["native_minus_legacy"]))
        largest_relative = (max(nonzero, key=lambda p:
                                abs(p[1]["relative_difference_percent_of_legacy"]))
                            if nonzero else None)
        summary[name] = {
            "compared_rows": len(values),
            "exact_decimal_matches": sum(m["exact_decimal_match"] for _, m in values),
            "zero_legacy_denominator_rows": len(values) - len(nonzero),
            "max_absolute_difference": abs(largest_absolute[1]["native_minus_legacy"]),
            "max_absolute_difference_energy_total_mev": largest_absolute[0],
            "max_absolute_relative_difference_percent": None if largest_relative is None
            else abs(largest_relative[1]["relative_difference_percent_of_legacy"]),
            "max_relative_difference_energy_total_mev": None if largest_relative is None
            else largest_relative[0],
            "min_signed_relative_difference_percent": min(
                (m["relative_difference_percent_of_legacy"] for _, m in nonzero), default=None),
            "max_signed_relative_difference_percent": max(
                (m["relative_difference_percent_of_legacy"] for _, m in nonzero), default=None),
        }
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def compare_legacy_table(
    output: SrimRangeOutput, csv_bytes: bytes, resource_path: str,
) -> dict[str, Any]:
    """Compare only uniquely matched native nodes, retaining every selected CSV row.

    Decimal differences are exact to the supplied printed columns. The relative
    mismatch denominator is always the legacy value; a zero denominator is null.
    Rounded/native equality is not upstream identification or physical validation.
    """
    if not isinstance(output, SrimRangeOutput):
        raise ValueError("A parsed SRIM range output is required.")
    with localcontext() as context:
        # Legacy tokens are <=128 characters and binary64-bounded; SRIM tokens
        # are <=64 characters with exponents bounded to 100. This covers every
        # allowed exact addition, subtraction and unit/density multiplication.
        context.prec = 1000
        legacy = _read_legacy(csv_bytes, resource_path)
        is_hze = "hze_" in resource_path
        is_al = "_al_" in resource_path
        if output.target_z != (13 if is_al else 14):
            raise ValueError("SRIM target does not match the legacy resource.")
        if (not is_hze and output.ion_z != 1) or (is_hze and output.ion_z not in _IONS):
            raise ValueError("SRIM ion is absent from this legacy resource.")
        selected = [row for row in legacy if row["z"] == output.ion_z]
        if is_hze and any(row["mass_u"] != output.ion_mass_u for row in selected):
            raise ValueError("SRIM ion mass differs from the legacy HZE mass.")
        rows: list[dict[str, Any]] = []
        used: set[int] = set()
        counts = {"matched_exact_decimal_energy": 0, "matched_binary64_roundtrip_energy": 0,
                  "unmatched_below_native_grid": 0, "unmatched_above_native_grid": 0,
                  "unmatched_inside_native_grid": 0}
        for old in selected:
            energy = old["energy_total_mev"]
            matches = [i for i, new in enumerate(output.rows)
                       if _roundtrip_match(energy, new.energy_total_mev)]
            if len(matches) > 1 or (matches and matches[0] in used):
                raise ValueError("Ambiguous many-to-one energy match; no nearest-node choice allowed.")
            entry: dict[str, Any] = {
                "legacy_source_row": old["source_row"], "legacy_raw_fields": old["raw_fields"],
                "legacy_energy_total_mev": energy,
            }
            if not matches:
                key = ("unmatched_below_native_grid" if energy < output.rows[0].energy_total_mev
                       else "unmatched_above_native_grid"
                       if energy > output.rows[-1].energy_total_mev
                       else "unmatched_inside_native_grid")
                counts[key] += 1
                entry["match_status"] = key
                rows.append(entry)
                continue
            position = matches[0]
            used.add(position)
            new = output.rows[position]
            key = ("matched_exact_decimal_energy" if energy == new.energy_total_mev
                   else "matched_binary64_roundtrip_energy")
            counts[key] += 1
            entry.update(match_status=key, native_row=position + 1,
                         native_energy_total_mev=new.energy_total_mev,
                         native_printed_tokens=new.printed_tokens,
                         native_minus_legacy_energy_mev=new.energy_total_mev - energy)
            comparisons: dict[str, dict[str, Any]] = {}
            total = new.electronic_mev_cm2_mg + new.nuclear_mev_cm2_mg
            if is_al:
                comparisons["range_cm_vs_srim_projected_cm"] = _metric(
                    old["range_cm"], new.projected_range_cm,
                    "Legacy range kind unverified; SRIM projected range is not CSDA.")
                for label, density in (("legacy_density", _LEGACY_DENSITY),
                                       ("native_density", output.density_g_cm3)):
                    comparisons[f"range_g_cm2_vs_projected_at_{label}"] = _metric(
                        old["range_g_cm2"], new.projected_range_cm * density,
                        "Declared density conversion only; legacy range semantics unverified.")
                entry["legacy_range_internal_mass_conversion_residual_g_cm2"] = (
                    old["range_g_cm2"] - old["range_cm"] * _LEGACY_DENSITY)
            else:
                comparisons["electronic_stopping_mev_cm2_mg"] = _metric(
                    old["electronic"], new.electronic_mev_cm2_mg,
                    "Electronic-labelled legacy column versus native electronic stopping.")
                if is_hze:
                    comparisons["nuclear_stopping_mev_cm2_mg"] = _metric(
                        old["nuclear"], new.nuclear_mev_cm2_mg,
                        "Native nuclear stopping versus legacy nuclear column.")
                else:
                    entry.update(legacy_nuclear_source_missing=True,
                                 legacy_nuclear_placeholder=old["nuclear"],
                                 native_nuclear_mev_cm2_mg=new.nuclear_mev_cm2_mg)
                entry["legacy_total_component_residual_mev_cm2_mg"] = (
                    old["stopping_total"] - old["electronic"] - old["nuclear"])
            if not is_hze or not is_al:
                comparisons["total_stopping_mev_cm2_mg"] = _metric(
                    old["stopping_total"], total,
                    "Legacy total is incomplete: nuclear source absent."
                    if not is_hze and not is_al else
                    "Native printed electronic+nuclear sum versus legacy total column.")
            entry["comparisons"] = comparisons
            rows.append(entry)
        report: dict[str, Any] = {
            "schema_version": "radar.srim_legacy_direct_nodes.v1",
            "resource_path": resource_path,
            "legacy_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "srim_output_sha256": output.raw_sha256,
            "native_ion_z": output.ion_z, "native_ion_mass_u": output.ion_mass_u,
            "legacy_ion_mass_recorded": is_hze,
            "native_target_z": output.target_z, "native_density_g_cm3": output.density_g_cm3,
            "legacy_al_density_conversion_g_cm3": _LEGACY_DENSITY if is_al else None,
            "legacy_range_kind_verified": False, "native_range_kind": "projected",
            "legacy_total_rows": len(legacy), "legacy_selected_ion_rows": len(selected),
            "native_row_count": len(output.rows), "counts": counts, "rows": rows,
            "native_rows_without_legacy_match": [
                {"native_row": i + 1, "energy_total_mev": row.energy_total_mev,
                 "native_printed_tokens": row.printed_tokens}
                for i, row in enumerate(output.rows) if i not in used
            ],
            "summary": _summary(rows),
            "energy_match_policy": "Exact Decimal or <=8 binary64 ULPs; unique direct node only.",
            "numeric_value_tolerance": "None; all differences retained.",
            "relative_percentage_decimal_precision": 80,
            "comparison_status": "INCOMPLETE",
            "legacy_upstream_identity_verified": False, "physical_accuracy_verified": False,
            "normative_export_allowed": False, "production_data_changed": False,
            "limitations": [
                "No interpolation, extrapolation, correction fit or source substitution.",
                "Exact printed equality is not equality to unprinted native arrays.",
                "This comparison does not establish SRIM ancestry of the workbook.",
                "A range difference is not an error until the legacy range definition is established.",
                "Unmatched energies and untested ions are not zero physical contributions.",
            ],
        }
        safe = _json_safe(report)
        assert isinstance(safe, dict)
        return safe
