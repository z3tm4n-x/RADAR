"""Inert current-online ASTAR parsing, semantic guards and printed precision."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from math import inf, nan
from pathlib import Path
import re

import pytest

from radar.validation.nist_astar_live import (
    CURRENT_DEFAULT_ENERGIES_MEV_TOTAL,
    LiveAstarResponse,
    half_printed_unit,
    half_printed_unit_decimal,
    parse_astar_live,
)


_ROOT = Path(__file__).resolve().parents[1]
_ROW = re.compile(
    r"(?<=<br>)[ \t]+(?:[1-9]\.[0-9]{3}E[+-][0-9]{2}[ \t]+){6}[01]\.[0-9]{4}(?=<br>)",
)


@pytest.fixture(scope="module")
def aluminum_html() -> str:
    return (_ROOT / "research/nist_star_20261031/live_astar_013.html").read_text(encoding="utf-8")


def _row_change(text: str, index: int, column: int, replacement: str) -> str:
    rows = tuple(_ROW.finditer(text))
    assert len(rows) == 121
    row = rows[index]
    tokens = row.group().split()
    tokens[column] = replacement
    return text[:row.start()] + "  " + "  ".join(tokens) + text[row.end():]


@pytest.mark.parametrize("material, name, first_electronic", [
    (13, "ALUMINUM", 55.80), (14, "SILICON", 47.32),
])
def test_retained_official_responses_preserve_whole_particle_axis_and_tokens(
    material: int, name: str, first_electronic: float,
) -> None:
    text = (_ROOT / f"research/nist_star_20261031/live_astar_{material:03d}.html").read_text(
        encoding="utf-8",
    )
    response = parse_astar_live(text, expected_material_id=material)
    assert response.material_id == material
    assert response.name == name
    assert response.row_count == 121
    assert response.rows[0].energy_mev_total == .001
    assert response.rows[-1].energy_mev_total == 1000.
    assert response.rows[0].stopping_electronic_mev_cm2_g == first_electronic
    assert response.rows[0].printed_tokens[0] == "1.000E-03"
    assert response.rows[-1].printed_tokens[0] == "1.000E+03"
    assert response.energy_coordinate == "whole_alpha_particle_kinetic_energy_MeV"
    assert not response.normative_export_allowed
    assert not response.physical_accuracy_verified
    assert not response.archive_grid_equivalence_verified
    assert tuple(row.energy_mev_total for row in response.rows) == CURRENT_DEFAULT_ENERGIES_MEV_TOTAL
    assert 22.5 not in CURRENT_DEFAULT_ENERGIES_MEV_TOTAL
    preceding = CURRENT_DEFAULT_ENERGIES_MEV_TOTAL.index(20.)
    assert (20., 25.) == CURRENT_DEFAULT_ENERGIES_MEV_TOTAL[preceding:preceding + 2]
    assert all(row.range_projected_g_cm2 < row.range_csda_g_cm2 for row in response.rows)


def test_total_stopping_is_preserved_not_replaced_by_rounded_component_sum(aluminum_html: str) -> None:
    row = parse_astar_live(aluminum_html, 13).rows[0]
    assert row.stopping_electronic_mev_cm2_g == 55.80
    assert row.stopping_nuclear_mev_cm2_g == 74.69
    assert row.stopping_total_mev_cm2_g == 130.5
    assert row.stopping_total_mev_cm2_g != (
        row.stopping_electronic_mev_cm2_g + row.stopping_nuclear_mev_cm2_g
    )
    assert row.range_csda_g_cm2 == 9.964e-6
    assert row.range_projected_g_cm2 == 2.529e-6
    assert row.detour_factor == .2538


@pytest.mark.parametrize("token, expected", [
    ("1.000E-03", Decimal("0.0000005")),
    ("5.580E+01", Decimal("0.005")),
    ("1.305E+02", Decimal("0.05")),
    ("1.000E+03", Decimal("0.5")),
    ("0.2538", Decimal("0.00005")),
    ("1.0000", Decimal("0.00005")),
])
def test_printing_half_unit_is_derived_from_token_precision(token: str, expected: Decimal) -> None:
    assert half_printed_unit_decimal(token) == expected
    assert half_printed_unit(token) == float(expected)


@pytest.mark.parametrize("token", [
    "nan", "NaN", "inf", "True", "1e-3", "1.000e-03", "1.0000E-03", "1.00E-03",
    "0.000E+00", "-1.000E+00", "1.000E+999", "0.25380", "0.25", "0.2538 ",
])
def test_malformed_printed_tokens_rejected(token: str) -> None:
    with pytest.raises(ValueError):
        half_printed_unit_decimal(token)


@pytest.mark.parametrize("old, new", [
    ("Stopping Power and Range Tables for Alpha Particles", "Stopping Tables for Protons"),
    ("ALPHA PARTICLE STOPPING POWER AND RANGE", "PROTON STOPPING POWER AND RANGE"),
    ("ALUMINUM", "SILICON"),
    ("kinetic energy, MeV", "kinetic energy, MeV/nucleon"),
    ("electronic stopping power, MeV cm2/g", "electronic stopping power, MeV cm2/mg"),
    ("nuclear stopping power, MeV cm2/g", "total stopping power, MeV cm2/g"),
    ("total stopping power, MeV cm2/g", "electronic stopping power, MeV cm2/g"),
    ("RANGE(c) = csda range, g/cm2", "RANGE(c) = projected range, g/cm2"),
    ("RANGE(p) = projected range, g/cm2", "RANGE(p) = csda range, g/cm2"),
    ("DETOUR = detour factor", "DETOUR = rate per second"),
    ("compos.pl?ap-text013", "compos.pl?ap-text014"),
    ("compos.pl?ap-text013", "https://untrusted.invalid/compos.pl?ap-text013"),
    (">compositional data</a>", ">different text</a>"),
    ("for this material.", "for a different material."),
])
def test_semantic_header_material_or_footer_changes_fail(
    aluminum_html: str, old: str, new: str,
) -> None:
    assert old in aluminum_html
    with pytest.raises(ValueError):
        parse_astar_live(aluminum_html.replace(old, new, 1), 13)


@pytest.mark.parametrize("column, value", [
    (0, "NaN"), (1, "True"), (1, "-1.000E+01"), (1, "0.000E+00"),
    (2, "7.469e+01"), (3, "inf"), (4, "9.964E-999"), (5, "2.529E-06 extra"),
    (6, "0.0000"), (6, "1.0001"), (6, "0.25380"),
])
def test_invalid_row_tokens_fail(aluminum_html: str, column: int, value: str) -> None:
    with pytest.raises(ValueError):
        parse_astar_live(_row_change(aluminum_html, 0, column, value), 13)


def test_total_stopping_error_below_one_percent_but_beyond_printing_is_rejected(
    aluminum_html: str,
) -> None:
    # 130.6 versus 55.80+74.69 has 0.11 discrepancy, beyond combined .06 half-units.
    with pytest.raises(ValueError, match="beyond printing"):
        parse_astar_live(_row_change(aluminum_html, 0, 3, "1.306E+02"), 13)


def test_projected_range_cannot_exceed_csda(aluminum_html: str) -> None:
    with pytest.raises(ValueError, match="projected range exceeds"):
        parse_astar_live(_row_change(aluminum_html, 0, 5, "1.100E-05"), 13)


def test_projected_range_detour_product_uses_printing_intervals(aluminum_html: str) -> None:
    row = parse_astar_live(aluminum_html, 13).rows[0]
    assert row.range_projected_g_cm2 != row.range_csda_g_cm2 * row.detour_factor
    with pytest.raises(ValueError, match="printing-interval overlap"):
        parse_astar_live(_row_change(aluminum_html, 0, 6, "0.2500"), 13)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "trailing", "short_columns"])
def test_missing_duplicate_extra_or_short_rows_are_not_silently_ignored(
    aluminum_html: str, mutation: str,
) -> None:
    row = tuple(_ROW.finditer(aluminum_html))[10]
    if mutation == "missing":
        changed = aluminum_html[:row.start()] + aluminum_html[row.end():]
    elif mutation == "duplicate":
        changed = aluminum_html[:row.start()] + row.group() + "<br>" + aluminum_html[row.start():]
    elif mutation == "trailing":
        changed = aluminum_html.replace("Show <a", "unexpected trailing data<br>Show <a", 1)
    else:
        changed = aluminum_html[:row.start()] + " ".join(row.group().split()[:-1]) + (
            aluminum_html[row.end():]
        )
    with pytest.raises(ValueError):
        parse_astar_live(changed, 13)


@pytest.mark.parametrize("energy", ["1.400E-03", "1.000E-03", "2.000E-03"])
def test_energy_grid_changed_duplicate_or_out_of_order_fails(
    aluminum_html: str, energy: str,
) -> None:
    with pytest.raises(ValueError, match="121-node"):
        parse_astar_live(_row_change(aluminum_html, 1, 0, energy), 13)


def test_new_22_5_mev_point_cannot_masquerade_as_current_default(aluminum_html: str) -> None:
    with pytest.raises(ValueError, match="121-node"):
        parse_astar_live(_row_change(
            aluminum_html, CURRENT_DEFAULT_ENERGIES_MEV_TOTAL.index(25.), 0, "2.250E+01",
        ), 13)


def test_range_order_is_enforced_even_with_individually_consistent_rows(aluminum_html: str) -> None:
    response = parse_astar_live(aluminum_html, 13)
    first = response.rows[0]
    second = response.rows[1]
    duplicate_range = replace(
        second, range_csda_g_cm2=first.range_csda_g_cm2,
        range_projected_g_cm2=first.range_projected_g_cm2, detour_factor=first.detour_factor,
        printed_tokens=(*second.printed_tokens[:4], *first.printed_tokens[4:]),
    )
    with pytest.raises(ValueError, match="ranges must increase"):
        replace(response, rows=(first, duplicate_range, *response.rows[2:]))


@pytest.mark.parametrize("markup", [
    "<pre>extra</pre>", "<title>extra</title>",
])
def test_multiple_pre_or_titles_fail(aluminum_html: str, markup: str) -> None:
    with pytest.raises(ValueError):
        parse_astar_live(aluminum_html + markup, 13)


@pytest.mark.parametrize("tag", ["</pre>", "</title>"])
def test_unclosed_required_block_fails(aluminum_html: str, tag: str) -> None:
    with pytest.raises(ValueError):
        parse_astar_live(aluminum_html.replace(tag, "", 1), 13)


def test_active_content_outside_table_is_inert_not_data(aluminum_html: str) -> None:
    plain = parse_astar_live(aluminum_html, 13)
    injected = aluminum_html.replace(
        "</body>", "<script>throw new Error('must never run'); var x='<pre>wrong</pre>';</script></body>",
    )
    assert parse_astar_live(injected, 13) == plain


def test_markup_inside_numeric_table_is_not_accepted(aluminum_html: str) -> None:
    with pytest.raises(ValueError, match="Unexpected markup"):
        parse_astar_live(aluminum_html.replace("<pre>", "<pre><script>1</script>", 1), 13)


@pytest.mark.parametrize("material_id", [True, False, 13.0, 0, 15, -1])
def test_only_explicit_al_si_integer_material_ids_supported(
    aluminum_html: str, material_id: int,
) -> None:
    with pytest.raises(ValueError):
        parse_astar_live(aluminum_html, material_id)


@pytest.mark.parametrize("value", [True, nan, inf, -inf, -1., 0., 10**1000])
def test_direct_row_construction_cannot_bypass_numeric_guards(
    aluminum_html: str, value: float,
) -> None:
    row = parse_astar_live(aluminum_html, 13).rows[0]
    with pytest.raises(ValueError):
        replace(row, energy_mev_total=value)


def test_response_cannot_be_constructed_with_wrong_name_or_empty_grid(aluminum_html: str) -> None:
    response = parse_astar_live(aluminum_html, 13)
    with pytest.raises(ValueError):
        LiveAstarResponse(13, "SILICON", response.rows)
    with pytest.raises(ValueError):
        LiveAstarResponse(13, "ALUMINUM", ())
