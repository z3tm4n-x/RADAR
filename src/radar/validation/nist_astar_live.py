"""Parse inert retained current ASTAR default-response HTML for Al and Si.

This reader never fetches or executes HTML. Its contract is the retained online
response, not STAR 1.2 archive data or the historical Fortran output format.
The observed default grid has 121 nodes: it does not include 22.5 MeV. That
absence must not be filled by interpolation or treated as a zero datum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from html.parser import HTMLParser
from math import isfinite
from numbers import Real
import re


_TITLE = "Stopping Power and Range Tables for Alpha Particles"
_NAMES = {13: "ALUMINUM", 14: "SILICON"}
_SCI = re.compile(r"[1-9]\.[0-9]{3}E[+-][0-9]{2}")
_DETOUR = re.compile(r"[01]\.[0-9]{4}")
_HEADERS = (
    "T = alpha particle (helium ion) kinetic energy, MeV",
    "STOP(e) = electronic stopping power, MeV cm2/g",
    "STOP(n) = nuclear stopping power, MeV cm2/g",
    "STOP(t) = total stopping power, MeV cm2/g",
    "RANGE(c) = csda range, g/cm2",
    "RANGE(p) = projected range, g/cm2",
    "DETOUR = detour factor",
    "T STOP(e) STOP(n) STOP(t) RANGE(c) RANGE(p) DETOUR",
)

# Explicit observed online default roster; deliberately independent of the
# archive reader and its 122-node ENG.ALF table. Energy is whole-alpha MeV.
CURRENT_DEFAULT_ENERGIES_MEV_TOTAL = tuple(float(value) for value in """
0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.006 0.007 0.008 0.009
0.01 0.0125 0.015 0.0175 0.02 0.0225 0.025 0.0275 0.03 0.035 0.04 0.045
0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09 0.095
0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.35 0.4 0.45
0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95
1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5
10 12.5 15 17.5 20 25 27.5 30 35 40 45 50 55 60 65 70 75 80 85 90 95
100 125 150 175 200 225 250 275 300 350 400 450 500 550 600 650 700 750 800 850 900 950
1000
""".split())


def half_printed_unit_decimal(token: str) -> Decimal:
    """Half of the final printed unit, not a physical uncertainty estimate."""
    if not isinstance(token, str):
        raise ValueError("Printed ASTAR token must be text.")
    if _SCI.fullmatch(token):
        return Decimal(5).scaleb(int(token[6:]) - 4)
    if _DETOUR.fullmatch(token):
        return Decimal("0.00005")
    raise ValueError("Expected a four-significant-digit E token or four-decimal detour token.")


def half_printed_unit(token: str) -> float:
    return float(half_printed_unit_decimal(token))


@dataclass(frozen=True)
class LiveAstarRow:
    """All seven printed columns are kept separately, with their exact tokens."""

    energy_mev_total: float
    stopping_electronic_mev_cm2_g: float
    stopping_nuclear_mev_cm2_g: float
    stopping_total_mev_cm2_g: float
    range_csda_g_cm2: float
    range_projected_g_cm2: float
    detour_factor: float
    printed_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        tokens = self.printed_tokens
        if (not isinstance(tokens, tuple) or len(tokens) != 7
                or any(not isinstance(token, str) for token in tokens)
                or any(_SCI.fullmatch(token) is None for token in tokens[:6])
                or _DETOUR.fullmatch(tokens[6]) is None):
            raise ValueError("Each ASTAR row requires six 4-digit E tokens and one detour token.")
        values = (
            self.energy_mev_total, self.stopping_electronic_mev_cm2_g,
            self.stopping_nuclear_mev_cm2_g, self.stopping_total_mev_cm2_g,
            self.range_csda_g_cm2, self.range_projected_g_cm2, self.detour_factor,
        )
        for value, token in zip(values, tokens, strict=True):
            try:
                invalid = (isinstance(value, bool) or not isinstance(value, Real)
                           or not isfinite(value) or value <= 0 or value != float(token))
            except OverflowError:
                invalid = True
            if invalid:
                raise ValueError("ASTAR row values must be finite positive numbers matching tokens.")
        if self.detour_factor > 1:
            raise ValueError("ASTAR detour factor cannot exceed one.")
        decimal_values = tuple(Decimal(token) for token in tokens)
        half_units = tuple(half_printed_unit_decimal(token) for token in tokens)
        if abs(decimal_values[3] - decimal_values[1] - decimal_values[2]) > sum(
            half_units[1:4], Decimal(0),
        ):
            raise ValueError("ASTAR total stopping differs from electronic+nuclear beyond printing.")
        csda, projected, detour = decimal_values[4:]
        dc, dp, dd = half_units[4:]
        if projected - csda > dc + dp:
            raise ValueError("ASTAR projected range exceeds CSDA range beyond printing.")
        # Test overlap of the printed intervals, rather than equating rounded
        # projected range with the product of two independently rounded values.
        if projected + dp < (csda - dc) * (detour - dd) or (
            projected - dp > (csda + dc) * (detour + dd)
        ):
            raise ValueError("ASTAR projected/CSDA/detour relation fails printing-interval overlap.")


@dataclass(frozen=True)
class LiveAstarResponse:
    material_id: int
    name: str
    rows: tuple[LiveAstarRow, ...]
    normative_export_allowed: bool = field(default=False, init=False)
    physical_accuracy_verified: bool = field(default=False, init=False)
    archive_grid_equivalence_verified: bool = field(default=False, init=False)
    energy_coordinate: str = field(default="whole_alpha_particle_kinetic_energy_MeV", init=False)

    def __post_init__(self) -> None:
        if type(self.material_id) is not int or self.material_id not in _NAMES:
            raise ValueError("Live ASTAR parser supports only material IDs 13 and 14.")
        if self.name != _NAMES[self.material_id]:
            raise ValueError("ASTAR response material name does not match its ID.")
        if (not isinstance(self.rows, tuple) or any(not isinstance(row, LiveAstarRow)
                                                  for row in self.rows)):
            raise ValueError("ASTAR response rows must be an immutable tuple of LiveAstarRow.")
        if tuple(row.energy_mev_total for row in self.rows) != CURRENT_DEFAULT_ENERGIES_MEV_TOTAL:
            raise ValueError("Expected the exact observed 121-node ASTAR online default grid.")
        for left, right in zip(self.rows, self.rows[1:]):
            if (left.range_csda_g_cm2 >= right.range_csda_g_cm2
                    or left.range_projected_g_cm2 >= right.range_projected_g_cm2):
                raise ValueError("ASTAR CSDA and projected ranges must increase strictly.")

    @property
    def row_count(self) -> int:
        return len(self.rows)


class _AstarHTML(HTMLParser):
    """Collect a single inert title/pre block; active content is never executed."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_count = 0
        self.pre_count = 0
        self.in_title = False
        self.in_pre = False
        self.in_anchor = False
        self.title_parts: list[str] = []
        self.pre_parts: list[str] = []
        self.anchor_parts: list[str] = []
        self.anchor_hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            if self.in_pre or self.in_title or self.title_count:
                raise ValueError("Exactly one non-nested ASTAR title is required.")
            self.title_count += 1
            self.in_title = True
        elif tag == "pre":
            if self.in_title or self.in_pre or self.pre_count:
                raise ValueError("Exactly one non-nested ASTAR pre block is required.")
            self.pre_count += 1
            self.in_pre = True
        elif self.in_title:
            raise ValueError("Unexpected markup inside ASTAR title.")
        elif self.in_pre:
            if tag in ("br", "p") and not attrs and not self.in_anchor:
                self.pre_parts.append("\n")
            elif tag == "a" and not self.in_anchor and len(attrs) == 1 and attrs[0][0] == "href":
                href = attrs[0][1]
                if href is None:
                    raise ValueError("Missing compositional link destination.")
                self.anchor_hrefs.append(href)
                self.in_anchor = True
            else:
                raise ValueError("Unexpected markup inside ASTAR numeric pre block.")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            if not self.in_title:
                raise ValueError("Unmatched ASTAR title end tag.")
            self.in_title = False
        elif tag == "pre":
            if not self.in_pre or self.in_anchor:
                raise ValueError("Unmatched ASTAR pre block or unclosed composition link.")
            self.in_pre = False
        elif self.in_pre:
            if tag == "a" and self.in_anchor:
                self.in_anchor = False
            elif tag not in ("br", "p"):
                raise ValueError("Unexpected closing markup inside ASTAR pre block.")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        elif self.in_pre:
            self.pre_parts.append(data)
            if self.in_anchor:
                self.anchor_parts.append(data)


def parse_astar_live(text: str, expected_material_id: int) -> LiveAstarResponse:
    """Validate a retained default Al/Si HTML response without fetching anything."""
    if not isinstance(text, str):
        raise ValueError("Retained ASTAR HTML must be provided as text.")
    if type(expected_material_id) is not int or expected_material_id not in _NAMES:
        raise ValueError("Expected ASTAR material ID must be integer 13 or 14, not bool.")
    parser = _AstarHTML()
    parser.feed(text)
    parser.close()
    if (parser.title_count != 1 or parser.pre_count != 1
            or parser.in_pre or parser.in_title or parser.in_anchor):
        raise ValueError("Expected exactly one closed ASTAR title and numeric pre block.")
    title = " ".join("".join(parser.title_parts).split())
    if title != _TITLE:
        raise ValueError("Response title is not the ASTAR alpha-particle title.")
    if (parser.anchor_hrefs != [f"compos.pl?ap-text{expected_material_id:03d}"]
            or "".join(parser.anchor_parts) != "compositional data"):
        raise ValueError("ASTAR composition link does not uniquely match the expected material ID.")
    lines = tuple(" ".join(line.split()) for line in "".join(parser.pre_parts).splitlines()
                  if line.strip())
    header = ("ALPHA PARTICLE STOPPING POWER AND RANGE", _NAMES[expected_material_id], *_HEADERS)
    if lines[:len(header)] != header:
        raise ValueError("ASTAR particle/material/column semantics or units do not match.")
    if len(lines) != len(header) + 121 + 1 or lines[-1] != (
        "Show compositional data for this material."
    ):
        raise ValueError("Expected exactly 121 numeric rows followed only by the composition footer.")
    rows = []
    for line in lines[len(header):-1]:
        tokens = tuple(line.split())
        if (len(tokens) != 7 or any(_SCI.fullmatch(token) is None for token in tokens[:6])
                or _DETOUR.fullmatch(tokens[6]) is None):
            raise ValueError("Malformed ASTAR seven-column printed numeric row.")
        values = tuple(float(token) for token in tokens)
        rows.append(LiveAstarRow(
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], tokens,
        ))
    return LiveAstarResponse(expected_material_id, _NAMES[expected_material_id], tuple(rows))
