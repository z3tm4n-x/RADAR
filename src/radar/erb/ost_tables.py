"""OST 134-1044-2007 Appendix A ERB table resources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from importlib import resources
from math import isfinite, log10
from typing import Literal, Mapping, Sequence, cast

from radar.core.types import Particle

OST_ERB_APPENDIX_A_DOCUMENT = "OST 134-1044-2007 Appendix A"
OST_ERB_RESOURCE_NAME = "erb/ost_appendix_a_tables.json"

OST_ERB_PROTON_ENERGIES_MEV: tuple[float, ...] = (
    0.2,
    0.6,
    1.25,
    3.0,
    5.0,
    12.5,
    30.0,
    50.0,
    125.0,
    300.0,
)

OST_ERB_ELECTRON_ENERGIES_MEV: tuple[float, ...] = (
    0.16,
    0.35,
    0.6,
    0.9,
    1.5,
    2.5,
    3.5,
    4.5,
    5.5,
    6.5,
)

OstErbTableName = Literal[
    "elec_min",
    "elec_max",
    "prot_min",
    "prot_max",
]

OstErbSolarState = Literal[
    "min",
    "max",
]


class OstErbInterpolationMode(StrEnum):
    """Interpolation policy for OST Appendix A tables."""

    STRICT = "strict"
    ENGINEERING = "engineering"


@dataclass(frozen=True)
class OstErbShell:
    """One L-shell slice of an OST ERB flux table."""

    l_shell: float
    b_over_b0: tuple[float, ...]
    flux: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isfinite(self.l_shell):
            msg = "OST ERB shell L value must be finite."
            raise ValueError(msg)

        if self.l_shell <= 0.0:
            msg = "OST ERB shell L value must be positive."
            raise ValueError(msg)

        if len(self.b_over_b0) < 1:
            msg = "OST ERB shell must contain at least one B/B0 value."
            raise ValueError(msg)

        if any(not isfinite(value) or value < 1.0 for value in self.b_over_b0):
            msg = "OST ERB B/B0 values must be finite and at least 1."
            raise ValueError(msg)

        if tuple(sorted(self.b_over_b0)) != self.b_over_b0:
            msg = "OST ERB B/B0 grid must be sorted."
            raise ValueError(msg)

        if len(set(self.b_over_b0)) != len(self.b_over_b0):
            msg = "OST ERB B/B0 grid values must be unique."
            raise ValueError(msg)

        if len(self.flux) != len(self.b_over_b0):
            msg = "OST ERB shell flux row count must match B/B0 grid length."
            raise ValueError(msg)

        if not self.flux:
            msg = "OST ERB shell flux table must not be empty."
            raise ValueError(msg)

        energy_count = len(self.flux[0])

        if energy_count < 1:
            msg = "OST ERB shell flux vectors must not be empty."
            raise ValueError(msg)

        for row in self.flux:
            if len(row) != energy_count:
                msg = "OST ERB shell flux vectors must have equal length."
                raise ValueError(msg)

            if any(not isfinite(value) or value < 0.0 for value in row):
                msg = "OST ERB flux values must be finite and non-negative."
                raise ValueError(msg)


@dataclass(frozen=True)
class OstErbFluxTable:
    """One OST Appendix A particle/state flux table."""

    name: OstErbTableName
    particle: Particle
    solar_state: OstErbSolarState
    energies_mev: tuple[float, ...]
    shells: tuple[OstErbShell, ...]

    def __post_init__(self) -> None:
        if self.particle not in (Particle.PROTON, Particle.ELECTRON):
            msg = "OST ERB table particle must be proton or electron."
            raise ValueError(msg)

        if len(self.energies_mev) < 1:
            msg = "OST ERB energy grid must not be empty."
            raise ValueError(msg)

        if any(not isfinite(value) or value <= 0.0 for value in self.energies_mev):
            msg = "OST ERB energy grid values must be finite and positive."
            raise ValueError(msg)

        if tuple(sorted(self.energies_mev)) != self.energies_mev:
            msg = "OST ERB energy grid must be sorted."
            raise ValueError(msg)

        if not self.shells:
            msg = "OST ERB flux table must contain at least one L shell."
            raise ValueError(msg)

        l_grid = tuple(shell.l_shell for shell in self.shells)

        if tuple(sorted(l_grid)) != l_grid:
            msg = "OST ERB L grid must be sorted."
            raise ValueError(msg)

        if len(set(l_grid)) != len(l_grid):
            msg = "OST ERB L grid values must be unique."
            raise ValueError(msg)

        for shell in self.shells:
            for row in shell.flux:
                if len(row) != len(self.energies_mev):
                    msg = "OST ERB shell flux width must match energy grid."
                    raise ValueError(msg)

    @property
    def l_grid(self) -> tuple[float, ...]:
        """Return the table L grid."""

        return tuple(shell.l_shell for shell in self.shells)

    def vector(
        self,
        *,
        l_shell: float,
        b_over_b0: float,
        mode: OstErbInterpolationMode = OstErbInterpolationMode.STRICT,
    ) -> tuple[float, ...]:
        """Return the interpolated differential flux vector."""

        if not isfinite(l_shell) or not isfinite(b_over_b0):
            return _zero_vector(len(self.energies_mev))

        if l_shell < self.l_grid[0] or l_shell > self.l_grid[-1]:
            return _zero_vector(len(self.energies_mev))

        l_indices = _pick3(self.l_grid, l_shell)
        selected_shells = tuple(self.shells[index] for index in l_indices)
        selected_l = tuple(shell.l_shell for shell in selected_shells)

        per_l: list[tuple[float, ...]] = []

        for shell in selected_shells:
            if b_over_b0 > shell.b_over_b0[-1] * (1.0 + 1.0e-12):
                per_l.append(_zero_vector(len(self.energies_mev)))
                continue

            bb0 = max(b_over_b0, shell.b_over_b0[0])
            b_indices = _pick3(shell.b_over_b0, bb0)
            selected_b = tuple(shell.b_over_b0[index] for index in b_indices)

            per_l.append(
                tuple(
                    _interp_positive(
                        x=bb0,
                        xs=selected_b,
                        ys=tuple(shell.flux[index][energy_index] for index in b_indices),
                        mode=mode,
                    )
                    for energy_index in range(len(self.energies_mev))
                )
            )

        return tuple(
            _interp_positive(
                x=l_shell,
                xs=selected_l,
                ys=tuple(values[energy_index] for values in per_l),
                mode=mode,
            )
            for energy_index in range(len(self.energies_mev))
        )


@dataclass(frozen=True)
class OstErbTableSet:
    """All OST Appendix A ERB flux tables."""

    electron_min: OstErbFluxTable
    electron_max: OstErbFluxTable
    proton_min: OstErbFluxTable
    proton_max: OstErbFluxTable

    def by_name(self, name: OstErbTableName) -> OstErbFluxTable:
        """Return a table by its resource name."""

        if name == "elec_min":
            return self.electron_min

        if name == "elec_max":
            return self.electron_max

        if name == "prot_min":
            return self.proton_min

        if name == "prot_max":
            return self.proton_max

        msg = f"Unknown OST ERB table name: {name}"
        raise ValueError(msg)

    @property
    def tables(self) -> tuple[OstErbFluxTable, ...]:
        """Return all tables in deterministic order."""

        return (
            self.electron_min,
            self.electron_max,
            self.proton_min,
            self.proton_max,
        )


def _zero_vector(size: int) -> tuple[float, ...]:
    return tuple(0.0 for _ in range(size))


def _lagrange3(
    *,
    x: float,
    xs: tuple[float, float, float],
    ys: tuple[float, float, float],
) -> float:
    x1, x2, x3 = xs
    y1, y2, y3 = ys

    return (
        ((x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3))) * y1
        + ((x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3))) * y2
        + ((x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2))) * y3
    )


def _pick3(grid: Sequence[float], x: float) -> tuple[int, ...]:
    size = len(grid)

    if size <= 3:
        return tuple(range(size))

    if x <= grid[1]:
        return (0, 1, 2)

    if x >= grid[-2]:
        return (size - 3, size - 2, size - 1)

    for index, value in enumerate(grid):
        if value >= x:
            return (index - 1, index, index + 1)

    return (size - 3, size - 2, size - 1)


def _interp_poly(
    *,
    x: float,
    xs: tuple[float, ...],
    ys: tuple[float, ...],
) -> float:
    if len(xs) >= 3:
        lagrange_value: float = _lagrange3(
            x=x,
            xs=(xs[0], xs[1], xs[2]),
            ys=(ys[0], ys[1], ys[2]),
        )
        return lagrange_value

    if len(xs) == 2:
        linear_value: float = ys[0] + (ys[1] - ys[0]) * (x - xs[0]) / (xs[1] - xs[0])
        return linear_value

    if len(xs) == 1:
        single_value: float = ys[0]
        return single_value

    return 0.0


def _interp_positive(
    *,
    x: float,
    xs: tuple[float, ...],
    ys: tuple[float, ...],
    mode: OstErbInterpolationMode,
) -> float:
    if not xs:
        return 0.0

    if all(y_value > 0.0 for y_value in ys):
        log_values = tuple(log10(y_value) for y_value in ys)
        positive_value: float = 10.0 ** _interp_poly(
            x=x,
            xs=xs,
            ys=log_values,
        )

        if mode is OstErbInterpolationMode.ENGINEERING:
            positive_value = min(max(positive_value, min(ys)), max(ys))

        return float(max(positive_value, 0.0))

    polynomial_value: float = _interp_poly(x=x, xs=xs, ys=ys)

    if mode is OstErbInterpolationMode.ENGINEERING:
        polynomial_value = min(max(polynomial_value, min(ys)), max(ys))

    return float(max(polynomial_value, 0.0))


def _resource_data() -> Mapping[str, object]:
    resource = resources.files("radar.data.normative").joinpath(OST_ERB_RESOURCE_NAME)
    data = json.loads(resource.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        msg = "OST ERB table resource root must be a JSON object."
        raise ValueError(msg)

    return cast("Mapping[str, object]", data)


def _as_sequence(value: object, *, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        msg = f"{context} must be a JSON array."
        raise ValueError(msg)

    return value


def _energy_from_header(value: object) -> float:
    token = str(value).strip()

    if not token.startswith("E="):
        msg = f"OST ERB energy header value is malformed: {token!r}."
        raise ValueError(msg)

    return float(token[2:].replace(",", "."))


def _validate_header_energies(
    *,
    header: Sequence[object],
    energies_mev: tuple[float, ...],
) -> None:
    if len(header) != len(energies_mev) + 2:
        msg = "OST ERB table header width does not match energy grid."
        raise ValueError(msg)

    parsed = tuple(_energy_from_header(value) for value in header[2:])

    for expected, actual in zip(energies_mev, parsed, strict=True):
        if abs(expected - actual) > 1.0e-12:
            msg = "OST ERB table header energy grid does not match constants."
            raise ValueError(msg)


def _json_float(value: object, *, context: str) -> float:
    if isinstance(value, int | float | str):
        return float(value)

    msg = f"{context} must be numeric."
    raise ValueError(msg)


def _float_row(value: object, *, expected_width: int) -> tuple[float, ...]:
    row = _as_sequence(value, context="OST ERB table row")

    if len(row) != expected_width:
        msg = "OST ERB table row width does not match header."
        raise ValueError(msg)

    return tuple(
        _json_float(item, context="OST ERB table row value")
        for item in row
    )


def _table_from_raw(
    *,
    name: OstErbTableName,
    raw_table: Mapping[str, object],
    particle: Particle,
    solar_state: OstErbSolarState,
    energies_mev: tuple[float, ...],
) -> OstErbFluxTable:
    header = _as_sequence(raw_table.get("header"), context=f"{name} header")
    _validate_header_energies(header=header, energies_mev=energies_mev)

    rows = tuple(
        _float_row(value, expected_width=len(header))
        for value in _as_sequence(raw_table.get("rows"), context=f"{name} rows")
    )

    grouped: dict[float, list[tuple[float, tuple[float, ...]]]] = {}

    for row in rows:
        l_shell = row[0]
        b_over_b0 = row[1]
        flux = row[2:]

        grouped.setdefault(l_shell, []).append((b_over_b0, flux))

    shells: list[OstErbShell] = []

    for l_shell in sorted(grouped):
        entries = sorted(grouped[l_shell], key=lambda item: item[0])
        shells.append(
            OstErbShell(
                l_shell=l_shell,
                b_over_b0=tuple(entry[0] for entry in entries),
                flux=tuple(entry[1] for entry in entries),
            )
        )

    return OstErbFluxTable(
        name=name,
        particle=particle,
        solar_state=solar_state,
        energies_mev=energies_mev,
        shells=tuple(shells),
    )


def _raw_tables(data: Mapping[str, object]) -> Mapping[str, object]:
    tables = data.get("tables")

    if not isinstance(tables, dict):
        msg = "OST ERB resource must contain a 'tables' object."
        raise ValueError(msg)

    return cast("Mapping[str, object]", tables)


def _raw_table(
    *,
    tables: Mapping[str, object],
    name: OstErbTableName,
) -> Mapping[str, object]:
    value = tables.get(name)

    if not isinstance(value, dict):
        msg = f"OST ERB resource table {name!r} must be a JSON object."
        raise ValueError(msg)

    return cast("Mapping[str, object]", value)


@lru_cache(maxsize=1)
def load_ost_erb_appendix_a_tables() -> OstErbTableSet:
    """Load OST 134-1044-2007 Appendix A ERB tables."""

    tables = _raw_tables(_resource_data())

    return OstErbTableSet(
        electron_min=_table_from_raw(
            name="elec_min",
            raw_table=_raw_table(tables=tables, name="elec_min"),
            particle=Particle.ELECTRON,
            solar_state="min",
            energies_mev=OST_ERB_ELECTRON_ENERGIES_MEV,
        ),
        electron_max=_table_from_raw(
            name="elec_max",
            raw_table=_raw_table(tables=tables, name="elec_max"),
            particle=Particle.ELECTRON,
            solar_state="max",
            energies_mev=OST_ERB_ELECTRON_ENERGIES_MEV,
        ),
        proton_min=_table_from_raw(
            name="prot_min",
            raw_table=_raw_table(tables=tables, name="prot_min"),
            particle=Particle.PROTON,
            solar_state="min",
            energies_mev=OST_ERB_PROTON_ENERGIES_MEV,
        ),
        proton_max=_table_from_raw(
            name="prot_max",
            raw_table=_raw_table(tables=tables, name="prot_max"),
            particle=Particle.PROTON,
            solar_state="max",
            energies_mev=OST_ERB_PROTON_ENERGIES_MEV,
        ),
    )


__all__ = [
    "OST_ERB_APPENDIX_A_DOCUMENT",
    "OST_ERB_ELECTRON_ENERGIES_MEV",
    "OST_ERB_PROTON_ENERGIES_MEV",
    "OST_ERB_RESOURCE_NAME",
    "OstErbFluxTable",
    "OstErbInterpolationMode",
    "OstErbShell",
    "OstErbTableName",
    "OstErbTableSet",
    "load_ost_erb_appendix_a_tables",
]
