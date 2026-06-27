"""IGRF coefficient parsing for ERB magnetic-field calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from radar.erb.constants import DEFAULT_IGRF_EPOCH


@dataclass(frozen=True)
class IgrfCoefficients:
    """Schmidt semi-normalised IGRF coefficients for one epoch."""

    epoch: float
    g: dict[tuple[int, int], float]
    h: dict[tuple[int, int], float]
    nmax: int

    def __post_init__(self) -> None:
        if not isfinite(self.epoch):
            msg = "IGRF epoch must be finite."
            raise ValueError(msg)

        if self.nmax < 1:
            msg = "IGRF maximum degree must be positive."
            raise ValueError(msg)

        if not self.g:
            msg = "IGRF g coefficient table must not be empty."
            raise ValueError(msg)

        if any(n < 1 or m < 0 or m > n for n, m in (*self.g, *self.h)):
            msg = "IGRF coefficient indices must satisfy n>=1 and 0<=m<=n."
            raise ValueError(msg)

    def g_nm(self, n: int, m: int) -> float:
        """Return g(n,m), or zero when the coefficient is absent."""

        return self.g.get((n, m), 0.0)

    def h_nm(self, n: int, m: int) -> float:
        """Return h(n,m), or zero when the coefficient is absent."""

        return self.h.get((n, m), 0.0)


def _interpolate_epoch(
    *,
    years: tuple[float, ...],
    values: tuple[float, ...],
    secular_variation: float,
    epoch: float,
) -> float:
    if epoch <= years[0]:
        return values[0]

    if epoch >= years[-1]:
        return values[-1] + secular_variation * (epoch - years[-1])

    for left_index, right_index in zip(
        range(len(years) - 1),
        range(1, len(years)),
        strict=True,
    ):
        left_year = years[left_index]
        right_year = years[right_index]

        if left_year <= epoch <= right_year:
            fraction = (epoch - left_year) / (right_year - left_year)
            return values[left_index] + fraction * (
                values[right_index] - values[left_index]
            )

    return values[-1]


def parse_igrf_coefficients(
    text: str,
    *,
    epoch: float = DEFAULT_IGRF_EPOCH,
) -> IgrfCoefficients:
    """Parse an IGRF coefficient text file."""

    if not isfinite(epoch):
        msg = "IGRF parse epoch must be finite."
        raise ValueError(msg)

    years: tuple[float, ...] | None = None
    g: dict[tuple[int, int], float] = {}
    h: dict[tuple[int, int], float] = {}
    nmax = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if parts[0] == "g/h":
            if len(parts) < 5:
                msg = "IGRF year header is malformed."
                raise ValueError(msg)

            years = tuple(float(value) for value in parts[3:-1])
            continue

        if parts[0] not in ("g", "h"):
            continue

        if years is None:
            msg = "IGRF coefficient row encountered before year header."
            raise ValueError(msg)

        n = int(parts[1])
        m = int(parts[2])

        if len(parts) < 3 + len(years):
            msg = "IGRF coefficient row has too few epoch values."
            raise ValueError(msg)

        values = tuple(float(value) for value in parts[3 : 3 + len(years)])
        secular_variation = (
            float(parts[3 + len(years)])
            if len(parts) > 3 + len(years)
            else 0.0
        )
        coefficient = _interpolate_epoch(
            years=years,
            values=values,
            secular_variation=secular_variation,
            epoch=epoch,
        )

        if parts[0] == "g":
            g[(n, m)] = coefficient
        else:
            h[(n, m)] = coefficient

        nmax = max(nmax, n)

    if years is None:
        msg = "IGRF years header not found."
        raise ValueError(msg)

    return IgrfCoefficients(
        epoch=epoch,
        g=g,
        h=h,
        nmax=nmax,
    )


__all__ = [
    "IgrfCoefficients",
    "parse_igrf_coefficients",
]
