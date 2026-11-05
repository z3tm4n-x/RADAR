"""TENDL/ENDF-6 parser for compact RADAR nuclear-data CSV conversion."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Iterable, Sequence

ENDT_PROTON_ZAP = 1001
EV_PER_MEV = 1.0e6


class EndfInterpolationLaw(StrEnum):
    """Interpolation laws retained by compact ENDF-derived resources."""

    HISTOGRAM_LEFT = "histogram_left"
    LINEAR_LINEAR = "linear_linear"


@dataclass(frozen=True)
class TendlMf6InterpolationMetadata:
    """Interpolation contract for one compact MF=6 proton distribution.

    The outgoing-energy ordinates in the bundled TENDL tables are ENDF
    histogram values: ``pdf[i]`` applies on ``[E[i], E[i + 1])``.  Yield and
    incident-energy PDF interpolation are linear between their own,
    independent incident-energy grids.  No extrapolation is permitted.
    """

    yield_incident_energy: EndfInterpolationLaw = EndfInterpolationLaw.LINEAR_LINEAR
    pdf_incident_energy: EndfInterpolationLaw = EndfInterpolationLaw.LINEAR_LINEAR
    pdf_outgoing_energy: EndfInterpolationLaw = EndfInterpolationLaw.HISTOGRAM_LEFT


DEFAULT_TENDL_MF6_INTERPOLATION = TendlMf6InterpolationMetadata()


@dataclass(frozen=True)
class TendlIncidentEnergySupport:
    """Evaluated incident-proton energy domain of one TENDL operator.

    Reaction thresholds inside this domain are represented by zero-valued
    cross sections/yields.  Energies outside the evaluated domain are not
    equivalent to a physical zero and must never be extrapolated silently.
    """

    minimum_mev: float
    maximum_mev: float
    dataset_ref: str

    def __post_init__(self) -> None:
        if not isfinite(self.minimum_mev) or not isfinite(self.maximum_mev):
            msg = "TENDL incident-energy support bounds must be finite."
            raise ValueError(msg)
        if self.minimum_mev < 0.0:
            msg = "TENDL incident-energy support minimum must be non-negative."
            raise ValueError(msg)
        if self.maximum_mev <= self.minimum_mev:
            msg = "TENDL incident-energy support maximum must exceed its minimum."
            raise ValueError(msg)
        if not self.dataset_ref.strip():
            msg = "TENDL incident-energy support dataset reference must not be empty."
            raise ValueError(msg)

    def contains(self, energy_mev: float) -> bool:
        """Return whether an energy is inside the evaluated domain."""

        return (
            isfinite(energy_mev)
            and self.minimum_mev <= energy_mev <= self.maximum_mev
        )

    def require_contains(self, energy_mev: float, *, operator: str) -> None:
        """Reject scalar evaluation outside support instead of extrapolating."""

        if self.contains(energy_mev):
            return
        msg = (
            f"{operator} incident energy {energy_mev:g} MeV is outside TENDL "
            f"support [{self.minimum_mev:g}, {self.maximum_mev:g}] MeV."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class TendlIncidentEnergyCoverage:
    """Typed coverage result for energies carrying a nonzero contribution."""

    support: TendlIncidentEnergySupport
    relevant_minimum_mev: float | None
    relevant_maximum_mev: float | None
    unsupported_minimum_mev: float | None
    unsupported_maximum_mev: float | None

    @property
    def complete(self) -> bool:
        """Return whether every relevant incident energy is evaluated."""

        return self.unsupported_minimum_mev is None

    def incomplete_reason(self, *, operator: str) -> str | None:
        """Return a stable release-blocking reason, if coverage is incomplete."""

        if self.complete:
            return None
        return (
            f"{operator} is incomplete: nonzero transported proton contribution exists "
            f"outside TENDL incident-energy support "
            f"[{self.support.minimum_mev:g}, {self.support.maximum_mev:g}] MeV"
        )


def assess_tendl_incident_energy_coverage(
    *,
    relevant_energies_mev: Iterable[float],
    support: TendlIncidentEnergySupport,
) -> TendlIncidentEnergyCoverage:
    """Classify only energies already known to carry a nonzero contribution."""

    relevant = tuple(relevant_energies_mev)
    if any(not isfinite(energy) or energy < 0.0 for energy in relevant):
        msg = "Relevant TENDL incident energies must be finite and non-negative."
        raise ValueError(msg)
    unsupported = tuple(energy for energy in relevant if not support.contains(energy))
    return TendlIncidentEnergyCoverage(
        support=support,
        relevant_minimum_mev=min(relevant, default=None),
        relevant_maximum_mev=max(relevant, default=None),
        unsupported_minimum_mev=min(unsupported, default=None),
        unsupported_maximum_mev=max(unsupported, default=None),
    )


@dataclass(frozen=True)
class TendlMf3CrossSectionPoint:
    """One MF=3 cross-section point."""

    mt: int
    energy_mev: float
    sigma_barn: float


@dataclass(frozen=True)
class TendlMf6ProtonYieldPoint:
    """One emitted-proton MF=6 product yield point."""

    mt: int
    product_index: int
    incident_energy_mev: float
    yield_value: float
    interpolation_law: EndfInterpolationLaw = EndfInterpolationLaw.LINEAR_LINEAR


@dataclass(frozen=True)
class TendlMf6ProtonPdfPoint:
    """One emitted-proton MF=6 LAW=1 outgoing-energy density point."""

    mt: int
    product_index: int
    incident_energy_mev: float
    emitted_energy_mev: float
    pdf_per_mev: float
    incident_energy_interpolation_law: EndfInterpolationLaw = (
        EndfInterpolationLaw.LINEAR_LINEAR
    )
    outgoing_energy_interpolation_law: EndfInterpolationLaw = (
        EndfInterpolationLaw.HISTOGRAM_LEFT
    )


@dataclass(frozen=True)
class TendlCsvConversionOutput:
    """Paths written by TENDL-to-CSV conversion."""

    mf3_xs: Path
    mf6_proton_yield: Path
    mf6_proton_pdf: Path


@dataclass(frozen=True)
class _EndfTab1:
    c1: float
    c2: float
    l1: int
    l2: int
    nr: int
    np: int
    interpolation_regions: tuple[tuple[int, int], ...]
    x: tuple[float, ...]
    y: tuple[float, ...]


@dataclass(frozen=True)
class _EndfTab2:
    c1: float
    c2: float
    l1: int
    l2: int
    nr: int
    nz: int
    interpolation_regions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _EndfList:
    c1: float
    c2: float
    l1: int
    l2: int
    nw: int
    n2: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class _ParsedMf6Product:
    mt: int
    product_index: int
    zap: int
    law: int
    yield_points: tuple[TendlMf6ProtonYieldPoint, ...]
    pdf_points: tuple[TendlMf6ProtonPdfPoint, ...]


def _endf_float(text: str) -> float:
    stripped = text.strip()

    if not stripped:
        return 0.0

    if "e" not in stripped.lower():
        for index in range(len(stripped) - 1, 0, -1):
            if stripped[index] in "+-":
                return float(stripped[:index] + "E" + stripped[index:])

    return float(stripped)


def _endf_int(text: str) -> int:
    stripped = text.strip()

    if not stripped:
        return 0

    return int(stripped)


def _endf_fields(line: str) -> tuple[str, str, str, str, str, str]:
    data = line[:66].ljust(66)
    fields = tuple(data[index : index + 11] for index in range(0, 66, 11))

    if len(fields) != 6:
        msg = "ENDF line must contain six 11-character data fields."
        raise ValueError(msg)

    return fields


def _line_identity(line: str) -> tuple[int, int, int]:
    if len(line) < 75:
        msg = "ENDF line is too short to contain MAT/MF/MT identifiers."
        raise ValueError(msg)

    return int(line[66:70]), int(line[70:72]), int(line[72:75])


def _parse_cont(line: str) -> tuple[float, float, int, int, int, int]:
    fields = _endf_fields(line)

    return (
        _endf_float(fields[0]),
        _endf_float(fields[1]),
        _endf_int(fields[2]),
        _endf_int(fields[3]),
        _endf_int(fields[4]),
        _endf_int(fields[5]),
    )


def _parse_values(
    lines: Sequence[str],
    index: int,
    count: int,
) -> tuple[tuple[float, ...], int]:
    values: list[float] = []
    current_index = index

    while len(values) < count:
        if current_index >= len(lines):
            msg = "Unexpected end of ENDF section while parsing numeric values."
            raise ValueError(msg)

        values.extend(_endf_float(field) for field in _endf_fields(lines[current_index]))
        current_index += 1

    return tuple(values[:count]), current_index


def _parse_interpolation_regions(
    values: tuple[float, ...],
    *,
    point_count: int,
    context: str,
) -> tuple[tuple[int, int], ...]:
    if len(values) % 2 != 0:
        msg = f"{context} interpolation metadata must contain NBT/INT pairs."
        raise ValueError(msg)

    regions: list[tuple[int, int]] = []
    previous_boundary = 0
    for raw_boundary, raw_law in zip(values[0::2], values[1::2], strict=True):
        boundary = int(raw_boundary)
        law = int(raw_law)
        if float(boundary) != raw_boundary or float(law) != raw_law:
            msg = f"{context} interpolation NBT/INT values must be integers."
            raise ValueError(msg)
        if boundary <= previous_boundary or boundary > point_count:
            msg = f"{context} interpolation boundaries are invalid."
            raise ValueError(msg)
        regions.append((boundary, law))
        previous_boundary = boundary

    if not regions or regions[-1][0] != point_count:
        msg = f"{context} interpolation regions must cover all {point_count} points."
        raise ValueError(msg)

    return tuple(regions)


def _require_endf_interpolation_law(
    regions: tuple[tuple[int, int], ...],
    *,
    supported_int: int,
    supported_name: str,
    context: str,
) -> None:
    unsupported = tuple(
        (boundary, law) for boundary, law in regions if law != supported_int
    )
    if unsupported:
        msg = (
            f"{context} uses unsupported ENDF interpolation regions {unsupported!r}; "
            f"only {supported_name} (INT={supported_int}) is implemented."
        )
        raise ValueError(msg)


def _section_lines(
    path: str | Path,
    *,
    mf: int,
    mt: int,
) -> tuple[str, ...]:
    rows: list[str] = []

    with Path(path).open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")

            try:
                _, current_mf, current_mt = _line_identity(line)
            except ValueError:
                continue

            if current_mf == mf and current_mt == mt:
                rows.append(line)

    return tuple(rows)


def _discover_mts(
    path: str | Path,
    *,
    mf: int,
) -> tuple[int, ...]:
    mts: list[int] = []
    seen: set[int] = set()

    with Path(path).open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")

            try:
                _, current_mf, current_mt = _line_identity(line)
            except ValueError:
                continue

            if current_mf == mf and current_mt not in seen:
                seen.add(current_mt)
                mts.append(current_mt)

    return tuple(mts)


def _parse_tab1(
    lines: Sequence[str],
    index: int,
) -> tuple[_EndfTab1, int]:
    c1, c2, l1, l2, nr, np_count = _parse_cont(lines[index])
    index += 1

    interpolation_values, index = _parse_values(lines, index, 2 * nr)
    interpolation_regions = _parse_interpolation_regions(
        interpolation_values,
        point_count=np_count,
        context="ENDF TAB1",
    )
    values, index = _parse_values(lines, index, 2 * np_count)

    return (
        _EndfTab1(
            c1=c1,
            c2=c2,
            l1=l1,
            l2=l2,
            nr=nr,
            np=np_count,
            interpolation_regions=interpolation_regions,
            x=values[0::2],
            y=values[1::2],
        ),
        index,
    )


def _parse_tab2(
    lines: Sequence[str],
    index: int,
) -> tuple[_EndfTab2, int]:
    c1, c2, l1, l2, nr, nz = _parse_cont(lines[index])
    index += 1
    interpolation_values, index = _parse_values(lines, index, 2 * nr)
    interpolation_regions = _parse_interpolation_regions(
        interpolation_values,
        point_count=nz,
        context="ENDF TAB2",
    )

    return (
        _EndfTab2(
            c1=c1,
            c2=c2,
            l1=l1,
            l2=l2,
            nr=nr,
            nz=nz,
            interpolation_regions=interpolation_regions,
        ),
        index,
    )


def _parse_list(
    lines: Sequence[str],
    index: int,
) -> tuple[_EndfList, int]:
    c1, c2, l1, l2, nw, n2 = _parse_cont(lines[index])
    index += 1
    values, index = _parse_values(lines, index, nw)

    return (
        _EndfList(c1=c1, c2=c2, l1=l1, l2=l2, nw=nw, n2=n2, values=values),
        index,
    )


def parse_mf3_cross_section_points(
    path: str | Path,
    *,
    mts: Iterable[int] | None = None,
    exclude_mts: Iterable[int] = (0, 2),
) -> tuple[TendlMf3CrossSectionPoint, ...]:
    """Parse MF=3 cross-section points from TENDL/ENDF-6.

    MT=2 is elastic scattering and is excluded by default because RADAR uses
    these compact tables for nonelastic survival and secondary-production
    kernels.
    """

    excluded = set(exclude_mts)
    selected_mts = tuple(mts) if mts is not None else _discover_mts(path, mf=3)
    points: list[TendlMf3CrossSectionPoint] = []

    for mt in selected_mts:
        if mt <= 0 or mt in excluded:
            continue

        lines = _section_lines(path, mf=3, mt=mt)

        if not lines:
            continue

        tab1, _ = _parse_tab1(lines, 1)
        _require_endf_interpolation_law(
            tab1.interpolation_regions,
            supported_int=2,
            supported_name="linear-linear",
            context=f"MF=3 MT={mt}",
        )

        for energy_ev, sigma_barn in zip(tab1.x, tab1.y, strict=True):
            if energy_ev < 0.0:
                msg = f"MF=3 MT={mt} contains negative energy."
                raise ValueError(msg)

            if sigma_barn < 0.0:
                msg = f"MF=3 MT={mt} contains negative cross section."
                raise ValueError(msg)

            points.append(
                TendlMf3CrossSectionPoint(
                    mt=mt,
                    energy_mev=energy_ev / EV_PER_MEV,
                    sigma_barn=sigma_barn,
                )
            )

    return tuple(sorted(points, key=lambda point: (point.mt, point.energy_mev)))


def _skip_unsupported_mf6_product(lines: Sequence[str], index: int, law: int) -> int:
    if law == 6:
        return index + 1

    tab2, index = _parse_tab2(lines, index)

    for _ in range(tab2.nz):
        _, index = _parse_list(lines, index)

    return index


def _parse_mf6_products_for_mt(
    path: str | Path,
    *,
    mt: int,
    target_zap: int,
) -> tuple[_ParsedMf6Product, ...]:
    lines = _section_lines(path, mf=6, mt=mt)

    if not lines:
        return ()

    _, _, _, _, product_count, _ = _parse_cont(lines[0])
    index = 1
    products: list[_ParsedMf6Product] = []

    for product_index in range(product_count):
        zap, _, _, law, nr, np_count = _parse_cont(lines[index])
        index += 1

        yield_interpolation_values, index = _parse_values(lines, index, 2 * nr)
        yield_interpolation_regions = _parse_interpolation_regions(
            yield_interpolation_values,
            point_count=np_count,
            context=f"MF=6 MT={mt} product={product_index} yield",
        )
        yield_values, index = _parse_values(lines, index, 2 * np_count)

        yield_points = tuple(
            TendlMf6ProtonYieldPoint(
                mt=mt,
                product_index=product_index,
                incident_energy_mev=energy_ev / EV_PER_MEV,
                yield_value=yield_value,
            )
            for energy_ev, yield_value in zip(yield_values[0::2], yield_values[1::2], strict=True)
        )

        pdf_points: list[TendlMf6ProtonPdfPoint] = []

        if law == 1:
            tab2, index = _parse_tab2(lines, index)

            for _ in range(tab2.nz):
                endf_list, index = _parse_list(lines, index)
                angular_coefficients = endf_list.l2
                emitted_point_count = endf_list.n2
                group_size = angular_coefficients + 2

                if emitted_point_count <= 0:
                    continue

                if group_size * emitted_point_count != endf_list.nw:
                    group_size = endf_list.nw // emitted_point_count

                if group_size <= 1 or group_size * emitted_point_count != endf_list.nw:
                    continue

                incident_energy_mev = endf_list.c2 / EV_PER_MEV

                for emitted_index in range(emitted_point_count):
                    offset = emitted_index * group_size
                    emitted_energy_mev = endf_list.values[offset] / EV_PER_MEV
                    pdf_per_mev = endf_list.values[offset + 1] * EV_PER_MEV

                    if emitted_energy_mev < 0.0:
                        msg = f"MF=6 MT={mt} contains negative emitted energy."
                        raise ValueError(msg)

                    if pdf_per_mev < 0.0:
                        msg = f"MF=6 MT={mt} contains negative emitted-proton PDF."
                        raise ValueError(msg)

                    pdf_points.append(
                        TendlMf6ProtonPdfPoint(
                            mt=mt,
                            product_index=product_index,
                            incident_energy_mev=incident_energy_mev,
                            emitted_energy_mev=emitted_energy_mev,
                            pdf_per_mev=pdf_per_mev,
                        )
                    )
        else:
            index = _skip_unsupported_mf6_product(lines, index, law)

        rounded_zap = int(round(zap))

        if rounded_zap == target_zap:
            if law != 1:
                msg = (
                    f"MF=6 MT={mt} emitted-proton product uses unsupported LAW={law}; "
                    "only LAW=1 is implemented."
                )
                raise ValueError(msg)
            _require_endf_interpolation_law(
                yield_interpolation_regions,
                supported_int=2,
                supported_name="linear-linear",
                context=f"MF=6 MT={mt} product={product_index} yield",
            )
            _require_endf_interpolation_law(
                tab2.interpolation_regions,
                supported_int=2,
                supported_name="linear-linear",
                context=f"MF=6 MT={mt} product={product_index} incident-energy PDF",
            )
            if tab2.l2 != 1:
                msg = (
                    f"MF=6 MT={mt} product={product_index} outgoing-energy PDF uses "
                    f"unsupported ENDF LEP={tab2.l2}; only histogram-left (LEP=1) "
                    "is implemented."
                )
                raise ValueError(msg)

        if rounded_zap == target_zap and pdf_points:
            products.append(
                _ParsedMf6Product(
                    mt=mt,
                    product_index=product_index,
                    zap=rounded_zap,
                    law=law,
                    yield_points=yield_points,
                    pdf_points=tuple(pdf_points),
                )
            )

    return tuple(products)


def parse_mf6_secondary_proton_points(
    path: str | Path,
    *,
    target_zap: int = ENDT_PROTON_ZAP,
    exclude_mts: Iterable[int] = (0, 2),
) -> tuple[tuple[TendlMf6ProtonYieldPoint, ...], tuple[TendlMf6ProtonPdfPoint, ...]]:
    """Parse emitted-proton MF=6 LAW=1 yields and PDFs from TENDL/ENDF-6."""

    excluded = set(exclude_mts)
    yield_points: list[TendlMf6ProtonYieldPoint] = []
    pdf_points: list[TendlMf6ProtonPdfPoint] = []

    for mt in _discover_mts(path, mf=6):
        if mt in excluded:
            continue

        for product in _parse_mf6_products_for_mt(path, mt=mt, target_zap=target_zap):
            yield_points.extend(product.yield_points)
            pdf_points.extend(product.pdf_points)

    return (
        tuple(
            sorted(
                yield_points,
                key=lambda point: (point.mt, point.product_index, point.incident_energy_mev),
            )
        ),
        tuple(
            sorted(
                pdf_points,
                key=lambda point: (
                    point.mt,
                    point.product_index,
                    point.incident_energy_mev,
                    point.emitted_energy_mev,
                ),
            )
        ),
    )


def convert_tendl_p_al27_to_csvs(
    *,
    tendl_path: str | Path,
    output_dir: str | Path,
    prefix: str = "p_al27",
) -> TendlCsvConversionOutput:
    """Convert p+Al27 TENDL data to compact RADAR CSV tables."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    mf3_points = parse_mf3_cross_section_points(tendl_path)
    yield_points, pdf_points = parse_mf6_secondary_proton_points(tendl_path)

    if not mf3_points:
        msg = "No MF=3 cross-section points were parsed from TENDL file."
        raise ValueError(msg)

    if not yield_points or not pdf_points:
        msg = "No MF=6 emitted-proton points were parsed from TENDL file."
        raise ValueError(msg)

    xs_path = destination / f"{prefix}_mf3_xs.csv"
    yield_path = destination / f"{prefix}_mf6_proton_yield.csv"
    pdf_path = destination / f"{prefix}_mf6_proton_pdf.csv"

    with xs_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("mt", "energy_mev", "sigma_barn"))
        for xs_point in mf3_points:
            writer.writerow((xs_point.mt, xs_point.energy_mev, xs_point.sigma_barn))

    with yield_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "mt",
                "product_index",
                "incident_energy_mev",
                "yield",
                "yield_interpolation",
            )
        )
        for yield_point in yield_points:
            writer.writerow(
                (
                    yield_point.mt,
                    yield_point.product_index,
                    yield_point.incident_energy_mev,
                    yield_point.yield_value,
                    yield_point.interpolation_law.value,
                )
            )

    with pdf_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "mt",
                "product_index",
                "incident_energy_mev",
                "emitted_energy_mev",
                "pdf_per_mev",
                "pdf_incident_interpolation",
                "pdf_outgoing_interpolation",
            )
        )
        for pdf_point in pdf_points:
            writer.writerow(
                (
                    pdf_point.mt,
                    pdf_point.product_index,
                    pdf_point.incident_energy_mev,
                    pdf_point.emitted_energy_mev,
                    pdf_point.pdf_per_mev,
                    pdf_point.incident_energy_interpolation_law.value,
                    pdf_point.outgoing_energy_interpolation_law.value,
                )
            )

    return TendlCsvConversionOutput(
        mf3_xs=xs_path,
        mf6_proton_yield=yield_path,
        mf6_proton_pdf=pdf_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert p+Al27 TENDL/ENDF-6 data to compact RADAR CSV tables."
    )
    parser.add_argument("tendl_path")
    parser.add_argument("output_dir")
    parser.add_argument("--prefix", default="p_al27")

    args = parser.parse_args(argv)

    output = convert_tendl_p_al27_to_csvs(
        tendl_path=args.tendl_path,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )

    print(f"MF=3 cross sections: {output.mf3_xs}")
    print(f"MF=6 proton yields:  {output.mf6_proton_yield}")
    print(f"MF=6 proton PDFs:    {output.mf6_proton_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
