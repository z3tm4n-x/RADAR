"""TENDL/ENDF-6 parser for compact RADAR nuclear-data CSV conversion."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ENDT_PROTON_ZAP = 1001
EV_PER_MEV = 1.0e6


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


@dataclass(frozen=True)
class TendlMf6ProtonPdfPoint:
    """One emitted-proton MF=6 LAW=1 outgoing-energy density point."""

    mt: int
    product_index: int
    incident_energy_mev: float
    emitted_energy_mev: float
    pdf_per_mev: float


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

    _, index = _parse_values(lines, index, 2 * nr)
    values, index = _parse_values(lines, index, 2 * np_count)

    return (
        _EndfTab1(
            c1=c1,
            c2=c2,
            l1=l1,
            l2=l2,
            nr=nr,
            np=np_count,
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
    _, index = _parse_values(lines, index, 2 * nr)

    return _EndfTab2(c1=c1, c2=c2, l1=l1, l2=l2, nr=nr, nz=nz), index


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

        _, index = _parse_values(lines, index, 2 * nr)
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

        if rounded_zap == target_zap and law == 1 and pdf_points:
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
        writer.writerow(("mt", "product_index", "incident_energy_mev", "yield"))
        for yield_point in yield_points:
            writer.writerow(
                (
                    yield_point.mt,
                    yield_point.product_index,
                    yield_point.incident_energy_mev,
                    yield_point.yield_value,
                )
            )

    with pdf_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("mt", "product_index", "incident_energy_mev", "emitted_energy_mev", "pdf_per_mev"))
        for pdf_point in pdf_points:
            writer.writerow(
                (
                    pdf_point.mt,
                    pdf_point.product_index,
                    pdf_point.incident_energy_mev,
                    pdf_point.emitted_energy_mev,
                    pdf_point.pdf_per_mev,
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
