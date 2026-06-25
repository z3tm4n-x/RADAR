"""Secondary proton production in aluminium shielding from compact TENDL CSV data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource
from radar.shielding.proton_al import (
    ProtonAlRangeEnergyTable,
    _interpolate_zero_outside,
    log_bin_edges_from_centers,
)
from radar.shielding.proton_al_survival import (
    AL27_MOLAR_MASS_G_MOL,
    AVOGADRO_CONSTANT_MOL_INV,
    BARN_CM2,
)

PROTON_AL_SECONDARY_MODEL = "proton_secondary_al_tendl_kernel"
PROTON_AL_TOTAL_MODEL = "proton_total_al_primary_plus_secondary"

_REQUIRED_YIELD_COLUMNS = {
    "mt",
    "product_index",
    "incident_energy_mev",
    "yield",
}

_REQUIRED_PDF_COLUMNS = {
    "mt",
    "product_index",
    "incident_energy_mev",
    "emitted_energy_mev",
    "pdf_per_mev",
}

_REQUIRED_XS_COLUMNS = {
    "mt",
    "energy_mev",
    "sigma_barn",
}


@dataclass(frozen=True)
class Al27CrossSectionTable:
    """One p+Al27 MF=3 cross-section table."""

    mt: int
    energy_mev: tuple[float, ...]
    sigma_barn: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.mt <= 0:
            msg = "Cross-section MT must be positive."
            raise ValueError(msg)

        if len(self.energy_mev) != len(self.sigma_barn):
            msg = "Cross-section energy and sigma arrays must have equal length."
            raise ValueError(msg)

        if len(self.energy_mev) < 2:
            msg = "Cross-section table must contain at least two points."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.energy_mev):
            msg = "Cross-section energies must be finite."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.sigma_barn):
            msg = "Cross-section values must be finite."
            raise ValueError(msg)

        if any(value < 0.0 for value in self.sigma_barn):
            msg = "Cross-section values must be non-negative."
            raise ValueError(msg)

        if any(right <= left for left, right in zip(self.energy_mev, self.energy_mev[1:])):
            msg = "Cross-section energies must be strictly increasing."
            raise ValueError(msg)

    def sigma_at_energy(self, energy_mev: float) -> float:
        """Return cross section in barns at proton kinetic energy."""

        return _interpolate_zero_outside(
            x=energy_mev,
            xp=self.energy_mev,
            fp=self.sigma_barn,
        )


@dataclass(frozen=True)
class SecondaryProtonProductDistribution:
    """MF=6 emitted-proton yield and outgoing-energy density for one product."""

    mt: int
    product_index: int
    incident_energy_mev: tuple[float, ...]
    yield_value: tuple[float, ...]
    emitted_energy_mev_by_incident: tuple[tuple[float, ...], ...]
    pdf_per_mev_by_incident: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.incident_energy_mev) != len(self.yield_value):
            msg = "MF=6 yield energy and value arrays must have equal length."
            raise ValueError(msg)

        if len(self.incident_energy_mev) != len(self.emitted_energy_mev_by_incident):
            msg = "MF=6 incident and emitted-energy grids must have equal length."
            raise ValueError(msg)

        if len(self.incident_energy_mev) != len(self.pdf_per_mev_by_incident):
            msg = "MF=6 incident and PDF arrays must have equal length."
            raise ValueError(msg)

        if len(self.incident_energy_mev) < 2:
            msg = "MF=6 product distribution must contain at least two incident energies."
            raise ValueError(msg)

        if any(right <= left for left, right in zip(self.incident_energy_mev, self.incident_energy_mev[1:])):
            msg = "MF=6 incident energies must be strictly increasing."
            raise ValueError(msg)

    def yield_at_incident_energy(self, incident_energy_mev: float) -> float:
        """Return product yield at incident proton energy."""

        return _interpolate_zero_outside(
            x=incident_energy_mev,
            xp=self.incident_energy_mev,
            fp=self.yield_value,
        )

    def pdf_at(self, *, incident_energy_mev: float, emitted_energy_mev: float) -> float:
        """Return emitted-proton PDF in 1/MeV by bilinear interpolation."""

        if incident_energy_mev < self.incident_energy_mev[0]:
            return 0.0

        if incident_energy_mev > self.incident_energy_mev[-1]:
            return 0.0

        if incident_energy_mev == self.incident_energy_mev[0]:
            return _pdf_at_index(self, 0, emitted_energy_mev)

        if incident_energy_mev == self.incident_energy_mev[-1]:
            return _pdf_at_index(self, len(self.incident_energy_mev) - 1, emitted_energy_mev)

        for index in range(len(self.incident_energy_mev) - 1):
            left_energy = self.incident_energy_mev[index]
            right_energy = self.incident_energy_mev[index + 1]

            if left_energy <= incident_energy_mev <= right_energy:
                left_pdf = _pdf_at_index(self, index, emitted_energy_mev)
                right_pdf = _pdf_at_index(self, index + 1, emitted_energy_mev)
                fraction = (incident_energy_mev - left_energy) / (right_energy - left_energy)
                return left_pdf + fraction * (right_pdf - left_pdf)

        return 0.0


@dataclass(frozen=True)
class SecondaryProtonKernel:
    """Secondary emitted-proton kernel for p+Al27."""

    products: tuple[SecondaryProtonProductDistribution, ...]
    cross_sections_by_mt: Mapping[int, Al27CrossSectionTable]

    def source_density_per_g_cm2(
        self,
        *,
        incident_energy_mev: float,
        emitted_energy_mev: float,
    ) -> float:
        """Return emitted proton yield density per g/cm² and MeV.

        This is sum_MT N_A/A * sigma_MT(E) * yield_MT(E) * pdf_MT(Eout|E).
        """

        number_per_gram = AVOGADRO_CONSTANT_MOL_INV / AL27_MOLAR_MASS_G_MOL
        density = 0.0

        for product in self.products:
            cross_section = self.cross_sections_by_mt.get(product.mt)

            if cross_section is None:
                continue

            sigma_cm2 = cross_section.sigma_at_energy(incident_energy_mev) * BARN_CM2
            yield_value = product.yield_at_incident_energy(incident_energy_mev)
            pdf = product.pdf_at(
                incident_energy_mev=incident_energy_mev,
                emitted_energy_mev=emitted_energy_mev,
            )
            density += number_per_gram * sigma_cm2 * yield_value * pdf

        return density


def _pdf_at_index(
    product: SecondaryProtonProductDistribution,
    index: int,
    emitted_energy_mev: float,
) -> float:
    return _interpolate_zero_outside(
        x=emitted_energy_mev,
        xp=product.emitted_energy_mev_by_incident[index],
        fp=product.pdf_per_mev_by_incident[index],
    )


def _required_text(row: Mapping[str, str | None], column: str, line_number: int) -> str:
    value = row.get(column)

    if value is None or not value.strip():
        msg = f"Missing required column {column!r} at CSV line {line_number}."
        raise ValueError(msg)

    return value.strip()


def _require_columns(fieldnames: set[str], required: set[str], name: str) -> None:
    missing_columns = required - fieldnames

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        msg = f"{name} CSV is missing columns: {missing}."
        raise ValueError(msg)


def load_al27_cross_section_tables(
    path: str | Path,
    *,
    mts: set[int] | None = None,
) -> dict[int, Al27CrossSectionTable]:
    """Load p+Al27 MF=3 cross-section tables from compact CSV."""

    rows_by_mt: dict[int, list[tuple[float, float]]] = {}

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        _require_columns(set(reader.fieldnames or ()), _REQUIRED_XS_COLUMNS, "MF=3")

        for line_number, row in enumerate(reader, start=2):
            mt = int(_required_text(row, "mt", line_number))

            if mts is not None and mt not in mts:
                continue

            rows_by_mt.setdefault(mt, []).append(
                (
                    float(_required_text(row, "energy_mev", line_number)),
                    float(_required_text(row, "sigma_barn", line_number)),
                )
            )

    tables: dict[int, Al27CrossSectionTable] = {}

    for mt, rows in rows_by_mt.items():
        rows.sort(key=lambda item: item[0])
        tables[mt] = Al27CrossSectionTable(
            mt=mt,
            energy_mev=tuple(row[0] for row in rows),
            sigma_barn=tuple(row[1] for row in rows),
        )

    return tables


def load_secondary_proton_products(
    *,
    yield_path: str | Path,
    pdf_path: str | Path,
) -> tuple[SecondaryProtonProductDistribution, ...]:
    """Load MF=6 emitted-proton product distributions from compact CSV."""

    yield_by_key: dict[tuple[int, int], list[tuple[float, float]]] = {}

    with Path(yield_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        _require_columns(set(reader.fieldnames or ()), _REQUIRED_YIELD_COLUMNS, "MF=6 yield")

        for line_number, row in enumerate(reader, start=2):
            mt = int(_required_text(row, "mt", line_number))
            product_index = int(_required_text(row, "product_index", line_number))
            incident_energy = float(_required_text(row, "incident_energy_mev", line_number))
            yield_value = float(_required_text(row, "yield", line_number))
            yield_by_key.setdefault((mt, product_index), []).append((incident_energy, yield_value))

    pdf_by_key_energy: dict[tuple[int, int], dict[float, list[tuple[float, float]]]] = {}

    with Path(pdf_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        _require_columns(set(reader.fieldnames or ()), _REQUIRED_PDF_COLUMNS, "MF=6 PDF")

        for line_number, row in enumerate(reader, start=2):
            mt = int(_required_text(row, "mt", line_number))
            product_index = int(_required_text(row, "product_index", line_number))
            incident_energy = float(_required_text(row, "incident_energy_mev", line_number))
            emitted_energy = float(_required_text(row, "emitted_energy_mev", line_number))
            pdf = float(_required_text(row, "pdf_per_mev", line_number))
            key = (mt, product_index)
            pdf_by_key_energy.setdefault(key, {}).setdefault(incident_energy, []).append(
                (emitted_energy, pdf)
            )

    products: list[SecondaryProtonProductDistribution] = []

    for key, yield_rows in yield_by_key.items():
        if key not in pdf_by_key_energy:
            continue

        mt, product_index = key
        yield_rows.sort(key=lambda row: row[0])
        incident_energies = tuple(row[0] for row in yield_rows)

        emitted_grids: list[tuple[float, ...]] = []
        pdf_grids: list[tuple[float, ...]] = []

        for incident_energy in incident_energies:
            pdf_rows = pdf_by_key_energy[key].get(incident_energy)

            if not pdf_rows:
                break

            pdf_rows.sort(key=lambda row: row[0])
            emitted_grids.append(tuple(row[0] for row in pdf_rows))
            pdf_grids.append(tuple(row[1] for row in pdf_rows))
        else:
            products.append(
                SecondaryProtonProductDistribution(
                    mt=mt,
                    product_index=product_index,
                    incident_energy_mev=incident_energies,
                    yield_value=tuple(row[1] for row in yield_rows),
                    emitted_energy_mev_by_incident=tuple(emitted_grids),
                    pdf_per_mev_by_incident=tuple(pdf_grids),
                )
            )

    return tuple(sorted(products, key=lambda product: (product.mt, product.product_index)))


def load_secondary_proton_kernel(
    *,
    xs_path: str | Path,
    yield_path: str | Path,
    pdf_path: str | Path,
) -> SecondaryProtonKernel:
    """Load p+Al27 secondary-proton kernel from compact CSV tables."""

    products = load_secondary_proton_products(yield_path=yield_path, pdf_path=pdf_path)
    mts = {product.mt for product in products}
    cross_sections = load_al27_cross_section_tables(xs_path, mts=mts)

    return SecondaryProtonKernel(
        products=products,
        cross_sections_by_mt=cross_sections,
    )


def _incident_energy_at_depth(
    *,
    entrance_energy_mev: float,
    depth_g_cm2: float,
    table: ProtonAlRangeEnergyTable,
) -> float | None:
    entrance_range = table.range_at_energy(entrance_energy_mev)
    residual_range = entrance_range - depth_g_cm2

    if residual_range <= 0.0:
        return None

    return table.energy_at_range(residual_range)


def _birth_energy_for_output(
    *,
    output_energy_mev: float,
    residual_thickness_g_cm2: float,
    table: ProtonAlRangeEnergyTable,
) -> float | None:
    output_range = table.range_at_energy(output_energy_mev)
    birth_range = output_range + residual_thickness_g_cm2

    if birth_range > table.max_range_g_cm2:
        return None

    return table.energy_at_range(birth_range)


def calculate_secondary_proton_spectrum_through_al(
    *,
    incident_spectrum: Spectrum1D,
    stopping_table: ProtonAlRangeEnergyTable,
    kernel: SecondaryProtonKernel,
    thickness_g_cm2: float,
    depth_steps: int = 24,
    model: str | None = None,
) -> Spectrum1D:
    """Return secondary-proton spectrum emitted in Al and transported to shield exit."""

    if thickness_g_cm2 < 0.0:
        msg = "Al shielding thickness must be non-negative."
        raise ValueError(msg)

    if depth_steps < 1:
        msg = "Secondary-proton depth integration must use at least one step."
        raise ValueError(msg)

    if incident_spectrum.particle is not Particle.PROTON:
        msg = "Secondary-proton Al kernel requires an incident proton spectrum."
        raise ValueError(msg)

    if incident_spectrum.source is not RadiationSource.SEP:
        msg = "Secondary-proton Al kernel currently supports SEP spectra only."
        raise ValueError(msg)

    if thickness_g_cm2 == 0.0:
        return Spectrum1D(
            x=incident_spectrum.x,
            y=tuple(0.0 for _ in incident_spectrum.x),
            x_unit=incident_spectrum.x_unit,
            y_unit=incident_spectrum.y_unit,
            quantity=incident_spectrum.quantity,
            particle=Particle.PROTON,
            source=incident_spectrum.source,
            model=model or f"{incident_spectrum.model}+{PROTON_AL_SECONDARY_MODEL}",
        )

    output_edges = log_bin_edges_from_centers(incident_spectrum.x)
    input_edges = output_edges
    secondary_values = [0.0 for _ in incident_spectrum.x]
    depth_step = thickness_g_cm2 / depth_steps

    for input_index, entrance_energy in enumerate(incident_spectrum.x):
        input_width = input_edges[input_index + 1] - input_edges[input_index]
        incident_count = incident_spectrum.y[input_index] * input_width

        if incident_count <= 0.0:
            continue

        for depth_index in range(depth_steps):
            depth = (depth_index + 0.5) * depth_step
            incident_energy_at_depth = _incident_energy_at_depth(
                entrance_energy_mev=entrance_energy,
                depth_g_cm2=depth,
                table=stopping_table,
            )

            if incident_energy_at_depth is None:
                continue

            residual_thickness = thickness_g_cm2 - depth

            for output_index, output_energy in enumerate(incident_spectrum.x):
                output_lo = output_edges[output_index]
                output_hi = output_edges[output_index + 1]
                output_width = output_hi - output_lo

                birth_lo = _birth_energy_for_output(
                    output_energy_mev=output_lo,
                    residual_thickness_g_cm2=residual_thickness,
                    table=stopping_table,
                )
                birth_hi = _birth_energy_for_output(
                    output_energy_mev=output_hi,
                    residual_thickness_g_cm2=residual_thickness,
                    table=stopping_table,
                )
                birth_center = _birth_energy_for_output(
                    output_energy_mev=output_energy,
                    residual_thickness_g_cm2=residual_thickness,
                    table=stopping_table,
                )

                if birth_lo is None or birth_hi is None or birth_center is None:
                    continue

                birth_width = birth_hi - birth_lo
                source_density = kernel.source_density_per_g_cm2(
                    incident_energy_mev=incident_energy_at_depth,
                    emitted_energy_mev=birth_center,
                )
                secondary_counts = incident_count * source_density * depth_step * birth_width
                secondary_values[output_index] += secondary_counts / output_width

    return Spectrum1D(
        x=incident_spectrum.x,
        y=tuple(secondary_values),
        x_unit=incident_spectrum.x_unit,
        y_unit=incident_spectrum.y_unit,
        quantity=incident_spectrum.quantity,
        particle=Particle.PROTON,
        source=incident_spectrum.source,
        model=model or f"{incident_spectrum.model}+{PROTON_AL_SECONDARY_MODEL}",
    )


def combine_primary_and_secondary_proton_spectra(
    *,
    primary: Spectrum1D,
    secondary: Spectrum1D,
    model: str | None = None,
) -> Spectrum1D:
    """Return total proton spectrum behind shield."""

    if primary.x != secondary.x:
        msg = "Primary and secondary proton spectra must share the same energy grid."
        raise ValueError(msg)

    if primary.x_unit is not secondary.x_unit or primary.y_unit is not secondary.y_unit:
        msg = "Primary and secondary proton spectra must share units."
        raise ValueError(msg)

    if primary.quantity is not secondary.quantity:
        msg = "Primary and secondary proton spectra must share the same quantity."
        raise ValueError(msg)

    total_y = tuple(left + right for left, right in zip(primary.y, secondary.y, strict=True))

    return Spectrum1D(
        x=primary.x,
        y=total_y,
        x_unit=primary.x_unit,
        y_unit=primary.y_unit,
        quantity=primary.quantity,
        particle=Particle.PROTON,
        source=primary.source,
        model=model or f"{primary.model}+{secondary.model}+{PROTON_AL_TOTAL_MODEL}",
    )
