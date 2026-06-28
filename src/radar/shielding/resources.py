"""Package resource loaders for shielding and LET normative tables."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Callable, TypeVar

from radar.shielding.electron_al import (
    ElectronAlRangeEnergyTable,
    load_electron_al_range_energy_table,
)
from radar.shielding.hze_al import HzeAlRangeTable, load_hze_al_range_tables
from radar.shielding.hze_si_let import HzeSiLetTable, load_hze_si_let_tables
from radar.shielding.proton_al import (
    ProtonAlRangeEnergyTable,
    load_proton_al_range_energy_table,
)
from radar.shielding.proton_al_secondary import (
    SecondaryProtonKernel,
    load_secondary_proton_kernel,
)
from radar.shielding.proton_al_survival import (
    DEFAULT_AL27_NONELASTIC_MT,
    Al27NonelasticCrossSectionTable,
    load_al27_nonelastic_cross_section_table,
)
from radar.shielding.proton_si_let import ProtonSiLetTable, load_proton_si_let_table

_NORMATIVE_PACKAGE = "radar.data.normative"
_T = TypeVar("_T")

PROTON_AL_RANGE_RESOURCE = "stopping/proton_al_range.csv"
ELECTRON_AL_RANGE_RESOURCE = "stopping/electron_al_range.csv"
PROTON_SI_LET_RESOURCE = "stopping/proton_si_let.csv"
HZE_AL_RANGE_RESOURCE = "stopping/hze_al_range.csv"
HZE_SI_LET_RESOURCE = "stopping/hze_si_let.csv"
P_AL27_MF3_XS_RESOURCE = "tendl/p_al27_mf3_xs.csv"
P_AL27_MF6_PROTON_YIELD_RESOURCE = "tendl/p_al27_mf6_proton_yield.csv"
P_AL27_MF6_PROTON_PDF_RESOURCE = "tendl/p_al27_mf6_proton_pdf.csv"


@dataclass(frozen=True)
class ShieldingNormativeTables:
    """Normative shielding/LET tables loaded from package resources."""

    proton_al_range: ProtonAlRangeEnergyTable
    proton_si_let: ProtonSiLetTable
    hze_al_ranges_by_z: dict[int, HzeAlRangeTable]
    hze_si_let_by_z: dict[int, HzeSiLetTable]
    al27_nonelastic_xs: Al27NonelasticCrossSectionTable
    secondary_proton_kernel: SecondaryProtonKernel
    electron_al_range: ElectronAlRangeEnergyTable | None = None


def _load_resource(relative_path: str, loader: Callable[[Path], _T]) -> _T:
    resource = files(_NORMATIVE_PACKAGE).joinpath(relative_path)

    with as_file(resource) as path:
        return loader(path)


def load_normative_proton_al_range_table() -> ProtonAlRangeEnergyTable:
    """Load packaged normalized proton range-energy table in aluminium."""

    return _load_resource(
        PROTON_AL_RANGE_RESOURCE,
        load_proton_al_range_energy_table,
    )


def load_normative_electron_al_range_table() -> ElectronAlRangeEnergyTable:
    """Load packaged normalized ESTAR electron range-energy table in aluminium."""

    return _load_resource(
        ELECTRON_AL_RANGE_RESOURCE,
        load_electron_al_range_energy_table,
    )


def load_normative_proton_si_let_table() -> ProtonSiLetTable:
    """Load packaged normalized proton LET table in silicon."""

    return _load_resource(
        PROTON_SI_LET_RESOURCE,
        load_proton_si_let_table,
    )


def load_normative_hze_al_range_tables() -> dict[int, HzeAlRangeTable]:
    """Load packaged normalized HZE range-energy tables in aluminium."""

    return _load_resource(
        HZE_AL_RANGE_RESOURCE,
        load_hze_al_range_tables,
    )


def load_normative_hze_si_let_tables() -> dict[int, HzeSiLetTable]:
    """Load packaged normalized HZE LET tables in silicon."""

    return _load_resource(
        HZE_SI_LET_RESOURCE,
        load_hze_si_let_tables,
    )


def load_normative_al27_nonelastic_cross_section_table(
    *,
    mt: int = DEFAULT_AL27_NONELASTIC_MT,
) -> Al27NonelasticCrossSectionTable:
    """Load packaged p+Al27 nonelastic/reaction cross section table."""

    return _load_resource(
        P_AL27_MF3_XS_RESOURCE,
        lambda path: load_al27_nonelastic_cross_section_table(path, mt=mt),
    )


def load_normative_secondary_proton_kernel() -> SecondaryProtonKernel:
    """Load packaged p+Al27 secondary-proton production kernel."""

    xs_resource = files(_NORMATIVE_PACKAGE).joinpath(P_AL27_MF3_XS_RESOURCE)
    yield_resource = files(_NORMATIVE_PACKAGE).joinpath(P_AL27_MF6_PROTON_YIELD_RESOURCE)
    pdf_resource = files(_NORMATIVE_PACKAGE).joinpath(P_AL27_MF6_PROTON_PDF_RESOURCE)

    with as_file(xs_resource) as xs_path:
        with as_file(yield_resource) as yield_path:
            with as_file(pdf_resource) as pdf_path:
                return load_secondary_proton_kernel(
                    xs_path=xs_path,
                    yield_path=yield_path,
                    pdf_path=pdf_path,
                )


def load_normative_shielding_tables() -> ShieldingNormativeTables:
    """Load all packaged shielding/LET normative tables required by SEP shielding."""

    return ShieldingNormativeTables(
        proton_al_range=load_normative_proton_al_range_table(),
        electron_al_range=load_normative_electron_al_range_table(),
        proton_si_let=load_normative_proton_si_let_table(),
        hze_al_ranges_by_z=load_normative_hze_al_range_tables(),
        hze_si_let_by_z=load_normative_hze_si_let_tables(),
        al27_nonelastic_xs=load_normative_al27_nonelastic_cross_section_table(),
        secondary_proton_kernel=load_normative_secondary_proton_kernel(),
    )


__all__ = [
    "ELECTRON_AL_RANGE_RESOURCE",
    "HZE_AL_RANGE_RESOURCE",
    "HZE_SI_LET_RESOURCE",
    "P_AL27_MF3_XS_RESOURCE",
    "P_AL27_MF6_PROTON_PDF_RESOURCE",
    "P_AL27_MF6_PROTON_YIELD_RESOURCE",
    "PROTON_AL_RANGE_RESOURCE",
    "PROTON_SI_LET_RESOURCE",
    "ShieldingNormativeTables",
    "load_normative_al27_nonelastic_cross_section_table",
    "load_normative_electron_al_range_table",
    "load_normative_hze_al_range_tables",
    "load_normative_hze_si_let_tables",
    "load_normative_proton_al_range_table",
    "load_normative_proton_si_let_table",
    "load_normative_secondary_proton_kernel",
    "load_normative_shielding_tables",
]
