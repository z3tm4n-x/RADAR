"""GCR output table builders.

Dose and single-event-effect output tables are intentionally not constructed
here. They are deferred until dose/SEE physical models are available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.output_tables import OutputTable, spectrum_output_table

from radar.gcr.shielding import (
    GcrHzeShieldingResult,
    GcrLetProductsResult,
    GcrProtonShieldingResult,
)
from radar.gcr.source_spectra import (
    GCR_AKL_COMPONENT,
    GCR_MAIN_COMPONENT,
    GCR_NEK_COMPONENT,
    GCR_TOTAL_COMPONENT,
    GOST_GCR_SOURCE_SPECTRA_MODEL,
    OST_GCR_SOURCE_SPECTRA_MODEL,
    GcrIonSourceSpectra,
    GcrSourceSpectra,
)

GCR_OUTPUT_LOCATION_OUTSIDE_MAGNETOSPHERE = "outside_magnetosphere"
GCR_OUTPUT_LOCATION_ON_ORBIT = "on_orbit"
GCR_OUTPUT_LOCATION_BEHIND_SHIELDING = "behind_shielding"

GCR_OUTPUT_SHIELD_MATERIAL_AL = "Al"
GCR_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL = "centered_spherical_shell"

GCR_OUTPUT_DOSE_STATUS_NOT_CALCULATED = "not_calculated"
GCR_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED = "not_calculated"
GCR_OUTPUT_DOSE_SEE_DEFERRED_REASON = "deferred_until_gcr_erb_models"

GCR_OUTPUT_MAIN_COMPONENT_GOST = "gost"
GCR_OUTPUT_MAIN_COMPONENT_OST = "ost"
GCR_OUTPUT_LOW_ENERGY_COMPONENT_INCLUDED = "included"


def _format_float(value: float) -> str:
    if not isfinite(value):
        msg = "GCR output table metadata float value must be finite."
        raise ValueError(msg)

    return f"{value:.12g}"


def _main_component_from_model(model: str) -> str:
    if model.startswith(GOST_GCR_SOURCE_SPECTRA_MODEL):
        return GCR_OUTPUT_MAIN_COMPONENT_GOST

    if model.startswith(OST_GCR_SOURCE_SPECTRA_MODEL):
        return GCR_OUTPUT_MAIN_COMPONENT_OST

    msg = f"Cannot infer GCR main component from model {model!r}."
    raise ValueError(msg)


def _symbol_from_spectrum_model(spectrum: Spectrum1D) -> str:
    parts = spectrum.model.split(":")

    if len(parts) < 2 or not parts[1]:
        msg = "GCR output spectrum model must contain an ion symbol tag."
        raise ValueError(msg)

    return parts[1].split("+", maxsplit=1)[0]


def _common_metadata(
    *,
    spectrum: Spectrum1D,
    stage: str,
    location: str,
    geomagnetic_penetration: str,
    shielding: str,
) -> dict[str, str]:
    return {
        "stage": stage,
        "location": location,
        "main_component": _main_component_from_model(spectrum.model),
        "nek": GCR_OUTPUT_LOW_ENERGY_COMPONENT_INCLUDED,
        "akl": GCR_OUTPUT_LOW_ENERGY_COMPONENT_INCLUDED,
        "geomagnetic_penetration": geomagnetic_penetration,
        "shielding": shielding,
        "dose_status": GCR_OUTPUT_DOSE_STATUS_NOT_CALCULATED,
        "single_event_effects_status": (
            GCR_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED
        ),
        "dose_see_dependency": GCR_OUTPUT_DOSE_SEE_DEFERRED_REASON,
    }


def _metadata_with(
    base: Mapping[str, str],
    extra: Mapping[str, str],
) -> dict[str, str]:
    metadata = dict(base)
    metadata.update(extra)
    return metadata


def _table_id_token(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("+", "_")
        .replace(":", "_")
    )


def _source_component_spectra(
    ion: GcrIonSourceSpectra,
    *,
    include_components: bool,
) -> tuple[tuple[str, Spectrum1D], ...]:
    if include_components:
        return (
            (GCR_MAIN_COMPONENT, ion.main),
            (GCR_NEK_COMPONENT, ion.nek),
            (GCR_AKL_COMPONENT, ion.akl),
            (GCR_TOTAL_COMPONENT, ion.total),
        )

    return ((GCR_TOTAL_COMPONENT, ion.total),)


def gcr_source_spectra_output_tables(
    source_spectra: GcrSourceSpectra,
    *,
    include_components: bool = False,
) -> tuple[OutputTable, ...]:
    """Return GCR source spectrum tables outside the magnetosphere."""

    tables: list[OutputTable] = []
    main_component = _main_component_from_model(source_spectra.model)

    for ion in source_spectra.ions:
        for component, spectrum in _source_component_spectra(
            ion,
            include_components=include_components,
        ):
            base = _common_metadata(
                spectrum=spectrum,
                stage="source",
                location=GCR_OUTPUT_LOCATION_OUTSIDE_MAGNETOSPHERE,
                geomagnetic_penetration="not_applied",
                shielding="not_applied",
            )
            tables.append(
                spectrum_output_table(
                    table_id=(
                        f"gcr_{main_component}_{ion.symbol.lower()}_"
                        f"{component}_source_outside_magnetosphere"
                    ),
                    title=(
                        f"ГКЛ. {ion.symbol}. {component}. "
                        "Исходный спектр вне магнитосферы"
                    ),
                    spectrum=spectrum,
                    metadata=_metadata_with(
                        base,
                        {
                            "component": component,
                            "z": str(ion.z),
                            "symbol": ion.symbol,
                        },
                    ),
                )
            )

    return tuple(tables)


def gcr_on_orbit_product_output_tables(
    products: Sequence[SpectrumProduct],
) -> tuple[OutputTable, ...]:
    """Return GCR on-orbit product tables after geomagnetic penetration."""

    if not products:
        msg = "GCR on-orbit output tables require at least one product."
        raise ValueError(msg)

    tables: list[OutputTable] = []

    for product in products:
        spectrum = product.spectrum
        symbol = _symbol_from_spectrum_model(spectrum)
        base = _common_metadata(
            spectrum=spectrum,
            stage="geomagnetic_penetration",
            location=GCR_OUTPUT_LOCATION_ON_ORBIT,
            geomagnetic_penetration="applied",
            shielding="not_applied",
        )
        tables.append(
            spectrum_output_table(
                table_id=(
                    f"gcr_{symbol.lower()}_{product.kind.value}_on_orbit"
                ),
                title=(
                    f"ГКЛ. {symbol}. {product.kind.value}. "
                    "Спектр на орбите после геомагнитного проникновения"
                ),
                spectrum=spectrum,
                metadata=_metadata_with(
                    base,
                    {
                        "component": GCR_TOTAL_COMPONENT,
                        "product_kind": product.kind.value,
                        "symbol": symbol,
                    },
                ),
            )
        )

    return tuple(tables)


def _shielding_metadata(
    *,
    spectrum: Spectrum1D,
    stage: str,
    thickness_g_cm2: float,
) -> dict[str, str]:
    return _metadata_with(
        _common_metadata(
            spectrum=spectrum,
            stage=stage,
            location=GCR_OUTPUT_LOCATION_BEHIND_SHIELDING,
            geomagnetic_penetration="applied",
            shielding="applied",
        ),
        {
            "shield_material": GCR_OUTPUT_SHIELD_MATERIAL_AL,
            "shield_geometry": GCR_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL,
            "thickness_g_cm2": _format_float(thickness_g_cm2),
        },
    )


def gcr_proton_shielding_output_tables(
    result: GcrProtonShieldingResult,
    *,
    include_components: bool = False,
) -> tuple[OutputTable, ...]:
    """Return GCR proton behind-shielding energy spectrum tables."""

    base = _shielding_metadata(
        spectrum=result.total,
        stage="shielding",
        thickness_g_cm2=result.thickness_g_cm2,
    )

    tables = [
        spectrum_output_table(
            table_id="gcr_proton_total_energy_behind_al",
            title="ГКЛ. Протоны за Al-защитой. Полный энергетический спектр",
            spectrum=result.total,
            metadata=_metadata_with(
                base,
                {
                    "component": "total",
                    "proton_nonelastic_survival": (
                        str(result.nonelastic_survival).lower()
                    ),
                    "secondary_protons": str(result.secondary_protons).lower(),
                },
            ),
        )
    ]

    if include_components:
        tables.extend(
            (
                spectrum_output_table(
                    table_id="gcr_proton_primary_energy_behind_al",
                    title="ГКЛ. Протоны за Al-защитой. Первичный CSDA-спектр",
                    spectrum=result.primary,
                    metadata=_metadata_with(base, {"component": "primary_csda"}),
                ),
                spectrum_output_table(
                    table_id="gcr_proton_primary_survived_energy_behind_al",
                    title=(
                        "ГКЛ. Протоны за Al-защитой. "
                        "Первичный спектр с survival"
                    ),
                    spectrum=result.primary_survived,
                    metadata=_metadata_with(
                        base,
                        {
                            "component": "primary_survived",
                            "proton_nonelastic_survival": (
                                str(result.nonelastic_survival).lower()
                            ),
                        },
                    ),
                ),
                spectrum_output_table(
                    table_id="gcr_proton_secondary_energy_behind_al",
                    title="ГКЛ. Протоны за Al-защитой. Вторичный протонный спектр",
                    spectrum=result.secondary,
                    metadata=_metadata_with(
                        base,
                        {
                            "component": "secondary",
                            "secondary_protons": str(result.secondary_protons).lower(),
                        },
                    ),
                ),
            )
        )

    return tuple(tables)


def gcr_hze_shielding_output_tables(
    results_by_z: Mapping[int, GcrHzeShieldingResult],
) -> tuple[OutputTable, ...]:
    """Return GCR HZE behind-shielding energy spectrum tables by ion charge."""

    tables: list[OutputTable] = []

    for z, result in sorted(results_by_z.items()):
        base = _shielding_metadata(
            spectrum=result.spectrum,
            stage="shielding",
            thickness_g_cm2=result.thickness_g_cm2,
        )
        tables.append(
            spectrum_output_table(
                table_id=f"gcr_hze_z{z}_energy_behind_al",
                title=(
                    f"ГКЛ. ТЗЧ Z={result.z} {result.symbol} за Al-защитой. "
                    "Энергетический спектр"
                ),
                spectrum=result.spectrum,
                metadata=_metadata_with(
                    base,
                    {
                        "component": "hze_primary_csda",
                        "z": str(result.z),
                        "symbol": result.symbol,
                        "heavy_ion_nuclear_attenuation": "not_included",
                        "heavy_ion_fragmentation": "not_included",
                    },
                ),
            )
        )

    return tuple(tables)


def gcr_let_output_tables(
    *,
    result: GcrLetProductsResult,
    thickness_g_cm2: float,
) -> tuple[OutputTable, ...]:
    """Return GCR proton, HZE and combined LET spectrum tables."""

    base = _shielding_metadata(
        spectrum=result.combined_let,
        stage="let",
        thickness_g_cm2=thickness_g_cm2,
    )
    tables: list[OutputTable] = [
        spectrum_output_table(
            table_id="gcr_proton_let_behind_al",
            title="ГКЛ. Протоны за Al-защитой. ЛПЭ-спектр в Si",
            spectrum=result.proton_let,
            metadata=_metadata_with(
                base,
                {
                    "component": "proton_let",
                    "let_target": "Si",
                },
            ),
        )
    ]

    for z, spectrum in sorted(result.hze_let_by_z.items()):
        tables.append(
            spectrum_output_table(
                table_id=f"gcr_hze_z{z}_let_behind_al",
                title=f"ГКЛ. ТЗЧ Z={z} за Al-защитой. ЛПЭ-спектр в Si",
                spectrum=spectrum,
                metadata=_metadata_with(
                    base,
                    {
                        "component": "hze_let",
                        "z": str(z),
                        "let_target": "Si",
                    },
                ),
            )
        )

    tables.append(
        spectrum_output_table(
            table_id="gcr_combined_let_behind_al",
            title="ГКЛ. Суммарный ЛПЭ-спектр за Al-защитой в Si",
            spectrum=result.combined_let,
            metadata=_metadata_with(
                base,
                {
                    "component": "combined_let",
                    "hze_z_values": ",".join(str(z) for z in sorted(result.hze_let_by_z)),
                    "let_target": "Si",
                },
            ),
        )
    )

    return tuple(tables)


def gcr_shielding_let_output_tables(
    *,
    proton: GcrProtonShieldingResult,
    hze_by_z: Mapping[int, GcrHzeShieldingResult],
    let_products: GcrLetProductsResult,
    include_proton_components: bool = False,
) -> tuple[OutputTable, ...]:
    """Return all GCR shielding and LET spectrum output tables."""

    return (
        *gcr_proton_shielding_output_tables(
            proton,
            include_components=include_proton_components,
        ),
        *gcr_hze_shielding_output_tables(hze_by_z),
        *gcr_let_output_tables(
            result=let_products,
            thickness_g_cm2=proton.thickness_g_cm2,
        ),
    )


__all__ = [
    "GCR_OUTPUT_DOSE_SEE_DEFERRED_REASON",
    "GCR_OUTPUT_DOSE_STATUS_NOT_CALCULATED",
    "GCR_OUTPUT_LOCATION_BEHIND_SHIELDING",
    "GCR_OUTPUT_LOCATION_ON_ORBIT",
    "GCR_OUTPUT_LOCATION_OUTSIDE_MAGNETOSPHERE",
    "GCR_OUTPUT_MAIN_COMPONENT_GOST",
    "GCR_OUTPUT_MAIN_COMPONENT_OST",
    "GCR_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL",
    "GCR_OUTPUT_SHIELD_MATERIAL_AL",
    "GCR_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED",
    "gcr_hze_shielding_output_tables",
    "gcr_let_output_tables",
    "gcr_on_orbit_product_output_tables",
    "gcr_proton_shielding_output_tables",
    "gcr_shielding_let_output_tables",
    "gcr_source_spectra_output_tables",
]
