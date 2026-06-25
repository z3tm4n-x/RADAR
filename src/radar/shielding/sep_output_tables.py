"""SEP shielding and LET output table builders.

Dose and single-event-effect output tables are intentionally not constructed
here. They are deferred until GCR and ERB physical models are available.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from radar.output_tables import OutputTable, spectrum_output_table
from radar.shielding.sep_hze import SepHzeShieldingResult
from radar.shielding.sep_let import SepLetProductsResult
from radar.shielding.sep_proton import SepProtonShieldingResult

SEP_OUTPUT_LOCATION_BEHIND_SHIELDING = "behind_shielding"
SEP_OUTPUT_SHIELD_MATERIAL_AL = "Al"
SEP_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL = "centered_spherical_shell"
SEP_OUTPUT_DOSE_STATUS_NOT_CALCULATED = "not_calculated"
SEP_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED = "not_calculated"
SEP_OUTPUT_DOSE_SEE_DEFERRED_REASON = "deferred_until_gcr_erb_models"


def _format_float(value: float) -> str:
    if not isfinite(value):
        msg = "SEP output table metadata float value must be finite."
        raise ValueError(msg)

    return f"{value:.12g}"


def _common_metadata(*, stage: str, thickness_g_cm2: float) -> dict[str, str]:
    return {
        "stage": stage,
        "location": SEP_OUTPUT_LOCATION_BEHIND_SHIELDING,
        "shield_material": SEP_OUTPUT_SHIELD_MATERIAL_AL,
        "shield_geometry": SEP_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL,
        "thickness_g_cm2": _format_float(thickness_g_cm2),
        "dose_status": SEP_OUTPUT_DOSE_STATUS_NOT_CALCULATED,
        "single_event_effects_status": SEP_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED,
        "dose_see_dependency": SEP_OUTPUT_DOSE_SEE_DEFERRED_REASON,
    }


def _metadata_with(
    base: Mapping[str, str],
    extra: Mapping[str, str],
) -> dict[str, str]:
    metadata = dict(base)
    metadata.update(extra)
    return metadata


def sep_proton_shielding_output_tables(
    result: SepProtonShieldingResult,
    *,
    include_components: bool = False,
) -> tuple[OutputTable, ...]:
    """Return SEP proton behind-shielding energy spectrum tables."""

    base = _common_metadata(
        stage="shielding",
        thickness_g_cm2=result.thickness_g_cm2,
    )

    tables = [
        spectrum_output_table(
            table_id="sep_proton_total_energy_behind_al",
            title="СКЛ. Протоны за Al-защитой. Полный энергетический спектр",
            spectrum=result.total,
            metadata=_metadata_with(
                base,
                {
                    "component": "total",
                    "proton_nonelastic_survival": str(result.nonelastic_survival).lower(),
                    "secondary_protons": str(result.secondary_protons).lower(),
                },
            ),
        )
    ]

    if include_components:
        tables.extend(
            (
                spectrum_output_table(
                    table_id="sep_proton_primary_energy_behind_al",
                    title="СКЛ. Протоны за Al-защитой. Первичный CSDA-спектр",
                    spectrum=result.primary,
                    metadata=_metadata_with(base, {"component": "primary_csda"}),
                ),
                spectrum_output_table(
                    table_id="sep_proton_primary_survived_energy_behind_al",
                    title="СКЛ. Протоны за Al-защитой. Первичный спектр с survival",
                    spectrum=result.primary_survived,
                    metadata=_metadata_with(
                        base,
                        {
                            "component": "primary_survived",
                            "proton_nonelastic_survival": str(result.nonelastic_survival).lower(),
                        },
                    ),
                ),
                spectrum_output_table(
                    table_id="sep_proton_secondary_energy_behind_al",
                    title="СКЛ. Протоны за Al-защитой. Вторичный протонный спектр",
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


def sep_hze_shielding_output_tables(
    results_by_z: Mapping[int, SepHzeShieldingResult],
) -> tuple[OutputTable, ...]:
    """Return SEP HZE behind-shielding energy spectrum tables by ion charge."""

    tables: list[OutputTable] = []

    for z, result in sorted(results_by_z.items()):
        base = _common_metadata(
            stage="shielding",
            thickness_g_cm2=result.thickness_g_cm2,
        )
        tables.append(
            spectrum_output_table(
                table_id=f"sep_hze_z{z}_energy_behind_al",
                title=(
                    f"СКЛ. ТЗЧ Z={result.z} {result.symbol} за Al-защитой. "
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


def sep_let_output_tables(
    *,
    result: SepLetProductsResult,
    thickness_g_cm2: float,
) -> tuple[OutputTable, ...]:
    """Return SEP proton, HZE and combined LET spectrum tables."""

    base = _common_metadata(
        stage="let",
        thickness_g_cm2=thickness_g_cm2,
    )
    tables: list[OutputTable] = [
        spectrum_output_table(
            table_id="sep_proton_let_behind_al",
            title="СКЛ. Протоны за Al-защитой. ЛПЭ-спектр в Si",
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
                table_id=f"sep_hze_z{z}_let_behind_al",
                title=f"СКЛ. ТЗЧ Z={z} за Al-защитой. ЛПЭ-спектр в Si",
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
            table_id="sep_combined_let_behind_al",
            title="СКЛ. Суммарный ЛПЭ-спектр за Al-защитой в Si",
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


def sep_shielding_let_output_tables(
    *,
    proton: SepProtonShieldingResult,
    hze_by_z: Mapping[int, SepHzeShieldingResult],
    let_products: SepLetProductsResult,
    include_proton_components: bool = False,
) -> tuple[OutputTable, ...]:
    """Return all SEP shielding and LET spectrum output tables."""

    return (
        *sep_proton_shielding_output_tables(
            proton,
            include_components=include_proton_components,
        ),
        *sep_hze_shielding_output_tables(hze_by_z),
        *sep_let_output_tables(
            result=let_products,
            thickness_g_cm2=proton.thickness_g_cm2,
        ),
    )


__all__ = [
    "SEP_OUTPUT_DOSE_SEE_DEFERRED_REASON",
    "SEP_OUTPUT_DOSE_STATUS_NOT_CALCULATED",
    "SEP_OUTPUT_LOCATION_BEHIND_SHIELDING",
    "SEP_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL",
    "SEP_OUTPUT_SHIELD_MATERIAL_AL",
    "SEP_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED",
    "sep_hze_shielding_output_tables",
    "sep_let_output_tables",
    "sep_proton_shielding_output_tables",
    "sep_shielding_let_output_tables",
]
