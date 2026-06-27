"""SEP output table builders.

Dose and single-event-effect output tables are intentionally not constructed
here. They are deferred until dose/SEE physical models are available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle
from radar.output_tables import OutputTable, spectrum_output_table
from radar.sep.pipeline_shielding import (
    SepHzeShieldingResult,
    SepLetProductsResult,
)
from radar.shielding.sep_proton import SepProtonShieldingResult

SEP_OUTPUT_LOCATION_ON_ORBIT = "on_orbit"
SEP_OUTPUT_LOCATION_BEHIND_SHIELDING = "behind_shielding"

SEP_OUTPUT_SHIELD_MATERIAL_AL = "Al"
SEP_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL = "centered_spherical_shell"

SEP_OUTPUT_DOSE_STATUS_NOT_CALCULATED = "not_calculated"
SEP_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED = "not_calculated"
SEP_OUTPUT_DOSE_SEE_DEFERRED_REASON = "deferred_until_sep_dose_see"


def _format_float(value: float) -> str:
    if not isfinite(value):
        msg = "SEP output table metadata float value must be finite."
        raise ValueError(msg)

    return f"{value:.12g}"


def _metadata_with(
    base: Mapping[str, str],
    extra: Mapping[str, str],
) -> dict[str, str]:
    metadata = dict(base)
    metadata.update(extra)
    return metadata


def _source_model_from_spectrum_model(model: str) -> str:
    source_model = model.split(":", maxsplit=1)[0].split("+", maxsplit=1)[0]

    if not source_model:
        msg = "Cannot infer SEP source model from empty spectrum model."
        raise ValueError(msg)

    return source_model


def _sep_hze_tag_from_spectrum_model(spectrum: Spectrum1D) -> str:
    parts = spectrum.model.split(":")

    if len(parts) < 3 or parts[1] != "hze" or not parts[2]:
        msg = "SEP HZE output spectrum model must contain an hze ion tag."
        raise ValueError(msg)

    return parts[2].split("+", maxsplit=1)[0]


def _sep_hze_z_symbol_from_spectrum_model(spectrum: Spectrum1D) -> tuple[int, str]:
    tag = _sep_hze_tag_from_spectrum_model(spectrum)

    if not tag.startswith("z"):
        msg = "SEP HZE output ion tag must start with 'z'."
        raise ValueError(msg)

    digit_chars: list[str] = []
    rest_start = 1

    for index, character in enumerate(tag[1:], start=1):
        if not character.isdigit():
            rest_start = index
            break

        digit_chars.append(character)
    else:
        rest_start = len(tag)

    if not digit_chars:
        msg = "SEP HZE output ion tag must contain a charge number."
        raise ValueError(msg)

    symbol = tag[rest_start:].lstrip("_")

    if not symbol:
        symbol = f"z{int(''.join(digit_chars))}"

    return int("".join(digit_chars)), symbol.capitalize()


def _particle_token(spectrum: Spectrum1D) -> str:
    if spectrum.particle is Particle.HZE:
        z, _symbol = _sep_hze_z_symbol_from_spectrum_model(spectrum)
        return f"hze_z{z}"

    return spectrum.particle.value


def _common_metadata(
    *,
    spectrum: Spectrum1D,
    stage: str,
    location: str,
    shielding: str,
) -> dict[str, str]:
    return {
        "stage": stage,
        "location": location,
        "source_model": _source_model_from_spectrum_model(spectrum.model),
        "shielding": shielding,
        "dose_status": SEP_OUTPUT_DOSE_STATUS_NOT_CALCULATED,
        "single_event_effects_status": (
            SEP_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED
        ),
        "dose_see_dependency": SEP_OUTPUT_DOSE_SEE_DEFERRED_REASON,
    }


def _on_orbit_product_table_token(product: SpectrumProduct) -> str:
    return product.kind.value


def sep_on_orbit_product_output_tables(
    products: Sequence[SpectrumProduct],
) -> tuple[OutputTable, ...]:
    """Return SEP source/on-orbit product tables before shielding."""

    if not products:
        msg = "SEP on-orbit output tables require at least one product."
        raise ValueError(msg)

    tables: list[OutputTable] = []

    for product in products:
        spectrum = product.spectrum
        particle_token = _particle_token(spectrum)
        base = _common_metadata(
            spectrum=spectrum,
            stage="source_model",
            location=SEP_OUTPUT_LOCATION_ON_ORBIT,
            shielding="not_applied",
        )
        product_token = _on_orbit_product_table_token(product)
        extra = {
            "product_kind": product.kind.value,
            "spectrum_quantity": spectrum.quantity.value,
            "particle_group": spectrum.particle.value,
        }

        if spectrum.particle is Particle.HZE:
            z, symbol = _sep_hze_z_symbol_from_spectrum_model(spectrum)
            extra.update({"z": str(z), "symbol": symbol})

        tables.append(
            spectrum_output_table(
                table_id=f"sep_{particle_token}_{product_token}_on_orbit",
                title=(
                    f"???. {particle_token}. {product.kind.value}. "
                    "?????? ?? ??????"
                ),
                spectrum=spectrum,
                metadata=_metadata_with(base, extra),
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
            location=SEP_OUTPUT_LOCATION_BEHIND_SHIELDING,
            shielding="applied",
        ),
        {
            "shield_material": SEP_OUTPUT_SHIELD_MATERIAL_AL,
            "shield_geometry": SEP_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL,
            "thickness_g_cm2": _format_float(thickness_g_cm2),
        },
    )


def sep_proton_shielding_output_tables(
    result: SepProtonShieldingResult,
    *,
    include_components: bool = False,
) -> tuple[OutputTable, ...]:
    """Return SEP proton behind-shielding energy spectrum tables."""

    base = _shielding_metadata(
        spectrum=result.total,
        stage="shielding",
        thickness_g_cm2=result.thickness_g_cm2,
    )

    tables = [
        spectrum_output_table(
            table_id="sep_proton_total_energy_behind_al",
            title="???. ??????? ?? Al-???????. ?????? ?????????????? ??????",
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
                    table_id="sep_proton_primary_energy_behind_al",
                    title="???. ??????? ?? Al-???????. ????????? CSDA-??????",
                    spectrum=result.primary,
                    metadata=_metadata_with(base, {"component": "primary_csda"}),
                ),
                spectrum_output_table(
                    table_id="sep_proton_primary_survived_energy_behind_al",
                    title=(
                        "???. ??????? ?? Al-???????. "
                        "????????? ?????? ? survival"
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
                    table_id="sep_proton_secondary_energy_behind_al",
                    title="???. ??????? ?? Al-???????. ????????? ????????? ??????",
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
        base = _shielding_metadata(
            spectrum=result.spectrum,
            stage="shielding",
            thickness_g_cm2=result.thickness_g_cm2,
        )
        tables.append(
            spectrum_output_table(
                table_id=f"sep_hze_z{z}_energy_behind_al",
                title=(
                    f"???. ??? Z={result.z} {result.symbol} ?? Al-???????. "
                    "?????????????? ??????"
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

    base = _shielding_metadata(
        spectrum=result.combined_let,
        stage="let",
        thickness_g_cm2=thickness_g_cm2,
    )
    tables: list[OutputTable] = [
        spectrum_output_table(
            table_id="sep_proton_let_behind_al",
            title="???. ??????? ?? Al-???????. ???-?????? ? Si",
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
                title=f"???. ??? Z={z} ?? Al-???????. ???-?????? ? Si",
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
            title="???. ????????? ???-?????? ?? Al-??????? ? Si",
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
    "SEP_OUTPUT_LOCATION_ON_ORBIT",
    "SEP_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL",
    "SEP_OUTPUT_SHIELD_MATERIAL_AL",
    "SEP_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED",
    "sep_hze_shielding_output_tables",
    "sep_let_output_tables",
    "sep_on_orbit_product_output_tables",
    "sep_proton_shielding_output_tables",
    "sep_shielding_let_output_tables",
]
