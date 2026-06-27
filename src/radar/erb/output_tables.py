"""ERB output table builders.

Dose, shielding, bremsstrahlung, and single-event-effect output tables are
intentionally not constructed here. They are deferred until the corresponding
physical models are available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from radar.core.products import SpectrumProduct
from radar.core.source_products import validate_product_allowed_for_source
from radar.core.types import Particle, RadiationSource
from radar.erb.shielding import ErbProtonShieldingResult
from radar.output_tables import OutputTable, spectrum_output_table

ERB_OUTPUT_LOCATION_ON_ORBIT = "on_orbit"
ERB_OUTPUT_LOCATION_BEHIND_SHIELDING = "behind_shielding"
ERB_OUTPUT_SHIELDING_NOT_APPLIED = "not_applied"
ERB_OUTPUT_SHIELDING_APPLIED = "applied"
ERB_OUTPUT_SHIELD_MATERIAL_AL = "Al"
ERB_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL = "centered_spherical_shell"
ERB_OUTPUT_DOSE_STATUS_NOT_CALCULATED = "not_calculated"
ERB_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED = "not_calculated"
ERB_OUTPUT_DOSE_SEE_DEFERRED_REASON = "deferred_until_erb_shielding_dose_see"

_PARTICLE_TITLES: dict[Particle, str] = {
    Particle.PROTON: "\u041f\u0440\u043e\u0442\u043e\u043d\u044b",
    Particle.ELECTRON: "\u042d\u043b\u0435\u043a\u0442\u0440\u043e\u043d\u044b",
}


def _table_id_token(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("+", "_")
        .replace(":", "_")
    )


def _particle_title(particle: Particle) -> str:
    try:
        return _PARTICLE_TITLES[particle]
    except KeyError as exc:
        msg = f"Unsupported ERB output particle: {particle!r}."
        raise ValueError(msg) from exc


def _metadata_with(
    base: Mapping[str, str],
    extra: Mapping[str, str],
) -> dict[str, str]:
    metadata = dict(base)
    metadata.update(extra)
    return metadata


def _product_metadata(product: SpectrumProduct) -> dict[str, str]:
    spectrum = product.spectrum

    return {
        "stage": "source_model",
        "location": ERB_OUTPUT_LOCATION_ON_ORBIT,
        "shielding": ERB_OUTPUT_SHIELDING_NOT_APPLIED,
        "product_kind": product.kind.value,
        "spectrum_quantity": spectrum.quantity.value,
        "dose_status": ERB_OUTPUT_DOSE_STATUS_NOT_CALCULATED,
        "single_event_effects_status": (
            ERB_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED
        ),
        "dose_see_dependency": ERB_OUTPUT_DOSE_SEE_DEFERRED_REASON,
    }


def _product_table_id(product: SpectrumProduct) -> str:
    spectrum = product.spectrum

    return (
        f"erb_{_table_id_token(spectrum.particle.value)}_"
        f"{_table_id_token(product.kind.value)}_"
        f"{_table_id_token(spectrum.quantity.value)}_on_orbit"
    )


def erb_product_output_tables(
    products: Sequence[SpectrumProduct],
) -> tuple[OutputTable, ...]:
    """Return ERB on-orbit product spectrum tables before shielding."""

    if not products:
        msg = "ERB output tables require at least one product."
        raise ValueError(msg)

    tables: list[OutputTable] = []

    for product in products:
        validate_product_allowed_for_source(
            product=product,
            source=RadiationSource.ERB,
        )

        spectrum = product.spectrum
        tables.append(
            spectrum_output_table(
                table_id=_product_table_id(product),
                title=(
                    f"\u0415\u0420\u041f\u0417. {_particle_title(spectrum.particle)}. "
                    f"{product.kind.value}. \u0421\u043f\u0435\u043a\u0442\u0440 "
                    f"\u043d\u0430 \u043e\u0440\u0431\u0438\u0442\u0435 "
                    f"\u0434\u043e \u0437\u0430\u0449\u0438\u0442\u044b"
                ),
                spectrum=spectrum,
                metadata=_metadata_with(
                    _product_metadata(product),
                    {
                        "shielding_material": "none",
                    },
                ),
            )
        )

    return tuple(tables)


def _format_float(value: float) -> str:
    if not isfinite(value):
        msg = "ERB output table metadata float value must be finite."
        raise ValueError(msg)

    return f"{value:.12g}"


def _shielding_metadata(
    *,
    result: ErbProtonShieldingResult,
    component: str,
) -> dict[str, str]:
    return {
        "stage": "shielding",
        "location": ERB_OUTPUT_LOCATION_BEHIND_SHIELDING,
        "shielding": ERB_OUTPUT_SHIELDING_APPLIED,
        "shield_material": ERB_OUTPUT_SHIELD_MATERIAL_AL,
        "shield_geometry": ERB_OUTPUT_SHIELD_GEOMETRY_CENTERED_SPHERICAL_SHELL,
        "thickness_g_cm2": _format_float(result.thickness_g_cm2),
        "component": component,
        "nonelastic_survival": str(result.nonelastic_survival).lower(),
        "secondary_protons": str(result.secondary_protons).lower(),
        "dose_status": ERB_OUTPUT_DOSE_STATUS_NOT_CALCULATED,
        "single_event_effects_status": (
            ERB_OUTPUT_SINGLE_EVENT_EFFECTS_STATUS_NOT_CALCULATED
        ),
        "dose_see_dependency": ERB_OUTPUT_DOSE_SEE_DEFERRED_REASON,
    }


def erb_proton_shielding_output_tables(
    result: ErbProtonShieldingResult,
    *,
    include_components: bool = False,
) -> tuple[OutputTable, ...]:
    """Return ERB proton behind-shielding energy spectrum tables."""

    tables = [
        spectrum_output_table(
            table_id="erb_proton_total_energy_behind_al",
            title=(
                "\u0415\u0420\u041f\u0417. \u041f\u0440\u043e\u0442\u043e\u043d\u044b "
                "\u0437\u0430 Al-\u0437\u0430\u0449\u0438\u0442\u043e\u0439. "
                "\u041f\u043e\u043b\u043d\u044b\u0439 "
                "\u044d\u043d\u0435\u0440\u0433\u0435\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u0439 "
                "\u0441\u043f\u0435\u043a\u0442\u0440"
            ),
            spectrum=result.total,
            metadata=_shielding_metadata(result=result, component="total"),
        )
    ]

    if include_components:
        tables.extend(
            (
                spectrum_output_table(
                    table_id="erb_proton_primary_energy_behind_al",
                    title=(
                        "\u0415\u0420\u041f\u0417. \u041f\u0440\u043e\u0442\u043e\u043d\u044b "
                        "\u0437\u0430 Al-\u0437\u0430\u0449\u0438\u0442\u043e\u0439. "
                        "\u041f\u0435\u0440\u0432\u0438\u0447\u043d\u044b\u0439 "
                        "CSDA-\u0441\u043f\u0435\u043a\u0442\u0440"
                    ),
                    spectrum=result.primary,
                    metadata=_shielding_metadata(result=result, component="primary_csda"),
                ),
                spectrum_output_table(
                    table_id="erb_proton_primary_survived_energy_behind_al",
                    title=(
                        "\u0415\u0420\u041f\u0417. \u041f\u0440\u043e\u0442\u043e\u043d\u044b "
                        "\u0437\u0430 Al-\u0437\u0430\u0449\u0438\u0442\u043e\u0439. "
                        "\u041f\u0435\u0440\u0432\u0438\u0447\u043d\u044b\u0439 "
                        "\u0441\u043f\u0435\u043a\u0442\u0440 "
                        "\u0441 survival"
                    ),
                    spectrum=result.primary_survived,
                    metadata=_shielding_metadata(
                        result=result,
                        component="primary_survived",
                    ),
                ),
                spectrum_output_table(
                    table_id="erb_proton_secondary_energy_behind_al",
                    title=(
                        "\u0415\u0420\u041f\u0417. \u041f\u0440\u043e\u0442\u043e\u043d\u044b "
                        "\u0437\u0430 Al-\u0437\u0430\u0449\u0438\u0442\u043e\u0439. "
                        "\u0412\u0442\u043e\u0440\u0438\u0447\u043d\u044b\u0439 "
                        "\u0441\u043f\u0435\u043a\u0442\u0440"
                    ),
                    spectrum=result.secondary,
                    metadata=_shielding_metadata(result=result, component="secondary"),
                ),
            )
        )

    return tuple(tables)

