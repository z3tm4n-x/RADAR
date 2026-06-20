"""Allowed radiation products by natural radiation source."""

from __future__ import annotations

from radar.core.products import SpectrumProduct
from radar.core.types import RadiationProductKind, RadiationSource

SOURCE_PRODUCT_KINDS: dict[RadiationSource, tuple[RadiationProductKind, ...]] = {
    RadiationSource.SEP: (
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.PEAK_FLUX,
    ),
    RadiationSource.GCR: (
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MISSION_LET_FLUENCE,
        RadiationProductKind.MEAN_LET_FLUX,
        RadiationProductKind.MAXIMUM_LET_FLUX,
    ),
    RadiationSource.ERB: (
        RadiationProductKind.ORBIT_AVERAGED_FLUX,
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.PEAK_FLUX,
    ),
}


def allowed_product_kinds_for_source(
    source: RadiationSource,
) -> tuple[RadiationProductKind, ...]:
    """Return product kinds allowed for a radiation source."""

    try:
        return SOURCE_PRODUCT_KINDS[source]
    except KeyError as exc:
        msg = f"Unsupported radiation source: {source}"
        raise ValueError(msg) from exc


def is_product_kind_allowed_for_source(
    product_kind: RadiationProductKind,
    source: RadiationSource,
) -> bool:
    """Return True if product kind is allowed for a radiation source."""

    return product_kind in allowed_product_kinds_for_source(source)


def validate_product_kind_allowed_for_source(
    product_kind: RadiationProductKind,
    source: RadiationSource,
) -> None:
    """Validate that a product kind is allowed for a radiation source."""

    if is_product_kind_allowed_for_source(
        product_kind=product_kind,
        source=source,
    ):
        return

    allowed_values = ", ".join(
        kind.value for kind in allowed_product_kinds_for_source(source)
    )
    msg = (
        f"Radiation product {product_kind.value} is not allowed for "
        f"radiation source {source.value}. Allowed products: {allowed_values}."
    )
    raise ValueError(msg)


def validate_product_allowed_for_source(
    product: SpectrumProduct,
    source: RadiationSource,
) -> None:
    """Validate that product kind and spectrum source match a radiation source."""

    validate_product_kind_allowed_for_source(
        product_kind=product.kind,
        source=source,
    )

    if product.spectrum.source is source:
        return

    msg = (
        f"Radiation product spectrum source {product.spectrum.source.value} "
        f"does not match expected radiation source {source.value}."
    )
    raise ValueError(msg)
