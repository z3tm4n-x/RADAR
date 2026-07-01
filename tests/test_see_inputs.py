from __future__ import annotations

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.see.inputs import (
    ERB_PROTON_SEE_MARKER,
    GCR_PROTON_SEE_MARKER,
    SEP_PROTON_SEE_MARKER,
    select_heavy_ion_let_see_inputs,
    select_proton_see_inputs,
    select_single_event_effect_inputs,
)


def _proton_marker(source: RadiationSource) -> str:
    if source is RadiationSource.ERB:
        return ERB_PROTON_SEE_MARKER
    if source is RadiationSource.SEP:
        return SEP_PROTON_SEE_MARKER
    if source is RadiationSource.GCR:
        return GCR_PROTON_SEE_MARKER
    raise AssertionError(source)


def _proton_product(
    source: RadiationSource,
    *,
    model_marker: str | None = None,
    particle: Particle = Particle.PROTON,
    kind: RadiationProductKind = RadiationProductKind.MISSION_FLUENCE,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
) -> SpectrumProduct:
    marker = _proton_marker(source) if model_marker is None else model_marker

    return SpectrumProduct(
        kind=kind,
        spectrum=Spectrum1D(
            x=(1.0, 10.0),
            y=(100.0, 10.0),
            x_unit=Unit.MEV,
            y_unit=y_unit,
            quantity=quantity,
            particle=particle,
            source=source,
            model=f"{source.value}_source+{marker}" if marker else f"{source.value}_source",
        ),
    )


def _let_product(
    source: RadiationSource,
    *,
    kind: RadiationProductKind = RadiationProductKind.MISSION_LET_FLUENCE,
    quantity: SpectrumQuantity = SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    particle: Particle = Particle.HZE,
    y_unit: Unit = Unit.DIFFERENTIAL_LET_FLUENCE,
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=Spectrum1D(
            x=(1.0, 10.0),
            y=(100.0, 10.0),
            x_unit=Unit.LET,
            y_unit=y_unit,
            quantity=quantity,
            particle=particle,
            source=source,
            model=f"{source.value}_combined_let",
        ),
    )


def test_selects_sep_gcr_erb_shielded_proton_see_inputs() -> None:
    products = (
        _proton_product(RadiationSource.SEP),
        _proton_product(RadiationSource.GCR),
        _proton_product(RadiationSource.ERB),
        _proton_product(RadiationSource.SEP, model_marker=""),
    )

    selected = select_proton_see_inputs(products)

    assert selected == products[:3]


def test_rejects_non_proton_and_non_mission_fluence_products() -> None:
    products = (
        _proton_product(RadiationSource.SEP, particle=Particle.ELECTRON),
        _proton_product(
            RadiationSource.SEP,
            kind=RadiationProductKind.MEAN_FLUX,
            quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_FLUX,
        ),
        _proton_product(RadiationSource.SEP),
    )

    assert select_proton_see_inputs(products) == (products[2],)


def test_selects_sep_gcr_heavy_ion_let_see_inputs() -> None:
    products = (
        _let_product(RadiationSource.SEP),
        _let_product(RadiationSource.GCR),
        _let_product(RadiationSource.ERB),
    )

    selected = select_heavy_ion_let_see_inputs(products)

    assert selected == products[:2]


def test_rejects_non_hze_or_non_mission_let_products() -> None:
    products = (
        _let_product(RadiationSource.SEP, particle=Particle.PROTON),
        _let_product(
            RadiationSource.SEP,
            kind=RadiationProductKind.MEAN_LET_FLUX,
            quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
            y_unit=Unit.DIFFERENTIAL_LET_FLUX,
        ),
        _let_product(RadiationSource.SEP),
    )

    assert select_heavy_ion_let_see_inputs(products) == (products[2],)


def test_selects_combined_single_event_effect_inputs() -> None:
    products = (
        _proton_product(RadiationSource.SEP),
        _proton_product(RadiationSource.GCR),
        _proton_product(RadiationSource.ERB),
        _let_product(RadiationSource.SEP),
        _let_product(RadiationSource.GCR),
        _let_product(RadiationSource.ERB),
    )

    selected = select_single_event_effect_inputs(products)

    assert selected.proton_products == products[:3]
    assert selected.heavy_ion_let_products == products[3:5]
