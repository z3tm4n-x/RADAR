from __future__ import annotations

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.dose.inputs import (
    select_erb_electron_shieldose2_inputs,
    select_hze_let_dose_inputs,
    select_proton_dose_inputs,
)


def _energy_product(
    *,
    source: RadiationSource,
    particle: Particle,
    kind: RadiationProductKind = RadiationProductKind.MISSION_FLUENCE,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    model: str = "source_model",
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=Spectrum1D(
            x=(1.0, 10.0, 100.0),
            y=(3.0, 2.0, 1.0),
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUENCE
            if quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
            else Unit.DIFFERENTIAL_FLUX,
            quantity=quantity,
            particle=particle,
            source=source,
            model=model,
        ),
    )


def _let_product(
    *,
    source: RadiationSource,
    kind: RadiationProductKind = RadiationProductKind.MISSION_LET_FLUENCE,
    quantity: SpectrumQuantity = SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    model: str = "let_model",
) -> SpectrumProduct:
    return SpectrumProduct(
        kind=kind,
        spectrum=Spectrum1D(
            x=(0.1, 1.0, 10.0),
            y=(30.0, 20.0, 10.0),
            x_unit=Unit.LET,
            y_unit=Unit.DIFFERENTIAL_LET_FLUENCE
            if quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
            else Unit.DIFFERENTIAL_LET_FLUX,
            quantity=quantity,
            particle=Particle.HZE,
            source=source,
            model=model,
        ),
    )


def test_select_erb_electron_shieldose2_inputs_uses_on_orbit_mission_fluence() -> None:
    on_orbit = _energy_product(
        source=RadiationSource.ERB,
        particle=Particle.ELECTRON,
        model="erb_electron_on_orbit",
    )
    shielded = _energy_product(
        source=RadiationSource.ERB,
        particle=Particle.ELECTRON,
        model="erb_electron_on_orbit+erb_electron_al_shielding_primary_csda",
    )
    proton = _energy_product(
        source=RadiationSource.ERB,
        particle=Particle.PROTON,
        model="erb_proton_on_orbit",
    )
    flux = _energy_product(
        source=RadiationSource.ERB,
        particle=Particle.ELECTRON,
        kind=RadiationProductKind.MEAN_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        model="erb_electron_mean_flux",
    )

    assert select_erb_electron_shieldose2_inputs(
        (on_orbit, shielded, proton, flux),
    ) == (on_orbit,)


def test_select_proton_dose_inputs_uses_only_shielded_proton_mission_fluence() -> None:
    erb = _energy_product(
        source=RadiationSource.ERB,
        particle=Particle.PROTON,
        model="erb_model+erb_proton_al_shielding_primary_survival_secondary",
    )
    sep = _energy_product(
        source=RadiationSource.SEP,
        particle=Particle.PROTON,
        model="sep_model+sep_proton_al_shielding_primary_survival_secondary",
    )
    gcr = _energy_product(
        source=RadiationSource.GCR,
        particle=Particle.PROTON,
        model="gcr_model+gcr_proton_al_shielding_primary_survival_secondary",
    )
    on_orbit = _energy_product(
        source=RadiationSource.SEP,
        particle=Particle.PROTON,
        model="sep_model",
    )
    hze = _energy_product(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        model="gcr_model+gcr_hze_al_shielding_primary_csda",
    )
    let = _let_product(source=RadiationSource.SEP)

    assert select_proton_dose_inputs(
        (on_orbit, erb, hze, sep, let, gcr),
    ) == (erb, sep, gcr)


def test_select_hze_let_dose_inputs_uses_only_sep_gcr_mission_let_fluence() -> None:
    sep = _let_product(source=RadiationSource.SEP)
    gcr = _let_product(source=RadiationSource.GCR)
    erb = _let_product(source=RadiationSource.ERB)
    sep_flux = _let_product(
        source=RadiationSource.SEP,
        kind=RadiationProductKind.MEAN_LET_FLUX,
        quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
        model="sep_let_flux",
    )
    proton_energy = _energy_product(
        source=RadiationSource.SEP,
        particle=Particle.PROTON,
        model="sep_model+sep_proton_al_shielding_primary_survival_secondary",
    )

    assert select_hze_let_dose_inputs(
        (sep, gcr, erb, sep_flux, proton_energy),
    ) == (sep, gcr)
