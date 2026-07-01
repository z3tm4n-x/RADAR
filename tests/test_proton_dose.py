from __future__ import annotations

import pytest

from radar.core.products import SpectrumProduct
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.dose.constants import MEV_PER_MG_TO_MEV_PER_G, MEV_TO_RAD
from radar.dose.proton import (
    calculate_proton_dose_from_shielded_fluence,
    integrate_proton_energy_deposition_MeV_per_mg,
)
from radar.shielding.proton_si_let import ProtonSiLetTable


def _table() -> ProtonSiLetTable:
    return ProtonSiLetTable(
        energy_mev=(0.5, 1.0, 4.0, 8.0),
        let_total_mev_cm2_mg=(3.0, 3.0, 3.0, 3.0),
        let_electronic_mev_cm2_mg=(2.0, 2.0, 2.0, 2.0),
        let_nuclear_mev_cm2_mg=(1.0, 1.0, 1.0, 1.0),
    )


def _shielding_marker(source: RadiationSource) -> str:
    if source is RadiationSource.ERB:
        return "erb_proton_al_shielding_primary_survival_secondary"
    if source is RadiationSource.SEP:
        return "sep_proton_al_shielding_primary_survival_secondary"
    if source is RadiationSource.GCR:
        return "gcr_proton_al_shielding_primary_survival_secondary"
    raise AssertionError(source)


def _proton_product(
    *,
    source: RadiationSource = RadiationSource.SEP,
    particle: Particle = Particle.PROTON,
    kind: RadiationProductKind = RadiationProductKind.MISSION_FLUENCE,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    x_unit: Unit = Unit.MEV,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
    model: str | None = None,
    y: tuple[float, ...] = (10.0, 0.0),
) -> SpectrumProduct:
    marker = _shielding_marker(source)
    return SpectrumProduct(
        kind=kind,
        spectrum=Spectrum1D(
            x=(1.0, 4.0),
            y=y,
            x_unit=x_unit,
            y_unit=y_unit,
            quantity=quantity,
            particle=particle,
            source=source,
            model=model if model is not None else f"source_model+{marker}",
        ),
    )


def test_calculates_proton_dose_from_shielded_fluence() -> None:
    product = _proton_product()

    # Energy centers 1 and 4 have logarithmic bin edges 0.5, 2, 8.
    # Only the first bin is nonzero:
    # 10 cm^-2 MeV^-1 * 2 MeV cm^2/mg * (2 - 0.5) MeV = 30 MeV/mg.
    expected_MeV_per_mg = 30.0
    expected_MeV_per_g = expected_MeV_per_mg * MEV_PER_MG_TO_MEV_PER_G

    result = calculate_proton_dose_from_shielded_fluence(
        product,
        table=_table(),
    )

    assert integrate_proton_energy_deposition_MeV_per_mg(
        product=product,
        table=_table(),
    ) == pytest.approx(expected_MeV_per_mg)
    assert result.energy_deposition_MeV_per_mg == pytest.approx(expected_MeV_per_mg)
    assert result.energy_deposition_MeV_per_g == pytest.approx(expected_MeV_per_g)
    assert result.dose_rad == pytest.approx(expected_MeV_per_g * MEV_TO_RAD)
    assert result.included_in_total is True


def test_zero_proton_fluence_gives_zero_dose() -> None:
    product = _proton_product(y=(0.0, 0.0))

    result = calculate_proton_dose_from_shielded_fluence(
        product,
        table=_table(),
    )

    assert result.energy_deposition_MeV_per_mg == 0.0
    assert result.energy_deposition_MeV_per_g == 0.0
    assert result.dose_rad == 0.0


def test_proton_dose_scales_linearly_with_fluence() -> None:
    product = _proton_product(y=(10.0, 2.0))
    doubled = _proton_product(y=(20.0, 4.0))

    dose = calculate_proton_dose_from_shielded_fluence(product, table=_table())
    doubled_dose = calculate_proton_dose_from_shielded_fluence(
        doubled,
        table=_table(),
    )

    assert doubled_dose.dose_rad == pytest.approx(2.0 * dose.dose_rad)


@pytest.mark.parametrize(
    "source",
    (RadiationSource.ERB, RadiationSource.SEP, RadiationSource.GCR),
)
def test_accepts_supported_shielded_proton_sources(source: RadiationSource) -> None:
    product = _proton_product(source=source)

    result = calculate_proton_dose_from_shielded_fluence(
        product,
        table=_table(),
    )

    assert result.dose_rad > 0.0


def test_rejects_on_orbit_proton_spectrum() -> None:
    product = _proton_product(model="source_model")

    with pytest.raises(ValueError, match="behind aluminium shielding"):
        calculate_proton_dose_from_shielded_fluence(product, table=_table())


def test_rejects_non_proton_product() -> None:
    product = _proton_product(particle=Particle.HZE)

    with pytest.raises(ValueError, match="behind aluminium shielding"):
        calculate_proton_dose_from_shielded_fluence(product, table=_table())


def test_rejects_flux_product() -> None:
    product = _proton_product(
        kind=RadiationProductKind.MEAN_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
    )

    with pytest.raises(ValueError, match="behind aluminium shielding"):
        calculate_proton_dose_from_shielded_fluence(product, table=_table())


def test_rejects_let_spectrum() -> None:
    product = SpectrumProduct(
        kind=RadiationProductKind.MISSION_LET_FLUENCE,
        spectrum=Spectrum1D(
            x=(0.1, 1.0),
            y=(10.0, 1.0),
            x_unit=Unit.LET,
            y_unit=Unit.DIFFERENTIAL_LET_FLUENCE,
            quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
            particle=Particle.HZE,
            source=RadiationSource.SEP,
            model="combined_proton_hze_let",
        ),
    )

    with pytest.raises(ValueError, match="behind aluminium shielding"):
        calculate_proton_dose_from_shielded_fluence(product, table=_table())


def test_rejects_energy_outside_stopping_table() -> None:
    product = _proton_product()

    short_table = ProtonSiLetTable(
        energy_mev=(2.0, 8.0),
        let_total_mev_cm2_mg=(3.0, 3.0),
        let_electronic_mev_cm2_mg=(2.0, 2.0),
        let_nuclear_mev_cm2_mg=(1.0, 1.0),
    )

    with pytest.raises(ValueError, match="outside the tabulated range"):
        calculate_proton_dose_from_shielded_fluence(product, table=short_table)
