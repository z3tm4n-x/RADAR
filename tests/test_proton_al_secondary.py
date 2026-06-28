from pathlib import Path

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.proton_al import ProtonAlRangeEnergyTable
from radar.shielding.proton_al_secondary import (
    Al27CrossSectionTable,
    SecondaryProtonKernel,
    SecondaryProtonProductDistribution,
    calculate_secondary_proton_spectrum_through_al,
    combine_primary_and_secondary_proton_spectra,
    load_secondary_proton_kernel,
)
from radar.shielding.proton_al_survival import apply_proton_nonelastic_survival_to_primary_spectrum
from radar.shielding.proton_al_survival import Al27NonelasticCrossSectionTable


def _stopping_table() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
        range_g_cm2=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
        stopping_mev_cm2_g=(100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0),
    )


def _spectrum() -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="sep_proton_orbit_spectrum",
    )


def _kernel(*, sigma_barn: float = 1000.0) -> SecondaryProtonKernel:
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(1.0, 1000.0),
        yield_value=(1.0, 1.0),
        emitted_energy_mev_by_incident=((1.0, 1000.0), (1.0, 1000.0)),
        pdf_per_mev_by_incident=((1.0e-3, 1.0e-3), (1.0e-3, 1.0e-3)),
    )
    xs = Al27CrossSectionTable(
        mt=5,
        energy_mev=(1.0, 1000.0),
        sigma_barn=(sigma_barn, sigma_barn),
    )
    return SecondaryProtonKernel(
        products=(product,),
        cross_sections_by_mt={5: xs},
    )


def test_secondary_product_interpolates_yield_and_pdf() -> None:
    product = SecondaryProtonProductDistribution(
        mt=5,
        product_index=0,
        incident_energy_mev=(10.0, 20.0),
        yield_value=(1.0, 3.0),
        emitted_energy_mev_by_incident=((1.0, 10.0), (1.0, 10.0)),
        pdf_per_mev_by_incident=((2.0, 4.0), (6.0, 8.0)),
    )

    assert product.yield_at_incident_energy(15.0) == pytest.approx(2.0)
    assert product.pdf_at(incident_energy_mev=15.0, emitted_energy_mev=5.5) == pytest.approx(5.0)


def test_secondary_spectrum_is_zero_for_zero_thickness() -> None:
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=0.0,
    )

    assert secondary.y == (0.0, 0.0, 0.0)


def test_secondary_spectrum_is_nonnegative_and_nonzero_for_positive_kernel() -> None:
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=1.0,
        depth_steps=4,
    )

    assert all(value >= 0.0 for value in secondary.y)
    assert any(value > 0.0 for value in secondary.y)


def test_secondary_spectrum_skips_below_minimum_residual_range() -> None:
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(),
        thickness_g_cm2=1.0,
        depth_steps=10,
    )

    assert all(value >= 0.0 for value in secondary.y)


def test_secondary_spectrum_is_zero_for_zero_cross_section() -> None:
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=_spectrum(),
        stopping_table=_stopping_table(),
        kernel=_kernel(sigma_barn=0.0),
        thickness_g_cm2=1.0,
        depth_steps=4,
    )

    assert secondary.y == pytest.approx((0.0, 0.0, 0.0))


def test_combine_primary_and_secondary_proton_spectra() -> None:
    primary = _spectrum()
    secondary = Spectrum1D(
        x=primary.x,
        y=(1.0, 2.0, 3.0),
        x_unit=primary.x_unit,
        y_unit=primary.y_unit,
        quantity=primary.quantity,
        particle=Particle.PROTON,
        source=primary.source,
        model="secondary",
    )

    total = combine_primary_and_secondary_proton_spectra(
        primary=primary,
        secondary=secondary,
    )

    assert total.y == pytest.approx((11.0, 12.0, 13.0))


def test_load_secondary_proton_kernel(tmp_path: Path) -> None:
    xs_path = tmp_path / "xs.csv"
    yield_path = tmp_path / "yield.csv"
    pdf_path = tmp_path / "pdf.csv"

    xs_path.write_text(
        "mt,energy_mev,sigma_barn\n"
        "5,1.0,10.0\n"
        "5,100.0,20.0\n",
        encoding="utf-8",
    )
    yield_path.write_text(
        "mt,product_index,incident_energy_mev,yield\n"
        "5,0,1.0,1.0\n"
        "5,0,100.0,2.0\n",
        encoding="utf-8",
    )
    pdf_path.write_text(
        "mt,product_index,incident_energy_mev,emitted_energy_mev,pdf_per_mev\n"
        "5,0,1.0,1.0,0.1\n"
        "5,0,1.0,10.0,0.2\n"
        "5,0,100.0,1.0,0.3\n"
        "5,0,100.0,10.0,0.4\n",
        encoding="utf-8",
    )

    kernel = load_secondary_proton_kernel(
        xs_path=xs_path,
        yield_path=yield_path,
        pdf_path=pdf_path,
    )

    assert len(kernel.products) == 1
    assert set(kernel.cross_sections_by_mt) == {5}
    assert kernel.source_density_per_g_cm2(
        incident_energy_mev=50.0,
        emitted_energy_mev=5.0,
    ) > 0.0


def test_primary_survived_plus_secondary_total_path() -> None:
    spectrum = _spectrum()
    stopping_table = _stopping_table()
    xs = Al27NonelasticCrossSectionTable(
        mt=5,
        energy_mev=(1.0, 1000.0),
        sigma_barn=(1000.0, 1000.0),
    )

    primary = apply_proton_nonelastic_survival_to_primary_spectrum(
        spectrum=spectrum,
        stopping_table=stopping_table,
        cross_section_table=xs,
        thickness_g_cm2=1.0,
    )
    secondary = calculate_secondary_proton_spectrum_through_al(
        incident_spectrum=spectrum,
        stopping_table=stopping_table,
        kernel=_kernel(),
        thickness_g_cm2=1.0,
        depth_steps=4,
    )
    total = combine_primary_and_secondary_proton_spectra(
        primary=primary,
        secondary=secondary,
    )

    assert all(t >= p for t, p in zip(total.y, primary.y, strict=True))
