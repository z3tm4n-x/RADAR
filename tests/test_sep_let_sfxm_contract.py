import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationProductKind, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.resources import load_normative_shielding_tables
from radar.shielding.sep_hze import calculate_sep_hze_shielding_spectra
from radar.shielding.sep_let import (
    SEP_LET_HZE_PRIMARY_AFTER_AL,
    SEP_LET_NO_HIDDEN_4PI,
    SEP_LET_PRODUCTS_MODEL,
    SEP_LET_PROTON_TOTAL_AFTER_AL,
    SEP_LET_SILICON_TARGET,
    calculate_sep_combined_let_product,
    calculate_sep_let_products,
)
from radar.shielding.sep_proton import calculate_sep_proton_shielding_spectrum

SFXM_PEAK_FLUX_ROWS_10_TO_100_MEV = """
1.00E+0001 8.45E+0004 3.16E+0003 0.00E+0000 0.00E+0000 0.00E+0000 2.26E+0001 6.32E+0000 5.08E+0001 2.31E-0004 6.98E+0000 5.78E-0001 9.45E+0000 6.78E-0001 7.73E+0000 3.54E-0002 1.69E+0000 1.57E-0002 1.57E-0001 2.49E-0002 5.14E-0001 1.77E-0003 2.82E-0003 2.57E-0004 1.05E-0001 3.85E-0002 6.93E+0000 8.73E-0002 2.47E-0001
1.26E+0001 6.16E+0004 1.59E+0003 0.00E+0000 0.00E+0000 0.00E+0000 1.14E+0001 3.17E+0000 2.55E+0001 1.16E-0004 3.51E+0000 2.90E-0001 4.75E+0000 3.40E-0001 3.88E+0000 1.78E-0002 8.51E-0001 7.87E-0003 7.87E-0002 1.25E-0002 2.58E-0001 8.90E-0004 1.42E-0003 1.29E-0004 5.29E-0002 1.93E-0002 3.48E+0000 4.38E-0002 1.24E-0001
1.58E+0001 4.50E+0004 7.95E+0002 0.00E+0000 0.00E+0000 0.00E+0000 5.70E+0000 1.59E+0000 1.28E+0001 5.82E-0005 1.76E+0000 1.46E-0001 2.38E+0000 1.71E-0001 1.95E+0000 8.92E-0003 4.27E-0001 3.94E-0003 3.94E-0002 6.27E-0003 1.29E-0001 4.46E-0004 7.11E-0004 6.47E-0005 2.65E-0002 9.70E-0003 1.75E+0000 2.20E-0002 6.21E-0002
2.00E+0001 2.68E+0004 3.98E+0002 0.00E+0000 0.00E+0000 0.00E+0000 2.85E+0000 7.97E-0001 6.41E+0000 2.92E-0005 8.81E-0001 7.29E-0002 1.19E+0000 8.55E-0002 9.75E-0001 4.47E-0003 2.14E-0001 1.98E-0003 1.98E-0002 3.14E-0003 6.48E-0002 2.24E-0004 3.56E-0004 3.24E-0005 1.33E-0002 4.86E-0003 8.75E-0001 1.10E-0002 3.11E-0002
2.51E+0001 1.49E+0004 1.99E+0002 0.00E+0000 0.00E+0000 0.00E+0000 1.43E+0000 3.99E-0001 3.21E+0000 1.46E-0005 4.41E-0001 3.65E-0002 5.96E-0001 4.28E-0002 4.88E-0001 2.24E-0003 1.07E-0001 9.88E-0004 9.88E-0003 1.57E-0003 3.24E-0002 1.12E-0004 1.78E-0004 1.62E-0005 6.64E-0003 2.43E-0003 4.37E-0001 5.51E-0003 1.56E-0002
3.16E+0001 8.29E+0003 9.95E+0001 0.00E+0000 0.00E+0000 0.00E+0000 7.13E-0001 1.99E-0001 1.60E+0000 7.28E-0006 2.20E-0001 1.82E-0002 2.98E-0001 2.14E-0002 2.43E-0001 1.12E-0003 5.34E-0002 4.93E-0004 4.93E-0003 7.84E-0004 1.62E-0002 5.58E-0005 8.90E-0005 8.09E-0006 3.32E-0003 1.21E-0003 2.18E-0001 2.75E-0003 7.76E-0003
3.98E+0001 4.60E+0003 4.95E+0001 0.00E+0000 0.00E+0000 0.00E+0000 3.55E-0001 9.90E-0002 7.97E-0001 3.62E-0006 1.10E-0001 9.06E-0003 1.48E-0001 1.06E-0002 1.21E-0001 5.56E-0004 2.66E-0002 2.46E-0004 2.46E-0003 3.91E-0004 8.05E-0003 2.78E-0005 4.43E-0005 4.03E-0006 1.65E-0003 6.04E-0004 1.09E-0001 1.37E-0003 3.87E-0003
5.01E+0001 2.54E+0003 2.46E+0001 0.00E+0000 0.00E+0000 0.00E+0000 1.76E-0001 4.91E-0002 3.96E-0001 1.80E-0006 5.43E-0002 4.50E-0003 7.35E-0002 5.27E-0003 6.01E-0002 2.76E-0004 1.32E-0002 1.22E-0004 1.22E-0003 1.94E-0004 4.00E-0003 1.38E-0005 2.20E-0005 2.00E-0006 8.19E-0004 3.00E-0004 5.39E-0002 6.79E-0004 1.92E-0003
6.31E+0001 1.40E+0003 1.21E+0001 0.00E+0000 0.00E+0000 0.00E+0000 8.70E-0002 2.43E-0002 1.96E-0001 8.89E-0007 2.69E-0002 2.22E-0003 3.63E-0002 2.61E-0003 2.97E-0002 1.36E-0004 6.52E-0003 6.02E-0005 6.02E-0004 9.58E-0005 1.97E-0003 6.81E-0006 1.09E-0005 9.87E-0007 4.05E-0004 1.48E-0004 2.67E-0002 3.36E-0004 9.48E-0004
7.94E+0001 7.70E+0002 5.97E+0000 0.00E+0000 0.00E+0000 0.00E+0000 4.28E-0002 1.19E-0002 9.62E-0002 4.37E-0007 1.32E-0002 1.09E-0003 1.79E-0002 1.28E-0003 1.46E-0002 6.70E-0005 3.21E-0003 2.96E-0005 2.96E-0004 4.71E-0005 9.71E-0004 3.35E-0006 5.34E-0006 4.86E-0007 1.99E-0004 7.28E-0005 1.31E-0002 1.65E-0004 4.66E-0004
1.00E+0002 4.21E+0002 2.92E+0000 0.00E+0000 0.00E+0000 0.00E+0000 2.09E-0002 5.84E-0003 4.70E-0002 2.14E-0007 6.46E-0003 5.34E-0004 8.73E-0003 6.27E-0004 7.14E-0003 3.28E-0005 1.57E-0003 1.45E-0005 1.45E-0004 2.30E-0005 4.75E-0004 1.64E-0006 2.61E-0006 2.37E-0007 9.73E-0005 3.56E-0005 6.41E-0003 8.07E-0005 2.28E-0004
"""

_SELECTED_Z = (1, 2, 8, 26)
_THICKNESS_G_CM2 = 0.01


def _parse_sfxm_peak_flux_rows() -> tuple[tuple[float, ...], dict[int, tuple[float, ...]]]:
    energies: list[float] = []
    values_by_z: dict[int, list[float]] = {z: [] for z in _SELECTED_Z}

    for line_number, line in enumerate(SFXM_PEAK_FLUX_ROWS_10_TO_100_MEV.splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 29:
            msg = f"SFXM row {line_number} must contain energy plus 28 ion columns."
            raise ValueError(msg)

        energies.append(float(parts[0]))

        for z in _SELECTED_Z:
            values_by_z[z].append(float(parts[z]))

    return tuple(energies), {
        z: tuple(values)
        for z, values in values_by_z.items()
    }


def _sfxm_peak_flux_spectrum(
    *,
    z: int,
    particle: Particle,
    energies: tuple[float, ...],
    values_by_z: dict[int, tuple[float, ...]],
) -> Spectrum1D:
    return Spectrum1D(
        x=energies,
        y=values_by_z[z],
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        particle=particle,
        source=RadiationSource.SEP,
        model=f"sfxm_peak_flux_Z{z}",
    )


def test_sfxm_peak_flux_fixture_uses_expected_columns() -> None:
    energies, values_by_z = _parse_sfxm_peak_flux_rows()

    assert energies[0] == pytest.approx(10.0)
    assert energies[-1] == pytest.approx(100.0)
    assert values_by_z[1][0] == pytest.approx(8.45e4)
    assert values_by_z[2][0] == pytest.approx(3.16e3)
    assert values_by_z[8][0] == pytest.approx(5.08e1)
    assert values_by_z[26][0] == pytest.approx(6.93)


def test_sep_let_sfxm_peak_flux_mini_chain_with_normative_resources() -> None:
    tables = load_normative_shielding_tables()
    energies, values_by_z = _parse_sfxm_peak_flux_rows()

    proton_spectrum = _sfxm_peak_flux_spectrum(
        z=1,
        particle=Particle.PROTON,
        energies=energies,
        values_by_z=values_by_z,
    )
    hze_spectra_by_z = {
        z: _sfxm_peak_flux_spectrum(
            z=z,
            particle=Particle.HZE,
            energies=energies,
            values_by_z=values_by_z,
        )
        for z in (2, 8, 26)
    }

    proton = calculate_sep_proton_shielding_spectrum(
        spectrum=proton_spectrum,
        tables=tables,
        thickness_g_cm2=_THICKNESS_G_CM2,
        include_nonelastic_survival=True,
        include_secondary_protons=True,
        survival_integration_steps=8,
        secondary_depth_steps=4,
    )
    hze_by_z = calculate_sep_hze_shielding_spectra(
        spectra_by_z=hze_spectra_by_z,
        tables=tables,
        thickness_g_cm2=_THICKNESS_G_CM2,
    )
    result = calculate_sep_let_products(
        proton=proton,
        hze_by_z=hze_by_z,
        tables=tables,
    )

    assert all(center > 0.0 for center in result.let_centers)
    assert all(right > left for left, right in zip(result.let_centers, result.let_centers[1:]))
    assert any(value > 0.0 for value in result.proton_let.y)
    assert any(value > 0.0 for spectrum in result.hze_let_by_z.values() for value in spectrum.y)
    assert any(value > 0.0 for value in result.combined_let.y)

    expected_combined = tuple(
        result.proton_let.y[index]
        + sum(spectrum.y[index] for spectrum in result.hze_let_by_z.values())
        for index in range(len(result.let_centers))
    )
    assert result.combined_let.y == pytest.approx(expected_combined)

    assert result.combined_let.x_unit is Unit.LET
    assert result.combined_let.y_unit is Unit.DIFFERENTIAL_LET_FLUX
    assert result.combined_let.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUX
    assert result.combined_let.particle is Particle.HZE

    assert SEP_LET_PRODUCTS_MODEL in result.combined_let.model
    assert SEP_LET_PROTON_TOTAL_AFTER_AL in result.combined_let.model
    assert SEP_LET_HZE_PRIMARY_AFTER_AL in result.combined_let.model
    assert SEP_LET_SILICON_TARGET in result.combined_let.model
    assert SEP_LET_NO_HIDDEN_4PI in result.combined_let.model


def test_sep_combined_let_sfxm_peak_flux_product_kind() -> None:
    tables = load_normative_shielding_tables()
    energies, values_by_z = _parse_sfxm_peak_flux_rows()

    proton = calculate_sep_proton_shielding_spectrum(
        spectrum=_sfxm_peak_flux_spectrum(
            z=1,
            particle=Particle.PROTON,
            energies=energies,
            values_by_z=values_by_z,
        ),
        tables=tables,
        thickness_g_cm2=_THICKNESS_G_CM2,
        include_nonelastic_survival=True,
        include_secondary_protons=True,
        survival_integration_steps=8,
        secondary_depth_steps=4,
    )
    hze_by_z = calculate_sep_hze_shielding_spectra(
        spectra_by_z={
            z: _sfxm_peak_flux_spectrum(
                z=z,
                particle=Particle.HZE,
                energies=energies,
                values_by_z=values_by_z,
            )
            for z in (2, 8, 26)
        },
        tables=tables,
        thickness_g_cm2=_THICKNESS_G_CM2,
    )

    product = calculate_sep_combined_let_product(
        proton=proton,
        hze_by_z=hze_by_z,
        tables=tables,
        label="SFXM SEP combined peak LET flux",
    )

    assert product.kind is RadiationProductKind.PEAK_LET_FLUX
    assert product.label == "SFXM SEP combined peak LET flux"
    assert product.spectrum.y_unit is Unit.DIFFERENTIAL_LET_FLUX
    assert product.spectrum.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUX
    assert any(value > 0.0 for value in product.spectrum.y)
