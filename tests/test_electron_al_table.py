from pathlib import Path

import pytest

from radar.core.spectra import ModelReference, Spectrum1D
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.shielding.electron_al import (
    ELECTRON_AL_MAPPED_RANGE_SUPPORT_INCOMPLETE,
    ELECTRON_AL_OUTPUT_ENERGY_SUPPORT_INCOMPLETE,
    ELECTRON_AL_PRIMARY_CSDA_MODEL,
    ELECTRON_AL_SOURCE_ENERGY_SUPPORT_INCOMPLETE,
    ElectronAlRangeEnergyTable,
    integrate_spectrum_loglog,
    load_electron_al_range_energy_table,
    log_bin_edges_from_centers,
    shield_electron_spectrum_primary_through_al,
)


def _range_table() -> ElectronAlRangeEnergyTable:
    return ElectronAlRangeEnergyTable(
        energy_mev=(1.0, 10.0, 100.0),
        range_g_cm2=(0.1, 1.0, 10.0),
        stopping_mev_cm2_g=(100.0, 10.0, 1.0),
    )


def _electron_spectrum(
    *,
    y: tuple[float, ...] = (10.0, 10.0, 10.0),
    particle: Particle = Particle.ELECTRON,
    source: RadiationSource = RadiationSource.ERB,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
) -> Spectrum1D:
    return Spectrum1D(
        x=(3.1622776601683795, 10.0, 31.622776601683793),
        y=y,
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=quantity,
        particle=particle,
        source=source,
        model="erb_electron_orbit_spectrum",
    )


def test_log_bin_edges_from_centers_returns_geometric_edges() -> None:
    edges = log_bin_edges_from_centers((10.0, 100.0, 1000.0))

    assert len(edges) == 4
    assert edges[1] == pytest.approx(10.0**1.5)
    assert edges[2] == pytest.approx(10.0**2.5)


def test_electron_al_table_interpolates_in_loglog_space() -> None:
    table = _range_table()

    assert table.range_at_energy(10.0) == pytest.approx(1.0)
    assert table.energy_at_range(1.0) == pytest.approx(10.0)
    assert table.stopping_at_energy(10.0) == pytest.approx(10.0)
    assert table.range_at_energy(31.622776601683793) == pytest.approx(3.1622776601683795)


def test_integrate_spectrum_loglog_uses_power_law_segments() -> None:
    spectrum = Spectrum1D(
        x=(1.0, 10.0),
        y=(1.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.ELECTRON,
        source=RadiationSource.ERB,
        model="linear_in_loglog",
    )

    assert integrate_spectrum_loglog(spectrum, (1.0,), (10.0,)) == pytest.approx((49.5,))


def test_integrate_spectrum_loglog_preserves_positive_to_zero_segment() -> None:
    spectrum = Spectrum1D(
        x=(1.0, 3.0),
        y=(2.0, 0.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.ELECTRON,
        source=RadiationSource.ERB,
        model="linear_through_zero",
    )

    full, partial = integrate_spectrum_loglog(
        spectrum,
        (1.0, 2.0),
        (3.0, 3.0),
    )

    assert full == pytest.approx(2.0)
    assert partial == pytest.approx(0.5)


def test_primary_electron_zero_thickness_preserves_spectrum() -> None:
    shielded = shield_electron_spectrum_primary_through_al(
        spectrum=_electron_spectrum(),
        table=_range_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.x == _electron_spectrum().x
    assert shielded.y == pytest.approx(_electron_spectrum().y)
    assert shielded.x_unit is Unit.MEV
    assert shielded.y_unit is Unit.DIFFERENTIAL_FLUENCE
    assert shielded.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert shielded.particle is Particle.ELECTRON
    assert shielded.source is RadiationSource.ERB
    assert shielded.model == f"erb_electron_orbit_spectrum+{ELECTRON_AL_PRIMARY_CSDA_MODEL}"


def test_primary_electron_zero_thickness_preserves_bin_contract_exactly() -> None:
    edges = (2.0, 6.0, 20.0, 50.0)
    source = _electron_spectrum()
    spectrum = Spectrum1D(
        x=source.x,
        y=source.y,
        x_unit=source.x_unit,
        y_unit=source.y_unit,
        quantity=source.quantity,
        particle=source.particle,
        source=source.source,
        model="bin_average_zero_identity",
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=edges,
    )

    shielded = shield_electron_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_range_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.x == spectrum.x
    assert shielded.y == spectrum.y
    assert shielded.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE
    assert shielded.bin_edges == edges
    assert shielded.location is SpectrumLocation.BEHIND_SHIELD
    assert shielded.shielding is not None
    assert shielded.shielding.areal_density_g_cm2 == 0.0
    assert len(shielded.transformation_chain) == len(spectrum.transformation_chain) + 1


def test_primary_electron_zero_thickness_does_not_add_stopping_provenance() -> None:
    source = _electron_spectrum()
    spectrum = Spectrum1D(
        x=source.x,
        y=source.y,
        x_unit=source.x_unit,
        y_unit=source.y_unit,
        quantity=source.quantity,
        particle=source.particle,
        source=source.source,
        model=source.model,
        model_reference=ModelReference(
            name=source.model,
            version="test-1",
            document_refs=("ERB source test document",),
            dataset_refs=("erb/source_spectrum.csv",),
        ),
    )

    shielded = shield_electron_spectrum_primary_through_al(
        spectrum=spectrum,
        table=None,
        thickness_g_cm2=0.0,
    )

    assert shielded.model_reference is not None
    assert shielded.model_reference.dataset_refs == ("erb/source_spectrum.csv",)
    assert "stopping/electron_al_range.csv" not in (shielded.model_reference.dataset_refs)


def test_primary_electron_remapping_rejects_bin_average_input() -> None:
    source = _electron_spectrum()
    spectrum = Spectrum1D(
        x=source.x,
        y=source.y,
        x_unit=source.x_unit,
        y_unit=source.y_unit,
        quantity=source.quantity,
        particle=source.particle,
        source=source.source,
        model="bin_average_requires_conservative_transport",
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=(2.0, 6.0, 20.0, 50.0),
    )

    with pytest.raises(ValueError, match="bin-average input"):
        shield_electron_spectrum_primary_through_al(
            spectrum=spectrum,
            table=_range_table(),
            thickness_g_cm2=0.1,
        )


def test_primary_electron_positive_thickness_remaps_input_bins() -> None:
    spectrum = _electron_spectrum(y=(1.0, 2.0, 3.0))

    shielded = shield_electron_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_range_table(),
        thickness_g_cm2=0.1,
    )

    assert shielded.x == spectrum.x
    assert shielded.y[0] > 0.0
    assert shielded.y[1] > 0.0
    assert shielded.y[1] != pytest.approx(spectrum.y[1])
    assert shielded.y[2] > 0.0
    assert all(value >= 0.0 for value in shielded.y)
    assert shielded.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE
    assert shielded.bin_edges == pytest.approx(log_bin_edges_from_centers(spectrum.x))
    assert ELECTRON_AL_SOURCE_ENERGY_SUPPORT_INCOMPLETE in (shielded.incomplete_reasons)
    assert shielded.completeness is SpectrumCompleteness.INCOMPLETE


def test_primary_electron_bins_without_available_range_are_zero() -> None:
    shielded = shield_electron_spectrum_primary_through_al(
        spectrum=_electron_spectrum(),
        table=_range_table(),
        thickness_g_cm2=1000.0,
    )

    assert shielded.y == (0.0, 0.0, 0.0)
    assert ELECTRON_AL_MAPPED_RANGE_SUPPORT_INCOMPLETE in (shielded.incomplete_reasons)
    assert shielded.completeness is SpectrumCompleteness.INCOMPLETE


def test_primary_electron_output_bin_uses_supported_fraction_and_is_incomplete() -> None:
    shielded = shield_electron_spectrum_primary_through_al(
        spectrum=_electron_spectrum(),
        table=_range_table(),
        thickness_g_cm2=0.1,
        output_energy_mev=(1.5, 10.0, 80.0),
    )

    output_edges = log_bin_edges_from_centers((1.5, 10.0, 80.0))
    expected_first = 10.0 * ((15.0**0.5 + 1.0) - 10.0**0.5) / (output_edges[1] - output_edges[0])
    assert shielded.y[0] == pytest.approx(expected_first)
    assert shielded.y[-1] > 0.0
    assert ELECTRON_AL_OUTPUT_ENERGY_SUPPORT_INCOMPLETE in (shielded.incomplete_reasons)
    assert ELECTRON_AL_SOURCE_ENERGY_SUPPORT_INCOMPLETE in (shielded.incomplete_reasons)


def test_loglog_integration_rejects_partial_source_interval() -> None:
    with pytest.raises(ValueError, match="outside the source spectrum support"):
        integrate_spectrum_loglog(
            _electron_spectrum(),
            (1.0,),
            (10.0,),
        )


def test_primary_electron_rejects_non_electron_spectrum() -> None:
    with pytest.raises(ValueError, match="electron"):
        shield_electron_spectrum_primary_through_al(
            spectrum=_electron_spectrum(particle=Particle.PROTON),
            table=_range_table(),
            thickness_g_cm2=1.0,
        )


def test_primary_electron_rejects_non_erb_spectrum() -> None:
    with pytest.raises(ValueError, match="ERB"):
        shield_electron_spectrum_primary_through_al(
            spectrum=_electron_spectrum(source=RadiationSource.SEP),
            table=_range_table(),
            thickness_g_cm2=1.0,
        )


def test_primary_electron_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        shield_electron_spectrum_primary_through_al(
            spectrum=_electron_spectrum(),
            table=_range_table(),
            thickness_g_cm2=-1.0,
        )


def test_electron_al_table_rejects_non_monotonic_range() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ElectronAlRangeEnergyTable(
            energy_mev=(1.0, 10.0, 100.0),
            range_g_cm2=(0.1, 0.1, 10.0),
            stopping_mev_cm2_g=(100.0, 10.0, 1.0),
        )


def test_load_electron_al_range_energy_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "electron_al_range.csv"
    csv_path.write_text(
        "energy_mev,stopping_mev_cm2_g,range_g_cm2,range_mg_cm2,source_sheet\n"
        "10.0,10.0,1.0,1000.0,e_Al\n"
        "1.0,100.0,0.1,100.0,e_Al\n"
        "100.0,1.0,10.0,10000.0,e_Al\n",
        encoding="utf-8",
    )

    table = load_electron_al_range_energy_table(csv_path)

    assert table.energy_mev == (1.0, 10.0, 100.0)
    assert table.range_at_energy(10.0) == pytest.approx(1.0)
    assert table.energy_at_range(1.0) == pytest.approx(10.0)
    assert table.stopping_at_energy(10.0) == pytest.approx(10.0)
