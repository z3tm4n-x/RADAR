from pathlib import Path

import pytest

from radar.core.spectra import ModelReference, Spectrum1D
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.shielding.proton_al import (
    PROTON_AL_PRIMARY_CSDA_MODEL,
    PROTON_AL_TRANSPORT_DOMAIN_REASON_PREFIX,
    ProtonAlRangeEnergyTable,
    load_proton_al_range_energy_table,
    log_bin_edges_from_centers,
    shield_proton_spectrum_primary_through_al,
)
from radar.shielding.resources import load_normative_shielding_tables


def _range_table() -> ProtonAlRangeEnergyTable:
    return ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
        range_g_cm2=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
        stopping_mev_cm2_g=(100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0),
    )


def _proton_spectrum(
    *,
    y: tuple[float, ...] = (10.0, 10.0, 10.0),
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.SEP,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=y,
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=quantity,
        particle=particle,
        source=source,
        model="sep_proton_orbit_spectrum",
    )


def test_log_bin_edges_from_centers_returns_geometric_edges() -> None:
    edges = log_bin_edges_from_centers((10.0, 100.0, 1000.0))

    assert len(edges) == 4
    assert edges[1] == pytest.approx(10.0**1.5)
    assert edges[2] == pytest.approx(10.0**2.5)


def test_proton_al_table_interpolates_in_both_directions() -> None:
    table = _range_table()

    assert table.range_at_energy(20.0) == pytest.approx(2.0)
    assert table.energy_at_range(2.0) == pytest.approx(20.0)
    assert table.stopping_at_energy(20.0) == pytest.approx(50.0)


def test_primary_proton_zero_thickness_preserves_flat_spectrum() -> None:
    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=_proton_spectrum(),
        table=_range_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.x == (10.0, 30.0, 100.0)
    assert shielded.y == pytest.approx((10.0, 10.0, 10.0))
    assert shielded.x_unit is Unit.MEV
    assert shielded.y_unit is Unit.DIFFERENTIAL_FLUENCE
    assert shielded.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert shielded.particle is Particle.PROTON
    assert shielded.source is RadiationSource.SEP
    assert shielded.model == f"sep_proton_orbit_spectrum+{PROTON_AL_PRIMARY_CSDA_MODEL}"


def test_primary_proton_zero_thickness_preserves_bin_contract_exactly() -> None:
    edges = (5.0, 20.0, 60.0, 150.0)
    spectrum = Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(1.25, 2.5, 5.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="bin_average_zero_identity",
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=edges,
    )

    shielded = shield_proton_spectrum_primary_through_al(
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


def test_primary_proton_zero_thickness_does_not_claim_stopping_table() -> None:
    source_reference = ModelReference(
        name="test source",
        version="1",
        dataset_refs=("source/proton.csv",),
    )
    spectrum = Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="zero_shield_provenance",
        model_reference=source_reference,
    )

    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_range_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.model_reference is not None
    assert shielded.model_reference.dataset_refs == source_reference.dataset_refs


def test_primary_proton_positive_thickness_rejects_bin_average_input() -> None:
    spectrum = Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="bin_average_requires_conservative_transport",
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=(5.0, 20.0, 60.0, 150.0),
    )

    with pytest.raises(ValueError, match="bin-average input"):
        shield_proton_spectrum_primary_through_al(
            spectrum=spectrum,
            table=_range_table(),
            thickness_g_cm2=1.0,
        )


def test_primary_proton_positive_thickness_moves_to_higher_input_energy() -> None:
    spectrum = _proton_spectrum(y=(1.0, 2.0, 3.0))

    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_range_table(),
        thickness_g_cm2=1.0,
    )

    assert shielded.y[0] > 0.0
    assert shielded.y[0] != pytest.approx(spectrum.y[0])
    assert all(value >= 0.0 for value in shielded.y)


def test_primary_proton_integrates_positive_to_zero_source_segment() -> None:
    spectrum = _proton_spectrum(y=(10.0, 0.0, 0.0))
    table = ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 10.0, 30.0, 100.0, 300.0),
        range_g_cm2=(1.0, 10.0, 30.0, 100.0, 300.0),
        stopping_mev_cm2_g=(100.0, 80.0, 60.0, 40.0, 20.0),
    )

    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=table,
        thickness_g_cm2=1.0,
    )

    edges = log_bin_edges_from_centers(spectrum.x)
    mapped_lo = edges[1] + 1.0
    ordinate_at_mapped_lo = 10.0 * (30.0 - mapped_lo) / 20.0
    expected_triangle = (0.5 * ordinate_at_mapped_lo * (30.0 - mapped_lo)) / (edges[2] - edges[1])
    assert shielded.y[1] == pytest.approx(expected_triangle)
    assert shielded.y[1] > 0.0


def test_primary_proton_bins_without_available_range_are_zero() -> None:
    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=_proton_spectrum(),
        table=_range_table(),
        thickness_g_cm2=1000.0,
    )

    assert shielded.y == (0.0, 0.0, 0.0)
    assert any(
        reason.startswith(PROTON_AL_TRANSPORT_DOMAIN_REASON_PREFIX)
        for reason in shielded.incomplete_reasons
    )


def test_primary_proton_marks_mapped_upper_source_domain_incomplete() -> None:
    spectrum = _proton_spectrum(y=(1.0, 2.0, 3.0))

    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_range_table(),
        thickness_g_cm2=1.0,
    )

    assert 0.0 < shielded.y[-1] < spectrum.y[-1]
    assert any(
        reason.startswith(PROTON_AL_TRANSPORT_DOMAIN_REASON_PREFIX)
        and "source upper support 100" in reason
        and "1 bin(s)" in reason
        for reason in shielded.incomplete_reasons
    )


@pytest.mark.parametrize(
    ("source", "last_decade"),
    (
        (RadiationSource.GCR, 4.9),
        (RadiationSource.SEP, 4.1),
    ),
)
def test_primary_proton_marks_stage4_last_bin_mapping_loss_incomplete(
    source: RadiationSource,
    last_decade: float,
) -> None:
    grid = tuple(10.0 ** (last_decade - offset) for offset in (0.2, 0.1, 0.0))
    spectrum = Spectrum1D(
        x=grid,
        y=(1.0, 1.0, 1.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=source,
        model=f"{source.value}_stage4_upper_domain",
    )
    table = load_normative_shielding_tables().proton_al_range

    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=table,
        thickness_g_cm2=10.0,
    )

    assert 0.0 < shielded.y[-1] < 1.0
    assert any(
        reason.startswith(PROTON_AL_TRANSPORT_DOMAIN_REASON_PREFIX)
        and "source upper support" in reason
        and "1 bin(s)" in reason
        for reason in shielded.incomplete_reasons
    )


def test_primary_proton_uses_supported_boundary_fraction_in_full_bin() -> None:
    table = ProtonAlRangeEnergyTable(
        energy_mev=(10.0, 30.0, 100.0, 300.0, 1000.0),
        range_g_cm2=(1.0, 3.0, 10.0, 30.0, 100.0),
        stopping_mev_cm2_g=(100.0, 80.0, 60.0, 40.0, 20.0),
    )
    spectrum = _proton_spectrum(y=(1.0, 1.0, 1.0))

    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=table,
        thickness_g_cm2=1.0,
    )

    expected_edges = log_bin_edges_from_centers(spectrum.x)
    assert expected_edges[0] < table.energy_mev[0]
    assert shielded.bin_edges == expected_edges
    expected_supported_fraction = (expected_edges[1] - table.energy_mev[0]) / (
        expected_edges[1] - expected_edges[0]
    )
    assert shielded.y[0] == pytest.approx(expected_supported_fraction)
    assert any("range-table energy support" in reason for reason in shielded.incomplete_reasons)


def test_primary_proton_bin_averages_conserve_valid_mapped_intervals() -> None:
    table = ProtonAlRangeEnergyTable(
        energy_mev=(1.0, 10.0, 100.0, 1000.0, 3000.0),
        range_g_cm2=(0.1, 1.0, 10.0, 100.0, 300.0),
        stopping_mev_cm2_g=(100.0, 80.0, 60.0, 40.0, 20.0),
    )
    value = 7.5
    spectrum = Spectrum1D(
        x=(10.0, 30.0, 100.0, 300.0, 1000.0),
        y=(value,) * 5,
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="constant_point_source",
    )

    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=spectrum,
        table=table,
        thickness_g_cm2=1.0,
    )

    assert shielded.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE
    assert shielded.bin_edges is not None
    assert shielded.y[:-1] == pytest.approx((value,) * 4)
    valid_integral = sum(
        ordinate * (right - left)
        for ordinate, left, right in zip(
            shielded.y[:-1],
            shielded.bin_edges[:-2],
            shielded.bin_edges[1:-1],
        )
    )
    represented_width = shielded.bin_edges[-2] - shielded.bin_edges[0]
    assert valid_integral == pytest.approx(value * represented_width)


def test_primary_proton_accepts_erb_spectrum() -> None:
    shielded = shield_proton_spectrum_primary_through_al(
        spectrum=_proton_spectrum(source=RadiationSource.ERB),
        table=_range_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.source is RadiationSource.ERB


def test_primary_proton_rejects_non_proton_spectrum() -> None:
    with pytest.raises(ValueError, match="proton"):
        shield_proton_spectrum_primary_through_al(
            spectrum=_proton_spectrum(particle=Particle.HZE),
            table=_range_table(),
            thickness_g_cm2=1.0,
        )


def test_primary_proton_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        shield_proton_spectrum_primary_through_al(
            spectrum=_proton_spectrum(),
            table=_range_table(),
            thickness_g_cm2=-1.0,
        )


def test_load_proton_al_range_energy_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "proton_al_range.csv"
    csv_path.write_text(
        "energy_mev,stopping_mev_cm2_g,range_g_cm2,source_sheet\n"
        "1.0,100.0,0.1,p_Al\n"
        "10.0,60.0,1.0,p_Al\n"
        "30.0,40.0,3.0,p_Al\n",
        encoding="utf-8",
    )

    table = load_proton_al_range_energy_table(csv_path)

    assert table.range_at_energy(20.0) == pytest.approx(2.0)
    assert table.energy_at_range(2.0) == pytest.approx(20.0)
    assert table.stopping_at_energy(20.0) == pytest.approx(50.0)
