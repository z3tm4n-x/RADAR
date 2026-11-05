from pathlib import Path

import pytest

from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.shielding.hze_al import (
    HZE_AL_MASS_NUMBER_MISMATCH_REASON_PREFIX,
    HZE_AL_NO_FRAGMENTATION_NOTE,
    HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX,
    HzeAlRangeTable,
    load_hze_al_range_tables,
    shield_hze_spectrum_primary_through_al,
)
from radar.shielding.resources import load_normative_shielding_tables


def _table() -> HzeAlRangeTable:
    return HzeAlRangeTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
        range_g_cm2=(0.02, 0.06, 0.2, 0.6, 2.0, 6.0, 20.0),
    )


def _identity_range_table(*, maximum_energy: float = 200.0) -> HzeAlRangeTable:
    energies = (
        (1.0, 10.0, 30.0, 100.0, maximum_energy)
        if maximum_energy > 100.0
        else (1.0, 10.0, 30.0, maximum_energy)
    )
    return HzeAlRangeTable(
        z=26,
        symbol="Fe",
        mass_number=56.0,
        mass_to_charge=56.0 / 26.0,
        energy_mev_per_nucleon=energies,
        range_g_cm2=energies,
    )


def _spectrum(
    *,
    y: tuple[float, ...] = (10.0, 10.0, 10.0),
    particle: Particle = Particle.HZE,
) -> Spectrum1D:
    return Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=y,
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=particle,
        source=RadiationSource.SEP,
        model="sep_fe_orbit_spectrum",
    )


def test_hze_al_table_interpolates_in_both_directions() -> None:
    table = _table()

    assert table.range_at_energy(20.0) == pytest.approx(0.4)
    assert table.energy_at_range(0.4) == pytest.approx(20.0)


def test_hze_zero_thickness_preserves_flat_spectrum() -> None:
    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=_spectrum(),
        table=_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.x == (10.0, 30.0, 100.0)
    assert shielded.y == pytest.approx((10.0, 10.0, 10.0))
    assert shielded.particle is Particle.HZE
    assert HZE_AL_NO_FRAGMENTATION_NOTE in shielded.model


def test_hze_zero_thickness_preserves_bin_contract_exactly() -> None:
    edges = (5.0, 20.0, 60.0, 150.0)
    spectrum = Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(1.25, 2.5, 5.0),
        x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.HZE,
        source=RadiationSource.SEP,
        model="bin_average_zero_identity",
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=edges,
    )

    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_table(),
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


def test_hze_zero_thickness_does_not_claim_stopping_table() -> None:
    source_reference = ModelReference(
        name="test source",
        version="1",
        dataset_refs=("source/hze.csv",),
    )
    spectrum = Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.HZE,
        source=RadiationSource.SEP,
        model="zero_shield_provenance",
        model_reference=source_reference,
    )

    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.model_reference is not None
    assert shielded.model_reference.dataset_refs == source_reference.dataset_refs


def test_hze_positive_thickness_rejects_bin_average_input() -> None:
    spectrum = Spectrum1D(
        x=(10.0, 30.0, 100.0),
        y=(1.0, 2.0, 3.0),
        x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.HZE,
        source=RadiationSource.SEP,
        model="bin_average_requires_conservative_transport",
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=(5.0, 20.0, 60.0, 150.0),
    )

    with pytest.raises(ValueError, match="bin-average input"):
        shield_hze_spectrum_primary_through_al(
            spectrum=spectrum,
            table=_table(),
            thickness_g_cm2=0.1,
        )


def test_hze_typed_axis_and_species_survive_support_aligned_transport() -> None:
    species = ParticleSpecies(symbol="Fe", atomic_number=26, mass_number=56)
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x_unit": Unit.MEV_PER_NUCLEON,
            "y_unit": Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
            "axis_kind": SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
            "species": species,
        }
    )

    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_table(),
        thickness_g_cm2=0.1,
    )

    assert shielded.axis_kind is SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
    assert shielded.species == species
    assert shielded.x_unit is Unit.MEV_PER_NUCLEON


def test_hze_typed_mass_mismatch_is_explicit_for_positive_shield() -> None:
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x_unit": Unit.MEV_PER_NUCLEON,
            "y_unit": Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
            "axis_kind": SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
            "species": ParticleSpecies(
                symbol="Fe",
                atomic_number=26,
                mass_number=55,
            ),
        }
    )

    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_table(),
        thickness_g_cm2=0.1,
    )

    assert any(
        reason.startswith(HZE_AL_MASS_NUMBER_MISMATCH_REASON_PREFIX)
        for reason in shielded.incomplete_reasons
    )


def test_hze_typed_charge_or_symbol_mismatch_is_rejected() -> None:
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x_unit": Unit.MEV_PER_NUCLEON,
            "y_unit": Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
            "axis_kind": SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
            "species": ParticleSpecies(
                symbol="Co",
                atomic_number=27,
                mass_number=59,
            ),
        }
    )

    with pytest.raises(ValueError, match="does not match"):
        shield_hze_spectrum_primary_through_al(
            spectrum=spectrum,
            table=_table(),
            thickness_g_cm2=0.1,
        )


def test_hze_zero_shield_mass_mismatch_remains_clean_identity() -> None:
    spectrum = Spectrum1D(
        **{
            **_spectrum().__dict__,
            "x_unit": Unit.MEV_PER_NUCLEON,
            "y_unit": Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
            "axis_kind": SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
            "species": ParticleSpecies(
                symbol="Fe",
                atomic_number=26,
                mass_number=55,
            ),
        }
    )

    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_table(),
        thickness_g_cm2=0.0,
    )

    assert shielded.x == spectrum.x
    assert shielded.y == spectrum.y
    assert not any(
        reason.startswith(HZE_AL_MASS_NUMBER_MISMATCH_REASON_PREFIX)
        for reason in shielded.incomplete_reasons
    )


def test_hze_positive_thickness_moves_to_higher_input_energy() -> None:
    spectrum = _spectrum(y=(1.0, 2.0, 3.0))

    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_table(),
        thickness_g_cm2=0.1,
    )

    assert shielded.y[0] > 0.0
    assert shielded.y[0] != pytest.approx(spectrum.y[0])
    assert all(value >= 0.0 for value in shielded.y)


def test_hze_support_aligned_transport_conserves_piecewise_linear_source() -> None:
    spectrum = _spectrum(y=(1.0, 3.0, 10.0))

    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_identity_range_table(),
        thickness_g_cm2=1.0,
    )

    assert shielded.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE
    assert shielded.bin_edges is not None
    assert any(edge == pytest.approx(9.0) for edge in shielded.bin_edges)
    assert any(edge == pytest.approx(99.0) for edge in shielded.bin_edges)
    # f(E)=0.1 E on 10..100 MeV/nucleon, mapped by E_in=E_out+1.
    assert shielded.integral_conserved() == pytest.approx(495.0)


def test_hze_unsupported_sub_bins_are_zero_not_partial_full_bin_averages() -> None:
    spectrum = _spectrum(y=(2.0, 2.0, 2.0))

    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_identity_range_table(),
        thickness_g_cm2=1.0,
    )

    assert shielded.bin_edges is not None
    for value, lower, upper in zip(
        shielded.y,
        shielded.bin_edges,
        shielded.bin_edges[1:],
    ):
        if lower >= 9.0 and upper <= 99.0:
            assert value == pytest.approx(2.0)
        else:
            assert value == 0.0

    assert shielded.integral_conserved() == pytest.approx(180.0)
    assert any("source lower support" in reason for reason in shielded.incomplete_reasons)
    assert any("source upper support" in reason for reason in shielded.incomplete_reasons)


def test_hze_table_and_range_boundaries_are_separate_explicit_zero_bins() -> None:
    spectrum = _spectrum(y=(1.0, 1.0, 1.0))

    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=spectrum,
        table=_identity_range_table(maximum_energy=80.0),
        thickness_g_cm2=1.0,
    )

    assert shielded.bin_edges is not None
    assert any(edge == pytest.approx(79.0) for edge in shielded.bin_edges)
    assert any(edge == pytest.approx(80.0) for edge in shielded.bin_edges)
    assert shielded.integral_conserved() == pytest.approx(70.0)
    assert any(
        "require aluminium range above" in reason
        for reason in shielded.incomplete_reasons
    )
    assert any(
        "exceed aluminium range-table energy support" in reason
        for reason in shielded.incomplete_reasons
    )


def test_hze_bins_without_available_range_are_zero() -> None:
    shielded = shield_hze_spectrum_primary_through_al(
        spectrum=_spectrum(),
        table=_table(),
        thickness_g_cm2=1000.0,
    )

    assert shielded.y == (0.0, 0.0, 0.0)
    assert any(
        reason.startswith(HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX)
        for reason in shielded.incomplete_reasons
    )


@pytest.mark.parametrize(
    ("source", "quantity", "y_unit"),
    (
        (
            RadiationSource.SEP,
            SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
            Unit.DIFFERENTIAL_FLUX,
        ),
        (
            RadiationSource.SEP,
            SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            Unit.DIFFERENTIAL_FLUENCE,
        ),
        (
            RadiationSource.GCR,
            SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY,
        ),
        (
            RadiationSource.GCR,
            SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,
        ),
    ),
)
def test_hze_zero_thickness_is_exact_for_every_supported_ion_and_mode(
    source: RadiationSource,
    quantity: SpectrumQuantity,
    y_unit: Unit,
) -> None:
    tables = load_normative_shielding_tables().hze_al_ranges_by_z
    spectrum = Spectrum1D(
        x=(1.0, 10.0, 100.0),
        y=(1.25, 2.5, 5.0),
        x_unit=Unit.MEV_PER_NUCLEON,
        y_unit=y_unit,
        quantity=quantity,
        particle=Particle.HZE,
        source=source,
        model="zero_shield_identity_regression",
    )

    for table in tables.values():
        shielded = shield_hze_spectrum_primary_through_al(
            spectrum=spectrum,
            table=table,
            thickness_g_cm2=0.0,
        )

        assert shielded.x == spectrum.x
        assert shielded.y == spectrum.y
        assert shielded.ordinate_basis is spectrum.ordinate_basis
        assert shielded.bin_edges == spectrum.bin_edges
        assert not any(
            reason.startswith(HZE_AL_TRANSPORT_DOMAIN_REASON_PREFIX)
            for reason in shielded.incomplete_reasons
        )


def test_hze_rejects_non_hze_spectrum() -> None:
    with pytest.raises(ValueError, match="HZE spectrum"):
        shield_hze_spectrum_primary_through_al(
            spectrum=_spectrum(particle=Particle.PROTON),
            table=_table(),
            thickness_g_cm2=0.1,
        )


def test_hze_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        shield_hze_spectrum_primary_through_al(
            spectrum=_spectrum(),
            table=_table(),
            thickness_g_cm2=-1.0,
        )


def test_load_hze_al_range_tables(tmp_path: Path) -> None:
    csv_path = tmp_path / "hze_al_range.csv"
    csv_path.write_text(
        "z,symbol,mass_number,mass_to_charge,energy_mev_per_nucleon,range_g_cm2,range_mm,source_sheet\n"
        "26,Fe,56,2.1538461538,10.0,0.2,0.7407,i_R_Al\n"
        "26,Fe,56,2.1538461538,30.0,0.6,2.2222,i_R_Al\n"
        "26,Fe,56,2.1538461538,100.0,2.0,7.4074,i_R_Al\n"
        "8,O,16,2.0,10.0,0.4,1.4815,i_R_Al\n"
        "8,O,16,2.0,30.0,1.2,4.4444,i_R_Al\n"
        "8,O,16,2.0,100.0,4.0,14.8148,i_R_Al\n",
        encoding="utf-8",
    )

    tables = load_hze_al_range_tables(csv_path)

    assert set(tables) == {8, 26}
    assert tables[26].symbol == "Fe"
    assert tables[26].range_at_energy(20.0) == pytest.approx(0.4)
    assert tables[8].energy_at_range(2.0) == pytest.approx(50.0)
