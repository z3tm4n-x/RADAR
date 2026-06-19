import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.geomagnetic.penetration import PenetrationFunction
from radar.geomagnetic.rigidity import RigidityGrid
from radar.geomagnetic.spectrum import apply_proton_penetration, proton_penetration_factors
from radar.physics.rigidity import proton_rigidity_to_kinetic_energy_mev


def _penetration() -> PenetrationFunction:
    return PenetrationFunction(
        rigidity_grid=RigidityGrid(values_gv=(1.0, 2.0, 3.0)),
        values=(0.1, 0.5, 0.9),
        model="test_penetration",
    )


def _proton_spectrum_for_rigidities(rigidities_gv: tuple[float, ...]) -> Spectrum1D:
    return Spectrum1D(
        x=tuple(proton_rigidity_to_kinetic_energy_mev(value) for value in rigidities_gv),
        y=tuple(10.0 for _ in rigidities_gv),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="test_spectrum",
    )


def test_proton_penetration_factors_are_calculated_from_energy_grid() -> None:
    spectrum = _proton_spectrum_for_rigidities((1.0, 2.0, 3.0))

    factors = proton_penetration_factors(
        spectrum=spectrum,
        penetration=_penetration(),
    )

    assert factors == pytest.approx((0.1, 0.5, 0.9))


def test_proton_penetration_factors_interpolate_in_rigidity() -> None:
    spectrum = _proton_spectrum_for_rigidities((1.5,))

    factors = proton_penetration_factors(
        spectrum=spectrum,
        penetration=_penetration(),
    )

    assert factors == pytest.approx((0.3,))


def test_apply_proton_penetration_multiplies_spectrum_values() -> None:
    spectrum = _proton_spectrum_for_rigidities((1.0, 2.0, 3.0))

    attenuated = apply_proton_penetration(
        spectrum=spectrum,
        penetration=_penetration(),
        model="attenuated_test_spectrum",
    )

    assert attenuated.x == spectrum.x
    assert attenuated.y == pytest.approx((1.0, 5.0, 9.0))
    assert attenuated.x_unit is Unit.MEV
    assert attenuated.y_unit is Unit.DIFFERENTIAL_FLUENCE
    assert attenuated.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert attenuated.particle is Particle.PROTON
    assert attenuated.source is RadiationSource.SEP
    assert attenuated.model == "attenuated_test_spectrum"


def test_apply_proton_penetration_preserves_original_spectrum() -> None:
    spectrum = _proton_spectrum_for_rigidities((1.0, 2.0, 3.0))

    attenuated = apply_proton_penetration(
        spectrum=spectrum,
        penetration=_penetration(),
    )

    assert spectrum.y == (10.0, 10.0, 10.0)
    assert attenuated.y == pytest.approx((1.0, 5.0, 9.0))
    assert attenuated.model == "test_spectrum+test_penetration"


def test_apply_proton_penetration_rejects_electron_spectrum() -> None:
    spectrum = Spectrum1D(
        x=(1.0, 2.0, 3.0),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.ELECTRON,
        source=RadiationSource.ERB,
        model="electron_spectrum",
    )

    with pytest.raises(ValueError, match="proton"):
        apply_proton_penetration(
            spectrum=spectrum,
            penetration=_penetration(),
        )


def test_apply_proton_penetration_rejects_non_mev_grid() -> None:
    spectrum = Spectrum1D(
        x=(1.0, 2.0, 3.0),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.GEV_PER_NUCLEON,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="bad_units",
    )

    with pytest.raises(ValueError, match="MeV"):
        apply_proton_penetration(
            spectrum=spectrum,
            penetration=_penetration(),
        )


def test_apply_proton_penetration_rejects_let_quantity() -> None:
    spectrum = Spectrum1D(
        x=(1.0, 2.0, 3.0),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_LET_FLUENCE,
        quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="let_spectrum",
    )

    with pytest.raises(ValueError, match="energy spectrum"):
        apply_proton_penetration(
            spectrum=spectrum,
            penetration=_penetration(),
        )


def test_apply_proton_penetration_rejects_zero_energy() -> None:
    spectrum = Spectrum1D(
        x=(0.0, 1.0, 2.0),
        y=(10.0, 10.0, 10.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="zero_energy_spectrum",
    )

    with pytest.raises(ValueError, match="positive"):
        apply_proton_penetration(
            spectrum=spectrum,
            penetration=_penetration(),
        )
