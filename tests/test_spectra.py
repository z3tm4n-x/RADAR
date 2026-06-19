import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit


def test_spectrum_integral_trapezoid() -> None:
    spectrum = Spectrum1D(
        x=(1.0, 2.0, 3.0),
        y=(2.0, 2.0, 2.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="test",
    )

    assert spectrum.integral_trapezoid() == pytest.approx(4.0)


def test_spectrum_grid_must_be_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        Spectrum1D(
            x=(1.0, 1.0, 2.0),
            y=(1.0, 2.0, 3.0),
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            particle=Particle.PROTON,
            source=RadiationSource.SEP,
            model="test",
        )


def test_spectrum_values_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Spectrum1D(
            x=(1.0, 2.0, 3.0),
            y=(1.0, -1.0, 3.0),
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            particle=Particle.PROTON,
            source=RadiationSource.SEP,
            model="test",
        )
