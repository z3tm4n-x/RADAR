import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationSource,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
)
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
    assert spectrum.integral_conserved() == pytest.approx(4.0)


def test_bin_average_integral_uses_explicit_bin_widths() -> None:
    spectrum = Spectrum1D(
        x=(1.0, 3.0),
        y=(2.0, 4.0),
        x_unit=Unit.LET,
        y_unit=Unit.DIFFERENTIAL_LET_FLUENCE,
        quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
        particle=Particle.HZE,
        source=RadiationSource.SEP,
        model="histogram",
        ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        bin_edges=(0.5, 2.0, 5.0),
    )

    assert spectrum.integral_conserved() == pytest.approx(15.0)
    with pytest.raises(ValueError, match="bin-average"):
        spectrum.integral_trapezoid()


def test_bin_average_spectrum_requires_explicit_edges() -> None:
    with pytest.raises(ValueError, match="explicit bin_edges"):
        Spectrum1D(
            x=(1.0, 3.0),
            y=(2.0, 4.0),
            x_unit=Unit.MEV,
            y_unit=Unit.DIFFERENTIAL_FLUENCE,
            quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
            particle=Particle.PROTON,
            source=RadiationSource.SEP,
            model="histogram",
            ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
        )


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


def test_daily_display_spectrum_units_exist() -> None:
    from radar.core.units import Unit

    assert Unit.DIFFERENTIAL_FLUENCE_PER_DAY.value == "cm^-2 day^-1 MeV^-1"
    assert (
        Unit.DIFFERENTIAL_LET_FLUENCE_PER_DAY.value
        == "cm^-2 day^-1 (MeV*cm^2/mg)^-1"
    )
