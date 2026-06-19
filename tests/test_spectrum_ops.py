import math

import pytest

from radar.core.spectra import Spectrum1D
from radar.core.spectrum_ops import add_spectra, check_spectra_compatible, scale_spectrum
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit


def _proton_spectrum(
    *,
    x: tuple[float, ...] = (1.0, 2.0, 3.0),
    y: tuple[float, ...] = (10.0, 20.0, 30.0),
    x_unit: Unit = Unit.MEV,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.SEP,
    model: str = "test",
) -> Spectrum1D:
    return Spectrum1D(
        x=x,
        y=y,
        x_unit=x_unit,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=source,
        model=model,
    )


def test_scale_spectrum_multiplies_values() -> None:
    spectrum = _proton_spectrum()

    scaled = scale_spectrum(
        spectrum=spectrum,
        factor=2.0,
        model="scaled",
    )

    assert scaled.x == spectrum.x
    assert scaled.y == pytest.approx((20.0, 40.0, 60.0))
    assert scaled.x_unit is spectrum.x_unit
    assert scaled.y_unit is spectrum.y_unit
    assert scaled.quantity is spectrum.quantity
    assert scaled.particle is spectrum.particle
    assert scaled.source is spectrum.source
    assert scaled.model == "scaled"


def test_scale_spectrum_preserves_original_spectrum() -> None:
    spectrum = _proton_spectrum()

    scaled = scale_spectrum(spectrum=spectrum, factor=0.5)

    assert spectrum.y == (10.0, 20.0, 30.0)
    assert scaled.y == pytest.approx((5.0, 10.0, 15.0))
    assert scaled.model == "test*0.5"


def test_scale_spectrum_rejects_negative_factor() -> None:
    spectrum = _proton_spectrum()

    with pytest.raises(ValueError, match="non-negative"):
        scale_spectrum(spectrum=spectrum, factor=-1.0)


def test_scale_spectrum_rejects_non_finite_factor() -> None:
    spectrum = _proton_spectrum()

    with pytest.raises(ValueError, match="finite"):
        scale_spectrum(spectrum=spectrum, factor=math.inf)


def test_add_spectra_adds_values() -> None:
    left = _proton_spectrum(y=(1.0, 2.0, 3.0), model="left")
    right = _proton_spectrum(y=(10.0, 20.0, 30.0), model="right")

    total = add_spectra(left=left, right=right, model="total")

    assert total.x == left.x
    assert total.y == pytest.approx((11.0, 22.0, 33.0))
    assert total.x_unit is Unit.MEV
    assert total.y_unit is Unit.DIFFERENTIAL_FLUENCE
    assert total.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert total.particle is Particle.PROTON
    assert total.source is RadiationSource.SEP
    assert total.model == "total"


def test_add_spectra_default_model_name() -> None:
    left = _proton_spectrum(y=(1.0, 2.0, 3.0), model="left")
    right = _proton_spectrum(y=(10.0, 20.0, 30.0), model="right")

    total = add_spectra(left=left, right=right)

    assert total.model == "left+right"


def test_add_spectra_rejects_different_grid() -> None:
    left = _proton_spectrum(x=(1.0, 2.0, 3.0))
    right = _proton_spectrum(x=(1.0, 2.5, 3.0))

    with pytest.raises(ValueError, match="grid"):
        add_spectra(left=left, right=right)


def test_add_spectra_rejects_different_x_unit() -> None:
    left = _proton_spectrum()
    right = _proton_spectrum(x_unit=Unit.GEV_PER_NUCLEON)

    with pytest.raises(ValueError, match="x unit"):
        add_spectra(left=left, right=right)


def test_add_spectra_rejects_different_y_unit() -> None:
    left = _proton_spectrum()
    right = _proton_spectrum(y_unit=Unit.DIFFERENTIAL_FLUX)

    with pytest.raises(ValueError, match="y unit"):
        add_spectra(left=left, right=right)


def test_add_spectra_rejects_different_quantity() -> None:
    left = _proton_spectrum()
    right = _proton_spectrum(quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX)

    with pytest.raises(ValueError, match="quantity"):
        add_spectra(left=left, right=right)


def test_add_spectra_rejects_different_particle() -> None:
    left = _proton_spectrum()
    right = _proton_spectrum(particle=Particle.ELECTRON)

    with pytest.raises(ValueError, match="particle"):
        add_spectra(left=left, right=right)


def test_add_spectra_rejects_different_source() -> None:
    left = _proton_spectrum()
    right = _proton_spectrum(source=RadiationSource.GCR)

    with pytest.raises(ValueError, match="source"):
        add_spectra(left=left, right=right)


def test_check_spectra_compatible_accepts_compatible_spectra() -> None:
    left = _proton_spectrum()
    right = _proton_spectrum(y=(5.0, 6.0, 7.0))

    check_spectra_compatible(left=left, right=right)
