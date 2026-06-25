import pytest

from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.shielding.let_combined import (
    COMBINED_LET_MODEL,
    combine_let_spectra,
    combine_proton_and_hze_let_spectra,
    rebin_let_spectrum_to_grid,
)


def _let_spectrum(
    *,
    x: tuple[float, ...] = (1.0, 10.0, 100.0),
    y: tuple[float, ...] = (1.0, 2.0, 3.0),
    particle: Particle = Particle.PROTON,
    quantity: SpectrumQuantity = SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
    y_unit: Unit = Unit.DIFFERENTIAL_LET_FLUENCE,
    model: str = "let",
) -> Spectrum1D:
    return Spectrum1D(
        x=x,
        y=y,
        x_unit=Unit.LET,
        y_unit=y_unit,
        quantity=quantity,
        particle=particle,
        source=RadiationSource.SEP,
        model=model,
    )


def _log_edges(centers: tuple[float, ...]) -> tuple[float, ...]:
    from math import exp, log

    logs = tuple(log(center) for center in centers)
    edges = [0.0] * (len(centers) + 1)

    for index in range(1, len(centers)):
        edges[index] = exp(0.5 * (logs[index - 1] + logs[index]))

    edges[0] = exp(logs[0] - 0.5 * (logs[1] - logs[0]))
    edges[-1] = exp(logs[-1] + 0.5 * (logs[-1] - logs[-2]))

    return tuple(edges)


def _counts_integral(spectrum: Spectrum1D) -> float:
    edges = _log_edges(spectrum.x)

    return sum(
        spectrum.y[index] * (edges[index + 1] - edges[index])
        for index in range(len(spectrum.x))
    )


def test_rebin_let_spectrum_preserves_integral_on_wider_grid() -> None:
    source = _let_spectrum(
        x=(1.0, 10.0, 100.0),
        y=(1.0, 2.0, 3.0),
    )

    rebinned = rebin_let_spectrum_to_grid(
        spectrum=source,
        let_centers=(0.31622776601683794, 1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0, 316.22776601683796),
    )

    assert _counts_integral(rebinned) == pytest.approx(_counts_integral(source))


def test_combine_let_spectra_on_same_grid_adds_values() -> None:
    left = _let_spectrum(y=(1.0, 2.0, 3.0), particle=Particle.PROTON, model="proton")
    right = _let_spectrum(y=(4.0, 5.0, 6.0), particle=Particle.HZE, model="hze")

    combined = combine_let_spectra(spectra=(left, right))

    assert combined.x == left.x
    assert combined.y == pytest.approx((5.0, 7.0, 9.0))
    assert combined.x_unit is Unit.LET
    assert combined.y_unit is Unit.DIFFERENTIAL_LET_FLUENCE
    assert combined.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
    assert combined.model == COMBINED_LET_MODEL


def test_combine_let_spectra_rebins_to_common_grid() -> None:
    left = _let_spectrum(x=(1.0, 10.0, 100.0), y=(1.0, 1.0, 1.0))
    right = _let_spectrum(x=(1.0, 100.0), y=(2.0, 2.0), particle=Particle.HZE)

    combined = combine_let_spectra(
        spectra=(left, right),
        let_centers=(1.0, 10.0, 100.0),
    )

    assert combined.x == (1.0, 10.0, 100.0)
    assert _counts_integral(combined) == pytest.approx(
        _counts_integral(left)
        + _counts_integral(
            rebin_let_spectrum_to_grid(
                spectrum=right,
                let_centers=(1.0, 10.0, 100.0),
            )
        )
    )


def test_combine_proton_and_hze_let_spectra() -> None:
    proton = _let_spectrum(y=(1.0, 2.0, 3.0), particle=Particle.PROTON)
    hze_fe = _let_spectrum(y=(4.0, 5.0, 6.0), particle=Particle.HZE, model="fe")
    hze_o = _let_spectrum(y=(7.0, 8.0, 9.0), particle=Particle.HZE, model="o")

    combined = combine_proton_and_hze_let_spectra(
        proton_let=proton,
        hze_let_spectra=(hze_fe, hze_o),
    )

    assert combined.y == pytest.approx((12.0, 15.0, 18.0))
    assert combined.particle is Particle.HZE


def test_combine_rejects_energy_spectrum() -> None:
    energy_spectrum = Spectrum1D(
        x=(1.0, 10.0),
        y=(1.0, 2.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        particle=Particle.PROTON,
        source=RadiationSource.SEP,
        model="energy",
    )

    with pytest.raises(ValueError, match="LET spectrum"):
        combine_let_spectra(spectra=(energy_spectrum,))


def test_combine_rejects_quantity_mismatch() -> None:
    fluence = _let_spectrum()
    flux = _let_spectrum(
        quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_LET_FLUX,
        particle=Particle.HZE,
    )

    with pytest.raises(ValueError, match="same quantity"):
        combine_let_spectra(spectra=(fluence, flux))


def test_proton_hze_wrapper_requires_hze_spectra() -> None:
    proton = _let_spectrum(particle=Particle.PROTON)
    not_hze = _let_spectrum(particle=Particle.PROTON)

    with pytest.raises(ValueError, match="HZE LET spectra"):
        combine_proton_and_hze_let_spectra(
            proton_let=proton,
            hze_let_spectra=(not_hze,),
        )
