import pytest

from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.sep.model import (
    SepModelInput,
    SepModelProtocol,
    SepModelResult,
    StaticSepModel,
    validate_sep_proton_fluence_spectrum,
)


def _sep_proton_fluence_spectrum(
    *,
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.SEP,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    x_unit: Unit = Unit.MEV,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
    x: tuple[float, ...] = (10.0, 100.0, 1000.0),
    y: tuple[float, ...] = (1.0, 2.0, 3.0),
    model: str = "annual_test_sep",
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


def _mission(lifetime_years: int = 5) -> MissionConfig:
    return MissionConfig(
        launch_year=2027,
        lifetime_years=lifetime_years,
        sep_exceedance_probability=0.1,
    )


def _run_sep_model(
    model: SepModelProtocol,
    model_input: SepModelInput,
) -> SepModelResult:
    return model.calculate(model_input)


def test_validate_sep_proton_fluence_spectrum_accepts_valid_spectrum() -> None:
    validate_sep_proton_fluence_spectrum(_sep_proton_fluence_spectrum())


def test_validate_sep_proton_fluence_spectrum_rejects_non_proton() -> None:
    with pytest.raises(ValueError, match="protons"):
        validate_sep_proton_fluence_spectrum(
            _sep_proton_fluence_spectrum(particle=Particle.ELECTRON)
        )


def test_validate_sep_proton_fluence_spectrum_rejects_non_sep_source() -> None:
    with pytest.raises(ValueError, match="SEP"):
        validate_sep_proton_fluence_spectrum(
            _sep_proton_fluence_spectrum(source=RadiationSource.GCR)
        )


def test_validate_sep_proton_fluence_spectrum_rejects_non_fluence_quantity() -> None:
    with pytest.raises(ValueError, match="differential fluence"):
        validate_sep_proton_fluence_spectrum(
            _sep_proton_fluence_spectrum(
                quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
                y_unit=Unit.DIFFERENTIAL_FLUX,
            )
        )


def test_validate_sep_proton_fluence_spectrum_rejects_non_mev_grid() -> None:
    with pytest.raises(ValueError, match="MeV"):
        validate_sep_proton_fluence_spectrum(
            _sep_proton_fluence_spectrum(x_unit=Unit.GEV_PER_NUCLEON)
        )


def test_validate_sep_proton_fluence_spectrum_rejects_bad_y_unit() -> None:
    with pytest.raises(ValueError, match="differential fluence units"):
        validate_sep_proton_fluence_spectrum(
            _sep_proton_fluence_spectrum(y_unit=Unit.DIFFERENTIAL_FLUX)
        )


def test_validate_sep_proton_fluence_spectrum_rejects_zero_energy() -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_sep_proton_fluence_spectrum(
            _sep_proton_fluence_spectrum(x=(0.0, 10.0, 100.0))
        )


def test_sep_model_input_exposes_mission_parameters() -> None:
    model_input = SepModelInput(mission=_mission(lifetime_years=7))

    assert model_input.lifetime_years == 7
    assert model_input.exceedance_probability == pytest.approx(0.1)


def test_static_sep_model_scales_annual_fluence_by_integer_lifetime() -> None:
    model = StaticSepModel(
        annual_fluence_spectrum=_sep_proton_fluence_spectrum(),
        model="static_test",
        document="test_document",
    )
    model_input = SepModelInput(mission=_mission(lifetime_years=5))

    result = _run_sep_model(model=model, model_input=model_input)

    assert result.spectrum.x == (10.0, 100.0, 1000.0)
    assert result.spectrum.y == pytest.approx((5.0, 10.0, 15.0))
    assert result.spectrum.model == "annual_test_sep*5years"
    assert result.lifetime_years == 5
    assert result.exceedance_probability == pytest.approx(0.1)
    assert result.model == "static_test"
    assert result.document == "test_document"


def test_static_sep_model_preserves_annual_spectrum() -> None:
    annual_spectrum = _sep_proton_fluence_spectrum()
    model = StaticSepModel(annual_fluence_spectrum=annual_spectrum)

    result = model.calculate(SepModelInput(mission=_mission(lifetime_years=3)))

    assert annual_spectrum.y == (1.0, 2.0, 3.0)
    assert result.spectrum.y == pytest.approx((3.0, 6.0, 9.0))


def test_static_sep_model_rejects_invalid_base_spectrum() -> None:
    with pytest.raises(ValueError, match="protons"):
        StaticSepModel(
            annual_fluence_spectrum=_sep_proton_fluence_spectrum(
                particle=Particle.ELECTRON
            )
        )


def test_sep_model_result_validates_probability() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        SepModelResult(
            spectrum=_sep_proton_fluence_spectrum(),
            lifetime_years=5,
            exceedance_probability=1.0,
            model="test",
            document="test",
        )


def test_sep_model_result_requires_model_name() -> None:
    with pytest.raises(ValueError, match="model"):
        SepModelResult(
            spectrum=_sep_proton_fluence_spectrum(),
            lifetime_years=5,
            exceedance_probability=0.1,
            model="",
            document="test",
        )


def test_sep_model_result_requires_document() -> None:
    with pytest.raises(ValueError, match="document"):
        SepModelResult(
            spectrum=_sep_proton_fluence_spectrum(),
            lifetime_years=5,
            exceedance_probability=0.1,
            model="test",
            document="",
        )
