import pytest

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.erb.model import (
    ErbModelInput,
    ErbModelProtocol,
    ErbModelResult,
    StaticErbModel,
    validate_erb_energy_spectrum,
)


def _config(lifetime_years: int = 7, kp: int = 3) -> CalculationConfig:
    mission = MissionConfig(
        launch_year=2027,
        lifetime_years=lifetime_years,
    )
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    return CalculationConfig(mission=mission, orbit=orbit, kp=kp)


def _erb_flux_spectrum(
    *,
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.ERB,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUX,
    x_unit: Unit = Unit.MEV,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
    x: tuple[float, ...] = (0.1, 1.0, 10.0),
    y: tuple[float, ...] = (1.0, 2.0, 3.0),
    model: str = "test_erb",
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


def _erb_proton_flux_spectrum() -> Spectrum1D:
    return _erb_flux_spectrum(
        particle=Particle.PROTON,
        model="test_erb_proton",
    )


def _erb_electron_flux_spectrum() -> Spectrum1D:
    return _erb_flux_spectrum(
        particle=Particle.ELECTRON,
        model="test_erb_electron",
    )


def _run_erb_model(
    model: ErbModelProtocol,
    model_input: ErbModelInput,
) -> ErbModelResult:
    return model.calculate(model_input)


def test_validate_erb_energy_spectrum_accepts_proton_flux() -> None:
    validate_erb_energy_spectrum(_erb_proton_flux_spectrum())


def test_validate_erb_energy_spectrum_accepts_electron_flux() -> None:
    validate_erb_energy_spectrum(_erb_electron_flux_spectrum())


def test_validate_erb_energy_spectrum_accepts_peak_maximum_and_mean_flux() -> None:
    for quantity in (
        SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    ):
        validate_erb_energy_spectrum(
            _erb_flux_spectrum(quantity=quantity)
        )


def test_validate_erb_energy_spectrum_rejects_hze() -> None:
    with pytest.raises(ValueError, match="protons or electrons"):
        validate_erb_energy_spectrum(
            _erb_flux_spectrum(particle=Particle.HZE)
        )


def test_validate_erb_energy_spectrum_rejects_non_erb_source() -> None:
    with pytest.raises(ValueError, match="ERB"):
        validate_erb_energy_spectrum(
            _erb_flux_spectrum(source=RadiationSource.SEP)
        )


def test_validate_erb_energy_spectrum_rejects_fluence_quantity() -> None:
    with pytest.raises(ValueError, match="differential flux"):
        validate_erb_energy_spectrum(
            _erb_flux_spectrum(
                quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
                y_unit=Unit.DIFFERENTIAL_FLUENCE,
            )
        )


def test_validate_erb_energy_spectrum_rejects_non_mev_grid() -> None:
    with pytest.raises(ValueError, match="MeV"):
        validate_erb_energy_spectrum(
            _erb_flux_spectrum(x_unit=Unit.GEV_PER_NUCLEON)
        )


def test_validate_erb_energy_spectrum_rejects_bad_y_unit() -> None:
    with pytest.raises(ValueError, match="differential flux units"):
        validate_erb_energy_spectrum(
            _erb_flux_spectrum(y_unit=Unit.DIFFERENTIAL_FLUENCE)
        )


def test_validate_erb_energy_spectrum_rejects_zero_energy() -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_erb_energy_spectrum(
            _erb_flux_spectrum(x=(0.0, 1.0, 10.0))
        )


def test_erb_model_input_exposes_config_parameters() -> None:
    model_input = ErbModelInput(config=_config(lifetime_years=9, kp=4))

    assert model_input.lifetime_years == 9
    assert model_input.kp == 4


def test_static_erb_model_returns_configured_spectra() -> None:
    proton_spectrum = _erb_proton_flux_spectrum()
    electron_spectrum = _erb_electron_flux_spectrum()

    model = StaticErbModel(
        spectra=(proton_spectrum, electron_spectrum),
        model="static_test",
        document="test_document",
    )

    result = _run_erb_model(
        model=model,
        model_input=ErbModelInput(config=_config(lifetime_years=5, kp=3)),
    )

    assert result.spectra == (proton_spectrum, electron_spectrum)
    assert result.lifetime_years == 5
    assert result.kp == 3
    assert result.model == "static_test"
    assert result.document == "test_document"


def test_static_erb_model_rejects_empty_spectra() -> None:
    with pytest.raises(ValueError, match="at least one spectrum"):
        StaticErbModel(spectra=())


def test_static_erb_model_rejects_invalid_spectrum() -> None:
    with pytest.raises(ValueError, match="protons or electrons"):
        StaticErbModel(
            spectra=(
                _erb_flux_spectrum(particle=Particle.HZE),
            )
        )


def test_erb_model_result_requires_spectra() -> None:
    with pytest.raises(ValueError, match="at least one spectrum"):
        ErbModelResult(
            spectra=(),
            lifetime_years=5,
            kp=3,
            model="test",
            document="test",
        )


def test_erb_model_result_validates_kp() -> None:
    with pytest.raises(ValueError, match="0..9"):
        ErbModelResult(
            spectra=(_erb_proton_flux_spectrum(),),
            lifetime_years=5,
            kp=10,
            model="test",
            document="test",
        )


def test_erb_model_result_requires_model_name() -> None:
    with pytest.raises(ValueError, match="model"):
        ErbModelResult(
            spectra=(_erb_proton_flux_spectrum(),),
            lifetime_years=5,
            kp=3,
            model="",
            document="test",
        )


def test_erb_model_result_requires_document() -> None:
    with pytest.raises(ValueError, match="document"):
        ErbModelResult(
            spectra=(_erb_proton_flux_spectrum(),),
            lifetime_years=5,
            kp=3,
            model="test",
            document="",
        )
