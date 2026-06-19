import pytest

from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import Particle, RadiationSource, SpectrumQuantity
from radar.core.units import Unit
from radar.gcr.model import (
    GcrModelInput,
    GcrModelProtocol,
    GcrModelResult,
    StaticGcrModel,
    validate_gcr_energy_spectrum,
)


def _gcr_proton_flux_spectrum(
    *,
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.GCR,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUX,
    x_unit: Unit = Unit.MEV,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUX,
    x: tuple[float, ...] = (10.0, 100.0, 1000.0),
    y: tuple[float, ...] = (1.0, 2.0, 3.0),
    model: str = "test_gcr_proton",
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


def _gcr_hze_flux_spectrum() -> Spectrum1D:
    return Spectrum1D(
        x=(0.1, 1.0, 10.0),
        y=(4.0, 5.0, 6.0),
        x_unit=Unit.GEV_PER_NUCLEON,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
        particle=Particle.HZE,
        source=RadiationSource.GCR,
        model="test_gcr_hze",
    )


def _mission(lifetime_years: int = 7) -> MissionConfig:
    return MissionConfig(
        launch_year=2027,
        lifetime_years=lifetime_years,
    )


def _run_gcr_model(
    model: GcrModelProtocol,
    model_input: GcrModelInput,
) -> GcrModelResult:
    return model.calculate(model_input)


def test_validate_gcr_energy_spectrum_accepts_proton_flux() -> None:
    validate_gcr_energy_spectrum(_gcr_proton_flux_spectrum())


def test_validate_gcr_energy_spectrum_accepts_hze_flux() -> None:
    validate_gcr_energy_spectrum(_gcr_hze_flux_spectrum())


def test_validate_gcr_energy_spectrum_accepts_fluence_quantity() -> None:
    spectrum = _gcr_proton_flux_spectrum(
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
    )

    validate_gcr_energy_spectrum(spectrum)


def test_validate_gcr_energy_spectrum_rejects_electrons() -> None:
    with pytest.raises(ValueError, match="electrons"):
        validate_gcr_energy_spectrum(
            _gcr_proton_flux_spectrum(particle=Particle.ELECTRON)
        )


def test_validate_gcr_energy_spectrum_rejects_non_gcr_source() -> None:
    with pytest.raises(ValueError, match="GCR"):
        validate_gcr_energy_spectrum(
            _gcr_proton_flux_spectrum(source=RadiationSource.SEP)
        )


def test_validate_gcr_energy_spectrum_rejects_let_quantity() -> None:
    with pytest.raises(ValueError, match="energy flux or fluence"):
        validate_gcr_energy_spectrum(
            _gcr_proton_flux_spectrum(
                quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
                y_unit=Unit.DIFFERENTIAL_LET_FLUX,
            )
        )


def test_validate_gcr_energy_spectrum_rejects_bad_x_unit() -> None:
    with pytest.raises(ValueError, match="MeV or GeV/nucleon"):
        validate_gcr_energy_spectrum(
            _gcr_proton_flux_spectrum(x_unit=Unit.LET)
        )


def test_validate_gcr_energy_spectrum_rejects_bad_flux_y_unit() -> None:
    with pytest.raises(ValueError, match="differential flux units"):
        validate_gcr_energy_spectrum(
            _gcr_proton_flux_spectrum(y_unit=Unit.DIFFERENTIAL_FLUENCE)
        )


def test_validate_gcr_energy_spectrum_rejects_bad_fluence_y_unit() -> None:
    with pytest.raises(ValueError, match="differential fluence units"):
        validate_gcr_energy_spectrum(
            _gcr_proton_flux_spectrum(
                quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
                y_unit=Unit.DIFFERENTIAL_FLUX,
            )
        )


def test_validate_gcr_energy_spectrum_rejects_zero_energy() -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_gcr_energy_spectrum(
            _gcr_proton_flux_spectrum(x=(0.0, 10.0, 100.0))
        )


def test_gcr_model_input_exposes_mission_parameters() -> None:
    model_input = GcrModelInput(mission=_mission(lifetime_years=9))

    assert model_input.launch_year == 2027
    assert model_input.lifetime_years == 9


def test_static_gcr_model_returns_configured_spectra() -> None:
    proton_spectrum = _gcr_proton_flux_spectrum()
    hze_spectrum = _gcr_hze_flux_spectrum()

    model = StaticGcrModel(
        spectra=(proton_spectrum, hze_spectrum),
        model="static_test",
        document="test_document",
    )

    result = _run_gcr_model(
        model=model,
        model_input=GcrModelInput(mission=_mission(lifetime_years=5)),
    )

    assert result.spectra == (proton_spectrum, hze_spectrum)
    assert result.lifetime_years == 5
    assert result.model == "static_test"
    assert result.document == "test_document"


def test_static_gcr_model_rejects_empty_spectra() -> None:
    with pytest.raises(ValueError, match="at least one spectrum"):
        StaticGcrModel(spectra=())


def test_static_gcr_model_rejects_invalid_spectrum() -> None:
    with pytest.raises(ValueError, match="electrons"):
        StaticGcrModel(
            spectra=(
                _gcr_proton_flux_spectrum(particle=Particle.ELECTRON),
            )
        )


def test_gcr_model_result_requires_spectra() -> None:
    with pytest.raises(ValueError, match="at least one spectrum"):
        GcrModelResult(
            spectra=(),
            lifetime_years=5,
            model="test",
            document="test",
        )


def test_gcr_model_result_requires_model_name() -> None:
    with pytest.raises(ValueError, match="model"):
        GcrModelResult(
            spectra=(_gcr_proton_flux_spectrum(),),
            lifetime_years=5,
            model="",
            document="test",
        )


def test_gcr_model_result_requires_document() -> None:
    with pytest.raises(ValueError, match="document"):
        GcrModelResult(
            spectra=(_gcr_proton_flux_spectrum(),),
            lifetime_years=5,
            model="test",
            document="",
        )
