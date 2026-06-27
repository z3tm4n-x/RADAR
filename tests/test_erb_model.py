import pytest

from radar.core.products import SpectrumProduct

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.erb.igrf import IgrfCoefficients
from radar.erb.model import (
    ErbModelInput,
    ErbModelProtocol,
    ErbModelResult,
    OstErbModel,
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


def _leo_ost_config(lifetime_years: int = 2) -> CalculationConfig:
    mission = MissionConfig(
        launch_year=2027,
        lifetime_years=lifetime_years,
    )
    orbit = OrbitConfig.circular(altitude_km=2_000.0, inclination_deg=0.0)

    return CalculationConfig(mission=mission, orbit=orbit, kp=3)


def _dipole_coefficients() -> IgrfCoefficients:
    return IgrfCoefficients(
        epoch=1985.0,
        g={(1, 0): -31_165.3},
        h={},
        nmax=1,
    )


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


def test_validate_erb_energy_spectrum_accepts_maximum_and_mean_flux() -> None:
    for quantity in (
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    ):
        validate_erb_energy_spectrum(
            _erb_flux_spectrum(quantity=quantity)
        )


def test_validate_erb_energy_spectrum_rejects_peak_flux() -> None:
    with pytest.raises(ValueError, match="supported differential flux"):
        validate_erb_energy_spectrum(
            _erb_flux_spectrum(quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX)
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


def test_validate_erb_energy_spectrum_accepts_fluence_quantity() -> None:
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

def test_erb_model_result_exposes_products() -> None:
    from radar.core.types import RadiationProductKind

    proton_spectrum = _erb_proton_flux_spectrum()
    electron_spectrum = _erb_electron_flux_spectrum()

    result = StaticErbModel(
        spectra=(proton_spectrum, electron_spectrum),
        model="static_test",
        document="test_document",
    ).calculate(ErbModelInput(config=_config(lifetime_years=5, kp=3)))

    assert result.spectra == (proton_spectrum, electron_spectrum)
    assert len(result.products) == 2
    assert result.products[0].kind is RadiationProductKind.ORBIT_AVERAGED_FLUX
    assert result.products[0].spectrum == proton_spectrum
    assert result.products[1].kind is RadiationProductKind.ORBIT_AVERAGED_FLUX
    assert result.products[1].spectrum == electron_spectrum


def test_erb_model_result_maps_maximum_and_mean_products() -> None:
    from radar.core.types import RadiationProductKind

    maximum_spectrum = _erb_flux_spectrum(
        quantity=SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
    )
    mean_spectrum = _erb_flux_spectrum(quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX)

    result = ErbModelResult(
        spectra=(maximum_spectrum, mean_spectrum),
        lifetime_years=5,
        kp=3,
        model="test",
        document="test",
    )

    assert tuple(product.kind for product in result.products) == (
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MEAN_FLUX,
    )


def test_erb_model_result_maps_fluence_product() -> None:
    from radar.core.types import RadiationProductKind

    fluence_spectrum = _erb_flux_spectrum(
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
    )

    result = ErbModelResult(
        spectra=(fluence_spectrum,),
        lifetime_years=5,
        kp=3,
        model="test",
        document="test",
    )

    assert result.products[0].kind is RadiationProductKind.MISSION_FLUENCE


def test_erb_model_result_rejects_mismatched_products() -> None:
    from radar.core.products import SpectrumProduct
    from radar.core.types import RadiationProductKind

    spectrum = _erb_proton_flux_spectrum()
    other_spectrum = _erb_electron_flux_spectrum()

    with pytest.raises(ValueError, match="product spectra"):
        ErbModelResult(
            spectra=(spectrum,),
            lifetime_years=5,
            kp=3,
            model="test",
            document="test",
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.ORBIT_AVERAGED_FLUX,
                    spectrum=other_spectrum,
                ),
            ),
        )

def test_erb_model_result_rejects_product_with_wrong_source() -> None:
    spectrum = _erb_flux_spectrum(
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
    )
    wrong_source_spectrum = Spectrum1D(
        x=spectrum.x,
        y=spectrum.y,
        x_unit=spectrum.x_unit,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=RadiationSource.SEP,
        model=spectrum.model,
    )

    with pytest.raises(ValueError, match="spectrum source"):
        ErbModelResult(
            spectra=(spectrum,),
            lifetime_years=5,
            kp=2,
            model="test",
            document="test",
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.ORBIT_AVERAGED_FLUX,
                    spectrum=wrong_source_spectrum,
                ),
            ),
        )

def test_normative_erb_model_stub_metadata() -> None:
    from radar.core.profiles import OST_134_1044_2007_DOCUMENT, SourceModelFamily
    from radar.core.types import RadiationSource
    from radar.erb.model import OstErbModel

    model = OstErbModel()

    assert model.metadata.source is RadiationSource.ERB
    assert model.metadata.model_family is SourceModelFamily.OST_134_1044_2007
    assert model.metadata.document == OST_134_1044_2007_DOCUMENT


def test_normative_ost_erb_model_calculates_products() -> None:
    from radar.core.types import RadiationProductKind

    model = OstErbModel(
        anomaly_samples=2,
        node_samples=2,
        igrf_coefficients=_dipole_coefficients(),
    )

    result = model.calculate(
        ErbModelInput(
            config=_leo_ost_config(lifetime_years=2),
        )
    )

    assert result.model == "ost_erb_model"
    assert result.lifetime_years == 2
    assert len(result.spectra) == 6
    assert tuple(product.kind for product in result.products) == (
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.MEAN_FLUX,
        RadiationProductKind.MAXIMUM_FLUX,
        RadiationProductKind.MISSION_FLUENCE,
    )
    assert result.spectra[0].particle is Particle.PROTON
    assert result.spectra[3].particle is Particle.ELECTRON
    assert result.spectra[2].quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert result.spectra[2].y_unit is Unit.DIFFERENTIAL_FLUENCE
    assert any(value > 0.0 for value in result.spectra[0].y)
    assert any(value > 0.0 for value in result.spectra[3].y)

def test_ost_erb_model_records_appendix_e_method_metadata() -> None:
    model = OstErbModel(
        anomaly_samples=2,
        node_samples=2,
        igrf_coefficients=_dipole_coefficients(),
    )

    result = model.calculate(
        ErbModelInput(
            config=_leo_ost_config(lifetime_years=2),
        )
    )

    metadata = dict(result.method_metadata)

    assert metadata["ost_appendices"] == "A,D,E"
    assert "Appendix A" in metadata["appendix_a_reference"]
    assert "Appendix D" in metadata["appendix_d_reference"]
    assert "Appendix E" in metadata["appendix_e_reference"]
    assert metadata["appendix_e_interpolation"] == "three_point_lagrange_with_log_positive_flux_values"
    assert metadata["appendix_e_averaging"] == "equal_weight_mean_anomaly_samples"
    assert (
        metadata["appendix_e_time_weighting_status"]
        == "approximated_by_uniform_mean_anomaly_sampling"
    )
    assert metadata["solar_reference_start_year"] == "2027"
    assert metadata["anomaly_samples"] == "2"
    assert metadata["node_samples"] == "2"
    assert metadata["peak_state"] == "worst"


def test_ost_erb_model_allows_explicit_solar_reference_start_year() -> None:
    model = OstErbModel(
        anomaly_samples=2,
        node_samples=2,
        igrf_coefficients=_dipole_coefficients(),
        solar_reference_start_year=2024,
    )

    result = model.calculate(
        ErbModelInput(
            config=_leo_ost_config(lifetime_years=2),
        )
    )

    metadata = dict(result.method_metadata)

    assert metadata["solar_reference_start_year"] == "2024"

def test_ost_erb_model_records_orbit_table_coverage_diagnostics() -> None:
    model = OstErbModel(
        anomaly_samples=2,
        node_samples=2,
        igrf_coefficients=_dipole_coefficients(),
    )

    result = model.calculate(
        ErbModelInput(
            config=_leo_ost_config(lifetime_years=2),
        )
    )

    metadata = dict(result.method_metadata)

    assert metadata["proton_total_samples"] == "4"
    assert metadata["electron_total_samples"] == "4"
    assert int(metadata["proton_valid_samples"]) > 0
    assert int(metadata["electron_valid_samples"]) > 0
    assert (
        int(metadata["proton_valid_samples"])
        + int(metadata["proton_invalid_samples"])
        == 4
    )
    assert (
        int(metadata["electron_valid_samples"])
        + int(metadata["electron_invalid_samples"])
        == 4
    )
    assert float(metadata["proton_valid_sample_fraction"]) > 0.0
    assert float(metadata["electron_valid_sample_fraction"]) > 0.0
    assert float(metadata["proton_table_l_min"]) > 0.0
    assert float(metadata["electron_table_l_min"]) > 0.0


def test_ost_erb_model_fluence_matches_mean_flux_times_mission_duration() -> None:
    from radar.erb.constants import ERB_SECONDS_PER_YEAR

    lifetime_years = 2
    model = OstErbModel(
        anomaly_samples=2,
        node_samples=2,
        igrf_coefficients=_dipole_coefficients(),
    )

    result = model.calculate(
        ErbModelInput(
            config=_leo_ost_config(lifetime_years=lifetime_years),
        )
    )

    mission_seconds = lifetime_years * ERB_SECONDS_PER_YEAR
    proton_mean_flux = result.spectra[0]
    proton_fluence = result.spectra[2]
    electron_mean_flux = result.spectra[3]
    electron_fluence = result.spectra[5]

    assert proton_fluence.y == pytest.approx(
        tuple(value * mission_seconds for value in proton_mean_flux.y)
    )
    assert electron_fluence.y == pytest.approx(
        tuple(value * mission_seconds for value in electron_mean_flux.y)
    )

def test_erb_integral_spectrum_validates_inputs() -> None:
    from radar.erb.model import ErbIntegralSpectrum

    with pytest.raises(ValueError, match="protons or electrons"):
        ErbIntegralSpectrum(
            energies_mev=(1.0,),
            integral_flux_gt_e=(1.0,),
            particle=Particle.HZE,
            model="test",
            document="test",
        )

    with pytest.raises(ValueError, match="equal length"):
        ErbIntegralSpectrum(
            energies_mev=(1.0, 2.0),
            integral_flux_gt_e=(1.0,),
            particle=Particle.PROTON,
            model="test",
            document="test",
        )

    with pytest.raises(ValueError, match="sorted"):
        ErbIntegralSpectrum(
            energies_mev=(2.0, 1.0),
            integral_flux_gt_e=(1.0, 1.0),
            particle=Particle.PROTON,
            model="test",
            document="test",
        )

    with pytest.raises(ValueError, match="non-negative"):
        ErbIntegralSpectrum(
            energies_mev=(1.0, 2.0),
            integral_flux_gt_e=(1.0, -1.0),
            particle=Particle.PROTON,
            model="test",
            document="test",
        )


def test_erb_model_result_rejects_integral_spectrum_with_wrong_document() -> None:
    from radar.erb.model import ErbIntegralSpectrum

    spectrum = _erb_proton_flux_spectrum()
    integral_spectrum = ErbIntegralSpectrum(
        energies_mev=spectrum.x,
        integral_flux_gt_e=(3.0, 2.0, 1.0),
        particle=Particle.PROTON,
        model="test",
        document="other_document",
    )

    with pytest.raises(ValueError, match="document"):
        ErbModelResult(
            spectra=(spectrum,),
            lifetime_years=5,
            kp=3,
            model="test",
            document="test_document",
            integral_spectra=(integral_spectrum,),
        )


def test_ost_erb_model_returns_appendix_e_integral_spectra() -> None:
    model = OstErbModel(
        anomaly_samples=2,
        node_samples=2,
        igrf_coefficients=_dipole_coefficients(),
    )

    result = model.calculate(
        ErbModelInput(
            config=_leo_ost_config(lifetime_years=2),
        )
    )

    assert len(result.integral_spectra) == 2

    proton_integral = result.integral_spectra[0]
    electron_integral = result.integral_spectra[1]

    assert proton_integral.particle is Particle.PROTON
    assert electron_integral.particle is Particle.ELECTRON
    assert proton_integral.energies_mev == result.spectra[0].x
    assert electron_integral.energies_mev == result.spectra[3].x
    assert proton_integral.document == result.document
    assert electron_integral.document == result.document
    assert any(value > 0.0 for value in proton_integral.integral_flux_gt_e)
    assert any(value > 0.0 for value in electron_integral.integral_flux_gt_e)
    assert all(
        left >= right
        for left, right in zip(
            proton_integral.integral_flux_gt_e,
            proton_integral.integral_flux_gt_e[1:],
            strict=False,
        )
    )
    assert all(
        left >= right
        for left, right in zip(
            electron_integral.integral_flux_gt_e,
            electron_integral.integral_flux_gt_e[1:],
            strict=False,
        )
    )


def test_ost_erb_model_records_appendix_e_integral_metadata() -> None:
    model = OstErbModel(
        anomaly_samples=2,
        node_samples=2,
        igrf_coefficients=_dipole_coefficients(),
    )

    result = model.calculate(
        ErbModelInput(
            config=_leo_ost_config(lifetime_years=2),
        )
    )

    metadata = dict(result.method_metadata)

    assert metadata["appendix_e_integral_spectrum"] == "tail_power_law_numeric_quadrature"
