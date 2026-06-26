import pytest

from radar.core.products import SpectrumProduct

from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.gcr.components import GostGcrSolarState, OstGcrSolarState
from radar.gcr.model import (
    GCR_MODEL_SECONDS_PER_YEAR,
    GcrModelInput,
    GcrModelProtocol,
    GcrModelResult,
    GostGcrModel,
    OstGcrModel,
    StaticGcrModel,
    validate_gcr_energy_spectrum,
)


def _gcr_proton_flux_spectrum(
    *,
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.GCR,
    quantity: SpectrumQuantity = SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
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
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
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

def test_gcr_model_result_exposes_products() -> None:
    from radar.core.types import RadiationProductKind

    proton_spectrum = _gcr_proton_flux_spectrum()
    hze_spectrum = _gcr_hze_flux_spectrum()

    result = StaticGcrModel(
        spectra=(proton_spectrum, hze_spectrum),
        model="static_test",
        document="test_document",
    ).calculate(GcrModelInput(mission=_mission(lifetime_years=5)))

    assert result.spectra == (proton_spectrum, hze_spectrum)
    assert len(result.products) == 2
    assert result.products[0].kind is RadiationProductKind.MEAN_FLUX
    assert result.products[0].spectrum == proton_spectrum
    assert result.products[1].kind is RadiationProductKind.MEAN_FLUX
    assert result.products[1].spectrum == hze_spectrum


def test_gcr_model_result_rejects_mismatched_products() -> None:
    from radar.core.products import SpectrumProduct
    from radar.core.types import RadiationProductKind

    spectrum = _gcr_proton_flux_spectrum()
    other_spectrum = _gcr_hze_flux_spectrum()

    with pytest.raises(ValueError, match="product spectra"):
        GcrModelResult(
            spectra=(spectrum,),
            lifetime_years=5,
            model="test",
            document="test",
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.MEAN_FLUX,
                    spectrum=other_spectrum,
                ),
            ),
        )

def test_gcr_model_result_rejects_ambiguous_differential_flux_product() -> None:
    spectrum = _gcr_proton_flux_spectrum(
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
    )

    with pytest.raises(ValueError, match="without specifying mean or maximum flux"):
        GcrModelResult(
            spectra=(spectrum,),
            lifetime_years=5,
            model="test",
            document="test",
        )

def test_validate_gcr_energy_spectrum_rejects_daily_display_fluence_unit() -> None:
    with pytest.raises(ValueError, match="differential fluence units"):
        validate_gcr_energy_spectrum(
            _gcr_proton_flux_spectrum(
                quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
                y_unit=Unit.DIFFERENTIAL_FLUENCE_PER_DAY,
            )
        )

def test_gcr_model_result_rejects_product_with_wrong_source() -> None:
    spectrum = _gcr_proton_flux_spectrum(
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
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
        GcrModelResult(
            spectra=(spectrum,),
            lifetime_years=5,
            model="test",
            document="test",
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.MEAN_FLUX,
                    spectrum=wrong_source_spectrum,
                ),
            ),
        )

def test_normative_gcr_model_metadata() -> None:
    from radar.core.profiles import (
        GOST_GCR_DOCUMENT,
        OST_134_1044_2007_DOCUMENT,
        SourceModelFamily,
    )

    ost_model = OstGcrModel(symbols=("H",))
    gost_model = GostGcrModel(symbols=("H",))

    assert ost_model.metadata.source is RadiationSource.GCR
    assert ost_model.metadata.model_family is SourceModelFamily.OST_134_1044_2007
    assert ost_model.metadata.document == OST_134_1044_2007_DOCUMENT
    assert ost_model.metadata.version == "source_spectra_outside_magnetosphere_v1"

    assert gost_model.metadata.source is RadiationSource.GCR
    assert gost_model.metadata.model_family is SourceModelFamily.GOST_GCR
    assert gost_model.metadata.document == GOST_GCR_DOCUMENT
    assert gost_model.metadata.version == "source_spectra_outside_magnetosphere_v1"


def test_gost_gcr_model_calculates_outside_magnetosphere_products() -> None:
    model = GostGcrModel(
        energy_grid_mev_per_nucleon=(10.0, 100.0),
        solar_state=GostGcrSolarState(wolf_number=70.0, wolf_version="1.0"),
        nek_wolf_number=70.0,
        akl_years_from_cycle_start=2.0,
        symbols=("H", "He"),
        model="gost_test",
        document="gost_document",
    )
    model_input = GcrModelInput(mission=_mission(lifetime_years=2))

    source_spectra = model.calculate_source_spectra()
    result = model.calculate(model_input)

    assert result.model == "gost_test"
    assert result.document == "gost_document"
    assert result.lifetime_years == 2
    assert len(result.products) == 6
    assert result.spectra == tuple(product.spectrum for product in result.products)

    proton_source = source_spectra.ion("H").total
    duration = 2.0 * GCR_MODEL_SECONDS_PER_YEAR

    assert result.products[0].kind is RadiationProductKind.MEAN_FLUX
    assert result.products[0].spectrum.particle is Particle.PROTON
    assert result.products[0].spectrum.y == pytest.approx(proton_source.y)

    assert result.products[1].kind is RadiationProductKind.MAXIMUM_FLUX
    assert result.products[1].spectrum.y == pytest.approx(proton_source.y)

    assert result.products[2].kind is RadiationProductKind.MISSION_FLUENCE
    assert result.products[2].spectrum.y == pytest.approx(
        tuple(value * duration for value in proton_source.y)
    )

    assert result.products[3].spectrum.particle is Particle.HZE


def test_ost_gcr_model_calculates_outside_magnetosphere_products() -> None:
    model = OstGcrModel(
        energy_grid_mev_per_nucleon=(10.0, 100.0),
        solar_state=OstGcrSolarState(
            wolf_current=60.0,
            wolf_lagged=50.0,
            wolf_min=10.0,
            wolf_max=100.0,
            cycle_number=24,
            after_polarity_reversal=True,
        ),
        nek_wolf_number=60.0,
        akl_years_from_cycle_start=2.0,
        symbols=("O",),
        model="ost_test",
        document="ost_document",
    )
    model_input = GcrModelInput(mission=_mission(lifetime_years=3))

    source_spectra = model.calculate_source_spectra()
    result = model.calculate(model_input)

    assert result.model == "ost_test"
    assert result.document == "ost_document"
    assert len(result.products) == 3

    source_total = source_spectra.ion("O").total
    duration = 3.0 * GCR_MODEL_SECONDS_PER_YEAR

    assert result.products[0].kind is RadiationProductKind.MEAN_FLUX
    assert result.products[0].spectrum.particle is Particle.HZE
    assert result.products[0].spectrum.y == pytest.approx(source_total.y)

    assert result.products[1].kind is RadiationProductKind.MAXIMUM_FLUX
    assert result.products[1].spectrum.y == pytest.approx(source_total.y)

    assert result.products[2].kind is RadiationProductKind.MISSION_FLUENCE
    assert result.products[2].spectrum.y == pytest.approx(
        tuple(value * duration for value in source_total.y)
    )


def test_normative_gcr_models_reject_bad_source_spectrum_configuration() -> None:
    with pytest.raises(ValueError, match="energy grid"):
        GostGcrModel(
            energy_grid_mev_per_nucleon=(10.0, 10.0),
            symbols=("H",),
        )

    with pytest.raises(ValueError, match="Unknown GCR ion symbol"):
        OstGcrModel(
            energy_grid_mev_per_nucleon=(10.0,),
            symbols=("Xx",),
        )
