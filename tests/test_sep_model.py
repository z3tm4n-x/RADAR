import math
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
from radar.sep.model import (
    SepModelInput,
    SepModelProtocol,
    GostSepModel,
    OstSepModel,
    SepModelResult,
    StaticSepModel,
    validate_sep_energy_spectrum,
    validate_sep_proton_fluence_spectrum,
)


def _sep_energy_spectrum(
    *,
    particle: Particle = Particle.PROTON,
    source: RadiationSource = RadiationSource.SEP,
    quantity: SpectrumQuantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    x_unit: Unit = Unit.MEV,
    y_unit: Unit = Unit.DIFFERENTIAL_FLUENCE,
    x: tuple[float, ...] = (10.0, 100.0, 1000.0),
    y: tuple[float, ...] = (1.0, 2.0, 3.0),
    model: str = "test_sep_energy",
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
    return _sep_energy_spectrum(
        particle=particle,
        source=source,
        quantity=quantity,
        x_unit=x_unit,
        y_unit=y_unit,
        x=x,
        y=y,
        model=model,
    )


def _sep_proton_peak_flux_spectrum() -> Spectrum1D:
    return _sep_energy_spectrum(
        particle=Particle.PROTON,
        quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        model="sep_proton_peak_flux",
    )


def _sep_proton_mean_flux_spectrum() -> Spectrum1D:
    return _sep_energy_spectrum(
        particle=Particle.PROTON,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        model="sep_proton_mean_flux",
    )


def _sep_hze_fluence_spectrum() -> Spectrum1D:
    return _sep_energy_spectrum(
        particle=Particle.HZE,
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
        model="sep_hze_fluence",
    )


def _sep_hze_peak_flux_spectrum() -> Spectrum1D:
    return _sep_energy_spectrum(
        particle=Particle.HZE,
        quantity=SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        model="sep_hze_peak_flux",
    )


def _sep_hze_mean_flux_spectrum() -> Spectrum1D:
    return _sep_energy_spectrum(
        particle=Particle.HZE,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        model="sep_hze_mean_flux",
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


def test_validate_sep_energy_spectrum_accepts_hze_energy_products() -> None:
    for spectrum in (
        _sep_hze_fluence_spectrum(),
        _sep_hze_peak_flux_spectrum(),
        _sep_hze_mean_flux_spectrum(),
    ):
        validate_sep_energy_spectrum(spectrum)


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

def test_sep_model_result_exposes_mission_fluence_product() -> None:
    from radar.core.types import RadiationProductKind

    result = StaticSepModel(
        annual_fluence_spectrum=_sep_proton_fluence_spectrum(),
        model="static_test",
        document="test_document",
    ).calculate(SepModelInput(mission=_mission(lifetime_years=5)))

    assert len(result.products) == 1
    assert result.product is result.products[0]
    assert result.product.kind is RadiationProductKind.MISSION_FLUENCE
    assert result.product.spectrum == result.spectrum


def test_sep_model_result_rejects_mismatched_product() -> None:
    from radar.core.products import SpectrumProduct
    from radar.core.types import RadiationProductKind

    spectrum = _sep_proton_fluence_spectrum()
    other_spectrum = _sep_proton_fluence_spectrum(model="other")

    with pytest.raises(ValueError, match="product spectrum"):
        SepModelResult(
            spectrum=spectrum,
            lifetime_years=5,
            exceedance_probability=0.1,
            model="test",
            document="test",
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.MISSION_FLUENCE,
                    spectrum=other_spectrum,
                ),
            ),
        )

def test_validate_sep_proton_fluence_spectrum_rejects_daily_display_unit() -> None:
    with pytest.raises(ValueError, match="differential fluence units"):
        validate_sep_proton_fluence_spectrum(
            _sep_proton_fluence_spectrum(
                y_unit=Unit.DIFFERENTIAL_FLUENCE_PER_DAY,
            )
        )

def test_sep_model_result_rejects_product_with_wrong_source() -> None:
    spectrum = _sep_proton_fluence_spectrum()
    wrong_source_spectrum = Spectrum1D(
        x=spectrum.x,
        y=spectrum.y,
        x_unit=spectrum.x_unit,
        y_unit=spectrum.y_unit,
        quantity=spectrum.quantity,
        particle=spectrum.particle,
        source=RadiationSource.GCR,
        model=spectrum.model,
    )

    with pytest.raises(ValueError, match="spectrum source"):
        SepModelResult(
            spectrum=spectrum,
            lifetime_years=5,
            exceedance_probability=0.9,
            model="test",
            document="test",
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.MISSION_FLUENCE,
                    spectrum=wrong_source_spectrum,
                ),
            ),
        )

def test_normative_sep_model_stub_metadata() -> None:
    from radar.core.profiles import (
        GOST_SEP_DOCUMENT,
        OST_134_1044_2007_DOCUMENT,
        SourceModelFamily,
    )
    from radar.core.types import RadiationSource
    from radar.sep.model import GostSepModel, OstSepModel

    ost_model = OstSepModel()
    gost_model = GostSepModel()

    assert ost_model.metadata.source is RadiationSource.SEP
    assert ost_model.metadata.model_family is SourceModelFamily.OST_134_1044_2007
    assert ost_model.metadata.document == OST_134_1044_2007_DOCUMENT

    assert gost_model.metadata.source is RadiationSource.SEP
    assert gost_model.metadata.model_family is SourceModelFamily.GOST_SEP
    assert gost_model.metadata.document == GOST_SEP_DOCUMENT


def test_normative_sep_model_stubs_raise_not_implemented() -> None:
    from radar.core.project import MissionConfig
    from radar.sep.model import GostSepModel, OstSepModel, SepModelInput

    model_input = SepModelInput(
        mission=MissionConfig(launch_year=2027, lifetime_years=5),
    )

    for model in (OstSepModel(), GostSepModel()):
        with pytest.raises(NotImplementedError, match="not implemented"):
            model.calculate(model_input)



def test_sep_model_result_accepts_multiple_source_products() -> None:
    mission_fluence = _sep_proton_fluence_spectrum()
    proton_peak = _sep_proton_peak_flux_spectrum()
    proton_mean = _sep_proton_mean_flux_spectrum()
    hze_fluence = _sep_hze_fluence_spectrum()
    hze_peak = _sep_hze_peak_flux_spectrum()
    hze_mean = _sep_hze_mean_flux_spectrum()

    mission_product = SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=mission_fluence,
    )

    result = SepModelResult(
        spectrum=mission_fluence,
        lifetime_years=5,
        exceedance_probability=0.1,
        model="test",
        document="test",
        products=(
            SpectrumProduct(
                kind=RadiationProductKind.PEAK_FLUX,
                spectrum=proton_peak,
            ),
            mission_product,
            SpectrumProduct(
                kind=RadiationProductKind.MEAN_FLUX,
                spectrum=proton_mean,
            ),
            SpectrumProduct(
                kind=RadiationProductKind.MISSION_FLUENCE,
                spectrum=hze_fluence,
            ),
            SpectrumProduct(
                kind=RadiationProductKind.PEAK_FLUX,
                spectrum=hze_peak,
            ),
            SpectrumProduct(
                kind=RadiationProductKind.MEAN_FLUX,
                spectrum=hze_mean,
            ),
        ),
    )

    assert result.spectrum == mission_fluence
    assert result.product == mission_product
    assert result.spectra == (
        proton_peak,
        mission_fluence,
        proton_mean,
        hze_fluence,
        hze_peak,
        hze_mean,
    )


def test_sep_model_result_requires_proton_mission_fluence_for_compatibility() -> None:
    mission_fluence = _sep_proton_fluence_spectrum()
    proton_peak = _sep_proton_peak_flux_spectrum()

    with pytest.raises(ValueError, match="proton mission fluence"):
        SepModelResult(
            spectrum=mission_fluence,
            lifetime_years=5,
            exceedance_probability=0.1,
            model="test",
            document="test",
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.PEAK_FLUX,
                    spectrum=proton_peak,
                ),
            ),
        )


def test_sep_model_result_rejects_let_products_before_shielding() -> None:
    mission_fluence = _sep_proton_fluence_spectrum()
    hze_let_flux = Spectrum1D(
        x=(1.0, 2.0, 3.0),
        y=(4.0, 5.0, 6.0),
        x_unit=Unit.LET,
        y_unit=Unit.DIFFERENTIAL_LET_FLUX,
        quantity=SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
        particle=Particle.HZE,
        source=RadiationSource.SEP,
        model="sep_hze_let_flux",
    )

    with pytest.raises(ValueError, match="energy fluence or flux"):
        SepModelResult(
            spectrum=mission_fluence,
            lifetime_years=5,
            exceedance_probability=0.1,
            model="test",
            document="test",
            products=(
                SpectrumProduct(
                    kind=RadiationProductKind.MISSION_FLUENCE,
                    spectrum=mission_fluence,
                ),
                SpectrumProduct(
                    kind=RadiationProductKind.PEAK_LET_FLUX,
                    spectrum=hze_let_flux,
                ),
            ),
        )



def _simple_ost_sep_proton_coefficient_records():
    from radar.sep.proton_spectrum import (
        SepProtonCoefficientName,
        SepProtonCoefficientRecord,
        SepProtonSpectrumProduct,
    )

    records = []
    values = {
        SepProtonSpectrumProduct.FLUENCE: {
            SepProtonCoefficientName.LOG10_C: 1.0,
            SepProtonCoefficientName.BREAK_ENERGY_MEV: 10.0,
            SepProtonCoefficientName.GAMMA1: 1.0,
            SepProtonCoefficientName.GAMMA2: 1.0,
        },
        SepProtonSpectrumProduct.PEAK_FLUX: {
            SepProtonCoefficientName.LOG10_C: 0.0,
            SepProtonCoefficientName.BREAK_ENERGY_MEV: 10.0,
            SepProtonCoefficientName.GAMMA1: 1.0,
            SepProtonCoefficientName.GAMMA2: 1.0,
        },
    }

    for product, parameters in values.items():
        for parameter, value in parameters.items():
            records.append(
                SepProtonCoefficientRecord(
                    model="ost_134_1044_2007",
                    product=product,
                    parameter=parameter,
                    event_count=2,
                    probability=0.5,
                    value=value,
                    source_table="test",
                )
            )

    return tuple(records)


def _simple_hze_ion_records():
    from radar.sep.hze_spectrum import OstSepHzeIonRecord

    return (
        OstSepHzeIonRecord(
            model="ost_134_1044_2007",
            z=2,
            symbol="He",
            mass_number=4.0,
            mass_to_charge=2.0,
            relative_abundance=1.23e-2,
            source_table="B.9",
        ),
    )


def _mission_with_probability(
    *,
    lifetime_years: int = 5,
    probability: float = 0.5,
) -> MissionConfig:
    return MissionConfig(
        launch_year=2027,
        lifetime_years=lifetime_years,
        sep_exceedance_probability=probability,
    )


def test_ost_sep_model_calculates_proton_source_products() -> None:
    model = OstSepModel(
        energy_grid_mev=(10.0,),
        version="protons_only_v1",
        monthly_smoothed_wolf_numbers=(2.0 / (60.0 * 0.0135),) * 60,
        coefficient_records=_simple_ost_sep_proton_coefficient_records(),
    )

    result = model.calculate(
        SepModelInput(
            mission=_mission_with_probability(lifetime_years=5, probability=0.5),
        )
    )

    products_by_kind = {product.kind: product for product in result.products}

    assert set(products_by_kind) == {
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.PEAK_FLUX,
        RadiationProductKind.MEAN_FLUX,
    }

    mission_product = products_by_kind[RadiationProductKind.MISSION_FLUENCE]
    peak_product = products_by_kind[RadiationProductKind.PEAK_FLUX]
    mean_product = products_by_kind[RadiationProductKind.MEAN_FLUX]

    assert result.spectrum == mission_product.spectrum
    assert result.product == mission_product

    assert mission_product.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert mission_product.spectrum.y_unit is Unit.DIFFERENTIAL_FLUENCE
    assert mission_product.spectrum.y == pytest.approx((10.0,))

    assert peak_product.spectrum.quantity is SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX
    assert peak_product.spectrum.y_unit is Unit.DIFFERENTIAL_FLUX
    assert peak_product.spectrum.y == pytest.approx((4.0 * math.pi,))

    duration_seconds = 5.0 * 365.25 * 24.0 * 60.0 * 60.0

    assert mean_product.spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert mean_product.spectrum.y_unit is Unit.DIFFERENTIAL_FLUX
    assert mean_product.spectrum.y == pytest.approx((10.0 / duration_seconds,))

    assert result.model == "ost_sep_model"
    assert result.document == "OST 134-1044-2007"


def test_gost_sep_model_calculates_proton_source_products_with_w1_policy() -> None:
    from radar.core.profiles import GOST_SEP_DOCUMENT
    from radar.sep.event_count import SepEventCountPolicy

    model = GostSepModel(
        energy_grid_mev=(10.0,),
        version="protons_only_v1",
        monthly_smoothed_wolf_numbers=(2.0 / (60.0 * 0.0130),) * 60,
        coefficient_records=_simple_ost_sep_proton_coefficient_records(),
    )

    result = model.calculate(
        SepModelInput(
            mission=_mission_with_probability(lifetime_years=5, probability=0.5),
        )
    )

    products_by_kind = {product.kind: product for product in result.products}

    assert model.event_count_policy is SepEventCountPolicy.GOST_R_25645_165_2025_W1_0
    assert result.model == "gost_sep_model"
    assert result.document == GOST_SEP_DOCUMENT
    assert result.spectrum.y == pytest.approx((10.0,))
    assert products_by_kind[RadiationProductKind.PEAK_FLUX].spectrum.y == pytest.approx(
        (4.0 * math.pi,)
    )


def test_ost_sep_model_uses_real_coefficient_tables() -> None:
    model = OstSepModel(
        energy_grid_mev=(10.0, 20.0),
        monthly_smoothed_wolf_numbers=(2.0 / (36.0 * 0.0135),) * 36,
    )

    result = model.calculate(
        SepModelInput(
            mission=_mission_with_probability(lifetime_years=3, probability=0.5),
        )
    )

    assert len(result.products) == 3
    assert result.spectrum.x == (10.0, 20.0)
    assert all(value > 0.0 for value in result.spectrum.y)


def test_ost_sep_model_rejects_partial_configuration() -> None:
    with pytest.raises(ValueError, match="both energy grid and Wolf number series"):
        OstSepModel(energy_grid_mev=(10.0,))


def test_ost_sep_model_rejects_bad_energy_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        OstSepModel(
            energy_grid_mev=(10.0, 10.0),
            monthly_smoothed_wolf_numbers=(100.0,),
        )


def test_ost_sep_model_rejects_negative_wolf_number() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        OstSepModel(
            energy_grid_mev=(10.0,),
            monthly_smoothed_wolf_numbers=(-1.0,),
        )



def test_ost_sep_model_rejects_wolf_number_series_length_mismatch() -> None:
    model = OstSepModel(
        energy_grid_mev=(10.0,),
        monthly_smoothed_wolf_numbers=(2.0 / 0.0135,),
        coefficient_records=_simple_ost_sep_proton_coefficient_records(),
        version="protons_only_v1",
    )

    with pytest.raises(ValueError, match="12 \\* mission lifetime years"):
        model.calculate(
            SepModelInput(
                mission=_mission_with_probability(lifetime_years=5, probability=0.5),
            )
        )


def test_ost_sep_model_product_labels_explain_derived_quantities() -> None:
    model = OstSepModel(
        energy_grid_mev=(10.0,),
        monthly_smoothed_wolf_numbers=(2.0 / (12.0 * 0.0135),) * 12,
        coefficient_records=_simple_ost_sep_proton_coefficient_records(),
        version="protons_only_v1",
    )

    result = model.calculate(
        SepModelInput(
            mission=_mission_with_probability(lifetime_years=1, probability=0.5),
        )
    )

    labels_by_kind = {product.kind: product.label for product in result.products}

    assert "4π times directional per-steradian peak flux" in labels_by_kind[
        RadiationProductKind.PEAK_FLUX
    ]
    assert "mission fluence divided by mission duration" in labels_by_kind[
        RadiationProductKind.MEAN_FLUX
    ]



def test_ost_sep_model_calculates_hze_source_products_when_configured() -> None:
    from radar.sep.hze_spectrum import (
        evaluate_ost_sep_hze_spectrum,
        ost_sep_hze_coefficients_from_proton,
    )
    from radar.sep.proton_spectrum import (
        SepProtonSpectrumCoefficients,
        directional_flux_to_omnidirectional_flux,
    )

    model = OstSepModel(
        energy_grid_mev=(10.0,),
        hze_energy_grid_mev_per_nucleon=(5.0,),
        version="protons_hze_v1",
        monthly_smoothed_wolf_numbers=(2.0 / (60.0 * 0.0135),) * 60,
        coefficient_records=_simple_ost_sep_proton_coefficient_records(),
        hze_ion_records=_simple_hze_ion_records(),
    )
    result = model.calculate(
        SepModelInput(
            mission=_mission_with_probability(lifetime_years=5, probability=0.5),
        )
    )

    hze_products = tuple(
        product
        for product in result.products
        if product.spectrum.particle is Particle.HZE
    )
    hze_products_by_kind = {product.kind: product for product in hze_products}

    assert len(result.products) == 6
    assert len(hze_products) == 3
    assert set(hze_products_by_kind) == {
        RadiationProductKind.MISSION_FLUENCE,
        RadiationProductKind.PEAK_FLUX,
        RadiationProductKind.MEAN_FLUX,
    }

    ion = _simple_hze_ion_records()[0]
    fluence_proton_coefficients = SepProtonSpectrumCoefficients(
        log10_c=1.0,
        break_energy_mev=10.0,
        gamma1=1.0,
        gamma2=1.0,
    )
    peak_proton_coefficients = SepProtonSpectrumCoefficients(
        log10_c=0.0,
        break_energy_mev=10.0,
        gamma1=1.0,
        gamma2=1.0,
    )
    fluence_hze_coefficients = ost_sep_hze_coefficients_from_proton(
        proton_coefficients=fluence_proton_coefficients,
        ion=ion,
    )
    peak_hze_coefficients = ost_sep_hze_coefficients_from_proton(
        proton_coefficients=peak_proton_coefficients,
        ion=ion,
    )

    expected_mission = evaluate_ost_sep_hze_spectrum(
        (5.0,),
        fluence_hze_coefficients,
    )
    expected_peak = directional_flux_to_omnidirectional_flux(
        evaluate_ost_sep_hze_spectrum(
            (5.0,),
            peak_hze_coefficients,
        )
    )
    duration_seconds = 5.0 * 365.25 * 24.0 * 60.0 * 60.0
    expected_mean = tuple(value / duration_seconds for value in expected_mission)

    mission_product = hze_products_by_kind[RadiationProductKind.MISSION_FLUENCE]
    peak_product = hze_products_by_kind[RadiationProductKind.PEAK_FLUX]
    mean_product = hze_products_by_kind[RadiationProductKind.MEAN_FLUX]

    assert mission_product.spectrum.x == (5.0,)
    assert mission_product.spectrum.x_unit is Unit.MEV
    assert mission_product.spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert mission_product.spectrum.y == pytest.approx(expected_mission)

    assert peak_product.spectrum.quantity is SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX
    assert peak_product.spectrum.y == pytest.approx(expected_peak)

    assert mean_product.spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert mean_product.spectrum.y == pytest.approx(expected_mean)

    assert "He Z=2" in mission_product.label
    assert "z02_he" in mission_product.spectrum.model


def test_ost_sep_model_rejects_bad_hze_energy_grid() -> None:
    with pytest.raises(ValueError, match="HZE energy grid must be strictly increasing"):
        OstSepModel(
            energy_grid_mev=(10.0,),
            hze_energy_grid_mev_per_nucleon=(5.0, 5.0),
            monthly_smoothed_wolf_numbers=(100.0,),
        )


def test_ost_sep_model_rejects_hze_grid_without_proton_configuration() -> None:
    with pytest.raises(ValueError, match="requires proton source configuration"):
        OstSepModel(
            hze_energy_grid_mev_per_nucleon=(5.0,),
        )
