import pytest

from radar.core.products import SpectrumProduct

from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.gcr.components import GostGcrSolarState, OstGcrSolarState, ost_gcr_delay_months
from radar.gcr.kinematics import gcr_rigidity_gv_from_mev_per_nucleon
from radar.gcr.model import (
    GCR_MODEL_SECONDS_PER_YEAR,
    OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX,
    GcrModelInput,
    GcrModelProtocol,
    GcrModelResult,
    GostGcrModel,
    OstGcrModel,
    StaticGcrModel,
    gost_gcr_delay_months_for_cycle,
    gost_gcr_solar_state_for_timeline_month,
    ost_gcr_delay_months_for_timeline_energy,
    ost_gcr_solar_state_for_timeline_energy,
    validate_gcr_energy_spectrum,
)
from radar.solar_activity.model import build_mission_solar_activity
from radar.solar_activity.ost import ost_wolf_number_cycle_table


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


def test_validate_gcr_energy_spectrum_rejects_typed_hze_total_mev() -> None:
    spectrum = Spectrum1D(
        x=(10.0, 100.0, 1000.0),
        y=(4.0, 5.0, 6.0),
        x_unit=Unit.MEV,
        y_unit=Unit.DIFFERENTIAL_FLUX,
        quantity=SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        particle=Particle.HZE,
        source=RadiationSource.GCR,
        model="typed_hze_total_mev",
        axis_kind=SpectrumAxisKind.KINETIC_ENERGY,
    )

    with pytest.raises(ValueError, match="Typed GCR HZE.*MeV/nucleon"):
        validate_gcr_energy_spectrum(spectrum)


def test_validate_gcr_energy_spectrum_accepts_fluence_quantity() -> None:
    spectrum = _gcr_proton_flux_spectrum(
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        y_unit=Unit.DIFFERENTIAL_FLUENCE,
    )

    validate_gcr_energy_spectrum(spectrum)


def test_validate_gcr_energy_spectrum_accepts_integral_quantities() -> None:
    validate_gcr_energy_spectrum(
        _gcr_proton_flux_spectrum(
            quantity=SpectrumQuantity.MEAN_INTEGRAL_FLUX,
            y_unit=Unit.INTEGRAL_FLUX,
        )
    )
    validate_gcr_energy_spectrum(
        _gcr_proton_flux_spectrum(
            quantity=SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX,
            y_unit=Unit.INTEGRAL_FLUX,
        )
    )
    validate_gcr_energy_spectrum(
        _gcr_proton_flux_spectrum(
            quantity=SpectrumQuantity.INTEGRAL_FLUENCE,
            y_unit=Unit.INTEGRAL_FLUENCE,
        )
    )


def test_validate_gcr_energy_spectrum_rejects_electrons() -> None:
    with pytest.raises(ValueError, match="electrons"):
        validate_gcr_energy_spectrum(_gcr_proton_flux_spectrum(particle=Particle.ELECTRON))


def test_validate_gcr_energy_spectrum_rejects_non_gcr_source() -> None:
    with pytest.raises(ValueError, match="GCR"):
        validate_gcr_energy_spectrum(_gcr_proton_flux_spectrum(source=RadiationSource.SEP))


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
        validate_gcr_energy_spectrum(_gcr_proton_flux_spectrum(x_unit=Unit.LET))


def test_validate_gcr_energy_spectrum_rejects_bad_flux_y_unit() -> None:
    with pytest.raises(ValueError, match="differential flux units"):
        validate_gcr_energy_spectrum(_gcr_proton_flux_spectrum(y_unit=Unit.DIFFERENTIAL_FLUENCE))


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
        validate_gcr_energy_spectrum(_gcr_proton_flux_spectrum(x=(0.0, 10.0, 100.0)))


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
        StaticGcrModel(spectra=(_gcr_proton_flux_spectrum(particle=Particle.ELECTRON),))


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
    from radar.gcr.source_spectra import GOST_GCR_SOURCE_VERSION, OST_GCR_SOURCE_VERSION

    ost_model = OstGcrModel(symbols=("H",))
    gost_model = GostGcrModel(symbols=("H",))

    assert ost_model.metadata.source is RadiationSource.GCR
    assert ost_model.metadata.model_family is SourceModelFamily.OST_134_1044_2007
    assert ost_model.metadata.document == OST_134_1044_2007_DOCUMENT
    assert ost_model.metadata.version == OST_GCR_SOURCE_VERSION

    assert gost_model.metadata.source is RadiationSource.GCR
    assert gost_model.metadata.model_family is SourceModelFamily.GOST_GCR
    assert gost_model.metadata.document == GOST_GCR_DOCUMENT
    assert gost_model.metadata.version == GOST_GCR_SOURCE_VERSION


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
    assert len(result.products) == 12
    assert result.spectra == tuple(product.spectrum for product in result.products)

    proton_source = source_spectra.ion("H").total
    duration = 2.0 * GCR_MODEL_SECONDS_PER_YEAR

    assert result.products[0].kind is RadiationProductKind.MEAN_FLUX
    assert result.products[0].spectrum.particle is Particle.PROTON
    assert result.products[0].spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert result.products[0].spectrum.y == pytest.approx(proton_source.y)

    assert result.products[1].kind is RadiationProductKind.MEAN_FLUX
    assert result.products[1].spectrum.quantity is SpectrumQuantity.MEAN_INTEGRAL_FLUX

    assert result.products[2].kind is RadiationProductKind.MAXIMUM_FLUX
    assert result.products[2].spectrum.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
    assert result.products[2].spectrum.y == pytest.approx(proton_source.y)

    assert result.products[3].kind is RadiationProductKind.MAXIMUM_FLUX
    assert result.products[3].spectrum.quantity is SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX

    assert result.products[4].kind is RadiationProductKind.MISSION_FLUENCE
    assert result.products[4].spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert result.products[4].spectrum.y == pytest.approx(
        tuple(value * duration for value in proton_source.y)
    )

    assert result.products[5].kind is RadiationProductKind.MISSION_FLUENCE
    assert result.products[5].spectrum.quantity is SpectrumQuantity.INTEGRAL_FLUENCE

    assert result.products[6].spectrum.particle is Particle.HZE


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
    assert len(result.products) == 6

    source_total = source_spectra.ion("O").total
    duration = 3.0 * GCR_MODEL_SECONDS_PER_YEAR

    assert result.products[0].kind is RadiationProductKind.MEAN_FLUX
    assert result.products[0].spectrum.particle is Particle.HZE
    assert result.products[0].spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert result.products[0].spectrum.y == pytest.approx(source_total.y)

    assert result.products[1].kind is RadiationProductKind.MEAN_FLUX
    assert result.products[1].spectrum.quantity is SpectrumQuantity.MEAN_INTEGRAL_FLUX

    assert result.products[2].kind is RadiationProductKind.MAXIMUM_FLUX
    assert result.products[2].spectrum.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
    assert result.products[2].spectrum.y == pytest.approx(source_total.y)

    assert result.products[3].kind is RadiationProductKind.MAXIMUM_FLUX
    assert result.products[3].spectrum.quantity is SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX

    assert result.products[4].kind is RadiationProductKind.MISSION_FLUENCE
    assert result.products[4].spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert result.products[4].spectrum.y == pytest.approx(
        tuple(value * duration for value in source_total.y)
    )


def test_ost_gcr_subset_marks_source_and_mission_products_incomplete() -> None:
    model = OstGcrModel(
        energy_grid_mev_per_nucleon=(1.0, 10.0**4.9),
        symbols=("H", "Fe"),
    )

    source = model.calculate_source_spectra()
    result = model.calculate(GcrModelInput(mission=_mission(lifetime_years=1)))

    assert all(
        spectrum.completeness is SpectrumCompleteness.INCOMPLETE
        for spectrum in source.all_component_spectra
    )
    assert all(
        any(
            reason.startswith(OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX)
            and "He" in reason
            and "U" in reason
            for reason in spectrum.incomplete_reasons
        )
        for spectrum in source.all_component_spectra
    )
    assert all(
        product.spectrum.completeness is SpectrumCompleteness.INCOMPLETE
        and any(
            reason.startswith(OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX)
            for reason in product.spectrum.incomplete_reasons
        )
        for product in result.products
    )


def test_ost_gcr_full_roster_has_no_ion_roster_blocker() -> None:
    model = OstGcrModel(
        energy_grid_mev_per_nucleon=(1.0, 10.0**4.9),
    )

    source = model.calculate_source_spectra()

    assert len(source.ions) == 92
    assert all(
        not any(
            reason.startswith(OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX)
            for reason in spectrum.incomplete_reasons
        )
        for spectrum in source.all_component_spectra
    )


def test_custom_gcr_subset_is_not_subject_to_ost_roster_contract() -> None:
    custom_spectrum = _gcr_proton_flux_spectrum()

    result = StaticGcrModel(spectra=(custom_spectrum,)).calculate(
        GcrModelInput(mission=_mission(lifetime_years=1))
    )

    assert len(result.spectra) == 1
    assert not any(
        reason.startswith(OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX)
        for reason in result.spectra[0].incomplete_reasons
    )


def _calendar_timeline(*, cycle_number: int = 25, lifetime_years: int = 2):
    mission = MissionConfig(launch_year=2020, lifetime_years=lifetime_years)
    return build_mission_solar_activity(
        mission=mission,
        cycle_table=ost_wolf_number_cycle_table(mission.solar_activity_level),
        reference_start_year=2020,
        reference_cycle_number=cycle_number,
    )


def test_gost_gcr_delay_uses_numbered_cycle_parity() -> None:
    assert gost_gcr_delay_months_for_cycle(24) == 5.5
    assert gost_gcr_delay_months_for_cycle(25) == 15.5


def test_gost_gcr_state_samples_required_premission_lag() -> None:
    odd_state = gost_gcr_solar_state_for_timeline_month(
        solar_activity=_calendar_timeline(cycle_number=25),
        mission_month_index=0,
    )
    even_state = gost_gcr_solar_state_for_timeline_month(
        solar_activity=_calendar_timeline(cycle_number=24),
        mission_month_index=0,
    )

    assert odd_state.wolf_number == 21.3
    assert even_state.wolf_number == 11.7


def test_ost_gcr_lag_is_evaluated_per_energy_not_as_one_scalar() -> None:
    timeline = _calendar_timeline(cycle_number=25)

    low_energy_state = ost_gcr_solar_state_for_timeline_energy(
        solar_activity=timeline,
        mission_month_index=0,
        energy_mev_per_nucleon=10.0,
        mass_to_charge=1.0,
    )
    high_energy_state = ost_gcr_solar_state_for_timeline_energy(
        solar_activity=timeline,
        mission_month_index=0,
        energy_mev_per_nucleon=10000.0,
        mass_to_charge=1.0,
    )

    assert low_energy_state.wolf_lagged == 21.3
    assert high_energy_state.wolf_lagged == 11.7
    assert low_energy_state.wolf_lagged != high_energy_state.wolf_lagged


def test_ost_gcr_tau_uses_wolf_number_at_t_minus_16_months() -> None:
    timeline = _calendar_timeline(cycle_number=25)
    rigidity = gcr_rigidity_gv_from_mev_per_nucleon(
        kinetic_energy_mev_per_nucleon=10.0,
        mass_to_charge=1.0,
    )

    actual = ost_gcr_delay_months_for_timeline_energy(
        solar_activity=timeline,
        mission_month_index=0,
        energy_mev_per_nucleon=10.0,
        mass_to_charge=1.0,
    )
    expected = ost_gcr_delay_months(
        rigidity_gv=rigidity,
        wolf_for_tau=21.3,  # W at the January-2020 midpoint minus 16 months.
        wolf_min=7.1,
        wolf_max=107.4,
        cycle_number=25,
    )
    wrong_current_w = ost_gcr_delay_months(
        rigidity_gv=rigidity,
        wolf_for_tau=7.1,
        wolf_min=7.1,
        wolf_max=107.4,
        cycle_number=25,
    )

    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(wrong_current_w)


def test_calendar_gcr_products_use_monthly_bins_and_true_time_integral() -> None:
    mission = MissionConfig(launch_year=2027, lifetime_years=2)
    timeline = build_mission_solar_activity(
        mission=mission,
        cycle_table=ost_wolf_number_cycle_table(mission.solar_activity_level),
        reference_start_year=2020,
        reference_cycle_number=25,
    )
    model = GostGcrModel(
        energy_grid_mev_per_nucleon=(10.0, 1000.0),
        solar_activity=timeline,
        symbols=("H",),
    )

    result = model.calculate(GcrModelInput(mission=mission, solar_activity=timeline))
    mean_flux = result.products[0].spectrum
    maximum_flux = result.products[2].spectrum
    mission_fluence = result.products[4].spectrum

    assert dict(result.method_metadata)["gcr_time_bin_count"] == "24"
    assert maximum_flux.y != pytest.approx(mean_flux.y)
    assert mission_fluence.y == pytest.approx(
        tuple(value * 2.0 * GCR_MODEL_SECONDS_PER_YEAR for value in mean_flux.y)
    )

    assert result.products[5].kind is RadiationProductKind.MISSION_FLUENCE
    assert result.products[5].spectrum.quantity is SpectrumQuantity.INTEGRAL_FLUENCE


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
