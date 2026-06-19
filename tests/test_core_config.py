import pytest

from radar.core.constants import DEFAULT_SHIELD_THICKNESSES_G_CM2
from radar.core.project import (
    CalculationConfig,
    MethodologyConfig,
    MissionConfig,
    OrbitConfig,
    ShieldingConfig,
)
from radar.core.types import DoseQuantity, OrbitType, ShieldGeometry, SolarActivityLevel
from radar.core.units import Unit


def test_default_calculation_config() -> None:
    mission = MissionConfig(
        launch_year=2027,
        lifetime_years=15,
        solar_activity_level=SolarActivityLevel.MEAN,
    )
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    config = CalculationConfig(mission=mission, orbit=orbit)

    assert config.kp == 3
    assert config.dose_quantity is DoseQuantity.ACCUMULATED_DOSE
    assert config.dose_unit is Unit.RAD
    assert config.shielding.geometry is ShieldGeometry.SPHERE
    assert config.shielding.thicknesses_g_cm2 == DEFAULT_SHIELD_THICKNESSES_G_CM2
    assert config.methodology.calculate_gcr_electrons is False
    assert config.methodology.include_bremsstrahlung is True
    assert config.methodology.include_protons_in_let is True


def test_lifetime_must_be_integer() -> None:
    with pytest.raises(ValueError, match="integer"):
        MissionConfig(launch_year=2027, lifetime_years=2.5)


def test_kp_default_and_range() -> None:
    mission = MissionConfig(launch_year=2027, lifetime_years=15)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    assert CalculationConfig(mission=mission, orbit=orbit, kp=3).kp == 3

    with pytest.raises(ValueError, match="0..9"):
        CalculationConfig(mission=mission, orbit=orbit, kp=10)


def test_dose_quantity_and_unit_must_match() -> None:
    mission = MissionConfig(launch_year=2027, lifetime_years=15)
    orbit = OrbitConfig.circular(altitude_km=35786.0, inclination_deg=0.0)

    accumulated_dose_config = CalculationConfig(
        mission=mission,
        orbit=orbit,
        dose_quantity=DoseQuantity.ACCUMULATED_DOSE,
        dose_unit=Unit.RAD,
    )

    assert accumulated_dose_config.dose_quantity is DoseQuantity.ACCUMULATED_DOSE
    assert accumulated_dose_config.dose_unit is Unit.RAD

    dose_rate_config = CalculationConfig(
        mission=mission,
        orbit=orbit,
        dose_quantity=DoseQuantity.DOSE_RATE,
        dose_unit=Unit.RAD_PER_SECOND,
    )

    assert dose_rate_config.dose_quantity is DoseQuantity.DOSE_RATE
    assert dose_rate_config.dose_unit is Unit.RAD_PER_SECOND

    with pytest.raises(ValueError, match="Accumulated"):
        CalculationConfig(
            mission=mission,
            orbit=orbit,
            dose_quantity=DoseQuantity.ACCUMULATED_DOSE,
            dose_unit=Unit.RAD_PER_SECOND,
        )

    with pytest.raises(ValueError, match="Dose rate"):
        CalculationConfig(
            mission=mission,
            orbit=orbit,
            dose_quantity=DoseQuantity.DOSE_RATE,
            dose_unit=Unit.RAD,
        )


def test_shielding_is_spherical_and_positive() -> None:
    shielding = ShieldingConfig(thicknesses_g_cm2=(1.0, 0.1, 10.0))

    assert shielding.thicknesses_g_cm2 == (0.1, 1.0, 10.0)
    assert shielding.geometry is ShieldGeometry.SPHERE

    with pytest.raises(ValueError, match="positive"):
        ShieldingConfig(thicknesses_g_cm2=(0.0, 1.0))

    with pytest.raises(ValueError, match="duplicates"):
        ShieldingConfig(thicknesses_g_cm2=(1.0, 1.0))


def test_first_version_methodology_switches_are_strict() -> None:
    with pytest.raises(ValueError, match="GCR electrons"):
        MethodologyConfig(calculate_gcr_electrons=True)

    with pytest.raises(ValueError, match="Bremsstrahlung"):
        MethodologyConfig(include_bremsstrahlung=False)

    with pytest.raises(ValueError, match="Protons"):
        MethodologyConfig(include_protons_in_let=False)


def test_circular_orbit_requires_equal_altitudes() -> None:
    with pytest.raises(ValueError, match="Circular orbit"):
        OrbitConfig(
            orbit_type=OrbitType.CIRCULAR,
            perigee_altitude_km=500.0,
            apogee_altitude_km=600.0,
            inclination_deg=82.0,
        )
