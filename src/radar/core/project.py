"""Typed project configuration objects."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.constants import (
    DEFAULT_DOSE_QUANTITY,
    DEFAULT_DOSE_UNIT,
    DEFAULT_KP,
    DEFAULT_SEP_EXCEEDANCE_PROBABILITY,
    DEFAULT_SHIELD_GEOMETRY,
    DEFAULT_SHIELD_THICKNESSES_G_CM2,
    MAX_KP,
    MIN_KP,
    MIN_LIFETIME_YEARS,
)
from radar.core.dose_units import validate_dose_unit_for_quantity
from radar.core.profiles import (
    DEFAULT_METHODOLOGY_PROFILE,
    MethodologyProfile,
    MethodologyProfileSpec,
    MethodologySourceModelContract,
    SourceModelFamily,
    methodology_profile_spec,
    source_model_contract_for_profile,
    validate_source_model_family_for_profile,
)
from radar.core.types import (
    DoseQuantity,
    OrbitType,
    RadiationSource,
    ShieldGeometry,
    SolarActivityLevel,
)
from radar.core.units import Unit


@dataclass(frozen=True)
class MissionConfig:
    """Mission input parameters."""

    launch_year: int
    lifetime_years: int
    solar_activity_level: SolarActivityLevel = SolarActivityLevel.MEAN
    sep_exceedance_probability: float = DEFAULT_SEP_EXCEEDANCE_PROBABILITY

    def __post_init__(self) -> None:
        if not isinstance(self.launch_year, int):
            msg = "Launch year must be an integer."
            raise ValueError(msg)

        if not isinstance(self.lifetime_years, int):
            msg = "Lifetime must be an integer number of years."
            raise ValueError(msg)

        if self.lifetime_years < MIN_LIFETIME_YEARS:
            msg = "Lifetime must be at least one year."
            raise ValueError(msg)

        if not 0.0 < self.sep_exceedance_probability < 1.0:
            msg = "SEP exceedance probability must be between 0 and 1."
            raise ValueError(msg)


@dataclass(frozen=True)
class OrbitConfig:
    """Orbit input parameters."""

    orbit_type: OrbitType
    perigee_altitude_km: float
    apogee_altitude_km: float
    inclination_deg: float
    argument_of_perigee_deg: float = 0.0

    @classmethod
    def circular(cls, altitude_km: float, inclination_deg: float) -> OrbitConfig:
        """Create circular orbit configuration."""

        return cls(
            orbit_type=OrbitType.CIRCULAR,
            perigee_altitude_km=altitude_km,
            apogee_altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            argument_of_perigee_deg=0.0,
        )

    def __post_init__(self) -> None:
        if self.perigee_altitude_km < 0.0:
            msg = "Perigee altitude must be non-negative."
            raise ValueError(msg)

        if self.apogee_altitude_km < self.perigee_altitude_km:
            msg = "Apogee altitude must be greater than or equal to perigee altitude."
            raise ValueError(msg)

        if not 0.0 <= self.inclination_deg <= 180.0:
            msg = "Inclination must be in the range 0..180 degrees."
            raise ValueError(msg)

        if not 0.0 <= self.argument_of_perigee_deg < 360.0:
            msg = "Argument of perigee must be in the range 0..360 degrees."
            raise ValueError(msg)

        if self.orbit_type is OrbitType.CIRCULAR:
            if self.perigee_altitude_km != self.apogee_altitude_km:
                msg = "Circular orbit must have equal perigee and apogee altitudes."
                raise ValueError(msg)


@dataclass(frozen=True)
class ShieldingConfig:
    """Shielding input parameters."""

    thicknesses_g_cm2: tuple[float, ...] = DEFAULT_SHIELD_THICKNESSES_G_CM2
    geometry: ShieldGeometry = DEFAULT_SHIELD_GEOMETRY

    def __post_init__(self) -> None:
        if not self.thicknesses_g_cm2:
            msg = "Shield thickness list must not be empty."
            raise ValueError(msg)

        if any(value <= 0.0 for value in self.thicknesses_g_cm2):
            msg = "Shield thicknesses must be positive."
            raise ValueError(msg)

        if len(set(self.thicknesses_g_cm2)) != len(self.thicknesses_g_cm2):
            msg = "Shield thicknesses must not contain duplicates."
            raise ValueError(msg)

        object.__setattr__(self, "thicknesses_g_cm2", tuple(sorted(self.thicknesses_g_cm2)))

        if self.geometry is not ShieldGeometry.SPHERE:
            msg = "Only spherical aluminum shielding is supported in the first version."
            raise ValueError(msg)


@dataclass(frozen=True)
class SourceModelSelectionConfig:
    """Selected source model families for a calculation."""

    profile: MethodologyProfile = DEFAULT_METHODOLOGY_PROFILE
    sep_model_family: SourceModelFamily = SourceModelFamily.OST_134_1044_2007
    gcr_model_family: SourceModelFamily = SourceModelFamily.OST_134_1044_2007
    erb_model_family: SourceModelFamily = SourceModelFamily.OST_134_1044_2007

    @classmethod
    def from_profile(cls, profile: MethodologyProfile) -> SourceModelSelectionConfig:
        """Create model family selection from a methodology profile."""

        contract = source_model_contract_for_profile(profile)
        return cls(
            profile=profile,
            sep_model_family=contract.sep_model_family,
            gcr_model_family=contract.gcr_model_family,
            erb_model_family=contract.erb_model_family,
        )

    @classmethod
    def from_methodology(
        cls,
        methodology: MethodologyConfig,
    ) -> SourceModelSelectionConfig:
        """Create model family selection from methodology settings."""

        return cls.from_profile(methodology.profile)

    @property
    def families_by_source(self) -> dict[RadiationSource, SourceModelFamily]:
        """Return selected model families keyed by radiation source."""

        return {
            RadiationSource.SEP: self.sep_model_family,
            RadiationSource.GCR: self.gcr_model_family,
            RadiationSource.ERB: self.erb_model_family,
        }

    def model_family_for_source(
        self,
        source: RadiationSource,
    ) -> SourceModelFamily:
        """Return selected model family for a radiation source."""

        try:
            return self.families_by_source[source]
        except KeyError as exc:
            msg = f"Unsupported radiation source: {source}"
            raise ValueError(msg) from exc

    def __post_init__(self) -> None:
        validate_source_model_family_for_profile(
            profile=self.profile,
            source=RadiationSource.SEP,
            model_family=self.sep_model_family,
        )
        validate_source_model_family_for_profile(
            profile=self.profile,
            source=RadiationSource.GCR,
            model_family=self.gcr_model_family,
        )
        validate_source_model_family_for_profile(
            profile=self.profile,
            source=RadiationSource.ERB,
            model_family=self.erb_model_family,
        )

@dataclass(frozen=True)
class MethodologyConfig:
    """Methodology switches fixed for the first RADAR version."""

    profile: MethodologyProfile = DEFAULT_METHODOLOGY_PROFILE
    calculate_gcr_electrons: bool = False
    include_bremsstrahlung: bool = True
    include_protons_in_let: bool = True

    @property
    def profile_spec(self) -> MethodologyProfileSpec:
        """Return explicit methodology profile specification."""

        return methodology_profile_spec(self.profile)

    @property
    def source_model_contract(self) -> MethodologySourceModelContract:
        """Return source model family contract for this methodology."""

        return source_model_contract_for_profile(self.profile)

    @property
    def source_model_selection(self) -> SourceModelSelectionConfig:
        """Return source model family selection for this methodology."""

        return SourceModelSelectionConfig.from_methodology(self)

    def __post_init__(self) -> None:
        if self.calculate_gcr_electrons:
            msg = "GCR electrons are not calculated in the first version."
            raise ValueError(msg)

        if not self.include_bremsstrahlung:
            msg = "Bremsstrahlung accounting is required."
            raise ValueError(msg)

        if not self.include_protons_in_let:
            msg = "Protons must be included in LET calculation for SEE."
            raise ValueError(msg)


@dataclass(frozen=True)
class CalculationConfig:
    """Complete calculation configuration."""

    mission: MissionConfig
    orbit: OrbitConfig
    shielding: ShieldingConfig = ShieldingConfig()
    methodology: MethodologyConfig = MethodologyConfig()
    kp: int = DEFAULT_KP
    dose_quantity: DoseQuantity = DEFAULT_DOSE_QUANTITY
    dose_unit: Unit = DEFAULT_DOSE_UNIT

    @property
    def source_model_selection(self) -> SourceModelSelectionConfig:
        """Return source model family selection for this calculation."""

        return self.methodology.source_model_selection

    def __post_init__(self) -> None:
        if not isinstance(self.kp, int):
            msg = "Kp must be an integer."
            raise ValueError(msg)

        if not MIN_KP <= self.kp <= MAX_KP:
            msg = "Kp must be in the range 0..9."
            raise ValueError(msg)

        validate_dose_unit_for_quantity(
            dose_quantity=self.dose_quantity,
            dose_unit=self.dose_unit,
        )
