"""Galactic cosmic ray model interface."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Final, Protocol, cast

from radar.core.constants import (
    DEFAULT_GCR_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON as _DEFAULT_GCR_GRID,
    GCR_PRODUCTION_FIRST_CENTER_MEV_PER_NUCLEON,
    GCR_PRODUCTION_LAST_CENTER_MEV_PER_NUCLEON,
    GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON,
)
from radar.core.products import SpectrumProduct
from radar.core.profiles import (
    GOST_GCR_DOCUMENT,
    OST_134_1044_2007_DOCUMENT,
    SourceModelFamily,
    SourceModelMetadata,
)
from radar.core.source_products import validate_product_allowed_for_source
from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.spectrum_ops import mark_spectrum_incomplete
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.gcr.components import (
    GostGcrSolarState,
    GostWolfVersion,
    OstGcrSolarState,
    ost_gcr_delay_months,
)
from radar.gcr.kinematics import gcr_rigidity_gv_from_mev_per_nucleon
from radar.gcr.parameters import GCR_ELEMENT_SYMBOLS, GcrIonParameters
from radar.gcr.products import GcrFluxTimeBin, calculate_gcr_mission_products
from radar.gcr.source_spectra import (
    GOST_GCR_SOURCE_VERSION,
    OST_GCR_SOURCE_VERSION,
    GcrIonSourceSpectra,
    GcrSourceSpectra,
    calculate_gost_gcr_source_spectra_outside_magnetosphere,
    calculate_ost_gcr_source_spectra_outside_magnetosphere,
)
from radar.solar_activity.model import MissionSolarActivity

GCR_MODEL_SECONDS_PER_YEAR: Final[float] = 365.25 * 24.0 * 60.0 * 60.0
GOST_GCR_EVEN_CYCLE_DELAY_MONTHS: Final[float] = 5.5
GOST_GCR_ODD_CYCLE_DELAY_MONTHS: Final[float] = 15.5
OST_GCR_TAU_WOLF_LAG_MONTHS: Final[float] = 16.0
GCR_SOURCE_DOMAIN_REASON_PREFIX: Final[str] = "GCR source energy domain incomplete"
OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX: Final[str] = (
    "OST GCR normative ion roster incomplete"
)
DEFAULT_GCR_ENERGY_GRID_MEV_PER_NUCLEON: Final[tuple[float, ...]] = _DEFAULT_GCR_GRID

GCR_ALLOWED_PARTICLES = (
    Particle.PROTON,
    Particle.HZE,
)

GCR_ALLOWED_X_UNITS = (
    Unit.MEV,
    Unit.MEV_PER_NUCLEON,
    Unit.GEV_PER_NUCLEON,
)


def _energy_mev_per_nucleon(value: float, unit: Unit) -> float:
    if unit is Unit.GEV_PER_NUCLEON:
        return value * 1_000.0
    return value


def gcr_source_energy_domain_reasons(spectrum: Spectrum1D) -> tuple[str, ...]:
    """Return explicit reasons why a source grid cannot support OST production."""

    lower_bound = _energy_mev_per_nucleon(spectrum.x[0], spectrum.x_unit)
    upper_bound = _energy_mev_per_nucleon(spectrum.x[-1], spectrum.x_unit)
    species = spectrum.species
    symbol = "unspecified" if species is None else species.symbol
    reasons: list[str] = []

    if lower_bound > GCR_PRODUCTION_FIRST_CENTER_MEV_PER_NUCLEON:
        reasons.append(
            f"{GCR_SOURCE_DOMAIN_REASON_PREFIX}: {symbol} source lower bound "
            f"{lower_bound:.17g} MeV/nucleon is above the required "
            f"COSRAD-compatible first centre "
            f"{GCR_PRODUCTION_FIRST_CENTER_MEV_PER_NUCLEON:.17g} MeV/nucleon"
        )
    if upper_bound < GCR_PRODUCTION_LAST_CENTER_MEV_PER_NUCLEON:
        reasons.append(
            f"{GCR_SOURCE_DOMAIN_REASON_PREFIX}: {symbol} source upper bound "
            f"{upper_bound:.17g} MeV/nucleon is below the required "
            f"COSRAD-compatible final centre "
            f"{GCR_PRODUCTION_LAST_CENTER_MEV_PER_NUCLEON:.17g} MeV/nucleon "
            f"within the {GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON:.17g} "
            "MeV/nucleon reference domain"
        )
    if upper_bound > GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON:
        reasons.append(
            f"{GCR_SOURCE_DOMAIN_REASON_PREFIX}: {symbol} source upper bound "
            f"{upper_bound:.17g} MeV/nucleon exceeds the declared "
            f"{GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON:.17g} MeV/nucleon "
            "reference domain"
        )

    return tuple(reasons)


def _apply_source_energy_domain_contract(
    source_spectra: GcrSourceSpectra,
) -> GcrSourceSpectra:
    ions: list[GcrIonSourceSpectra] = []
    for ion in source_spectra.ions:
        reasons = gcr_source_energy_domain_reasons(ion.total)
        if not reasons:
            ions.append(ion)
            continue
        ions.append(
            replace(
                ion,
                main=mark_spectrum_incomplete(ion.main, *reasons),
                nek=mark_spectrum_incomplete(ion.nek, *reasons),
                akl=mark_spectrum_incomplete(ion.akl, *reasons),
                total=mark_spectrum_incomplete(ion.total, *reasons),
            )
        )
    return replace(source_spectra, ions=tuple(ions))


def ost_gcr_normative_ion_roster_reasons(
    symbols: tuple[str, ...],
) -> tuple[str, ...]:
    """Return explicit blockers unless the OST source covers every table ion."""

    observed_set = set(symbols)
    required_set = set(GCR_ELEMENT_SYMBOLS)
    missing = tuple(symbol for symbol in GCR_ELEMENT_SYMBOLS if symbol not in observed_set)
    unexpected = tuple(symbol for symbol in symbols if symbol not in required_set)
    duplicates = tuple(
        symbol for symbol in dict.fromkeys(symbols) if symbols.count(symbol) > 1
    )
    reasons: list[str] = []
    if missing:
        reasons.append(
            f"{OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX}: missing required ions "
            f"{', '.join(missing)}"
        )
    if unexpected:
        reasons.append(
            f"{OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX}: unexpected ions "
            f"{', '.join(unexpected)}"
        )
    if duplicates:
        reasons.append(
            f"{OST_GCR_NORMATIVE_ION_ROSTER_REASON_PREFIX}: duplicate ions "
            f"{', '.join(duplicates)}"
        )
    return tuple(reasons)


def _apply_ost_gcr_source_contract(
    source_spectra: GcrSourceSpectra,
) -> GcrSourceSpectra:
    contracted = _apply_source_energy_domain_contract(source_spectra)
    reasons = ost_gcr_normative_ion_roster_reasons(
        tuple(ion.symbol for ion in contracted.ions)
    )
    if not reasons:
        return contracted
    return replace(
        contracted,
        ions=tuple(
            replace(
                ion,
                main=mark_spectrum_incomplete(ion.main, *reasons),
                nek=mark_spectrum_incomplete(ion.nek, *reasons),
                akl=mark_spectrum_incomplete(ion.akl, *reasons),
                total=mark_spectrum_incomplete(ion.total, *reasons),
            )
            for ion in contracted.ions
        ),
    )


def _gcr_integral_upper_bound(spectrum: Spectrum1D) -> float:
    """Return a finite bound supported by the source calculation grid."""

    unit_scale = 1_000.0 if spectrum.x_unit is Unit.GEV_PER_NUCLEON else 1.0
    last_center_mev = spectrum.x[-1] * unit_scale
    if (
        GCR_PRODUCTION_LAST_CENTER_MEV_PER_NUCLEON
        <= last_center_mev
        <= GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON
    ):
        return GCR_REFERENCE_UPPER_BOUND_MEV_PER_NUCLEON / unit_scale
    return spectrum.x[-1]

GCR_ALLOWED_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.INTEGRAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_INTEGRAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX,
    SpectrumQuantity.DIFFERENTIAL_FLUENCE,
    SpectrumQuantity.INTEGRAL_FLUENCE,
)


def validate_gcr_energy_spectrum(spectrum: Spectrum1D) -> None:
    """Validate that a spectrum is a supported GCR energy spectrum."""

    if spectrum.source is not RadiationSource.GCR:
        msg = "GCR model spectrum source must be GCR."
        raise ValueError(msg)

    if spectrum.particle is Particle.ELECTRON:
        msg = "GCR electrons are not calculated."
        raise ValueError(msg)

    if spectrum.particle not in GCR_ALLOWED_PARTICLES:
        msg = "GCR model spectrum must describe protons or HZE particles."
        raise ValueError(msg)

    if spectrum.quantity not in GCR_ALLOWED_QUANTITIES:
        msg = "GCR model spectrum quantity must be an energy flux or fluence quantity."
        raise ValueError(msg)

    if spectrum.x_unit not in GCR_ALLOWED_X_UNITS:
        msg = "GCR model energy grid must be in MeV or GeV/nucleon."
        raise ValueError(msg)

    if spectrum.particle is Particle.HZE and spectrum.axis_kind is not None:
        if (
            spectrum.axis_kind is not SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
            or spectrum.x_unit is not Unit.MEV_PER_NUCLEON
        ):
            msg = "Typed GCR HZE energy spectra must use MeV/nucleon."
            raise ValueError(msg)

    if spectrum.quantity in (
        SpectrumQuantity.DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    ):
        allowed_units = (
            (Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY,)
            if spectrum.x_unit is Unit.MEV_PER_NUCLEON
            else (Unit.DIFFERENTIAL_FLUX,)
        )
        if spectrum.y_unit not in allowed_units:
            msg = "GCR differential flux spectrum values must use differential flux units."
            raise ValueError(msg)

    if spectrum.quantity in (
        SpectrumQuantity.INTEGRAL_FLUX,
        SpectrumQuantity.MEAN_INTEGRAL_FLUX,
        SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX,
    ):
        if spectrum.y_unit is not Unit.INTEGRAL_FLUX:
            msg = "GCR integral flux spectrum values must use integral flux units."
            raise ValueError(msg)

    if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        allowed_units = (
            (Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY,)
            if spectrum.x_unit is Unit.MEV_PER_NUCLEON
            else (Unit.DIFFERENTIAL_FLUENCE,)
        )
        if spectrum.y_unit not in allowed_units:
            msg = "GCR differential fluence spectrum values must use differential fluence units."
            raise ValueError(msg)

    if spectrum.quantity is SpectrumQuantity.INTEGRAL_FLUENCE:
        if spectrum.y_unit is not Unit.INTEGRAL_FLUENCE:
            msg = "GCR integral fluence spectrum values must use integral fluence units."
            raise ValueError(msg)

    if any(energy <= 0.0 for energy in spectrum.x):
        msg = "GCR model energy grid values must be positive."
        raise ValueError(msg)


def _default_gcr_product_kind(spectrum: Spectrum1D) -> RadiationProductKind:
    if spectrum.quantity in (
        SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        SpectrumQuantity.INTEGRAL_FLUENCE,
    ):
        return RadiationProductKind.MISSION_FLUENCE

    if spectrum.quantity in (
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_INTEGRAL_FLUX,
    ):
        return RadiationProductKind.MEAN_FLUX

    if spectrum.quantity in (
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX,
    ):
        return RadiationProductKind.MAXIMUM_FLUX

    if spectrum.quantity in (
        SpectrumQuantity.DIFFERENTIAL_FLUX,
        SpectrumQuantity.INTEGRAL_FLUX,
    ):
        msg = (
            "GCR differential or integral flux cannot be converted to a radiation "
            "product without specifying mean or maximum flux."
        )
        raise ValueError(msg)

    msg = f"Unsupported GCR spectrum quantity: {spectrum.quantity}"
    raise ValueError(msg)


def _default_gcr_products(spectra: tuple[Spectrum1D, ...]) -> tuple[SpectrumProduct, ...]:
    return tuple(
        SpectrumProduct(
            kind=_default_gcr_product_kind(spectrum),
            spectrum=spectrum,
            label=f"GCR {spectrum.particle.value} {spectrum.quantity.value}",
        )
        for spectrum in spectra
    )


@dataclass(frozen=True)
class GcrModelInput:
    """Input parameters passed to a GCR model."""

    mission: MissionConfig
    solar_activity: MissionSolarActivity | None = None

    def __post_init__(self) -> None:
        if self.solar_activity is None:
            return
        self.solar_activity.validate_for_mission(self.mission)

    @property
    def launch_year(self) -> int:
        """Return mission launch year."""

        return self.mission.launch_year

    @property
    def lifetime_years(self) -> int:
        """Return mission lifetime in integer years."""

        return self.mission.lifetime_years


@dataclass(frozen=True)
class GcrModelResult:
    """Result returned by a GCR model."""

    spectra: tuple[Spectrum1D, ...]
    lifetime_years: int
    model: str
    document: str
    products: tuple[SpectrumProduct, ...] = ()
    method_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "GCR model result must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_gcr_energy_spectrum(spectrum)

        products = self.products or _default_gcr_products(self.spectra)

        if not products:
            msg = "GCR model result must contain at least one radiation product."
            raise ValueError(msg)

        for product in products:
            validate_product_allowed_for_source(
                product=product,
                source=RadiationSource.GCR,
            )

        if tuple(product.spectrum for product in products) != self.spectra:
            msg = "GCR model product spectra must match result spectra."
            raise ValueError(msg)

        object.__setattr__(self, "products", products)

        if not isinstance(self.lifetime_years, int):
            msg = "GCR result lifetime must be an integer number of years."
            raise ValueError(msg)

        if self.lifetime_years < 1:
            msg = "GCR result lifetime must be at least one year."
            raise ValueError(msg)

        if not self.model:
            msg = "GCR model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "GCR source document must not be empty."
            raise ValueError(msg)

        metadata_keys = tuple(key for key, _ in self.method_metadata)
        if len(set(metadata_keys)) != len(metadata_keys):
            msg = "GCR method metadata keys must be unique."
            raise ValueError(msg)


class GcrModelProtocol(Protocol):
    """Protocol implemented by concrete GCR models."""

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""
        ...

    def calculate(self, model_input: GcrModelInput) -> GcrModelResult:
        """Calculate mission GCR spectra."""


@dataclass(frozen=True)
class StaticGcrModel:
    """Simple deterministic GCR model used for tests and integration plumbing.

    Normative OST/GOST GCR models must be implemented separately.
    """

    spectra: tuple[Spectrum1D, ...]
    model: str = "static_gcr_model"
    document: str = "test"
    model_family: SourceModelFamily = SourceModelFamily.CUSTOM

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.GCR,
            model_family=self.model_family,
            name=self.model,
            document=self.document,
        )

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "Static GCR model must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_gcr_energy_spectrum(spectrum)

        if not self.model:
            msg = "GCR model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "GCR source document must not be empty."
            raise ValueError(msg)

        _ = self.metadata

    def calculate(self, model_input: GcrModelInput) -> GcrModelResult:
        """Return configured test GCR spectra."""

        return GcrModelResult(
            spectra=self.spectra,
            lifetime_years=model_input.lifetime_years,
            model=self.model,
            document=self.document,
        )


def _mission_duration_seconds(model_input: GcrModelInput) -> float:
    return float(model_input.lifetime_years * GCR_MODEL_SECONDS_PER_YEAR)


def _products_from_source_spectra(
    *,
    source_spectra: GcrSourceSpectra,
    duration_seconds: float,
) -> tuple[SpectrumProduct, ...]:
    products: list[SpectrumProduct] = []

    for ion_spectra in source_spectra.ions:
        mission_products = calculate_gcr_mission_products(
            (
                GcrFluxTimeBin(
                    spectrum=ion_spectra.total,
                    duration_seconds=duration_seconds,
                ),
            ),
            upper_energy_bound=_gcr_integral_upper_bound(ion_spectra.total),
        )
        products.extend(mission_products.products)

    return tuple(products)


def _products_from_source_spectra_time_series(
    *,
    source_spectra_by_month: tuple[GcrSourceSpectra, ...],
    duration_seconds_per_month: float,
) -> tuple[SpectrumProduct, ...]:
    """Aggregate calendar-month source states independently for every ion."""

    if not source_spectra_by_month:
        msg = "GCR monthly source spectrum sequence must not be empty."
        raise ValueError(msg)

    reference_symbols = tuple(ion.symbol for ion in source_spectra_by_month[0].ions)
    for source_spectra in source_spectra_by_month[1:]:
        if tuple(ion.symbol for ion in source_spectra.ions) != reference_symbols:
            msg = "GCR monthly source spectra must contain the same ordered ions."
            raise ValueError(msg)

    products: list[SpectrumProduct] = []
    for symbol in reference_symbols:
        time_bins = tuple(
            GcrFluxTimeBin(
                spectrum=source_spectra.ion(symbol).total,
                duration_seconds=duration_seconds_per_month,
            )
            for source_spectra in source_spectra_by_month
        )
        products.extend(
            calculate_gcr_mission_products(
                time_bins,
                upper_energy_bound=_gcr_integral_upper_bound(
                    time_bins[0].spectrum
                ),
            ).products
        )
    return tuple(products)


def _result_from_source_spectra(
    *,
    source_spectra: GcrSourceSpectra,
    model_input: GcrModelInput,
    model: str,
    document: str,
) -> GcrModelResult:
    products = _products_from_source_spectra(
        source_spectra=source_spectra,
        duration_seconds=_mission_duration_seconds(model_input),
    )

    return GcrModelResult(
        spectra=tuple(product.spectrum for product in products),
        lifetime_years=model_input.lifetime_years,
        model=model,
        document=document,
        products=products,
    )


def _result_from_source_spectra_time_series(
    *,
    source_spectra_by_month: tuple[GcrSourceSpectra, ...],
    solar_activity: MissionSolarActivity,
    model_input: GcrModelInput,
    model: str,
    document: str,
    additional_metadata: tuple[tuple[str, str], ...] = (),
) -> GcrModelResult:
    products = _products_from_source_spectra_time_series(
        source_spectra_by_month=source_spectra_by_month,
        duration_seconds_per_month=solar_activity.month_duration_seconds,
    )
    return GcrModelResult(
        spectra=tuple(product.spectrum for product in products),
        lifetime_years=model_input.lifetime_years,
        model=model,
        document=document,
        products=products,
        method_metadata=(
            *solar_activity.metadata,
            ("gcr_time_bin_policy", "calendar_month_midpoint_piecewise_constant"),
            ("gcr_time_bin_count", str(len(source_spectra_by_month))),
            *additional_metadata,
        ),
    )


def _model_solar_activity(
    *,
    configured: MissionSolarActivity | None,
    model_input: GcrModelInput,
) -> MissionSolarActivity | None:
    solar_activity = configured or model_input.solar_activity
    if solar_activity is None:
        return None
    solar_activity.validate_for_mission(model_input.mission)
    return solar_activity


def gost_gcr_delay_months_for_cycle(cycle_number: int) -> float:
    """Return the fixed GOST lag selected by numbered-cycle parity."""

    if not isinstance(cycle_number, int) or cycle_number < 1:
        msg = "GOST solar cycle number must be a positive integer."
        raise ValueError(msg)
    if cycle_number % 2 == 0:
        return GOST_GCR_EVEN_CYCLE_DELAY_MONTHS
    return GOST_GCR_ODD_CYCLE_DELAY_MONTHS


def gost_gcr_solar_state_for_timeline_month(
    *,
    solar_activity: MissionSolarActivity,
    mission_month_index: int,
) -> GostGcrSolarState:
    """Return GOST state with W sampled at the parity-dependent lag."""

    point = solar_activity.point_at_mission_month(mission_month_index)
    delay_months = gost_gcr_delay_months_for_cycle(point.cycle_number)
    wolf_lagged = solar_activity.wolf_number_at_mission_month_coordinate(
        mission_month_index + 0.5 - delay_months,
    )
    return GostGcrSolarState(
        wolf_number=wolf_lagged,
        wolf_version=cast(
            "GostWolfVersion",
            solar_activity.wolf_number_version,
        ),
    )


def ost_gcr_delay_months_for_timeline_energy(
    *,
    solar_activity: MissionSolarActivity,
    mission_month_index: int,
    energy_mev_per_nucleon: float,
    mass_to_charge: float,
) -> float:
    """Evaluate OST (V.8)-(V.9), including W(t-16 months) in tau."""

    point = solar_activity.point_at_mission_month(mission_month_index)
    month_coordinate = mission_month_index + 0.5
    # OST formula (V.9): tau uses W(t - delta_t), delta_t = 16 months.
    # The cycle number and its Wmin/Wmax remain those of the current time t.
    wolf_for_tau = solar_activity.wolf_number_at_mission_month_coordinate(
        month_coordinate - OST_GCR_TAU_WOLF_LAG_MONTHS,
    )
    rigidity_gv = gcr_rigidity_gv_from_mev_per_nucleon(
        kinetic_energy_mev_per_nucleon=energy_mev_per_nucleon,
        mass_to_charge=mass_to_charge,
    )
    return ost_gcr_delay_months(
        rigidity_gv=rigidity_gv,
        wolf_for_tau=wolf_for_tau,
        wolf_min=solar_activity.wolf_min,
        wolf_max=solar_activity.wolf_max,
        cycle_number=point.cycle_number,
    )


def ost_gcr_solar_state_for_timeline_energy(
    *,
    solar_activity: MissionSolarActivity,
    mission_month_index: int,
    energy_mev_per_nucleon: float,
    mass_to_charge: float,
) -> OstGcrSolarState:
    """Return OST state after evaluating its lag for one ion and energy."""

    point = solar_activity.point_at_mission_month(mission_month_index)
    month_coordinate = mission_month_index + 0.5
    delay_months = ost_gcr_delay_months_for_timeline_energy(
        solar_activity=solar_activity,
        mission_month_index=mission_month_index,
        energy_mev_per_nucleon=energy_mev_per_nucleon,
        mass_to_charge=mass_to_charge,
    )
    wolf_lagged = solar_activity.wolf_number_at_mission_month_coordinate(
        month_coordinate - delay_months,
    )
    return OstGcrSolarState(
        wolf_current=point.wolf_number,
        wolf_lagged=wolf_lagged,
        wolf_min=solar_activity.wolf_min,
        wolf_max=solar_activity.wolf_max,
        cycle_number=point.cycle_number,
        after_polarity_reversal=point.after_polarity_reversal,
    )


@dataclass(frozen=True)
class OstGcrModel:
    """Normative OST GCR model outside the magnetosphere."""

    energy_grid_mev_per_nucleon: tuple[float, ...] = DEFAULT_GCR_ENERGY_GRID_MEV_PER_NUCLEON
    solar_state: OstGcrSolarState = field(
        default_factory=lambda: OstGcrSolarState(
            wolf_current=60.0,
            wolf_lagged=50.0,
            wolf_min=10.0,
            wolf_max=100.0,
            cycle_number=24,
            after_polarity_reversal=True,
        )
    )
    nek_wolf_number: float = 60.0
    akl_years_from_cycle_start: float = 2.0
    solar_activity: MissionSolarActivity | None = None
    symbols: tuple[str, ...] = GCR_ELEMENT_SYMBOLS
    model: str = "ost_gcr_model"
    document: str = OST_134_1044_2007_DOCUMENT
    version: str = OST_GCR_SOURCE_VERSION

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.GCR,
            model_family=SourceModelFamily.OST_134_1044_2007,
            name=self.model,
            document=self.document,
            version=self.version,
        )

    def __post_init__(self) -> None:
        _ = self.metadata
        _ = self.calculate_source_spectra()

    def _source_spectra_for_month(
        self,
        solar_activity: MissionSolarActivity,
        mission_month_index: int,
    ) -> GcrSourceSpectra:
        point = solar_activity.point_at_mission_month(mission_month_index)
        base_state = OstGcrSolarState(
            wolf_current=point.wolf_number,
            wolf_lagged=point.wolf_number,
            wolf_min=solar_activity.wolf_min,
            wolf_max=solar_activity.wolf_max,
            cycle_number=point.cycle_number,
            after_polarity_reversal=point.after_polarity_reversal,
        )

        def solar_state_for_energy(
            energy_mev_per_nucleon: float,
            parameters: GcrIonParameters,
        ) -> OstGcrSolarState:
            return ost_gcr_solar_state_for_timeline_energy(
                solar_activity=solar_activity,
                mission_month_index=mission_month_index,
                energy_mev_per_nucleon=energy_mev_per_nucleon,
                mass_to_charge=parameters.mass_to_charge,
            )

        return _apply_ost_gcr_source_contract(
            calculate_ost_gcr_source_spectra_outside_magnetosphere(
                energy_grid_mev_per_nucleon=self.energy_grid_mev_per_nucleon,
                solar_state=base_state,
                nek_wolf_number=point.wolf_number,
                akl_years_from_cycle_start=point.years_from_cycle_start,
                symbols=self.symbols,
                solar_state_for_energy=solar_state_for_energy,
                version=self.version,
                document_refs=(self.document,),
            )
        )

    def calculate_source_spectra(
        self,
        model_input: GcrModelInput | None = None,
    ) -> GcrSourceSpectra:
        """Calculate OST source spectra at the launch month or static state."""

        solar_activity = (
            _model_solar_activity(configured=self.solar_activity, model_input=model_input)
            if model_input is not None
            else self.solar_activity
        )
        if solar_activity is not None:
            return self._source_spectra_for_month(solar_activity, 0)

        return _apply_ost_gcr_source_contract(
            calculate_ost_gcr_source_spectra_outside_magnetosphere(
                energy_grid_mev_per_nucleon=self.energy_grid_mev_per_nucleon,
                solar_state=self.solar_state,
                nek_wolf_number=self.nek_wolf_number,
                akl_years_from_cycle_start=self.akl_years_from_cycle_start,
                symbols=self.symbols,
                version=self.version,
                document_refs=(self.document,),
            )
        )

    def calculate(self, model_input: GcrModelInput) -> GcrModelResult:
        """Calculate OST GCR mission products outside the magnetosphere."""

        solar_activity = _model_solar_activity(
            configured=self.solar_activity,
            model_input=model_input,
        )
        if solar_activity is not None:
            monthly_spectra = tuple(
                self._source_spectra_for_month(solar_activity, month_index)
                for month_index in range(solar_activity.mission_month_count)
            )
            return _result_from_source_spectra_time_series(
                source_spectra_by_month=monthly_spectra,
                solar_activity=solar_activity,
                model_input=model_input,
                model=self.model,
                document=self.document,
                additional_metadata=(
                    (
                        "ost_gcr_lag_policy",
                        "energy_rigidity_dependent_appendix_v_evaluated_per_ion_energy_month",
                    ),
                    ("ost_gcr_tau_wolf_lag_months", "16"),
                    (
                        "ost_gcr_tau_cycle_policy",
                        "W_from_t_minus_16_months_but_parity_and_extrema_from_cycle_at_t",
                    ),
                    (
                        "ost_gcr_polarity_reversal_policy",
                        "application_discretization_first_month_of_tabulated_maximum_year",
                    ),
                    (
                        "ost_gcr_polarity_reversal_normative_rule",
                        "reversal_coincides_with_solar_activity_maximum;month_not_tabulated",
                    ),
                    ("ost_gcr_nek_wolf_policy", "current_calendar_month"),
                    ("ost_gcr_akl_time_policy", "calendar_month_midpoint"),
                ),
            )

        return _result_from_source_spectra(
            source_spectra=self.calculate_source_spectra(),
            model_input=model_input,
            model=self.model,
            document=self.document,
        )


@dataclass(frozen=True)
class GostGcrModel:
    """Normative GOST GCR model outside the magnetosphere.

    The main GCR component uses the GOST formula. The low-energy NEK and AKL
    components use OST Appendix V.
    """

    energy_grid_mev_per_nucleon: tuple[float, ...] = DEFAULT_GCR_ENERGY_GRID_MEV_PER_NUCLEON
    solar_state: GostGcrSolarState = field(
        default_factory=lambda: GostGcrSolarState(
            wolf_number=70.0,
            wolf_version="1.0",
        )
    )
    nek_wolf_number: float = 70.0
    akl_years_from_cycle_start: float = 2.0
    solar_activity: MissionSolarActivity | None = None
    symbols: tuple[str, ...] = GCR_ELEMENT_SYMBOLS
    model: str = "gost_gcr_model"
    document: str = GOST_GCR_DOCUMENT
    version: str = GOST_GCR_SOURCE_VERSION

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.GCR,
            model_family=SourceModelFamily.GOST_GCR,
            name=self.model,
            document=self.document,
            version=self.version,
        )

    def __post_init__(self) -> None:
        _ = self.metadata
        _ = self.calculate_source_spectra()

    def _source_spectra_for_month(
        self,
        solar_activity: MissionSolarActivity,
        mission_month_index: int,
    ) -> GcrSourceSpectra:
        point = solar_activity.point_at_mission_month(mission_month_index)
        return _apply_source_energy_domain_contract(
            calculate_gost_gcr_source_spectra_outside_magnetosphere(
                energy_grid_mev_per_nucleon=self.energy_grid_mev_per_nucleon,
                solar_state=gost_gcr_solar_state_for_timeline_month(
                    solar_activity=solar_activity,
                    mission_month_index=mission_month_index,
                ),
                nek_wolf_number=point.wolf_number,
                akl_years_from_cycle_start=point.years_from_cycle_start,
                symbols=self.symbols,
                version=self.version,
                document_refs=(self.document, OST_134_1044_2007_DOCUMENT),
            )
        )

    def calculate_source_spectra(
        self,
        model_input: GcrModelInput | None = None,
    ) -> GcrSourceSpectra:
        """Calculate GOST source spectra at the launch month or static state."""

        solar_activity = (
            _model_solar_activity(configured=self.solar_activity, model_input=model_input)
            if model_input is not None
            else self.solar_activity
        )
        if solar_activity is not None:
            return self._source_spectra_for_month(solar_activity, 0)

        return _apply_source_energy_domain_contract(
            calculate_gost_gcr_source_spectra_outside_magnetosphere(
                energy_grid_mev_per_nucleon=self.energy_grid_mev_per_nucleon,
                solar_state=self.solar_state,
                nek_wolf_number=self.nek_wolf_number,
                akl_years_from_cycle_start=self.akl_years_from_cycle_start,
                symbols=self.symbols,
                version=self.version,
                document_refs=(self.document, OST_134_1044_2007_DOCUMENT),
            )
        )

    def calculate(self, model_input: GcrModelInput) -> GcrModelResult:
        """Calculate GOST GCR mission products outside the magnetosphere."""

        solar_activity = _model_solar_activity(
            configured=self.solar_activity,
            model_input=model_input,
        )
        if solar_activity is not None:
            monthly_spectra = tuple(
                self._source_spectra_for_month(solar_activity, month_index)
                for month_index in range(solar_activity.mission_month_count)
            )
            return _result_from_source_spectra_time_series(
                source_spectra_by_month=monthly_spectra,
                solar_activity=solar_activity,
                model_input=model_input,
                model=self.model,
                document=self.document,
                additional_metadata=(
                    ("gost_gcr_even_cycle_lag_months", "5.5"),
                    ("gost_gcr_odd_cycle_lag_months", "15.5"),
                    ("gost_gcr_lag_sampling", "calendar_month_midpoint"),
                    ("gost_gcr_nek_wolf_policy", "current_calendar_month"),
                    ("gost_gcr_akl_source", "OST 134-1044-2007 Appendix V"),
                ),
            )

        return _result_from_source_spectra(
            source_spectra=self.calculate_source_spectra(),
            model_input=model_input,
            model=self.model,
            document=self.document,
        )
