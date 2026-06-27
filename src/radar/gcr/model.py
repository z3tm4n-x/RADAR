"""Galactic cosmic ray model interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Protocol

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
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit
from radar.gcr.components import GostGcrSolarState, OstGcrSolarState
from radar.gcr.parameters import GCR_ELEMENT_SYMBOLS
from radar.gcr.products import GcrFluxTimeBin, calculate_gcr_mission_products
from radar.gcr.source_spectra import (
    GcrSourceSpectra,
    calculate_gost_gcr_source_spectra_outside_magnetosphere,
    calculate_ost_gcr_source_spectra_outside_magnetosphere,
)

GCR_MODEL_SECONDS_PER_YEAR: Final[float] = 365.25 * 24.0 * 60.0 * 60.0
DEFAULT_GCR_ENERGY_GRID_MEV_PER_NUCLEON: Final[tuple[float, ...]] = (
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
)

GCR_ALLOWED_PARTICLES = (
    Particle.PROTON,
    Particle.HZE,
)

GCR_ALLOWED_X_UNITS = (
    Unit.MEV,
    Unit.GEV_PER_NUCLEON,
)

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

    if spectrum.quantity in (
        SpectrumQuantity.DIFFERENTIAL_FLUX,
        SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    ):
        if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUX:
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
        if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUENCE:
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
            )
        )
        products.extend(mission_products.products)

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
    symbols: tuple[str, ...] = GCR_ELEMENT_SYMBOLS
    model: str = "ost_gcr_model"
    document: str = OST_134_1044_2007_DOCUMENT
    version: str = "source_spectra_outside_magnetosphere_v1"

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

    def calculate_source_spectra(self) -> GcrSourceSpectra:
        """Calculate OST source spectra outside the magnetosphere."""

        return calculate_ost_gcr_source_spectra_outside_magnetosphere(
            energy_grid_mev_per_nucleon=self.energy_grid_mev_per_nucleon,
            solar_state=self.solar_state,
            nek_wolf_number=self.nek_wolf_number,
            akl_years_from_cycle_start=self.akl_years_from_cycle_start,
            symbols=self.symbols,
        )

    def calculate(self, model_input: GcrModelInput) -> GcrModelResult:
        """Calculate OST GCR mission products outside the magnetosphere."""

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
    symbols: tuple[str, ...] = GCR_ELEMENT_SYMBOLS
    model: str = "gost_gcr_model"
    document: str = GOST_GCR_DOCUMENT
    version: str = "source_spectra_outside_magnetosphere_v1"

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

    def calculate_source_spectra(self) -> GcrSourceSpectra:
        """Calculate GOST source spectra outside the magnetosphere."""

        return calculate_gost_gcr_source_spectra_outside_magnetosphere(
            energy_grid_mev_per_nucleon=self.energy_grid_mev_per_nucleon,
            solar_state=self.solar_state,
            nek_wolf_number=self.nek_wolf_number,
            akl_years_from_cycle_start=self.akl_years_from_cycle_start,
            symbols=self.symbols,
        )

    def calculate(self, model_input: GcrModelInput) -> GcrModelResult:
        """Calculate GOST GCR mission products outside the magnetosphere."""

        return _result_from_source_spectra(
            source_spectra=self.calculate_source_spectra(),
            model_input=model_input,
            model=self.model,
            document=self.document,
        )
