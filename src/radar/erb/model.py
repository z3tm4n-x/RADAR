"""Earth radiation belt model interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from radar.core.products import SpectrumProduct
from radar.core.profiles import (
    OST_134_1044_2007_DOCUMENT,
    SourceModelFamily,
    SourceModelMetadata,
)
from radar.core.source_products import validate_product_allowed_for_source
from radar.core.project import CalculationConfig
from radar.core.spectra import Spectrum1D
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit

ERB_ALLOWED_PARTICLES = (
    Particle.PROTON,
    Particle.ELECTRON,
)

ERB_ALLOWED_QUANTITIES = (
    SpectrumQuantity.DIFFERENTIAL_FLUX,
    SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
    SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
)


def validate_erb_energy_spectrum(spectrum: Spectrum1D) -> None:
    """Validate that a spectrum is a supported ERB energy flux spectrum."""

    if spectrum.source is not RadiationSource.ERB:
        msg = "ERB model spectrum source must be ERB."
        raise ValueError(msg)

    if spectrum.particle not in ERB_ALLOWED_PARTICLES:
        msg = "ERB model spectrum must describe protons or electrons."
        raise ValueError(msg)

    if spectrum.quantity not in ERB_ALLOWED_QUANTITIES:
        msg = "ERB model spectrum quantity must be a supported differential flux quantity."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "ERB model energy grid must be in MeV."
        raise ValueError(msg)

    if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUX:
        msg = "ERB model spectrum values must use differential flux units."
        raise ValueError(msg)

    if any(energy <= 0.0 for energy in spectrum.x):
        msg = "ERB model energy grid values must be positive."
        raise ValueError(msg)


def _default_erb_product_kind(spectrum: Spectrum1D) -> RadiationProductKind:
    if spectrum.quantity is SpectrumQuantity.DIFFERENTIAL_FLUX:
        return RadiationProductKind.ORBIT_AVERAGED_FLUX

    if spectrum.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX:
        return RadiationProductKind.MEAN_FLUX

    if spectrum.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX:
        return RadiationProductKind.MAXIMUM_FLUX

    if spectrum.quantity is SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX:
        return RadiationProductKind.PEAK_FLUX

    msg = f"Unsupported ERB spectrum quantity: {spectrum.quantity}"
    raise ValueError(msg)


def _default_erb_products(spectra: tuple[Spectrum1D, ...]) -> tuple[SpectrumProduct, ...]:
    return tuple(
        SpectrumProduct(
            kind=_default_erb_product_kind(spectrum),
            spectrum=spectrum,
            label=f"ERB {spectrum.particle.value} {spectrum.quantity.value}",
        )
        for spectrum in spectra
    )


@dataclass(frozen=True)
class ErbModelInput:
    """Input parameters passed to an ERB model."""

    config: CalculationConfig

    @property
    def lifetime_years(self) -> int:
        """Return mission lifetime in integer years."""

        return self.config.mission.lifetime_years

    @property
    def kp(self) -> int:
        """Return geomagnetic Kp index."""

        return self.config.kp


@dataclass(frozen=True)
class ErbModelResult:
    """Result returned by an ERB model."""

    spectra: tuple[Spectrum1D, ...]
    lifetime_years: int
    kp: int
    model: str
    document: str
    products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "ERB model result must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_erb_energy_spectrum(spectrum)

        products = self.products or _default_erb_products(self.spectra)

        if not products:
            msg = "ERB model result must contain at least one radiation product."
            raise ValueError(msg)

        for product in products:
            validate_product_allowed_for_source(
                product=product,
                source=RadiationSource.ERB,
            )

        if tuple(product.spectrum for product in products) != self.spectra:
            msg = "ERB model product spectra must match result spectra."
            raise ValueError(msg)

        object.__setattr__(self, "products", products)

        if not isinstance(self.lifetime_years, int):
            msg = "ERB result lifetime must be an integer number of years."
            raise ValueError(msg)

        if self.lifetime_years < 1:
            msg = "ERB result lifetime must be at least one year."
            raise ValueError(msg)

        if not isinstance(self.kp, int):
            msg = "ERB result Kp must be an integer."
            raise ValueError(msg)

        if not 0 <= self.kp <= 9:
            msg = "ERB result Kp must be in the range 0..9."
            raise ValueError(msg)

        if not self.model:
            msg = "ERB model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "ERB source document must not be empty."
            raise ValueError(msg)


class ErbModelProtocol(Protocol):
    """Protocol implemented by concrete ERB models."""

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""
        ...

    def calculate(self, model_input: ErbModelInput) -> ErbModelResult:
        """Calculate mission ERB spectra."""


@dataclass(frozen=True)
class StaticErbModel:
    """Simple deterministic ERB model used for tests and integration plumbing.

    Normative ERB models must be implemented separately.
    Dose and bremsstrahlung calculations must be implemented separately.
    """

    spectra: tuple[Spectrum1D, ...]
    model: str = "static_erb_model"
    document: str = "test"
    model_family: SourceModelFamily = SourceModelFamily.CUSTOM

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.ERB,
            model_family=self.model_family,
            name=self.model,
            document=self.document,
        )

    def __post_init__(self) -> None:
        if not self.spectra:
            msg = "Static ERB model must contain at least one spectrum."
            raise ValueError(msg)

        for spectrum in self.spectra:
            validate_erb_energy_spectrum(spectrum)

        if not self.model:
            msg = "ERB model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "ERB source document must not be empty."
            raise ValueError(msg)

        _ = self.metadata

    def calculate(self, model_input: ErbModelInput) -> ErbModelResult:
        """Return configured test ERB spectra."""

        return ErbModelResult(
            spectra=self.spectra,
            lifetime_years=model_input.lifetime_years,
            kp=model_input.kp,
            model=self.model,
            document=self.document,
        )

@dataclass(frozen=True)
class OstErbModel:
    """Placeholder for normative OST ERB model.

    The class declares metadata and profile compatibility only.
    Numerical OST ERB equations are not implemented yet.
    """

    model: str = "ost_erb_model"
    document: str = OST_134_1044_2007_DOCUMENT
    version: str = "not_implemented"

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.ERB,
            model_family=SourceModelFamily.OST_134_1044_2007,
            name=self.model,
            document=self.document,
            version=self.version,
        )

    def __post_init__(self) -> None:
        _ = self.metadata

    def calculate(self, model_input: ErbModelInput) -> ErbModelResult:
        """Raise until the normative OST ERB model is implemented."""

        raise NotImplementedError("OST ERB model is not implemented yet.")
