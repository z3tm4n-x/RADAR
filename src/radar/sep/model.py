"""Solar energetic particle model interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from radar.core.products import SpectrumProduct
from radar.core.profiles import (
    GOST_SEP_DOCUMENT,
    OST_134_1044_2007_DOCUMENT,
    SourceModelFamily,
    SourceModelMetadata,
)
from radar.core.source_products import validate_product_allowed_for_source
from radar.core.project import MissionConfig
from radar.core.spectra import Spectrum1D
from radar.core.spectrum_ops import scale_spectrum
from radar.core.types import (
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumQuantity,
)
from radar.core.units import Unit


def validate_sep_proton_fluence_spectrum(spectrum: Spectrum1D) -> None:
    """Validate that a spectrum is a SEP proton differential fluence spectrum."""

    if spectrum.particle is not Particle.PROTON:
        msg = "SEP model spectrum must describe protons."
        raise ValueError(msg)

    if spectrum.source is not RadiationSource.SEP:
        msg = "SEP model spectrum source must be SEP."
        raise ValueError(msg)

    if spectrum.quantity is not SpectrumQuantity.DIFFERENTIAL_FLUENCE:
        msg = "SEP model spectrum quantity must be differential fluence."
        raise ValueError(msg)

    if spectrum.x_unit is not Unit.MEV:
        msg = "SEP model energy grid must be in MeV."
        raise ValueError(msg)

    if spectrum.y_unit is not Unit.DIFFERENTIAL_FLUENCE:
        msg = "SEP model spectrum values must use differential fluence units."
        raise ValueError(msg)

    if any(energy <= 0.0 for energy in spectrum.x):
        msg = "SEP model energy grid values must be positive."
        raise ValueError(msg)


def _mission_fluence_product(spectrum: Spectrum1D) -> SpectrumProduct:
    return SpectrumProduct(
        kind=RadiationProductKind.MISSION_FLUENCE,
        spectrum=spectrum,
        label="SEP proton mission fluence",
    )


@dataclass(frozen=True)
class SepModelInput:
    """Input parameters passed to a SEP model."""

    mission: MissionConfig

    @property
    def lifetime_years(self) -> int:
        """Return mission lifetime in integer years."""

        return self.mission.lifetime_years

    @property
    def exceedance_probability(self) -> float:
        """Return SEP exceedance probability."""

        return self.mission.sep_exceedance_probability


@dataclass(frozen=True)
class SepModelResult:
    """Result returned by a SEP model."""

    spectrum: Spectrum1D
    lifetime_years: int
    exceedance_probability: float
    model: str
    document: str
    products: tuple[SpectrumProduct, ...] = ()

    def __post_init__(self) -> None:
        validate_sep_proton_fluence_spectrum(self.spectrum)

        products = self.products or (_mission_fluence_product(self.spectrum),)

        if len(products) != 1:
            msg = "SEP model result must contain exactly one radiation product."
            raise ValueError(msg)

        product = products[0]

        if product.kind is not RadiationProductKind.MISSION_FLUENCE:
            msg = "SEP model product must be mission fluence."
            raise ValueError(msg)

        validate_product_allowed_for_source(
            product=product,
            source=RadiationSource.SEP,
        )

        if product.spectrum != self.spectrum:
            msg = "SEP model product spectrum must match result spectrum."
            raise ValueError(msg)

        object.__setattr__(self, "products", products)

        if not isinstance(self.lifetime_years, int):
            msg = "SEP result lifetime must be an integer number of years."
            raise ValueError(msg)

        if self.lifetime_years < 1:
            msg = "SEP result lifetime must be at least one year."
            raise ValueError(msg)

        if not 0.0 < self.exceedance_probability < 1.0:
            msg = "SEP exceedance probability must be between 0 and 1."
            raise ValueError(msg)

        if not self.model:
            msg = "SEP model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "SEP source document must not be empty."
            raise ValueError(msg)

    @property
    def product(self) -> SpectrumProduct:
        """Return the SEP mission fluence product."""

        return self.products[0]


class SepModelProtocol(Protocol):
    """Protocol implemented by concrete SEP models."""

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""
        ...

    def calculate(self, model_input: SepModelInput) -> SepModelResult:
        """Calculate mission SEP fluence spectrum."""


@dataclass(frozen=True)
class StaticSepModel:
    """Simple deterministic SEP model used for tests and integration plumbing.

    The input spectrum is interpreted as annual differential fluence.
    The model scales it by integer mission lifetime.

    Normative OST/GOST SEP models must be implemented separately.
    """

    annual_fluence_spectrum: Spectrum1D
    model: str = "static_sep_model"
    document: str = "test"
    model_family: SourceModelFamily = SourceModelFamily.CUSTOM

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.SEP,
            model_family=self.model_family,
            name=self.model,
            document=self.document,
        )

    def __post_init__(self) -> None:
        validate_sep_proton_fluence_spectrum(self.annual_fluence_spectrum)

        if not self.model:
            msg = "SEP model name must not be empty."
            raise ValueError(msg)

        if not self.document:
            msg = "SEP source document must not be empty."
            raise ValueError(msg)

        _ = self.metadata

    def calculate(self, model_input: SepModelInput) -> SepModelResult:
        """Return annual test fluence scaled by integer mission lifetime."""

        mission_spectrum = scale_spectrum(
            spectrum=self.annual_fluence_spectrum,
            factor=float(model_input.lifetime_years),
            model=f"{self.annual_fluence_spectrum.model}*{model_input.lifetime_years}years",
        )

        return SepModelResult(
            spectrum=mission_spectrum,
            lifetime_years=model_input.lifetime_years,
            exceedance_probability=model_input.exceedance_probability,
            model=self.model,
            document=self.document,
        )

@dataclass(frozen=True)
class OstSepModel:
    """Placeholder for normative OST SEP model.

    The class declares metadata and profile compatibility only.
    Numerical OST SEP equations are not implemented yet.
    """

    model: str = "ost_sep_model"
    document: str = OST_134_1044_2007_DOCUMENT
    version: str = "not_implemented"

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.SEP,
            model_family=SourceModelFamily.OST_134_1044_2007,
            name=self.model,
            document=self.document,
            version=self.version,
        )

    def __post_init__(self) -> None:
        _ = self.metadata

    def calculate(self, model_input: SepModelInput) -> SepModelResult:
        """Raise until the normative OST SEP model is implemented."""

        raise NotImplementedError("OST SEP model is not implemented yet.")


@dataclass(frozen=True)
class GostSepModel:
    """Placeholder for normative GOST SEP model.

    The class declares metadata and profile compatibility only.
    Numerical GOST SEP equations are not implemented yet.
    """

    model: str = "gost_sep_model"
    document: str = GOST_SEP_DOCUMENT
    version: str = "not_implemented"

    @property
    def metadata(self) -> SourceModelMetadata:
        """Return source model metadata."""

        return SourceModelMetadata(
            source=RadiationSource.SEP,
            model_family=SourceModelFamily.GOST_SEP,
            name=self.model,
            document=self.document,
            version=self.version,
        )

    def __post_init__(self) -> None:
        _ = self.metadata

    def calculate(self, model_input: SepModelInput) -> SepModelResult:
        """Raise until the normative GOST SEP model is implemented."""

        raise NotImplementedError("GOST SEP model is not implemented yet.")
