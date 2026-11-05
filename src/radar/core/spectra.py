"""Typed one-dimensional spectra."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from radar.core.types import (
    AngularBasis,
    AssessmentMode,
    LetSpectrumRole,
    Particle,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit


@dataclass(frozen=True)
class ParticleSpecies:
    """Machine-readable particle or nuclide identity.

    ``atomic_number`` and ``mass_number`` are deliberately separate from
    ``symbol`` so calculation code never has to recover Z or A by parsing a
    display label.  Aggregate spectra (for example a total HZE LET spectrum)
    are marked explicitly and do not pretend to be a nuclide.
    """

    symbol: str
    atomic_number: int | None = None
    mass_number: int | None = None
    is_aggregate: bool = False

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            msg = "Particle species symbol must not be empty."
            raise ValueError(msg)

        has_atomic_number = self.atomic_number is not None
        has_mass_number = self.mass_number is not None
        if has_atomic_number != has_mass_number:
            msg = "Particle species Z and A must either both be set or both be omitted."
            raise ValueError(msg)

        if self.is_aggregate:
            if has_atomic_number:
                msg = "Aggregate particle species must not claim a single Z or A."
                raise ValueError(msg)
            return

        if self.atomic_number is not None:
            mass_number = self.mass_number
            assert mass_number is not None
            if (
                isinstance(self.atomic_number, bool)
                or not isinstance(self.atomic_number, int)
                or self.atomic_number <= 0
            ):
                msg = "Particle species Z must be a positive integer."
                raise ValueError(msg)
            if (
                isinstance(mass_number, bool)
                or not isinstance(mass_number, int)
                or mass_number <= 0
            ):
                msg = "Particle species A must be a positive integer."
                raise ValueError(msg)
            if mass_number < self.atomic_number:
                msg = "Particle species A must not be smaller than Z."
                raise ValueError(msg)


@dataclass(frozen=True)
class ShieldingDescriptor:
    """Shield material and spherical areal density attached to a spectrum."""

    material: str
    areal_density_g_cm2: float
    areal_density_unit: Unit = Unit.THICKNESS

    def __post_init__(self) -> None:
        if not self.material.strip():
            msg = "Shield material must not be empty."
            raise ValueError(msg)
        if not isfinite(self.areal_density_g_cm2):
            msg = "Shield areal density must be finite."
            raise ValueError(msg)
        if self.areal_density_g_cm2 < 0.0:
            msg = "Shield areal density must be non-negative."
            raise ValueError(msg)
        if self.areal_density_unit is not Unit.THICKNESS:
            msg = "Shield areal density unit must be g/cm^2."
            raise ValueError(msg)


@dataclass(frozen=True)
class ModelReference:
    """Traceable model, normative-document and dataset references."""

    name: str
    version: str
    document_refs: tuple[str, ...] = ()
    dataset_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            msg = "Model reference name must not be empty."
            raise ValueError(msg)
        if not self.version.strip():
            msg = "Model reference version must not be empty."
            raise ValueError(msg)

        self._validate_references(self.document_refs, "document")
        self._validate_references(self.dataset_refs, "dataset")

    @staticmethod
    def _validate_references(references: tuple[str, ...], reference_kind: str) -> None:
        if any(not reference.strip() for reference in references):
            msg = f"Model {reference_kind} references must not contain empty values."
            raise ValueError(msg)
        if len(set(references)) != len(references):
            msg = f"Model {reference_kind} references must be unique."
            raise ValueError(msg)


@dataclass(frozen=True)
class TransformationStep:
    """One traceable transformation in the source-to-product calculation chain."""

    name: str
    version: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            msg = "Transformation step name must not be empty."
            raise ValueError(msg)
        if not self.version.strip():
            msg = "Transformation step version must not be empty."
            raise ValueError(msg)
        if any(not key.strip() or not value.strip() for key, value in self.parameters):
            msg = "Transformation parameters must have non-empty names and values."
            raise ValueError(msg)
        keys = tuple(key for key, _ in self.parameters)
        if len(set(keys)) != len(keys):
            msg = "Transformation parameter names must be unique within a step."
            raise ValueError(msg)


@dataclass(frozen=True)
class Spectrum1D:
    """One-dimensional spectrum with explicit physical meaning.

    The fields through ``model`` are the legacy construction contract.  The
    optional fields that follow form the production metadata contract.  This
    split lets existing calculation stages migrate without silently declaring
    their historical products production-ready.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    x_unit: Unit
    y_unit: Unit
    quantity: SpectrumQuantity
    particle: Particle
    source: RadiationSource
    model: str
    axis_kind: SpectrumAxisKind | None = None
    angular_basis: AngularBasis | None = None
    temporal_basis: TemporalBasis | None = None
    location: SpectrumLocation | None = None
    species: ParticleSpecies | None = None
    shielding: ShieldingDescriptor | None = None
    assessment_mode: AssessmentMode | None = None
    mission_duration_s: float | None = None
    model_reference: ModelReference | None = None
    transformation_chain: tuple[TransformationStep, ...] = ()
    completeness: SpectrumCompleteness | None = None
    incomplete_reasons: tuple[str, ...] = ()
    let_role: LetSpectrumRole | None = None
    ordinate_basis: SpectrumOrdinateBasis = SpectrumOrdinateBasis.POINT_SAMPLE
    bin_edges: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            msg = "Spectrum grid and values must have the same length."
            raise ValueError(msg)

        if not self.x:
            msg = "Spectrum must contain at least one point."
            raise ValueError(msg)

        if not self.model:
            msg = "Spectrum model name must not be empty."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.x):
            msg = "Spectrum grid must contain only finite values."
            raise ValueError(msg)

        if any(not isfinite(value) for value in self.y):
            msg = "Spectrum values must contain only finite values."
            raise ValueError(msg)

        if any(value < 0.0 for value in self.y):
            msg = "Spectrum values must be non-negative."
            raise ValueError(msg)

        if any(right <= left for left, right in zip(self.x, self.x[1:])):
            msg = "Spectrum grid must be strictly increasing."
            raise ValueError(msg)

        if self.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE:
            if self.bin_edges is None:
                msg = "Bin-average spectra require explicit bin_edges."
                raise ValueError(msg)
            if len(self.bin_edges) != len(self.x) + 1:
                msg = "Spectrum bin_edges length must equal len(x) + 1."
                raise ValueError(msg)
            if any(not isfinite(value) for value in self.bin_edges):
                msg = "Spectrum bin_edges must contain only finite values."
                raise ValueError(msg)
            if any(
                right <= left
                for left, right in zip(self.bin_edges, self.bin_edges[1:])
            ):
                msg = "Spectrum bin_edges must be strictly increasing."
                raise ValueError(msg)
            if any(
                not (left < center < right)
                for left, center, right in zip(
                    self.bin_edges,
                    self.x,
                    self.bin_edges[1:],
                )
            ):
                msg = "Each spectrum center must lie inside its bin edges."
                raise ValueError(msg)
        elif self.bin_edges is not None:
            msg = "Point-sample spectra must not carry bin_edges."
            raise ValueError(msg)

        if self.mission_duration_s is not None:
            if not isfinite(self.mission_duration_s):
                msg = "Spectrum mission duration must be finite."
                raise ValueError(msg)
            if self.mission_duration_s <= 0.0:
                msg = "Spectrum mission duration must be positive."
                raise ValueError(msg)

        if any(not reason.strip() for reason in self.incomplete_reasons):
            msg = "Spectrum incomplete reasons must not contain empty values."
            raise ValueError(msg)
        if len(set(self.incomplete_reasons)) != len(self.incomplete_reasons):
            msg = "Spectrum incomplete reasons must be unique."
            raise ValueError(msg)

    def production_contract_violations(self) -> tuple[str, ...]:
        """Return every violation of the canonical production contract.

        Legacy spectra intentionally return violations instead of failing in
        ``__post_init__``.  Production boundaries must call
        :meth:`validate_production_contract` explicitly.
        """

        violations: list[str] = []

        if self.axis_kind is None:
            violations.append("axis_kind is required")
        else:
            expected_x_unit = {
                SpectrumAxisKind.KINETIC_ENERGY: Unit.MEV,
                SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON: Unit.MEV_PER_NUCLEON,
                SpectrumAxisKind.LINEAR_ENERGY_TRANSFER: Unit.LET,
            }[self.axis_kind]
            if self.x_unit is not expected_x_unit:
                violations.append(
                    f"axis_kind {self.axis_kind.value} requires x_unit "
                    f"{expected_x_unit.value}"
                )

        if self.angular_basis is None:
            violations.append("angular_basis is required")
        elif self.angular_basis is not AngularBasis.OMNIDIRECTIONAL_4PI:
            violations.append("production spectra must be omnidirectional over 4pi")

        if self.temporal_basis is None:
            violations.append("temporal_basis is required")
        if self.location is None:
            violations.append("location is required")
        if self.species is None:
            violations.append("structured particle species is required")
        else:
            violations.extend(self._species_contract_violations())

        if self.location is SpectrumLocation.BEHIND_SHIELD:
            if self.shielding is None:
                violations.append("behind-shield spectra require shielding metadata")
        elif self.location is not None and self.shielding is not None:
            violations.append("shielding metadata is only valid behind the shield")

        if self.assessment_mode is AssessmentMode.CUMULATIVE:
            if self.mission_duration_s is None:
                violations.append("cumulative spectra require mission_duration_s")
            if (
                self.temporal_basis is not None
                and self.temporal_basis is not TemporalBasis.MISSION_TOTAL
            ):
                violations.append("cumulative spectra require temporal_basis mission_total")
        elif self.assessment_mode is AssessmentMode.PEAK:
            if self.mission_duration_s is not None:
                violations.append("peak spectra must not carry mission_duration_s")
            if self.temporal_basis not in (
                None,
                TemporalBasis.SEP_PEAK,
                TemporalBasis.SOURCE_MAXIMUM,
            ):
                violations.append(
                    "peak spectra require temporal_basis sep_peak or source_maximum"
                )

        if self.temporal_basis is TemporalBasis.MISSION_TOTAL:
            if self.assessment_mode is not AssessmentMode.CUMULATIVE:
                violations.append("mission_total spectra require cumulative assessment_mode")
            if self.mission_duration_s is None:
                violations.append("mission_total spectra require mission_duration_s")
        elif self.temporal_basis in (
            TemporalBasis.SEP_PEAK,
            TemporalBasis.SOURCE_MAXIMUM,
        ):
            if self.assessment_mode is not AssessmentMode.PEAK:
                violations.append("peak/maximum spectra require peak assessment_mode")
        elif self.temporal_basis in (
            TemporalBasis.MEAN_INTERNAL,
            TemporalBasis.ORBIT_AVERAGED,
        ):
            if self.assessment_mode is not None:
                violations.append("internal mean spectra must not claim an assessment_mode")
            if self.mission_duration_s is not None:
                violations.append("internal mean spectra must not carry mission_duration_s")

        if self.temporal_basis is TemporalBasis.SEP_PEAK and self.source is not RadiationSource.SEP:
            violations.append("temporal_basis sep_peak is only valid for the SEP source")

        if self.model_reference is None:
            violations.append("model_reference is required")
        else:
            if self.model_reference.name != self.model:
                violations.append("model_reference name must match the legacy model name")
            if not self.model_reference.document_refs:
                violations.append("at least one normative document reference is required")
            if not self.model_reference.dataset_refs:
                violations.append("at least one dataset reference is required")
            if self.model_reference.version.strip().lower() in {
                "unversioned",
                "not_implemented",
            }:
                violations.append("production model_reference requires a real version")

        if not self.transformation_chain:
            violations.append("transformation_chain is required")

        if self.completeness is None:
            violations.append("completeness is required")
        elif self.completeness is SpectrumCompleteness.INCOMPLETE:
            violations.append("incomplete spectra cannot be released as production results")

        if self.completeness is SpectrumCompleteness.COMPLETE and self.incomplete_reasons:
            violations.append("complete spectra must not contain incomplete reasons")
        if self.completeness is SpectrumCompleteness.INCOMPLETE and not self.incomplete_reasons:
            violations.append("incomplete spectra require at least one reason")

        let_quantities = (
            SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE,
            SpectrumQuantity.LET_DIFFERENTIAL_FLUX,
        )
        if self.quantity in let_quantities:
            if self.axis_kind not in (None, SpectrumAxisKind.LINEAR_ENERGY_TRANSFER):
                violations.append("LET quantities require a linear-energy-transfer axis")
            if self.let_role is None:
                violations.append("LET quantities require an explicit let_role")
            else:
                violations.extend(self._let_role_contract_violations())
        elif self.axis_kind is SpectrumAxisKind.LINEAR_ENERGY_TRANSFER:
            violations.append("a linear-energy-transfer axis requires a LET quantity")
        elif self.let_role is not None:
            violations.append("let_role is only valid for LET quantities")

        expected_y_unit = self._expected_production_y_unit()
        if expected_y_unit is not None and self.y_unit is not expected_y_unit:
            violations.append(
                f"quantity {self.quantity.value} on axis "
                f"{self.axis_kind.value if self.axis_kind is not None else 'unspecified'} "
                f"requires y_unit {expected_y_unit.value}"
            )

        return tuple(dict.fromkeys(violations))

    def _expected_production_y_unit(self) -> Unit | None:
        """Return the canonical ordinate unit for a complete axis contract."""

        differential_fluence_quantities = (
            SpectrumQuantity.DIFFERENTIAL_FLUENCE,
        )
        differential_flux_quantities = (
            SpectrumQuantity.DIFFERENTIAL_FLUX,
            SpectrumQuantity.PEAK_DIFFERENTIAL_FLUX,
            SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX,
            SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX,
        )
        integral_flux_quantities = (
            SpectrumQuantity.INTEGRAL_FLUX,
            SpectrumQuantity.PEAK_INTEGRAL_FLUX,
            SpectrumQuantity.MAXIMUM_INTEGRAL_FLUX,
            SpectrumQuantity.MEAN_INTEGRAL_FLUX,
        )

        if self.axis_kind is None:
            return None
        if self.quantity in differential_fluence_quantities:
            if self.axis_kind is SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON:
                return Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY
            return Unit.DIFFERENTIAL_FLUENCE
        if self.quantity in differential_flux_quantities:
            if self.axis_kind is SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON:
                return Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY
            return Unit.DIFFERENTIAL_FLUX
        if self.quantity is SpectrumQuantity.INTEGRAL_FLUENCE:
            return Unit.INTEGRAL_FLUENCE
        if self.quantity in integral_flux_quantities:
            return Unit.INTEGRAL_FLUX
        if self.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE:
            return Unit.DIFFERENTIAL_LET_FLUENCE
        if self.quantity is SpectrumQuantity.LET_DIFFERENTIAL_FLUX:
            return Unit.DIFFERENTIAL_LET_FLUX
        return None

    def _species_contract_violations(self) -> tuple[str, ...]:
        """Return particle/species consistency violations."""

        if self.species is None:
            return ()

        violations: list[str] = []
        species = self.species

        if self.particle is Particle.PROTON:
            if species.is_aggregate or (
                species.atomic_number,
                species.mass_number,
            ) != (1, 1):
                violations.append("proton spectra require species Z=1, A=1")
            proton_let_axis = (
                self.let_role is LetSpectrumRole.PROTON_DIAGNOSTIC
                and self.axis_kind is SpectrumAxisKind.LINEAR_ENERGY_TRANSFER
            )
            if (
                self.axis_kind not in (None, SpectrumAxisKind.KINETIC_ENERGY)
                and not proton_let_axis
            ):
                violations.append("proton spectra require a kinetic-energy axis")
        elif self.particle is Particle.ELECTRON:
            if species.is_aggregate or species.atomic_number is not None:
                violations.append("electron spectra require a non-nuclide species")
            if self.axis_kind not in (None, SpectrumAxisKind.KINETIC_ENERGY):
                violations.append("electron spectra require a kinetic-energy axis")
        elif self.particle is Particle.HZE:
            if not species.is_aggregate and (
                species.atomic_number is None or species.atomic_number < 2
            ):
                violations.append("individual HZE spectra require species Z>=2 and explicit A")
            if self.axis_kind not in (
                None,
                SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON,
                SpectrumAxisKind.LINEAR_ENERGY_TRANSFER,
            ):
                violations.append("HZE spectra require an MeV/nucleon or LET axis")
        elif self.particle is Particle.MIXED_ION:
            if not species.is_aggregate:
                violations.append("mixed-ion spectra require aggregate species metadata")
            if self.axis_kind not in (
                None,
                SpectrumAxisKind.LINEAR_ENERGY_TRANSFER,
            ):
                violations.append("mixed-ion spectra are only defined on a LET axis")

        return tuple(violations)

    def _let_role_contract_violations(self) -> tuple[str, ...]:
        """Return consistency violations for structured LET semantics."""

        role = self.let_role
        species = self.species
        if role is None:
            return ()

        violations: list[str] = []
        if role is LetSpectrumRole.PROTON_DIAGNOSTIC:
            if self.particle is not Particle.PROTON:
                violations.append("proton diagnostic LET requires particle proton")
        elif role is LetSpectrumRole.HZE_SPECIES:
            if self.particle is not Particle.HZE:
                violations.append("HZE species LET requires particle HZE")
            if species is not None and species.is_aggregate:
                violations.append("HZE species LET requires non-aggregate species")
        elif role is LetSpectrumRole.HZE_DIRECT_ION:
            if self.particle is not Particle.HZE:
                violations.append("direct-ion LET requires particle HZE")
            if species is not None and not species.is_aggregate:
                violations.append("direct-ion LET requires aggregate HZE species")
        elif role is LetSpectrumRole.MIXED_DIAGNOSTIC:
            if self.particle is not Particle.MIXED_ION:
                violations.append("mixed diagnostic LET requires particle mixed_ion")
            if species is not None and not species.is_aggregate:
                violations.append("mixed diagnostic LET requires aggregate species")
        return tuple(violations)

    def validate_production_contract(self) -> None:
        """Raise if the spectrum is not safe to expose as a RADAR result."""

        violations = self.production_contract_violations()
        if violations:
            details = "; ".join(violations)
            msg = f"Spectrum does not satisfy the production contract: {details}."
            raise ValueError(msg)

    def integral_trapezoid(self) -> float:
        """Return trapezoidal integral over the spectrum grid."""

        if self.ordinate_basis is SpectrumOrdinateBasis.BIN_AVERAGE:
            msg = (
                "Trapezoidal integration is invalid for bin-average spectra; "
                "use integral_conserved()."
            )
            raise ValueError(msg)

        if len(self.x) == 1:
            return 0.0

        total = 0.0
        for left_x, right_x, left_y, right_y in zip(
            self.x,
            self.x[1:],
            self.y,
            self.y[1:],
        ):
            total += 0.5 * (left_y + right_y) * (right_x - left_x)

        return total

    def integral_conserved(self) -> float:
        """Integrate according to the typed ordinate/bin contract."""

        if self.ordinate_basis is SpectrumOrdinateBasis.POINT_SAMPLE:
            return self.integral_trapezoid()

        edges = self.bin_edges
        assert edges is not None
        return sum(
            value * (right - left)
            for value, left, right in zip(self.y, edges, edges[1:])
        )
