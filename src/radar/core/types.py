"""Common enumerations for the RADAR calculation core."""

from enum import StrEnum


class Particle(StrEnum):
    """Particle groups used by RADAR."""

    PROTON = "proton"
    ELECTRON = "electron"
    HZE = "hze"
    MIXED_ION = "mixed_ion"


class LetSpectrumRole(StrEnum):
    """Physical role of a LET spectrum.

    The role deliberately separates the HZE spectrum used by the direct-ion
    SEE model from proton and mixed-particle diagnostic spectra.  Particle
    type alone is insufficient because a mixed proton+HZE histogram must not
    masquerade as an HZE assessment input.
    """

    HZE_SPECIES = "hze_species"
    HZE_DIRECT_ION = "hze_direct_ion"
    PROTON_DIAGNOSTIC = "proton_diagnostic"
    MIXED_DIAGNOSTIC = "mixed_proton_hze_diagnostic"


class RadiationSource(StrEnum):
    """Natural radiation sources used by RADAR."""

    SEP = "sep"
    GCR = "gcr"
    ERB = "erb"


class RadiationQuantityMode(StrEnum):
    """Quantity class of a radiation product."""

    FLUENCE = "fluence"
    FLUX = "flux"


class AssessmentMode(StrEnum):
    """The two user-facing RADAR spectral assessment modes."""

    PEAK = "peak"
    CUMULATIVE = "cumulative"


class SpectrumAxisKind(StrEnum):
    """Physical variable represented by a spectrum's abscissa."""

    KINETIC_ENERGY = "kinetic_energy"
    KINETIC_ENERGY_PER_NUCLEON = "kinetic_energy_per_nucleon"
    LINEAR_ENERGY_TRANSFER = "linear_energy_transfer"


class SpectrumOrdinateBasis(StrEnum):
    """Mathematical meaning of the tabulated ordinate values."""

    POINT_SAMPLE = "point_sample"
    BIN_AVERAGE = "bin_average"


class AngularBasis(StrEnum):
    """Angular normalization of a spectrum.

    RADAR's canonical production product is the omnidirectional spectrum
    integrated over the full solid angle.  Directional spectra must be
    converted before they enter that product contract.
    """

    OMNIDIRECTIONAL_4PI = "omnidirectional_4pi"


class TemporalBasis(StrEnum):
    """Time semantics of the ordinates in a spectral product."""

    SEP_PEAK = "sep_peak"
    SOURCE_MAXIMUM = "source_maximum"
    MISSION_TOTAL = "mission_total"
    MEAN_INTERNAL = "mean_internal"
    ORBIT_AVERAGED = "orbit_averaged"
    SOURCE_TIME_BIN = "source_time_bin"


class SpectrumLocation(StrEnum):
    """Physical stage at which a spectrum is defined."""

    OUTSIDE_MAGNETOSPHERE = "outside_magnetosphere"
    ON_ORBIT = "on_orbit"
    BEHIND_SHIELD = "behind_shield"


class SpectrumCompleteness(StrEnum):
    """Whether every required component of a spectral product is present."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class RadiationProductKind(StrEnum):
    """Named radiation products required by OST/GOST calculation chains."""

    MISSION_FLUENCE = "mission_fluence"
    ORBIT_AVERAGED_FLUX = "orbit_averaged_flux"
    MEAN_FLUX = "mean_flux"
    MAXIMUM_FLUX = "maximum_flux"
    PEAK_FLUX = "peak_flux"

    MISSION_LET_FLUENCE = "mission_let_fluence"
    MEAN_LET_FLUX = "mean_let_flux"
    MAXIMUM_LET_FLUX = "maximum_let_flux"
    PEAK_LET_FLUX = "peak_let_flux"


class SpectrumQuantity(StrEnum):
    """Supported spectrum quantity types."""

    DIFFERENTIAL_FLUENCE = "differential_fluence"
    INTEGRAL_FLUENCE = "integral_fluence"
    DIFFERENTIAL_FLUX = "differential_flux"
    INTEGRAL_FLUX = "integral_flux"
    PEAK_DIFFERENTIAL_FLUX = "peak_differential_flux"
    PEAK_INTEGRAL_FLUX = "peak_integral_flux"
    MAXIMUM_DIFFERENTIAL_FLUX = "maximum_differential_flux"
    MAXIMUM_INTEGRAL_FLUX = "maximum_integral_flux"
    MEAN_DIFFERENTIAL_FLUX = "mean_differential_flux"
    MEAN_INTEGRAL_FLUX = "mean_integral_flux"
    LET_DIFFERENTIAL_FLUENCE = "let_differential_fluence"
    LET_DIFFERENTIAL_FLUX = "let_differential_flux"


class DoseQuantity(StrEnum):
    """Supported dose result quantity types."""

    ACCUMULATED_DOSE = "accumulated_dose"
    DOSE_RATE = "dose_rate"


class ShieldGeometry(StrEnum):
    """Shield geometry options."""

    SPHERE = "sphere"


class SolarActivityLevel(StrEnum):
    """Solar activity levels used in the non-expert mode."""

    MINIMUM = "minimum"
    MEAN = "mean"
    MAXIMUM = "maximum"


class OrbitType(StrEnum):
    """Supported orbit types."""

    CIRCULAR = "circular"
    ELLIPTIC = "elliptic"
