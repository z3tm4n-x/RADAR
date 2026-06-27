"""Earth radiation belt models."""

from radar.erb.field import (
    ErbFieldLineSample,
    MagneticFieldVector,
    magnetic_field_at_orbit_sample,
    magnetic_field_vector_km,
    trace_l_shell_b_over_b0,
)
from radar.erb.igrf import IgrfCoefficients, parse_igrf_coefficients
from radar.erb.model import (
    OstErbModel,
    ErbModelInput,
    ErbModelProtocol,
    ErbModelResult,
    StaticErbModel,
    validate_erb_energy_spectrum,
)
from radar.erb.orbit import ErbOrbitGrid, ErbOrbitSample, sample_erb_orbit_grid
from radar.erb.ost_tables import (
    OST_ERB_ELECTRON_ENERGIES_MEV,
    OST_ERB_PROTON_ENERGIES_MEV,
    OstErbFluxTable,
    OstErbInterpolationMode,
    OstErbShell,
    OstErbTableSet,
    load_ost_erb_appendix_a_tables,
)
from radar.erb.radbelt import RadbeltAscMap, parse_radbelt_asc_text

__all__ = [
    "ErbFieldLineSample",
    "ErbOrbitGrid",
    "ErbOrbitSample",
    "IgrfCoefficients",
    "MagneticFieldVector",
    "OST_ERB_ELECTRON_ENERGIES_MEV",
    "OST_ERB_PROTON_ENERGIES_MEV",
    "OstErbFluxTable",
    "OstErbInterpolationMode",
    "OstErbModel",
    "OstErbShell",
    "OstErbTableSet",
    "ErbModelInput",
    "ErbModelProtocol",
    "ErbModelResult",
    "RadbeltAscMap",
    "StaticErbModel",
    "magnetic_field_at_orbit_sample",
    "magnetic_field_vector_km",
    "load_ost_erb_appendix_a_tables",
    "parse_igrf_coefficients",
    "parse_radbelt_asc_text",
    "sample_erb_orbit_grid",
    "trace_l_shell_b_over_b0",
    "validate_erb_energy_spectrum",
]
