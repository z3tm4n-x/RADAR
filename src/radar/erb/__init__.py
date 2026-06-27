"""Earth radiation belt models."""

from radar.erb.appendix_e import (
    OstErbApproximationFitQuality,
    OstErbElectronApproximationCoefficients,
    OstErbElectronDifferentialApproximationCoefficients,
    OstErbElectronIntegralApproximation,
    OstErbElectronIntegralApproximationFit,
    OstErbProtonApproximationCoefficients,
    OstErbProtonDifferentialApproximationCoefficients,
    OstErbProtonIntegralApproximation,
    OstErbProtonIntegralApproximationFit,
    fit_ost_erb_electron_integral_approximation,
    fit_ost_erb_proton_integral_approximation,
    integrate_differential_flux_tail_power_law,
)
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
    ErbIntegralSpectrum,
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
from radar.erb.resources import ERB_IGRF14_RESOURCE_NAME, load_igrf14_coefficients

__all__ = [
    "ERB_IGRF14_RESOURCE_NAME",
    "OstErbApproximationFitQuality",
    "OstErbElectronApproximationCoefficients",
    "OstErbElectronDifferentialApproximationCoefficients",
    "OstErbElectronIntegralApproximation",
    "OstErbElectronIntegralApproximationFit",
    "OstErbProtonApproximationCoefficients",
    "OstErbProtonDifferentialApproximationCoefficients",
    "OstErbProtonIntegralApproximation",
    "OstErbProtonIntegralApproximationFit",
    "fit_ost_erb_electron_integral_approximation",
    "fit_ost_erb_proton_integral_approximation",
    "integrate_differential_flux_tail_power_law",
    "ErbFieldLineSample",
    "ErbOrbitGrid",
    "ErbOrbitSample",
    "ErbIntegralSpectrum",
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
    "load_igrf14_coefficients",
    "load_ost_erb_appendix_a_tables",
    "parse_igrf_coefficients",
    "parse_radbelt_asc_text",
    "sample_erb_orbit_grid",
    "trace_l_shell_b_over_b0",
    "validate_erb_energy_spectrum",
]
