"""RADAR calculation core."""

from radar.core.dose_units import (
    default_dose_unit_for_radiation_quantity_mode,
    dose_quantity_for_radiation_quantity_mode,
    dose_quantity_for_spectrum_quantity,
    is_accumulated_dose_unit,
    is_dose_rate_unit,
    radiation_quantity_mode_for_spectrum_quantity,
    validate_dose_quantity_for_radiation_quantity_mode,
    validate_dose_settings_for_radiation_quantity_mode,
    validate_dose_unit_for_quantity,
)
from radar.core.log import CalculationLog, CalculationLogEntry, LogLevel
from radar.core.product_display import (
    average_daily_spectrum_for_mission_years,
    average_daily_spectrum_for_product,
    daily_display_unit_for_product_kind,
)
from radar.core.products import (
    SpectrumProduct,
    allowed_spectrum_quantities_for_product_kind,
    default_dose_unit_for_product_kind,
    dose_quantity_for_product_kind,
    radiation_quantity_mode_for_product_kind,
    expected_spectrum_unit_for_product_kind,
    validate_spectrum_matches_product_kind,
)
from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig, ShieldingConfig
from radar.core.result import CalculationResult, ComponentStatus, ModelInfo
from radar.core.spectra import Spectrum1D
from radar.core.source_products import (
    allowed_product_kinds_for_source,
    is_product_kind_allowed_for_source,
    validate_product_allowed_for_source,
    validate_product_spectra_match_spectra,
    validate_products_allowed_for_source,
    validate_products_match_spectra_and_source,
    validate_product_kind_allowed_for_source,
    validate_spectra_match_expected,
)
from radar.core.spectrum_ops import add_spectra, check_spectra_compatible, scale_spectrum
from radar.core.types import DoseQuantity, RadiationProductKind, RadiationQuantityMode

__all__ = [
    "CalculationConfig",
    "CalculationLog",
    "CalculationLogEntry",
    "CalculationResult",
    "ComponentStatus",
    "DoseQuantity",
    "LogLevel",
    "MissionConfig",
    "ModelInfo",
    "OrbitConfig",
    "RadiationProductKind",
    "RadiationQuantityMode",
    "ShieldingConfig",
    "Spectrum1D",
    "SpectrumProduct",
    "average_daily_spectrum_for_mission_years",
    "average_daily_spectrum_for_product",
    "add_spectra",
    "allowed_product_kinds_for_source",
    "allowed_spectrum_quantities_for_product_kind",
    "check_spectra_compatible",
    "daily_display_unit_for_product_kind",
    "default_dose_unit_for_product_kind",
    "expected_spectrum_unit_for_product_kind",
    "default_dose_unit_for_radiation_quantity_mode",
    "dose_quantity_for_product_kind",
    "dose_quantity_for_radiation_quantity_mode",
    "dose_quantity_for_spectrum_quantity",
    "is_accumulated_dose_unit",
    "is_dose_rate_unit",
    "is_product_kind_allowed_for_source",
    "radiation_quantity_mode_for_product_kind",
    "radiation_quantity_mode_for_spectrum_quantity",
    "scale_spectrum",
    "validate_dose_quantity_for_radiation_quantity_mode",
    "validate_dose_settings_for_radiation_quantity_mode",
    "validate_dose_unit_for_quantity",
    "validate_product_allowed_for_source",
    "validate_product_spectra_match_spectra",
    "validate_products_allowed_for_source",
    "validate_products_match_spectra_and_source",
    "validate_product_kind_allowed_for_source",
    "validate_spectra_match_expected",
    "validate_spectrum_matches_product_kind",
]
