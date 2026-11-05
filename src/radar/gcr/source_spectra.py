"""GCR source spectra outside the magnetosphere."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Final

from radar.core.profiles import GOST_GCR_DOCUMENT, OST_134_1044_2007_DOCUMENT
from radar.core.spectra import ModelReference, ParticleSpecies, Spectrum1D, TransformationStep
from radar.core.types import (
    AngularBasis,
    Particle,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit
from radar.gcr.components import (
    GostGcrSolarState,
    OstGcrSolarState,
    gost_main_gcr_differential_flux_directional_m2_s_sr_gev,
    ost_akl_differential_flux_directional_cm2_s_sr_mev,
    ost_akl_gamma,
    ost_main_gcr_differential_flux_directional_cm2_s_sr_mev,
    ost_nek_differential_flux_directional_cm2_s_sr_mev,
)
from radar.gcr.kinematics import (
    low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev,
    main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev,
    mev_per_nucleon_to_gev_per_nucleon,
)
from radar.gcr.parameters import (
    GCR_ELEMENT_SYMBOLS,
    AklParameters,
    GcrIonParameters,
    OST_NEK_HE_BC_RELEASE_ELIGIBLE,
    OST_NEK_HE_BC_USED,
    OST_NEK_HE_BC_VALIDATION_STATUS,
    get_gost_gcr_ion_parameters,
    get_ost_akl_parameters,
    get_ost_gcr_ion_parameters,
    get_ost_nek_parameter,
)

GOST_GCR_SOURCE_SPECTRA_MODEL: Final[str] = "gost_gcr_source_outside_magnetosphere"
OST_GCR_SOURCE_SPECTRA_MODEL: Final[str] = "ost_gcr_source_outside_magnetosphere"

GCR_MAIN_COMPONENT: Final[str] = "main"
GCR_NEK_COMPONENT: Final[str] = "nek"
GCR_AKL_COMPONENT: Final[str] = "akl"
GCR_TOTAL_COMPONENT: Final[str] = "total"

OST_GCR_SOURCE_VERSION: Final[str] = "ost_134_1044_2007_change_2_appendix_v_v3"
GOST_GCR_SOURCE_VERSION: Final[str] = "gost_gcr_draft_main_plus_ost_appendix_v_v2"
GCR_WOLF_DATASET_REF: Final[str] = "ost_134_1044_2007/table_g_1_wolf_numbers.csv"
GOST_GCR_DRAFT_INCOMPLETE_REASON: Final[str] = (
    "GOST GCR source parameters were transcribed from a supplied draft whose "
    "publication status has not been established"
)
OST_NEK_HE_BC_INCOMPLETE_REASON: Final[str] = (
    "OST NEK helium b_c is pending an isolated COSRAD comparison; the printed "
    "6.6e-2 value is provisional"
)
OST_AKL_CONVENTIONS_INCOMPLETE_REASON: Final[str] = (
    "OST Appendix V.13 AKL angular, energy and time conventions are unconfirmed: "
    "the printed cm^-2 s^-1 MeV^-1 and unspecified time unit do not establish "
    "the implemented directional per-nucleon spectrum, 4*pi conversion and years"
)
OST_AKL_GAMMA_GUARD_INCOMPLETE_REASON: Final[str] = (
    "OST Appendix V.13 AKL was set to zero by the implementation guard for "
    "gamma <= 0; this suppression rule is not stated in the supplied standard"
)

GCR_SOURCE_SPECTRA_UNITS_NOTE: Final[str] = (
    "GCR source spectra are returned in RADAR omnidirectional differential-flux "
    "units cm^-2 s^-1 (MeV/nucleon)^-1 on a MeV/nucleon energy grid. The GOST "
    "main component is converted from m^-2 s^-1 sr^-1 (GeV/nucleon)^-1 with "
    "4*pi*1e-7. OST components are evaluated under the directional "
    "cm^-2 s^-1 sr^-1 (MeV/nucleon)^-1 convention and converted only with 4*pi. "
    "For AKL, this angular/energy convention and time in years are provisional: "
    "the supplied V.13 prints cm^-2 s^-1 MeV^-1 and no time unit."
)

OstSolarStateForEnergy = Callable[[float, GcrIonParameters], OstGcrSolarState]


@dataclass(frozen=True)
class GcrIonSourceSpectra:
    """Main, NEK, AKL and total source spectra for one GCR ion."""

    symbol: str
    z: int
    main: Spectrum1D
    nek: Spectrum1D
    akl: Spectrum1D
    total: Spectrum1D

    @property
    def particle(self) -> Particle:
        """Return RADAR particle group for this ion."""

        return _particle_for_z(self.z)

    @property
    def all_spectra(self) -> tuple[Spectrum1D, ...]:
        """Return all component spectra in deterministic order."""

        return (self.main, self.nek, self.akl, self.total)

    def __post_init__(self) -> None:
        if not self.symbol:
            msg = "GCR ion source spectrum symbol must not be empty."
            raise ValueError(msg)

        if self.z < 1:
            msg = "GCR ion source spectrum charge Z must be positive."
            raise ValueError(msg)

        expected_particle = _particle_for_z(self.z)
        expected_grid = self.total.x

        for spectrum in self.all_spectra:
            if spectrum.source is not RadiationSource.GCR:
                msg = "GCR source spectrum source must be GCR."
                raise ValueError(msg)

            if spectrum.particle is not expected_particle:
                msg = "GCR source spectrum particle does not match ion charge."
                raise ValueError(msg)

            expected_x_unit = Unit.MEV if self.z == 1 else Unit.MEV_PER_NUCLEON
            if spectrum.x_unit is not expected_x_unit:
                msg = "GCR source spectrum energy grid uses the wrong typed energy unit."
                raise ValueError(msg)

            expected_y_unit = (
                Unit.DIFFERENTIAL_FLUX
                if self.z == 1
                else Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY
            )
            if spectrum.y_unit is not expected_y_unit:
                msg = "GCR source spectrum values must use differential-flux units."
                raise ValueError(msg)

            if spectrum.quantity is not SpectrumQuantity.DIFFERENTIAL_FLUX:
                msg = "GCR source spectrum quantity must be differential flux."
                raise ValueError(msg)

            if spectrum.x != expected_grid:
                msg = "GCR source spectrum components must share the same energy grid."
                raise ValueError(msg)


@dataclass(frozen=True)
class GcrSourceSpectra:
    """Collection of GCR source spectra outside the magnetosphere."""

    model: str
    ions: tuple[GcrIonSourceSpectra, ...]

    @property
    def total_spectra(self) -> tuple[Spectrum1D, ...]:
        """Return one total spectrum per ion."""

        return tuple(ion.total for ion in self.ions)

    @property
    def all_component_spectra(self) -> tuple[Spectrum1D, ...]:
        """Return all component spectra for all ions."""

        return tuple(spectrum for ion in self.ions for spectrum in ion.all_spectra)

    def by_symbol(self) -> dict[str, GcrIonSourceSpectra]:
        """Return ion spectra indexed by element symbol."""

        return {ion.symbol: ion for ion in self.ions}

    def ion(self, symbol: str) -> GcrIonSourceSpectra:
        """Return ion spectra for one element symbol."""

        spectra = self.by_symbol().get(symbol)

        if spectra is None:
            msg = f"GCR source spectra do not contain ion symbol {symbol!r}."
            raise KeyError(msg)

        return spectra

    def __post_init__(self) -> None:
        if not self.model:
            msg = "GCR source spectra model name must not be empty."
            raise ValueError(msg)

        if not self.ions:
            msg = "GCR source spectra result must contain at least one ion."
            raise ValueError(msg)

        seen_symbols: set[str] = set()
        for ion in self.ions:
            if ion.symbol in seen_symbols:
                msg = f"Duplicate GCR source spectra for ion symbol {ion.symbol!r}."
                raise ValueError(msg)
            seen_symbols.add(ion.symbol)


def _validate_non_negative_finite(
    *,
    value: float,
    name: str,
) -> None:
    if not isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)

    if value < 0.0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)


def _validate_energy_grid(
    energy_grid_mev_per_nucleon: tuple[float, ...],
) -> None:
    if not energy_grid_mev_per_nucleon:
        msg = "GCR source spectrum energy grid must not be empty."
        raise ValueError(msg)

    if any(not isfinite(energy) for energy in energy_grid_mev_per_nucleon):
        msg = "GCR source spectrum energy grid must contain only finite values."
        raise ValueError(msg)

    if any(energy <= 0.0 for energy in energy_grid_mev_per_nucleon):
        msg = "GCR source spectrum energy grid values must be positive."
        raise ValueError(msg)

    if any(
        right <= left
        for left, right in zip(
            energy_grid_mev_per_nucleon,
            energy_grid_mev_per_nucleon[1:],
        )
    ):
        msg = "GCR source spectrum energy grid must be strictly increasing."
        raise ValueError(msg)


def _particle_for_z(z: int) -> Particle:
    return Particle.PROTON if z == 1 else Particle.HZE


def _component_model_name(
    *,
    model: str,
    symbol: str,
    component: str,
) -> str:
    return f"{model}:{symbol}:{component}"


def _make_spectrum(
    *,
    energy_grid_mev_per_nucleon: tuple[float, ...],
    values: tuple[float, ...],
    particle: Particle,
    model: str,
    parameters: GcrIonParameters,
    component: str,
    version: str,
    document_refs: tuple[str, ...],
    completeness: SpectrumCompleteness,
    incomplete_reasons: tuple[str, ...],
    additional_transformation_parameters: tuple[tuple[str, str], ...] = (),
) -> Spectrum1D:
    is_proton = particle is Particle.PROTON
    transformation_parameters = [
        ("component", component),
        ("particle", f"Z={parameters.z},A={int(round(parameters.mass_number))}"),
        # The source formula uses the tabulated (possibly fractional) parameter.
        # The legacy integer species label is NOT an identified isotope or an
        # approved mass for a stopping-power calculation.
        ("source_tabulated_mass_parameter", f"{parameters.mass_number:.17g}"),
        ("species_mass_label_policy", "legacy_round_not_isotope_identification"),
        ("transport_mass_policy", "not_established"),
        ("energy_basis", "MeV_per_nucleon"),
        ("angular_conversion", "directional_times_4pi"),
        *additional_transformation_parameters,
    ]
    is_ost_nek_he_candidate = (
        model == _component_model_name(
            model=OST_GCR_SOURCE_SPECTRA_MODEL,
            symbol="He",
            component=component,
        )
        and component in (GCR_NEK_COMPONENT, GCR_TOTAL_COMPONENT)
    )
    if is_ost_nek_he_candidate:
        transformation_parameters.extend(
            (
                ("ost_nek_he_bc_used", f"{OST_NEK_HE_BC_USED:.17g}"),
                (
                    "ost_nek_he_bc_validation_status",
                    OST_NEK_HE_BC_VALIDATION_STATUS,
                ),
            )
        )
        if not OST_NEK_HE_BC_RELEASE_ELIGIBLE:
            completeness = SpectrumCompleteness.INCOMPLETE
            incomplete_reasons = tuple(
                dict.fromkeys(
                    (*incomplete_reasons, OST_NEK_HE_BC_INCOMPLETE_REASON),
                )
            )

    return Spectrum1D(
        x=energy_grid_mev_per_nucleon,
        y=values,
        x_unit=Unit.MEV if is_proton else Unit.MEV_PER_NUCLEON,
        y_unit=(
            Unit.DIFFERENTIAL_FLUX
            if is_proton
            else Unit.DIFFERENTIAL_FLUX_PER_NUCLEON_ENERGY
        ),
        quantity=SpectrumQuantity.DIFFERENTIAL_FLUX,
        particle=particle,
        source=RadiationSource.GCR,
        model=model,
        axis_kind=(
            SpectrumAxisKind.KINETIC_ENERGY
            if is_proton
            else SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
        ),
        angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
        temporal_basis=TemporalBasis.SOURCE_TIME_BIN,
        location=SpectrumLocation.OUTSIDE_MAGNETOSPHERE,
        species=ParticleSpecies(
            symbol=parameters.symbol,
            atomic_number=parameters.z,
            mass_number=int(round(parameters.mass_number)),
        ),
        model_reference=ModelReference(
            name=model,
            version=version,
            document_refs=document_refs,
            dataset_refs=(GCR_WOLF_DATASET_REF,),
        ),
        transformation_chain=(
            TransformationStep(
                name="gcr_source_spectrum",
                version=version,
                parameters=tuple(transformation_parameters),
            ),
        ),
        completeness=completeness,
        incomplete_reasons=incomplete_reasons,
    )


def _zero_values(
    energy_grid_mev_per_nucleon: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(0.0 for _ in energy_grid_mev_per_nucleon)


def _select_ion_parameters(
    *,
    parameters_by_symbol: dict[str, GcrIonParameters],
    symbols: tuple[str, ...],
) -> tuple[GcrIonParameters, ...]:
    selected: list[GcrIonParameters] = []

    for symbol in symbols:
        parameters = parameters_by_symbol.get(symbol)

        if parameters is None:
            msg = f"Unknown GCR ion symbol {symbol!r}."
            raise ValueError(msg)

        selected.append(parameters)

    return tuple(selected)


def _gost_main_values(
    *,
    energy_grid_mev_per_nucleon: tuple[float, ...],
    parameters: GcrIonParameters,
    solar_state: GostGcrSolarState,
) -> tuple[float, ...]:
    values: list[float] = []

    for energy_mev_per_nucleon in energy_grid_mev_per_nucleon:
        directional_flux = gost_main_gcr_differential_flux_directional_m2_s_sr_gev(
            kinetic_energy_gev_per_nucleon=mev_per_nucleon_to_gev_per_nucleon(
                energy_mev_per_nucleon,
            ),
            parameters=parameters,
            solar_state=solar_state,
        )
        values.append(
            main_gcr_directional_m2_gev_to_omnidirectional_cm2_mev(
                directional_flux,
            )
        )

    return tuple(values)


def _ost_main_values(
    *,
    energy_grid_mev_per_nucleon: tuple[float, ...],
    parameters: GcrIonParameters,
    solar_state: OstGcrSolarState,
    solar_state_for_energy: OstSolarStateForEnergy | None = None,
) -> tuple[float, ...]:
    values: list[float] = []

    for energy_mev_per_nucleon in energy_grid_mev_per_nucleon:
        energy_solar_state = (
            solar_state_for_energy(energy_mev_per_nucleon, parameters)
            if solar_state_for_energy is not None
            else solar_state
        )
        directional_flux = ost_main_gcr_differential_flux_directional_cm2_s_sr_mev(
            kinetic_energy_gev_per_nucleon=mev_per_nucleon_to_gev_per_nucleon(
                energy_mev_per_nucleon,
            ),
            parameters=parameters,
            solar_state=energy_solar_state,
        )
        values.append(
            low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(
                directional_flux,
            )
        )

    return tuple(values)


def _nek_values(
    *,
    energy_grid_mev_per_nucleon: tuple[float, ...],
    z: int,
    wolf_number: float,
) -> tuple[float, ...]:
    parameters = get_ost_nek_parameter(z)

    if parameters is None:
        return _zero_values(energy_grid_mev_per_nucleon)

    values: list[float] = []

    for energy_mev_per_nucleon in energy_grid_mev_per_nucleon:
        directional_flux = ost_nek_differential_flux_directional_cm2_s_sr_mev(
            kinetic_energy_mev_per_nucleon=energy_mev_per_nucleon,
            parameters=parameters,
            wolf_number=wolf_number,
        )
        values.append(
            low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(
                directional_flux,
            )
        )

    return tuple(values)


def _akl_values(
    *,
    energy_grid_mev_per_nucleon: tuple[float, ...],
    parameters: AklParameters | None,
    years_from_cycle_start: float,
) -> tuple[float, ...]:
    if parameters is None:
        return _zero_values(energy_grid_mev_per_nucleon)

    values: list[float] = []

    for energy_mev_per_nucleon in energy_grid_mev_per_nucleon:
        directional_flux = ost_akl_differential_flux_directional_cm2_s_sr_mev(
            kinetic_energy_mev_per_nucleon=energy_mev_per_nucleon,
            parameters=parameters,
            years_from_cycle_start=years_from_cycle_start,
        )
        values.append(
            low_energy_directional_cm2_mev_to_omnidirectional_cm2_mev(
                directional_flux,
            )
        )

    return tuple(values)


def _total_values(
    *,
    main_values: tuple[float, ...],
    nek_values: tuple[float, ...],
    akl_values: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        main + nek + akl
        for main, nek, akl in zip(
            main_values,
            nek_values,
            akl_values,
            strict=True,
        )
    )


def _akl_method_provenance(
    *,
    parameters: AklParameters | None,
    years_from_cycle_start: float,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Keep unresolved V.13 assumptions local to actual AKL contributors."""

    if parameters is None or parameters.c == 0.0:
        return (), ()

    gamma = ost_akl_gamma(years_from_cycle_start)
    guard_active = gamma <= 0.0
    reasons: tuple[str, ...] = (OST_AKL_CONVENTIONS_INCOMPLETE_REASON,)
    if guard_active:
        reasons = (*reasons, OST_AKL_GAMMA_GUARD_INCOMPLETE_REASON)
    return reasons, (
        ("ost_akl_conventions_status", "pending_source_or_reference_comparison"),
        ("ost_akl_printed_flux_units", "cm^-2 s^-1 MeV^-1"),
        ("ost_akl_assumed_input_flux_units", "cm^-2 sr^-1 s^-1 (MeV/nucleon)^-1"),
        ("ost_akl_angular_conversion", "directional_times_4pi_provisional"),
        ("ost_akl_assumed_time_unit", "years_since_solar_cycle_start"),
        ("ost_akl_years_from_cycle_start", f"{years_from_cycle_start:.17g}"),
        ("ost_akl_gamma", f"{gamma:.17g}"),
        ("ost_akl_gamma_nonpositive_policy", "set_zero_not_stated_in_standard"),
        ("ost_akl_gamma_zeroing_guard_active", str(guard_active).lower()),
    )


def _build_ion_source_spectra(
    *,
    model: str,
    energy_grid_mev_per_nucleon: tuple[float, ...],
    parameters: GcrIonParameters,
    main_values: tuple[float, ...],
    nek_values: tuple[float, ...],
    akl_values: tuple[float, ...],
    akl_parameters: AklParameters | None,
    akl_years_from_cycle_start: float,
    version: str,
    document_refs: tuple[str, ...],
    completeness: SpectrumCompleteness,
    incomplete_reasons: tuple[str, ...],
) -> GcrIonSourceSpectra:
    particle = _particle_for_z(parameters.z)
    total_values = _total_values(
        main_values=main_values,
        nek_values=nek_values,
        akl_values=akl_values,
    )
    akl_reasons, akl_metadata = _akl_method_provenance(
        parameters=akl_parameters,
        years_from_cycle_start=akl_years_from_cycle_start,
    )
    akl_completeness = SpectrumCompleteness.INCOMPLETE if akl_reasons else completeness
    combined_akl_reasons = tuple(dict.fromkeys((*incomplete_reasons, *akl_reasons)))

    return GcrIonSourceSpectra(
        symbol=parameters.symbol,
        z=parameters.z,
        main=_make_spectrum(
            energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
            values=main_values,
            particle=particle,
            parameters=parameters,
            component=GCR_MAIN_COMPONENT,
            version=version,
            document_refs=document_refs,
            completeness=completeness,
            incomplete_reasons=incomplete_reasons,
            model=_component_model_name(
                model=model,
                symbol=parameters.symbol,
                component=GCR_MAIN_COMPONENT,
            ),
        ),
        nek=_make_spectrum(
            energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
            values=nek_values,
            particle=particle,
            parameters=parameters,
            component=GCR_NEK_COMPONENT,
            version=version,
            document_refs=document_refs,
            completeness=completeness,
            incomplete_reasons=incomplete_reasons,
            model=_component_model_name(
                model=model,
                symbol=parameters.symbol,
                component=GCR_NEK_COMPONENT,
            ),
        ),
        akl=_make_spectrum(
            energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
            values=akl_values,
            particle=particle,
            parameters=parameters,
            component=GCR_AKL_COMPONENT,
            version=version,
            document_refs=document_refs,
            completeness=akl_completeness,
            incomplete_reasons=combined_akl_reasons,
            additional_transformation_parameters=akl_metadata,
            model=_component_model_name(
                model=model,
                symbol=parameters.symbol,
                component=GCR_AKL_COMPONENT,
            ),
        ),
        total=_make_spectrum(
            energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
            values=total_values,
            particle=particle,
            parameters=parameters,
            component=GCR_TOTAL_COMPONENT,
            version=version,
            document_refs=document_refs,
            completeness=akl_completeness,
            incomplete_reasons=combined_akl_reasons,
            additional_transformation_parameters=akl_metadata,
            model=_component_model_name(
                model=model,
                symbol=parameters.symbol,
                component=GCR_TOTAL_COMPONENT,
            ),
        ),
    )


def calculate_gost_gcr_source_spectra_outside_magnetosphere(
    *,
    energy_grid_mev_per_nucleon: tuple[float, ...],
    solar_state: GostGcrSolarState,
    nek_wolf_number: float,
    akl_years_from_cycle_start: float,
    symbols: tuple[str, ...] = GCR_ELEMENT_SYMBOLS,
    version: str = GOST_GCR_SOURCE_VERSION,
    document_refs: tuple[str, ...] = (
        GOST_GCR_DOCUMENT,
        OST_134_1044_2007_DOCUMENT,
    ),
) -> GcrSourceSpectra:
    """Calculate GOST main GCR plus OST NEK/AKL source spectra.

    Returned spectra are outside the magnetosphere, before geomagnetic
    penetration, shielding, LET, dose or SEE processing.
    """

    _validate_energy_grid(energy_grid_mev_per_nucleon)
    _validate_non_negative_finite(
        value=nek_wolf_number,
        name="GCR NEK Wolf number",
    )
    _validate_non_negative_finite(
        value=akl_years_from_cycle_start,
        name="GCR AKL time from solar-cycle start",
    )

    akl_parameters_by_symbol = get_ost_akl_parameters()
    ions: list[GcrIonSourceSpectra] = []

    for parameters in _select_ion_parameters(
        parameters_by_symbol=get_gost_gcr_ion_parameters(),
        symbols=symbols,
    ):
        main_values = _gost_main_values(
            energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
            parameters=parameters,
            solar_state=solar_state,
        )
        nek_values = _nek_values(
            energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
            z=parameters.z,
            wolf_number=nek_wolf_number,
        )
        akl_values = _akl_values(
            energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
            parameters=akl_parameters_by_symbol.get(parameters.symbol),
            years_from_cycle_start=akl_years_from_cycle_start,
        )
        ions.append(
            _build_ion_source_spectra(
                model=GOST_GCR_SOURCE_SPECTRA_MODEL,
                energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
                parameters=parameters,
                main_values=main_values,
                nek_values=nek_values,
                akl_values=akl_values,
                akl_parameters=akl_parameters_by_symbol.get(parameters.symbol),
                akl_years_from_cycle_start=akl_years_from_cycle_start,
                version=version,
                document_refs=document_refs,
                completeness=SpectrumCompleteness.INCOMPLETE,
                incomplete_reasons=(GOST_GCR_DRAFT_INCOMPLETE_REASON,),
            )
        )

    return GcrSourceSpectra(
        model=GOST_GCR_SOURCE_SPECTRA_MODEL,
        ions=tuple(ions),
    )


def calculate_ost_gcr_source_spectra_outside_magnetosphere(
    *,
    energy_grid_mev_per_nucleon: tuple[float, ...],
    solar_state: OstGcrSolarState,
    nek_wolf_number: float,
    akl_years_from_cycle_start: float,
    symbols: tuple[str, ...] = GCR_ELEMENT_SYMBOLS,
    solar_state_for_energy: OstSolarStateForEnergy | None = None,
    version: str = OST_GCR_SOURCE_VERSION,
    document_refs: tuple[str, ...] = (OST_134_1044_2007_DOCUMENT,),
) -> GcrSourceSpectra:
    """Calculate OST main GCR plus OST NEK/AKL source spectra.

    Returned spectra are outside the magnetosphere, before geomagnetic
    penetration, shielding, LET, dose or SEE processing.
    """

    _validate_energy_grid(energy_grid_mev_per_nucleon)
    _validate_non_negative_finite(
        value=nek_wolf_number,
        name="GCR NEK Wolf number",
    )
    _validate_non_negative_finite(
        value=akl_years_from_cycle_start,
        name="GCR AKL time from solar-cycle start",
    )

    akl_parameters_by_symbol = get_ost_akl_parameters()
    ions: list[GcrIonSourceSpectra] = []

    for parameters in _select_ion_parameters(
        parameters_by_symbol=get_ost_gcr_ion_parameters(),
        symbols=symbols,
    ):
        main_values = _ost_main_values(
            energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
            parameters=parameters,
            solar_state=solar_state,
            solar_state_for_energy=solar_state_for_energy,
        )
        nek_values = _nek_values(
            energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
            z=parameters.z,
            wolf_number=nek_wolf_number,
        )
        akl_values = _akl_values(
            energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
            parameters=akl_parameters_by_symbol.get(parameters.symbol),
            years_from_cycle_start=akl_years_from_cycle_start,
        )
        ions.append(
            _build_ion_source_spectra(
                model=OST_GCR_SOURCE_SPECTRA_MODEL,
                energy_grid_mev_per_nucleon=energy_grid_mev_per_nucleon,
                parameters=parameters,
                main_values=main_values,
                nek_values=nek_values,
                akl_values=akl_values,
                akl_parameters=akl_parameters_by_symbol.get(parameters.symbol),
                akl_years_from_cycle_start=akl_years_from_cycle_start,
                version=version,
                document_refs=document_refs,
                completeness=SpectrumCompleteness.COMPLETE,
                incomplete_reasons=(),
            )
        )

    return GcrSourceSpectra(
        model=OST_GCR_SOURCE_SPECTRA_MODEL,
        ions=tuple(ions),
    )


__all__ = [
    "GCR_AKL_COMPONENT",
    "GCR_MAIN_COMPONENT",
    "GCR_NEK_COMPONENT",
    "GCR_SOURCE_SPECTRA_UNITS_NOTE",
    "GCR_TOTAL_COMPONENT",
    "GCR_WOLF_DATASET_REF",
    "GOST_GCR_DRAFT_INCOMPLETE_REASON",
    "OST_NEK_HE_BC_INCOMPLETE_REASON",
    "OST_AKL_CONVENTIONS_INCOMPLETE_REASON",
    "OST_AKL_GAMMA_GUARD_INCOMPLETE_REASON",
    "GOST_GCR_SOURCE_VERSION",
    "OST_GCR_SOURCE_VERSION",
    "GOST_GCR_SOURCE_SPECTRA_MODEL",
    "OST_GCR_SOURCE_SPECTRA_MODEL",
    "OstSolarStateForEnergy",
    "GcrIonSourceSpectra",
    "GcrSourceSpectra",
    "calculate_gost_gcr_source_spectra_outside_magnetosphere",
    "calculate_ost_gcr_source_spectra_outside_magnetosphere",
]
