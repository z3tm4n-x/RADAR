"""Source radiation product specification for RADAR calculation chains."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from radar.core.types import Particle, RadiationProductKind, RadiationSource


class RadiationProductPurpose(StrEnum):
    """Calculation purpose of a radiation product."""

    ACCUMULATED_DOSE = "accumulated_dose"
    DOSE_RATE = "dose_rate"
    SINGLE_EVENT_EFFECT = "single_event_effect"
    OUTPUT_CHARACTERISTIC = "output_characteristic"
    INTERMEDIATE = "intermediate"


class RadiationProductLocation(StrEnum):
    """Calculation location where a radiation product is defined."""

    BEFORE_SHIELDING = "before_shielding"
    BEHIND_SHIELDING = "behind_shielding"


class RadiationProductNormativeStatus(StrEnum):
    """Normative role of a radiation product."""

    OST_MINIMUM = "ost_minimum"
    EXTENDED_BALANCE = "extended_balance"
    OUTPUT_FORM = "output_form"
    INTERMEDIATE = "intermediate"


@dataclass(frozen=True)
class SourceRadiationProductSpec:
    """Specification of a source radiation product required by RADAR."""

    source: RadiationSource
    particle: Particle
    product_kind: RadiationProductKind
    location: RadiationProductLocation
    purposes: tuple[RadiationProductPurpose, ...]
    normative_status: RadiationProductNormativeStatus
    title: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.purposes:
            msg = "Source radiation product spec must contain at least one purpose."
            raise ValueError(msg)

        if not self.title:
            msg = "Source radiation product spec title must not be empty."
            raise ValueError(msg)


SOURCE_RADIATION_PRODUCT_SPECS: tuple[SourceRadiationProductSpec, ...] = (
    SourceRadiationProductSpec(
        source=RadiationSource.SEP,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MISSION_FLUENCE,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.ACCUMULATED_DOSE,),
        normative_status=RadiationProductNormativeStatus.OST_MINIMUM,
        title="Флюенс протонов СКЛ за САС",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.SEP,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.PEAK_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.SINGLE_EVENT_EFFECT,),
        normative_status=RadiationProductNormativeStatus.OST_MINIMUM,
        title="Пиковый поток протонов СКЛ",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.SEP,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MEAN_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(
            RadiationProductPurpose.INTERMEDIATE,
            RadiationProductPurpose.OUTPUT_CHARACTERISTIC,
        ),
        normative_status=RadiationProductNormativeStatus.OUTPUT_FORM,
        title="Средний поток протонов СКЛ",
        note="Продукт для полной формы выходных характеристик по средним потокам.",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.SEP,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.PEAK_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(
            RadiationProductPurpose.INTERMEDIATE,
            RadiationProductPurpose.SINGLE_EVENT_EFFECT,
        ),
        normative_status=RadiationProductNormativeStatus.INTERMEDIATE,
        title="Пиковый поток ТЗЧ СКЛ до защиты",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.SEP,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MEAN_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(
            RadiationProductPurpose.INTERMEDIATE,
            RadiationProductPurpose.SINGLE_EVENT_EFFECT,
        ),
        normative_status=RadiationProductNormativeStatus.INTERMEDIATE,
        title="Средний поток ТЗЧ СКЛ до защиты",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.SEP,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MISSION_FLUENCE,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(
            RadiationProductPurpose.INTERMEDIATE,
            RadiationProductPurpose.ACCUMULATED_DOSE,
        ),
        normative_status=RadiationProductNormativeStatus.EXTENDED_BALANCE,
        title="Флюенс ТЗЧ СКЛ за САС до защиты",
        note="Промежуточная величина для ЛПЭ-спектра за защитой.",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.SEP,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.PEAK_LET_FLUX,
        location=RadiationProductLocation.BEHIND_SHIELDING,
        purposes=(RadiationProductPurpose.SINGLE_EVENT_EFFECT,),
        normative_status=RadiationProductNormativeStatus.OUTPUT_FORM,
        title="ЛПЭ пикового потока ТЗЧ СКЛ за защитой",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.SEP,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MEAN_LET_FLUX,
        location=RadiationProductLocation.BEHIND_SHIELDING,
        purposes=(RadiationProductPurpose.SINGLE_EVENT_EFFECT,),
        normative_status=RadiationProductNormativeStatus.OUTPUT_FORM,
        title="ЛПЭ среднего потока ТЗЧ СКЛ за защитой",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.SEP,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MISSION_LET_FLUENCE,
        location=RadiationProductLocation.BEHIND_SHIELDING,
        purposes=(RadiationProductPurpose.ACCUMULATED_DOSE,),
        normative_status=RadiationProductNormativeStatus.EXTENDED_BALANCE,
        title="ЛПЭ-флюенс ТЗЧ СКЛ за САС",
        note="Расширенный расчётный вклад, если используется баланс дозы от ТЗЧ.",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.GCR,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MEAN_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(
            RadiationProductPurpose.INTERMEDIATE,
            RadiationProductPurpose.ACCUMULATED_DOSE,
        ),
        normative_status=RadiationProductNormativeStatus.EXTENDED_BALANCE,
        title="Средний поток протонов ГКЛ",
        note="Используется для получения флюенса ГКЛ за САС.",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.GCR,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MISSION_FLUENCE,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.ACCUMULATED_DOSE,),
        normative_status=RadiationProductNormativeStatus.EXTENDED_BALANCE,
        title="Флюенс протонов ГКЛ за САС",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.GCR,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MAXIMUM_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.SINGLE_EVENT_EFFECT,),
        normative_status=RadiationProductNormativeStatus.OST_MINIMUM,
        title="Максимальный поток протонов ГКЛ",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MEAN_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(
            RadiationProductPurpose.INTERMEDIATE,
            RadiationProductPurpose.SINGLE_EVENT_EFFECT,
        ),
        normative_status=RadiationProductNormativeStatus.INTERMEDIATE,
        title="Средний поток ТЗЧ ГКЛ до защиты",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MAXIMUM_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(
            RadiationProductPurpose.INTERMEDIATE,
            RadiationProductPurpose.SINGLE_EVENT_EFFECT,
        ),
        normative_status=RadiationProductNormativeStatus.INTERMEDIATE,
        title="Максимальный поток ТЗЧ ГКЛ до защиты",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MISSION_FLUENCE,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(
            RadiationProductPurpose.INTERMEDIATE,
            RadiationProductPurpose.ACCUMULATED_DOSE,
        ),
        normative_status=RadiationProductNormativeStatus.EXTENDED_BALANCE,
        title="Флюенс ТЗЧ ГКЛ за САС до защиты",
        note="Промежуточная величина для ЛПЭ-спектра за защитой.",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MAXIMUM_LET_FLUX,
        location=RadiationProductLocation.BEHIND_SHIELDING,
        purposes=(RadiationProductPurpose.SINGLE_EVENT_EFFECT,),
        normative_status=RadiationProductNormativeStatus.OUTPUT_FORM,
        title="ЛПЭ максимального потока ТЗЧ ГКЛ за защитой",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MEAN_LET_FLUX,
        location=RadiationProductLocation.BEHIND_SHIELDING,
        purposes=(RadiationProductPurpose.SINGLE_EVENT_EFFECT,),
        normative_status=RadiationProductNormativeStatus.OUTPUT_FORM,
        title="ЛПЭ среднего потока ТЗЧ ГКЛ за защитой",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.GCR,
        particle=Particle.HZE,
        product_kind=RadiationProductKind.MISSION_LET_FLUENCE,
        location=RadiationProductLocation.BEHIND_SHIELDING,
        purposes=(RadiationProductPurpose.ACCUMULATED_DOSE,),
        normative_status=RadiationProductNormativeStatus.EXTENDED_BALANCE,
        title="ЛПЭ-флюенс ТЗЧ ГКЛ за САС",
        note="Расширенный расчётный вклад, если используется баланс дозы от ТЗЧ.",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.ERB,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.ORBIT_AVERAGED_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.INTERMEDIATE,),
        normative_status=RadiationProductNormativeStatus.OST_MINIMUM,
        title="Орбитально-усреднённый поток протонов ЕРПЗ",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.ERB,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MEAN_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.INTERMEDIATE,),
        normative_status=RadiationProductNormativeStatus.INTERMEDIATE,
        title="Средний поток протонов ЕРПЗ",
        note="Допустимая промежуточная величина; для ОЭ используется максимальный поток.",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.ERB,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MISSION_FLUENCE,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.ACCUMULATED_DOSE,),
        normative_status=RadiationProductNormativeStatus.OST_MINIMUM,
        title="Флюенс протонов ЕРПЗ за САС",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.ERB,
        particle=Particle.PROTON,
        product_kind=RadiationProductKind.MAXIMUM_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.SINGLE_EVENT_EFFECT,),
        normative_status=RadiationProductNormativeStatus.OST_MINIMUM,
        title="Максимальный поток протонов ЕРПЗ",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.ERB,
        particle=Particle.ELECTRON,
        product_kind=RadiationProductKind.ORBIT_AVERAGED_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.INTERMEDIATE,),
        normative_status=RadiationProductNormativeStatus.OST_MINIMUM,
        title="Орбитально-усреднённый поток электронов ЕРПЗ",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.ERB,
        particle=Particle.ELECTRON,
        product_kind=RadiationProductKind.MEAN_FLUX,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.INTERMEDIATE,),
        normative_status=RadiationProductNormativeStatus.INTERMEDIATE,
        title="Средний поток электронов ЕРПЗ",
        note="Допустимая промежуточная величина для дозового тракта электронов ЕРПЗ.",
    ),
    SourceRadiationProductSpec(
        source=RadiationSource.ERB,
        particle=Particle.ELECTRON,
        product_kind=RadiationProductKind.MISSION_FLUENCE,
        location=RadiationProductLocation.BEFORE_SHIELDING,
        purposes=(RadiationProductPurpose.ACCUMULATED_DOSE,),
        normative_status=RadiationProductNormativeStatus.OST_MINIMUM,
        title="Флюенс электронов ЕРПЗ за САС",
    ),
)


def source_radiation_product_specs() -> tuple[SourceRadiationProductSpec, ...]:
    """Return all source radiation product specifications."""

    return SOURCE_RADIATION_PRODUCT_SPECS


def source_radiation_product_specs_for_source(
    source: RadiationSource,
) -> tuple[SourceRadiationProductSpec, ...]:
    """Return source radiation product specifications for a source."""

    return tuple(spec for spec in SOURCE_RADIATION_PRODUCT_SPECS if spec.source is source)


def source_radiation_product_specs_for_purpose(
    purpose: RadiationProductPurpose,
) -> tuple[SourceRadiationProductSpec, ...]:
    """Return source radiation product specifications for a purpose."""

    return tuple(
        spec
        for spec in SOURCE_RADIATION_PRODUCT_SPECS
        if purpose in spec.purposes
    )


def source_radiation_product_specs_for_status(
    status: RadiationProductNormativeStatus,
) -> tuple[SourceRadiationProductSpec, ...]:
    """Return source radiation product specifications for a normative status."""

    return tuple(
        spec
        for spec in SOURCE_RADIATION_PRODUCT_SPECS
        if spec.normative_status is status
    )


def source_radiation_product_specs_for_source_and_particle(
    source: RadiationSource,
    particle: Particle,
) -> tuple[SourceRadiationProductSpec, ...]:
    """Return source radiation product specifications for source and particle."""

    return tuple(
        spec
        for spec in SOURCE_RADIATION_PRODUCT_SPECS
        if spec.source is source and spec.particle is particle
    )


def has_source_radiation_product_spec(
    *,
    source: RadiationSource,
    particle: Particle,
    product_kind: RadiationProductKind,
    purpose: RadiationProductPurpose | None = None,
) -> bool:
    """Return True if a matching source radiation product specification exists."""

    for spec in SOURCE_RADIATION_PRODUCT_SPECS:
        if (
            spec.source is source
            and spec.particle is particle
            and spec.product_kind is product_kind
            and (purpose is None or purpose in spec.purposes)
        ):
            return True

    return False
