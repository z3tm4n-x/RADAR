"""Single event effect result structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class SingleEventEffectMechanism(StrEnum):
    """Single event effect contribution mechanism."""

    SEP_PROTON = "sep_proton"
    GCR_PROTON = "gcr_proton"
    ERB_PROTON = "erb_proton"
    SEP_HEAVY_ION = "sep_heavy_ion"
    GCR_HEAVY_ION = "gcr_heavy_ion"


_MECHANISM_TITLES: dict[SingleEventEffectMechanism, str] = {
    SingleEventEffectMechanism.SEP_PROTON: "протоны СКЛ",
    SingleEventEffectMechanism.GCR_PROTON: "протоны ГКЛ",
    SingleEventEffectMechanism.ERB_PROTON: "протоны ЕРПЗ",
    SingleEventEffectMechanism.SEP_HEAVY_ION: "ТЗЧ СКЛ",
    SingleEventEffectMechanism.GCR_HEAVY_ION: "ТЗЧ ГКЛ",
}


def single_event_effect_mechanism_title(
    mechanism: SingleEventEffectMechanism,
) -> str:
    """Return Russian title for a single event effect mechanism."""

    return _MECHANISM_TITLES[mechanism]


def _validate_non_negative_finite(value: float, name: str) -> None:
    """Validate finite non-negative value."""

    if not isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)

    if value < 0.0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)


@dataclass(frozen=True)
class SingleEventEffectContribution:
    """Contribution of one mechanism to single event effects."""

    mechanism: SingleEventEffectMechanism
    event_rate_per_day: float
    expected_events: float

    def __post_init__(self) -> None:
        _validate_non_negative_finite(
            self.event_rate_per_day,
            "Single event effect rate",
        )
        _validate_non_negative_finite(
            self.expected_events,
            "Single event effect expected events",
        )


@dataclass(frozen=True)
class SingleEventEffectPoint:
    """Single event effect result at one shielding thickness."""

    thickness_g_cm2: float
    contributions: tuple[SingleEventEffectContribution, ...]

    def __post_init__(self) -> None:
        _validate_non_negative_finite(
            self.thickness_g_cm2,
            "Shielding thickness",
        )

        if not self.contributions:
            msg = "Single event effect point must contain at least one contribution."
            raise ValueError(msg)

        mechanisms = tuple(contribution.mechanism for contribution in self.contributions)
        if len(set(mechanisms)) != len(mechanisms):
            msg = "Single event effect mechanisms must be unique within one point."
            raise ValueError(msg)

    @property
    def total_event_rate_per_day(self) -> float:
        """Return total event rate at this shielding thickness."""

        return sum(
            contribution.event_rate_per_day
            for contribution in self.contributions
        )

    @property
    def total_expected_events(self) -> float:
        """Return total expected events at this shielding thickness."""

        return sum(contribution.expected_events for contribution in self.contributions)

    def contribution_for(
        self,
        mechanism: SingleEventEffectMechanism,
    ) -> SingleEventEffectContribution | None:
        """Return contribution for a mechanism, if present."""

        for contribution in self.contributions:
            if contribution.mechanism is mechanism:
                return contribution

        return None
