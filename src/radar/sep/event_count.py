"""SEP expected solar proton event count policies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class WolfNumberVersion(StrEnum):
    """Wolf number scale used by a SEP event count policy."""

    W1_0 = "W1.0"
    W2_0 = "W2.0"
    W1_0_IMPLICIT = "W1.0_implicit"


class SepEventCountPolicy(StrEnum):
    """Supported SEP expected event count policies."""

    OST_134_1044_2007 = "ost_134_1044_2007"
    GOST_R_25645_165_2025_W1_0 = "gost_r_25645_165_2025_w1_0"
    GOST_R_25645_165_2025_W2_0 = "gost_r_25645_165_2025_w2_0"


@dataclass(frozen=True)
class SepEventCountResult:
    """Result of an expected SEP event count calculation."""

    expected_events: float
    wolf_number_sum: float
    months: int
    coefficient: float
    wolf_number_version: WolfNumberVersion
    policy: SepEventCountPolicy
    document: str


@dataclass(frozen=True)
class _SepEventCountPolicyConfig:
    coefficient: float
    wolf_number_version: WolfNumberVersion
    document: str


_POLICY_CONFIGS: dict[SepEventCountPolicy, _SepEventCountPolicyConfig] = {
    SepEventCountPolicy.OST_134_1044_2007: _SepEventCountPolicyConfig(
        coefficient=0.0135,
        wolf_number_version=WolfNumberVersion.W1_0_IMPLICIT,
        document="ОСТ 134-1044-2007, приложение Б",
    ),
    SepEventCountPolicy.GOST_R_25645_165_2025_W1_0: _SepEventCountPolicyConfig(
        coefficient=0.0130,
        wolf_number_version=WolfNumberVersion.W1_0,
        document="ГОСТ Р 25645.165-2025, проект, окончательная редакция",
    ),
    SepEventCountPolicy.GOST_R_25645_165_2025_W2_0: _SepEventCountPolicyConfig(
        coefficient=0.00925,
        wolf_number_version=WolfNumberVersion.W2_0,
        document="ГОСТ Р 25645.165-2025, проект, окончательная редакция",
    ),
}


def _policy_config(policy: SepEventCountPolicy) -> _SepEventCountPolicyConfig:
    try:
        return _POLICY_CONFIGS[policy]
    except KeyError as exc:
        msg = f"Unsupported SEP event count policy: {policy!r}."
        raise ValueError(msg) from exc


def _normalized_wolf_numbers(
    monthly_smoothed_wolf_numbers: Sequence[float],
) -> tuple[float, ...]:
    if not monthly_smoothed_wolf_numbers:
        msg = "Monthly smoothed Wolf number sequence must not be empty."
        raise ValueError(msg)

    values = tuple(float(value) for value in monthly_smoothed_wolf_numbers)

    for value in values:
        if not math.isfinite(value):
            msg = "Monthly smoothed Wolf numbers must be finite."
            raise ValueError(msg)

        if value < 0.0:
            msg = "Monthly smoothed Wolf numbers must be non-negative."
            raise ValueError(msg)

    return values


def calculate_sep_expected_events(
    monthly_smoothed_wolf_numbers: Sequence[float],
    policy: SepEventCountPolicy,
) -> SepEventCountResult:
    """Calculate expected solar proton event count for a Wolf number series."""

    config = _policy_config(policy)

    if config.coefficient <= 0.0:
        msg = "SEP event count coefficient must be positive."
        raise ValueError(msg)

    values = _normalized_wolf_numbers(monthly_smoothed_wolf_numbers)
    wolf_number_sum = float(sum(values))
    expected_events = float(config.coefficient * wolf_number_sum)

    return SepEventCountResult(
        expected_events=expected_events,
        wolf_number_sum=wolf_number_sum,
        months=len(values),
        coefficient=config.coefficient,
        wolf_number_version=config.wolf_number_version,
        policy=policy,
        document=config.document,
    )
