import pytest

from radar.single_event_effects import (
    SingleEventEffectContribution,
    SingleEventEffectMechanism,
    SingleEventEffectPoint,
    single_event_effect_mechanism_title,
)


def _contribution(
    mechanism: SingleEventEffectMechanism,
    *,
    event_rate_per_day: float,
    expected_events: float,
) -> SingleEventEffectContribution:
    return SingleEventEffectContribution(
        mechanism=mechanism,
        event_rate_per_day=event_rate_per_day,
        expected_events=expected_events,
    )


def test_single_event_effect_mechanism_title() -> None:
    assert (
        single_event_effect_mechanism_title(SingleEventEffectMechanism.SEP_PROTON)
        == "протоны СКЛ"
    )
    assert (
        single_event_effect_mechanism_title(SingleEventEffectMechanism.GCR_HEAVY_ION)
        == "ТЗЧ ГКЛ"
    )


def test_single_event_effect_point_sums_contributions() -> None:
    point = SingleEventEffectPoint(
        thickness_g_cm2=1.0,
        contributions=(
            _contribution(
                SingleEventEffectMechanism.SEP_PROTON,
                event_rate_per_day=0.2,
                expected_events=3.0,
            ),
            _contribution(
                SingleEventEffectMechanism.GCR_HEAVY_ION,
                event_rate_per_day=0.3,
                expected_events=4.0,
            ),
        ),
    )

    assert point.total_event_rate_per_day == pytest.approx(0.5)
    assert point.total_expected_events == pytest.approx(7.0)
    assert (
        point.contribution_for(SingleEventEffectMechanism.SEP_PROTON)
        is point.contributions[0]
    )
    assert point.contribution_for(SingleEventEffectMechanism.ERB_PROTON) is None


def test_single_event_effect_contribution_rejects_negative_rate() -> None:
    with pytest.raises(
        ValueError,
        match="Single event effect rate must be non-negative",
    ):
        _contribution(
            SingleEventEffectMechanism.SEP_PROTON,
            event_rate_per_day=-0.1,
            expected_events=0.0,
        )


def test_single_event_effect_contribution_rejects_negative_expected_events() -> None:
    with pytest.raises(
        ValueError,
        match="Single event effect expected events must be non-negative",
    ):
        _contribution(
            SingleEventEffectMechanism.SEP_PROTON,
            event_rate_per_day=0.0,
            expected_events=-1.0,
        )


def test_single_event_effect_point_rejects_negative_thickness() -> None:
    with pytest.raises(ValueError, match="Shielding thickness must be non-negative"):
        SingleEventEffectPoint(
            thickness_g_cm2=-1.0,
            contributions=(
                _contribution(
                    SingleEventEffectMechanism.SEP_PROTON,
                    event_rate_per_day=0.0,
                    expected_events=0.0,
                ),
            ),
        )


def test_single_event_effect_point_rejects_empty_contributions() -> None:
    with pytest.raises(
        ValueError,
        match="must contain at least one contribution",
    ):
        SingleEventEffectPoint(thickness_g_cm2=1.0, contributions=())


def test_single_event_effect_point_rejects_duplicate_mechanisms() -> None:
    with pytest.raises(
        ValueError,
        match="mechanisms must be unique",
    ):
        SingleEventEffectPoint(
            thickness_g_cm2=1.0,
            contributions=(
                _contribution(
                    SingleEventEffectMechanism.SEP_PROTON,
                    event_rate_per_day=0.0,
                    expected_events=0.0,
                ),
                _contribution(
                    SingleEventEffectMechanism.SEP_PROTON,
                    event_rate_per_day=0.1,
                    expected_events=1.0,
                ),
            ),
        )
