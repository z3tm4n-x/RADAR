import math

import pytest

from radar.sep.event_count import (
    SepEventCountPolicy,
    WolfNumberVersion,
    calculate_sep_expected_events,
)


def test_ost_sep_event_count_uses_implicit_w1_scale() -> None:
    result = calculate_sep_expected_events(
        monthly_smoothed_wolf_numbers=(100.0, 100.0),
        policy=SepEventCountPolicy.OST_134_1044_2007,
    )

    assert result.expected_events == pytest.approx(2.7)
    assert result.wolf_number_sum == pytest.approx(200.0)
    assert result.months == 2
    assert result.coefficient == pytest.approx(0.0135)
    assert result.wolf_number_version is WolfNumberVersion.W1_0_IMPLICIT
    assert result.policy is SepEventCountPolicy.OST_134_1044_2007
    assert "ОСТ 134-1044-2007" in result.document


def test_gost_2025_sep_event_count_uses_w1_scale() -> None:
    result = calculate_sep_expected_events(
        monthly_smoothed_wolf_numbers=(100.0, 100.0),
        policy=SepEventCountPolicy.GOST_R_25645_165_2025_W1_0,
    )

    assert result.expected_events == pytest.approx(2.6)
    assert result.wolf_number_sum == pytest.approx(200.0)
    assert result.months == 2
    assert result.coefficient == pytest.approx(0.0130)
    assert result.wolf_number_version is WolfNumberVersion.W1_0
    assert result.policy is SepEventCountPolicy.GOST_R_25645_165_2025_W1_0
    assert "ГОСТ Р 25645.165-2025" in result.document


def test_gost_2025_sep_event_count_uses_w2_scale() -> None:
    result = calculate_sep_expected_events(
        monthly_smoothed_wolf_numbers=(140.0, 140.0),
        policy=SepEventCountPolicy.GOST_R_25645_165_2025_W2_0,
    )

    assert result.expected_events == pytest.approx(2.59)
    assert result.wolf_number_sum == pytest.approx(280.0)
    assert result.months == 2
    assert result.coefficient == pytest.approx(0.00925)
    assert result.wolf_number_version is WolfNumberVersion.W2_0
    assert result.policy is SepEventCountPolicy.GOST_R_25645_165_2025_W2_0


def test_ost_event_count_is_slightly_higher_than_gost_2025_w1_for_same_wolf_series() -> None:
    wolf_numbers = (50.0, 75.0, 125.0)

    ost = calculate_sep_expected_events(
        monthly_smoothed_wolf_numbers=wolf_numbers,
        policy=SepEventCountPolicy.OST_134_1044_2007,
    )
    gost = calculate_sep_expected_events(
        monthly_smoothed_wolf_numbers=wolf_numbers,
        policy=SepEventCountPolicy.GOST_R_25645_165_2025_W1_0,
    )

    assert ost.expected_events / gost.expected_events == pytest.approx(0.0135 / 0.0130)


def test_sep_event_count_accepts_zero_wolf_numbers() -> None:
    result = calculate_sep_expected_events(
        monthly_smoothed_wolf_numbers=(0.0, 0.0),
        policy=SepEventCountPolicy.OST_134_1044_2007,
    )

    assert result.expected_events == pytest.approx(0.0)
    assert result.wolf_number_sum == pytest.approx(0.0)
    assert result.months == 2


def test_sep_event_count_rejects_empty_wolf_number_series() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_sep_expected_events(
            monthly_smoothed_wolf_numbers=(),
            policy=SepEventCountPolicy.OST_134_1044_2007,
        )


def test_sep_event_count_rejects_negative_wolf_number() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        calculate_sep_expected_events(
            monthly_smoothed_wolf_numbers=(100.0, -1.0),
            policy=SepEventCountPolicy.OST_134_1044_2007,
        )


@pytest.mark.parametrize("bad_value", (math.nan, math.inf, -math.inf))
def test_sep_event_count_rejects_non_finite_wolf_number(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate_sep_expected_events(
            monthly_smoothed_wolf_numbers=(100.0, bad_value),
            policy=SepEventCountPolicy.OST_134_1044_2007,
        )


def test_sep_event_count_preserves_fractional_wolf_number_sum() -> None:
    result = calculate_sep_expected_events(
        monthly_smoothed_wolf_numbers=(1.5, 2.25, 3.25),
        policy=SepEventCountPolicy.GOST_R_25645_165_2025_W1_0,
    )

    assert result.wolf_number_sum == pytest.approx(7.0)
    assert result.expected_events == pytest.approx(0.0130 * 7.0)
    assert result.months == 3
