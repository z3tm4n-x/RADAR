import pytest

from radar.core.display_units import (
    DAYS_PER_YEAR,
    SECONDS_PER_DAY,
    accumulated_rad_to_average_rad_per_day,
    accumulated_rad_to_average_rad_per_second,
    accumulated_values_to_average_daily_values,
    days_for_mission_years,
    rad_per_day_to_rad_per_second,
    rad_per_second_to_rad_per_day,
    seconds_for_mission_years,
)


def test_mission_duration_conversions_for_integer_years() -> None:
    assert days_for_mission_years(2) == pytest.approx(2.0 * DAYS_PER_YEAR)
    assert seconds_for_mission_years(2) == pytest.approx(
        2.0 * DAYS_PER_YEAR * SECONDS_PER_DAY
    )


def test_mission_duration_rejects_invalid_lifetime() -> None:
    with pytest.raises(ValueError, match="integer"):
        days_for_mission_years(2.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least one"):
        days_for_mission_years(0)


def test_rad_per_second_and_rad_per_day_conversions() -> None:
    assert rad_per_second_to_rad_per_day(1.0) == pytest.approx(SECONDS_PER_DAY)
    assert rad_per_day_to_rad_per_second(SECONDS_PER_DAY) == pytest.approx(1.0)


def test_accumulated_rad_to_average_rates() -> None:
    assert accumulated_rad_to_average_rad_per_day(
        accumulated_rad=365.0,
        duration_days=365.0,
    ) == pytest.approx(1.0)

    assert accumulated_rad_to_average_rad_per_second(
        accumulated_rad=SECONDS_PER_DAY,
        duration_seconds=SECONDS_PER_DAY,
    ) == pytest.approx(1.0)


def test_accumulated_values_to_average_daily_values() -> None:
    assert accumulated_values_to_average_daily_values(
        values=(365.0, 730.0),
        duration_days=365.0,
    ) == pytest.approx((1.0, 2.0))


def test_average_rate_conversions_reject_zero_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        accumulated_rad_to_average_rad_per_day(
            accumulated_rad=1.0,
            duration_days=0.0,
        )

    with pytest.raises(ValueError, match="positive"):
        accumulated_rad_to_average_rad_per_second(
            accumulated_rad=1.0,
            duration_seconds=0.0,
        )

    with pytest.raises(ValueError, match="positive"):
        accumulated_values_to_average_daily_values(
            values=(1.0,),
            duration_days=0.0,
        )
