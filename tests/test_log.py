import pytest

from radar.core.log import CalculationLog, CalculationLogEntry, LogLevel


def test_log_entry_requires_stage() -> None:
    with pytest.raises(ValueError, match="stage"):
        CalculationLogEntry(
            level=LogLevel.INFO,
            stage="",
            message="message",
        )


def test_log_entry_requires_message() -> None:
    with pytest.raises(ValueError, match="message"):
        CalculationLogEntry(
            level=LogLevel.INFO,
            stage="stage",
            message="",
        )


def test_log_adds_entries_immutably() -> None:
    original_log = CalculationLog()

    updated_log = original_log.info(
        stage="solar_activity",
        message="Solar activity sequence created.",
        details={"lifetime_years": "15", "level": "mean"},
    )

    assert original_log.entries == ()
    assert len(updated_log.entries) == 1

    entry = updated_log.entries[0]

    assert entry.level is LogLevel.INFO
    assert entry.stage == "solar_activity"
    assert entry.message == "Solar activity sequence created."
    assert entry.details == (("level", "mean"), ("lifetime_years", "15"))


def test_log_filters_warnings_and_errors() -> None:
    log = CalculationLog()
    log = log.info(stage="input", message="Input checked.")
    log = log.warning(stage="validation", message="Reference differs.")
    log = log.error(stage="export", message="Export failed.")

    assert len(log.entries) == 3
    assert len(log.warnings()) == 1
    assert len(log.errors()) == 1
    assert log.has_errors() is True
