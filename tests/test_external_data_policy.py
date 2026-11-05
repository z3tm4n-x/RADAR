"""Publication gating never converts bad evidence or synthetic tests into skips."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import conftest as policy


@pytest.mark.parametrize("module,function,fixtures", [
    ("test_nist_astar.py", "test_row_rejects_invalid_numbers", set()),
    ("test_nist_astar.py", "test_no_filesystem_extraction_or_unsafe_archive_members", set()),
    ("test_nist_astar_live.py", "test_printing_half_unit_is_derived_from_token_precision", set()),
    ("test_astar_helium_controls.py", "test_exact_total_to_per_mass_mapping_no_stopping_multiplier", {"generator"}),
    ("test_astar_helium_comparison.py", "test_reference_total_and_csda_not_projected", {"runner"}),
    ("test_catima_candidate_generator.py", "test_changed_source_is_rejected", {"generator"}),
    ("test_iaea_stopping_controls.py", "test_bad_archives_fail_before_native_loading", {"generator"}),
    ("test_srim_source_catalog.py", "test_noncontrol_values_rejected_before_formatting", set()),
    ("test_srim_source_extension.py", "test_noncontrol_cannot_format_native_input", set()),
    ("test_gsi_atima_anchors.py", "test_collect", {"native_report"}),
])
def test_synthetic_items_stay_enabled(module: str, function: str, fixtures: set[str]) -> None:
    assert policy.required_artifacts(module, function, fixtures) == ()


@pytest.mark.parametrize("module,fixture,expected", [
    ("test_nist_astar.py", "members", policy.ARCHIVE),
    ("test_nist_astar_live.py", "aluminum_html", policy.STAR + "live_astar_013.html"),
    ("test_astar_helium_controls.py", "control_context", policy.HEADER),
    ("test_astar_helium_comparison.py", "comparison_context", policy.STAR + "download.html"),
    ("test_srim_source_catalog.py", "copied", policy.MANIFEST),
    ("test_srim_source_extension.py", "baseline", policy.MANIFEST),
    ("test_srim_source_campaign.py", "catalog", policy.MANIFEST),
    ("test_srim_extension_campaign.py", "catalog", policy.MANIFEST),
    ("test_srim_source_summary.py", "controls", policy.MANIFEST),
    ("test_srim_upper_full_comparison.py", "roster", policy.MANIFEST),
])
def test_fixture_closure_marks_only_actual_dependency(module: str, fixture: str, expected: str) -> None:
    assert expected in policy.required_artifacts(module, "test_any", {fixture})
    assert policy.required_artifacts("unrelated.py", "test_any", {fixture}) == ()


@pytest.mark.parametrize("material", [13, 14])
def test_direct_parameterized_html_uses_only_selected_material(material: int) -> None:
    paths = policy.required_artifacts(
        "test_nist_astar_live.py",
        "test_retained_official_responses_preserve_whole_particle_axis_and_tokens", set(),
        {"material": material})
    assert paths == (policy.STAR + f"live_astar_{material:03d}.html",)
    paths = policy.required_artifacts(
        "test_astar_helium_comparison.py",
        "test_live_and_archive_are_separate_and_missing_22_5_is_explicit", set(),
        {"material_id": material})
    assert set(paths) == {policy.ARCHIVE, policy.STAR + f"live_astar_{material:03d}.html"}


def test_existing_corrupt_bytes_are_not_a_reason_to_skip(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    assert policy.missing_artifacts(("archive.zip",), tmp_path) == ("archive.zip",)
    path.write_bytes(b"wrong bytes -- original integrity test must fail")
    assert policy.missing_artifacts(("archive.zip",), tmp_path) == ()
    assert path.read_bytes() == b"wrong bytes -- original integrity test must fail"


def _item(module: str, function: str, fixtures: tuple[str, ...] = ()) -> tuple[pytest.Item, list[Any]]:
    marks: list[Any] = []
    value = SimpleNamespace(path=Path(module), originalname=function,
                            name=function + "[parameter]", fixturenames=fixtures,
                            add_marker=marks.append)
    return cast(pytest.Item, value), marks


def test_collection_marker_does_not_skip_whole_module() -> None:
    external, marked = _item("test_nist_astar.py", "test_invalid_lookup")
    synthetic, untouched = _item("test_nist_astar.py", "test_row_rejects_invalid_numbers")
    policy.pytest_collection_modifyitems([external, synthetic])
    assert [mark.name for mark in marked] == ["external_data"]
    assert untouched == []


def test_missing_skip_names_exact_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    item, _ = _item("test_nist_astar.py", "test_invalid_lookup")
    monkeypatch.setattr(policy, "missing_artifacts", lambda paths: paths)
    with pytest.raises(pytest.skip.Exception, match="research/nist_star_20261031/STAR.tar.gz"):
        policy.pytest_runtest_setup(item)


def test_present_artifact_setup_does_not_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    item, _ = _item("test_nist_astar.py", "test_invalid_lookup")
    monkeypatch.setattr(policy, "missing_artifacts", lambda paths: ())
    policy.pytest_runtest_setup(item)


def test_fixture_requirements_are_deduplicated() -> None:
    actual = policy.required_artifacts("test_srim_source_extension.py", "test_any",
                                       {"catalog", "baseline", "copied"})
    assert actual == (policy.MANIFEST,)
