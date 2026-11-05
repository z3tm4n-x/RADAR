"""Keep external-evidence checks explicit without weakening integrity failures.

Public checkout excludes research downloads, native runtimes and local outputs.
Only tests that actually read those artifacts (directly or through a fixture)
receive ``external_data``. Missing artifacts skip with their paths; present
artifacts still run the original hash/content checks and can fail. Synthetic
tests in the same modules remain enabled. Nothing is downloaded or regenerated.
Use ``pytest -m 'not external_data'`` for the explicitly portable subset, or
restore the pinned local evidence and run the full suite, reporting skips.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAR = "research/nist_star_20261031/"
CATIMA = "research/ion_stopping_20261031/"
ARCHIVE = STAR + "STAR.tar.gz"
HEADER = CATIMA + "runtime/include/catima/data_ionisation_potential.h"
MANIFEST = "outputs/stopping_mass_manifest_20261031.json"
RADEF = "research/iaea_stopping_20261031/javanainen_esa_2009.pdf"
ASTAR_EVIDENCE = tuple(STAR + name for name in (
    "download.html", "programs.html", "readme.html", "composition_al.html",
    "composition_si.html", "live_astar_013.html", "live_astar_014.html",
))
CATIMA_RUNTIME = tuple(CATIMA + name for name in (
    "pycatima-1.982-cp312-cp312-win_amd64.whl", "pycatima-1.982.tar.gz",
    "catima-v1.91.zip", "source/catima-1.91/src/calculations.cpp",
    "source/catima-1.91/src/catima.cpp", "source/catima-1.91/include/catima/config.h",
    "runtime",
))

FIXTURE_ARTIFACTS: dict[str, dict[str, tuple[str, ...]]] = {
    "test_nist_astar.py": {"members": (ARCHIVE,)},
    "test_nist_astar_live.py": {"aluminum_html": (STAR + "live_astar_013.html",)},
    "test_astar_helium_controls.py": {"control_context": (ARCHIVE, HEADER)},
    "test_astar_helium_comparison.py": {
        "controls": (ARCHIVE, HEADER), "comparison_context": ASTAR_EVIDENCE,
    },
    "test_ion_stopping_comparison.py": {"candidate": tuple(CATIMA + name for name in (
        "gsi_fe56_al_100.html", "gsi_fe56_si_1000.html", "gsi_he4_si_100.html"))},
    "test_srim_source_catalog.py": {"catalog": (MANIFEST,), "copied": (MANIFEST,)},
    "test_srim_source_extension.py": {
        "baseline": (MANIFEST,), "catalog": (MANIFEST,), "copied": (MANIFEST,),
    },
    "test_srim_source_campaign.py": {"catalog": (MANIFEST,)},
    "test_srim_extension_campaign.py": {"catalog": (MANIFEST,)},
    "test_srim_source_summary.py": {"controls": (MANIFEST,)},
    "test_srim_upper_full_comparison.py": {"roster": (MANIFEST,)},
}

TEST_ARTIFACTS: dict[str, dict[str, tuple[str, ...]]] = {
    "test_nist_astar.py": {name: (ARCHIVE,) for name in (
        "test_pinned_archive_complete_native_grid_and_nonrelease_flags",
        "test_invalid_lookup", "test_artifact_is_immutable",
        "test_public_loader_refuses_unpinned_bytes", "test_changed_archive_during_load_is_rejected",
    )},
    "test_astar_helium_controls.py": {
        "test_wheel_header_values_are_read_without_setting_i": (HEADER,),
        "test_bad_archive_rejected_before_native_load": (HEADER,),
    },
    "test_astar_helium_comparison.py": {
        "test_exact_decimal_rounding_tie_is_not_an_online_discrepancy": (
            ARCHIVE, STAR + "live_astar_013.html"),
    },
    "test_catima_candidate_generator.py": {
        "test_retained_source_and_runtime_hashes": CATIMA_RUNTIME,
    },
    "test_iaea_stopping_controls.py": {
        "test_native_elemental_i_header_is_unchanged": (HEADER,),
        "test_generator_records_full_selection_and_all_qualification_boundaries": (HEADER,),
        "test_changed_pins_or_reference_inventory_rejected": (HEADER,),
        "test_radef_document_identity_is_verified": (RADEF,),
        "test_bad_archives_fail_before_native_loading": (HEADER, RADEF),
        "test_real_pinned_inventory_request_mapping": (
            "research/iaea_stopping_20261031/data.zip",
            "research/iaea_stopping_20261031/references.zip", RADEF),
    },
}


def required_artifacts(module: str, function: str, fixtures: set[str],
                       params: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return explicit external dependencies only, not broad module-level skips."""
    result = set(TEST_ARTIFACTS.get(module, {}).get(function, ()))
    for fixture, paths in FIXTURE_ARTIFACTS.get(module, {}).items():
        if fixture in fixtures:
            result.update(paths)
    values = params or {}
    if (module == "test_nist_astar_live.py"
            and function == "test_retained_official_responses_preserve_whole_particle_axis_and_tokens"):
        result.add(STAR + f"live_astar_{values['material']:03d}.html")
    if (module == "test_astar_helium_comparison.py"
            and function == "test_live_and_archive_are_separate_and_missing_22_5_is_explicit"):
        result.update((ARCHIVE, STAR + f"live_astar_{values['material_id']:03d}.html"))
    return tuple(sorted(result))


def _item_artifacts(item: pytest.Item) -> tuple[str, ...]:
    callspec = getattr(item, "callspec", None)
    return required_artifacts(
        item.path.name, getattr(item, "originalname", None) or item.name.split("[", 1)[0],
        set(getattr(item, "fixturenames", ())), {} if callspec is None else callspec.params,
    )


def missing_artifacts(paths: tuple[str, ...], root: Path = ROOT) -> tuple[str, ...]:
    # Do not replace integrity validation with a hash-based skip: corrupted,
    # altered or partially installed *present* artifacts must still fail tests.
    return tuple(path for path in paths if not (root / path).exists())


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "external_data: requires retained local scientific evidence")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if _item_artifacts(item):
            item.add_marker(pytest.mark.external_data)


def pytest_runtest_setup(item: pytest.Item) -> None:
    missing = missing_artifacts(_item_artifacts(item))
    if missing:
        pytest.skip("External scientific evidence not included in checkout: " + ", ".join(missing))
