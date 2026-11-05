from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import radar.data.integrity as integrity_module
import radar.data_tables as data_tables_module
import radar.erb.ost_tables as ost_tables_module
import radar.erb.resources as erb_resources_module
import radar.pipelines.sep_source_products as sep_source_products_module
import radar.sep.model as sep_model_module
import radar.shielding.resources as shielding_resources_module
from radar.sep.hze_spectrum import OST_SEP_HZE_B9_RESOURCE


def test_data_table_loader_validates_index_manifest_and_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated_paths: list[str] = []

    def record_validation(path: str) -> None:
        validated_paths.append(path)

    monkeypatch.setattr(
        data_tables_module,
        "validate_normative_resource_integrity",
        record_validation,
    )

    rows = data_tables_module.load_normative_table_rows(
        "ost_134_1044_2007",
        "ost_134_1044_2007_table_g_1_wolf_numbers",
    )

    assert rows
    assert validated_paths == [
        "index.json",
        "ost_134_1044_2007/manifest.json",
        "ost_134_1044_2007/table_g_1_wolf_numbers.csv",
    ]


def test_erb_resource_loaders_validate_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated_paths: list[str] = []

    def record_validation(path: str) -> None:
        validated_paths.append(path)

    monkeypatch.setattr(
        erb_resources_module,
        "validate_normative_resource_integrity",
        record_validation,
    )
    erb_resources_module.load_igrf14_coefficients.cache_clear()
    erb_resources_module.load_radbelt_asc_map.cache_clear()

    try:
        erb_resources_module.load_igrf14_coefficients()
        erb_resources_module.load_radbelt_asc_map(
            name="AE8MAX",
            resource_name=erb_resources_module.ERB_RADBELT_AE8MAX_RESOURCE_NAME,
        )
    finally:
        erb_resources_module.load_igrf14_coefficients.cache_clear()
        erb_resources_module.load_radbelt_asc_map.cache_clear()

    assert validated_paths == [
        erb_resources_module.ERB_IGRF14_RESOURCE_NAME,
        erb_resources_module.ERB_RADBELT_AE8MAX_RESOURCE_NAME,
    ]


def test_ost_appendix_a_loader_validates_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated_paths: list[str] = []

    def record_validation(path: str) -> None:
        validated_paths.append(path)

    monkeypatch.setattr(
        ost_tables_module,
        "validate_normative_resource_integrity",
        record_validation,
    )
    ost_tables_module.load_ost_erb_appendix_a_tables.cache_clear()

    try:
        tables = ost_tables_module.load_ost_erb_appendix_a_tables()
    finally:
        ost_tables_module.load_ost_erb_appendix_a_tables.cache_clear()

    assert tables.tables
    assert validated_paths == [ost_tables_module.OST_ERB_RESOURCE_NAME]


def test_shielding_loaders_validate_single_and_multi_file_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated_paths: list[str] = []
    single_table = object()
    kernel = object()

    def record_validation(path: str) -> None:
        validated_paths.append(path)

    monkeypatch.setattr(
        shielding_resources_module,
        "validate_normative_resource_integrity",
        record_validation,
    )
    monkeypatch.setattr(
        shielding_resources_module,
        "load_proton_al_range_energy_table",
        lambda _path: single_table,
    )
    monkeypatch.setattr(
        shielding_resources_module,
        "load_secondary_proton_kernel",
        lambda **_paths: kernel,
    )

    assert shielding_resources_module.load_normative_proton_al_range_table() is single_table
    assert shielding_resources_module.load_normative_secondary_proton_kernel() is kernel
    assert validated_paths == [
        shielding_resources_module.PROTON_AL_RANGE_RESOURCE,
        shielding_resources_module.P_AL27_MF3_XS_RESOURCE,
        shielding_resources_module.P_AL27_MF6_PROTON_YIELD_RESOURCE,
        shielding_resources_module.P_AL27_MF6_PROTON_PDF_RESOURCE,
    ]


def test_sep_penetration_hze_loader_validates_before_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated_paths: list[str] = []
    ion_records = (object(),)

    def record_validation(path: str) -> None:
        validated_paths.append(path)

    monkeypatch.setattr(
        sep_source_products_module,
        "validate_normative_resource_integrity",
        record_validation,
    )
    monkeypatch.setattr(
        sep_source_products_module,
        "load_ost_sep_hze_ion_records",
        lambda _path: ion_records,
    )

    assert (
        sep_source_products_module._load_ost_sep_hze_ion_records_for_penetration()
        is ion_records
    )
    assert validated_paths == [OST_SEP_HZE_B9_RESOURCE]


def test_sep_source_loaders_validate_before_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Start from the warm-cache condition seen in a whole-suite run.
    sep_model_module._load_ost_sep_hze_ion_records()
    validated_paths: list[str] = []
    proton_records = (object(),)
    ion_records = (object(),)

    monkeypatch.setattr(
        sep_model_module,
        "validate_normative_resource_integrity",
        validated_paths.append,
    )
    monkeypatch.setattr(
        sep_model_module,
        "load_sep_proton_coefficient_records",
        lambda _path: proton_records,
    )
    monkeypatch.setattr(
        sep_model_module,
        "load_ost_sep_hze_ion_records",
        lambda _path: ion_records,
    )

    # Exercise the parser even if an earlier model test populated the cache,
    # and never leave the mocked records for a later real pipeline calculation.
    sep_model_module._load_ost_sep_hze_ion_records.cache_clear()
    try:
        assert (
            sep_model_module._load_sep_proton_coefficient_records(
                sep_model_module.OST_SEP_PROTON_COEFFICIENT_RESOURCE,
            )
            is proton_records
        )
        assert sep_model_module._load_ost_sep_hze_ion_records() is ion_records
    finally:
        sep_model_module._load_ost_sep_hze_ion_records.cache_clear()
    assert validated_paths == [
        sep_model_module.OST_SEP_PROTON_COEFFICIENT_RESOURCE,
        OST_SEP_HZE_B9_RESOURCE,
    ]


def test_tampered_shielding_resource_is_rejected_before_application_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_path = shielding_resources_module.PROTON_AL_RANGE_RESOURCE
    expected_payload = b"pristine normative payload\n"
    tampered_payload = b"tampered normative payload\n"
    entry = integrity_module.NormativeResourceIntegrity(
        resource_id="test_proton_al_range",
        path=resource_path,
        sha256=hashlib.sha256(expected_payload).hexdigest(),
        byte_count=len(expected_payload),
        record_count=0,
        status=integrity_module.NormativeResourceStatus.PROVISIONAL,
        calculation_input=True,
        release_eligible=False,
        source_ref="isolated test fixture",
        authority_status="test",
        media_type="text/csv",
        validator="isolated test",
        verification_method="fixture digest",
        verification_note="Resource bytes intentionally differ from the registry.",
    )
    manifest = integrity_module.NormativeIntegrityManifest(
        schema_version="2.0.0",
        release_id="isolated-tamper-test",
        resources=(entry,),
    )

    def fake_joinpath(path: str) -> SimpleNamespace:
        assert path == resource_path
        return SimpleNamespace(read_bytes=lambda: tampered_payload)

    monkeypatch.setattr(
        integrity_module,
        "resources",
        SimpleNamespace(
            files=lambda _package: SimpleNamespace(joinpath=fake_joinpath),
        ),
    )
    monkeypatch.setattr(
        integrity_module,
        "load_normative_integrity_manifest",
        lambda: manifest,
    )
    parser_called = False

    def application_parser(_path: Path) -> object:
        nonlocal parser_called
        parser_called = True
        return object()

    monkeypatch.setattr(
        shielding_resources_module,
        "load_proton_al_range_energy_table",
        application_parser,
    )

    with pytest.raises(ValueError, match="integrity check failed"):
        shielding_resources_module.load_normative_proton_al_range_table()

    assert parser_called is False
