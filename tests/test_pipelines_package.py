from radar.pipelines import (
    SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
    SEP_LET_COMPONENT,
    SEP_MODEL_COMPONENT,
    SEP_OUTPUT_TABLES_COMPONENT,
    SEP_PIPELINE_COMPONENT,
    SEP_PROTON_PIPELINE_COMPONENT,
    SEP_SHIELDING_COMPONENT,
    SepPipelineResult,
    SepProtonPipelineResult,
    calculate_sep_pipeline,
    calculate_sep_proton_pipeline,
)


def test_pipelines_package_exports_full_sep_pipeline() -> None:
    assert SEP_PIPELINE_COMPONENT == "sep_pipeline"
    assert SEP_MODEL_COMPONENT == "sep_model"
    assert SEP_GEOMAGNETIC_PENETRATION_COMPONENT == "sep_geomagnetic_penetration"
    assert SEP_SHIELDING_COMPONENT == "sep_shielding"
    assert SEP_LET_COMPONENT == "sep_let"
    assert SEP_OUTPUT_TABLES_COMPONENT == "sep_output_tables"
    assert SepPipelineResult.__name__ == "SepPipelineResult"
    assert callable(calculate_sep_pipeline)


def test_pipelines_package_keeps_legacy_sep_proton_pipeline_export() -> None:
    assert SEP_PROTON_PIPELINE_COMPONENT == "sep_proton_pipeline"
    assert SepProtonPipelineResult.__name__ == "SepProtonPipelineResult"
    assert callable(calculate_sep_proton_pipeline)
