"""Calculation pipelines."""

from radar.pipelines.erb import (
    ERB_MODEL_COMPONENT,
    ERB_PIPELINE_COMPONENT,
    ErbPipelineResult,
    calculate_erb_pipeline,
)
from radar.pipelines.execution import (
    PipelineSourceModels,
    build_pipeline_source_models,
    calculate_erb_pipeline_for_config,
    calculate_gcr_pipeline_for_config,
    calculate_sep_pipeline_for_config,
)
from radar.pipelines.gcr import (
    GCR_MODEL_COMPONENT,
    GCR_PIPELINE_COMPONENT,
    GcrPipelineResult,
    calculate_gcr_pipeline,
)
from radar.pipelines.sep import (
    SEP_GEOMAGNETIC_PENETRATION_COMPONENT,
    SEP_LET_COMPONENT,
    SEP_MODEL_COMPONENT,
    SEP_OUTPUT_TABLES_COMPONENT,
    SEP_PIPELINE_COMPONENT,
    SEP_SHIELDING_COMPONENT,
    SepPipelineResult,
    calculate_sep_pipeline,
)
from radar.pipelines.sep_protons import (
    GEOMAGNETIC_PENETRATION_COMPONENT,
    SEP_PROTON_PIPELINE_COMPONENT,
    SepProtonPipelineResult,
    calculate_sep_proton_pipeline,
)

__all__ = [
    "ERB_MODEL_COMPONENT",
    "ERB_PIPELINE_COMPONENT",
    "GCR_MODEL_COMPONENT",
    "GCR_PIPELINE_COMPONENT",
    "GEOMAGNETIC_PENETRATION_COMPONENT",
    "SEP_GEOMAGNETIC_PENETRATION_COMPONENT",
    "SEP_LET_COMPONENT",
    "SEP_MODEL_COMPONENT",
    "SEP_OUTPUT_TABLES_COMPONENT",
    "SEP_PIPELINE_COMPONENT",
    "SEP_PROTON_PIPELINE_COMPONENT",
    "SEP_SHIELDING_COMPONENT",
    "ErbPipelineResult",
    "GcrPipelineResult",
    "PipelineSourceModels",
    "SepPipelineResult",
    "SepProtonPipelineResult",
    "build_pipeline_source_models",
    "calculate_erb_pipeline",
    "calculate_erb_pipeline_for_config",
    "calculate_gcr_pipeline",
    "calculate_gcr_pipeline_for_config",
    "calculate_sep_pipeline",
    "calculate_sep_pipeline_for_config",
    "calculate_sep_proton_pipeline",
]
