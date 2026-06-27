"""Helpers for executing configured calculation pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from radar.core.constants import (
    DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON,
    DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV,
)
from radar.core.project import CalculationConfig
from radar.erb.model import ErbModelProtocol
from radar.gcr.model import GcrModelProtocol
from radar.model_registry import source_model_instances_for_pipeline_config
from radar.pipelines.erb import ErbPipelineResult, calculate_erb_pipeline
from radar.pipelines.gcr import GcrPipelineResult, calculate_gcr_pipeline
from radar.pipelines.sep import SepPipelineResult, calculate_sep_pipeline
from radar.sep.model import SepModelProtocol


@dataclass(frozen=True)
class PipelineSourceModels:
    """Source model instances configured for calculation pipeline execution."""

    sep_model: SepModelProtocol
    gcr_model: GcrModelProtocol
    erb_model: ErbModelProtocol


def build_pipeline_source_models(
    config: CalculationConfig,
    *,
    sep_energy_grid_mev: tuple[float, ...] | None = (
        DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV
    ),
    sep_hze_energy_grid_mev_per_nucleon: tuple[float, ...] | None = (
        DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON
    ),
) -> PipelineSourceModels:
    """Build source model instances for configured pipeline execution."""

    sep_model, gcr_model, erb_model = source_model_instances_for_pipeline_config(
        config,
        sep_energy_grid_mev=sep_energy_grid_mev,
        sep_hze_energy_grid_mev_per_nucleon=sep_hze_energy_grid_mev_per_nucleon,
    )

    return PipelineSourceModels(
        sep_model=sep_model,
        gcr_model=gcr_model,
        erb_model=erb_model,
    )


def calculate_sep_pipeline_for_config(
    config: CalculationConfig,
    *,
    sep_energy_grid_mev: tuple[float, ...] | None = (
        DEFAULT_SEP_PROTON_PIPELINE_ENERGY_GRID_MEV
    ),
    sep_hze_energy_grid_mev_per_nucleon: tuple[float, ...] | None = (
        DEFAULT_SEP_HZE_PIPELINE_ENERGY_GRID_MEV_PER_NUCLEON
    ),
) -> SepPipelineResult:
    """Calculate the configured SEP pipeline."""

    models = build_pipeline_source_models(
        config,
        sep_energy_grid_mev=sep_energy_grid_mev,
        sep_hze_energy_grid_mev_per_nucleon=sep_hze_energy_grid_mev_per_nucleon,
    )

    return calculate_sep_pipeline(
        config=config,
        sep_model=models.sep_model,
    )


def calculate_gcr_pipeline_for_config(config: CalculationConfig) -> GcrPipelineResult:
    """Calculate the configured GCR pipeline."""

    models = build_pipeline_source_models(config)
    return calculate_gcr_pipeline(
        config=config,
        gcr_model=models.gcr_model,
    )


def calculate_erb_pipeline_for_config(config: CalculationConfig) -> ErbPipelineResult:
    """Calculate the configured ERB pipeline."""

    models = build_pipeline_source_models(config)
    return calculate_erb_pipeline(
        config=config,
        erb_model=models.erb_model,
    )


__all__ = [
    "PipelineSourceModels",
    "build_pipeline_source_models",
    "calculate_erb_pipeline_for_config",
    "calculate_gcr_pipeline_for_config",
    "calculate_sep_pipeline_for_config",
]
