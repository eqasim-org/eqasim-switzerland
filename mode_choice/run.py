"""Read tour batches, select/calibrate modes, and merge choices into the population."""
import logging
from pathlib import Path
import polars as pl
from mode_choice.defaults import RUN_DEFAULTS, CALIBRATION_DEFAULTS, SOURCE_TRIP_ID
from mode_choice.calibration import population as calibration
from mode_choice.model.parameters import load
from mode_choice.selection.mode_choice import ModeChoice
from mode_choice.selection.sampler import TourSampler
from mode_choice.selection.cache import TopologyCache
from mode_choice.routing.RoutingServer import RoutingServer

logger = logging.getLogger(__name__)


def configure(context):
    context.stage("mode_choice.defaults")
    context.stage("mode_choice.data.trips")
    context.stage("mode_choice.tours.batches")
    context.stage("dmc.params")
    context.config("random_seed")
    for name, default in {**RUN_DEFAULTS, **CALIBRATION_DEFAULTS}.items():
        context.config(name, default)
    if context.config("mode_choice_calibrate_alphas"):
        calibration.calibration_level(context)
        context.stage("data.microcensus.shares")


def apply_choices(original, choices):
    """Preserve every source trip and its original survey ID; replace modes by key."""
    if choices["trip_id"].n_unique() != choices.height:
        raise ValueError("Duplicate mode-choice trip IDs")
    if choices.select("trip_id").join(original.select("trip_id"), on="trip_id", how="anti").height:
        raise ValueError("Choices contain unknown trip IDs")
    result = original.join(choices.rename({"mode": "chosen_mode"}), on="trip_id", how="left",
                           validate="1:1", maintain_order="left").with_columns(
        pl.coalesce("chosen_mode", "mode").alias("mode")).drop("chosen_mode", "trip_id")
    if SOURCE_TRIP_ID in result.columns:
        result = result.rename({SOURCE_TRIP_ID: "trip_id"})
    return result


def execute(context):
    try:
        return _execute(context)
    finally:
        RoutingServer.stop_server()


def _execute(context):
    original = context.stage("mode_choice.data.trips")
    paths = context.stage("mode_choice.tours.batches")
    if not paths and context.config("mode_choice_calibrate_alphas"):
        raise ValueError("Cannot calibrate alphas: no eligible population trips")
    sampler = TourSampler(
        method=context.config("mode_choice_sampling_method"),
        random_cache_max_values=context.config("mode_choice_random_cache_max_values"),
        max_trips=context.config("mode_choice_max_trips_per_tour"),
        cache=TopologyCache(context.config("mode_choice_cache_max_entries"), context.config("mode_choice_cache_max_edges")),
    )
    with ModeChoice(paths, context.config("random_seed"), sampler,
                    cache_prepared=(context.config("mode_choice_calibrate_alphas")
                                    and context.config("mode_choice_cache_prepared_batches"))) as model:
        choices = (calibration.run(context, model, load(context)) if context.config("mode_choice_calibrate_alphas")
                   else model.run())
    choices.write_parquet(Path(context.path()) / "choices.parquet")
    logger.info("Assigned modes to %d trips; retained %d source modes. Topology cache: %s",
                choices.height, original.height - choices.height, sampler.cache.info())
    return apply_choices(original, choices)
