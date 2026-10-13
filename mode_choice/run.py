"""Assign tour-level modes using existing dmc.params; never fit another model."""
import logging
import polars as pl
from mode_choice.defaults import RUN_DEFAULTS, CALIBRATION_DEFAULTS
from mode_choice.calibration import population as calibration
from mode_choice.model import costs, utilities
from mode_choice.model.parameters import load
from mode_choice.tours.build import choose
from mode_choice.tours.batching import person_batches
from mode_choice.selection.sampler import TourSampler
from mode_choice.selection.cache import TopologyCache

logger = logging.getLogger(__name__)


def configure(context):
    context.stage("mode_choice.defaults")
    context.stage("mode_choice.routing.java")
    context.stage("synthesis.population.trips_with_departures")
    context.stage("dmc.params")
    context.config("random_seed")
    for name, default in {**RUN_DEFAULTS, **CALIBRATION_DEFAULTS}.items():
        context.config(name, default)
    if context.config("mode_choice_calibrate_alphas"):
        calibration.calibration_level(context)
        context.stage("data.microcensus.shares")


def apply_choices(original, choices):
    """Replace modes by stable keys; preserve all source rows and metadata."""
    if choices.duplicated(["person_id", "trip_index"]).any():
        raise ValueError("Duplicate mode choices for a person/trip")
    result = original.merge(choices.rename(columns={"mode": "chosen_mode"}),
                            on=["person_id", "trip_index"], how="left", validate="1:1", sort=False)
    result["mode"] = result["chosen_mode"].fillna(result["mode"])
    return result.drop(columns="chosen_mode")


def execute(context):
    original = context.stage("synthesis.population.trips_with_departures")
    trips = context.stage("mode_choice.routing.java")
    if trips.is_empty():
        if context.config("mode_choice_calibrate_alphas"):
            raise ValueError("Cannot calibrate alphas: no eligible population trips")
        # Always replace the diagnostic artifact, including an all-special-tour
        # population, so an earlier run's choices cannot be mistaken for this run.
        pl.from_pandas(original.head(0)[["person_id", "trip_index", "mode"]]).write_parquet(
            context.path() + "/choices.parquet")
        return original.copy()
    sampler = TourSampler(
        method=context.config("mode_choice_sampling_method"),
        random_cache_max_values=context.config("mode_choice_random_cache_max_values"),
        max_trips=context.config("mode_choice_max_trips_per_tour"),
        cache=TopologyCache(context.config("mode_choice_cache_max_entries"), context.config("mode_choice_cache_max_edges")),
    )
    p = load(context)
    for name in ("walk_speed_m_per_s", "walk_distance_factor", "bike_speed_m_per_s", "bike_distance_factor"):
        if context.config(name) <= 0:
            raise ValueError(f"{name} must be positive")
    trips = trips.sort(["person_id", "trip_index"])
    batches = prepared_batches(context, trips, p)
    if context.config("mode_choice_calibrate_alphas"):
        choices = calibration.run(context, batches, p, sampler)
    else:
        choices = pl.concat([
            choose(batch, context.config("random_seed"), assume_sorted=True, sampler=sampler)
            for batch in batches
        ])
    logger.info("Mode-choice topology cache: %s; peak retained random values: %d",
                sampler.cache.info(), sampler.random_budget.peak)
    if choices.height != trips.height:
        raise ValueError("Tour selection did not return exactly one mode per trip")
    logger.info("Assigned modes to %d trips; preserved %d trips in special or open tours", choices.height, len(original) - choices.height)
    choices.write_parquet(context.path() + "/choices.parquet")
    return apply_choices(original, choices.to_pandas())


def prepared_batches(context, trips, p):
    """Compute costs and base utilities once per complete-person batch."""
    batch_size = context.config("mode_choice_person_batch_size")
    utility_expressions = utilities.expressions(
        p, distance_interaction=context.config("distance_cost_interaction"),
        income_interaction=context.config("income_cost_interaction") and context.config("use_income_in_dmc"))
    for batch in person_batches(trips, batch_size):
        batch = costs.prepare(batch, p, car_cost_model=context.config("car_cost_model").lower(),
                              walk_speed=context.config("walk_speed_m_per_s"), walk_factor=context.config("walk_distance_factor"),
                              bike_speed=context.config("bike_speed_m_per_s"), bike_factor=context.config("bike_distance_factor"),
                              assume_sorted=True)
        # Lazy evaluation shares common subexpressions between mode utilities.
        batch = batch.lazy().with_columns(utility_expressions).collect()
        invalid = batch.select([
            (pl.col(f"{mode}_availability") & ~pl.col(f"utility_{mode}").is_finite().fill_null(False)).sum().alias(mode)
            for mode in utilities.MODES
        ]).row(0)
        for mode, count in zip(utilities.MODES, invalid):
            if count:
                raise ValueError(f"Non-finite {mode} utilities for {count} available trips")
        yield batch
