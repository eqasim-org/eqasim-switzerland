"""Build home-based tours and draw one feasible alternative for each tour."""
import polars as pl
from mode_choice.selection.sampler import TourSampler
from mode_choice.defaults import MODES, HOME_PURPOSES


def assign_tours(trips):
    trips = trips.sort(["person_id", "trip_index"])
    return trips.with_columns(
        pl.col("preceding_purpose").is_in(HOME_PURPOSES).cast(pl.Int64)
        .cum_sum().over("person_id").alias("tour_index")
    )


def choose(trips, seed, *, assume_sorted=False, sampler=None):
    """One-shot public wrapper; calibration retains PreparedBatch directly."""
    from mode_choice.tours.prepared import PreparedBatch

    if trips.is_empty():
        return trips.select("person_id", "trip_index").with_columns(pl.lit(None, dtype=pl.String).alias("mode"))
    sampler = sampler if sampler is not None else TourSampler()
    with PreparedBatch(trips, seed, sampler, assume_sorted=assume_sorted) as prepared:
        return prepared.materialize(prepared.sample_codes())


def configure(context):
    context.stage("mode_choice.data.trips")
    context.stage("mode_choice.defaults")


def execute(context):
    """Identify tours before routing, preserving special/open chains unchanged."""
    trips = assign_tours(context.stage("mode_choice.data.trips"))
    group = ["person_id", "tour_index"]
    special = (pl.col("mode").str.ends_with("_loop") | pl.col("mode").eq("remote_walk")
               | pl.col("preceding_purpose").eq("border") | pl.col("following_purpose").eq("border"))
    return trips.select(
        "trip_id", "tour_index",
        (~special.any().over(group)
         & pl.col("preceding_purpose").first().over(group).is_in(HOME_PURPOSES)
         & pl.col("following_purpose").last().over(group).is_in(HOME_PURPOSES)).alias("mode_choice_eligible"),
    )
