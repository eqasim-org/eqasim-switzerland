"""Write shared multimodal-server requests, including attributes for fares."""
from pathlib import Path
import polars as pl
from mode_choice.defaults import PT_GA, PT_HALBTAX, PT_VERBUND, PT_SUBSCRIPTIONS


def configure(context):
    context.stage("mode_choice.data.prepare")
    context.stage("mode_choice.defaults")


def execute(context):
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        trips = trips.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in
                                   ("origin_x", "origin_y", "destination_x", "destination_y",
                                    "departure_time", "home_x", "home_y", "age", "pt_subscription")])

    # Use the same subscription coding as population export, without silently
    # assigning full fares to missing or unknown subscription categories.
    if trips.filter(~pl.col("pt_subscription").is_in(list(PT_SUBSCRIPTIONS)).fill_null(False)).height:
        raise ValueError("Router requests require pt_subscription codes 0 through 4")
    valid_age = pl.col("age").is_finite() & (pl.col("age") >= 0) & (pl.col("age") == pl.col("age").floor())
    if trips.filter(~valid_age.fill_null(False)).height:
        raise ValueError("Router requests require a finite, nonnegative integer age")

    # A regional pass is centred on the person's residence, including on trips
    # starting at work or elsewhere. Never substitute the trip origin for home.
    valid_home = pl.all_horizontal([pl.col(c).is_finite() for c in ("home_x", "home_y")]).fill_null(False)
    if trips.filter(pl.col("pt_subscription").is_in(PT_VERBUND) & ~valid_home).height:
        raise ValueError("Regional PT subscriptions require finite residence home_x and home_y coordinates")

    path = Path(context.path()) / "requests.csv"
    trips.select(
        "trip_id", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time",
        "home_x", "home_y", pl.col("age").cast(pl.Int64),
        pl.col("pt_subscription").replace_strict(PT_SUBSCRIPTIONS, return_dtype=pl.String).alias("subscriptions"),
        # The new server prioritizes subscriptions. Keep equivalent flags while
        # older JARs are still in use; all fares then use the same person data.
        pl.col("pt_subscription").is_in(PT_GA).alias("hasGA"),
        pl.col("pt_subscription").is_in(PT_HALBTAX).alias("hasHalbtaxSubscription"),
        pl.col("pt_subscription").is_in(PT_VERBUND).alias("hasVerbundAbo"),
        pl.lit(False).alias("hasStreckenAbo"),
        pl.lit(False).alias("hasGleis7Abo"),
        pl.lit(False).alias("hasJuniorAbo"),
    ).rename(
        {"trip_id": "identifier"}).write_csv(path)
    return str(path)
