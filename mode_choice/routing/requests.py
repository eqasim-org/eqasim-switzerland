"""Write shared multimodal-server requests, including attributes for fares."""
from pathlib import Path
import polars as pl
from mode_choice.defaults import PT_GA, PT_HALBTAX, PT_VERBUND, PT_SUBSCRIPTIONS
from mode_choice.routing import options, endpoints


def configure(context):
    options.configure(context)
    context.stage("mode_choice.data.prepare")
    context.stage("mode_choice.defaults")
    if context.config("route_population_once"):
        context.stage("mode_choice.routing.preparation")
        context.stage("mode_choice.routing.config")
        context.stage("synthesis.population.spatial.locations")
        context.stage("synthesis.population.models.subscriptions")


def execute(context):

    # Load trips and align reusable routes with MATSim's prepared endpoints.

    trips = context.stage("mode_choice.data.prepare")
    if context.config("route_population_once"):
        trips = endpoints.attach(context, trips)
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

    # Write shared requests with stable population matching keys and fare data.

    path = Path(context.path()) / "requests.csv"
    trips.select(
        "trip_id", "person_id", "trip_index", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time",
        *(["origin_link_id", "destination_link_id"] if context.config("route_population_once") else []),
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
