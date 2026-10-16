"""Write one shared request CSV, consumed by all Java mode routers."""
from pathlib import Path
import polars as pl
from mode_choice.defaults import PT_GA, PT_HALBTAX, PT_VERBUND


def configure(context):
    context.stage("mode_choice.data.prepare")
    context.stage("mode_choice.defaults")


def execute(context):
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        trips = trips.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in
                                   ("origin_x", "origin_y", "destination_x", "destination_y",
                                    "departure_time", "home_x", "home_y", "age", "pt_subscription")])
    path = Path(context.path()) / "requests.csv"
    trips.select(
        "trip_id", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time",
        "home_x", "home_y", pl.col("age").cast(pl.Int64),
        pl.col("pt_subscription").is_in(PT_GA).alias("hasGA"),
        pl.col("pt_subscription").is_in(PT_HALBTAX).alias("hasHalbtaxSubscription"),
        pl.col("pt_subscription").is_in(PT_VERBUND).alias("hasVerbundAbo"),
        pl.lit(False).alias("hasStreckenAbo"),
        pl.lit(False).alias("hasGleis7Abo"),
        pl.lit(False).alias("hasJuniorAbo"),
    ).rename(
        {"trip_id": "identifier"}).write_csv(path)
    return str(path)
