"""Write one shared request CSV, consumed by all Java mode routers."""
from pathlib import Path
import polars as pl


def configure(context):
    context.stage("mode_choice.data.prepare")


def execute(context):
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        trips = trips.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in
                                   ("origin_x", "origin_y", "destination_x", "destination_y")])
    path = Path(context.path()) / "requests.csv"
    trips.select("trip_id", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time").rename(
        {"trip_id": "identifier"}).write_csv(path)
    return str(path)
