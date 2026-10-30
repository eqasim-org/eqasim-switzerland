"""One mixed-mode road request: cars, freight and light commercial vehicles."""
from pathlib import Path
import polars as pl
from mode_choice.routing import options


def configure(context):
    options.configure(context)
    context.stage("mode_choice.routing.requests")
    context.stage("mode_choice.routing.freight")
    if context.config("route_population_once"):
        context.stage("mode_choice.routing.preparation")


def execute(context):

    # Use prepared freight coordinates when routes will be reused in MATSim.

    freight = (pl.read_parquet(context.stage("mode_choice.routing.preparation")["freight"])
               if context.config("route_population_once") else context.stage("mode_choice.routing.freight"))
    cars = pl.scan_csv(context.stage("mode_choice.routing.requests"),
                       schema_overrides={"identifier": pl.String, "person_id": pl.String}).with_columns(pl.lit("car").alias("mode"))

    # Send all demand together: each MARouter request starts with zero flow.

    path = Path(context.path()) / "road_requests.csv"
    pl.concat([cars, freight.lazy()], how="diagonal_relaxed").sink_csv(path)
    return str(path)
