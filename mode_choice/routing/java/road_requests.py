"""One mixed-mode road request: cars, freight and light commercial vehicles."""
from pathlib import Path
import polars as pl
from mode_choice.routing import options


def configure(context):
    options.configure(context)
    context.stage("mode_choice.routing.java.requests")
    context.stage("mode_choice.routing.java.freight")
    if context.config("use_marouter_in_mode_choice"):
        context.stage("mode_choice.routing.java.weights")
    if context.config("route_population_once"):
        context.stage("mode_choice.routing.java.preparation")


def execute(context):

    # Use prepared freight coordinates when routes will be reused in MATSim.

    freight = (pl.read_parquet(context.stage("mode_choice.routing.java.preparation")["freight"])
               if context.config("route_population_once") else context.stage("mode_choice.routing.java.freight"))
    cars = pl.scan_csv(context.stage("mode_choice.routing.java.requests"),
                       schema_overrides={"identifier": pl.String, "person_id": pl.String})

    # Weigh driver demand; non-drivers are zero-flow passenger probes only.

    if context.config("use_marouter_in_mode_choice"):
        weights = context.stage("mode_choice.routing.java.weights").lazy()
        cars = cars.join(weights, on="identifier", how="inner", validate="1:1", maintain_order="left")
        freight = freight.drop("mode").lazy().join(weights, on="identifier", how="inner", validate="1:1", maintain_order="left")
    else:
        cars = cars.with_columns(pl.lit("car").alias("mode"))
        freight = freight.lazy()

    # Send all demand together: each MARouter request starts with zero flow.

    path = Path(context.path()) / "road_requests.csv"
    pl.concat([cars, freight], how="diagonal_relaxed").sink_csv(path)
    return str(path)
