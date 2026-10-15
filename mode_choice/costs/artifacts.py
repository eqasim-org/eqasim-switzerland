"""Read the trip-keyed cost artifact owned by each routing stage."""
from pathlib import Path
import polars as pl

from mode_choice.defaults import ROUTING_COST_FILENAME
from mode_choice.data.frames import join_variables


def read_costs(context, mode):
    stage = f"mode_choice.routing_variables.{mode}"
    trips = context.stage(stage).select("trip_id")
    costs = pl.read_parquet(Path(context.path(stage)) / ROUTING_COST_FILENAME,
                            columns=["trip_id", f"{mode}_cost_CHF"])
    return join_variables(trips, costs)
