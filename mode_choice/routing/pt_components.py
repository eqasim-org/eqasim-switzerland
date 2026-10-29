"""DMC components from the exported PT itinerary, without routing it again."""
import polars as pl

from mode_choice.routing.results import complete_routes

PART = pl.Struct({"kind": pl.String, "mode": pl.String, "travelTime": pl.Float64})
PT_LEG = pl.Struct({"departureTime": pl.Float64, "boardingTime": pl.Float64,
                    "travelTime": pl.Float64, "distance": pl.Float64})


def read_components(path, trips):
    saved = pl.read_csv(path, columns=["identifier", "route", "pt_cost_data"],
                        schema_overrides={"identifier": pl.String, "route": pl.String, "pt_cost_data": pl.String})
    saved = saved.rename({"identifier": "trip_id"})
    itineraries = saved.select("trip_id", pl.col("route").str.json_decode(
        pl.Struct({"parts": pl.List(PART)})).struct.field("parts").alias("parts"))
    legs = itineraries.explode("parts").unnest("parts").filter(pl.col("kind") == "leg").with_columns(
        pl.int_range(pl.len()).over("trip_id").alias("index"), pl.len().over("trip_id").alias("count"))
    walking = pl.col("mode").str.contains("walk")
    endpoint = (pl.col("index") == 0) | (pl.col("index") == pl.col("count") - 1)
    walks = legs.group_by("trip_id").agg(
        pl.col("travelTime").filter(walking & endpoint).sum().truediv(60).alias("pt_access_egress_time_min"),
        pl.col("travelTime").filter(walking & ~endpoint).sum().truediv(60).alias("transfer_walk"))
    fares = saved.select("trip_id", pl.col("pt_cost_data").str.json_decode(
        pl.Struct({"legs": pl.List(PT_LEG)})).struct.field("legs").alias("legs"))
    part = pl.element().struct
    waiting = part.field("boardingTime") - part.field("departureTime")
    metrics = fares.select(
        "trip_id",
        pl.col("legs").list.eval(part.field("travelTime") - waiting).list.sum().truediv(60).alias("pt_in_vehicle_time_min"),
        pl.col("legs").list.eval(part.field("distance")).list.sum().truediv(1000).alias("pt_in_vehicle_distance_km"),
        (pl.col("legs").list.len().cast(pl.Int64) - 1).clip(lower_bound=0).cast(pl.Float64).alias("pt_transfers"),
        # Initial waiting is excluded, matching the DMC survey variables.
        pl.col("legs").list.slice(1).list.eval(waiting).list.sum().truediv(60).alias("transfer_wait"),
    ).join(walks, on="trip_id", how="left", validate="1:1").with_columns(
        (pl.col("transfer_wait") + pl.col("transfer_walk")).alias("pt_transfer_time_min"),
    ).drop("transfer_wait", "transfer_walk")
    return complete_routes(trips, metrics, ["pt_in_vehicle_time_min", "pt_access_egress_time_min", "pt_transfer_time_min"])
