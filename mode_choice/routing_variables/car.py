"""Car time and distance, from Java or the hourly Pandana network."""
from mode_choice.routing.java import configure_mode, car_backend, route_mode
from mode_choice.routing.congestion import input_path


def configure(context):
    configure_mode(context, "car")
    if car_backend(context) == "pandana":
        context.stage("matsim.scenario.network.mapped")


def validate(context):
    path = input_path(context, car_backend(context))
    if path is not None:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino


def execute(context):
    if car_backend(context) == "java":
        return route_mode(context, "car")
    from mode_choice.routing.pandana import route
    trips = context.stage("mode_choice.data.prepare").rename({"trip_id": "identifier"})
    if trips.is_empty():
        import polars as pl
        return trips.select(pl.col("identifier").alias("trip_id")).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("car_travel_time_min"),
            pl.lit(None, dtype=pl.Float64).alias("car_distance_km"))
    return route(context, trips, context.config("mode_choice_routing_batch_size")).rename({"identifier": "trip_id"})
