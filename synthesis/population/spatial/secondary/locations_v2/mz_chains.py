import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")


def execute(context):
    persons = context.stage("data.microcensus.persons")[["person_id", "home_x", "home_y", "work_x", "work_y"]]
    trips = context.stage("data.microcensus.trips")[0].copy()

    trips = trips.merge(persons, on="person_id", how="left")

    trips_sorted = trips.sort_values(["person_id", "trip_id"]).copy()

    trips_sorted["trip_destination_distance_from_home"] = np.sqrt(
        (trips_sorted["destination_x"] - trips_sorted["home_x"]) ** 2
        + (trips_sorted["destination_y"] - trips_sorted["home_y"]) ** 2
    )

    trips_sorted["trip_destination_distance_from_work"] = 0.0
    has_work_location = trips_sorted["work_x"].notna() & trips_sorted["work_y"].notna() & ~trips_sorted["work_x"].isin([np.inf, -np.inf]) & ~trips_sorted["work_y"].isin([np.inf, -np.inf])
    trips_sorted.loc[has_work_location, "trip_destination_distance_from_work"] = np.sqrt(
        (trips_sorted.loc[has_work_location, "destination_x"] - trips_sorted.loc[has_work_location, "work_x"]) ** 2
        + (trips_sorted.loc[has_work_location, "destination_y"] - trips_sorted.loc[has_work_location, "work_y"]) ** 2
    )

    trips_sorted["crowfly_distance"] = trips_sorted["crowfly_distance"].to_numpy(dtype=np.float64)
    trips_sorted["crowfly_distance"] = np.where(
        np.isfinite(trips_sorted["crowfly_distance"]) & (trips_sorted["crowfly_distance"] >= 0.0),
        trips_sorted["crowfly_distance"],
        0.0,
    )

    trips_sorted["daily_longest_distance_from_home"] = (
        trips_sorted.groupby("person_id")["trip_destination_distance_from_home"].transform("max")
    )

    trips_sorted["daily_crowfly_total"] = trips_sorted.groupby("person_id")["crowfly_distance"].transform("sum")

    trips_sorted["crowfly_consumed_before_trip"] = (
        trips_sorted.groupby("person_id")["crowfly_distance"].cumsum() - trips_sorted["crowfly_distance"]
    )

    # Normalized trip progression within person-day in (0, 1].
    trip_progress = trips_sorted.groupby("person_id").cumcount()
    trips_per_person = trips_sorted.groupby("person_id")["trip_id"].transform("size")
    trips_sorted["trip_position_class"] = trip_progress / np.maximum(trips_per_person-1, 1)


    return trips_sorted[["person_id", "trip_id", 
                         "trip_destination_distance_from_home", "trip_destination_distance_from_work", 
                         "daily_longest_distance_from_home", "daily_crowfly_total", 
                         "crowfly_consumed_before_trip", "trip_position_class"]]
