import numpy as np
import pandas as pd
from .hierarchical_utils import encode_purpose

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

    trips_sorted["daily_longest_distance_from_work"] = (
        trips_sorted.groupby("person_id")["trip_destination_distance_from_work"].transform("max")
    )

    trips_sorted["daily_crowfly_total"] = trips_sorted.groupby("person_id")["crowfly_distance"].transform("sum")

    trips_sorted["crowfly_consumed_before_trip"] = (
        trips_sorted.groupby("person_id")["crowfly_distance"].cumsum() - trips_sorted["crowfly_distance"]
    )

    # Normalized trip progression within person-day in (0, 1].
    trip_progress = trips_sorted.groupby("person_id").cumcount()
    trips_per_person = trips_sorted.groupby("person_id")["trip_id"].transform("size")
    trips_sorted["trip_position_class"] = trip_progress / np.maximum(trips_per_person-1, 1)

    # Normalize trip departure times to [0, 1] within person-day.
    min_departure = 0.0
    max_departure = 3600.0 * 24.0
    trips_sorted["departure_time_normalized"] = (trips_sorted["departure_time"] % max_departure - min_departure) / (max_departure - min_departure)

    # Adding activity duration (we assume the last activity duration is 8h, which is a reasonable assumption for a night activity)
    trips_sorted["activity_duration"] = trips_sorted["departure_time"].shift(-1) - trips_sorted["arrival_time"]
    sel = trips_sorted["person_id"].shift(-1) != trips_sorted["person_id"]
    trips_sorted.loc[sel, "activity_duration"] = 3600.0 * 8.0

    trips_sorted["activity_duration_h"] = trips_sorted["activity_duration"].clip(0.0, 3600.0 * 16.0)/3600.0
    # Adding activity chain as sum of one hot encoded purposes
    
    def build_activity_chain(group):
        sequence = [group.iloc[0]["origin_purpose"]] + group["purpose"].tolist()
        vectors = [encode_purpose(p) for p in sequence]
        return np.sum(vectors, axis=0)

    _chain_series = trips_sorted.groupby("person_id", sort=False).apply(build_activity_chain)
    trips_sorted["activity_chain"] = trips_sorted["person_id"].map(_chain_series)
    

    return trips_sorted[["person_id", "trip_id", 
                         "trip_destination_distance_from_home", "trip_destination_distance_from_work", 
                         "daily_longest_distance_from_home", "daily_longest_distance_from_work", "daily_crowfly_total", 
                         "crowfly_consumed_before_trip", "trip_position_class","departure_time_normalized", "activity_duration_h",
                         "activity_chain"]]
