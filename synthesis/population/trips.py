import numpy as np
import pandas as pd
import logging
from data.utils import coerce_boolean_series
from data.microcensus.secondary_trips import recode_nonstudent_education
from data.cross_border.travel_times import estimate_travel_time

logger = logging.getLogger("synpp")
"""
This stage attaches all trip relevant information to the synthetic population.
"""


BORDER_MATCH_COLUMNS = [
    "cross_border_person_id", "destination_country_raw",
    "interview_geometry_point", "border_crossing_trip_mode", "interview_point_id",
    "border_crossing_point", "border_crossing_direction", "border_interview_place",
]

ORIGINAL_PURPOSE_COLUMNS = ["original_preceding_purpose", "original_following_purpose"]


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.constants")
    context.stage("synthesis.population.spatial.primary.work.work_remotly")

    context.config("random_seed")
    context.config("specific_day_scenario", default = "workday")


def execute(context):

    # Load recipients and donor trips, preserving the assigned crossing event.

    c      = context.stage("data.constants")

    if c.census == "statpop":
        df_persons = context.stage("synthesis.population.enriched")[[
            "person_id", "mz_person_id", "age", "is_truck_driver", "is_outside_of_switzerland", "is_crossing_the_border", "canton_id",
            "home_x", "home_y",
        ] + BORDER_MATCH_COLUMNS].copy()

    elif c.census == "are_synpop":
        df_persons = context.stage("synthesis.population.enriched")[[
            "person_id", "mz_person_id", "age_class", "is_truck_driver", "is_outside_of_switzerland", "is_crossing_the_border", "canton_id",
            "home_x", "home_y",
        ] + BORDER_MATCH_COLUMNS].copy()

    df_trips = pd.DataFrame(context.stage("data.microcensus.trips")[0], copy=True)[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose", "origin_purpose"
    ]]
    df_trips.columns = ["mz_person_id", "trip_id", "departure_time", "arrival_time", "mode", "following_purpose", "preceding_purpose"]

    if df_persons["person_id"].duplicated().any():
        raise ValueError("enriched population contains duplicate person_id values")
    if df_trips.duplicated(["mz_person_id", "trip_id"]).any():
        raise ValueError("microcensus trips contain duplicate person/trip keys")

    df_trips = pd.merge(df_persons, df_trips, on="mz_person_id")
    if df_trips.duplicated(["person_id", "trip_id"]).any():
        raise ValueError("trip attachment produced duplicate synthetic person/trip keys")

    # Children do not have any trips from the microcensus
    f = np.isnan(df_trips["mz_person_id"])
    if c.census == "statpop":
        assert ((df_trips[f]["age"] > c.MZ_AGE_THRESHOLD).all())
    elif c.census == "are_synpop":
        assert ((df_trips[f]["age_class"] >= 1).all())

    # We deliberately delete them here, since other persons also may not have any
    # trips. May be improved later. TODO
    df_trips = df_trips[~f]

    if c.census == "statpop":
        # Use the recipient's final student status, after the stock correction.
        # Raw donor diaries remain unchanged for matching and survey audits.
        persons = context.stage("synthesis.population.enriched")
        nonstudents = persons.loc[pd.to_numeric(persons["is_student"], errors="raise").eq(0), "person_id"]
        df_trips = recode_nonstudent_education(
            df_trips, nonstudents, ("preceding_purpose", "following_purpose"))
    else:
        for column in ["preceding_purpose", "following_purpose"]:
            df_trips["original_" + column] = df_trips[column]

    df_trips.loc[:, "travel_time"] = df_trips.loc[:, "arrival_time"] - df_trips.loc[:, "departure_time"]

    df_trips = df_trips[["person_id", "mz_person_id", "trip_id",
                         "departure_time", "arrival_time",
                         "travel_time", "mode",
                         "preceding_purpose", "following_purpose",
                         "is_truck_driver", "is_outside_of_switzerland",
                         "is_crossing_the_border"] + ORIGINAL_PURPOSE_COLUMNS].sort_values(by=["person_id", "trip_id"])

    # Diversify departure times
    counts = df_trips[["person_id", "trip_id"]].groupby("person_id").size().reset_index(name="count")["count"].values

    interval = (df_trips[["person_id", "departure_time"]]
                .groupby("person_id")
                .min()
                .reset_index()["departure_time"]
                .values)

    # If first departure time is just 5min after midnight, we only add a deviation of 5min
    interval = np.minimum(1800.0, interval)

    # Set up RNG
    rng    = np.random.RandomState(context.config("random_seed"))
    offset = rng.random_sample(size=(len(counts),)) * interval * 2.0 - interval
    offset = np.repeat(offset, counts)

    df_trips["departure_time"] += offset
    df_trips["arrival_time"]   += offset
    df_trips["departure_time"]  = np.round(df_trips["departure_time"])
    df_trips["arrival_time"]    = np.round(df_trips["arrival_time"])
    df_trips["trip_duration"]   = df_trips["arrival_time"] - df_trips["departure_time"]

    # Define trip index
    df_trips = df_trips.sort_values(by=["person_id", "trip_id"])
    df_count = df_trips.groupby("person_id").size().reset_index(name="count")
    df_trips["trip_index"] = (np.hstack([np.arange(count) for count in df_count["count"].values])
                              if len(df_count) else np.array([], dtype=int))

    # Define remote_walk as main mode to go to work for agents working from home
    remote_agents = context.stage("synthesis.population.spatial.primary.work.work_remotly")
    remote_agents = set(remote_agents["person_id"].values)
    f = (df_trips["following_purpose"] == "work") & (df_trips["person_id"].isin(remote_agents))
    df_trips.loc[f, "mode"] = "remote_walk"

    # Delete trips for truck drivers and agents not in Switzerland
    initial_length = len(df_trips)
    is_truck_driver = coerce_boolean_series(
        df_trips["is_truck_driver"], name="is_truck_driver")
    df_trips = df_trips[~is_truck_driver]
    final_length   = len(df_trips)
    share          = round((final_length - initial_length) / initial_length * 100, 2)

    logger.info(f"Removed {initial_length - final_length} ({share}%) trips (truck drivers)")

    initial_length = len(df_trips)
    is_outside = coerce_boolean_series(
        df_trips["is_outside_of_switzerland"], name="is_outside_of_switzerland")
    df_trips = df_trips[~is_outside]
    final_length   = len(df_trips)
    share          = round((final_length - initial_length) / initial_length * 100, 2)

    logger.info(f"Removed {initial_length - final_length} ({share}%) trips (people outside of Switzerland)")

    # Adapt trips for agents crossing the border: they get a single trip
    # (home-border or border-home) instead of their regular
    # microcensus-derived chain. The destination country, crossing point, and
    # trip mode are the ones synthesis.population.models.cross_border already
    # matched this person to (from data.cross_border.swiss_residents_od) --
    # reading that match here instead of sampling independently keeps it
    # consistent with matsim/scenario/population.py's crossBorderOD attribute
    # and with the location synthesis.population.spatial.locations assigns.
    is_crossing = coerce_boolean_series(
        df_trips["is_crossing_the_border"], name="is_crossing_the_border")
    df_trips_noncb = df_trips[~is_crossing].copy()
    df_trips_noncb["destination_country_raw"] = None
    df_trips_noncb["interview_geometry_point"] = None
    df_trips_noncb["interview_point_id"]       = None
    df_trips_noncb["border_crossing_point"]    = None

    is_cb_person = coerce_boolean_series(
        df_persons["is_crossing_the_border"], name="is_crossing_the_border")
    is_cb_person &= ~coerce_boolean_series(
        df_persons["is_truck_driver"], name="is_truck_driver")
    is_cb_person &= ~coerce_boolean_series(
        df_persons["is_outside_of_switzerland"], name="is_outside_of_switzerland")
    is_cb_person &= df_persons["cross_border_person_id"].notna()

    df_cb_persons = df_persons.loc[is_cb_person, ["person_id", "home_x", "home_y"] + BORDER_MATCH_COLUMNS].copy()

    final_columns = [
        "person_id", "mz_person_id", "trip_id", "trip_index",
        "departure_time", "arrival_time",
        "preceding_purpose",
        "following_purpose",
        "trip_duration",
        "mode",
        "destination_country_raw", "interview_geometry_point", "interview_point_id",
        "border_crossing_point",
    ] + ORIGINAL_PURPOSE_COLUMNS

    _write_border_trip_audit(context, df_persons, is_cb_person)

    if len(df_cb_persons) == 0:
        return df_trips_noncb[final_columns]

    # Consume the event's assigned direction. Redrawing 50/50 here would
    # discard the entry/exit controls established before population sampling.

    direction = df_cb_persons["border_crossing_direction"]
    if not direction.isin(["entry", "exit"]).all():
        raise ValueError("Swiss crossing direction is missing/invalid; rerun synthesis.population.models.cross_border")
    is_leaving = direction.eq("exit").to_numpy()

    sampled_times = sample_border_trip_times(context, df_cb_persons, is_leaving, rng)

    df_trips_cb = pd.DataFrame({
        "person_id": df_cb_persons["person_id"].values,
        "mz_person_id": df_cb_persons["cross_border_person_id"].values,
        "trip_id": 1,
        "trip_index": 0,
        "departure_time": sampled_times["departure_time"].values,
        "arrival_time": sampled_times["arrival_time"].values,
        "preceding_purpose": np.where(is_leaving, "home", "border"),
        "following_purpose": np.where(is_leaving, "border", "home"),
        "mode": df_cb_persons["border_crossing_trip_mode"].values,
        "destination_country_raw": df_cb_persons["destination_country_raw"].values,
        "interview_geometry_point": df_cb_persons["interview_geometry_point"].values,
        "interview_point_id": df_cb_persons["interview_point_id"].values,
        "border_crossing_point": df_cb_persons["border_crossing_point"].values,
    })
    df_trips_cb["trip_duration"] = df_trips_cb["arrival_time"] - df_trips_cb["departure_time"]
    # These are newly constructed border trips, not recoded diary education.
    for column in ["preceding_purpose", "following_purpose"]:
        df_trips_cb["original_" + column] = df_trips_cb[column]

    df_trips = pd.concat([df_trips_noncb, df_trips_cb], ignore_index=True, sort=False)

    return df_trips[final_columns]


def sample_border_trip_times(context, df_cb_persons, is_leaving, rng):
    """
    Departure and arrival times of the single border trip of the Swiss residents
    crossing the border.

    The time of day comes from the microcensus trips that really cross the border
    (on the scenario's day type), drawn by weight among those going in the same
    direction (leaving / entering Switzerland) with the same mode (car or pt), so
    that people leave in the morning and come back in the evening. A uniform
    +/- 15 min offset spreads them over the rounded microcensus times.

    The duration comes from the distance between the home and the border activity
    (data.cross_border.travel_times), and the microcensus time is kept at the Swiss
    end of the trip: the departure from home when leaving, the arrival at home
    when entering (the departure from the border is computed backwards).
    """

    df_mz_trips = context.stage("data.microcensus.trips")[0]
    df_mz_persons = context.stage("data.microcensus.persons")

    day = context.config("specific_day_scenario")
    if day == "weekend":
        day_persons = df_mz_persons.loc[df_mz_persons["weekend"], "person_id"]
    elif day == "workday":
        day_persons = df_mz_persons.loc[df_mz_persons["workday"], "person_id"]
    else:
        day_persons = df_mz_persons.loc[df_mz_persons["day"] == day, "person_id"]

    pool = df_mz_trips[df_mz_trips["trip_crossing_border"] & df_mz_trips["person_id"].isin(day_persons)].copy()
    pool = pool[pool["mode"].isin(["car", "car_passenger", "pt"])]
    pool["leaving"] = pool["origin_in_ch"].astype(bool)
    pool["mode_group"] = np.where(pool["mode"] == "pt", "pt", "car")

    modes = df_cb_persons["border_crossing_trip_mode"].values
    mode_groups = np.where(modes == "pt", "pt", "car")

    swiss_end_time = np.zeros(len(df_cb_persons))

    for leaving in (True, False):
        for mode_group in ("car", "pt"):
            selection = (is_leaving == leaving) & (mode_groups == mode_group)
            if not selection.any():
                continue

            candidates = pool[(pool["leaving"] == leaving) & (pool["mode_group"] == mode_group)]
            if len(candidates) == 0:  # no microcensus trip for this mode: any mode, same direction
                candidates = pool[pool["leaving"] == leaving]
            if len(candidates) == 0:
                raise RuntimeError("No border-crossing trip in the microcensus to sample the cross-border departure times from")

            weights = candidates["weight_person"].values.astype(float)
            weights = weights / weights.sum()
            drawn = candidates.iloc[rng.choice(len(candidates), size = selection.sum(), p = weights)]

            # The Swiss end of the trip: the departure from home, or the arrival at home
            swiss_end_time[selection] = drawn["departure_time" if leaving else "arrival_time"].values

    anchor = np.clip(swiss_end_time + rng.uniform(-900.0, 900.0, size = len(swiss_end_time)), 0.0, None)

    distance = np.array([
        np.hypot(point.x - x, point.y - y) if hasattr(point, "x") else np.nan
        for point, x, y in zip(df_cb_persons["border_crossing_point"].values,
                               df_cb_persons["home_x"].values, df_cb_persons["home_y"].values)
    ])
    travel_time = estimate_travel_time(distance, modes)

    departure_time = np.where(is_leaving, anchor, np.clip(anchor - travel_time, 0.0, None))
    arrival_time = np.where(is_leaving, anchor + travel_time, np.maximum(anchor, travel_time))

    logger.info("Border trips of the Swiss residents: median travel time %.0f min (%d leaving, %d entering Switzerland, from %d microcensus border-crossing trips)",
                np.nanmedian(travel_time) / 60, int(is_leaving.sum()), int((~is_leaving).sum()), len(pool))

    return pd.DataFrame({"departure_time": np.round(departure_time), "arrival_time": np.round(arrival_time)})


def validate(context):
    # Synpp hashes this stage, but imported helpers also change its output.
    import hashlib
    from pathlib import Path
    import data.microcensus.secondary_trips
    return hashlib.sha256(Path(data.microcensus.secondary_trips.__file__).read_bytes()).hexdigest()


def _write_border_trip_audit(context, persons, generated):
    """Separate sampled selections from downstream truck/outside exclusions."""
    selected = coerce_boolean_series(persons["is_crossing_the_border"], name="is_crossing_the_border")
    selected &= persons["cross_border_person_id"].notna()
    audit = persons.loc[selected, ["border_interview_place", "border_crossing_trip_mode",
                                   "border_crossing_direction"]].copy()
    audit["selected_sample_events"] = 1
    truck = coerce_boolean_series(persons["is_truck_driver"], name="is_truck_driver")
    outside = coerce_boolean_series(persons["is_outside_of_switzerland"], name="is_outside_of_switzerland")
    # Mutually exclusive exclusions keep the accounting additive.
    audit["excluded_truck_driver"] = truck.loc[selected].astype(int)
    audit["excluded_outside"] = (outside & ~truck).loc[selected].astype(int)
    audit["generated_events"] = generated.loc[selected].astype(int)
    keys = ["border_interview_place", "border_crossing_trip_mode", "border_crossing_direction"]
    audit.groupby(keys, dropna=False).sum().to_csv(f"{context.path()}/swiss_crossing_trip_counts.csv")
