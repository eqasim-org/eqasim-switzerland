import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString


def configure(context):
    context.config("random_seed")
    context.config("output_path")

    context.stage("data.cross_border.destinations")
    context.stage("data.microcensus.trips")


def execute(context):
    df       = context.stage("data.cross_border.destinations").copy()
    mz_trips = context.stage("data.microcensus.trips")[0].copy()

    population = df[["cross_border_person_id", "label", "mz_person_id",
                     "origin_x", "origin_y",
                     "destination_x", "destination_y", "destination_id",
                     "is_border_point_projected",
                     "interview_place", "interview_point_id", "interview_geometry_point",
                     "entry_interview_point_id", "entry_interview_geometry_point",
                     "exit_interview_point_id", "exit_interview_geometry_point"]]
    
    mz_trips   = mz_trips[["person_id", "trip_id", 
                           "departure_time", "arrival_time", 
                           "mode", "purpose"]].rename(columns = {"purpose": "following_purpose"})
    
    mz_trips["preceding_purpose"] = mz_trips["following_purpose"].shift(1)
    mz_trips.loc[mz_trips["trip_id"] == 1, "preceding_purpose"] = "home"

    df_trips = pd.merge(population, mz_trips, left_on="mz_person_id", right_on = "person_id")
    del df_trips["person_id"]

    df_trips = df_trips[["cross_border_person_id", "label", "trip_id",
                         "origin_x", "origin_y",
                         "destination_x", "destination_y",
                         "destination_id",
                         "entry_interview_point_id", "entry_interview_geometry_point",
                         "exit_interview_point_id", "exit_interview_geometry_point",
                         "departure_time", "arrival_time",
                         "mode",
                         "preceding_purpose", "following_purpose"]].sort_values(by=["cross_border_person_id", "trip_id"])
    
    # Diversify departure times
    counts = df_trips[["cross_border_person_id", "trip_id"]].groupby("cross_border_person_id").size().reset_index(name="count")["count"].values

    interval = (df_trips[["cross_border_person_id", "departure_time"]]
                .groupby("cross_border_person_id")
                .min()
                .reset_index()["departure_time"]
                .values)

    # If first departure time is just 5min after midnight, we only add a deviation of 5min
    interval = np.minimum(1800.0, interval)

    # Set up RNG
    rng = np.random.RandomState(context.config("random_seed"))
    offset = rng.random_sample(size=(len(counts),)) * interval * 2.0 - interval
    offset = np.repeat(offset, counts)

    df_trips["departure_time"] += offset
    df_trips["arrival_time"]   += offset
    df_trips["departure_time"]  = np.round(df_trips["departure_time"])
    df_trips["arrival_time"]    = np.round(df_trips["arrival_time"])

    # Define trip index
    df_trips = df_trips.sort_values(by=["cross_border_person_id", "trip_id"])
    df_count = df_trips.groupby("cross_border_person_id").size().reset_index(name="count")
    df_trips["trip_index"] = np.hstack([np.arange(count) for count in df_count["count"].values])

    # Adjust origin and destination coordinates
    mask = df_trips["trip_index"] == 1
    df_trips.loc[mask, ["origin_x", "destination_x"]] = df_trips.loc[mask, ["destination_x", "origin_x"]].values
    df_trips.loc[mask, ["origin_y", "destination_y"]] = df_trips.loc[mask, ["destination_y", "origin_y"]].values

    df_trips = df_trips.drop(columns = ["trip_index"])
    df_trips = df_trips.rename(columns = {"cross_border_person_id": "person_id",
                                          "trip_id": "trip_index"})
    
    df_trips = df_trips.sort_values(by=["person_id", "trip_index"]).reset_index(drop=True)

    # For the people going through Switzerland (label = "Through"), only the first trip is relevant
    mask = (df_trips["label"] != "Through") | ((df_trips["label"] == "Through") & (df_trips["trip_index"] == 1))
    df_trips = df_trips[mask]

    activities = pd.DataFrame({
        "person_id": df_trips["person_id"],
        "label": df_trips["label"],
        "activity_index": df_trips["trip_index"],
        "end_time": df_trips["departure_time"],
        "purpose": df_trips["preceding_purpose"],
        "location_x": df_trips["origin_x"],
        "location_y": df_trips["origin_y"],
        "destination_id": None,
        "following_mode": df_trips["mode"]
    })

    activities["start_time"] = df_trips.groupby("person_id")["arrival_time"].shift()
    first_trip_idx = df_trips.groupby("person_id")["trip_index"].idxmin()
    activities.loc[first_trip_idx, "start_time"] = 0.0

    activities["destination_id"] = df_trips.groupby("person_id")["destination_id"].shift()

    last_activities = df_trips.groupby("person_id").tail(1).copy()
    final_activities = pd.DataFrame({
        "person_id": last_activities["person_id"],
        "label": last_activities["label"],
        "activity_index": last_activities["trip_index"] + 1,
        "start_time": last_activities["arrival_time"],
        "end_time": 30*3600,
        "purpose": last_activities["following_purpose"],
        "location_x": last_activities["destination_x"],
        "location_y": last_activities["destination_y"],
        "destination_id": None,
        "following_mode": None
    })

    df_activities = pd.concat([activities, final_activities], ignore_index=True)
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"]).reset_index(drop=True)
    df_activities["start_time"]     = df_activities["start_time"].fillna(0)
    df_activities["destination_id"] = df_activities["destination_id"].fillna(-1)

    geometry = gpd.GeoSeries.from_xy(df_activities["location_x"], df_activities["location_y"])

    df_activities["geometry"] = geometry

    # Through traffic does not perform a real activity in Switzerland. Its first
    # and last activities are therefore the two border anchors used for routing
    # across the country, with direction-specific facility IDs.
    through_entry = (df_activities["label"] == "Through") & (df_activities["activity_index"] == 1)
    through_exit = (df_activities["label"] == "Through") & df_activities["following_mode"].isna()
    df_activities.loc[through_entry, "purpose"] = "border"
    df_activities.loc[through_entry, "destination_id"] = df_trips.loc[
        df_trips["label"] == "Through", "entry_interview_point_id"
    ].values
    df_activities.loc[through_entry, "geometry"] = df_trips.loc[
        df_trips["label"] == "Through", "entry_interview_geometry_point"
    ].values
    df_activities.loc[through_exit, "purpose"] = "border"
    df_activities.loc[through_exit, "destination_id"] = df_trips.loc[
        df_trips["label"] == "Through", "exit_interview_point_id"
    ].values
    df_activities.loc[through_exit, "geometry"] = df_trips.loc[
        df_trips["label"] == "Through", "exit_interview_geometry_point"
    ].values

    df_activities["duration"] = df_activities["end_time"] - df_activities["start_time"]
    df_activities["is_last"]  = df_activities["following_mode"].isna()

    df_activities = df_activities[["person_id", "activity_index", "label",
                                   "start_time", "end_time", "duration",
                                   "purpose", "is_last",
                                   "geometry", "destination_id", "following_mode"
                                   ]]
    
    # ── 0. Sort ────────────────────────────────────────────────────────────────────
    df_activities = df_activities.sort_values(["person_id", "activity_index"]).reset_index(drop=True)

    # ── 1. Compute trip durations ──────────────────────────────────────────────────
    df_activities["next_start_time"] = df_activities.groupby("person_id")["start_time"].shift(-1)
    df_activities["trip_duration"]   = df_activities["next_start_time"] - df_activities["end_time"]

    # ── 2. Extract nth activities per person ───────────────────────────────────────
    def nth_activity(df, n):
        return (
            df.groupby("person_id")
            .nth(n)
            .reset_index()[["person_id", "geometry", "end_time", "trip_duration", "following_mode"]]
        )

    act1 = nth_activity(df_activities, 0).rename(columns={"geometry": "geom_1", "end_time": "end_time_1", "trip_duration": "trip_duration_12"})
    act2 = nth_activity(df_activities, 1).rename(columns={"geometry": "geom_2", "end_time": "end_time_2", "trip_duration": "trip_duration_23"})
    act3 = nth_activity(df_activities, 2).rename(columns={"geometry": "geom_3"})

    # ── 3. Helper: compute fake activity timing ────────────────────────────────────
    def compute_fake_activity(fake_df, geom_before, geom_after, end_time_col, trip_duration_col):
        """
        Interpolates the start_time of a fake activity between two real activities
        based on the ratio of distances.
        
        fake_df         : dataframe of fake activities, must have a 'geometry' column (border point)
        geom_before     : column name for the geometry of the activity before
        geom_after      : column name for the geometry of the activity after
        end_time_col    : column name for end_time of the activity before
        trip_duration_col: column name for the trip duration between the two real activities
        """
        df = fake_df.copy()
        
        df["dist_before"] = df.apply(lambda r: r[geom_before].distance(r["geometry"]), axis=1)
        df["dist_after"]  = df.apply(lambda r: r[geom_after].distance(r["geometry"]),  axis=1)
        df["ratio"]       = df["dist_before"] / (df["dist_before"] + df["dist_after"])
        
        df["start_time"]  = df[end_time_col] + df[trip_duration_col] * df["ratio"]
        df["end_time"]    = df["start_time"] + df["duration"]
        
        return df.drop(columns=["dist_before", "dist_after", "ratio", geom_before, geom_after,
                                end_time_col, trip_duration_col])

    # ── 4. Build fake activities at index 1.5 (all persons with mode car) ───────────────────────
    car_users      = (df_activities["following_mode"] == "car") | (df_activities["following_mode"] == "car_passenger")
    car_users      = df_activities[car_users]["person_id"].values.tolist()

    # From-To car users get two artificial border activities. We also insert them
    # for projected origins so the routing link comes from the directional border
    # facility instead of the generic home facility at the same coordinate.
    mask_car_users = population["cross_border_person_id"].isin(car_users) & (population["label"] == "From-To")

    activities1point5 = (
        population[mask_car_users].copy()
        .rename(columns={"cross_border_person_id": "person_id", "entry_interview_geometry_point": "geometry"})
    )
    activities1point5["activity_index"] = 1.5
    activities1point5["duration"]       = 1
    activities1point5["is_last"]        = False
    activities1point5["destination_id"] = activities1point5["entry_interview_point_id"]
    activities1point5["purpose"]        = "border"

    activities1point5 = activities1point5.merge(act1[["person_id", "geom_1", "end_time_1", "trip_duration_12", "following_mode"]], on="person_id", how="left")
    activities1point5 = activities1point5.merge(act2[["person_id", "geom_2"]], on="person_id", how="left")

    activities1point5 = compute_fake_activity(
        activities1point5,
        geom_before="geom_1", geom_after="geom_2",
        end_time_col="end_time_1", trip_duration_col="trip_duration_12"
    )

    # ── 5. Build fake activities at index 2.5 (only persons with 3+ activities) ───
    persons_with_3_acts = df_activities.groupby("person_id").size()
    persons_with_3_acts = persons_with_3_acts[persons_with_3_acts >= 3].index

    activities2point5 = (
        population[population["cross_border_person_id"].isin(persons_with_3_acts) & mask_car_users]
        .copy()
        .rename(columns={"cross_border_person_id": "person_id", "exit_interview_geometry_point": "geometry"})
    )
    activities2point5["activity_index"] = 2.5
    activities2point5["duration"]       = 1
    activities2point5["is_last"]        = False
    activities2point5["destination_id"] = activities2point5["exit_interview_point_id"]
    activities2point5["purpose"]        = "border"

    activities2point5 = activities2point5.merge(act2[["person_id", "geom_2", "end_time_2", "trip_duration_23", "following_mode"]], on="person_id", how="left")
    activities2point5 = activities2point5.merge(act3[["person_id", "geom_3"]], on="person_id", how="left")

    activities2point5 = compute_fake_activity(
        activities2point5,
        geom_before="geom_2", geom_after="geom_3",
        end_time_col="end_time_2", trip_duration_col="trip_duration_23"
    )

    # ── 6. Combine and re-sort ─────────────────────────────────────────────────────
    df_activities = df_activities.drop(columns=["next_start_time", "trip_duration"])

    df_activities = (
        pd.concat([df_activities, activities1point5, activities2point5])
        .sort_values(["person_id", "activity_index"])
        .reset_index(drop=True)
    )

    # ── 7. Fix is_last ─────────────────────────────────────────────────────────────
    df_activities["is_last"] = (
        df_activities["activity_index"] == df_activities.groupby("person_id")["activity_index"].transform("max")
    )

    df_activities["activity_index"] = df_activities.groupby("person_id").cumcount() + 1

    df_activities = df_activities[["person_id", "activity_index", "label",
                                   "start_time", "end_time", "duration",
                                   "purpose", "is_last",
                                   "geometry", "destination_id", "following_mode"
                                   ]]
    
    df_sorted = df_activities.sort_values(["person_id", "activity_index"])

    # Shift to get origin and destination activity side by side
    trips = pd.DataFrame({
        "person_id"       : df_sorted["person_id"],
        "trip_index"      : df_sorted["activity_index"],
        "origin_id"       : df_sorted["destination_id"],
        "departure_time"  : df_sorted["end_time"],
        "geom_origin"     : df_sorted["geometry"].values,
        "mode"            : df_sorted["following_mode"],
        "geom_destination": df_sorted.groupby("person_id")["geometry"].shift(-1).values,
        "destination_id"  : df_sorted.groupby("person_id")["destination_id"].shift(-1).values,
        "arrival_time"    : df_sorted.groupby("person_id")["start_time"].shift(-1).values,
    })

    # Drop last activity of each person (no outgoing trip)
    trips = trips[trips["mode"].notna()].copy()

    # Compute trip duration
    trips["trip_duration"] = trips["arrival_time"] - trips["departure_time"]

    # Build LineString geometry
    trips["geometry"] = trips.apply(
        lambda r: LineString([r["geom_origin"], r["geom_destination"]]), axis=1
    )

    # Drop helper columns and convert to GeoDataFrame
    trips = trips.drop(columns=["geom_origin", "geom_destination"])
    trips = gpd.GeoDataFrame(trips, geometry="geometry", crs = "EPSG:2056")

    # Reindex trip index as integer
    trips["trip_index"] = trips.groupby("person_id").cumcount() + 1

    #trips.to_file(f"{context.config("output_path")}/trips_crossborder.shp")

    return df_activities
