import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import LineString

from data.cross_border.destinations import make_entry_border_facility_id, make_exit_border_facility_id


def merge_refined_points(population, network_projection, real_x_col, real_y_col,
                          interview_x_col, interview_y_col, out_x_col, out_y_col):
    """
    Adds out_x_col/out_y_col to `population`: the data.cross_border.network_projection-
    refined point for (real_x_col, real_y_col) -> (interview_x_col, interview_y_col),
    where one was computed - NaN where the end wasn't projected in the first
    place (real_x_col is NaN, e.g. destination_residence_x/y for "From-To"
    rows - only "Through" trips can have a projected destination) or no
    network route was found for it.

    Merge keys are int-truncated to match data.cross_border.destinations' own
    int cast of these coordinate columns (both sides trace back to the same
    underlying data.cross_border.generate_od float values, left untouched in
    between, so the truncation is identical and the merge is exact).
    """
    population = population.copy()
    population[out_x_col] = np.nan
    population[out_y_col] = np.nan

    valid = population[real_x_col].notna() & population[interview_x_col].notna()
    if len(network_projection) == 0 or not valid.any():
        return population

    df_refined = network_projection[["real_x", "real_y", "interview_x", "interview_y", "refined_x", "refined_y"]].copy()
    key_columns = ["real_x", "real_y", "interview_x", "interview_y"]
    df_refined[key_columns] = df_refined[key_columns].astype(int)
    df_refined = df_refined.drop_duplicates(subset = key_columns, keep = "first")

    keys = pd.DataFrame({
        "real_x": population.loc[valid, real_x_col].astype(int),
        "real_y": population.loc[valid, real_y_col].astype(int),
        "interview_x": population.loc[valid, interview_x_col].astype(int),
        "interview_y": population.loc[valid, interview_y_col].astype(int),
    })
    merged = keys.merge(df_refined, on = key_columns, how = "left")

    population.loc[valid, out_x_col] = merged["refined_x"].values
    population.loc[valid, out_y_col] = merged["refined_y"].values

    return population


def rename_projected_ends(df_activities, population):
    """
    Relocates the "From-To" home/other activity ends that sit on a projected
    origin, since those are NOT border crossings for "From-To" agents - see
    below. "Through" agents' origin/destination bookends and border
    crossings are built separately, in execute() (see build_through_ends).

    data.cross_border.generate_od moves an origin that is more than 20 km
    from the border onto the nearest surveyed interview place, so the RAW
    interview place coordinate is not where a "From-To" agent lives - it is
    (approximately) the point where they cross the border. "From-To" agents
    DO reside at that end (just not at the raw interview place - their real
    home is data.cross_border.network_projection's refined, network-real
    point, in population's refined_origin_x/y - see merge_refined_points).
    Collapsing their home/other activity onto "border" would hide the fact
    they have an actual residence; instead, the home/other activity here gets
    relocated to that refined point (purpose and destination_id both left
    untouched), and execute()'s activities1point5/activities2point5 insert an
    actual "border" activity right next to it, at the raw interview place,
    modeling the crossing itself.

    network_projection never refines a projected PT origin (see its
    docstring) and can occasionally fail to find a route for a projected car
    origin - in both cases refined_origin_x/y is NaN and there is nothing to
    relocate onto. Left alone, that activity would keep its survey purpose
    (usually "home") while sitting exactly on the raw interview place (e.g. a
    train station platform), which reads as "this agent lives at the
    station". Its purpose is relabelled "other" here instead, to flag that
    the point isn't a real, geocoded residence - only the label changes, the
    activity stays at the raw interview place.
    """

    projection = population.rename(columns = {"cross_border_person_id": "person_id"})
    projection = projection[["person_id", "origin_is_projected", "refined_origin_x", "refined_origin_y"]]

    df = df_activities.merge(projection, on = "person_id", how = "left")

    is_first   = df["activity_index"] == df.groupby("person_id")["activity_index"].transform("min")
    is_through = df["label"] == "Through"
    is_home_end = ~is_through & (is_first | df["is_last"]) & df["origin_is_projected"].fillna(False)

    has_refined_origin = df["refined_origin_x"].notna()

    on_home_relocate = is_home_end & has_refined_origin
    on_home_unrefined = is_home_end & ~has_refined_origin

    df.loc[on_home_relocate, "geometry"] = gpd.points_from_xy(
        df.loc[on_home_relocate, "refined_origin_x"], df.loc[on_home_relocate, "refined_origin_y"],
    )
    df.loc[on_home_unrefined, "purpose"] = "other"

    return df.drop(columns = ["origin_is_projected", "refined_origin_x", "refined_origin_y"])


def build_through_ends(df_activities, df_trips, population):
    """
    Gives "Through" agents (who cross Switzerland without a real activity in
    it - see the module-level notes in data.cross_border.destinations/
    generate_od) real trips before and after the border crossing, instead of
    starting/ending the day sitting at the crossing itself:

        origin -> border (entry) -> border (exit) -> destination

    `origin`/`destination` sit at data.cross_border.network_projection's
    refined, network-real point where that end was projected (too far from
    the border to use directly), or at the real (already close-to-border)
    point otherwise - see merge_refined_points. The two `border` activities
    keep the exact geometry/facility ids the rest of this module already
    assigns them (data.cross_border.destinations' entry/exit interview
    points) - only their timing shrinks to a near-instant pass-through, to
    make room for the new bookend activities.

    Timing: the single kept trip for a "Through" person (see execute()'s "For
    the people going through Switzerland ... only the first trip is
    relevant") carries a departure_time/arrival_time pair that used to be the
    entire border_entry -> border_exit window. That's now split: `origin`
    occupies the day up to the original departure_time, `border (entry)` is
    a 1-second pass-through right after, the original departure_time ->
    arrival_time span becomes the border (entry) -> border (exit) transit,
    `border (exit)` is another 1-second pass-through, and `destination`
    occupies the rest of the day. This keeps the in-Switzerland transit
    duration exactly as before, just bookended rather than treated as the
    start/end of the whole day.
    """

    through_population = population[population["label"] == "Through"].rename(
        columns = {"cross_border_person_id": "person_id"},
    )

    if len(through_population) == 0:
        return df_activities

    through_trip = df_trips.loc[df_trips["label"] == "Through", ["person_id", "departure_time", "arrival_time", "mode"]]
    through_trip = through_trip.set_index("person_id")

    origin_end_time     = through_trip["departure_time"]
    entry_start_time    = origin_end_time
    entry_end_time      = entry_start_time + 1
    exit_start_time      = entry_end_time + (through_trip["arrival_time"] - through_trip["departure_time"])
    exit_end_time        = exit_start_time + 1

    through_entry = (df_activities["label"] == "Through") & (df_activities["activity_index"] == 1)
    through_exit  = (df_activities["label"] == "Through") & df_activities["following_mode"].isna()

    df_activities.loc[through_entry, "start_time"]     = df_activities.loc[through_entry, "person_id"].map(entry_start_time)
    df_activities.loc[through_entry, "end_time"]       = df_activities.loc[through_entry, "person_id"].map(entry_end_time)
    df_activities.loc[through_entry, "duration"]       = 1
    df_activities.loc[through_exit, "start_time"]       = df_activities.loc[through_exit, "person_id"].map(exit_start_time)
    df_activities.loc[through_exit, "end_time"]         = df_activities.loc[through_exit, "person_id"].map(exit_end_time)
    df_activities.loc[through_exit, "duration"]         = 1
    df_activities.loc[through_exit, "following_mode"]   = df_activities.loc[through_exit, "person_id"].map(through_trip["mode"])

    origin_bookend = pd.DataFrame({
        "person_id": through_population["person_id"].values,
        "activity_index": 0.5,
        "label": "Through",
        "start_time": 0.0,
        "end_time": through_population["person_id"].map(origin_end_time).values,
        "purpose": "other",
        "is_last": False,
        "geometry": gpd.points_from_xy(through_population["origin_point_x"], through_population["origin_point_y"]),
        "destination_id": -1,
        "following_mode": through_population["person_id"].map(through_trip["mode"]).values,
    })
    origin_bookend["duration"] = origin_bookend["end_time"] - origin_bookend["start_time"]

    destination_bookend = pd.DataFrame({
        "person_id": through_population["person_id"].values,
        "activity_index": 2.5,
        "label": "Through",
        "start_time": through_population["person_id"].map(exit_end_time).values,
        "end_time": 30 * 3600,
        "purpose": "other",
        "is_last": True,
        "geometry": gpd.points_from_xy(through_population["destination_point_x"], through_population["destination_point_y"]),
        "destination_id": -1,
        "following_mode": None,
    })
    destination_bookend["duration"] = destination_bookend["end_time"] - destination_bookend["start_time"]

    return pd.concat([df_activities, origin_bookend, destination_bookend], ignore_index = True)


def configure(context):
    context.config("random_seed")
    context.config("output_path")

    context.stage("data.cross_border.destinations")
    context.stage("data.cross_border.network_projection")
    context.stage("data.microcensus.trips")


def execute(context):
    df                = context.stage("data.cross_border.destinations").copy()
    network_projection = context.stage("data.cross_border.network_projection")
    mz_trips          = context.stage("data.microcensus.trips")[0].copy()

    population = df[["cross_border_person_id", "label", "mz_person_id",
                     "origin_x", "origin_y",
                     "destination_x", "destination_y", "destination_id",
                     "residence_x", "residence_y",
                     "destination_residence_x", "destination_residence_y",
                     "is_border_point_projected", "origin_is_projected", "destination_is_projected",
                     "interview_place", "interview_point_id", "interview_geometry_point",
                     "entry_interview_point_id", "entry_interview_geometry_point",
                     "exit_interview_point_id", "exit_interview_geometry_point"]]

    # data.cross_border.network_projection refines the teleported ("projected")
    # origin (all labels) and destination ("Through" only - see that stage's
    # docstring) onto a real point on the MATSim car network, closer to an
    # actual road entry point than the raw survey interview place coordinate.
    # Used below by rename_projected_ends ("From-To" home relocation) and
    # build_through_ends ("Through" origin/destination bookends).
    population = merge_refined_points(
        population, network_projection,
        "residence_x", "residence_y", "origin_x", "origin_y",
        "refined_origin_x", "refined_origin_y",
    )
    population = merge_refined_points(
        population, network_projection,
        "destination_residence_x", "destination_residence_y", "destination_x", "destination_y",
        "refined_destination_x", "refined_destination_y",
    )

    # The point each "Through" bookend activity sits at: the refined point
    # when one was found, else the raw origin_x/destination_x (the interview
    # place) directly. network_projection only ever computes a refined point
    # for a projected end in the first place, so falling back to origin_x/
    # destination_x here is correct in every case it's reached: a
    # non-projected end (where origin_x/destination_x already equals the
    # real residence_x/destination_residence_x point), a projected PT end
    # (deliberately never refined - see network_projection's docstring), and
    # a projected car end network_projection couldn't find a route for.
    # Falling back to residence_x/destination_residence_x instead would be
    # wrong for the PT/no-route cases: that's the real, far-from-the-border
    # point project_point_series_close_to_border decided NOT to use directly
    # for exactly that reason.
    population["origin_point_x"] = population["refined_origin_x"].where(
        population["refined_origin_x"].notna(), population["origin_x"],
    )
    population["origin_point_y"] = population["refined_origin_y"].where(
        population["refined_origin_y"].notna(), population["origin_y"],
    )
    population["destination_point_x"] = population["refined_destination_x"].where(
        population["refined_destination_x"].notna(), population["destination_x"],
    )
    population["destination_point_y"] = population["refined_destination_y"].where(
        population["refined_destination_y"].notna(), population["destination_y"],
    )

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
    # across the country, with direction-specific facility IDs. (build_through_ends,
    # called below, then bookends these with real origin/destination activities
    # and shrinks these two to a near-instant pass-through.)
    through_entry = (df_activities["label"] == "Through") & (df_activities["activity_index"] == 1)
    through_exit = (df_activities["label"] == "Through") & df_activities["following_mode"].isna()

    through_info = df_trips.loc[df_trips["label"] == "Through", [
        "person_id", "entry_interview_point_id", "entry_interview_geometry_point",
        "exit_interview_point_id", "exit_interview_geometry_point",
    ]]
    df_activities = df_activities.merge(through_info, on = "person_id", how = "left")

    df_activities.loc[through_entry, "purpose"] = "border"
    df_activities.loc[through_entry, "destination_id"] = df_activities.loc[through_entry, "entry_interview_point_id"]
    df_activities.loc[through_entry, "geometry"] = df_activities.loc[through_entry, "entry_interview_geometry_point"]
    df_activities.loc[through_exit, "purpose"] = "border"
    df_activities.loc[through_exit, "destination_id"] = df_activities.loc[through_exit, "exit_interview_point_id"]
    df_activities.loc[through_exit, "geometry"] = df_activities.loc[through_exit, "exit_interview_geometry_point"]

    df_activities = df_activities.drop(columns = ["entry_interview_point_id", "entry_interview_geometry_point",
                                                   "exit_interview_point_id", "exit_interview_geometry_point"])

    df_activities["duration"] = df_activities["end_time"] - df_activities["start_time"]
    df_activities["is_last"]  = df_activities["following_mode"].isna()

    df_activities = df_activities[["person_id", "activity_index", "label",
                                   "start_time", "end_time", "duration",
                                   "purpose", "is_last",
                                   "geometry", "destination_id", "following_mode"
                                   ]]

    df_activities = rename_projected_ends(df_activities, population)
    df_activities = build_through_ends(df_activities, df_trips, population)

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

        if len(df) == 0:
            # apply() over an empty frame gives back a DataFrame rather than a
            # Series, which the single-column assignments below cannot take.
            return df.drop(columns = [geom_before, geom_after, end_time_col, trip_duration_col])

        df["dist_before"] = df.apply(lambda r: r[geom_before].distance(r["geometry"]), axis=1)
        df["dist_after"]  = df.apply(lambda r: r[geom_after].distance(r["geometry"]),  axis=1)
        df["ratio"]       = df["dist_before"] / (df["dist_before"] + df["dist_after"])

        df["start_time"]  = df[end_time_col] + df[trip_duration_col] * df["ratio"]
        df["end_time"]    = df["start_time"] + df["duration"]

        return df.drop(columns=["dist_before", "dist_after", "ratio", geom_before, geom_after,
                                end_time_col, trip_duration_col])

    # ── 4. Build fake activities at index 1.5 (every mode) ─────────────────────────
    # The interview point each record carries is the one that serves its own mode
    # (road points for car and its passengers, pt points for public transport,
    # see sample_point in data.cross_border.generate_od), so public transport
    # agents get their border activity the same way car agents do. Uses
    # entry_interview_geometry_point rather than the plain interview_geometry_point
    # (they're the same coordinate for every non-"Through" row - see
    # data.cross_border.destinations) because its geometry has to match the
    # entry facility's own registered coordinate, not the generic interview
    # point, or MATSim's ScenarioValidator rejects the scenario ("facility and
    # activity do not have same coordinates").
    #
    # "Through" persons are always skipped here: their border crossings and
    # origin/destination bookends are built by build_through_ends instead
    # (called above), which already handles both the projected and
    # non-projected case - inserting a 1.5/2.5 activity here too would
    # duplicate that. "From-To" persons are never skipped, projected or not:
    # rename_projected_ends leaves their home/other activity as a real
    # activity (just relocated onto the refined point when projected - see
    # its docstring), so they always need this actual "border" activity
    # inserted next to it to model the crossing.
    mask_border_activity = population["label"] != "Through"

    activities1point5 = (
        population[mask_border_activity].copy()
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
        population[population["cross_border_person_id"].isin(persons_with_3_acts) & mask_border_activity]
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
