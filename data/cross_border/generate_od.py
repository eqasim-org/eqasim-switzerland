import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import random
from pathlib import Path
import logging

logger = logging.getLogger("synpp")

# Which border crossing points serve which mode. "label" comes from
# data.cross_border.interview_places; car passengers cross with their driver.
MODE_TO_LABEL = {"car": "road", "car_passenger": "road", "pt": "pt"}


def configure(context):
    context.config("data_path")
    context.config("random_seed")
    context.config("specific_day_scenario", default = "workday")

    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.swiss_border")
    context.stage("data.cross_border.interview_places")

    context.config("cross_border_countries", default = "All")
    context.config("cross_border_exclude_shapefiles", default=None)


def sample_rows_by_weight(df2, rng, weight_col="weight"):
    df = df2.copy()

    # Separate integer and fractional parts
    df["int_part"]  = df[weight_col].astype(int)
    df["frac_part"] = df[weight_col] - df["int_part"]

    # Repeat rows according to integer part
    repeated = df.iloc[np.repeat(np.arange(len(df)), df["int_part"])].copy().drop(columns=["int_part", "frac_part"])

    # Handle fractional part with Bernoulli sampling (rng is seeded from
    # random_seed so that the generated population is reproducible)
    fractional_mask = rng.random_sample(len(df)) < df["frac_part"]
    fractional = df[fractional_mask].drop(columns=["int_part", "frac_part"])
    
    # Combine both
    sampled_df = pd.concat([repeated, fractional], ignore_index=True)
    return sampled_df


def sample_points_in_polygon(polygon, n):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < n:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            points.append(p)
    return points


def sample_candidates_by_importance(candidates, n, rng):
    """
    Draws n candidates (with replacement), weighted by their importance -
    same process read_2021_data.sample_point uses to match a non-projected
    respondent to their interview point. Falls back to a uniform draw when
    the weights are degenerate (missing/non-finite, or all zero), same as
    data.cross_border.destinations.sample_destinations.
    """

    weights = candidates["importance"]

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = None

    return candidates.sample(n = n, replace = True, weights = weights, random_state = rng)


def project_point_series_close_to_border(df, x, y, distance_threshold, default_purpose, projected_purpose, column_name, context, rng, mode_column = "trip_mode", interview_place_column = "interview_place"):
    df = df.copy()
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df[x], df[y]), crs = "EPSG:4326").to_crs("EPSG:2056")
    points["record"] = range(len(points))

    # A projected point becomes that agent's border activity, so it has to be a
    # crossing its own mode can use.
    points["label"] = df[mode_column].map(MODE_TO_LABEL).values if mode_column in df.columns else None

    # ... and, where available, at the interview_place the respondent was
    # actually surveyed at (data.cross_border.interview_places can hold
    # several points under the same interview_place, e.g. "Bardonnex (625)"
    # covers 2 - so this still leaves an importance-weighted choice among those).
    points["interview_place"] = df[interview_place_column].values if interview_place_column in df.columns else None

    ch_borders        = context.stage("data.spatial.swiss_border").copy()[0]
    ch_borders_simple = ch_borders.simplify(50)

    points["dist_to_border"] = points.geometry.apply(lambda g: g.distance(ch_borders_simple)) / 1000
    close_mask               = points["dist_to_border"] < distance_threshold
    far_mask                 = ~close_mask

    far_points   = points[far_mask].copy()
    close_points = points[close_mask].copy()

    # Points too far from the border are projected onto a surveyed interview
    # place instead of the nearest other observed (and already close) record.
    interview_points = context.stage("data.cross_border.interview_places").copy()[
        ["geometry", "border_crossing_point_id", "label", "interview_place", "importance"]].reset_index(drop = True)
    interview_points["record"] = range(len(interview_points))
    merging_aux_df    = interview_points.copy().rename(columns = {"geometry": "close_point_geometry"})

    # A weighted-by-importance draw among the crossings serving the agent's
    # mode, preferring candidates at the same reported interview_place
    # (falling back to every point of the right mode, then to every point at
    # all, if none of those match) - same process read_2021_data.sample_point
    # uses to match a non-projected respondent to their interview point, so
    # that a pt agent does not end up projected onto a motorway crossing,
    # and a respondent surveyed at e.g. "Bardonnex" is projected onto a
    # Bardonnex point rather than some other, merely closer, crossing.
    nearest_parts = []

    for (label, interview_place), group in far_points.groupby(["label", "interview_place"], dropna = False):
        candidates = interview_points[
            (interview_points["label"] == label) & (interview_points["interview_place"] == interview_place)
        ]

        if len(candidates) == 0:  # no point at this interview_place for this mode: fall back to mode alone
            candidates = interview_points[interview_points["label"] == label]

        if len(candidates) == 0:  # unknown mode: fall back to every crossing
            candidates = interview_points

        sampled = sample_candidates_by_importance(candidates, len(group), rng)

        part = group.copy()
        part["record_left"]  = part["record"]
        part["record_right"] = sampled["record"].values
        nearest_parts.append(part)

    nearest = pd.concat(nearest_parts) if len(nearest_parts) > 0 else far_points.assign(record_left = None, record_right = None)

    if "index_right" in nearest: del nearest["index_right"]
    nearest = pd.merge(nearest, merging_aux_df, left_on = "record_right", right_on = "record", how = "left")
    nearest = nearest[["record_left", "dist_to_border", "close_point_geometry", "geometry", "border_crossing_point_id"]]
    nearest.columns = ["record", "dist_to_border", "geometry", "geometry_before_projection", "point_id"]
    nearest["purpose"]      = projected_purpose
    nearest["is_projected"] = True

    close_points["geometry_before_projection"] = close_points["geometry"]
    close_points["purpose"]             = default_purpose
    close_points["is_projected"]        = False
    close_points["point_id"]            = None  # a real location, not a crossing

    points = pd.concat([nearest, close_points])
    points = points.sort_values(by = "record")

    df[column_name + "_point"]             = points["geometry"].values
    df[column_name + "_before_projection"] = points["geometry_before_projection"].values
    df[column_name + "_purpose"]           = points["purpose"].values
    df[column_name + "_is_projected"]      = points["is_projected"].values
    df[column_name + "_point_id"]          = points["point_id"].values

    del df[x]
    del df[y]

    df[column_name +  "_x"] = df[column_name + "_point"].apply(lambda p : p.x)
    df[column_name +  "_y"] = df[column_name + "_point"].apply(lambda p : p.y)

    return df


def expand_and_sample(df, expand_column, weight_column, rng):
    df = df.copy()

    # Expand: one row per occupant of the vehicle / member of the group
    df_expanded = df.loc[df.index.repeat(df[expand_column])].copy()
    df_expanded["passenger_index"] = df_expanded.groupby(df_expanded.index).cumcount() + 1
    df_expanded["trip_mode"]       = assign_car_passengers(df_expanded["trip_mode"], df_expanded["passenger_index"])

    del df_expanded[expand_column]
    del df_expanded["passenger_index"]

    # Sample
    df_sampled = sample_rows_by_weight(df_expanded, rng, weight_col = weight_column)
    del df_sampled[weight_column]

    return df_sampled.copy().reset_index()


def sjoin_within_unique(points, polygons):
    """
    Point-in-polygon join returning exactly one row per point, in the order of
    `points` (a GeoSeries). A plain sjoin returns one row per match, and these
    layers do overlap: data.spatial.cantons and data.spatial.municipalities
    append the external-population region as an extra polygon covering the real
    ones, so every point inside it matches twice. That silently breaks the
    positional `.values` assignments the callers do afterwards. The region is
    appended last, so keeping the first match keeps the real canton /
    municipality.
    """

    points = gpd.GeoDataFrame(geometry = points.reset_index(drop = True))

    joined = gpd.sjoin(points, polygons, how = "left", predicate = "within")
    joined = joined[~joined.index.duplicated(keep = "first")]

    return joined.reindex(points.index)


def assign_car_passengers(trip_mode, passenger_index):
    """
    In a car, only the first occupant drives; everyone else rides along. The
    survey codes the whole group with a single vehicle type ("car" here, since
    FAHRZEUGTYP is mapped to car/pt when the data is read), so the passengers
    have to be derived from their position within the expanded group.
    """

    return np.where((trip_mode == "car") & (passenger_index > 1), "car_passenger", trip_mode)


def process_from_to_trips(df_trips, context, rng):
    # Load municipalities
    df_municipalities, _ = context.stage("data.spatial.municipalities")

    # 1. Remove "through" trips that were not classified properly
    trips    = df_trips[(df_trips["origin_country"]=="CH") | (df_trips["destination_country"]=="CH")].copy()
    trips_od = trips[["origin_country", "destination_country", "origin_country_raw", "destination_country_raw",
        "start_x", "start_y", "end_x", "end_y", "trip_mode", "trip_purpose", "weight", "nb_passengers",
        "interview_place", "interview_point_id", "interview_geometry_point"]].copy()

    # 2. Remove trips with missing information on start or end point
    mask_missing_start = pd.to_numeric(trips_od["start_x"], errors="coerce").isna() # If start_x is missing, so is origin_place, so we cannot use one value to compensate the absence of the other.
    mask_missing_end   = pd.to_numeric(trips_od["end_x"], errors="coerce").isna()   # Same with destinations
    
     # This removes 1.49% of the records
    df = trips_od[~(mask_missing_start) & ~(mask_missing_end)].copy()

    # Reorder start and end so that all trips end in CH. The raw country codes
    # have to be swapped along with everything else, otherwise the records
    # interviewed on the way out of Switzerland keep a reversed country pair
    # (which is what matsim/scenario/population.py builds cross_border_od from).
    mask = df["origin_country"] == "CH"
    df.loc[mask, ["origin_country", "destination_country"]] = df.loc[mask, ["destination_country", "origin_country"]].values
    df.loc[mask, ["origin_country_raw", "destination_country_raw"]] = df.loc[mask, ["destination_country_raw", "origin_country_raw"]].values
    df.loc[mask, ["start_x", "end_x"]] = df.loc[mask, ["end_x", "start_x"]].values
    df.loc[mask, ["start_y", "end_y"]] = df.loc[mask, ["end_y", "start_y"]].values

    # Prepare to sample points from destination municipality
    destinations = df.copy().apply(lambda row: Point(row["end_x"], row["end_y"]), axis = 1)
    destinations = gpd.GeoSeries(destinations, crs = "EPSG:4326").to_crs("EPSG:2056")

    joined = sjoin_within_unique(destinations, df_municipalities)

    df["destination_municipality"] = joined["municipality_id"].values

    # In 23 cases, corresponding mostly to people going to Liechtenstein or to points exactly on the border
    # in le Locle or Saint-Gingolph, the municipality cannot be found. 
    # Let's remove these observations.
    df = df[df["destination_municipality"].notna()].copy()

    df = expand_and_sample(df.copy(), "nb_passengers", "weight", rng)

    # Fix the origins
    df = project_point_series_close_to_border(df.copy(), "start_x", "start_y", 20, "home", "other", "origin", context, rng)

    # Re-create the destinations
    destinations = df.copy().apply(lambda row: Point(row["end_x"], row["end_y"]), axis = 1)
    destinations = gpd.GeoSeries(destinations, crs = "EPSG:4326").to_crs("EPSG:2056")

    df["destination_x"] = destinations.apply(lambda p : p.x)
    df["destination_y"] = destinations.apply(lambda p : p.y)

    df["cross_border_person_id"] = range(len(df))
    df["cross_border_person_id"] = "CBS_" + df["cross_border_person_id"].astype(str)

    df["residence_x"] =  df["origin_before_projection"].apply(lambda p: p.x)
    df["residence_y"] =  df["origin_before_projection"].apply(lambda p: p.y)

    df["label"] = "From-To"

    # Only the origin (residence) can be projected here, the destination is always a real
    # STATENT-sampled point inside Switzerland (see data.cross_border.destinations).
    df["destination_is_projected"]  = False
    df["destination_point_id"]      = None
    df["is_border_point_projected"] = df["origin_is_projected"]

    df = df[["cross_border_person_id", "label",
        "origin_x", "origin_y", "destination_x", "destination_y",
        "residence_x", "residence_y",
        "trip_mode", "trip_purpose",
        "is_border_point_projected", "origin_is_projected", "destination_is_projected",
        "origin_point_id", "destination_point_id",
        "interview_place", "interview_point_id", "interview_geometry_point",
        "origin_country", "destination_country", "origin_country_raw", "destination_country_raw"]]

    return df


def process_through_trips(through_trips, N, context, rng):
    through_od = through_trips[
        ["origin_country", "destination_country", "origin_country_raw", "destination_country_raw",
        "start_x", "start_y", "end_x", "end_y", "trip_mode", "trip_purpose", "weight", "nb_passengers",
        "interview_place", "interview_point_id", "interview_geometry_point"]
    ]

    mask_missing_start = pd.to_numeric(through_od["start_x"], errors="coerce").isna()  # If start_x is missing, so is origin_place, so we cannot use one value to compensate the absence of the other.
    mask_missing_end   = pd.to_numeric(through_od["end_x"], errors="coerce").isna()

    df = through_od[~(mask_missing_start) & ~(mask_missing_end)].copy() # This removes 9% / 11.3% of the records (unweighted/weighted)

    df_expanded = df.loc[df.index.repeat(df["nb_passengers"])].copy()
    df_expanded["passenger_index"] = df_expanded.groupby(df_expanded.index).cumcount() + 1
    df_expanded["trip_mode"]       = assign_car_passengers(df_expanded["trip_mode"], df_expanded["passenger_index"])

    del df_expanded["nb_passengers"]
    del df_expanded["passenger_index"]

    df_sampled = sample_rows_by_weight(df_expanded, rng, weight_col = "weight")
    del df_sampled["weight"]

    df = df_sampled.copy().reset_index()

    df = project_point_series_close_to_border(df.copy(), "start_x", "start_y", 20, "other", "other", "origin", context, rng)
    df = project_point_series_close_to_border(df.copy(), "end_x", "end_y", 20, "other", "other", "destination", context, rng)

    df["cross_border_person_id"] = range(N, N + len(df))
    df["cross_border_person_id"] = "CBS_" + df["cross_border_person_id"].astype(str)

    df["residence_x"] =  df["origin_before_projection"].apply(lambda p: p.x)
    df["residence_y"] =  df["origin_before_projection"].apply(lambda p: p.y)

    df["label"] = "Through"

    # Either end of a through-trip can be projected, since neither is a real point inside CH.
    df["is_border_point_projected"] = df["origin_is_projected"] | df["destination_is_projected"]

    df = df[["cross_border_person_id", "label",
        "origin_x", "origin_y", "destination_x", "destination_y",
        "residence_x", "residence_y",
        "trip_mode", "trip_purpose",
        "is_border_point_projected", "origin_is_projected", "destination_is_projected",
        "origin_point_id", "destination_point_id",
        "interview_place", "interview_point_id", "interview_geometry_point",
        "origin_country", "destination_country", "origin_country_raw", "destination_country_raw"]]

    return df


def read_2021_data(context):
    # Load data
    # We are using the 2021 release because the 2015 one doesn't provide reliable destination coordinates.

    data_path = context.config("data_path")
    data_path = f"{data_path}/crossborder/AuGQPV_2021/AGQPV21_finale_Auswertungsdatenbank.csv"

    df2021 = pd.read_csv(data_path, encoding="latin1", sep = ";")

    df2021 = df2021[["INTERVIEWID", "BEFRAGUNGSORTID", "BEFRAGUNGSORT", "GRENZABSCHNITT", "TAGESTYP", "VERKEHRSTRAEGER", "UEBERGANGSART", "FAHRZEUGTYP", "GRUPPENGROESSE",
                 "WOHNORTLANDISO", "WOHNORT_GISCO_ID", "STARTORTLANDISO", "STARTORT_GISCO_ID", "ZIELORTLANDISO", "ZIELORT_GISCO_ID",
                 "STARTORTORTLATITUDE", "STARTORTORTLONGITUDE",
                 "ZIELORTORTLATITUDE", "ZIELORTORTLONGITUDE",
                 "FAHRTZWECK", "ANZAHLUEBERNACHTUNGEN", "AUFENTHALTSLAND1ISO", "AUFENTHALTSLAND2ISO", "AUFENTHALTSLAND3ISO",
                 "ZUGTYP", "FAHRTRICHTUNGGU", "FAHRTRICHTUNGAU", "VERKEHRSART", "GEWICHT_Personen", "GEWICHT_Fahrzeuge"]]

    df2021.columns = ["interview_id", "interview_place_id", "interview_place", "neighbor_country", "day_cat", "road_type", "crossing_cat", "vehicle_type", "nb_passengers",
                  "residence_country", "residence_place", "origin_country", "origin_place", "destination_country", "destination_place",
                  "start_y", "start_x", "end_y", "end_x", 
                  "trip_purpose", "nb_nights", "country1", "country2", "country3",
                  "train_type", "direction_crossing", "direction_alps", "travel_cat", "weight", "weight_vehicles"]
    
    # Process the columns
    # 1. Rename countries
    # Keep the unprocessed country codes around, since the grouping below collapses
    # everything outside of swiss_neighbors into "other".
    df2021["origin_country_raw"]      = df2021["origin_country"]
    df2021["destination_country_raw"] = df2021["destination_country"]

    swiss_neighbors = ["CH", "FR", "DE", "IT", "AT", "LI"]
    for column in ["residence_country", "origin_country", "destination_country"]:
        df2021.loc[:, column] = df2021[column].apply(lambda x: x if x in swiss_neighbors else "other")
    
    # 2. Separate road and rail observations
    df2021["road_type"] = df2021["road_type"].astype(str)
    df2021.loc[df2021["road_type"]=="1", "road_type"] = "road"
    df2021.loc[df2021["road_type"]=="2", "road_type"] = "rail"
    
    # 3. Identify vehicles
    df2021["vehicle_type"] = df2021["vehicle_type"].astype(str)
    df2021.loc[df2021["vehicle_type"]=="1", "vehicle_type"] = "car"
    df2021.loc[df2021["vehicle_type"]=="2", "vehicle_type"] = "car" #"motorcycle"
    df2021.loc[df2021["vehicle_type"]=="3", "vehicle_type"] = "pt"  #"long distance bus"
    
    # 4. Identify trip purpose
    df2021["trip_purpose"] = df2021["trip_purpose"].astype(str)
    df2021.loc[df2021["trip_purpose"]=="1", "trip_purpose"] = "work"
    df2021.loc[df2021["trip_purpose"]=="2", "trip_purpose"] = "education"
    df2021.loc[df2021["trip_purpose"]=="3", "trip_purpose"] = "shop"
    df2021.loc[df2021["trip_purpose"]=="4", "trip_purpose"] = "work"
    df2021.loc[df2021["trip_purpose"]=="5", "trip_purpose"] = "leisure"
    df2021.loc[df2021["trip_purpose"]=="6", "trip_purpose"] = "other"
    df2021.loc[df2021["trip_purpose"]=="7", "trip_purpose"] = "freight"
    
    # 5. Is the respondent entering or leaving CH?
    df2021["direction_crossing"] = df2021["direction_crossing"].astype(str)
    df2021.loc[df2021["direction_crossing"]=="1", "direction_crossing"] = "entering CH"
    df2021.loc[df2021["direction_crossing"]=="2", "direction_crossing"] = "leaving CH"
    
    # 6. For the Alps crossing, not used currently
    df2021["direction_alps"] = df2021["direction_alps"].astype(str)
    df2021.loc[df2021["direction_alps"]=="1", "direction_alps"] = "North"
    df2021.loc[df2021["direction_alps"]=="2", "direction_alps"] = "South"
    
    # 7. More detailed compared to direction_crossing, but obviously there are inconsistencies betweeen these two columns
    df2021["travel_cat"] = df2021["travel_cat"].astype(str)
    df2021.loc[df2021["travel_cat"]=="1", "travel_cat"] = "Within CH"
    df2021.loc[df2021["travel_cat"]=="2", "travel_cat"] = "Through CH"
    df2021.loc[df2021["travel_cat"]=="3", "travel_cat"] = "From CH"
    df2021.loc[df2021["travel_cat"]=="4", "travel_cat"] = "To CH"
    
    # 8. Identify the observation day
    df2021.loc[df2021["day_cat"]=="Werktag", "day_cat"]    = "Mo-Fr"
    df2021.loc[df2021["day_cat"]=="Samstag", "day_cat"]    = "WE"
    df2021.loc[df2021["day_cat"]=="Sonntag", "day_cat"]    = "WE"
    df2021.loc[df2021["day_cat"]=="Wochenende", "day_cat"] = "WE"
    
    # 9. Identify the train category for rail observations
    df2021.loc[df2021["train_type"]=="1", "train_type"] = "pt" #"long distance train"
    df2021.loc[df2021["train_type"]=="2", "train_type"] = "pt" #"regional train"

    # 10. Aggregate vehicle_type and train_type
    df2021["trip_mode"] = np.where(df2021["vehicle_type"].str.strip() != '', df2021["vehicle_type"], df2021["train_type"])

    df2021["weight"]          = df2021["weight"].astype(float)
    df2021["weight_vehicles"] = df2021["weight_vehicles"].replace("", 0).replace(" ", 0).fillna(0).astype(float)
    df2021.loc[df2021["road_type"]=="road", "weight"] = 2 * df2021[df2021["road_type"]=="road"]["weight_vehicles"] 
    
    df2021.loc[:, "group_weight"] = df2021["weight"]
    df2021.loc[df2021["road_type"]=="road", "group_weight"] = df2021[df2021["road_type"]=="road"]["weight"] * df2021[df2021["road_type"]=="road"]["nb_passengers"]
    df2021["group_weight"] = df2021["group_weight"].astype(float)
    
    del df2021["road_type"]
    del df2021["vehicle_type"]
    del df2021["train_type"]

    # 11. Adjust weight
    days    = {"Mo-Fr": 5, "WE": 2}

    day = context.config("specific_day_scenario")

    if day == "weekend":
        day_key = "WE"
    elif day == "workday":
        day_key = "Mo-Fr"
    elif day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        # The survey only distinguishes Mo-Fr from the weekend, so a single
        # weekday gets the regular workday demand (and a single weekend day
        # the regular weekend demand).
        day_key = "Mo-Fr"
    elif day in ["Saturday", "Sunday"]:
        day_key = "WE"
    else:
        raise ValueError(f"Unsupported specific_day_scenario: '{day}'")

    day_value = days[day_key]
    df_days   = df2021[df2021["day_cat"]==day_key].copy()
    df_days["weight"] = df_days["weight"] / (52 * day_value)
    df_days["weight"] = df_days["weight"] / 2 # Because the persons entering the country have to leave it too
    df_days["group_weight"] = df_days["group_weight"] / (52 * day_value)
    df_days["group_weight"] = df_days["group_weight"] / 2
    del df_days["day_cat"]

    # 12. Only select border crossing data - remove Alps crossing data
    borders = df_days[df_days["crossing_cat"]==1].copy()
    
    del borders["crossing_cat"]
    del borders["direction_alps"]

    # 13. Remove Swiss residents, their mobility should be covered in the Microcensus
    residents_ch_mask = borders["residence_country"] == "CH"
    borders = borders[~ residents_ch_mask].copy()

    for col in ["start_x", "start_y", "end_x", "end_y"]:
        borders[col] = pd.to_numeric(borders[col].str.strip(), errors="coerce")

    borders = borders[borders["start_x"].notna() & borders["start_y"].notna()]
    borders = borders[borders["end_x"].notna() & borders["end_y"].notna()]

    # 14. Match to an interview point    
    #borders.loc[borders["interview_place"] == "Vallorbe (544)", "interview_place"] = "Gruppe O: VD/NE -> F. Dép. Jura. Doubs"

    points = context.stage("data.cross_border.interview_places").copy()[["interview_place", "border_crossing_point_id", "geometry", "importance", "label"]]
    points["interview_point_id"] = points["border_crossing_point_id"]

    # "label" indicates which mode a point was surveyed for ("road" serves car trips,
    # "pt" serves public transport trips), so candidates are grouped by place AND mode.
    mode_to_label = MODE_TO_LABEL

    grouped_points = {k: v for k, v in points.groupby(["interview_place", "label"])}

    # Shared, seeded RNG: passing it to every .sample() call keeps the draws
    # reproducible while still giving each row its own draw.
    point_rng = np.random.RandomState(context.config("random_seed"))

    def sample_point(row):
        label = mode_to_label.get(row["trip_mode"])
        candidates = grouped_points.get((row["interview_place"], label))
        if candidates is not None and "importance" in candidates.columns:
            sampled = candidates.sample(n=1, weights=candidates["importance"], random_state=point_rng)
            return sampled.iloc[0][["geometry", "importance", "interview_point_id", "label"]]

        # Fall back to the closest point that still matches the trip's mode, if any exist
        same_label = points[points["label"] == label] if label is not None else points
        if len(same_label) == 0:
            same_label = points

        origin    = Point(row["start_x"], row["start_y"])
        distances = same_label["geometry"].apply(lambda geom: origin.distance(geom))
        closest   = same_label.loc[distances.idxmin()]
        return closest[["geometry", "importance", "interview_point_id", "label"]]

    result = borders[["interview_place", "start_x", "start_y", "trip_mode"]].apply(sample_point, axis=1)
    borders[["interview_geometry_point", "importance", "interview_point_id", "interview_point_label"]] = result

    # The point has to serve the mode the agent travels with, since it becomes
    # that agent's border activity in data.cross_border.activities: car (and
    # its passengers) cross at a road point, public transport at a pt one. That
    # holds by construction above, including in the fallback, so this only
    # catches a mode that mode_to_label does not know about.
    expected_label = borders["trip_mode"].map(mode_to_label)
    mismatched     = expected_label.notna() & (borders["interview_point_label"] != expected_label)

    assert not mismatched.any(), (
        "%d records got a border crossing point that does not serve their mode, e.g. %s"
        % (int(mismatched.sum()),
           borders.loc[mismatched, ["trip_mode", "interview_point_label"]].head().to_dict("records"))
    )

    logger.info("Border crossing points by mode: %s",
                borders.groupby(["trip_mode", "interview_point_label"]).size().to_dict())

    return borders


def read_2015_data(context):
    data_path = context.config("data_path")
    data_path = f"{data_path}/crossborder/AuGQPV_2015/Finale_Auswertungsdatenbank_AGQPV2015_V2.csv"

    df2015   = pd.read_csv(data_path, encoding = "latin1", sep = ",")

    df2015   = df2015[["INTERVIEWID", "BEFRAGUNGSORTID", "BEFRAGUNGSORT", "GRENZABSCHNITT", "TAGESTYP", 
                    "VERKEHRSTRAEGER", "UEBERGANGSART", "FAHRZEUGTYP", "GRUPPENGROESSE", 
                    "WOHNORTLANDISO", "STARTORTLANDISO", "ZIELORTLANDISO", 
                    "STARTORTORTLATITUDE", "STARTORTORTLONGITUDE",
                    "ZIELORTORTLATITUDE","ZIELORTORTLONGITUDE",
                    "FAHRTZWECK", "ANZAHLUEBERNACHTUNGEN",
                    "AUFENTHALTSLAND1ISO", "AUFENTHALTSLAND2ISO", "AUFENTHALTSLAND3ISO",
                    "ZUGTYP", "FAHRTRICHTUNGGU", "FAHRTRICHTUNGAU", "VERKEHRSART", "GEWICHT"]]
    
    df2015.columns = ["interview_id", "interview_place_id", "interview_place", "neighbor_country", "day_cat", 
                  "road_type", "crossing_cat", "vehicle_type", "nb_passengers",
                  "residence_country", "origin_country", "destination_country", 
                  "start_y", "start_x", 
                  "end_y", "end_x", 
                  "trip_purpose", "nb_nights", 
                  "country1", "country2", "country3",
                  "train_type", "direction_crossing", "direction_alps", "travel_cat", "weight"]
    
    swiss_neighbors = ['CH', 'FR', 'DE', 'IT', 'AT', 'LI']
    for column in ["residence_country", "origin_country", "destination_country"]:
        df2015.loc[:, column] = df2015[column].apply(lambda x: x if x in swiss_neighbors else "other")

    df2015["road_type"] = df2015["road_type"].astype(str)
    df2015.loc[df2015["road_type"]=="1", "road_type"] = "road"
    df2015.loc[df2015["road_type"]=="2", "road_type"] = "rail"

    df2015["vehicle_type"] = df2015["vehicle_type"].astype(str)
    df2015.loc[df2015["vehicle_type"]=="1", "vehicle_type"] = "car"
    df2015.loc[df2015["vehicle_type"]=="3", "vehicle_type"] = "car"
    df2015.loc[df2015["vehicle_type"]=="4", "vehicle_type"] = "pt"

    df2015["trip_purpose"] = df2015["trip_purpose"].astype(str)
    df2015.loc[df2015["trip_purpose"]=="1", "trip_purpose"] = "work"
    df2015.loc[df2015["trip_purpose"]=="2", "trip_purpose"] = "education"
    df2015.loc[df2015["trip_purpose"]=="3", "trip_purpose"] = "shop"
    df2015.loc[df2015["trip_purpose"]=="4", "trip_purpose"] = "work"
    df2015.loc[df2015["trip_purpose"]=="5", "trip_purpose"] = "leisure"
    df2015.loc[df2015["trip_purpose"]=="6", "trip_purpose"] = "other"
    df2015.loc[df2015["trip_purpose"]=="7", "trip_purpose"] = "freight"

    df2015["direction_crossing"] = df2015["direction_crossing"].astype(str)
    df2015.loc[df2015["direction_crossing"]=="1", "direction_crossing"] = "entering CH"
    df2015.loc[df2015["direction_crossing"]=="2", "direction_crossing"] = "leaving CH"

    df2015["direction_alps"] = df2015["direction_alps"].astype(str)
    df2015.loc[df2015["direction_alps"]=="1", "direction_alps"] = "North"
    df2015.loc[df2015["direction_alps"]=="2", "direction_alps"] = "South"

    df2015["travel_cat"] = df2015["travel_cat"].astype(str)
    df2015.loc[df2015["travel_cat"]=="1", "travel_cat"] = "Within CH"
    df2015.loc[df2015["travel_cat"]=="2", "travel_cat"] = "Through CH"
    df2015.loc[df2015["travel_cat"]=="3", "travel_cat"] = "From CH"
    df2015.loc[df2015["travel_cat"]=="4", "travel_cat"] = "To CH"
    df2015.loc[~df2015["travel_cat"].isin(["Within CH", "Through CH", "From CH", "To CH"]), "travel_cat"]  = "Unknown"

    df2015.loc[df2015["day_cat"]=="Werktag", "day_cat"]    = "Mo-Fr"
    df2015.loc[df2015["day_cat"]=="Samstag", "day_cat"]    = "WE"
    df2015.loc[df2015["day_cat"]=="Sonntag", "day_cat"]    = "WE"
    df2015.loc[df2015["day_cat"]=="Wochenende", "day_cat"] = "WE"

    df2015.loc[df2015["train_type"]=="FV", "train_type"] = "pt"
    df2015.loc[df2015["train_type"]=="RV", "train_type"] = "pt"

    df2015["trip_mode"] = np.where(df2015["vehicle_type"].str.strip() != '', df2015["vehicle_type"], df2015["train_type"])

    days    = {"Mo-Fr": 5, "WE": 2}

    day = context.config("specific_day_scenario")

    if day == "weekend":
        day_key = "WE"
    elif day == "workday":
        day_key = "Mo-Fr"
    elif day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        # The survey only distinguishes Mo-Fr from the weekend, so a single
        # weekday gets the regular workday demand (and a single weekend day
        # the regular weekend demand).
        day_key = "Mo-Fr"
    elif day in ["Saturday", "Sunday"]:
        day_key = "WE"
    else:
        raise ValueError(f"Unsupported specific_day_scenario: '{day}'")

    day_value = days[day_key]
    df_days   = df2015[df2015["day_cat"]==day_key].copy()
    df_days["weight"] = df_days["weight"] / (52 * day_value)
    df_days["weight"] = df_days["weight"] / 2 # Because the persons entering the country have to leave it too
    del df_days["day_cat"]

    df2015 = df_days.copy()

    df2015.loc[df2015["road_type"]=="road", "weight"] = df2015[df2015["road_type"]=="road"]["weight"] * 2
    df2015.loc[:, "group_weight"] = df2015["weight"] 
    df2015.loc[df2015["road_type"]=="road", "group_weight"] = df2015[df2015["road_type"]=="road"]["weight"] * df2015[df2015["road_type"]=="road"]["nb_passengers"]

    del df2015["vehicle_type"]
    del df2015["train_type"]

    borders2015 = df2015[df2015["crossing_cat"]==1]

    del borders2015["crossing_cat"]
    del borders2015["direction_alps"]

    residents_ch_mask = borders2015["residence_country"] == "CH"
    borders2015 = borders2015[~ residents_ch_mask].copy()

    return borders2015


def execute(context):
    rng = np.random.RandomState(context.config("random_seed"))

    borders2021 = read_2021_data(context)
    borders2015 = read_2015_data(context)

    grouped2021 = borders2021.groupby(["trip_mode", "trip_purpose", "origin_country", "destination_country"], as_index = False)["group_weight"].sum().rename(columns = {"group_weight": "group_weight_2021"})
    grouped2015 = borders2015.groupby(["trip_mode", "trip_purpose", "origin_country", "destination_country"], as_index = False)["group_weight"].sum().rename(columns = {"group_weight": "group_weight_2015"})

    grouped = grouped2021.merge(grouped2015, how = "left", on = ["trip_mode", "trip_purpose", "origin_country", "destination_country"])

    grouped.loc[grouped["group_weight_2015"].isna(), "scaling_factor"]  = 1
    grouped.loc[~grouped["group_weight_2015"].isna(), "scaling_factor"] = grouped[~grouped["group_weight_2015"].isna()]["group_weight_2015"] / grouped[~grouped["group_weight_2015"].isna()]["group_weight_2021"]
    grouped = grouped[["trip_mode", "trip_purpose", "origin_country", "destination_country", "scaling_factor"]]

    borders = borders2021.merge(
        grouped,
        on=["trip_mode", "trip_purpose", "origin_country", "destination_country"],
        how="left"
    )

    borders["weight"] = (
        borders["weight"] *
        borders["scaling_factor"]
    )

    borders["group_weight"] = (
        borders["group_weight"] *
        borders["scaling_factor"]
    )

    # Selector by origin country
    allowed_countries  = ["FR", "DE", "AT", "LI", "IT"]
    selected_countries = context.config("cross_border_countries")

    if selected_countries != "All":
        if isinstance(selected_countries, list):
            selected_countries = [c for c in selected_countries if c in allowed_countries]
            if not selected_countries:
                raise ValueError(
                    f"No valid countries in selection. Must be within {allowed_countries}."
                )
            
            borders = borders.loc[borders["origin_country"].within(selected_countries)].copy()

        elif isinstance(selected_countries, str):
            if selected_countries not in allowed_countries:
                raise ValueError(
                    f"Invalid country code '{selected_countries}'. Must be one of {allowed_countries}."
                )
            borders = borders.loc[borders["origin_country"] == selected_countries].copy()

        else:
            raise TypeError("cross_border_countries must be a list, string, or 'All'.")

    # Now process the trips
    trips = borders[borders["travel_cat"].isin(["From CH", "To CH"])]   
    from_to_trips = process_from_to_trips(trips, context, rng)

    through = borders[borders["travel_cat"]=="Through CH"]
    through_trips = process_through_trips(through, len(from_to_trips), context, rng)

    df = pd.concat([from_to_trips, through_trips])
        
    # Remove people who really live in the spatial file to be excluded. This
    # has to be checked against residence_x/residence_y (the real home
    # location, before project_point_series_close_to_border may have snapped
    # a far-away home onto a nearby border crossing point for the "origin"/
    # "destination" columns) - checking origin_x/origin_y instead would catch
    # a teleported person whose real home is nowhere near the excluded area,
    # just because the crossing point their trip got projected onto happens
    # to sit inside it.
    exclude_file = context.config("cross_border_exclude_shapefiles")

    if not exclude_file is None:
        if isinstance(exclude_file, (str, Path)):
            exclude_file = [exclude_file]

        if not isinstance(exclude_file, (list, tuple)):
            raise TypeError(
                "cross_border_exclude_shapefiles must be a path or a list of paths."
            )

        gdfs = []

        for f in exclude_file:
            suffix = Path(f).suffix.lower()
            if suffix not in {".gpkg", ".shp"}:
                raise TypeError(
                    f"{f} is not a .gpkg or .shp file."
                )

            gdf = gpd.read_file(f).to_crs("EPSG:2056")
            gdfs.append(gdf)

        exclude_region = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:2056")
        residences     = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["residence_x"], df["residence_y"]), crs="EPSG:2056")

        joined           = gpd.sjoin(residences, exclude_region[["geometry"]], how = "left", predicate = "within")
        is_within_region = joined["index_right"].notna()
        df["exclude"]    = is_within_region.values

        excluded_ids = df.loc[df["exclude"], "cross_border_person_id"].unique()
        df = df[~df["cross_border_person_id"].isin(excluded_ids)].copy()

        del df["exclude"]

    return df
