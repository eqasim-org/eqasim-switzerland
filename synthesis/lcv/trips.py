
"""
Generate trip list from a float OD matrix using enterprises as origins/destinations.

Steps:
1. Load OD matrix from OMX (via h5py).
2. Turn OD into a long table (origin, destination, value).
3. Sample integer trips from float values with a stochastic (Poisson) approach.
4. Load zone polygons from shapefile.
5. Load enterprise points (df_statent-like), spatially join to zones to get zone IDs.
6. For each trip, randomly choose an origin and destination enterprise (or zone centroid if no enterprise).
7. Return a DataFrame with one row per trip:
   trip_id, origin_x, origin_y, destination_x, destination_y, origin_zone, destination_zone
"""

import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.statent.statent")
    context.stage("synthesis.freight.trips")

    context.stage("data.microcensus.21.trips")
    context.stage("data.microcensus.21.persons")

    context.stage("data.statpop.persons")
    context.config("lcv_home_destination_after", default=8 * 3600)

    context.config("input_downsampling")
    context.config("random_seed")
    context.config("data_path")
    context.config("lcv_poisson_sampling", default=False)
    context.config("use_freight")    

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def build_home_sampling_points(
    context,
    gpkg_path,
    layer_name,
    zone_id_col,
    home_x_col="home_x",
    home_y_col="home_y",
):
    """
    Build zone_id -> array([[home_x, home_y], ...]) using data.statpop.persons.

    Homes are assigned to zones via spatial join based on home_x/home_y.
    No centroid fallback here: if a zone has no homes, it simply has no home candidates.
    """
    if layer_name is None:
        zones_gdf = gpd.read_file(gpkg_path)
    else:
        zones_gdf = gpd.read_file(gpkg_path, layer=layer_name)

    zones_gdf = zones_gdf[[zone_id_col, "geometry"]].copy()
    zones_gdf = zones_gdf.rename(columns={zone_id_col: "zone_id"})
    zones_gdf["zone_id"] = zones_gdf["zone_id"].astype(int)
    zones_gdf = zones_gdf.drop_duplicates(subset="zone_id").reset_index(drop=True)

    valid_zone_ids = set(zones_gdf["zone_id"].unique())

    df_persons = context.stage("data.statpop.persons")
    df_persons = df_persons[[home_x_col, home_y_col]].dropna().copy()

    homes_gdf = gpd.GeoDataFrame(
        df_persons,
        geometry=gpd.points_from_xy(df_persons[home_x_col], df_persons[home_y_col]),
        crs=zones_gdf.crs,
    )

    homes_joined = gpd.sjoin(
        homes_gdf,
        zones_gdf[["zone_id", "geometry"]],
        how="inner",
        predicate="within",
    ).reset_index(drop=True)

    homes_joined["zone_id"] = homes_joined["zone_id"].astype(int)
    homes_joined = homes_joined[homes_joined["zone_id"].isin(valid_zone_ids)].copy()

    homes_joined["x"] = homes_joined.geometry.x
    homes_joined["y"] = homes_joined.geometry.y

    home_zone_to_points = {}
    for zid, grp in homes_joined.groupby("zone_id"):
        home_zone_to_points[int(zid)] = grp[["x", "y"]].to_numpy()

    return home_zone_to_points


def add_home_destinations_after_time(
    trips_df,
    enterprise_zone_to_points,
    home_zone_to_points,
    after_time_sec=8 * 3600,
    seed=None,
):
    """
    For trips departing at/after after_time_sec, allow destinations to be sampled
    from enterprise + home points in the destination zone.

    Before after_time_sec, destinations stay exactly as originally sampled.
    """
    rng = np.random.default_rng(seed)
    out = trips_df.copy()

    eligible = out["departure_time"] >= after_time_sec
    eligible_idx = out.index[eligible].to_numpy()

    if len(eligible_idx) == 0:
        return out

    for dest_zone, idx in out.loc[eligible_idx].groupby("destination_zone").groups.items():
        dest_zone = int(dest_zone)

        enterprise_pts = enterprise_zone_to_points.get(dest_zone)
        home_pts = home_zone_to_points.get(dest_zone)

        if home_pts is None or len(home_pts) == 0:
            continue

        if enterprise_pts is None or len(enterprise_pts) == 0:
            candidate_pts = home_pts
        else:
            candidate_pts = np.vstack([enterprise_pts, home_pts])

        idx = np.asarray(list(idx))
        sampled_idx = rng.integers(0, len(candidate_pts), size=len(idx))
        sampled_pts = candidate_pts[sampled_idx]

        out.loc[idx, "destination_x"] = sampled_pts[:, 0]
        out.loc[idx, "destination_y"] = sampled_pts[:, 1]

    return out


def load_od_from_omx(path, matrix_key, lookup_key):
    """Load OD matrix and zone IDs from OMX file, return long OD DataFrame."""
    with h5py.File(path, "r") as f:
        od_array = f["data"][matrix_key][()]
        zone_ids = f["lookup"][lookup_key][()]

    # Make sure zone IDs are 1D and use a "clean" dtype
    zone_ids = np.asarray(zone_ids).astype(int)

    # Build wide matrix with zone IDs as index/columns
    od_wide_df = pd.DataFrame(od_array, index=zone_ids, columns=zone_ids)
    od_wide_df.index.name = "origin"
    od_wide_df.columns.name = "destination"

    # Long form: origin, destination, value
    od_long_df = (
        od_wide_df
        .stack()
        .rename_axis(["origin", "destination"])
        .reset_index(name="value")
    )

    return od_long_df


def sample_integer_trips(od_long_df, value_col="value",
                         use_poisson=True, seed=None, max_total_trips=None):
    """
    From float OD values, generate integer trip counts per OD pair.

    - If use_poisson: N_ij ~ Poisson(lambda = value)
    - Else: stochastic rounding (floor + 0/1 with prob = fractional part)
    """
    rng = np.random.default_rng(seed)
    vals = od_long_df[value_col].to_numpy()

    if use_poisson:
        n_trips = rng.poisson(lam=vals)
    else:
        floors = np.floor(vals)
        fracs = vals - floors
        u = rng.random(len(vals))
        add_one = (u < fracs).astype(int)
        n_trips = floors.astype(int) + add_one

    od_counts = od_long_df.copy()
    od_counts["n_trips"] = n_trips
    od_counts = od_counts.loc[od_counts["n_trips"] > 0].reset_index(drop=True)

    if max_total_trips is not None:
        total = od_counts["n_trips"].sum()
        if total > max_total_trips and total > 0:
            # Downscale trip counts proportionally
            scale = max_total_trips / total
            od_counts["n_trips"] = np.floor(od_counts["n_trips"] * scale).astype(int)
            od_counts = od_counts.loc[od_counts["n_trips"] > 0].reset_index(drop=True)
            logger.info(
                "Total trips exceeded max_total_trips, "
                "downscaled by factor %.3f.", scale
            )

    return od_counts


def build_zone_sampling_points(
    context,
    gpkg_path,
    layer_name,
    zone_id_col,
    statent_x_col,
    statent_y_col,
):
    """
    Build a mapping: zone_id -> array([[x, y], ...]) of candidate points.

    - zone_id comes strictly from the zone layer (gpkg), not from enterprises.
    - If a zone has no enterprises, we use the zone centroid as fallback.
    """

    # ----------------------------------------------------------------------
    # 1) Load zones from GPKG, enforce ONE clean 'zone_id' column
    # ----------------------------------------------------------------------
    if layer_name is None:
        zones_gdf = gpd.read_file(gpkg_path)
    else:
        zones_gdf = gpd.read_file(gpkg_path, layer=layer_name)

    zones_gdf = zones_gdf.copy()

    # Keep just geometry + the ID column we care about
    zones_gdf = zones_gdf[[zone_id_col, "geometry"]].copy()
    zones_gdf = zones_gdf.rename(columns={zone_id_col: "zone_id"})
    zones_gdf["zone_id"] = zones_gdf["zone_id"].astype(int)

    # Deduplicate if the source has multiple rows per zone_id
    zones_gdf = zones_gdf.drop_duplicates(subset="zone_id").reset_index(drop=True)

    # Centroids for fallback
    zones_gdf["centroid"] = zones_gdf.geometry.centroid
    zones_gdf["centroid_x"] = zones_gdf["centroid"].x
    zones_gdf["centroid_y"] = zones_gdf["centroid"].y

    valid_zone_ids = set(zones_gdf["zone_id"].unique())
    n_valid_zones = len(valid_zone_ids)
    logger.debug("Zones in GPKG (unique zone_id): %d", n_valid_zones)

    # ----------------------------------------------------------------------
    # 2) Load enterprises and prepare as GeoDataFrame
    # ----------------------------------------------------------------------
    df_statent = context.stage("data.statent.statent")

    # Make sure enterprises table does NOT have a conflicting 'zone_id' column
    if "zone_id" in df_statent.columns:
        logger.debug("Dropping 'zone_id' column from enterprises to avoid conflict.")
        df_statent = df_statent.drop(columns=["zone_id"])

    enterprises_gdf = gpd.GeoDataFrame(
        df_statent,
        geometry=gpd.points_from_xy(
            df_statent[statent_x_col],
            df_statent[statent_y_col],
        ),
        crs=zones_gdf.crs,
    )

    # ----------------------------------------------------------------------
    # 3) Spatial join: attach zone_id from zones_gdf to enterprises
    # ----------------------------------------------------------------------
    # Note: we only pass geometry + 'zone_id' from zones, so the sjoin result
    # will have a single 'zone_id' column from the right GeoDataFrame.
    enterprises_joined = gpd.sjoin(
        enterprises_gdf,
        zones_gdf[["zone_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    # Reset index to simplify things
    enterprises_joined = enterprises_joined.reset_index(drop=True)

    # Ensure 'zone_id' is present and 1D
    if "zone_id" not in enterprises_joined.columns:
        raise RuntimeError(
            "Expected 'zone_id' in enterprises_joined after sjoin, got columns: "
            f"{list(enterprises_joined.columns)}"
        )

    enterprises_joined["zone_id"] = enterprises_joined["zone_id"].astype(int)

    # Filter out any zone_ids that are not in the valid set (paranoid check)
    enterprises_joined = enterprises_joined[
        enterprises_joined["zone_id"].isin(valid_zone_ids)
    ].copy()

    # Coordinates from geometry
    enterprises_joined["x"] = enterprises_joined.geometry.x
    enterprises_joined["y"] = enterprises_joined.geometry.y

    # ----------------------------------------------------------------------
    # 4) Build dict: zone_id -> array of [x, y] candidate points
    # ----------------------------------------------------------------------
    zone_to_points = {}

    grouped = enterprises_joined.groupby("zone_id")
    for zid, grp in grouped:
        pts = grp[["x", "y"]].to_numpy()
        zone_to_points[int(zid)] = pts

    # Fallback: centroids for zones with no enterprises
    for row in zones_gdf.itertuples():
        zid = int(row.zone_id)
        if zid not in zone_to_points:
            zone_to_points[zid] = np.array([[row.centroid_x, row.centroid_y]])

    return zone_to_points


def generate_trip_list(od_counts, zone_to_points, seed=None):
    """
    Expand OD counts into one row per trip with random enterprise origin/destination.

    Returns a DataFrame:
        trip_id, origin_zone, destination_zone,
        origin_x, origin_y, destination_x, destination_y
    """
    rng = np.random.default_rng(seed)

    trip_records = []

    for row in od_counts.itertuples(index=False):
        origin_zone = int(row.origin)
        dest_zone = int(row.destination)
        n = int(row.n_trips)

        if n <= 0:
            continue

        # Get candidate points for origin and destination zones
        if origin_zone not in zone_to_points:
            # Should not happen if dictionary built from full zone set,
            # but we guard anyway (fallback: skip)
            continue
        if dest_zone not in zone_to_points:
            continue

        origin_pts = zone_to_points[origin_zone]
        dest_pts = zone_to_points[dest_zone]

        # Sample indices
        origin_idx = rng.integers(0, len(origin_pts), size=n)
        dest_idx = rng.integers(0, len(dest_pts), size=n)

        origin_samples = origin_pts[origin_idx]  # shape (n, 2)
        dest_samples = dest_pts[dest_idx]        # shape (n, 2)

        trip_df = pd.DataFrame({
            "origin_zone": np.full(n, origin_zone, dtype=int),
            "destination_zone": np.full(n, dest_zone, dtype=int),
            "origin_x": origin_samples[:, 0],
            "origin_y": origin_samples[:, 1],
            "destination_x": dest_samples[:, 0],
            "destination_y": dest_samples[:, 1],
        })

        trip_records.append(trip_df)

    if not trip_records:
        return pd.DataFrame(
            columns=[
                "trip_id",
                "origin_zone",
                "destination_zone",
                "origin_x",
                "origin_y",
                "destination_x",
                "destination_y",
            ]
        )

    trips_df = pd.concat(trip_records, ignore_index=True)
    trips_df.insert(0, "trip_id", np.arange(1, len(trips_df) + 1, dtype=int))

    return trips_df


def build_departure_time_bins(dep_df,
                              weight_col="person_weight",
                              time_col="departure_time",
                              bin_width_sec=1800):

    df = dep_df.copy()

    # Clean up
    df = df[[time_col, weight_col]].dropna()
    df[weight_col] = df[weight_col].astype(float)

    # Keep only positive-weight rows
    df = df[df[weight_col] > 0]

    if df.empty:
        raise ValueError("No positive-weight observations in departure time data.")

    # Clip to [0, 24h) if needed
    day_secs = 24 * 3600
    df[time_col] = df[time_col].clip(lower=0, upper=day_secs - 1)

    # Bin index (integer)
    df["bin_idx"] = (df[time_col] // bin_width_sec).astype(int)

    # Aggregate weights per bin
    bins = (
        df.groupby("bin_idx", as_index=False)[weight_col]
        .sum()
        .rename(columns={weight_col: "weight"})
    )

    total_w = bins["weight"].sum()
    if total_w <= 0:
        raise ValueError("Sum of weights across bins must be > 0.")

    bins["prob"] = bins["weight"] / total_w

    # Compute start/end of each bin in seconds
    bins["start_sec"] = bins["bin_idx"] * bin_width_sec
    bins["end_sec"] = bins["start_sec"] + bin_width_sec
    bins["end_sec"] = bins["end_sec"].clip(upper=day_secs)

    # Filter out any bins with zero width (just in case)
    bins = bins[bins["end_sec"] > bins["start_sec"]].reset_index(drop=True)

    return bins


def sample_departure_times_binned(trips_df, bins_df, seed=None):
   
    rng = np.random.default_rng(seed)

    n_trips = len(trips_df)
    if n_trips == 0:
        trips_df["departure_time"] = []
        return trips_df

    probs = bins_df["prob"].to_numpy()
    starts = bins_df["start_sec"].to_numpy()
    ends = bins_df["end_sec"].to_numpy()
    widths = ends - starts

    # Sample which bin each trip falls in (by index in bins_df)
    bin_choices = rng.choice(len(bins_df), size=n_trips, p=probs)

    # For each chosen bin, sample a uniform offset inside that bin
    u = rng.random(n_trips)
    chosen_starts = starts[bin_choices]
    chosen_widths = widths[bin_choices]

    dep_times = chosen_starts + (u * chosen_widths)
    dep_times = dep_times.astype(int)

    out = trips_df.copy()
    out["departure_time"] = dep_times
    return out


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def execute(context):
    seed = context.config("random_seed")
    data_path = context.config("data_path")
    # 1) Load OD from OMX and make it long
    logger.info("[1/4] Loading OD matrix from OMX...")
    od_long = load_od_from_omx("%s/npvm/LI_Binnen.omx" % data_path, "10", "NO")
    logger.info("    OD long shape: %s, total demand = %.2f", od_long.shape, od_long['value'].sum())

    od_long["value"] *= context.config("input_downsampling")

    logger.info("    OD long shape: %s, total demand after downsampling = %.2f", od_long.shape, od_long['value'].sum())
    
    # 2) Sample integer trips per OD pair
    logger.info("[2/4] Sampling integer trips from float OD...")
    od_counts = sample_integer_trips(
        od_long,
        value_col="value",
        use_poisson=context.config("lcv_poisson_sampling"),
        seed=seed,
        max_total_trips=None,
    )
    logger.info("    Non-zero OD pairs: %d, total integer trips = %d", len(od_counts), od_counts['n_trips'].sum())

    # 3) Build sampling points per zone (enterprises or centroids)
    logger.info("[3/4] Building zone->points dictionary from shapefile + enterprises...")
    zone_to_points = build_zone_sampling_points(context,
        "%s/npvm/1_Verkehrszonen_Schweiz_NPVM_2023.gpkg" % data_path, None,
        "No",
        "x",
        "y",
    )
    logger.info("    Zones with sampling points: %d", len(zone_to_points))

    # 4) Generate the trip list with coordinates
    logger.info("[4/4] Generating trip list...")
    trips_df = generate_trip_list(
        od_counts,
        zone_to_points,
        seed=seed,
    )
    logger.info("    Generated %d trips.", len(trips_df))

     # 5) Sample departure times and attach to trips
    logger.info("[5/5] Sampling departure times for each trip...")

    # --- Microcensus 2015 ---
    df_mz15_trips = pd.read_csv("%s/microcensus/wege.csv" % data_path, encoding="latin1")
    df_mz15_trips = df_mz15_trips.drop(columns=["WP"])
    df_mz15_persons = pd.read_csv("%s/microcensus/zielpersonen.csv" % data_path, encoding="latin1")

    df_mz15_trips = df_mz15_trips.merge(
        df_mz15_persons[["HHNR", "WP"]],
        on="HHNR",
        how="left",
    )

    df_mz15_trips["mz_year"] = 2015
    mode_map = {
        -99: "unknown",  # Pseudo stage
        1: "pt",         # Plane
        2: "pt",         # Train
        3: "pt",         # Postauto
        4: "pt",         # Ship
        5: "pt",         # Tram
        6: "pt",         # Bus
        7: "pt",         # Other PT
        8: "pt",         # Reisecar (coach)
        9: "car",        # Car
        10: "truck",       # Truck
        11: "pt",        # Taxi
        12: "car",       # Motorbike
        13: "car",       # Mofa
        14: "bike",      # Bicycle / E-bike
        15: "walk",      # Walking
        16: "bike",      # Machines similar to a vehicle
        17: "unknown"    # Other / don't know
    }

    df_mz15_trips["mode"] = df_mz15_trips["wmittel"].map(mode_map)
    #keeping only the truck trips to obtain the departure time curve
    df_mz15_trips = df_mz15_trips[df_mz15_trips["mode"]=="truck"]
    df_mz15_trips = df_mz15_trips[[
        "HHNR", "f51100", "mode", "WP"
    ]]

    # --- Microcensus 2021 ---
    df_mz21_trips = pd.read_csv("%s/microcensus/21/wege.csv" % data_path, sep=";", encoding="latin1")
    df_mz21_persons = pd.read_csv("%s/microcensus/21/zielpersonen.csv" % data_path, sep=";", encoding="latin1")
    df_mz21_trips = df_mz21_trips.drop(columns=["WP"])
    df_mz21_trips = df_mz21_trips.merge(
        df_mz21_persons[["HHNR", "WP"]],
        on="HHNR",
        how="left",
    )
   
    df_mz21_trips["mz_year"] = 2021
    #TODO: wmittel1 is the mode in mz 21
    mode_map = {
        -99: "unknown",  # Pseudo stage
        1: "pt",         # Plane
        2: "pt",         # Train
        3: "pt",         # Ship
        4: "pt",         # Tram
        5: "pt",         # Bus, postauto
        6: "pt",         # other PT
        7: "pt",         # Reisecar (coach)
        8: "car",        # Car
        9: "truck",       # Truck
        10: "pt",        # Taxi
        11: "pt",        # Taxi-like modes e.g. Uber
        12: "car",       # Motorbike
        13: "car",       # Mofa
        14: "bike",      # E-bike
        15: "bike",      # bike
        16: "walk",      # Walking
        17: "bike",      # Machines similar to a vehicle
        18: "unknown"    # Other / don't know
    }

    df_mz21_trips["mode"] = df_mz21_trips["wmittel1"].map(mode_map)
    df_mz21_trips = df_mz21_trips[df_mz21_trips["mode"]=="truck"]
    df_mz21_trips = df_mz21_trips[[
        "HHNR", "f51100", "mode", "WP"
    ]]
    # --- Combine both years ---
    df_mz_trips = pd.concat(
        [df_mz15_trips, df_mz21_trips],
        ignore_index=True,
        sort=False,
    )
    df_mz_trips = df_mz_trips[[
        "HHNR", "f51100", "mode", "WP"
    ]]

    ##NOTE: there are also departure times between 24:00 and 30:00 we currently do not do anything special about it
    ## we treat it the same way as for the movement of people
    df_mz_trips.loc[:, "departure_time"] = df_mz_trips["f51100"] * 60
    df_mz_trips["person_weight"] = df_mz_trips["WP"]
    dep_df = df_mz_trips.copy()

    bins_df = build_departure_time_bins(
        dep_df,
        weight_col="person_weight",
        time_col="departure_time",
        bin_width_sec=3600,
    )

    trips_df = sample_departure_times_binned(
        trips_df,
        bins_df,
        seed=seed,
    )

    logger.info("[5/5] Adding home locations as possible destinations after configured time...")

    home_zone_to_points = build_home_sampling_points(
        context,
        "%s/npvm/1_Verkehrszonen_Schweiz_NPVM_2023.gpkg" % data_path,
        None,
        "No",
        "home_x",
        "home_y",
    )

    trips_df = add_home_destinations_after_time(
        trips_df,
        zone_to_points,
        home_zone_to_points,
        after_time_sec=context.config("lcv_home_destination_after"),
        seed=seed,
    )

    logger.info(
        "    Home destinations available after %d seconds; home zones with points: %d",
        context.config("lcv_home_destination_after"),
        len(home_zone_to_points),
    )

    # if the freight module is used, make sure that the trip ids do not overlap
    if context.config("use_freight"):
        id_offset = context.stage("synthesis.freight.trips")["agent_id"].max() + 1
        trips_df["trip_id"] += id_offset

    bin_width = 3600

    departure_distribution = (
        trips_df
        .assign(departure_hour_bin=(trips_df["departure_time"] // bin_width).astype(int))
        .groupby("departure_hour_bin")
        .size()
        .reset_index(name="n_trips")
    )

    departure_distribution["start_sec"] = departure_distribution["departure_hour_bin"] * bin_width
    departure_distribution["end_sec"] = departure_distribution["start_sec"] + bin_width
    departure_distribution["share"] = departure_distribution["n_trips"] / departure_distribution["n_trips"].sum()

    return trips_df
