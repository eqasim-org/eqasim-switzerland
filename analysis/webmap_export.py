import pandas as pd
import numpy as np
import geopandas as gpd
import json
import math
import os
import gzip
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict, Counter


from shapely.geometry import Point, LineString, mapping
from pyproj import Transformer


# === GLOBAL SETTINGS ===
DISTANCE_COLS = ["euclidean_distance", "network_distance"]
WEIGHT_COL = "person_weight"  # Used only if `weighted=True`

GEO_LEVEL = "canton_name"
AGGREGATION_COL = "purpose" 
GROUP_BY_COLS = [GEO_LEVEL, AGGREGATION_COL] 

DEFAULT_WORKDIR = None

# === FUNCTIONS ===

# === DATA IMPORT AND PREPROCESSING ===

def assign_cantons(df, canton_boundaries, x_col="start_x", y_col="start_y"):
    """Assigns canton names and IDs to a dataset based on origin coordinates."""

    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Missing required columns '{x_col}' or '{y_col}' in dataframe.")

    geometry = gpd.points_from_xy(df[x_col], df[y_col])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:2056")

    # Spatial join on cleaned columns
    within = gdf.sjoin(canton_boundaries[['canton_id', 'canton_name', 'geometry']], how="left", predicate="within")

    # Handle points that didn't fall within a canton geometry
    non_match = within[within['canton_name'].isna()].drop(columns=['index_right', 'canton_id', 'canton_name'], errors='ignore')
    match = within[within['canton_name'].notna()]

    match_nearest = non_match.sjoin_nearest(
        canton_boundaries[['canton_id', 'canton_name', 'geometry']],
        max_distance=3500,
        distance_col="distance"
    )

    # Combine matched and nearest-matched
    result = pd.concat([match, match_nearest], ignore_index=True)

    # Clean up columns
    result = result.drop(columns=['geometry', 'index_right'], errors='ignore')

    return result


def import_data(eqasim, trips, persons, filter=250000):
    """Imports and preprocesses synthetic and microcensus datasets, filtering and renaming columns to consistency across datasets."""
    
    # Add previous purpose to each row, by looking at the trip_id for the each unique person_id, and the purpose for the previous trip_id
    trips = trips.sort_values(by=["person_id", "trip_id"]).reset_index(drop=True)
    trips["prev_purpose"] = trips.groupby("person_id")["purpose"].shift(1)

    # Reorder columns to place 'prev_purpose' right before 'purpose'
    cols = list(trips.columns)
    purpose_index = cols.index("purpose")
    cols.insert(purpose_index, cols.pop(cols.index("prev_purpose")))
    trips = trips[cols]

    eqasim_filt = eqasim[~eqasim["person"].str.startswith("freight")] #filter out freight
    eqasim_filt = eqasim_filt[~(eqasim_filt["euclidean_distance"] == 0)] #remove loop trips

    trips = trips[trips["person_id"].isin(persons["person_id"])] # filter to only weekdays
    trips = trips[~(trips["crowfly_distance"] == 0)]             # remove loop trips

    # Add person weight to data
    trips = trips.join(persons[["person_id", "person_weight"]].set_index("person_id"), on="person_id")
    
    #filter by distance
    trips = trips.loc[trips["crowfly_distance"] < filter] 
    eqasim_filt = eqasim_filt.loc[eqasim_filt["euclidean_distance"] < filter]

    # FILTER OUT NA CANTONS (from microcensus)
    trips = trips[trips["canton_name"].notna()]

    # RENAME TO euclidean_distance, network_distance
    trips = trips.rename(columns={'crowfly_distance': 'euclidean_distance'})
    eqasim_filt = eqasim_filt.rename(columns={'traveled_distance': 'network_distance'})
    eqasim_filt = eqasim_filt.rename(columns={"end_activity_type": "purpose"})
    eqasim_filt = eqasim_filt.rename(columns={"main_mode": "mode"})
    eqasim_filt = eqasim_filt.rename(columns={"dep_time": "departure_time"})

    ## convert departure time from hh:mm:ss to seconds after midnight
    def time_to_seconds(x):
        if isinstance(x, str):
            try:
                h, m, s = map(int, x.split(":"))
                return h * 3600 + m * 60 + s
            except:
                return pd.NA
        return x

    eqasim_filt["departure_time"] = eqasim_filt["departure_time"].apply(time_to_seconds)

    return eqasim_filt, trips

def clean_geo_name(name):
    """Normalize canton names: take first part before slash, remove accents/punctuation."""
    name = name.split("/")[0]  # Take first name (e.g., 'Bern' from 'Bern/Berne')
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    return name.replace(" ", "").replace(".", "").replace("/", "")

def weighted_mean(group, value_col, weight_col):
    """Compute the weighted mean if a weight column is provided, else return the regular mean."""
    if not weight_col:
        return group[value_col].mean()
    return (group[value_col] * group[weight_col]).sum() / group[weight_col].sum()

# === AVERAGE DISTANCE ===

def process_avg_dist(data, dataset_type, weighted=False):
    """Group by AGGREGATION_COLS, then compute mean (or weighted mean) DISTANCE_COLS."""
    aggregated = data.groupby(GROUP_BY_COLS).agg({col: "mean" for col in DISTANCE_COLS}).reset_index()
    aggregated["sample_size"] = data.groupby(GROUP_BY_COLS).size().values
    aggregated["dataset"] = dataset_type

    if weighted:
        grouped = data.groupby(GROUP_BY_COLS, group_keys=False)
        for col in DISTANCE_COLS:
            weighted_series = grouped.apply(
                lambda g: weighted_mean(g, col, WEIGHT_COL),
            ).reset_index(drop=True)
            aggregated[col] = weighted_series.values

    # Clean geographic column
    aggregated[GEO_LEVEL] = aggregated[GEO_LEVEL].apply(clean_geo_name)
    return aggregated

def export_avg_dist(microcensus_data, synthetic_data, weighted=True):
    """Compute and save average (or weighted) distance per aggregation group and geographic region."""

    microcensus_grouped = process_avg_dist(microcensus_data, "Microcensus", weighted=True)
    synthetic_grouped = process_avg_dist(synthetic_data, "Synthetic", weighted=False)

    json_data = {}

    # Generate JSON structure for output
    for df in [microcensus_grouped, synthetic_grouped]:
        for _, row in df.iterrows():
            region = row[GEO_LEVEL]
            group = row[AGGREGATION_COL]
            dataset = row["dataset"]

            if region not in json_data:
                json_data[region] = {"Microcensus": {}, "Synthetic": {}}

            json_data[region][dataset][group] = {
                col: row[col] for col in DISTANCE_COLS
            }
            json_data[region][dataset][group]["sample_size"] = int(row["sample_size"])

    # Compute Switzerland-wide totals (aggregated only on AGGREGATION_COL)
    def aggregate_all(df):
        grouped = df.groupby(AGGREGATION_COL, group_keys=False)
        return grouped.apply(
            lambda g: pd.Series({
                col: weighted_mean(g, col, "sample_size") for col in DISTANCE_COLS
            } | {
                "sample_size": g["sample_size"].sum()
            }),
        ).to_dict(orient="index")

    all_microcensus = aggregate_all(microcensus_grouped)
    all_synthetic = aggregate_all(synthetic_grouped)

    json_data["All"] = {
        "Microcensus": {group: {k: float(v) for k, v in data.items()} for group, data in all_microcensus.items()},
        "Synthetic": {group: {k: float(v) for k, v in data.items()} for group, data in all_synthetic.items()}
    }

    # Save to /public/data/avg_dist_data.json
    output_path = os.path.join(DEFAULT_WORKDIR, "public", "data", f"avg_dist_data_{AGGREGATION_COL}.json")

    # Save to file
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=4)

    return f"Average Distances saved to: {output_path}"

# === HISTOGRAM DATA ===

def preprocess_histogram_data(microcensus_data, synthetic_data, data_type, weighted=True, max_iqr=3, num_bins=25):
    """Compute histogram distributions and means by GEO_LEVEL and AGGREGATION_COL for a given data column."""
    
    result = {}

    # Get unique geographic units (unmodified for filtering)
    geo_units = ["All"] + sorted(microcensus_data[GEO_LEVEL].dropna().unique())

    # Get all group values (e.g., modes)
    group_values = sorted(microcensus_data[AGGREGATION_COL].astype(str).unique())

    for geo_unit in geo_units:
        if geo_unit == "All":
            filtered_microcensus = microcensus_data
            filtered_synthetic = synthetic_data
        else:
            filtered_microcensus = microcensus_data[microcensus_data[GEO_LEVEL] == geo_unit]
            filtered_synthetic = synthetic_data[synthetic_data[GEO_LEVEL] == geo_unit]

        group_data = {}

        for group in group_values:
            mc_filtered = filtered_microcensus[filtered_microcensus[AGGREGATION_COL] == group]
            syn_filtered = filtered_synthetic[filtered_synthetic[AGGREGATION_COL] == group]

            microcensus_values = mc_filtered[data_type]
            synthetic_values = syn_filtered[data_type]
            person_weight = mc_filtered[WEIGHT_COL] if weighted and WEIGHT_COL in mc_filtered.columns else None

            if len(microcensus_values) == 0 or len(synthetic_values) == 0:
                continue

            # Compute bins using IQR rule
            q1, q3 = synthetic_values.quantile(0.25), synthetic_values.quantile(0.75)
            iqr = q3 - q1
            range_max = q3 + max_iqr * iqr
            bin_size = range_max / num_bins

            mc_bins = np.arange(0, max(microcensus_values) + bin_size, bin_size)
            syn_bins = np.arange(0, max(synthetic_values) + bin_size, bin_size)

            # Compute histograms
            mc_hist, bins = np.histogram(
                microcensus_values, bins=mc_bins, range=(0, max(microcensus_values)),
                weights=person_weight, density=False
            )
            syn_hist, _ = np.histogram(
                synthetic_values, bins=syn_bins, range=(0, max(synthetic_values)),
                density=False
            )

            # Convert to percentages
            mc_hist = (mc_hist / mc_hist.sum()) * 100
            syn_hist = (syn_hist / syn_hist.sum()) * 100

            # Compute means
            mc_mean = np.average(microcensus_values, weights=person_weight) if weighted and person_weight is not None else microcensus_values.mean()
            syn_mean = synthetic_values.mean()

            group_data[group] = {
                "bin_width": bin_size,
                "bins": bins.tolist(),
                "microcensus_histogram": mc_hist.tolist(),
                "synthetic_histogram": syn_hist.tolist(),
                "microcensus_mean": mc_mean,
                "synthetic_mean": syn_mean,
                "microcensus_sample_size": len(microcensus_values),
                "synthetic_sample_size": len(synthetic_values)
            }

        # Clean canton names
        result[clean_geo_name(geo_unit)] = group_data

    return result

def export_histogram(microcensus_data, synthetic_data, weighted=True):
    """Generate and export histogram data for all distance columns defined globally."""
    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data")
    os.makedirs(output_dir, exist_ok=True)

    for dist_col in DISTANCE_COLS:
        data = preprocess_histogram_data(microcensus_data, synthetic_data, data_type=dist_col, weighted=weighted)
        output_path = os.path.join(output_dir, f"histogram_{dist_col}_{AGGREGATION_COL}.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

    return f"Histogram data exported to {output_dir}"

# === STACKED BAR PLOT DATA ===
def process_stacked_bar_data(data, mode_order, weighted=True):
    """Aggregates mode share percentages by distance category and dataset."""
    if weighted:
        grouped_data = data.groupby(['distance_category', AGGREGATION_COL, 'dataset'], observed=False).agg(
            count=(WEIGHT_COL, 'sum') 
        ).reset_index()

        # Synthetic data gets simple counts
        grouped_data.loc[grouped_data['dataset'] == 'Synthetic', 'count'] = data[
            data['dataset'] == 'Synthetic'
        ].groupby(['distance_category', AGGREGATION_COL], observed=False).size().values
    else:
        grouped_data = data.groupby(['distance_category', AGGREGATION_COL, 'dataset'], observed=False).size().reset_index(name='count')

    total_weight = grouped_data.groupby(['distance_category', 'dataset'], observed=False)['count'].transform('sum')
    grouped_data['percentage'] = (grouped_data['count'] / total_weight) * 100
    grouped_data['percentage'] = grouped_data['percentage'].round(1)

    # Only enforce mode order if AGGREGATION_COL is "mode"
    if AGGREGATION_COL == "mode":
        grouped_data[AGGREGATION_COL] = pd.Categorical(
            grouped_data[AGGREGATION_COL],
            categories=mode_order,
            ordered=True
        )
    else:
        unique_agg_values = sorted(grouped_data[AGGREGATION_COL].dropna().unique())
        grouped_data[AGGREGATION_COL] = pd.Categorical(
            grouped_data[AGGREGATION_COL],
            categories=unique_agg_values,
            ordered=True
        )

    return grouped_data

def export_stacked_bar(microcensus_data, synthetic_data, distance_bins=None, weighted=True):
    """Process and export stacked bar plot data for each distance type (e.g. euclidean and network)."""

    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data")

    if distance_bins is None:
        distance_bins = [0, 1000, 5000, 25000, float('inf')]

    bin_labels = [f"{distance_bins[i]}-{distance_bins[i+1]}" if distance_bins[i+1] != float('inf') 
                  else f"{distance_bins[i]}+" for i in range(len(distance_bins) - 1)]

    mode_order = ['car', 'car_passenger', 'pt', 'bike', 'walk']

    for distance_col in DISTANCE_COLS:
        mc = microcensus_data.copy()
        syn = synthetic_data.copy()
        mc['dataset'] = "Microcensus"
        syn['dataset'] = "Synthetic"

        mc['distance_category'] = pd.cut(mc[distance_col], bins=distance_bins, labels=bin_labels, right=False)
        syn['distance_category'] = pd.cut(syn[distance_col], bins=distance_bins, labels=bin_labels, right=False)

        flat_data = {}

        # === All data (keyed as "All")
        combined_all = pd.concat([mc, syn])
        grouped_all = process_stacked_bar_data(combined_all, mode_order, weighted)
        flat_data["All"] = grouped_all.to_dict(orient='records')

        # === Per region
        geo_units = sorted(mc[GEO_LEVEL].dropna().unique())

        for geo in geo_units:
            mc_region = mc[mc[GEO_LEVEL] == geo]
            syn_region = syn[syn[GEO_LEVEL] == geo]
            combined = pd.concat([mc_region, syn_region])
            grouped = process_stacked_bar_data(combined, mode_order, weighted)
            flat_data[clean_geo_name(geo)] = grouped.to_dict(orient='records')

        # === Export flat JSON
        output_path = os.path.join(output_dir, f"stacked_bar_{distance_col}_{AGGREGATION_COL}.json")
        with open(output_path, "w") as f:
            json.dump(flat_data, f, indent=4)

    return f"Stacked bar data exported to: {output_dir}"

# === LINEPLOT DATA ===
def process_line_plot_data(data, variable, max_value, unit_conversion, num_bins, weighted):
    """Processes data into bins, applies unit conversion, and calculates mode share percentage."""
    data = data.copy()
    bin_interval = max_value / num_bins
    tick_labels, tick_vals = None, None

    # Unit conversion
    if unit_conversion == 's_to_h':
        tick_interval = 30 * 60 if max_value <= 6 * 3600 else max(3600, 3600 * round((max_value / 12) / 3600))
        tick_vals = np.arange(0, max_value + 1, tick_interval)
        tick_labels = [(datetime(1970, 1, 1) + timedelta(seconds=int(t))).strftime("%I:%M %p") for t in tick_vals]
        data[variable] /= 3600
        max_value /= 3600
        bin_interval = max_value / num_bins
        tick_vals = tick_vals / 3600

    elif unit_conversion == 'm_to_km':
        data[variable] /= 1000
        max_value /= 1000
        bin_interval = max_value / num_bins

        if max_value <= 5:
            tick_interval = 0.5
        elif max_value <= 10:
            tick_interval = 1
        elif max_value <= 25:
            tick_interval = 2
        else:
            raw_interval = max_value / 10
            magnitude = 10 ** math.floor(math.log10(raw_interval))
            tick_interval = math.ceil(raw_interval / magnitude) * magnitude

        tick_vals = np.arange(0, max_value + tick_interval, tick_interval)
        tick_labels = [f"{int(val)} km" if val.is_integer() else f"{round(val, 1)} km" for val in tick_vals]

    # Bin and group
    bins = np.arange(0, max_value + bin_interval, bin_interval)
    bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    data['variable_bin'] = pd.cut(data[variable], bins=bins, labels=bin_labels, right=False, ordered=False)

    if weighted and WEIGHT_COL in data.columns:
        grouped = data.groupby(['variable_bin', AGGREGATION_COL], observed=False).agg(
            count=(WEIGHT_COL, 'sum')
        ).reset_index()
    else:
        grouped = data.groupby(['variable_bin', AGGREGATION_COL], observed=False).size().reset_index(name='count')

    grouped['percentage'] = grouped.groupby('variable_bin', observed=False)['count'].transform(lambda x: (x / x.sum()) * 100).fillna(0)

    # Midpoints
    bin_midpoints = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    midpoint_map = dict(zip(bin_labels, bin_midpoints))
    grouped['variable_midpoint'] = grouped['variable_bin'].map(midpoint_map)

    return grouped, tick_labels, tick_vals, max_value


def export_line_chart(microcensus, synthetic, max_euclidean=20000, max_network=10000, max_departure=60 * 60 * 30):
    """Generates lineplot JSON files by GEO_LEVEL."""
    variables = {
        "departure_time": ("s_to_h", max_departure, 32),
        "euclidean_distance": ("m_to_km", max_euclidean, 20),
        "network_distance": ("m_to_km", max_network, 20),
    }

    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data")

    for variable, (unit_conversion, max_value, num_bins) in variables.items():
        region_data = {}

        geo_units = ['All'] + sorted(microcensus[GEO_LEVEL].dropna().astype(str).unique())

        for geo in geo_units:
            mc_filtered = microcensus if geo == 'All' else microcensus[microcensus[GEO_LEVEL] == geo]
            syn_filtered = synthetic if geo == 'All' else synthetic[synthetic[GEO_LEVEL] == geo]

            grouped_mc, tick_labels, tick_vals, max_val_out = process_line_plot_data(
                mc_filtered, variable, max_value, unit_conversion, num_bins, weighted=True
            )
            grouped_syn, *_ = process_line_plot_data(
                syn_filtered, variable, max_value, unit_conversion, num_bins, weighted=False
            )

            region_data[clean_geo_name(geo)] = {
                "microcensus": grouped_mc.to_dict(orient="records"),
                "synthetic": grouped_syn.to_dict(orient="records"),
                "tick_labels": tick_labels,
                "tick_vals": tick_vals.tolist(),
                "max_value": max_val_out
            }

        with open(os.path.join(output_dir, f"lineplot_{variable}_data_{AGGREGATION_COL}.json"), "w") as f:
            json.dump(region_data, f, indent=4)

    return f"Lineplot JSONs exported to {output_dir}"

# === AGGREGATION SHARE ===
def compute_aggregation_share(microcensus_data, synthetic_data, weighted=True):
    """Compute share of AGGREGATION_COL within each GEO_LEVEL for both datasets."""

    def compute_share(df, weighted):
        if weighted and WEIGHT_COL in df.columns:
            total = df.groupby(GEO_LEVEL, observed=False)[WEIGHT_COL].sum().reset_index(name='total')
            group = df.groupby([GEO_LEVEL, AGGREGATION_COL], observed=False)[WEIGHT_COL].sum().reset_index(name='count')
        else:
            total = df.groupby(GEO_LEVEL, observed=False)[AGGREGATION_COL].count().reset_index(name='total')
            group = df.groupby([GEO_LEVEL, AGGREGATION_COL], observed=False).size().reset_index(name='count')

        merged = group.merge(total, on=GEO_LEVEL)
        merged['share'] = merged['count'] / merged['total']
        merged[GEO_LEVEL] = merged[GEO_LEVEL].apply(clean_geo_name)
        return merged[[GEO_LEVEL, AGGREGATION_COL, 'share']]
    # Compute shares
    mc_share = compute_share(microcensus_data, weighted=weighted)
    syn_share = compute_share(synthetic_data, weighted=False)

    # Combine for max share per group
    combined = pd.concat([mc_share, syn_share])
    max_share = (
        combined.groupby(AGGREGATION_COL)["share"]
        .max()
        .apply(lambda x: math.ceil(x * 20) / 20)
        .to_dict()
    )

    result = {
        "Microcensus": mc_share.to_dict(orient="records"),
        "Synthetic": syn_share.to_dict(orient="records"),
        f"max_share_per_{AGGREGATION_COL}": max_share
    }

    output_path = os.path.join(DEFAULT_WORKDIR, "public", "data", f"{AGGREGATION_COL}_share.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

def export_network_by_canton(network_gdf, cantons_gdf, skip_cantons=None):
    """
    Intersects a MATSim network with canton geometries and exports each as a GeoJSON.
    Also saves a JSON of available transport modes per canton.
    
    Parameters:
    - network_gdf: GeoDataFrame of the MATSim network (already projected).
    - cantons_gdf: GeoDataFrame of canton geometries.
    - skip_cantons: Optional list of canton names to skip.
    """
    if skip_cantons is None:
        skip_cantons = []

    matsim_output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data", "matsim")

    canton_modes = {}

    for _, canton in cantons_gdf.iterrows():
        canton_name = canton["canton_name"]
        if canton_name in skip_cantons:
            print(f"Skipping {canton_name}")
            continue

        print(f"Processing {canton_name}")
        canton_geom = canton.geometry

        intersected = network_gdf[network_gdf.intersects(canton_geom)].copy()
        intersected["geometry"] = intersected.geometry.intersection(canton_geom)

        essential_cols = ["link_id", "geometry", "length", "freespeed", "capacity", "permlanes", "modes"]
        intersected = intersected[essential_cols].rename(columns={"link_id": "id"})

        intersected = intersected.sort_values(by="capacity", ascending=True).reset_index(drop=True)
        intersected = intersected.to_crs("EPSG:4326")

        output_canton_name = clean_geo_name(canton_name)
        output_path = os.path.join(matsim_output_dir, f"matsim_network_{output_canton_name}.geojson")
        intersected.to_file(output_path, driver="GeoJSON")
        print(f"Saved: {output_path}")

        # Extract and store modes
        mode_set = set()
        for mode_str in intersected["modes"]:
            mode_set.update(mode_str.split(","))
        canton_modes[output_canton_name] = sorted(mode_set)

    # Save all modes to JSON
    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data")
    modes_path = os.path.join(output_dir, "modes_by_canton.json")
    with open(modes_path, "w") as f:
        json.dump(canton_modes, f, indent=2)
    print(f"Saved mode summary to: {modes_path}")

def export_link_volumes_by_canton(cantons_gdf, linkstats_path, skip_cantons=None):
    """
    Merges link traffic volumes from linkstats.txt with MATSim canton GeoJSONs and exports:
    - A JSON of hourly and daily volumes per link
    - An updated GeoJSON with daily_avg_volume
    
    Parameters:
    - cantons_gdf: GeoDataFrame containing canton geometries and names
    - linkstats_path: Path to the linkstats file (e.g. "50.linkstats.txt")
    - skip_cantons: Optional set of canton names to skip
    """

    if skip_cantons is None:
        skip_cantons = set()

    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data", "matsim")

    linkstats = pd.read_csv(linkstats_path, sep='\t', comment='#', dtype=str)
    linkstats.columns = linkstats.columns.str.strip()
    linkstats = linkstats.apply(pd.to_numeric, errors='ignore')

    for _, canton in cantons_gdf.iterrows():
        canton_name = canton['canton_name']
        if canton_name in skip_cantons:
            print(f"Skipping {canton_name}")
            continue

        print(f"Processing {canton_name}")
        clean_name = clean_geo_name(canton_name)

        # Load the matching GeoJSON
        geojson_path = os.path.join(output_dir, f"matsim_network_{clean_name}.geojson")
        if not os.path.exists(geojson_path):
            print(f"  → Skipped: GeoJSON not found for {canton_name}")
            continue

        gdf = gpd.read_file(geojson_path)

        # Ensure string type for merging
        gdf["id"] = gdf["id"].astype(str)
        linkstats["LINK"] = linkstats["LINK"].astype(str)

        # Merge linkstats into GeoDataFrame
        merged = gdf.merge(linkstats, left_on="id", right_on="LINK", how="inner")

        # Extract hourly averages
        hourly_avg_cols = [col for col in merged.columns if col.startswith("HRS") and col.endswith("avg") and col != "HRS0-24avg"]

        # Create JSON summary
        link_summaries = []
        for _, row in merged.iterrows():
            hourly_data = {col: row[col] for col in hourly_avg_cols}
            summary = {
                "link_id": row["id"],
                "hourly_avg_volumes": hourly_data,
                "daily_avg_volume": row["HRS0-24avg"]
            }
            link_summaries.append(summary)

        # Save JSON file
        json_path = os.path.join(output_dir, f"{clean_name}_link_traffic_volumes.json")
        with open(json_path, "w") as f:
            json.dump(link_summaries, f, indent=2)
        print(f"Saved: {json_path}")

        # Overwrite if col already exists in geojson
        if "daily_avg_volume" in gdf.columns:
            gdf = gdf.drop(columns=["daily_avg_volume"])

        # Merge daily avg back into GeoDataFrame and export
        gdf_with_avg = gdf.merge(
            linkstats[["LINK", "HRS0-24avg"]],
            left_on="id",
            right_on="LINK",
            how="left"
        ).rename(columns={"HRS0-24avg": "daily_avg_volume"})

        gdf_with_avg.drop(columns=["LINK"], inplace=True)

        geojson_with_avg_path = os.path.join(output_dir, f"matsim_network_{clean_name}.geojson")
        gdf_with_avg.to_file(geojson_with_avg_path, driver="GeoJSON")
        print(f"Saved: {geojson_with_avg_path}")

def export_by_aggregation(mc_data, syn_data, aggregation_col):
    """Run only the exports that depend on AGGREGATION_COL."""
    global AGGREGATION_COL, GROUP_BY_COLS
    AGGREGATION_COL = aggregation_col
    GROUP_BY_COLS = [GEO_LEVEL, AGGREGATION_COL]

    print(f"\n=== Exporting for AGGREGATION_COL = '{aggregation_col}' ===")

    print("Generating average distances data")
    export_avg_dist(mc_data, syn_data)

    print("Generating histogram data")
    export_histogram(mc_data, syn_data)

    print("Generating stacked bar plot data")
    export_stacked_bar(mc_data, syn_data)

    print("Generating line plot data")
    export_line_chart(mc_data, syn_data)

    print("Generating aggregation share data")
    compute_aggregation_share(mc_data, syn_data)


## TRANSIT STOPS PROCESSING

def parse_stops(schedule_path):
    """
    Parse MATSim schedule XML and return a GeoDataFrame of *aggregated* stops 
    (one feature per stop-name) in WGS84, with deduped lines, modes_list,
    and predominant_mode.
    """
    # 0) XML + CRS set-up
    if schedule_path.endswith(".gz"):
        with gzip.open(schedule_path, 'rb') as f:
            tree = ET.parse(f)
    else:
        tree = ET.parse(schedule_path)

    root = tree.getroot()
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    # 1) pull out every stopFacility into a dict
    stop_facilities = {}
    for stop in root.find('transitStops'):
        sid  = stop.attrib['id']
        x,y  = float(stop.attrib['x']), float(stop.attrib['y'])
        name = stop.attrib.get('name','')
        stop_facilities[sid] = {
            'x': x, 'y': y,
            'name': name,
            'lines_routes': []
        }

    # 2) walk every line / routeProfile / stop → attach to stop_facilities
    for line in root.findall('transitLine'):
        line_id = line.attrib['id']
        line_name = line.attrib.get('name') or line.attrib['id'] # fall back to id if name is not available
        for route in line.findall('transitRoute'):
            route_id = route.attrib.get('id', '')
            mode = route.findtext('transportMode')
            route_profile = route.find('routeProfile')
            if route_profile is not None:
                for stop in route_profile.findall('stop'):
                    stop_ref = stop.attrib['refId']
                    if stop_ref not in stop_facilities:
                        stop_facilities[stop_ref] = {
                            'x': None, 'y': None, 'name': '', 'attributes': {}, 'lines_routes': []
                        }
                    stop_facilities[stop_ref]['lines_routes'].append({
                        'line_id': line_id,
                        'line_name': line_name, 
                        'route_id': route_id,
                        'mode': mode
                    })

    # 3) flatten into a pandas DataFrame with lon/lat + dedupe per stop_id
    rows = []
    for sid, info in stop_facilities.items():
        if info['x'] is None:
            continue
        lon, lat = transformer.transform(info['x'], info['y'])

        # dedupe any repeated (line,route,mode)
        unique_lines = list({
            (lr['line_id'], lr['route_id'], lr['mode'], lr.get('line_name', ''))
            for lr in info['lines_routes']
        })

        deduped = [
            {
                'line_id': lid,
                'route_id': rid,
                'mode': mode,
                'line_name': lname
            }
            for (lid, rid, mode, lname) in unique_lines
        ]


        # modes_list + predominant_mode
        modes = sorted({d['mode'] for d in deduped})
        cnt   = Counter(d['mode'] for d in deduped)
        if   not cnt:                     predominant = None
        elif len(cnt)==1:                 predominant = modes[0]
        elif cnt.most_common(2)[0][1] == cnt.most_common(2)[1][1]:
                                          predominant = "Mixed"
        else:                             predominant = cnt.most_common(1)[0][0]

        rows.append({
            'stop_id': sid,
            'name': info['name'],
            'lon': lon, 'lat': lat,
            'lines': deduped,
            'modes_list': modes,
            'predominant_mode': predominant
        })

    df = pd.DataFrame(rows)

    # 4) group by name → avg lon/lat, concat stop_ids, flatten lines, recompute modes
    agg = (
        df.groupby('name')
          .agg({
            'lon':              'mean',
            'lat':              'mean',
            'stop_id':          lambda x: list(x),
            'lines':            lambda ll: sum(ll, []),
            'modes_list':       lambda ml: sorted({m for sub in ml for m in sub}),
            'predominant_mode': lambda pm: Counter(pm).most_common(1)[0][0]
          })
          .reset_index()
    )

    # 5) rebuild geometry and return a GeoDataFrame in EPSG:4326
    agg['geometry'] = agg.apply(lambda r: Point(r.lon, r.lat), axis=1)
    return gpd.GeoDataFrame(agg, geometry='geometry', crs="EPSG:4326")


def assign_cantons_stops(stops_gdf, canton_gdf):
    """
    Spatially join stops to cantons—both assumed in EPSG:2056—then for any
    unassigned stops snap to the nearest canton within 10 000 m.  Returns
    joined GeoDataFrame still in EPSG:2056.
    """

    # 1) within‐join
    joined = gpd.sjoin(
        stops_gdf, 
        canton_gdf, 
        how='left', 
        predicate='within'
    ).rename(columns={'canton_name':'assigned_canton'}) \
     .drop(columns=['index_right'], errors='ignore')

    # 2) nearest‐canton fallback
    unassigned = joined['assigned_canton'].isna()
    if unassigned.any():
        canton_geoms = canton_gdf.geometry
        canton_names = canton_gdf['canton_name']
        for idx in joined[unassigned].index:
            pt    = joined.at[idx, 'geometry']
            dists = canton_geoms.distance(pt)
            i_min = dists.idxmin()
            if dists.loc[i_min] <= 10000:
                joined.at[idx, 'assigned_canton'] = canton_names.loc[i_min]

    return joined


def export_per_canton_stops(joined_gdf):
    """
    Export one GeoJSON per canton, keeping only stop name & geometry.
    Assumes joined_gdf is still in EPSG:2056 and has an 'assigned_canton' column.
    """
    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data", "matsim", "transit", "stops_by_canton")
    os.makedirs(output_dir, exist_ok=True)

    for canton, grp in joined_gdf.groupby('assigned_canton'):
        if pd.isna(canton):
            continue

        # keep only the stop name and geometry
        stops_simple = grp[[
            'name', 'lon', 'lat', 'stop_id', 'lines',
            'predominant_mode', 'assigned_canton', 'modes_list', 'geometry'
        ]].copy()

        # reproject to WGS84 just for writing
        out_path = os.path.join(output_dir, f"{clean_geo_name(canton)}_stops.geojson")
        # Reproject and manually write to GeoJSON, preserving lists
        stops_geojson = stops_simple.to_crs(epsg=4326).to_json()

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(stops_geojson)


def generate_modes_by_canton(joined_gdf):
    """
    Aggregate the unique modes_list values per assigned_canton
    directly from the joined GeoDataFrame (avoids re-reading GeoJSONs).
    """
    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data", "matsim", "transit")
    os.makedirs(output_dir, exist_ok=True)

    # group by canton, flatten and sort
    modes_per = (
        joined_gdf
        .dropna(subset=['assigned_canton'])
        .groupby('assigned_canton')['modes_list']
        .apply(lambda lists:
               sorted({mode for sub in lists for mode in sub}))
        .to_dict()
    )

    # write out
    output_path = os.path.join(output_dir, "modes_by_canton.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(modes_per, f, ensure_ascii=False, indent=2)


def build_route_lines(schedule_path, stops_gdf):
    """
    Build route geometries from schedule stops and write to GeoJSON.
    Any route with fewer than two stops is skipped.
    Assumes stops_gdf is in Swiss LV95 (EPSG:2056); output will be in WGS84.
    """

    # 0) unzip schedule if needed
    if schedule_path.endswith(".gz"):
        with gzip.open(schedule_path, 'rb') as f:
            tree = ET.parse(f)
    else:
        tree = ET.parse(schedule_path)

    # 1) parse XML
    root = tree.getroot()
    stop_coords = {}

    # 2) pull stop coordinates from stops_gdf
    #    ensure we're in EPSG:2056 so coords are in LV95 metres
    stops_lv95 = stops_gdf.to_crs(epsg=2056)

    for _, row in stops_lv95.iterrows():
            raw = row["stop_id"]
            # row.geometry is a shapely Point in LV95
            coords = (row.geometry.x, row.geometry.y)

            # support both "a,b,c" and ["a","b","c"]
            if isinstance(raw, str):
                ids = [s.strip() for s in raw.split(",") if s.strip()]
            elif isinstance(raw, (list, tuple)):
                ids = raw
            else:
                continue

            for sid in ids:
                stop_coords[sid] = coords


    # 3) build feature list
    features = []
    for line in root.findall('transitLine'):
        lid = line.attrib['id']
        for route in line.findall('transitRoute'):
            rid  = route.attrib.get('id', '')
            mode = route.findtext('transportMode')
            prof = route.find('routeProfile')
            if prof is None:
                continue

            # collect LV95 coords for each stop in this route
            coords = [
                stop_coords[rs.attrib['refId']]
                for rs in prof.findall('stop')
                if rs.attrib['refId'] in stop_coords
            ]

            # skip any degenerate route
            if len(coords) < 2:
                continue

            features.append({
                'type': 'Feature',
                'geometry': mapping(LineString(coords)),
                'properties': {
                    'line_id': lid,
                    'route_id': rid,
                    'mode': mode
                }
            })

    # 4) construct GeoDataFrame in LV95, then reproject to WGS84 for output
    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data", "matsim", "transit" , "routes")
    os.makedirs(output_dir, exist_ok=True)

    routes_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:2056")
    routes_gdf = routes_gdf.to_crs(epsg=4326)
    output_path = os.path.join(output_dir, "transit_routes.geojson")
    routes_gdf.to_file(output_path, driver="GeoJSON")

def compute_passenger_counts(joined_gdf, counts_df):
    """
    Generate per-canton passenger counts JSON files.
    joined_gdf: GeoDataFrame (possibly aggregated) with columns
                'stop_id' (either list[str] or comma-joined str) and 'assigned_canton'.
    counts_csv: path to pt_passenger_counts.csv with a column 'stop_id'.
    """

    # 1) pull out stop_id ↔ canton mapping and explode to one row per stop_id
    stop_canton = joined_gdf[['stop_id','assigned_canton']].copy()
    stop_canton = stop_canton.explode('stop_id').reset_index(drop=True)

    # 2) merge passenger counts to their canton
    merged = (
        counts_df
        .merge(stop_canton, on='stop_id', how='left')
        .dropna(subset=['assigned_canton'])
    )

    # 3) write one JSON per canton
    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data", "matsim", "transit", "per_canton_counts")
    os.makedirs(output_dir, exist_ok=True)

    for canton, grp in merged.groupby('assigned_canton'):
        out = []
        for (sid, lid), sub in grp.groupby(['stop_id','line_id']):
            data = sub[['time_bin','boardings','alightings']].to_dict('records')
            out.append({'stop_id': sid, 'line_id': lid, 'data': data})

        fname = f"{clean_geo_name(canton)}_counts.json"
        with open(os.path.join(output_dir, fname), 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)