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
from shapely.ops import transform as shapely_transform
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

        # add histogram for combined data_type (still to be tested to make sure results are good)
        mc_all_vals = filtered_microcensus[data_type]
        syn_all_vals = filtered_synthetic[data_type]
        mc_all_w = (
            filtered_microcensus[WEIGHT_COL]
            if weighted and WEIGHT_COL in filtered_microcensus.columns
            else None
        )

        if len(mc_all_vals) > 0 and len(syn_all_vals) > 0:
            # bins based on synthetic, same logic as before
            q1, q3 = syn_all_vals.quantile(0.25), syn_all_vals.quantile(0.75)
            iqr = q3 - q1
            range_max = q3 + max_iqr * iqr
            bin_size = range_max / num_bins if range_max > 0 else 1.0  # safety

            bins_all = np.arange(0, max(mc_all_vals.max(), syn_all_vals.max()) + bin_size, bin_size)

            mc_hist_all, bins_returned = np.histogram(
                mc_all_vals, bins=bins_all, range=(0, mc_all_vals.max()),
                weights=mc_all_w, density=False
            )
            syn_hist_all, _ = np.histogram(
                syn_all_vals, bins=bins_all, range=(0, syn_all_vals.max()),
                density=False
            )

            if mc_hist_all.sum() > 0:
                mc_hist_all = (mc_hist_all / mc_hist_all.sum()) * 100
            if syn_hist_all.sum() > 0:
                syn_hist_all = (syn_hist_all / syn_hist_all.sum()) * 100

            mc_mean_all = (
                np.average(mc_all_vals, weights=mc_all_w)
                if weighted and mc_all_w is not None
                else mc_all_vals.mean()
            )
            syn_mean_all = syn_all_vals.mean()

            group_data["All"] = {
                "bin_width": bin_size,
                "bins": bins_returned.tolist(),
                "microcensus_histogram": mc_hist_all.tolist(),
                "synthetic_histogram": syn_hist_all.tolist(),
                "microcensus_mean": mc_mean_all,
                "synthetic_mean": syn_mean_all,
                "microcensus_sample_size": len(mc_all_vals),
                "synthetic_sample_size": len(syn_all_vals),
            }

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
        # Select the rows in grouped_data corresponding to Synthetic
        mask = grouped_data['dataset'] == 'Synthetic'

        # Aggregate counts for synthetic data by distance_category and aggregation column
        synthetic_counts = (
            data[data['dataset'] == 'Synthetic']
            .groupby(['distance_category', AGGREGATION_COL], observed=False)
            .size()
        )

        # Align synthetic_counts with grouped_data[mask] index to avoid length mismatch
        synthetic_counts_aligned = synthetic_counts.reindex(
            grouped_data.loc[mask, ['distance_category', AGGREGATION_COL]]
            .set_index(['distance_category', AGGREGATION_COL]).index,
            fill_value=0
        )

        # Assign safely
        grouped_data.loc[mask, 'count'] = synthetic_counts_aligned.values
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


# ==== MERGED NETWORK LINKS ====     

def _round_to(value, decimals=0):
    """Round to specified decimals, return None if not finite."""
    if value is None or not math.isfinite(value):
        return None
    factor = 10 ** decimals
    return round(value * factor) / factor

def _mps_to_kmh(mps):
    """Convert m/s to km/h, return None if not valid."""
    if mps is None:
        return None
    try:
        n = float(mps)
        return n * 3.6 if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None

def _to_num(v):
    """Convert to number, return None if not finite."""
    if v is None:
        return None
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None

def _norm_key(coords):
    fwd = tuple(map(tuple, coords))
    rev = tuple(map(tuple, reversed(coords)))
    return min(fwd, rev)

def _angle_for_segment(coords):
    if not coords or len(coords) < 2:
        return None
    x0, y0 = coords[0][:2]
    x1, y1 = coords[-1][:2]
    dx, dy = (x1 - x0), (y1 - y0)
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))

def _arrow_for_segment(coords):
    if not coords or len(coords) < 2:
        return None
    start_lon, start_lat = coords[0][:2]
    end_lon,   end_lat   = coords[-1][:2]
    if start_lon > end_lon:
        return "←"
    elif start_lon < end_lon:
        return "→"

## todo? add case for perfectly vertical roads, north should be <-, south should be ->

def _sanitize_for_json(obj):
    """Recursively replace NaN/inf with None (-> null in JSON)."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    else:
        return obj

def _parse_modes(val):
    """
    Accepts a string like 'car,car_passenger,truck' or a list; returns a list of unique modes.
    """
    modes = []
    if isinstance(val, str):
        modes = [m.strip() for m in val.split(",")]
    elif isinstance(val, (list, tuple)):
        modes = [str(m).strip() for m in val]
    # dedupe, keep stable order
    seen = set()
    uniq = []
    for m in modes:
        if m and m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq

def merge_geojson_segments_per_id(
    input_df,
    output_path,
    id_key="id",
    per_id_key="per_id",
    per_id_fields=("length", "freespeed", "capacity", "permlanes", "daily_avg_volume"),
    sum_field="daily_avg_volume",
    props_to_ignore=None,
    debug=True,
):
    """
    Merge reversed-duplicate LineStrings (A->B == B->A) ONLY IF the coordinate
    sequence is an EXACT forward or EXACT reverse match to the base geometry.

    Outputs per feature:
      - properties.angle : float degrees (0=east, 90=north)
      - properties.per_id_keys : pipe-delimited link IDs
      - properties.per_id_capacities : pipe-delimited capacities
      - properties.per_id_lengths : pipe-delimited lengths (rounded to 1 decimal)
      - properties.per_id_freespeeds : pipe-delimited freespeeds in km/h (rounded to 1 decimal)
      - properties.per_id_daily_avgs : pipe-delimited daily avg volumes
      - properties.searchable_text : all values concatenated for search
      - properties.daily_avg_volume : sum of per-id sum_field
      - properties.modes : comma-separated union across all member segments (top-level only)
    """

    data = input_df

    features = data.get("features", [])
    if props_to_ignore is None:
        props_to_ignore = set()
    else:
        props_to_ignore = set(props_to_ignore)

    # We exclude id/per_id/per-id fields from shared-prop merging, and also exclude 'modes'
    always_ignore = {id_key, per_id_key, *per_id_fields}
    ignore_set = props_to_ignore | always_ignore | {"modes"}
    
    groups = {}

    for feat in features:
        geom = feat.get("geometry", {})
        ftype = geom.get("type")
        coords = geom.get("coordinates", [])
        props = dict(feat.get("properties", {}))
        feat_id = props.pop(id_key, None)

        # collect modes for this source feature (used for link-level union)
        src_modes = set(_parse_modes(props.get("modes")))

        # Non-LineStrings: pass-through
        if ftype != "LineString" or not coords:
            key = ("__non_linestring__", id(feat))
            base_feature = {
                "type": "Feature",
                "geometry": geom.copy() if isinstance(geom, dict) else geom,
                "properties": {k: v for k, v in props.items() if k not in ignore_set},
            }
            entry = {
                "feature": base_feature,
                "per_id": {} if feat_id is None else {
                    str(feat_id): {
                        **{fld: props[fld] for fld in per_id_fields if fld in props},
                        "direction": None,
                        "arrow": None,
                    }
                },
                "base_coords": tuple(map(tuple, coords)) if coords else (),
                "base_id": str(feat_id) if feat_id is not None else None,
                "modes_union": set(src_modes),
            }
            groups.setdefault(key, []).append(entry)
            continue

        key = _norm_key(coords)
        fwd_tuple = tuple(map(tuple, coords))
        rev_tuple = tuple(map(tuple, reversed(coords)))

        # Build per-id payload (NO 'modes' per-id)
        base_per_id_payload = {}
        if feat_id is not None:
            for fld in per_id_fields:
                if fld in props:
                    base_per_id_payload[fld] = props[fld]
            base_per_id_payload["arrow"] = _arrow_for_segment(coords)

        bucket = groups.setdefault(key, [])
        placed = False

        # Try to merge into an existing entry that has exact fwd or exact rev coords
        for entry in bucket:
            base_coords = entry["base_coords"]
            eq_fwd = (fwd_tuple == base_coords)
            eq_rev = (rev_tuple == base_coords)
            if eq_fwd or eq_rev:
                if debug:
                    print(f"[merge check] base_id={entry['base_id']} curr_id={feat_id} "
                          f"eq_fwd={eq_fwd} eq_rev={eq_rev}")

                # Merge shared (non per-id) props (keep first on mismatch)
                base_props = entry["feature"]["properties"]
                for k, v in props.items():
                    if k in ignore_set:
                        continue
                    if k not in base_props:
                        base_props[k] = v
                    elif base_props[k] != v:
                        print(
                            f"[merge warn] Property mismatch on key '{k}' for merged segment. "
                            f"Keeping first value.\n"
                            f"  first: {base_props[k]!r}\n"
                            f"  this : {v!r}\n"
                            f"  base_id: {entry['base_id']}  curr_id: {feat_id}"
                        )

                # Update union of modes at the link level
                entry["modes_union"].update(src_modes)

                # Add per-id info
                if feat_id is not None and base_per_id_payload:
                    per_id_payload = dict(base_per_id_payload)
                    per_id_payload["direction"] = "forward" if eq_fwd else "reverse"
                    entry["per_id"][str(feat_id)] = per_id_payload

                placed = True
                break

        if not placed:
            if debug and key in groups:
                for entry in bucket:
                    base_coords = entry["base_coords"]
                    eq_fwd = (fwd_tuple == base_coords)
                    eq_rev = (rev_tuple == base_coords)
                    print(f"[no-merge] base_id={entry['base_id']} curr_id={feat_id} "
                          f"eq_fwd={eq_fwd} eq_rev={eq_rev} -> creating new group for this geometry")

            existing_angle = props.get("angle", None)
            angle = existing_angle if isinstance(existing_angle, (int, float)) else _angle_for_segment(coords)
            props_clean = {k: v for k, v in props.items() if k not in ignore_set}
            if angle is not None:
                props_clean["angle"] = angle

            base_feature = {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": list(coords) if isinstance(coords, list) else coords},
                "properties": props_clean,
            }
            entry = {
                "feature": base_feature,
                "per_id": {},
                "base_coords": fwd_tuple,
                "base_id": str(feat_id) if feat_id is not None else None,
                "modes_union": set(src_modes),
            }
            if feat_id is not None and base_per_id_payload:
                per_id_payload = dict(base_per_id_payload)
                per_id_payload["direction"] = "forward"
                entry["per_id"][str(feat_id)] = per_id_payload

            bucket.append(entry)

    # Finalize: inject flattened per_id arrays and searchable text
    merged = []
    for bucket in groups.values():
        for entry in bucket:
            f = entry["feature"]
            per_id_map = entry["per_id"]

            # Flatten per_id into pipe-delimited arrays
            if per_id_map:
                link_ids = []
                capacities = []
                lengths = []
                freespeeds = []
                daily_avgs = []
                permlanes_list = []
                arrows = []
                directions = []
                
                total_volume = 0.0
                total_capacity = 0.0
                max_freespeed = None
                
                for key in sorted(per_id_map.keys()):
                    data = per_id_map[key]
                    
                    # Extract and transform values exactly like client-side
                    length = _to_num(data.get("length"))
                    freespeed_mps = _to_num(data.get("freespeed"))
                    freespeed_kmh = _mps_to_kmh(freespeed_mps)
                    capacity = _to_num(data.get("capacity"))
                    daily_avg = _to_num(data.get("daily_avg_volume"))
                    permlanes = _to_num(data.get("permlanes"))
                    arrow = data.get("arrow", "")
                    direction = data.get("direction", "")
                    
                    # Add to arrays (with appropriate formatting)
                    link_ids.append(key)
                    
                    if capacity is not None:
                        capacities.append(str(int(capacity)))
                        total_capacity += capacity
                    else:
                        capacities.append("")
                        
                    if length is not None:
                        lengths.append(str(_round_to(length, 1)))
                    else:
                        lengths.append("")
                        
                    if freespeed_kmh is not None:
                        freespeeds.append(str(_round_to(freespeed_kmh, 1)))
                        if max_freespeed is None or freespeed_kmh > max_freespeed:
                            max_freespeed = freespeed_kmh
                    else:
                        freespeeds.append("")
                        
                    if daily_avg is not None:
                        daily_avgs.append(str(int(daily_avg)))
                        total_volume += daily_avg
                    else:
                        daily_avgs.append("")
                    
                    if permlanes is not None:
                        permlanes_list.append(str(int(permlanes)))
                    else:
                        permlanes_list.append("")
                    
                    # Arrow and direction are already strings (or empty)
                    arrows.append(str(arrow) if arrow else "")
                    directions.append(str(direction) if direction else "")
                
                # Store as pipe-delimited strings
                f["properties"]["per_id_keys"] = "|".join(link_ids)
                f["properties"]["per_id_capacities"] = "|".join(capacities)
                f["properties"]["per_id_lengths"] = "|".join(lengths)
                f["properties"]["per_id_freespeeds"] = "|".join(freespeeds)
                f["properties"]["per_id_daily_avgs"] = "|".join(daily_avgs)
                f["properties"]["per_id_permlanes"] = "|".join(permlanes_list)
                f["properties"]["per_id_arrows"] = "|".join(arrows)
                f["properties"]["per_id_directions"] = "|".join(directions)
                
                # Create searchable text (include all non-empty values)
                all_values = link_ids + [c for c in capacities if c] + [l for l in lengths if l] + \
                            [f for f in freespeeds if f] + [d for d in daily_avgs if d]
                modes_str = f["properties"].get("modes", "")
                if modes_str:
                    all_values.append(modes_str)
                f["properties"]["searchable_text"] = "|".join(all_values).lower()
                
                # Set aggregate values
                f["properties"]["daily_avg_volume"] = total_volume
                f["properties"]["capacity"] = total_capacity
                f["properties"]["freespeed"] = max_freespeed
            else:
                # Remove fields if no per_id data
                for key in ["daily_avg_volume", "capacity", "freespeed",
                        "per_id_keys", "per_id_capacities", "per_id_lengths", 
                        "per_id_freespeeds", "per_id_daily_avgs", "per_id_permlanes",
                        "per_id_arrows", "per_id_directions", "searchable_text"]:
                    f["properties"].pop(key, None)

            # angle (safety)
            if "angle" not in f["properties"]:
                geom = f.get("geometry", {})
                if geom.get("type") == "LineString" and geom.get("coordinates"):
                    ang = _angle_for_segment(geom["coordinates"])
                    if ang is not None:
                        f["properties"]["angle"] = ang

            # modes (top-level): union across all member segments
            modes_union = entry.get("modes_union", set())
            f["properties"]["modes"] = ",".join(sorted(modes_union)) if modes_union else ""

            merged.append(f)

    out = {
        "type": "FeatureCollection",
        **({"crs": data["crs"]} if "crs" in data else {}),
        "features": merged,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(out), f, ensure_ascii=False, indent=2)

    print(f"Merged GeoJSON written to: {output_path}")

def export_merged_segments_by_canton(
    network_gdf: gpd.GeoDataFrame,
    cantons_gdf: gpd.GeoDataFrame,
    linkstats_path: str,
    output_dir: str,
    skip_cantons=None,
    id_col="link_id",
    canton_name_col="canton_name",
    network_modes_col="modes",
    target_crs="EPSG:4326",
    write_link_hourly_json=True,
):
    """
    1) Build a MASTER on the full network (rename id→'id', normalize 'modes', join daily_avg_volume)
    2) For each canton: select/intersect geometry ONLY, keep attributes, then export/merge.
    """
    os.makedirs(output_dir, exist_ok=True)
    if skip_cantons is None:
        skip_cantons = []

    # --- linkstats ----------------------------------------------------
    linkstats = pd.read_csv(linkstats_path, sep="\t", comment="#", dtype=str)
    linkstats.columns = linkstats.columns.str.strip()
    linkstats = linkstats.apply(pd.to_numeric, errors="ignore")
    if "LINK" not in linkstats.columns or "HRS0-24avg" not in linkstats.columns:
        raise ValueError("linkstats must contain 'LINK' and 'HRS0-24avg' columns.")

    # pre-compute hourly cols for optional JSON
    hourly_avg_cols = [
        c for c in linkstats.columns
        if c.startswith("HRS") and c.endswith("avg") and c != "HRS0-24avg"
    ]

    # --- CRS reconcile ------------------------------------------------
    if network_gdf.crs is None:
        raise ValueError("network_gdf must have a CRS.")
    if cantons_gdf.crs is None:
        cantons_gdf = cantons_gdf.set_crs(network_gdf.crs)
    elif cantons_gdf.crs != network_gdf.crs:
        cantons_gdf = cantons_gdf.to_crs(network_gdf.crs)

    # --- add attributes on the full network -------------
    # Drop Z on LineStrings
    def _drop_z_linestring(g):
        if g is None or g.is_empty:
            return g
        # ignore the z argument for any input geometry
        return shapely_transform(lambda x, y, z=None: (x, y), g)

    net = network_gdf.copy()
    net["geometry"] = net["geometry"].map(_drop_z_linestring)
    net = net[net.geometry.notna() & ~net.geometry.is_empty]
    net = net[net.geometry.geom_type == "LineString"]

    # keep requried attributes
    essential = [id_col, "geometry", "length", "freespeed", "capacity", "permlanes", network_modes_col]
    missing = [c for c in essential if c not in net.columns]
    if missing:
        raise ValueError(f"Network missing columns: {missing}")

    master = net[essential].rename(columns={id_col: "id", network_modes_col: "modes"}).copy()
    master["id"] = master["id"].astype(str)
    linkstats["LINK"] = linkstats["LINK"].astype(str)

    # Normalize modes once
    master["modes"] = master["modes"].apply(lambda v: ",".join(sorted(set(_parse_modes(v)))))

    # Join daily average volume ON THE FULL NETWORK
    master = master.merge(
        linkstats[["LINK", "HRS0-24avg"]],
        left_on="id", right_on="LINK", how="left"
    ).rename(columns={"HRS0-24avg": "daily_avg_volume"}).drop(columns=["LINK"])

    # --- Per-canton export (clip geometries) -------
    canton_modes = {}

    for _, row in cantons_gdf.iterrows():
        canton_name = row[canton_name_col]
        if canton_name in skip_cantons:
            print(f"Skipping {canton_name}")
            continue

        print(f"Processing {canton_name}")
        geom = row.geometry

        # Select from MASTER
        sub = master[master.intersects(geom)].copy()
        if sub.empty:
            print(f"  → No features in {canton_name}")
            continue

        # Clip geometry ONLY; keep attributes (id, modes, daily_avg_volume, etc.)
        sub["geometry"] = sub.geometry.intersection(geom)
        sub = sub[~sub.geometry.is_empty]

        # modes summary per canton
        mode_set = set()
        for s in sub["modes"]:
            mode_set.update(_parse_modes(s))
        canton_modes[clean_geo_name(canton_name)] = sorted(mode_set)

        # optional per-link hourly JSON 
        if write_link_hourly_json:
            merged_full = sub.merge(
                linkstats[["LINK", "HRS0-24avg"] + hourly_avg_cols].rename(columns={"LINK": "id"}),
                on="id", how="left"
            )

            link_summaries = []
            for _, r in merged_full.iterrows():
                # extract hourly volumes in the correct order (0–23)
                hourly_values = [
                    _to_num(r.get(c, None)) for c in sorted(
                        hourly_avg_cols,
                        key=lambda col: int(col.split("HRS")[1].split("-")[0])
                    )
                ]

                link_summaries.append({
                    "link_id": str(r["id"]),
                    "hourly_avg_volumes": hourly_values,  # compact array instead of dict
                    "daily_avg_volume": _to_num(
                        r.get("HRS0-24avg", r.get("daily_avg_volume"))
                    ),
                })
            json_path = os.path.join(output_dir, f"{clean_geo_name(canton_name)}_link_traffic_volumes.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(link_summaries, f, indent=2)
            print(f"Saved: {json_path}")

        # to WGS84 + in-memory FeatureCollection
        gdf_wgs84 = sub.to_crs(target_crs)
        fc = json.loads(gdf_wgs84.to_json())

        out_path = os.path.join(output_dir, f"{clean_geo_name(canton_name)}_merged_segments.geojson")
        merge_geojson_segments_per_id(
            input_df=fc,
            output_path=str(out_path),
            id_key="id", 
            per_id_key="per_id",
            per_id_fields=("length", "freespeed", "capacity", "permlanes", "daily_avg_volume"),
            sum_field="daily_avg_volume",
            props_to_ignore=None,
            debug=False,
        )

    # modes_by_canton.json (to keep track of which modes exist per canton for filtering)
    modes_path = os.path.join(os.path.dirname(output_dir), "modes_by_canton.json")
    with open(modes_path, "w", encoding="utf-8") as f:
        json.dump(canton_modes, f, indent=2)
    print(f"Saved mode summary to: {modes_path}")

# PROCESS JSONS FOR GRAPHS

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
        .assign(assigned_canton=lambda df: df['assigned_canton'].apply(clean_geo_name))
        .groupby('assigned_canton')['modes_list']
        .apply(lambda lists:
               sorted({mode for sub in lists for mode in sub}))
        .to_dict()
    )

    # write out
    output_path = os.path.join(output_dir, "transit_modes_by_canton.json")
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

def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _normalize_stop_ids_col(df):
    """Ensure df['stop_id'] is a list[str] for explode()."""
    s = df["stop_id"]
    # Fast path: many values already list-like? fall back safely
    def as_list(x):
        if isinstance(x, list): return x
        if pd.isna(x): return []
        if isinstance(x, str) and ("," in x):
            return [t.strip() for t in x.split(",") if t.strip()]
        return [x]
    return s.apply(as_list)

def compute_passenger_counts(joined_gdf: pd.DataFrame, counts_df: pd.DataFrame):
    """
    Faster pipeline:
      1) Build stop_id -> assigned_canton dict once (explode only the mapping, not the fact table)
      2) Pre-aggregate counts_df by (stop_id, line_id) to lists
      3) Map canton onto that aggregated table
      4) Split & write per canton
    """

    # ---- 1) stop -> canton mapping (small) ----
    map_df = joined_gdf[["stop_id", "assigned_canton"]].copy()
    map_df["stop_id"] = _normalize_stop_ids_col(map_df)
    map_df = map_df.explode("stop_id", ignore_index=True)
    map_df = map_df.dropna(subset=["stop_id", "assigned_canton"])
    # If a stop_id appears multiple times (rare), keep first (or choose your rule)
    map_df = map_df.drop_duplicates(subset=["stop_id"])
    stop_to_canton = pd.Series(map_df["assigned_canton"].values, index=map_df["stop_id"].values)

    # ---- 2) pre-aggregate counts_df once (heavy but only 2 keys) ----
    counts = counts_df[["stop_id","line_id","time_bin","boardings","alightings"]].copy()

    # Ensure numeric
    for col in ("boardings", "alightings"):
        counts[col] = pd.to_numeric(counts[col], errors="coerce").fillna(0)

    # Optional: keep time_bin order stable without full sort by pre-sorting only necessary cols
    counts = counts.sort_values(["stop_id","line_id","time_bin"], kind="stable")

    agg = (
        counts.groupby(["stop_id","line_id"], sort=False, as_index=False)
              .agg({
                  "time_bin": list,
                  "boardings": list,
                  "alightings": list,
              })
    )

    # ---- 3) map canton after aggregation ----
    agg["assigned_canton"] = agg["stop_id"].map(stop_to_canton)
    agg = agg.dropna(subset=["assigned_canton"])

    # Optional memory/speed: category helps splitting/groupby
    if agg["assigned_canton"].dtype != "category":
        agg["assigned_canton"] = agg["assigned_canton"].astype("category")

    # Build per-row 'data' payloads using list(zip(...)) which is fast
    # (Construct once per (stop_id,line_id))
    agg["data"] = [
        [{"time_bin": tb, "boardings": b, "alightings": a}
         for tb, b, a in zip(tbs, bs, als)]
        for tbs, bs, als in zip(agg["time_bin"], agg["boardings"], agg["alightings"])
    ]
    agg = agg.drop(columns=["time_bin","boardings","alightings"])

    # ---- 4) write per-canton files (I/O often the real bottleneck) ----
    output_dir = os.path.join(
        DEFAULT_WORKDIR, "public", "data", "matsim", "transit", "per_canton_counts"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Avoid Python for-loops inside inner groups; just loop cantons once
    for canton, df_c in agg.groupby("assigned_canton", sort=False):
        # Construct final records in one go
        out_records = [
            {"stop_id": sid, "line_id": lid, "data": data}
            for sid, lid, data in zip(df_c["stop_id"].values, df_c["line_id"].values, df_c["data"].values)
        ]
        fname = f"{clean_geo_name(str(canton))}_counts.json"
        with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
            f.write(_dumps(out_records))

def _to_list(x):
    if isinstance(x, list): return x
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    return [x]

def _extract_line_ids(lines):
    # lines is typically a list[dict], but be defensive
    if isinstance(lines, (list, tuple)):
        out = []
        for d in lines:
            if isinstance(d, dict):
                lid = d.get("line_id")
                if lid is not None:
                    out.append(lid)
        return set(out)
    # if someone stored a single dict
    if isinstance(lines, dict):
        lid = lines.get("line_id")
        return {lid} if lid is not None else set()
    # otherwise nothing useful
    return set()

def export_inter_cantonal_stops(joined_gdf: gpd.GeoDataFrame, volumes_df: pd.DataFrame):
    """
    Faster version:
      - vectorized line→cantons detection
      - one-time volume aggregation
      - dedupe by canonical stop_key
      - bulk reprojection
    Requires column 'assigned_canton' to decide inter-cantonal lines.
    """
    # ---- sanity checks (fail fast but friendly) ----
    required_cols = {"stop_id", "lines", "geometry"}
    missing = [c for c in required_cols if c not in joined_gdf.columns]
    if missing:
        raise ValueError(f"joined_gdf is missing columns: {missing}")
    if "assigned_canton" not in joined_gdf.columns:
        raise ValueError("joined_gdf must contain 'assigned_canton' to determine inter-cantonal lines.")
    if not {"stop_id", "boardings", "alightings"}.issubset(volumes_df.columns):
        raise ValueError("volumes_df must contain 'stop_id', 'boardings', 'alightings'.")

    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data", "matsim", "transit", "stops_by_canton")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "inter_cantonal_stops.geojson")

    # 0) prune rows without canton early
    gdf = joined_gdf[~joined_gdf["assigned_canton"].isna()].copy()

    # 1) normalize stop_ids and line_ids
    gdf["stop_ids_list"] = gdf["stop_id"].map(_to_list)
    gdf["line_ids_set"] = gdf["lines"].map(_extract_line_ids)
    gdf["assigned_canton"] = gdf["assigned_canton"].apply(clean_geo_name)
    # 2) line_id -> distinct canton count (vectorized)
    # explode line sets to rows; skip entries with no lines
    tmp = gdf.loc[gdf["line_ids_set"].map(bool), ["assigned_canton", "line_ids_set"]].copy()
    # convert sets to lists before explode
    tmp["line_ids_list"] = tmp["line_ids_set"].map(list)
    exploded = tmp.explode("line_ids_list").rename(columns={"line_ids_list": "line_id"})
    cantons_per_line = exploded.groupby("line_id", sort=False)["assigned_canton"].nunique()
    inter_cantonal_lines = set(cantons_per_line[cantons_per_line > 1].index)

    # 3) keep only stops on inter-cantonal lines
    if inter_cantonal_lines:
        mask = gdf["line_ids_set"].apply(lambda s: bool(s & inter_cantonal_lines))
        gdf = gdf[mask].copy()
    else:
        gdf = gdf.iloc[0:0].copy()  # nothing qualifies

    if gdf.empty:
        # write a valid empty GeoJSON
        empty = gpd.GeoDataFrame(geometry=[], crs=(joined_gdf.crs or "EPSG:4326"))
        empty.to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")
        print(f"Saved 0 inter-cantonal stops with volume to {output_path}")
        return

    # 4) dedupe by canonical stop_key
    gdf["stop_key"] = gdf["stop_ids_list"].apply(lambda lst: tuple(sorted(lst)))
    gdf = gdf.drop_duplicates(subset="stop_key").copy()

    # 5) pre-aggregate volumes once
    vol = volumes_df[["stop_id", "boardings", "alightings"]].copy()
    vol["boardings"] = vol["boardings"].fillna(0)
    vol["alightings"] = vol["alightings"].fillna(0)
    vol["volume"] = vol["boardings"] + vol["alightings"]
    volume_by_stop = vol.groupby("stop_id", dropna=False)["volume"].sum().to_dict()

    gdf["volume"] = gdf["stop_ids_list"].apply(lambda ids: int(sum(volume_by_stop.get(s, 0) for s in ids)))

    # 6) keep props (adapt to what you actually have; these are safe w/ your dtypes)
    keep_keys = ["name", "stop_id", "lines", "assigned_canton", "predominant_mode", "modes_list"]
    present = [k for k in keep_keys if k in gdf.columns]
    gdf["properties"] = gdf.apply(lambda r: {k: r[k] for k in present} | {"volume": r["volume"]}, axis=1)

    # 7) bulk reproject to WGS84
    src = joined_gdf.crs or "EPSG:4326"
    gdf = gdf.set_geometry("geometry")
    gdf = gdf.set_crs(src, allow_override=True)
    gdf_wgs84 = gdf.to_crs("EPSG:4326")

    # 8) write GeoJSON (keep nested properties as-is)
    features = [
    {
        "type": "Feature",
        "geometry": mapping(gdf_wgs84.geometry.iloc[i]),
        "properties": gdf_wgs84.properties.iloc[i],
    }
    for i in range(len(gdf_wgs84))
    ]
    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(features)} inter-cantonal stops with volume to {output_path}")


def export_inter_cantonal_stops_2(joined_gdf, volumes_df):
    """
    Identify inter-cantonal stops and export them as GeoJSON with volume.
    """
    output_dir = os.path.join(DEFAULT_WORKDIR, "public", "data", "matsim", "transit", "stops_by_canton")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "inter_cantonal_stops.geojson")

    # Transformer from original CRS to WGS84
    transformer = Transformer.from_crs(joined_gdf.crs, "EPSG:4326", always_xy=True)

    # Step 1: Build mapping from line_id → set of cantons
    line_to_cantons = defaultdict(set)
    all_features = []

    for _, row in joined_gdf.iterrows():
        
        if pd.isna(row["assigned_canton"]):
            continue
        
        props = row.drop("geometry").to_dict()
        geometry = row.geometry
        lines = props.get("lines", [])
        for line in lines:
            line_id = line.get("line_id")
            if line_id:
                line_to_cantons[line_id].add(props.get("assigned_canton"))
        all_features.append({"properties": props, "geometry": geometry})

    # Step 2: Identify inter-cantonal lines
    inter_cantonal_lines = {lid for lid, cantons in line_to_cantons.items() if len(cantons) > 1}

    # Step 3: Collect final features with deduplication and volume assignment
    seen_keys = set()
    final_features = []

    for f in all_features:
        props = f["properties"]
        geometry = f["geometry"]
        lines = props.get("lines", [])
        stop_ids = props.get("stop_id", [])

        stop_key = tuple(sorted(stop_ids)) if isinstance(stop_ids, list) else (stop_ids,)
        if stop_key in seen_keys:
            continue

        if any(line.get("line_id") in inter_cantonal_lines for line in lines):
            seen_keys.add(stop_key)

            # Compute volume
            ids = stop_ids if isinstance(stop_ids, list) else [stop_ids]
            total_volume = sum(
                volumes_df[volumes_df['stop_id'].isin(ids)]['boardings'].fillna(0) +
                volumes_df[volumes_df['stop_id'].isin(ids)]['alightings'].fillna(0)
            )

            # Retain only desired props
            keep_keys = ["name", "stop_id", "lines", "assigned_canton", "predominant_mode", "modes_list"]
            filtered_props = {k: props[k] for k in keep_keys if k in props}
            filtered_props["volume"] = total_volume
            filtered_props["assigned_canton"] = clean_geo_name(filtered_props["assigned_canton"])
            # Reproject geometry to WGS84
            geometry_wgs = shapely_transform(transformer.transform, geometry)

            final_features.append({
                "type": "Feature",
                "geometry": mapping(geometry_wgs),
                "properties": filtered_props
            })

    # Save as GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": final_features
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    print(f"Saved {len(final_features)} inter-cantonal stops with volume to {output_path}")


# === VOLUMES BY LINK LINE ===

def _generate_15min_bins():
    """Return list of 96 time labels from 00:00 to 23:45 inclusive."""
    bins = []
    t = datetime.strptime("00:00", "%H:%M")
    for _ in range(96):
        bins.append(t.strftime("%H:%M"))
        t += timedelta(minutes=15)
    return bins

_TIME_BINS = _generate_15min_bins()
_TIME_BIN_SET = set(_TIME_BINS)


def _clean_link_id(link_id):
    """For each part separated by '_', remove from first ':' onward.
    Example: 'A:foo_B:bar_123' -> 'A_B_123'
    """
    parts = str(link_id).split("_")
    return "_".join(part.split(":")[0] for part in parts)


def _normalize_time_bin(tb):
    """Normalize a time bin to 'HH:MM' on a 15-minute grid.
    Accepts int/float index [0..95], or 'H:MM'/'HH:MM' strings.
    Returns None for anything unrecognised.
    """
    if pd.isna(tb):
        return None

    if isinstance(tb, (int, float)) and not isinstance(tb, bool):
        if math.isfinite(tb):
            idx = int(tb)
            if 0 <= idx < 96:
                return _TIME_BINS[idx]
        return None

    s = str(tb).strip()
    try:
        dt = datetime.strptime(s, "%H:%M")
        if dt.minute % 15 == 0:
            return dt.strftime("%H:%M")
        return None
    except ValueError:
        pass

    try:
        parts = s.split(":")
        if len(parts) == 2:
            hour, minute = int(parts[0]), int(parts[1])
            if 0 <= hour <= 23 and minute in (0, 15, 30, 45):
                return f"{hour:02d}:{minute:02d}"
    except Exception:
        pass
    return None


def build_volumes_by_link_line(pt_link_volumes_path):
    """
    Build per-canton JSON files with PT link volumes broken down by line.

    Reads:
      - pt_link_volumes.csv.gz  (param)
      - transit_modes_by_canton.json   (from DEFAULT_WORKDIR)
      - transit_routes.geojson         (from DEFAULT_WORKDIR)
      - {canton}_merged_segments.geojson (from DEFAULT_WORKDIR)

    Writes:
      - volumes_by_link_line/pt_link_volumes_by_link_line_{canton}.json
        per canton under DEFAULT_WORKDIR/public/data/matsim/transit/
    """
    transit_dir = os.path.join(DEFAULT_WORKDIR, "public", "data", "matsim", "transit")
    network_dir = os.path.join(DEFAULT_WORKDIR, "public", "data", "matsim")
    output_dir = os.path.join(transit_dir, "volumes_by_link_line")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load CSV ---
    print("  Loading pt_link_volumes CSV...")
    df = pd.read_csv(pt_link_volumes_path, compression="gzip")
    df["linkId"] = df["linkId"].astype(str).apply(_clean_link_id)
    df["timeBin"] = df["timeBin"].apply(_normalize_time_bin)
    df = df[df["timeBin"].notna()].copy()

    if not pd.api.types.is_numeric_dtype(df["passengers"]):
        df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce").fillna(0.0)
    else:
        df["passengers"] = df["passengers"].fillna(0.0).astype(float)

    # --- Load modes per canton ---
    modes_path = os.path.join(transit_dir, "transit_modes_by_canton.json")
    with open(modes_path, "r", encoding="utf-8") as f:
        pt_modes_dict = json.load(f)

    # --- Load route metadata (line_id -> mode, lineName) ---
    routes_path = os.path.join(transit_dir, "routes", "transit_routes.geojson")
    with open(routes_path, "r", encoding="utf-8") as f:
        routes_geo = json.load(f)

    line_metadata = {}
    for feat in routes_geo.get("features", []):
        props = feat.get("properties", {})
        line_id = props.get("line_id")
        if line_id is None:
            continue
        line_id = str(line_id)
        if line_id.lower() == "none":
            continue
        line_metadata[line_id] = {
            "mode": props.get("mode"),
            "lineName": props.get("line_name") or props.get("name"),
        }

    # --- Process each canton ---
    for canton, pt_modes in pt_modes_dict.items():
        pt_modes_set = set(pt_modes if isinstance(pt_modes, list) else [])
        network_path = os.path.join(network_dir, f"{canton}_merged_segments.geojson")
        output_path = os.path.join(output_dir, f"pt_link_volumes_by_link_line_{canton}.json")

        if not os.path.exists(network_path):
            print(f"  Skipping {canton} -- missing merged_segments")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            continue

        with open(network_path, "r", encoding="utf-8") as f:
            network = json.load(f)

        # Collect valid PT links from merged segments via per_id_keys
        valid_links_modes = defaultdict(set)
        for feat in network.get("features", []):
            props = feat.get("properties", {}) or {}
            modes_str = props.get("modes", "")
            seg_modes = {m.strip() for m in modes_str.split(",") if m.strip()}
            allowed = seg_modes & pt_modes_set
            if not allowed:
                continue

            per_id_keys = props.get("per_id_keys", "")
            if isinstance(per_id_keys, str):
                for link_id in per_id_keys.split("|"):
                    link_id = link_id.strip()
                    if link_id:
                        valid_links_modes[_clean_link_id(link_id)].update(allowed)

        if not valid_links_modes:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            print(f"  {canton}: 0 links (no PT modes in merged segments)")
            continue

        df_canton = df[df["linkId"].isin(valid_links_modes)].copy()

        if df_canton.empty:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            print(f"  {canton}: 0 links (no volumes after filtering)")
            continue

        # Aggregate by (link, line, timeBin)
        result = defaultdict(lambda: defaultdict(
            lambda: {"total": 0.0, "timeBins": defaultdict(float), "mode": None, "lineName": None}
        ))

        for _, row in df_canton.iterrows():
            link = str(row["linkId"])
            line = str(row["lineId"])
            tb = row["timeBin"]
            count = float(row["passengers"])

            node = result[link][line]
            node["total"] += count
            node["timeBins"][tb] += count

            meta = line_metadata.get(line, {})
            node["mode"] = meta.get("mode")
            node["lineName"] = meta.get("lineName") or row.get("lineName")

        # Build output list, stripping zero time bins and minifying
        final_list = []
        for link_id, lines in result.items():
            modes_list = sorted(valid_links_modes.get(link_id, []))
            link_obj = {
                "link_id": link_id,
                "modes_list": modes_list,
                "lines": [],
            }

            for line_id, line_data in lines.items():
                # Only include non-zero time bins
                hourly = {}
                for tb, val in line_data["timeBins"].items():
                    if tb in _TIME_BIN_SET and val != 0:
                        hourly[tb] = float(val)

                daily_avg = sum(hourly.values())

                link_obj["lines"].append({
                    "line_id": line_id,
                    "line_name": line_data["lineName"],
                    "mode": line_data["mode"],
                    "hourly_avg_volumes": hourly,
                    "daily_avg_volume": daily_avg,
                })

            link_obj["lines"].sort(key=lambda d: (str(d.get("mode")), str(d["line_id"])))
            final_list.append(link_obj)

        final_list.sort(key=lambda d: d["link_id"])

        # Write minified JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_list, f, ensure_ascii=False, separators=(",", ":"))

        print(f"  {canton}: {len(final_list)} links -> {os.path.basename(output_path)}")