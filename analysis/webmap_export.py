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


# ==== MERGED NETWORK LINKS ====     

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
    # NOTE: 'modes' is intentionally EXCLUDED here so it's NOT stored per-id.
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
      - properties.per_id: { "<id>": { per-id fields..., direction, arrow }, ... }
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
                "geometry": geom.copy() if isinstance(geom, dict) else geom,  # shallow copy
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
                    per_id_payload = dict(base_per_id_payload)  # copy
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
                per_id_payload["direction"] = "forward"  # base geometry defines forward
                entry["per_id"][str(feat_id)] = per_id_payload

            bucket.append(entry)

    # Finalize: inject per_id, summed daily_avg_volume, and unioned modes
    merged = []
    for bucket in groups.values():
        for entry in bucket:
            f = entry["feature"]
            per_id_map = entry["per_id"]

            # per_id mapping
            if per_id_map:
                f["properties"][per_id_key] = per_id_map

                # Sum the chosen field across per-id entries
                total = 0.0
                for v in per_id_map.values():
                    if sum_field in v:
                        try:
                            total += float(v[sum_field])
                        except (TypeError, ValueError):
                            pass
                f["properties"][sum_field] = total

                cap_total = 0.0
                for v in per_id_map.values():
                    if "capacity" in v:
                        try:
                            cap_total += float(v["capacity"])
                        except (TypeError, ValueError):
                            pass
                f["properties"]["capacity"] = cap_total

                freespeed_max = None
                for v in per_id_map.values():
                    if "freespeed" in v:
                        try:
                            fv = float(v["freespeed"])
                        except (TypeError, ValueError):
                            continue
                        if freespeed_max is None or fv > freespeed_max:
                            freespeed_max = fv
                # store in top-level 'freespeed' so your Mapbox color ramp still works
                f["properties"]["freespeed"] = freespeed_max if freespeed_max is not None else None
            else:
                if sum_field in f["properties"]:
                    del f["properties"][sum_field]

            # angle (safety)
            if "angle" not in f["properties"]:
                geom = f.get("geometry", {})
                if geom.get("type") == "LineString" and geom.get("coordinates"):
                    ang = _angle_for_segment(geom["coordinates"])
                    if ang is not None:
                        f["properties"]["angle"] = ang

            # modes (top-level): union across all member segments
            modes_union = entry.get("modes_union", set())
            # store as comma-separated string (matches your existing Mapbox filter logic)
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
    SAME logic as before, but reordered:
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
                hourly = {c: r.get(c, None) for c in hourly_avg_cols}
                link_summaries.append({
                    "link_id": r["id"],
                    "hourly_avg_volumes": hourly,
                    "daily_avg_volume": r.get("HRS0-24avg", r.get("daily_avg_volume")),
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

def export_inter_cantonal_stops(joined_gdf, volumes_df):
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