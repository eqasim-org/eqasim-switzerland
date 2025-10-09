import pickle
import os 
import numpy as np
import pandas as pd
import geopandas as gpd
import unicodedata
from tqdm import tqdm
import json

def remove_accents(text):
    if isinstance(text, str):
        return ''.join(
            c for c in unicodedata.normalize('NFKD', text)
            if not unicodedata.combining(c)
        )
    return text

def add_canton_name(dataset, x_col, y_col, coord_system=2056, distance=3500, canton_name="canton_name"):
    """
    Adds the cantons of a datapoint based on coordinates.
    Adapted from Andrew 

    Args:
        x_col: column for x-coordinate
        y_col: column for y-coordinate
        coord_system: input coordinate system (default 2056 for LV95)
        distance: maximum distance for nearest canton matching
        add_coords: if True, adds longitude/latitude columns
    """
    if x_col not in dataset.columns or y_col not in dataset.columns:
        raise ValueError(f"Columns '{x_col}' and '{y_col}' must exist in the provided file.")

    geojson_path = "/cluster/work/ivt_vpl/anding/data/TLM_KANTONSGEBIET.json"
    canton_boundaries = gpd.read_file(geojson_path).to_crs(epsg=coord_system)

    geometry = gpd.points_from_xy(dataset[x_col], dataset[y_col])

    dataset_gdf = gpd.GeoDataFrame(dataset, geometry=geometry, crs=f"EPSG:{coord_system}")

    print("Finished assigning points!")

    within_canton = dataset_gdf.sjoin(canton_boundaries[['KANTONSNUMMER', 'NAME', 'geometry']], how="left", predicate='within')

    print("Finished within canton matches!")
    print("Checking non-matches...")

    non_match = within_canton.loc[within_canton['NAME'].isna()]
    match = within_canton.loc[within_canton['NAME'].notna()]

    non_match = non_match.drop(columns=["index_right", 'KANTONSNUMMER', 'NAME'], errors="ignore")
    match_closest = non_match.sjoin_nearest(canton_boundaries[['KANTONSNUMMER', 'NAME', 'geometry']], how="left", max_distance=distance, distance_col="distance")

    print("Non-matches finished!")
    print("Concatenating results...")

    result = pd.concat([match, match_closest], ignore_index=True)

    result_filt = result.drop(columns=["geometry", "index_right"], errors="ignore")
    result_filt = result_filt.rename(columns={
        "NAME": canton_name,
        "KANTONSNUMMER": "canton_id"
    })

    missing_matches = len(result_filt.loc[result_filt[canton_name].isna()])

    if missing_matches > 0:
        print(f'Warning: {missing_matches} trips not assigned a canton (try increasing the distance parameter)')

    assert len(dataset) == len(result_filt), "Input/Output number of rows not matching"

    result_df = pd.DataFrame(result_filt)
    result_df[canton_name] = result_df[canton_name].apply(remove_accents)
    return result_df

def preprocess(path):
    path = '/cluster/project/cmdp/chaoch/switzerland_data/cache/matsim.simulation.run__a08f881cee4c481521a1ae6c8d50fea3.cache/simulation_output/output_trips.csv'
    
    # Read the CSV file
    df = pd.read_csv(path, sep=';')
    
    # Print all column names
    print("Columns in the dataset:")
    print(df.columns.tolist())
    
    # Create copies and add canton names separately
    df_starts = add_canton_name(df.copy(), 'start_x', 'start_y', canton_name='start_canton')
    df_ends = add_canton_name(df.copy(), 'end_x', 'end_y', canton_name='end_canton')
    
    # Keep only the new canton columns from df_ends
    df_ends = df_ends[['end_canton', 'canton_id']]
    
    # Merge the results back
    df = pd.merge(df_starts, df_ends, left_index=True, right_index=True)
    print(list(df.columns))
    print("\nFirst 10 rows of the merged dataset:")
    print(df_starts.head(10))
    # Save the DataFrame to CSV
    output_path = 'output_trips.csv'
    df.to_csv(output_path, sep=';', index=False)
    print(f"Data saved to {output_path}")

def to_time_bin(time_str):
    # Parse hours, minutes, seconds from string
    hours, minutes, _ = map(int, time_str.split(":"))
    
    total_minutes = hours * 60 + minutes
    
    bin_minutes = (total_minutes // 15) * 15
    
    binned_hours = bin_minutes // 60
    binned_minutes = bin_minutes % 60
    
    return f"{binned_hours:02d}:{binned_minutes:02d}"

def get_canton_trip_data_combined(path="output_trips.csv"):
    """
    Outputs one JSON file per canton, containing records for both when the canton is treated as origin and as destination.
    """
    df = pd.read_csv(path, sep=';')
    df["time_bin"] = df["dep_time"].apply(to_time_bin)

    df = df[[
        "start_canton", "end_canton", "main_mode", "end_activity_type", "time_bin"
    ]].rename(columns={
        "start_canton": "origin",
        "end_canton": "destination",
        "main_mode": "mode",
        "end_activity_type": "purpose"
    })

    # Group by all columns to get trip counts per time bin
    time_counts = df.groupby(
        ["origin", "destination", "mode", "purpose", "time_bin"]
    ).size().reset_index(name="trip_count")

    # Restructure the data to create nested time_bins for both roles
    nested_data = {}
    for _, row in time_counts.iterrows():
        key = (row["origin"], row["destination"], row["mode"], row["purpose"])
        if key not in nested_data:
            nested_data[key] = {
                "destination": row["destination"],
                "origin": row["origin"],
                "mode": row["mode"],
                "purpose": row["purpose"],
                "time_bins": {}
            }
        nested_data[key]["time_bins"][row["time_bin"]] = row["trip_count"]

    output_dir = "plot_data_combined"
    os.makedirs(output_dir, exist_ok=True)

    canton_list = np.unique(np.concatenate([df["origin"].unique(), df["destination"].unique()]))
    all_modes = df["mode"].unique()
    all_purposes = df["purpose"].unique()

    for canton in canton_list:
        records = []
        # As origin
        for key, data in nested_data.items():
            if key[0] == canton:
                record = data.copy()
                record["role"] = "origin"
                records.append(record)
        # As destination
        for key, data in nested_data.items():
            if key[1] == canton:
                record = data.copy()
                record["role"] = "destination"
                records.append(record)
        # Ensure all cantons have an entry for trips to themselves for both roles
        for mode in all_modes:
            for purpose in all_purposes:
                key = (canton, canton, mode, purpose)
                if key not in nested_data:
                    entry = {
                        "destination": canton,
                        "origin": canton,
                        "mode": mode,
                        "purpose": purpose,
                        "time_bins": {}
                    }
                    # Add for both roles
                    entry_origin = entry.copy()
                    entry_origin["role"] = "origin"
                    records.append(entry_origin)
                    entry_dest = entry.copy()
                    entry_dest["role"] = "destination"
                    records.append(entry_dest)
                else:
                    # If exists, ensure both roles are present
                    found_origin = any(r for r in records if r["mode"] == mode and r["purpose"] == purpose and r["role"] == "origin" and r["origin"] == canton and r["destination"] == canton)
                    found_dest = any(r for r in records if r["mode"] == mode and r["purpose"] == purpose and r["role"] == "destination" and r["origin"] == canton and r["destination"] == canton)
                    if not found_origin:
                        rec = nested_data[key].copy()
                        rec["role"] = "origin"
                        records.append(rec)
                    if not found_dest:
                        rec = nested_data[key].copy()
                        rec["role"] = "destination"
                        records.append(rec)
        with open(os.path.join(output_dir, f"{canton}.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    directory = '/cluster/project/cmdp/chaoch/switzerland_data/cache'
    get_canton_trip_data_combined("output_trips.csv")