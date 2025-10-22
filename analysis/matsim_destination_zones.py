import os 
import numpy as np
import pandas as pd
import geopandas as gpd
import unicodedata
import json

def configure(context):
    context.stage("matsim.simulation.run")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")

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
        canton_name: name for the canton column
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
    # Use a unique distance column name to avoid conflicts
    distance_col_name = f"distance_{canton_name}"
    match_closest = non_match.sjoin_nearest(canton_boundaries[['KANTONSNUMMER', 'NAME', 'geometry']], how="left", max_distance=distance, distance_col=distance_col_name)

    print("Non-matches finished!")
    print("Concatenating results...")

    result = pd.concat([match, match_closest], ignore_index=True)

    result_filt = result.drop(columns=["geometry", "index_right"], errors="ignore")
    # Keep the distance column if it exists
    if "distance" not in result_filt.columns and "distance" in result.columns:
        result_filt["distance"] = result["distance"]
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

def preprocess_trips(input_path="output_trips.csv"):
    """
    Preprocesses the trips file to add canton information for start and end coordinates.
    Returns the processed DataFrame in memory without saving to file.
    """
    # Read the CSV file
    df = pd.read_csv(input_path, sep=';')
    
    print("Columns in the dataset:")
    print(df.columns.tolist())
    
    # Add start canton information
    df_with_start = add_canton_name(df, 'start_x', 'start_y', canton_name='start_canton')
    
    # Add end canton information to the same DataFrame
    df_final = add_canton_name(df_with_start, 'end_x', 'end_y', canton_name='end_canton')
    
    print(f"\nProcessed dataset columns: {list(df_final.columns)}")

    return df_final

def to_time_bin(time_str):
    # Parse hours, minutes, seconds from string
    hours, minutes, _ = map(int, time_str.split(":"))
    
    total_minutes = hours * 60 + minutes
    
    bin_minutes = (total_minutes // 15) * 15
    
    binned_hours = bin_minutes // 60
    binned_minutes = bin_minutes % 60
    
    return f"{binned_hours:02d}:{binned_minutes:02d}"

def get_canton_trip_data(df, work_dir):
    """
    Outputs one JSON file per canton, containing records for both when the canton is treated as origin and as destination.
    Takes a DataFrame as input instead of reading from file.
    """
    df = df.copy()  # Work with a copy to avoid modifying the original
    df["time_bin"] = df["dep_time"].apply(to_time_bin)

    df = df[[
        "start_canton", "end_canton", "main_mode", "end_activity_type", "time_bin"
    ]].rename(columns={
        "start_canton": "origin",
        "end_canton": "destination",
        "main_mode": "mode",
        "end_activity_type": "purpose"
    })
    
    # Remove rows with missing canton information
    df = df.dropna(subset=['origin', 'destination'])
    print(f"Processing {len(df)} trips after removing entries with missing canton data")

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

    output_dir = os.path.join(work_dir, "public", "data", "trips_by_canton")
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


def generate_source_destination_data(input_path, work_dir):
    """
    Complete pipeline: preprocesses trips to add canton info, then generates canton trip data.
    """
    print("Step 1: Adding canton information to trips...")
    df_with_cantons = preprocess_trips(input_path)
    
    print("\nStep 2: Generating canton trip data...")
    get_canton_trip_data(df_with_cantons, work_dir)
    
    print("\nProcessing complete! Check the 'trips_by_canton' directory for output files.")


def execute(context):
    matsim_dir = context.stage("matsim.simulation.run")
    simulation_output = os.path.join(matsim_dir, "simulation_output")

    # Create webmap output directory
    output_dir = os.path.join(matsim_dir, "simulation_output", "webmap")
    os.makedirs(output_dir, exist_ok=True)

    # Create all necessary subfolders
    os.makedirs(os.path.join(output_dir, "public", "data"), exist_ok=True)

    # Input files
    trips_path = os.path.join(simulation_output, "output_trips.csv.gz")
    passenger_counts_path = os.path.join(simulation_output, "pt_passenger_counts.csv.gz")
    legs_path = os.path.join(simulation_output, "output_legs.csv.gz")

    # functions for creating the jsons
    print("Generating modes_by_canton.json...")
    generate_source_destination_data(trips_path, work_dir=output_dir)

    print("Webmap export complete. Output saved to:", output_dir)


if __name__ == '__main__':
    generate_source_destination_data("output_trips.csv")