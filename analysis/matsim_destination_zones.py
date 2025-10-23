import os 
import numpy as np
import pandas as pd
import geopandas as gpd
import unicodedata
import json
from webmap_export import assign_cantons  # Import the function

def configure(context):
    context.stage("matsim.simulation.run")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.spatial.cantons") # get canton boundaries

def remove_accents(text):
    if isinstance(text, str):
        return ''.join(
            c for c in unicodedata.normalize('NFKD', text)
            if not unicodedata.combining(c)
        )
    return text

def preprocess_trips(input_path, canton_boundaries):
    """
    Preprocesses the trips file to add canton information for start and end coordinates.
    Returns the processed DataFrame in memory without saving to file.
    """
    # Read the CSV file
    df = pd.read_csv(input_path, sep=';')
    
    print("Columns in the dataset:")
    print(df.columns.tolist())
    
    # Add start canton information
    df_with_start = assign_cantons(df, canton_boundaries, x_col='start_x', y_col='start_y')
    df_with_start = df_with_start.rename(columns={'canton_name': 'start_canton'})
    
    # Add end canton information to the same DataFrame
    df_final = assign_cantons(df_with_start, canton_boundaries, x_col='end_x', y_col='end_y')
    df_final = df_final.rename(columns={'canton_name': 'end_canton'})
    
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


def generate_source_destination_data(input_path, work_dir, canton_boundaries):
    """
    Complete pipeline: preprocesses trips to add canton info, then generates canton trip data.
    """
    print("Step 1: Adding canton information to trips...")
    df_with_cantons = preprocess_trips(input_path, canton_boundaries)
    
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
    canton_boundaries = context.stage("data.spatial.cantons")
    generate_source_destination_data(trips_path, work_dir=output_dir, canton_boundaries=canton_boundaries)

    print("Webmap export complete. Output saved to:", output_dir)


if __name__ == '__main__':
    generate_source_destination_data("output_trips.csv")