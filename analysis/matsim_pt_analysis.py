import os 
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from analysis.webmap_export import assign_cantons  # Import the function
import glob

def preprocess_trips(input_path, canton_boundaries):
    """
    Preprocesses the trips file to add canton information for start and end coordinates.
    Returns the processed DataFrame in memory without saving to file.
    """
    # Read the CSV file
    df = pd.read_csv(input_path, sep=';')

    df_with_start = assign_cantons(df, canton_boundaries, x_col='start_x', y_col='start_y')
    df_with_start = df_with_start.rename(columns={'canton_name': 'start_canton'})

    df_with_end = assign_cantons(df, canton_boundaries, x_col='end_x', y_col='end_y')
    df_with_end = df_with_end.rename(columns={'canton_name': 'end_canton'})
  
    # combine start and end canton
    df_final = pd.merge(df_with_start, df_with_end[['trip_id', 'end_canton']], on='trip_id', how='left')
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


def add_canton_to_pt_passenger(passenger_counts_path, stops_by_canton_dir='stops_by_canton'):   

    df = pd.read_csv(passenger_counts_path, compression='gzip')
    canton_stop_mapping = {}

    # Get all geojson files in the directory
    geojson_files = glob.glob(os.path.join(stops_by_canton_dir, '*.geojson'))
    print(f"Found {len(geojson_files)} canton files")

    for file_path in geojson_files:
        #  extract canton name
        canton_name = os.path.basename(file_path).replace('_stops.geojson', '')
        
        canton_stops = gpd.read_file(file_path)
        
        # Extract all stop_ids for this canton
        for stop_id_list in canton_stops['stop_id']:
            if isinstance(stop_id_list, list):
                for stop_id in stop_id_list:
                    canton_stop_mapping[stop_id] = canton_name
            else:
                canton_stop_mapping[stop_id_list] = canton_name

    print(f"Total unique stop_ids across all cantons: {len(canton_stop_mapping)}")

    # Add origin_canton column to the dataframe
    if 'stop_id' in df.columns:
        # Map stop_id to canton
        df['origin_canton'] = df['stop_id'].map(canton_stop_mapping)
        
        unmapped_count = df['origin_canton'].isna().sum()
        total_count = len(df)
        
        if unmapped_count > 0:
            print(f"Warning: {unmapped_count} out of {total_count} rows could not be mapped to cantons ({unmapped_count/total_count*100:.1f}%)")
        
    else:
        print("Error: 'stop_id' column not found in the passenger counts data.")
        print("Available columns:", list(df.columns))
        df['origin_canton'] = None

    return df

#  ============== Code for generating how PT lines travel through cantons ==============

def create_boarding_json(df, output_dir, csv_file=None):
    """
    Create a JSON structure with boarding data aggregated by line_id+line_name, time_bin, and canton.
    
    Args:
        df: DataFrame with canton and boarding information (preferred)
        csv_file: Path to CSV file (fallback if df not provided)
    
    Returns:
        dict: JSON structure with boarding data
    """
    print("Creating boarding JSON from passenger counts data...")
    
    if df is None:
        df = pd.read_csv(csv_file)
    
    # Remove rows with missing canton information
    initial_count = len(df)
    df = df[df['origin_canton'].notna()]
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} rows with missing canton information")
    
    # Create a unique key combining line_id and line_name
    df['unique_line_key'] = df['line_id'] + '_' + df['line_name']
    
    # Group by unique_line_key to get metadata
    line_metadata = df.groupby('unique_line_key').agg({
        'vehicle_id': lambda x: list(x.unique()),
        'line_id': lambda x: list(x.unique()),
        'line_name': lambda x: list(x.unique()),
        'route_id': lambda x: list(x.unique()),
        'origin_canton': lambda x: list(x.unique())
    }).reset_index()
    
    # Aggregate boardings by unique_line_key, time_bin, and canton
    boarding_aggregation = df.groupby(['unique_line_key', 'time_bin', 'origin_canton'])['boardings'].sum().reset_index()
    
    # Create the JSON structure
    json_data = {}
    
    for _, meta_row in line_metadata.iterrows():
        unique_key = meta_row['unique_line_key']
        
        # Handle single values vs lists for metadata
        vehicle_id_raw = meta_row['vehicle_id'][0] if len(meta_row['vehicle_id']) == 1 else meta_row['vehicle_id']
        line_id = meta_row['line_id'][0] if len(meta_row['line_id']) == 1 else meta_row['line_id']
        line_name = meta_row['line_name'][0] if len(meta_row['line_name']) == 1 else meta_row['line_name']
        route_id = meta_row['route_id'][0] if len(meta_row['route_id']) == 1 else meta_row['route_id']
        cantons = sorted(meta_row['origin_canton'])
        
        # Extract vehicle type from vehicle_id (text after last underscore)
        if isinstance(vehicle_id_raw, list):
            # If multiple vehicle_ids, extract type from each
            vehicle_types = []
            for vid in vehicle_id_raw:
                vehicle_type = vid.split("_")[-1] if "_" in vid else vid
                vehicle_types.append(vehicle_type)
            # Remove duplicates and keep as list if multiple unique types
            unique_vehicle_types = list(set(vehicle_types))
            vehicle = unique_vehicle_types[0] if len(unique_vehicle_types) == 1 else unique_vehicle_types
        else:
            # Single vehicle_id
            vehicle = vehicle_id_raw.split("_")[-1] if "_" in vehicle_id_raw else vehicle_id_raw
        
        # Initialize the structure for this line using the unique key
        json_data[unique_key] = {
            "vehicle": vehicle,
            "cantons": cantons,
            "line_id": line_id,
            "line_name": line_name,
            "route_id": route_id,
            "boardings": {}
        }
        
        # Get boarding data for this line
        line_boardings = boarding_aggregation[boarding_aggregation['unique_line_key'] == unique_key]
        
        # Group by time_bin
        for time_bin in sorted(line_boardings['time_bin'].unique()):
            time_data = line_boardings[line_boardings['time_bin'] == time_bin]
            
            # Create canton boarding data for this time bin
            canton_boardings = {}
            for _, boarding_row in time_data.iterrows():
                canton = boarding_row['origin_canton']
                boardings_count = int(boarding_row['boardings'])  # Convert to int for JSON
                if boardings_count > 0:  # Only include non-zero boardings
                    canton_boardings[canton] = boardings_count
            
            # Only add time_bin if there are non-zero boardings
            if canton_boardings:
                json_data[unique_key]["boardings"][time_bin] = canton_boardings
    
    # Remove lines with no boardings
    json_data = {k: v for k, v in json_data.items() if v["boardings"]}
    
    # Save to file
    output_file = 'boarding_data_by_line.json'
    with open(os.path.join(output_dir, output_file), "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # Calculate statistics
    total_boardings = sum(
        sum(canton_data.values()) 
        for line_data in json_data.values() 
        for canton_data in line_data["boardings"].values()
    )
    
    print(f"Created JSON with {len(json_data)} lines and {total_boardings:,} total boardings")
    print(f"Saved to: {output_file}")
    
    return json_data

