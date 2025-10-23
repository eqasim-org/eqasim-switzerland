import os
import json
import pandas as pd
import glob
import geopandas as gpd
import analysis.webmap_export as webmap_export


def configure(context):
    """Configure the required stages for transit stop processing."""
    context.stage("matsim.simulation.run")  # get simulation output directory
    context.stage("data.spatial.cantons")   # get canton boundaries


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


def execute(context):
    """
    Main execution function that creates individual GeoJSON files for each canton's transit stops.
    This is a simplified version that focuses only on the transit stop processing.
    """
    
    # Get required inputs from pipeline context
    matsim_dir = context.stage("matsim.simulation.run")
    canton_boundaries = context.stage("data.spatial.cantons")
    
    # Define paths
    simulation_output = os.path.join(matsim_dir, "simulation_output")
    transit_schedule_path = os.path.join(simulation_output, "output_transitSchedule.xml.gz")
    passenger_counts_path = os.path.join(simulation_output, "pt_passenger_counts.csv.gz")
    
    # Create output directory structure
    output_dir = os.path.join(matsim_dir, "simulation_output", "webmap")
    os.makedirs(output_dir, exist_ok=True)
    webmap_export.DEFAULT_WORKDIR = output_dir
    
    # Create transit subdirectories
    os.makedirs(os.path.join(output_dir, "public", "data", "matsim", "transit"), exist_ok=True)
    
    # Verify required files exist
    if not os.path.exists(transit_schedule_path):
        raise FileNotFoundError(f"Transit schedule file not found: {transit_schedule_path}")
    
    print("=" * 60)
    print("CREATING CANTON-SPECIFIC TRANSIT STOP GEOJSON FILES")
    print("=" * 60)
    
    # Step 1: Parse transit stops from MATSim schedule XML
    print("Step 1: Parsing transit stops from schedule XML...")
    stops_gdf = webmap_export.parse_stops(transit_schedule_path)
    stops_gdf = stops_gdf.to_crs(epsg=2056)
    print(f"  → Parsed {len(stops_gdf)} unique stops")
    
    # Step 2: Assign each stop to its canton
    print("Step 2: Assigning each stop to a canton...")
    joined = webmap_export.assign_cantons_stops(stops_gdf, canton_boundaries)
    
    assigned_count = (~joined['assigned_canton'].isna()).sum()
    print(f"  → Successfully assigned {assigned_count} stops to cantons")
    print(f"  → {len(joined) - assigned_count} stops could not be assigned to any canton")
    
    # Step 3: Export individual GeoJSON files for each canton
    print("Step 3: Exporting per-canton stop GeoJSONs...")
    webmap_export.export_per_canton_stops(joined)
    
    # directory contain stops by canton
    stops_dir = os.path.join(output_dir, "public", "data", "matsim", "transit", "stops_by_canton")
    
    # Generate the canton travel information for PT lines
    df_with_cantons = add_canton_to_pt_passenger(passenger_counts_path, stops_dir)
    create_boarding_json(df_with_cantons, output_dir, )


if __name__ == '__main__':
    df = pd.read_csv('/cluster/project/cmdp/chaoch/pt_passenger_counts.csv.gz', compression='gzip')
    print(f"Loaded passenger data with {len(df)} rows and columns: {list(df.columns)}")
 
    # Step 1: Add canton information to passenger data
    print("\n=== ADDING CANTON INFORMATION ===")
    df_with_cantons = add_canton_to_pt_passenger(df)

    # Step 2: Create JSON structure for boarding data
    print("\n=== CREATING BOARDING JSON ===")
    create_boarding_json(df_with_cantons, output_dir='')
