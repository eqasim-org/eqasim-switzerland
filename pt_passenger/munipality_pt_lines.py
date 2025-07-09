import pandas as pd
import geopandas as gpd
import os
import glob
import json

def analyze_multi_canton_lines(csv_file='pt_passenger_counts_with_cantons.csv'):

    print("Reading processed passenger counts data...")
    df = pd.read_csv(csv_file)
    
    # Group by line_id and find unique cantons for each line
    line_canton_analysis = df.groupby('line_id').agg({
        'origin_canton': lambda x: x.dropna().unique().tolist(),
        'line_name': 'first',  # Get the line name
        'stop_id': 'count'  # Count total stops per line
    }).reset_index()
    
    # Add column for number of cantons per line
    line_canton_analysis['num_cantons'] = line_canton_analysis['origin_canton'].apply(len)
    line_canton_analysis['canton_list'] = line_canton_analysis['origin_canton'].apply(lambda x: ', '.join(sorted(x)) if x else 'No Canton')
    
    # Filter for multi-canton lines
    multi_canton_lines = line_canton_analysis[line_canton_analysis['num_cantons'] > 1].copy()
    
    # Sort by number of cantons (descending) and then by number of stops
    multi_canton_lines = multi_canton_lines.sort_values(['num_cantons', 'stop_id'], ascending=[False, False])
    
    print(f"\n=== MULTI-CANTON LINE ANALYSIS ===")
    print(f"Total unique lines: {len(line_canton_analysis)}")
    print(f"Lines crossing multiple cantons: {len(multi_canton_lines)}")
    print(f"Percentage of multi-canton lines: {len(multi_canton_lines)/len(line_canton_analysis)*100:.2f}%")
    
    # Distribution of lines by number of cantons
    canton_distribution = line_canton_analysis['num_cantons'].value_counts().sort_index()
    print(f"\nDistribution by number of cantons:")
    for num_cantons, count in canton_distribution.items():
        print(f"  {num_cantons} canton{'s' if num_cantons != 1 else ''}: {count} lines")
    
    # Top 20 lines with most cantons
    print(f"\nTop 20 lines crossing most cantons:")
    print("=" * 80)
    for idx, row in multi_canton_lines.head(20).iterrows():
        print(f"{row['line_id']} ({row['line_name']}): {row['num_cantons']} cantons")
        print(f"  Cantons: {row['canton_list']}")
        print(f"  Total stops: {row['stop_id']}")
        print("-" * 40)
    
    # Analysis by canton pairs/combinations
    print(f"\nMost common canton combinations:")
    canton_combo_counts = multi_canton_lines['canton_list'].value_counts().head(10)
    for combo, count in canton_combo_counts.items():
        print(f"  {combo}: {count} lines")
    
    # Save detailed results
    output_file = 'multi_canton_lines_analysis.csv'
    multi_canton_lines.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return multi_canton_lines


def add_canton_to_pt_passenger(df, stops_by_canton_dir='stops_by_canton'):   

    canton_stop_mapping = {}

    # Get all geojson files in the directory
    geojson_files = glob.glob(os.path.join(stops_by_canton_dir, '*.geojson'))
    print(f"Found {len(geojson_files)} canton files")

    for file_path in geojson_files:
        #  extract canton name
        canton_name = os.path.basename(file_path).replace('_stops.geojson', '')
        print(f"Processing {canton_name}...")
        
        canton_stops = gpd.read_file(file_path)
        
        # Extract all stop_ids for this canton
        for stop_id_list in canton_stops['stop_id']:
            if isinstance(stop_id_list, list):
                for stop_id in stop_id_list:
                    canton_stop_mapping[stop_id] = canton_name
            else:
                canton_stop_mapping[stop_id_list] = canton_name

    print(f"\nTotal unique stop_ids across all cantons: {len(canton_stop_mapping)}")
    print("Sample canton mappings:", dict(list(canton_stop_mapping.items())[:5]))

    # Add origin_canton column to the dataframe
    # Check if the dataframe has a stop_id column to join with
    if 'stop_id' in df.columns:
        # Map stop_id to canton
        df['origin_canton'] = df['stop_id'].map(canton_stop_mapping)
        
        canton_counts = df['origin_canton'].value_counts()
        total_count = len(df)
        unmapped_count = df['origin_canton'].isna().sum()
        
        print(f"\nBreakdown by canton:")
        for canton, count in canton_counts.items():
            print(f"{canton}: {count} rows ({count/total_count*100:.2f}%)")
        
        if unmapped_count > 0:
            print(f"\nUnmapped rows (no canton found): {unmapped_count} out of {total_count} total rows ({unmapped_count/total_count*100:.2f}%)")
        
        # some examples
        print("\nSample of data with origin_canton:")
        sample_cols = ['stop_id', 'origin_canton']
        if 'line_name' in df.columns:
            sample_cols.append('line_name')
        print(df[sample_cols].head(10))
        
    else:
        print("'stop_id' column not found in the passenger counts data.")
        print("Available columns in passenger data:", list(df.columns))
        # Set all to None as fallback
        df['origin_canton'] = None

    output_filename = 'pt_passenger_counts_with_cantons.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nData with origin_canton column saved to: {output_filename}")
    print(f"Final dataframe shape: {df.shape}")


def create_boarding_json(csv_file='pt_passenger_counts_with_cantons.csv'):
    """
    Create a JSON structure with boarding data aggregated by line_id+line_name, time_bin, and canton.
    
    Args:
        csv_file: Path to the CSV file with canton and boarding information
    
    Returns:
        dict: JSON structure with boarding data
    """
    print("Reading passenger counts data for JSON creation...")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Columns: {list(df.columns)}")
    
    # Remove rows with missing canton information
    df = df[df['origin_canton'].notna()]
    print(f"After removing rows with no canton: {len(df)} rows")
    
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
    
    print(f"\nCreated JSON data for {len(json_data)} lines")
    
    # Save to file
    output_file = 'boarding_data_by_line.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"JSON data saved to: {output_file}")
    
    # Print some statistics
    total_lines = len(json_data)
    lines_with_boardings = sum(1 for line_data in json_data.values() if line_data["boardings"])
    total_boardings = sum(
        sum(canton_data.values()) 
        for line_data in json_data.values() 
        for canton_data in line_data["boardings"].values()
    )
    
    print(f"\nStatistics:")
    print(f"Total lines: {total_lines}")
    print(f"Lines with boardings: {lines_with_boardings}")
    print(f"Total boardings across all lines: {total_boardings}")
    
    # Show sample of the data
    print(f"\nSample of JSON structure (first 2 lines):")
    sample_lines = list(json_data.keys())[:2]
    for unique_key in sample_lines:
        print(f"\n{unique_key}:")
        sample_data = json_data[unique_key].copy()
        # Limit boarding data for display
        if sample_data["boardings"]:
            first_time_bin = list(sample_data["boardings"].keys())[0]
            sample_data["boardings"] = {first_time_bin: sample_data["boardings"][first_time_bin]}
        print(json.dumps(sample_data, indent=2))
    
    return json_data

if __name__ == '__main__':
    df = pd.read_csv('/cluster/project/cmdp/chaoch/pt_passenger_counts.csv.gz', compression='gzip')

    # Explore the dataframe
    print(list(df.columns))
    print(df.head(10))
    print("\nFirst 20 lines and their names:")
    print(df[['line_id', 'line_name']].head(20))


    # Run multi-canton analysis
    print("\n" + "="*60)
    print("STARTING MULTI-CANTON LINE ANALYSIS")
    print("="*60)
    output_filename = 'pt_passenger_counts_with_cantons.csv'
    multi_canton_results = analyze_multi_canton_lines(output_filename)

    # # Add canton information to passenger data
    # print("\n" + "="*60)
    # print("ADDING CANTON INFORMATION TO PASSENGER DATA")
    # print("="*60)
    # add_canton_to_pt_passenger(df)

    # Create JSON structure for boarding data
    # print("\n" + "="*60)
    # print("CREATING JSON STRUCTURE FOR BOARDING DATA")
    # print("="*60)
    json_data = create_boarding_json()
    # Write the JSON data to a file
    # output_file = 'boarding_data_by_line.json'
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(json_data, f, ensure_ascii=False, indent=2)

    # print(f"\nJSON data has been written to {output_file}")
