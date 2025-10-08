#!/usr/bin/env python3
"""
Complete transfer data processing pipeline:
1. Load transfer analysis data from pt_transfer_analysis.csv
2. Generate stop-level transfer JSON with full stop IDs as keys
3. Group transfer data by Swiss cantons
4. Output both stop_transfer_data.json and stop_transfer_data_by_canton.json

This script combines the functionality of generate_stop_transfer_json.py and group_transfers_by_canton.py
"""

import pandas as pd
import json
import os
from pathlib import Path
from collections import defaultdict


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_base_stop_id(stop_string):
    """
    Extract the base stop ID (number before first colon/dot) from a full stop ID.
    This consolidates different platform/direction IDs to the same base stop.
    
    Examples:
    - '8508318:0:1.link:pt_8508318:0:1' -> '8508318'
    - '8591093.link:pt_8591093' -> '8591093'
    - '8591093.link:339638' -> '8591093'
    
    This approach fixes the matching issue where the same physical stop has
    different ID formats in MATSim output vs GeoJSON files.
    """
    if pd.isna(stop_string) or not stop_string:
        return None
    
    stop_str = str(stop_string).strip()
    # Extract number before first colon or dot - this is the base stop ID
    base_part = stop_str.split(':')[0].split('.')[0]
    return base_part


# =============================================================================
# STEP 1: LOAD AND PROCESS TRANSFER DATA
# =============================================================================

def load_transfer_data():
    """Load the transfer analysis data"""
    try:
        df = pd.read_csv('pt_transfer_analysis.csv')
        print(f"✅ Loaded {len(df)} transfer records from pt_transfer_analysis.csv")
        return df
    except FileNotFoundError:
        print("❌ Error: pt_transfer_analysis.csv not found. Please run analyze_transfers.py first.")
        return None


def build_stop_id_mapping(df, legs_df=None):
    """
    Build mapping from base stop IDs to representative full stop IDs.
    
    Key insight: The same physical stop can have multiple stop IDs with different
    formats (e.g. '8591093.link:pt_8591093' vs '8591093.link:339638'). We consolidate
    these by base ID ('8591093') and pick one representative full ID.
    
    Preference order:
    1. pt_ format (e.g. '8591093.link:pt_8591093') - most standard
    2. Most frequently used format - indicates primary usage
    """
    base_to_full = defaultdict(lambda: defaultdict(int))
    
    # Collect all base->full mappings from transfer data
    for _, row in df.iterrows():
        # Process egress stops (where people get off)
        if 'current_egress_stop' in df.columns:
            full_stop = row['current_egress_stop']
            if pd.notna(full_stop):
                base_stop = extract_base_stop_id(full_stop)
                if base_stop:
                    base_to_full[base_stop][full_stop] += 1
        
        # Process access stops (where people board)
        if 'next_access_stop' in df.columns:
            full_stop = row['next_access_stop']
            if pd.notna(full_stop):
                base_stop = extract_base_stop_id(full_stop)
                if base_stop:
                    base_to_full[base_stop][full_stop] += 1
    
    # Also collect from legs data if available (for comprehensive mapping)
    if legs_df is not None:
        pt_legs = legs_df[legs_df['mode'] == 'pt']
        for col in ['access_stop_id', 'egress_stop_id']:
            if col in pt_legs.columns:
                for full_stop in pt_legs[col].dropna():
                    base_stop = extract_base_stop_id(full_stop)
                    if base_stop:
                        base_to_full[base_stop][full_stop] += 1
    
    # For each base stop, pick the best representative full stop ID
    final_mapping = {}
    for base_stop, full_stops in base_to_full.items():
        if full_stops:
            # Prefer pt_ format (standard format)
            pt_format_stops = [stop for stop in full_stops.keys() if f'pt_{base_stop}' in stop]
            if pt_format_stops:
                representative_stop = max(pt_format_stops, key=lambda x: full_stops[x])
            else:
                # No pt_ format, use most frequent
                representative_stop = max(full_stops.items(), key=lambda x: x[1])[0]
            final_mapping[base_stop] = representative_stop
    
    print(f"✅ Built mapping for {len(final_mapping)} base stops to representative full stop IDs")
    return final_mapping


# =============================================================================
# STEP 2: AGGREGATE TRANSFER DATA BY STOP
# =============================================================================

def aggregate_stop_transfers(df):
    """
    Aggregate transfer data by base stop ID to consolidate different platform/direction IDs.
    
    Key change: Instead of using exact stop ID matching, we extract base stop IDs
    (e.g. '8591093' from '8591093.link:pt_8591093') and aggregate all transfers
    for the same physical stop location. This fixes the issue where major stops
    like Paradeplatz appeared to have zero transfers due to ID format mismatches.
    """
    stop_data = defaultdict(lambda: {
        'total_boardings': 0,
        'total_transfers_in': 0, 
        'total_transfers_out': 0,
        'line_transfers': defaultdict(lambda: defaultdict(int)),
        'stop_transfers': defaultdict(int)
    })
    
    print("🔄 Processing transfer records with base stop ID aggregation...")
    
    for _, row in df.iterrows():
        # Extract base stop IDs (this is the key fix)
        egress_stop_base = extract_base_stop_id(row['current_egress_stop'])
        access_stop_base = extract_base_stop_id(row['next_access_stop'])
        
        from_line = row['current_line']
        to_line = row['next_line']
        transfer_type = row['transfer_type']
        
        # Skip if we couldn't extract valid base stop IDs
        if pd.isna(egress_stop_base) or pd.isna(access_stop_base):
            continue
        
        egress_stop_base = str(egress_stop_base)
        access_stop_base = str(access_stop_base)
        
        if transfer_type == 'same_stop':
            # Same stop transfer: line change at same base location
            stop_data[access_stop_base]['total_transfers_in'] += 1
            stop_data[egress_stop_base]['total_transfers_out'] += 1
            stop_data[access_stop_base]['line_transfers'][from_line][to_line] += 1
            
        else:  # different_stop
            # Different stop transfer: walk between stops
            stop_data[egress_stop_base]['total_transfers_out'] += 1
            stop_data[access_stop_base]['total_transfers_in'] += 1
            stop_data[egress_stop_base]['stop_transfers'][access_stop_base] += 1
            stop_data[access_stop_base]['line_transfers'][from_line][to_line] += 1
    
    # Convert defaultdicts to regular dicts for JSON serialization
    result = {}
    for stop_id, data in stop_data.items():
        result[stop_id] = {
            'total_boardings': data['total_boardings'],
            'total_transfers_in': data['total_transfers_in'],
            'total_transfers_out': data['total_transfers_out'],
            'line_transfers': {
                origin_line: dict(dest_lines) 
                for origin_line, dest_lines in data['line_transfers'].items()
            },
            'stop_transfers': dict(data['stop_transfers'])
        }
    
    print(f"✅ Aggregated data for {len(result)} base stops with transfers")
    return result


def add_all_pt_boardings(stop_data):
    """
    Add ALL PT boardings to the total_boardings count from output_legs.csv.
    This includes both transfer boardings and standalone PT trips.
    """
    legs_df = None
    try:
        print("🔄 Loading output_legs.csv to count ALL PT boardings...")
        legs_df = pd.read_csv('output_legs.csv', sep=';')
        
        # Filter for PT legs only
        pt_legs = legs_df[legs_df['mode'] == 'pt'].copy()
        print(f"✅ Found {len(pt_legs)} PT legs total")
        
        # Extract base stop IDs from access_stop_id
        pt_legs['access_stop_base'] = pt_legs['access_stop_id'].apply(extract_base_stop_id)
        
        # Count ALL boardings per base stop ID (including standalone trips)
        stop_boardings = pt_legs['access_stop_base'].value_counts()
        
        print(f"🔄 Adding ALL PT boardings for {len(stop_boardings)} base stops...")
        
        for base_stop_id, boarding_count in stop_boardings.items():
            if pd.notna(base_stop_id):
                base_stop_str = str(base_stop_id)
                if base_stop_str in stop_data:
                    stop_data[base_stop_str]['total_boardings'] = boarding_count
                else:
                    # Create new entry for stops with only standalone PT trips
                    stop_data[base_stop_str] = {
                        'total_boardings': boarding_count,
                        'total_transfers_in': 0,
                        'total_transfers_out': 0,
                        'line_transfers': {},
                        'stop_transfers': {}
                    }
                
    except FileNotFoundError:
        print("⚠️ Warning: output_legs.csv not found. Only showing transfer boardings.")
    except Exception as e:
        print(f"⚠️ Warning: Error processing boardings: {e}")
    
    return stop_data, legs_df


def convert_to_representative_stop_ids(stop_data, base_to_full_mapping):
    """
    Convert aggregated base stop data to use representative full stop IDs as final keys.
    
    This step converts our internal base stop aggregation (e.g. '8591093') back to 
    full stop IDs (e.g. '8591093.link:pt_8591093') for the final output, while 
    preserving all the consolidated transfer data.
    """
    full_stop_data = {}
    
    print("🔄 Converting base stop IDs to representative full stop IDs for output...")
    
    major_stops_examples = []
    
    for base_stop_id, data in stop_data.items():
        # Get representative full stop ID (preferring pt_ format)
        representative_stop_id = base_to_full_mapping.get(base_stop_id, base_stop_id)
        
        # Track examples of major stops for logging
        if data['total_boardings'] > 50 or data['total_transfers_in'] > 20:
            major_stops_examples.append(
                f"  📍 {base_stop_id} → {representative_stop_id} "
                f"(boardings: {data['total_boardings']}, transfers: {data['total_transfers_in']})"
            )
        
        # Convert stop_transfers to use representative stop IDs as destination keys
        converted_stop_transfers = {}
        for dest_base_stop, count in data['stop_transfers'].items():
            dest_representative_stop = base_to_full_mapping.get(dest_base_stop, dest_base_stop)
            converted_stop_transfers[dest_representative_stop] = count
        
        # Create final data entry with representative stop ID as key
        full_stop_data[representative_stop_id] = {
            'total_boardings': data['total_boardings'],
            'total_transfers_in': data['total_transfers_in'],
            'total_transfers_out': data['total_transfers_out'],
            'line_transfers': data['line_transfers'],
            'stop_transfers': converted_stop_transfers
        }
    
    print(f"✅ Converted {len(stop_data)} base stops to {len(full_stop_data)} representative stop IDs")
    
    # Show examples of major stop consolidations
    if major_stops_examples:
        print("🔍 Major stop consolidations (base → representative):")
        for example in major_stops_examples[:8]:  # Show top 8
            print(example)
        if len(major_stops_examples) > 8:
            print(f"  ... and {len(major_stops_examples) - 8} more major stops")
    
    return full_stop_data


# =============================================================================
# STEP 3: GROUP BY CANTON
# =============================================================================

def load_canton_stop_mapping():
    """
    Load all canton GeoJSON files and create a mapping from base stop_id to canton name.
    """
    canton_stops = {}
    base_to_canton = {}
    stops_dir = Path("stops_by_canton")
    
    print(f"🔄 Loading canton stop data from {stops_dir}...")
    
    if not stops_dir.exists():
        print(f"❌ Error: {stops_dir} directory not found!")
        return {}
    
    geojson_files = list(stops_dir.glob("*_stops.geojson"))
    print(f"✅ Found {len(geojson_files)} canton files")
    
    for geojson_file in geojson_files:
        canton_name = geojson_file.stem.replace("_stops", "")
        
        try:
            with open(geojson_file, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            stop_count = 0
            for feature in geojson_data.get('features', []):
                properties = feature.get('properties', {})
                stop_ids = properties.get('stop_id', [])
                
                for stop_id in stop_ids:
                    canton_stops[stop_id] = canton_name
                    base_id = extract_base_stop_id(stop_id)
                    if base_id:
                        base_to_canton[base_id] = canton_name
                    stop_count += 1
            
            print(f"  ✅ {canton_name}: {stop_count} stop IDs")
            
        except Exception as e:
            print(f"❌ Error reading {geojson_file}: {e}")
    
    print(f"✅ Total base stop IDs mapped: {len(base_to_canton)}")
    return base_to_canton


def group_transfers_by_canton(transfer_data, canton_base_mapping):
    """
    Group transfer data by canton using the base stop-to-canton mapping.
    """
    canton_data = defaultdict(dict)
    unmapped_stops = set()

    print("🔄 Grouping transfer data by canton...")

    mapped_count = 0
    total_count = 0

    for stop_id, stop_data in transfer_data.items():
        total_count += 1
        base_stop_id = extract_base_stop_id(stop_id)
        canton = canton_base_mapping.get(base_stop_id)

        if canton:
            canton_data[canton][stop_id] = stop_data
            mapped_count += 1
        else:
            unmapped_stops.add(stop_id)

    print(f"✅ Grouping results:")
    print(f"   📊 Total stops processed: {total_count}")
    print(f"   ✅ Successfully mapped: {mapped_count}")
    print(f"   ❌ Unmapped: {len(unmapped_stops)}")
    
    if unmapped_stops and len(unmapped_stops) < 50:  # Only show if manageable number
        print(f"   ⚠️ Sample unmapped stops: {list(unmapped_stops)[:10]}")

    return dict(canton_data), unmapped_stops


def calculate_canton_statistics(canton_data):
    """
    Calculate summary statistics for each canton.
    """
    print("🔄 Calculating canton-level statistics...")
    
    for canton_name, stops_data in canton_data.items():
        total_boardings = sum(data.get('total_boardings', 0) for data in stops_data.values())
        total_transfers_in = sum(data.get('total_transfers_in', 0) for data in stops_data.values())
        total_transfers_out = sum(data.get('total_transfers_out', 0) for data in stops_data.values())
        
        # Count unique lines across all stops
        all_lines = set()
        for stop_data in stops_data.values():
            line_transfers = stop_data.get('line_transfers', {})
            for origin_line in line_transfers.keys():
                all_lines.add(origin_line)
                for dest_line in line_transfers[origin_line].keys():
                    all_lines.add(dest_line)
        
        # Find busiest stop
        busiest_stop = None
        max_boardings = 0
        for stop_id, stop_data in stops_data.items():
            if stop_id != '_canton_summary':
                boardings = stop_data.get('total_boardings', 0)
                if boardings > max_boardings:
                    max_boardings = boardings
                    busiest_stop = stop_id
        
        # Add canton summary
        canton_data[canton_name]['_canton_summary'] = {
            'total_stops': len(stops_data),
            'total_boardings': total_boardings,
            'total_transfers_in': total_transfers_in,
            'total_transfers_out': total_transfers_out,
            'unique_lines': len(all_lines),
            'busiest_stop': busiest_stop if busiest_stop else 'None'
        }
        
        # Correct the total_stops count to exclude the summary itself
        canton_data[canton_name]['_canton_summary']['total_stops'] = len(stops_data) - 1
    
    return canton_data


# =============================================================================
# STEP 4: MAIN PROCESSING PIPELINE
# =============================================================================

def process_transfer_data():
    """
    Main function to process transfer data and generate both output files.
    """
    print("=" * 60)
    print("🚀 SWISS PT TRANSFER DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load transfer data
    df = load_transfer_data()
    if df is None:
        return
    
    print(f"📊 Transfer data columns: {list(df.columns)}")
    print(f"📊 Transfer types: {df['transfer_type'].value_counts().to_dict()}")
    
    # Step 2: Aggregate transfer data by stop (using base stop IDs)
    stop_data = aggregate_stop_transfers(df)
    
    # Step 3: Add all PT boardings from output_legs.csv
    stop_data, legs_df = add_all_pt_boardings(stop_data)
    print(f"✅ Final data includes {len(stop_data)} base stops total")
    
    # Step 4: Build mapping from base stop IDs to full stop IDs
    base_to_full_mapping = build_stop_id_mapping(df, legs_df)
    
    # Step 5: Convert stop data to use full stop IDs as keys
    full_stop_data = convert_to_representative_stop_ids(stop_data, base_to_full_mapping)
    
    # Step 6: Calculate and print statistics
    total_boardings = sum(data['total_boardings'] for data in full_stop_data.values())
    total_transfers_in = sum(data['total_transfers_in'] for data in full_stop_data.values())
    standalone_trips = total_boardings - total_transfers_in
    
    print("\n" + "=" * 40)
    print("📊 BOARDING STATISTICS")
    print("=" * 40)
    print(f"  🚌 Total PT boardings: {total_boardings:,}")
    print(f"  🔄 Transfer boardings: {total_transfers_in:,}")
    print(f"  🎯 Standalone trips: {standalone_trips:,}")
    if total_boardings > 0:
        print(f"  📈 Transfer rate: {total_transfers_in/total_boardings*100:.1f}%")
    
    # Step 7: Save stop-level JSON
    output_file = 'stop_transfer_data.json'
    with open(output_file, 'w') as f:
        json.dump(full_stop_data, f, indent=2)
    print(f"\n✅ Stop-level JSON saved to {output_file}")
    
    # Step 8: Load canton mapping and group by canton
    canton_base_mapping = load_canton_stop_mapping()
    if not canton_base_mapping:
        print("❌ Error: No canton-stop mapping loaded.")
        return
    
    canton_data, unmapped_stops = group_transfers_by_canton(full_stop_data, canton_base_mapping)
    
    # Step 9: Calculate canton statistics
    canton_data = calculate_canton_statistics(canton_data)
    
    # Step 10: Create final canton output structure
    output_data = {
        'metadata': {
            'description': 'Public transport transfer data grouped by Swiss cantons',
            'total_cantons': len(canton_data),
            'total_stops_mapped': sum(len(stops) - 1 for stops in canton_data.values()),  # -1 for summary
            'unmapped_stops': len(unmapped_stops),
            'data_structure': {
                'canton_name': {
                    'stop_id': {
                        'total_boardings': 'Total boardings at this stop',
                        'total_transfers_in': 'Total people transferring into this stop',
                        'total_transfers_out': 'Total people leaving this stop via transfer',
                        'line_transfers': 'origin_line -> destination_line transfer counts',
                        'stop_transfers': 'transfers from this stop to other stops'
                    },
                    '_canton_summary': 'Aggregated statistics for the canton'
                }
            }
        },
        'cantons': canton_data
    }
    
    # Step 11: Save canton-grouped JSON
    canton_output_file = "stop_transfer_data_by_canton.json"
    with open(canton_output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Canton-grouped JSON saved to {canton_output_file}")
    
    # Step 12: Print final summary
    print("\n" + "=" * 60)
    print("🎉 PROCESSING COMPLETE - CANTON SUMMARY")
    print("=" * 60)
    for canton_name in sorted(canton_data.keys()):
        summary = canton_data[canton_name]['_canton_summary']
        print(f"{canton_name:20s}: {summary['total_stops']:4d} stops, "
              f"{summary['total_boardings']:5d} boardings, "
              f"{summary['total_transfers_in']:4d} transfers")
    
    print(f"\n🎯 Final outputs:")
    print(f"   📄 {output_file}")
    print(f"   📄 {canton_output_file}")
    print(f"   📊 {len(canton_data)} cantons processed")
    print("=" * 60)


if __name__ == "__main__":
    process_transfer_data()
