#!/usr/bin/env python3
"""
Analyze PT transfer patterns in MATSim output_legs.csv dataset.

This script identifies transfer patterns: PT leg -> walking -> PT leg
- Same stop transfers: base stop IDs match (platform changes)  
- Different stop transfers: base stop IDs differ (walking between stops)

**: pt_transfer_analysis.csv with detailed transfer information
NOTE: this the information that's used to generate the transfer matrix
The script extracts transfer and boarding data to `pt_transfer_analysis.csv`,
so that irrelevant data is not included. 
"""

import pandas as pd
import traceback
from collections import Counter
from datetime import datetime, timedelta


def extract_base_stop_id(stop_id):
    """
    Extract the base stop ID (number before first colon) from a full stop ID.
    Example: '8508318:0:1.link:pt_8508318:0:1' -> '8508318'
    """
    if pd.isna(stop_id) or stop_id == '':
        return None
    
    stop_str = str(stop_id).strip()
    base_id = stop_str.split(':')[0]
    return base_id


def parse_time_24plus(time_str):
    """
    Parse time string that may have hours >= 24 (transit times past midnight).
    Convert to datetime object by handling 24+ hour format.
    """
    if pd.isna(time_str) or time_str == '':
        return None
    
    time_parts = str(time_str).strip().split(':')
    if len(time_parts) != 3:
        return None
    
    try:
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(time_parts[2])
        
        # Handle 24+ hour format by converting to next day
        if hours >= 24:
            base_date = datetime(2000, 1, 1)
            extra_days = hours // 24
            remaining_hours = hours % 24
            result = base_date + timedelta(days=extra_days, hours=remaining_hours, minutes=minutes, seconds=seconds)
        else:
            result = datetime(2000, 1, 1, hours, minutes, seconds)
        
        return result
    except (ValueError, IndexError):
        return None


def parse_travel_time(trav_time_str):
    """
    Parse travel time string (HH:MM:SS) and return as timedelta.
    """
    if pd.isna(trav_time_str) or trav_time_str == '':
        return timedelta(0)
    
    time_parts = str(trav_time_str).strip().split(':')
    if len(time_parts) != 3:
        return timedelta(0)
    
    try:
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(time_parts[2])
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except (ValueError, IndexError):
        return timedelta(0)


def categorize_transfer_distance(distance):
    """Categorize transfer distance into meaningful groups."""
    if distance == 0:
        return "0m (same platform)"
    elif distance <= 50:
        return "1-50m (very short)"
    elif distance <= 100:
        return "51-100m (short)"
    elif distance <= 200:
        return "101-200m (medium)"
    elif distance <= 500:
        return "201-500m (long)"
    else:
        return "500m+ (very long)"


def categorize_transfer_time(time_str):
    """Categorize transfer time into meaningful groups."""
    if pd.isna(time_str) or time_str == '':
        return "unknown"
    
    # Convert time string to minutes
    time_parts = str(time_str).split(':')
    if len(time_parts) == 3:
        try:
            hours, minutes, seconds = map(int, time_parts)
            total_minutes = hours * 60 + minutes + seconds / 60
        except ValueError:
            return "unknown"
    else:
        return "unknown"
    
    if total_minutes <= 2:
        return "0-2min (very quick)"
    elif total_minutes <= 5:
        return "2-5min (quick)"
    elif total_minutes <= 10:
        return "5-10min (medium)"
    elif total_minutes <= 15:
        return "10-15min (long)"
    else:
        return "15min+ (very long)"


def extract_line_type(line_id):
    """Extract the line type from transit line ID."""
    if pd.isna(line_id) or line_id == '':
        return "unknown"
    
    line_str = str(line_id)
    # Extract the part after the first dash and before the second dash
    parts = line_str.split('-')
    if len(parts) >= 2:
        return parts[1]  # e.g., "91-55-j24-1" -> "55"
    return "unknown"


def analyze_pt_transfers_with_walking(csv_file):
    """
    Analyze PT transfers with walking segments between them.
    
    Identifies and counts transfers following the pattern: pt -> walk -> pt
    where walking segments separate consecutive PT legs.
    
    Returns:
        dict: Transfer analysis results and details
    """
    print("===== PT TRANSFER ANALYSIS (pt -> walk -> pt) =====")
    
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file, sep=';')
    print(f"Dataset loaded: {len(df)} rows")
    
    # Sort by person, trip_id, and departure time
    df_sorted = df.sort_values(['person', 'trip_id', 'dep_time']).reset_index(drop=True)
    
    # Initialize counters
    same_stop_transfers = 0
    different_stop_transfers = 0
    total_pt_legs = 0
    transfer_details = []
    
    # Analyze each trip
    for (person, trip_id), trip_group in df_sorted.groupby(['person', 'trip_id']):
        trip_legs = trip_group.reset_index(drop=True)
        pt_legs_indices = trip_legs[trip_legs['mode'] == 'pt'].index.tolist()
        total_pt_legs += len(pt_legs_indices)
        
        # Check consecutive PT legs for transfers
        for i in range(len(pt_legs_indices) - 1):
            current_pt_idx = pt_legs_indices[i]
            next_pt_idx = pt_legs_indices[i + 1]
            current_pt_leg = trip_legs.iloc[current_pt_idx]
            next_pt_leg = trip_legs.iloc[next_pt_idx]
            
            # Validate stop information exists
            if (pd.notna(current_pt_leg['egress_stop_id']) and 
                current_pt_leg['egress_stop_id'] != '' and
                pd.notna(next_pt_leg['access_stop_id']) and 
                next_pt_leg['access_stop_id'] != ''):
                
                # Extract base stop IDs for comparison
                current_egress_base = extract_base_stop_id(current_pt_leg['egress_stop_id'])
                next_access_base = extract_base_stop_id(next_pt_leg['access_stop_id'])
                
                # Determine transfer type based on base stop IDs
                if current_egress_base == next_access_base:
                    transfer_type = 'same_stop'
                    same_stop_transfers += 1
                else:
                    transfer_type = 'different_stop'
                    different_stop_transfers += 1
                
                # Calculate walking segments between PT legs
                walking_legs = trip_legs.iloc[current_pt_idx + 1:next_pt_idx]
                walking_distance = walking_legs[walking_legs['mode'] == 'walk']['distance'].sum()
                walking_time = sum(parse_travel_time(t) for t in walking_legs[walking_legs['mode'] == 'walk']['trav_time'])
                
                # Parse times
                current_departure = parse_time_24plus(current_pt_leg['dep_time'])
                current_arrival = parse_time_24plus(current_pt_leg['dep_time']) + parse_travel_time(current_pt_leg['trav_time']) if current_departure else None
                next_departure = parse_time_24plus(next_pt_leg['dep_time'])
                
                # Calculate transfer time
                transfer_time_minutes = 0
                if current_arrival and next_departure:
                    transfer_timedelta = next_departure - current_arrival
                    transfer_time_minutes = transfer_timedelta.total_seconds() / 60
                
                # Get line information
                current_line = current_pt_leg['transit_line']
                next_line = next_pt_leg['transit_line']
                line_change = current_line != next_line
                
                # Extract line types for analysis
                current_line_type = extract_line_type(current_line)
                next_line_type = extract_line_type(next_line)
                line_type_change = current_line_type != next_line_type
                
                # Get time of day (hour from departure time)
                time_of_day = current_departure.hour if current_departure else 0
                
                # Store detailed transfer information
                transfer_detail = {
                    'person': person,
                    'trip_id': trip_id,
                    'transfer_type': transfer_type,
                    'current_egress_stop': current_pt_leg['egress_stop_id'],
                    'next_access_stop': next_pt_leg['access_stop_id'],
                    'current_egress_stop_base': current_egress_base,
                    'next_access_stop_base': next_access_base,
                    'current_line': current_line,
                    'next_line': next_line,
                    'current_line_type': current_line_type,
                    'next_line_type': next_line_type,
                    'line_change': line_change,
                    'line_type_change': line_type_change,
                    'walking_legs_between': len(walking_legs),
                    'walking_distance': walking_distance,
                    'walking_distance_category': categorize_transfer_distance(walking_distance),
                    'walking_time': str(walking_time),
                    'walking_time_category': categorize_transfer_time(str(walking_time)),
                    'transfer_time_minutes': transfer_time_minutes,
                    'transfer_time_category': categorize_transfer_time(f"00:{int(transfer_time_minutes):02d}:00"),
                    'current_pt_departure': str(current_departure.time()) if current_departure else '',
                    'current_pt_arrival': str(current_arrival.time()) if current_arrival else '',
                    'next_pt_departure': str(next_departure.time()) if next_departure else '',
                    'time_of_day': time_of_day
                }
                
                transfer_details.append(transfer_detail)

    return _print_transfer_results(same_stop_transfers, different_stop_transfers, 
                                 total_pt_legs, transfer_details)


def _print_transfer_results(same_stop_transfers, different_stop_transfers, total_pt_legs, transfer_details):
    """Print formatted transfer analysis results."""
    print("\n=== TRANSFER ANALYSIS RESULTS (PT -> WALK -> PT) ===")
    print("Note: Transfer matching uses base stop IDs (numbers before first colon)")
    print(f"Total PT legs: {total_pt_legs}")
    print(f"Transfers on the same stop: {same_stop_transfers}")
    print(f"Transfers on different stops: {different_stop_transfers}")
    print(f"Total PT transfers: {same_stop_transfers + different_stop_transfers}")
    
    total_transfers = same_stop_transfers + different_stop_transfers
    if total_transfers > 0:
        print("\n=== PERCENTAGES ===")
        print(f"Same stop transfers: {same_stop_transfers/total_transfers*100:.1f}% of all PT transfers")
        print(f"Different stop transfers: {different_stop_transfers/total_transfers*100:.1f}% of all PT transfers")
        print(f"PT transfer rate: {total_transfers/total_pt_legs*100:.1f}% of all PT legs")
        
        # Save transfer details
        if transfer_details:
            transfer_df = pd.DataFrame(transfer_details)
            transfer_df.to_csv('pt_transfer_analysis.csv', index=False)
            print(f"\n✅ Detailed transfer analysis saved to pt_transfer_analysis.csv ({len(transfer_details)} transfers)")
            
            # Print some basic statistics
            line_changes = sum(1 for details in transfer_details if details['line_change'])
            print(f"\n=== LINE CHANGE ANALYSIS ===")
            print(f"Transfers with line change: {line_changes:,} ({100*line_changes/total_transfers:.1f}%)")
            
            # Most common line type combinations
            type_combinations = Counter(
                (details['current_line_type'], details['next_line_type']) 
                for details in transfer_details
            )
            print("\nMost common line type combinations:")
            for (from_type, to_type), count in type_combinations.most_common(5):
                pct = (count / total_transfers) * 100
                print(f"  {from_type} → {to_type}: {count} ({pct:.1f}%)")
    else:
        print("\nNo PT transfers found in the dataset")
    
    return {
        'same_stop_transfers': same_stop_transfers,
        'different_stop_transfers': different_stop_transfers,
        'total_pt_legs': total_pt_legs,
        'transfer_details': transfer_details
    }


if __name__ == "__main__":
    csv_file = "output_legs.csv"
    
    try:
        analyze_pt_transfers_with_walking(csv_file)
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        traceback.print_exc()
