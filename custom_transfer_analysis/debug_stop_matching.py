#!/usr/bin/env python3
"""
Debug stop ID matching between MATSim output_legs.csv and Zurich_stops.geojson.
- Extract all unique stop IDs from output_legs.csv
- Extract all stop IDs from Zurich_stops.geojson
- Show exact values and matching details
- Analyze distribution of stop IDs per stop and matching rates
"""
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict, Counter

# --- Load Zurich stops from GeoJSON ---
from pathlib import Path
print("=== Loading Zurich GeoJSON File ===")
geojson_path = Path("stops_by_canton/Zurich_stops.geojson")
with open(geojson_path, "r", encoding="utf-8") as f:
    zurich_geo = json.load(f)

# Build mapping of stop IDs to stop names and analyze stop structure
zurich_stop_ids = set()
id_to_name = {}
zurich_base_to_stops = defaultdict(list)  # base_id -> list of (name, [full_ids])
stop_name_to_ids = defaultdict(list)  # Group stop IDs by stop name
stop_id_counts = Counter()  # Count how many IDs each stop has

for feature in zurich_geo["features"]:
    props = feature["properties"]
    stop_id_list = props.get("stop_id", [])
    name = props.get("name", "Unknown")
    
    stop_id_counts[len(stop_id_list)] += 1  # Count distribution
    stop_name_to_ids[name] = stop_id_list
    
    for sid in stop_id_list:
        zurich_stop_ids.add(sid)
        id_to_name[sid] = name
        base_id = sid.split(":")[0].split(".")[0]
        zurich_base_to_stops[base_id].append((name, stop_id_list))

print(f"Loaded {len(zurich_stop_ids)} unique stop IDs from {len(zurich_geo['features'])} stops in Zurich")

# Show distribution of stop IDs per stop
print(f"\n=== Stop ID Distribution Analysis ===")
print("Number of stop IDs per stop location:")
for num_ids, count in sorted(stop_id_counts.items()):
    print(f"  {count} stops have {num_ids} stop ID(s)")

# Show examples of stops with multiple IDs
print(f"\nExamples of stops with multiple stop IDs:")
multi_id_stops = [(name, ids) for name, ids in stop_name_to_ids.items() if len(ids) > 1]
for name, ids in multi_id_stops[:5]:
    print(f"  {name}: {len(ids)} IDs -> {ids}")

# --- Load all stop IDs from output_legs.csv ---
print("\n=== Loading MATSim output_legs.csv ===")
legs_path = Path("output_legs.csv")
df = pd.read_csv(legs_path, sep=";")

# Extract all stop IDs 
stop_id_cols = [col for col in df.columns if "stop_id" in col]
print(f"Columns containing 'stop_id': {stop_id_cols}")

output_stop_ids = set()
output_id_to_context = {}
output_base_to_ids = defaultdict(list)
for col in stop_id_cols:
    for sid in df[col].dropna().unique():
        output_stop_ids.add(sid)
        if sid not in output_id_to_context:
            output_id_to_context[sid] = []
        output_id_to_context[sid].append(col)
        base_id = sid.split(":")[0].split(".")[0]
        output_base_to_ids[base_id].append(sid)

print(f"Found {len(output_stop_ids)} unique stop IDs in output_legs.csv")

# Show sample stop IDs from output_legs.csv with exact formats
print(f"\nSample stop IDs from output_legs.csv (first 20):")
sample_output_ids = list(output_stop_ids)[:20]
for sid in sample_output_ids:
    print(f"  '{sid}' (columns: {output_id_to_context[sid]})")

# --- Attempt exact matching ---
print("\n=== Exact Matching Analysis ===")
matched = set()
unmatched = set()

for sid in output_stop_ids:
    if sid in zurich_stop_ids:
        matched.add(sid)
    else:
        unmatched.add(sid)

never_matched = zurich_stop_ids - matched

print(f"Exact matches: {len(matched)} out of {len(output_stop_ids)} output stop IDs")
print(f"Unmatched output stop IDs: {len(unmatched)}")
print(f"Zurich stops never matched: {len(never_matched)} out of {len(zurich_stop_ids)}")

# Show exact matching results
print(f"\n=== Exact Matches Found (first 20) ===")
for i, sid in enumerate(list(matched)[:20]):
    print(f"  MATCH: '{sid}' -> {id_to_name[sid]}")

print(f"\n=== Unmatched Output Stop IDs (first 20) ===")
for sid in list(unmatched)[:20]:
    print(f"  NO MATCH: '{sid}'")

print(f"\n=== Zurich Stops Never Matched (first 20) ===")
for sid in list(never_matched)[:20]:
    print(f"  NEVER USED: '{sid}' ({id_to_name[sid]})")

# --- Stop-level matching analysis ---
print(f"\n=== Stop-Level Matching Analysis ===")
stops_with_matches = set()
stops_without_matches = set()

for stop_name, stop_id_list in stop_name_to_ids.items():
    has_match = any(sid in matched for sid in stop_id_list)
    if has_match:
        stops_with_matches.add(stop_name)
        matched_ids = [sid for sid in stop_id_list if sid in matched]
        unmatched_ids = [sid for sid in stop_id_list if sid not in matched]
        if len(stop_id_list) > 1:  # Only show multi-ID stops
            print(f"  {stop_name}: {len(matched_ids)}/{len(stop_id_list)} IDs matched")
            if matched_ids:
                print(f"    Matched: {matched_ids}")
            if unmatched_ids:
                print(f"    Not matched: {unmatched_ids}")
    else:
        stops_without_matches.add(stop_name)

print(f"\nStop-level summary:")
print(f"  Stops with at least one matched ID: {len(stops_with_matches)}")
print(f"  Stops with no matched IDs: {len(stops_without_matches)}")

# --- Base Stop ID Analysis ---
print(f"\n=== Base Stop ID Analysis ===")
def extract_base_stop_id(stop_string):
    """Extract base stop ID (number before first colon)."""
    if pd.isna(stop_string) or not stop_string:
        return None
    try:
        return stop_string.split(':')[0].split('.')[0]
    except:
        return None

# Extract base IDs from output stops
output_base_ids = set()
for sid in output_stop_ids:
    base_id = extract_base_stop_id(sid)
    if base_id:
        output_base_ids.add(base_id)

# Extract base IDs from Zurich stops
zurich_base_ids = set()
for sid in zurich_stop_ids:
    base_id = extract_base_stop_id(sid)
    if base_id:
        zurich_base_ids.add(base_id)

base_matches = output_base_ids & zurich_base_ids
base_output_only = output_base_ids - zurich_base_ids
base_zurich_only = zurich_base_ids - output_base_ids

print(f"Base stop IDs in output_legs.csv: {len(output_base_ids)}")
print(f"Base stop IDs in Zurich GeoJSON: {len(zurich_base_ids)}")
print(f"Base IDs that match: {len(base_matches)}")
print(f"Base IDs only in output: {len(base_output_only)}")
print(f"Base IDs only in Zurich: {len(base_zurich_only)}")

print(f"\nSample base IDs only in output: {list(base_output_only)[:10]}")
print(f"Sample base IDs only in Zurich: {list(base_zurich_only)[:10]}")
print(f"Sample matching base IDs: {list(base_matches)[:10]}")

# --- DETAILED ANALYSIS: BUCHEGGPLATZ STOP ---
print(f"\n" + "=" * 60)
print(f"🔍 DETAILED ANALYSIS: BUCHEGGPLATZ STOP")
print(f"=" * 60)

# Bucheggplatz details from GeoJSON
bucheggplatz_base_id = "8591101"
bucheggplatz_stop_ids = [
    "8591101.link:373245", 
    "8591101.link:373246", 
    "8591101.link:824719", 
    "8591101.link:824720", 
    "8591101.link:pt_8591101"
]

print(f"📍 Stop Name: Zürich, Bucheggplatz")
print(f"📍 Base Stop ID: {bucheggplatz_base_id}")
print(f"📍 All Stop IDs in GeoJSON: {len(bucheggplatz_stop_ids)} variants")
for i, stop_id in enumerate(bucheggplatz_stop_ids, 1):
    print(f"   {i}. {stop_id}")

# Check which Bucheggplatz stop IDs appear in MATSim output
print(f"\n🔍 BUCHEGGPLATZ IN MATSIM OUTPUT:")
buchegg_exact_matches = [sid for sid in bucheggplatz_stop_ids if sid in output_stop_ids]
buchegg_missing = [sid for sid in bucheggplatz_stop_ids if sid not in output_stop_ids]

print(f"✅ Exact matches found: {len(buchegg_exact_matches)}")
for sid in buchegg_exact_matches:
    print(f"   ✓ {sid} (appears in: {output_id_to_context[sid]})")

print(f"❌ Missing from output: {len(buchegg_missing)}")
for sid in buchegg_missing:
    print(f"   ✗ {sid}")

# Count total rows in output_legs.csv related to Bucheggplatz (any variant)
print(f"\n📊 BUCHEGGPLATZ USAGE IN OUTPUT_LEGS.CSV:")
buchegg_access_count = 0
buchegg_egress_count = 0
buchegg_total_rows = 0

# Check both access_stop_id and egress_stop_id columns
for col in stop_id_cols:
    for sid in bucheggplatz_stop_ids:
        if sid in df[col].values:
            count = (df[col] == sid).sum()
            if col == 'access_stop_id':
                buchegg_access_count += count
            elif col == 'egress_stop_id':
                buchegg_egress_count += count
            buchegg_total_rows += count
            if count > 0:
                print(f"   {col}: {sid} appears {count} times")

# Also check for any base ID matches
base_matches_in_output = 0
for col in stop_id_cols:
    for sid in df[col].dropna():
        if extract_base_stop_id(sid) == bucheggplatz_base_id:
            base_matches_in_output += 1

print(f"\n📈 BUCHEGGPLATZ SUMMARY:")
print(f"   🚌 Total access boardings: {buchegg_access_count}")
print(f"   🚪 Total egress alightings: {buchegg_egress_count}")
print(f"   📋 Total rows mentioning Bucheggplatz: {buchegg_total_rows}")
print(f"   🔢 Base ID matches (any variant): {base_matches_in_output}")

# Base ID consolidation benefit
if base_matches_in_output > buchegg_total_rows:
    print(f"   💡 Base ID matching captures {base_matches_in_output - buchegg_total_rows} additional variants!")

# Check if Bucheggplatz appears in transfer analysis
print(f"\n🔄 BUCHEGGPLATZ IN TRANSFER ANALYSIS:")
try:
    transfer_df = pd.read_csv('pt_transfer_analysis.csv')
    buchegg_transfer_rows = 0
    
    # Check current_egress_stop and next_access_stop
    for col in ['current_egress_stop', 'next_access_stop']:
        if col in transfer_df.columns:
            # Check exact matches
            for sid in bucheggplatz_stop_ids:
                count = (transfer_df[col] == sid).sum()
                buchegg_transfer_rows += count
                if count > 0:
                    print(f"   {col}: {sid} appears {count} times")
            
            # Check base ID matches
            base_matches = transfer_df[col].apply(
                lambda x: extract_base_stop_id(x) == bucheggplatz_base_id if pd.notna(x) else False
            ).sum()
            if base_matches > 0:
                print(f"   {col}: base ID {bucheggplatz_base_id} matches {base_matches} times")
    
    print(f"   📊 Total transfer-related rows: {buchegg_transfer_rows}")
    
except FileNotFoundError:
    print(f"   ⚠️ pt_transfer_analysis.csv not found")

# Check final results in transfer data
print(f"\n📋 BUCHEGGPLATZ IN FINAL TRANSFER DATA:")
try:
    with open('stop_transfer_data_by_canton.json', 'r') as f:
        transfer_data = json.load(f)
    
    zurich_data = transfer_data.get('cantons', {}).get('Zurich', {})
    
    # Look for Bucheggplatz in the final data
    buchegg_found = False
    for stop_id, data in zurich_data.items():
        if stop_id != '_canton_summary' and bucheggplatz_base_id in stop_id:
            buchegg_found = True
            print(f"   ✅ Found: {stop_id}")
            print(f"      🚌 Total boardings: {data.get('total_boardings', 0)}")
            print(f"      🔄 Transfers in: {data.get('total_transfers_in', 0)}")
            print(f"      🔄 Transfers out: {data.get('total_transfers_out', 0)}")
            
            # Show line transfers if any
            line_transfers = data.get('line_transfers', {})
            if line_transfers:
                print(f"      🚏 Line transfer patterns:")
                for from_line, to_lines in line_transfers.items():
                    for to_line, count in to_lines.items():
                        print(f"         {from_line} → {to_line}: {count} transfers")
            break
    
    if not buchegg_found:
        print(f"   ❌ Bucheggplatz not found in final transfer data")
        
except FileNotFoundError:
    print(f"   ⚠️ stop_transfer_data_by_canton.json not found")

print(f"=" * 60)
