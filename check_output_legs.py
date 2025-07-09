import pickle
import pandas as pd
import numpy as np
import glob

# Read the pickle file
pickle_file_path = '/cluster/project/cmdp/chaoch/switzerland_data/cache/matsim.simulation.run__a08f881cee4c481521a1ae6c8d50fea3.p'

print("Loading pickle file...")
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

print(f"Pickle file loaded successfully!")
print(f"Type of loaded data: {type(data)}")

# Basic exploration
if isinstance(data, dict):
    print(f"\nDictionary with {len(data)} keys:")
    print("Keys:", list(data.keys())[:10])  # Show first 10 keys
    
    # Explore each key
    for key in list(data.keys())[:5]:  # Show details for first 5 keys
        print(f"\nKey: '{key}'")
        print(f"  Type: {type(data[key])}")
        if hasattr(data[key], 'shape'):
            print(f"  Shape: {data[key].shape}")
        elif hasattr(data[key], '__len__'):
            print(f"  Length: {len(data[key])}")
        
        # If it's a DataFrame or similar, show basic info
        if isinstance(data[key], pd.DataFrame):
            print(f"  Columns: {list(data[key].columns)}")
            print(f"  First few rows:")
            print(data[key].head(3))
        elif isinstance(data[key], (list, tuple)) and len(data[key]) > 0:
            print(f"  First few elements: {data[key][:3]}")
        elif isinstance(data[key], dict):
            print(f"  Dict keys: {list(data[key].keys())[:5]}")

elif isinstance(data, pd.DataFrame):
    print(f"\nDataFrame with shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Data types:")
    print(data.dtypes)
    print(f"\nFirst 10 rows:")
    print(data.head(10))
    print(f"\nBasic statistics:")
    print(data.describe())

elif isinstance(data, (list, tuple)):
    print(f"\nList/Tuple with {len(data)} elements")
    print(f"Type of first element: {type(data[0]) if len(data) > 0 else 'Empty'}")
    if len(data) > 0:
        print(f"First few elements: {data[:5]}")

elif isinstance(data, np.ndarray):
    print(f"\nNumPy array with shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"First few elements: {data.flat[:10]}")

else:
    print(f"\nData content (first 1000 chars): {str(data)[:1000]}")

print(f"\nMemory usage: {data.__sizeof__() / (1024*1024):.2f} MB" if hasattr(data, '__sizeof__') else "Memory usage: Unknown")

def explore_output_legs():
    """
    Explores the output_legs.csv file to understand its structure and contents.
    """
    print("\n" + "="*60)
    print("📊 EXPLORING OUTPUT_LEGS.CSV FILE")
    print("="*60)
    
    legs_file = "output_legs.csv"
    
    try:
        # Read first 1000 rows for quick analysis
        print(f"Reading {legs_file} (first 1000 rows)...")
        df_legs = pd.read_csv(legs_file, sep=';', nrows=1000)
        
        print(f"✓ Successfully loaded {legs_file}")
        print(f"📋 Shape: {df_legs.shape} (rows, columns)")
        print(f"💾 Memory usage: {df_legs.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column overview
        print(f"\n📄 COLUMNS ({len(df_legs.columns)} total):")
        for i, col in enumerate(df_legs.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Missing values summary
        print(f"\n❌ MISSING VALUES:")
        missing_counts = df_legs.isnull().sum()
        missing_with_data = missing_counts[missing_counts > 0]
        if len(missing_with_data) > 0:
            for col, count in missing_with_data.items():
                pct = (count / len(df_legs)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print("  No missing values found")
        
        # Sample data
        print(f"\n👁️  SAMPLE DATA (first 3 rows):")
        print(df_legs.head(3).to_string())
        
        # Key categorical columns
        print(f"\n🏷️  TRANSPORT MODES:")
        if 'mode' in df_legs.columns:
            mode_counts = df_legs['mode'].value_counts()
            for mode, count in mode_counts.items():
                pct = (count / len(df_legs)) * 100
                print(f"  {mode}: {count} ({pct:.1f}%)")
        
        return df_legs
        
    except FileNotFoundError:
        print(f"❌ Error: {legs_file} not found in current directory")
        csv_files = glob.glob("*.csv")
        if csv_files:
            print("Available CSV files:")
            for f in csv_files:
                print(f"  - {f}")
        return None
        
    except Exception as e:
        print(f"❌ Error reading {legs_file}: {str(e)}")
        return None

def check_transit_missing_by_mode():
    """
    Checks how transit_line and transit_route values are missing for different transport modes
    and provides examples of each transport mode.
    """
    print("\n" + "="*60)
    print("🚌 TRANSIT LINE & ROUTE ANALYSIS BY MODE")
    print("="*60)
    
    legs_file = "output_legs.csv"
    
    try:
        # Read sample data
        print(f"Reading {legs_file} (first 5000 rows for analysis)...")
        df_legs = pd.read_csv(legs_file, sep=';', nrows=5000)
        
        print(f"✓ Loaded sample: {df_legs.shape}")
        
        # Check if required columns exist
        required_cols = ['mode', 'transit_line', 'transit_route']
        missing_cols = [col for col in required_cols if col not in df_legs.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print("Available columns:", ', '.join(df_legs.columns))
            return
        
        # Transport modes overview
        print(f"\n📊 TRANSPORT MODES:")
        mode_counts = df_legs['mode'].value_counts()
        total_trips = len(df_legs)
        
        for mode, count in mode_counts.items():
            percentage = (count / total_trips) * 100
            print(f"  {mode}: {count} ({percentage:.1f}%)")
        
        # Missing values analysis by mode
        print(f"\n🚇 MISSING TRANSIT DATA BY MODE:")
        print(f"{'Mode':<15} {'Total':<8} {'Line Missing':<15} {'Route Missing':<16}")
        print("-" * 60)
        
        for mode in mode_counts.index:
            mode_data = df_legs[df_legs['mode'] == mode]
            mode_count = len(mode_data)
            
            line_missing = mode_data['transit_line'].isnull().sum()
            route_missing = mode_data['transit_route'].isnull().sum()
            
            line_pct = (line_missing / mode_count) * 100
            route_pct = (route_missing / mode_count) * 100
            
            print(f"{mode:<15} {mode_count:<8} {line_missing} ({line_pct:.1f}%){'':<5} {route_missing} ({route_pct:.1f}%)")
        
        # Examples by mode
        print(f"\n🎯 EXAMPLES BY TRANSPORT MODE:")
        for mode in mode_counts.index[:5]:  # Show first 5 modes
            mode_data = df_legs[df_legs['mode'] == mode]
            
            # Get examples of lines and routes
            line_examples = mode_data['transit_line'].dropna().unique()[:2]
            route_examples = mode_data['transit_route'].dropna().unique()[:2]
            
            print(f"\n📍 {mode.upper()}:")
            print(f"   Lines: {list(line_examples) if len(line_examples) > 0 else 'None'}")
            print(f"   Routes: {list(route_examples) if len(route_examples) > 0 else 'None'}")
        
        # Overall summary
        has_line = (~df_legs['transit_line'].isnull()).sum()
        has_route = (~df_legs['transit_route'].isnull()).sum()
        
        print(f"\n📈 OVERALL SUMMARY:")
        print(f"  Trips with transit line: {has_line} ({has_line/total_trips*100:.1f}%)")
        print(f"  Trips with transit route: {has_route} ({has_route/total_trips*100:.1f}%)")
        print(f"  Unique lines: {df_legs['transit_line'].nunique()}")
        print(f"  Unique routes: {df_legs['transit_route'].nunique()}")
        
    except FileNotFoundError:
        print(f"❌ Error: {legs_file} not found in current directory")
        csv_files = glob.glob("*.csv")
        if csv_files:
            print("Available CSV files:", ', '.join(csv_files))
    except Exception as e:
        print(f"❌ Error analyzing transit data: {str(e)}")

# Call both exploration functions
if __name__ == "__main__":
    # Explore output_legs.csv
    df_legs = explore_output_legs()
    
    # Check transit missing values by mode
    check_transit_missing_by_mode()