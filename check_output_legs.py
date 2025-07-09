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
    print("\n" + "="*50)
    print("EXPLORING OUTPUT_LEGS.CSV FILE")
    print("="*50)
    
    legs_file = "output_legs.csv"
    
    try:
        # Read the CSV file
        print(f"Reading {legs_file}...")
        df_legs = pd.read_csv(legs_file, sep=';')
        
        print(f"✓ Successfully loaded {legs_file}")
        print(f"Shape: {df_legs.shape} (rows, columns)")
        
        # Basic information
        print("\n--- BASIC INFORMATION ---")
        print(f"Number of rows: {len(df_legs):,}")
        print(f"Number of columns: {len(df_legs.columns)}")
        print(f"Memory usage: {df_legs.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column information
        print("\n--- COLUMNS ---")
        print("Column names:")
        for i, col in enumerate(df_legs.columns, 1):
            print(f"{i:2d}. {col}")
        
        # Data types
        print("\n--- DATA TYPES ---")
        print(df_legs.dtypes)
        
        # Missing values
        print("\n--- MISSING VALUES ---")
        missing_counts = df_legs.isnull().sum()
        missing_pct = (missing_counts / len(df_legs)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_counts,
            'Missing %': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # First few rows
        print("\n--- FIRST 5 ROWS ---")
        print(df_legs.head())
        
        # Unique values for categorical columns
        print("\n--- UNIQUE VALUES IN KEY COLUMNS ---")
        potential_categorical = ['mode', 'start_activity_type', 'end_activity_type']
        
        for col in potential_categorical:
            if col in df_legs.columns:
                unique_count = df_legs[col].nunique()
                print(f"\n{col}: {unique_count} unique values")
                if unique_count <= 20:
                    value_counts = df_legs[col].value_counts()
                    print(value_counts)
                else:
                    print("Top 10 most common values:")
                    print(df_legs[col].value_counts().head(10))
        
        # Time-related analysis
        time_columns = [col for col in df_legs.columns if 'time' in col.lower()]
        if time_columns:
            print(f"\n--- TIME COLUMNS ANALYSIS ---")
            for col in time_columns:
                print(f"\n{col}:")
                print(f"  Min: {df_legs[col].min()}")
                print(f"  Max: {df_legs[col].max()}")
                print(f"  Sample values: {df_legs[col].dropna().head(5).tolist()}")
        
        # Distance analysis if available
        distance_columns = [col for col in df_legs.columns if 'distance' in col.lower()]
        if distance_columns:
            print(f"\n--- DISTANCE COLUMNS ANALYSIS ---")
            for col in distance_columns:
                print(f"\n{col}:")
                print(df_legs[col].describe())
        
        # Trip/Person ID analysis
        id_columns = [col for col in df_legs.columns if 'id' in col.lower()]
        if id_columns:
            print(f"\n--- ID COLUMNS ANALYSIS ---")
            for col in id_columns:
                unique_count = df_legs[col].nunique()
                print(f"{col}: {unique_count:,} unique values")
        
        # Sample of random rows
        print("\n--- RANDOM SAMPLE (5 ROWS) ---")
        print(df_legs.sample(5))
        
        # Summary statistics for numeric columns
        print("\n--- NUMERIC COLUMNS SUMMARY ---")
        numeric_cols = df_legs.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df_legs[numeric_cols].describe())
        
        # Link analysis (start_link and end_link)
        link_columns = ['start_link', 'end_link']
        existing_link_cols = [col for col in link_columns if col in df_legs.columns]
        
        if existing_link_cols:
            print("\n--- LINK COLUMNS ANALYSIS ---")
            for col in existing_link_cols:
                print(f"\n{col}:")
                unique_count = df_legs[col].nunique()
                null_count = df_legs[col].isnull().sum()
                print(f"  Unique values: {unique_count:,}")
                print(f"  Null values: {null_count:,} ({null_count/len(df_legs)*100:.2f}%)")
                
                # Sample values
                sample_values = df_legs[col].dropna().head(10).tolist()
                print(f"  Sample values: {sample_values}")
                
                # Check for patterns in link IDs
                non_null_links = df_legs[col].dropna()
                if len(non_null_links) > 0:
                    # Check if links contain certain patterns
                    link_samples = non_null_links.astype(str)
                    has_colon = link_samples.str.contains(':').sum()
                    has_dot = link_samples.str.contains('\.').sum()
                    has_underscore = link_samples.str.contains('_').sum()
                    
                    print(f"  Links with ':': {has_colon:,} ({has_colon/len(link_samples)*100:.1f}%)")
                    print(f"  Links with '.': {has_dot:,} ({has_dot/len(link_samples)*100:.1f}%)")
                    print(f"  Links with '_': {has_underscore:,} ({has_underscore/len(link_samples)*100:.1f}%)")
                    
                    # Check length distribution
                    link_lengths = link_samples.str.len()
                    print(f"  Link ID length - Min: {link_lengths.min()}, Max: {link_lengths.max()}, Mean: {link_lengths.mean():.1f}")
            
            # Compare start_link and end_link if both exist
            if 'start_link' in df_legs.columns and 'end_link' in df_legs.columns:
                print(f"\n--- START_LINK vs END_LINK COMPARISON ---")
                
                # Same link trips (where start_link == end_link)
                same_link_mask = df_legs['start_link'] == df_legs['end_link']
                same_link_count = same_link_mask.sum()
                print(f"Trips with same start and end link: {same_link_count:,} ({same_link_count/len(df_legs)*100:.2f}%)")
                
                # Links that appear in both start and end
                start_links = set(df_legs['start_link'].dropna())
                end_links = set(df_legs['end_link'].dropna())
                common_links = start_links.intersection(end_links)
                print(f"Links that appear as both start and end: {len(common_links):,}")
                print(f"Only start links: {len(start_links - end_links):,}")
                print(f"Only end links: {len(end_links - start_links):,}")
                
                # Most frequent start and end links
                print(f"\nTop 10 most frequent start_links:")
                print(df_legs['start_link'].value_counts().head(10))
                
                print(f"\nTop 10 most frequent end_links:")
                print(df_legs['end_link'].value_counts().head(10))
        
        return df_legs
        
    except FileNotFoundError:
        print(f"❌ Error: {legs_file} not found in current directory")
        print("Available CSV files:")
        csv_files = glob.glob("*.csv")
        for f in csv_files:
            print(f"  - {f}")
        return None
        
    except Exception as e:
        print(f"❌ Error reading {legs_file}: {str(e)}")
        return None

# Call both exploration functions
if __name__ == "__main__":
    # Explore output_legs.csv
    df_legs = explore_output_legs()