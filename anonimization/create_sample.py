"""
Create a fixed sample from the original Zurich data for anonymization testing.
"""

import pandas as pd
import numpy as np

def create_fixed_sample(input_file: str, output_file: str, sample_size: int = 10000, random_state: int = 42):
    """
    Create a fixed sample from the original data.
    
    Parameters:
    -----------
    input_file : str
        Path to the original data file
    output_file : str
        Path to save the sampled data
    sample_size : int
        Number of records to sample
    random_state : int
        Random seed for reproducibility
    """
    
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset has {len(df)} records")
    
    # Filter out records with missing coordinates
    df_valid = df.dropna(subset=['home_x', 'home_y'])
    print(f"After removing missing coordinates: {len(df_valid)} records")
    
    # Sample the data
    if len(df_valid) > sample_size:
        df_sample = df_valid.sample(n=sample_size, random_state=random_state)
        print(f"Sampled {sample_size} records")
    else:
        df_sample = df_valid
        print(f"Using all {len(df_valid)} records (less than requested sample size)")
    
    # Save the sample
    df_sample.to_csv(output_file, index=False)
    print(f"Sample saved to {output_file}")
    
    # Print some statistics
    print("\nSample statistics:")
    print(f"  X coordinate range: [{df_sample['home_x'].min():.0f}, {df_sample['home_x'].max():.0f}]")
    print(f"  Y coordinate range: [{df_sample['home_y'].min():.0f}, {df_sample['home_y'].max():.0f}]")
    
    return df_sample


if __name__ == "__main__":
    input_file = "statpop_original_zurich.csv"
    output_file = "statpop_sample_10k.csv"
    
    create_fixed_sample(input_file, output_file, sample_size=10000)
