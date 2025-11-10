"""
Concise analysis of anonymization results.
Creates two key plots:
1. Distribution of anonymization distances
2. Distribution of original neighbors within displacement radius
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_anonymization_plots():
    """Create the two main analysis plots."""
    
    # File mapping
    files = {
        'Donut Geomask': '/cluster/home/chaoch/full_donut_geomask.csv',
        'K-Anonymity': '/cluster/home/chaoch/full_k_anonymity.csv', 
        'Differential Privacy': '/cluster/home/chaoch/full_differential_privacy.csv',
        'Voronoi Mask': '/cluster/home/chaoch/full_voronoi_mask.csv'
    }
    
    # Colors for each method
    colors = ['#FF1744', '#00E676', '#2979FF', '#FFD600']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Anonymization Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Anonymization Distance Distribution
    ax1 = axes[0]
    distance_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    distance_labels = ['0-50m', '50-100m', '100-200m', '200-500m', '500-1000m', '>1000m']
    
    x_pos = np.arange(len(distance_labels))
    width = 0.2
    
    for i, (method, file_path) in enumerate(files.items()):
        try:
            df = pd.read_csv(file_path)
            distances = df['anonymization_distance_m'].values
            
            # Count in each bin
            bin_counts = []
            for j in range(len(distance_bins)-1):
                start, end = distance_bins[j], distance_bins[j+1]
                if end == float('inf'):
                    count = np.sum(distances >= start)
                else:
                    count = np.sum((distances >= start) & (distances < end))
                bin_counts.append(count)
            
            # Plot bars
            offset = (i - 1.5) * width
            bars = ax1.bar(x_pos + offset, bin_counts, width, 
                          label=method, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + max(bin_counts)*0.01,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)
                    
        except Exception as e:
            print(f"Error processing {method}: {e}")
    
    ax1.set_xlabel('Anonymization Distance')
    ax1.set_ylabel('Number of Households')
    ax1.set_title('How Far Are Homes Moved?')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(distance_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Original Neighbors Within Displacement
    ax2 = axes[1]
    neighbor_bins = [0, 5, 10, 20, 50, float('inf')]
    neighbor_labels = ['0-5', '6-10', '11-20', '21-50', '>50']
    
    x_pos2 = np.arange(len(neighbor_labels))
    
    for i, (method, file_path) in enumerate(files.items()):
        try:
            df = pd.read_csv(file_path)
            neighbors = df['original_neighbors_within_displacement'].values
            
            # Count in each bin
            bin_counts = []
            for j in range(len(neighbor_bins)-1):
                start, end = neighbor_bins[j], neighbor_bins[j+1]
                if j == 0:  # First bin: 0-5
                    count = np.sum((neighbors >= start) & (neighbors <= end))
                elif end == float('inf'):  # Last bin: >50
                    count = np.sum(neighbors > neighbor_bins[j])
                else:  # Middle bins: 6-10, 11-20, 21-50
                    count = np.sum((neighbors > start) & (neighbors <= end))
                bin_counts.append(count)
            
            # Plot bars
            offset = (i - 1.5) * width
            bars = ax2.bar(x_pos2 + offset, bin_counts, width,
                          label=method, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(bin_counts)*0.01,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)
                    
        except Exception as e:
            print(f"Error processing {method}: {e}")
    
    ax2.set_xlabel('Number of Original Homes Within Displacement')
    ax2.set_ylabel('Number of Households')
    ax2.set_title('How Many Neighbors Are Within Anonymization Radius?')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(neighbor_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('anonymization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create spatial visualization plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    colors = ['#FF1744', '#00E676', '#2979FF', '#FFD600']
    method_labels = ['Donut Geomask', 'K-Anonymity', 'Differential Privacy', 'Voronoi Mask']
    
    # Load original data for comparison
    spatial_files = [
        'full_donut_geomask.csv',
        'full_k_anonymity.csv', 
        'full_differential_privacy.csv',
        'full_voronoi.csv'
    ]
    
    # Get original coordinates from first file
    df_orig = pd.read_csv('statpop_original_zurich.csv')
    orig_x = df_orig['home_x'].values
    orig_y = df_orig['home_y'].values
    
    # Sample for visualization (to avoid overcrowding)
    n_sample = min(2000, len(orig_x))
    sample_idx = np.random.choice(len(orig_x), n_sample, replace=False)
    orig_x_sample = orig_x[sample_idx]
    orig_y_sample = orig_y[sample_idx]
    
    for i, (file, color, label) in enumerate(zip(spatial_files, colors, method_labels)):
        if os.path.exists(file):
            df = pd.read_csv(file)
            anon_x = df['home_x'].values[sample_idx]
            anon_y = df['home_y'].values[sample_idx]
            
            # Plot original locations (light gray background)
            axes[i].scatter(orig_x, orig_y, alpha=0.2, s=0.5, color='lightgray', 
                          label='All original', rasterized=True)
            
            # Plot sample original locations
            axes[i].scatter(orig_x_sample, orig_y_sample, alpha=0.6, s=8, color='black', 
                          label='Original (sample)', zorder=2)
            
            # Plot anonymized locations
            axes[i].scatter(anon_x, anon_y, alpha=0.8, s=10, color=color, 
                          label='Anonymized', zorder=3)
            
            # Draw displacement vectors for a subset
            vector_sample = min(200, len(orig_x_sample))
            for j in range(vector_sample):
                if j < len(orig_x_sample) and j < len(anon_x):
                    axes[i].plot([orig_x_sample[j], anon_x[j]], 
                               [orig_y_sample[j], anon_y[j]], 
                               color=color, alpha=0.3, linewidth=0.5, zorder=1)
            
            axes[i].set_title(f'{label}\nSpatial Distribution', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('X Coordinate (m)')
            axes[i].set_ylabel('Y Coordinate (m)')
            axes[i].legend(markerscale=2)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('spatial_anonymization_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Analysis complete! Plots saved to:")
    print("- anonymization_analysis.png")
    print("- spatial_anonymization_map.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    
    for method, file_path in files.items():
        try:
            df = pd.read_csv(file_path)
            distances = df['anonymization_distance_m'].values
            neighbors = df['original_neighbors_within_displacement'].values
            
            print(f"\n{method}:")
            print(f"  Distance - Mean: {np.mean(distances):.1f}m, Median: {np.median(distances):.1f}m")
            print(f"  Neighbors - Mean: {np.mean(neighbors):.1f}, Median: {np.median(neighbors):.1f}")
            print(f"  Zero displacement: {np.sum(distances == 0)} ({np.sum(distances == 0)/len(distances)*100:.1f}%)")
            print(f"  Zero neighbors: {np.sum(neighbors == 0)} ({np.sum(neighbors == 0)/len(neighbors)*100:.1f}%)")
            print(f"  5+ neighbors: {np.sum(neighbors >= 5)} ({np.sum(neighbors >= 5)/len(neighbors)*100:.1f}%)")
            
        except Exception as e:
            print(f"Error analyzing {method}: {e}")

if __name__ == "__main__":
    # Set style
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8')
    
    print("Creating anonymization analysis plots...")
    create_anonymization_plots()
