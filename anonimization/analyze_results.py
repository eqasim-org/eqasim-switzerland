"""
Analysis and comparison of anonymization methods results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree
import pyproj
from typing import List, Dict, Tuple
import os

# Import enhanced metrics
from enhanced_metrics import calculate_all_enhanced_metrics

class AnonymizationAnalyzer:
    """Analyze and compare anonymization results."""
    
    def __init__(self, crs_from="EPSG:2056", crs_to="EPSG:3857"):
        self.crs_from = crs_from
        self.crs_to = crs_to
        self.transformer_to_meters = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    
    def project_to_meters(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Project coordinates to metric CRS."""
        x_proj, y_proj = self.transformer_to_meters.transform(x_coords, y_coords)
        return np.column_stack([x_proj, y_proj])
    
    def calculate_displacement_stats(self, 
                                   original_x: np.ndarray, 
                                   original_y: np.ndarray,
                                   masked_x: np.ndarray, 
                                   masked_y: np.ndarray) -> Dict[str, float]:
        """Calculate displacement statistics between original and masked coordinates."""

        orig_coords = self.project_to_meters(original_x, original_y)
        masked_coords = self.project_to_meters(masked_x, masked_y)
        
        # Calculate Euclidean distances
        distances = np.sqrt(np.sum((orig_coords - masked_coords)**2, axis=1))
        
        return {
            'mean_displacement_m': np.mean(distances),
            'median_displacement_m': np.median(distances),
            'min_displacement_m': np.min(distances),
            'max_displacement_m': np.max(distances),
            'std_displacement_m': np.std(distances),
            'q25_displacement_m': np.percentile(distances, 25),
            'q75_displacement_m': np.percentile(distances, 75),
            'q95_displacement_m': np.percentile(distances, 95),
            'q99_displacement_m': np.percentile(distances, 99)
        }
    
    def calculate_privacy_metrics(self, 
                                original_x: np.ndarray, 
                                original_y: np.ndarray,
                                masked_x: np.ndarray, 
                                masked_y: np.ndarray,
                                k_values: List[int] = [3, 5, 10]) -> Dict:
        """Calculate privacy-related metrics."""
        
        orig_coords = self.project_to_meters(original_x, original_y)
        masked_coords = self.project_to_meters(masked_x, masked_y)
        
        metrics = {}
        
        # Build trees for nearest neighbor queries
        orig_tree = KDTree(orig_coords)
        masked_tree = KDTree(masked_coords)
        
        for k in k_values:
            # Calculate k-anonymity preservation
            k_preserved = 0
            
            for i in range(min(1000, len(orig_coords))):  # Sample for efficiency
                # Find k nearest neighbors in original data
                orig_dists, orig_idxs = orig_tree.query(orig_coords[i].reshape(1, -1), k=k+1)
                orig_neighbors = set(orig_idxs[0][1:])  # Exclude self
                
                # Find k nearest neighbors in masked data
                masked_dists, masked_idxs = masked_tree.query(masked_coords[i].reshape(1, -1), k=k+1)
                masked_neighbors = set(masked_idxs[0][1:])  # Exclude self
                
                # Check if neighborhoods overlap significantly
                overlap = len(orig_neighbors.intersection(masked_neighbors))
                if overlap >= k // 2:  # At least half of neighbors preserved
                    k_preserved += 1
            
            metrics[f'k{k}_anonymity_preservation'] = k_preserved / min(1000, len(orig_coords))
        
        # Calculate displacement metrics
        displacements = np.sqrt(np.sum((orig_coords - masked_coords)**2, axis=1))
        metrics['mean_displacement_m'] = np.mean(displacements)
        metrics['median_displacement_m'] = np.median(displacements)
        metrics['max_displacement_m'] = np.max(displacements)
        metrics['std_displacement_m'] = np.std(displacements)
        
        return metrics
    
    def calculate_utility_metrics(self, 
                                original_x: np.ndarray, 
                                original_y: np.ndarray,
                                masked_x: np.ndarray, 
                                masked_y: np.ndarray) -> Dict:
        """Calculate data utility metrics."""
        
        # Spatial distribution preservation
        orig_center = np.array([np.mean(original_x), np.mean(original_y)])
        masked_center = np.array([np.mean(masked_x), np.mean(masked_y)])
        
        orig_coords_proj = self.project_to_meters(original_x, original_y)
        masked_coords_proj = self.project_to_meters(masked_x, masked_y)
        
        center_shift = np.linalg.norm(
            self.project_to_meters(masked_center.reshape(1), orig_center.reshape(1))[0] - 
            self.project_to_meters(orig_center.reshape(1), orig_center.reshape(1))[0]
        )
        
        # Standard deviation preservation
        orig_std = np.std(orig_coords_proj, axis=0)
        masked_std = np.std(masked_coords_proj, axis=0)
        
        return {
            'center_shift_m': center_shift,
            'x_std_preservation': masked_std[0] / orig_std[0] if orig_std[0] > 0 else 1.0,
            'y_std_preservation': masked_std[1] / orig_std[1] if orig_std[1] > 0 else 1.0,
            'area_coverage_ratio': (
                (np.max(masked_x) - np.min(masked_x)) * (np.max(masked_y) - np.min(masked_y))
            ) / (
                (np.max(original_x) - np.min(original_x)) * (np.max(original_y) - np.min(original_y))
            ) if (np.max(original_x) - np.min(original_x)) > 0 and (np.max(original_y) - np.min(original_y)) > 0 else 1.0
        }
    
    def analyze_all_methods(self, original_file: str, anonymized_files: Dict[str, str]) -> pd.DataFrame:
        """Analyze all anonymization methods and compare results."""
        
        # Load original data
        print(f"Loading original data from {original_file}")
        df_orig = pd.read_csv(original_file)
        print(f"Analyzing {len(df_orig)} records")
        
        results = []
        
        for method_name, file_path in anonymized_files.items():
            print(f"\nAnalyzing {method_name}...")
            
            try:
                df_anon = pd.read_csv(file_path)
                
                if len(df_anon) != len(df_orig):
                    print(f"Warning: Size mismatch for {method_name}: {len(df_anon)} vs {len(df_orig)}")
                    continue
                
                # Extract coordinates
                orig_x = df_orig['home_x'].values
                orig_y = df_orig['home_y'].values
                masked_x = df_anon['home_x'].values
                masked_y = df_anon['home_y'].values
                
                # Remove NaN values
                valid_mask = (~pd.isna(orig_x) & ~pd.isna(orig_y) & 
                             ~pd.isna(masked_x) & ~pd.isna(masked_y))
                
                orig_x = orig_x[valid_mask]
                orig_y = orig_y[valid_mask]
                masked_x = masked_x[valid_mask]
                masked_y = masked_y[valid_mask]
                
                if len(orig_x) == 0:
                    print(f"Warning: No valid coordinates for {method_name}")
                    continue
                
                print(f"  Comparing {len(orig_x)} coordinate pairs...")
                
                # Basic privacy metrics
                metrics = self.calculate_privacy_metrics(orig_x, orig_y, masked_x, masked_y)
                metrics['method'] = method_name
                
                # Add enhanced metrics
                print(f"  Computing enhanced metrics...")
                orig_coords_meters = self.project_to_meters(orig_x, orig_y)
                mask_coords_meters = self.project_to_meters(masked_x, masked_y)
                enhanced = calculate_all_enhanced_metrics(orig_coords_meters, mask_coords_meters)
                
                # Flatten enhanced metrics into main metrics dict
                for category, value in enhanced.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, (int, float, bool, np.integer, np.floating)):
                                metrics[f'{category}_{k}'] = float(v)
                    elif isinstance(value, (int, float, bool, np.integer, np.floating)):
                        metrics[category] = float(value)
                    elif isinstance(value, np.ndarray):
                        # For arrays like Ripley's K, store as string or skip
                        metrics[f'{category}_array'] = str(value.tolist())
                
                results.append(metrics)
                print(f"  ✓ Completed analysis for {method_name}")
                
            except Exception as e:
                print(f"Error analyzing {method_name}: {e}")
                import traceback
                traceback.print_exc()
        
        if not results:
            print("No results to display")
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def create_comparison_plots(self, original_file: str, anonymized_files: Dict[str, str], output_dir: str = "."):
        """Create comparison plots for anonymization methods."""
        
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Load original data (sample for plotting to avoid overcrowding)
        df_orig = pd.read_csv(original_file)
        if len(df_orig) > 2000:
            plot_sample = df_orig.sample(n=2000, random_state=42)
        else:
            plot_sample = df_orig
        
        orig_x = plot_sample['home_x'].values
        orig_y = plot_sample['home_y'].values
        valid_mask = ~(pd.isna(orig_x) | pd.isna(orig_y))
        orig_x = orig_x[valid_mask]
        orig_y = orig_y[valid_mask]
        
        n_methods = len(anonymized_files)
        
        # Create overlay comparison plots (original + anonymized on same plot)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        # Use bright, saturated colors for better visibility
        colors = ['#FF1744', '#00E676', '#2979FF', '#FFD600']  # Bright red, green, blue, yellow
        
        for i, (method_name, file_path) in enumerate(anonymized_files.items()):
            if i >= 4:
                break
                
            try:
                df_anon = pd.read_csv(file_path)
                
                # Use same sample indices for comparison
                if len(df_anon) > 2000:
                    anon_sample = df_anon.sample(n=2000, random_state=42)
                else:
                    anon_sample = df_anon
                
                masked_x = anon_sample['home_x'].values
                masked_y = anon_sample['home_y'].values
                
                # Remove NaN values
                mask_valid = ~(pd.isna(masked_x) | pd.isna(masked_y))
                masked_x = masked_x[mask_valid]
                masked_y = masked_y[mask_valid]
                
                if len(masked_x) > 0:
                    # Plot original points in dark gray with higher opacity
                    axes[i].scatter(orig_x, orig_y, alpha=0.6, s=8, color='#424242', 
                                  label='Original', edgecolors='none', zorder=1)
                    # Plot anonymized points in bright color with higher opacity
                    axes[i].scatter(masked_x, masked_y, alpha=0.8, s=8, 
                                  color=colors[i], label='Anonymized', edgecolors='none', zorder=2)
                    axes[i].set_title(f'{method_name.replace("_", " ").title()}', 
                                     fontsize=14, fontweight='bold')
                    axes[i].set_xlabel('X Coordinate', fontsize=12)
                    axes[i].set_ylabel('Y Coordinate', fontsize=12)
                    axes[i].legend(loc='upper right', markerscale=2, fontsize=11)
                    axes[i].grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error plotting {method_name}: {e}")
        
        # Hide unused subplots
        for i in range(len(anonymized_files), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/anonymization_comparison_overlay.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved overlay comparison plot to {output_dir}/anonymization_comparison_overlay.png")
        
        # Create displacement vector plots (showing movement with arrows)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        # Sample fewer points for arrow visualization
        n_arrows = min(200, len(plot_sample))
        arrow_indices = np.random.choice(len(orig_x), size=min(n_arrows, len(orig_x)), replace=False)
        
        for i, (method_name, file_path) in enumerate(anonymized_files.items()):
            if i >= 4:
                break
                
            try:
                df_anon = pd.read_csv(file_path)
                
                # Get corresponding anonymized points
                if len(df_anon) > 2000:
                    anon_sample = df_anon.sample(n=2000, random_state=42)
                else:
                    anon_sample = df_anon
                
                masked_x = anon_sample['home_x'].values
                masked_y = anon_sample['home_y'].values
                
                # Remove NaN values
                mask_valid = ~(pd.isna(masked_x) | pd.isna(masked_y))
                masked_x = masked_x[mask_valid]
                masked_y = masked_y[mask_valid]
                
                if len(masked_x) > 0:
                    # Plot all original points in background with darker color
                    axes[i].scatter(orig_x, orig_y, alpha=0.3, s=4, color='#9E9E9E', label='Original (all)')
                    
                    # Plot displacement arrows for sample with more visible color
                    n_plot = min(len(arrow_indices), len(orig_x), len(masked_x))
                    for j in range(n_plot):
                        idx = arrow_indices[j] if j < len(arrow_indices) else j
                        if idx < len(orig_x) and idx < len(masked_x):
                            axes[i].annotate('', 
                                           xy=(masked_x[idx], masked_y[idx]), 
                                           xytext=(orig_x[idx], orig_y[idx]),
                                           arrowprops=dict(arrowstyle='->', color=colors[i], 
                                                         alpha=0.7, lw=1.2))
                    
                    # Plot sample of original and anonymized points with more intense colors
                    axes[i].scatter(orig_x[arrow_indices], orig_y[arrow_indices], 
                                  alpha=0.8, s=25, color='#1565C0', marker='o', 
                                  label='Original (sample)', zorder=3, edgecolors='white', linewidths=0.5)
                    axes[i].scatter(masked_x[arrow_indices], masked_y[arrow_indices], 
                                  alpha=0.9, s=30, color=colors[i], marker='x', 
                                  label='Anonymized', zorder=4, linewidths=2)
                    
                    axes[i].set_title(f'{method_name.replace("_", " ").title()}\nDisplacement Vectors', 
                                     fontsize=14, fontweight='bold')
                    axes[i].set_xlabel('X Coordinate', fontsize=12)
                    axes[i].set_ylabel('Y Coordinate', fontsize=12)
                    axes[i].legend(loc='upper right', markerscale=1.5, fontsize=11)
                    axes[i].grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error creating displacement plot for {method_name}: {e}")
        
        # Hide unused subplots
        for i in range(len(anonymized_files), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/displacement_vectors.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved displacement vector plot to {output_dir}/displacement_vectors.png")
        """Create comparison plots for anonymization methods."""
        
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Load original data
        df_orig = pd.read_csv(original_file)
        
        # For plotting, sample if dataset is too large
        if len(df_orig) > 5000:
            df_plot = df_orig.sample(n=5000, random_state=42)
        else:
            df_plot = df_orig
        
        orig_x = df_plot['home_x'].values
        orig_y = df_plot['home_y'].values
        valid_mask_orig = ~(pd.isna(orig_x) | pd.isna(orig_y))
        orig_x = orig_x[valid_mask_orig]
        orig_y = orig_y[valid_mask_orig]
        
        n_methods = len(anonymized_files)
        fig, axes = plt.subplots(2, (n_methods + 1) // 2 + 1, figsize=(20, 12))
        axes = axes.flatten()
        
        # Plot original data
        sns.scatterplot(x=orig_x, y=orig_y, alpha=0.5, s=1, color='black', ax=axes[0])
        axes[0].set_title('Original Data', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        
        # Plot anonymized data
        colors = sns.color_palette("husl", n_methods)
        for i, (method_name, file_path) in enumerate(anonymized_files.items()):
            try:
                df_anon = pd.read_csv(file_path)
                
                # Use same subset for plotting
                if len(df_anon) > 5000:
                    df_anon_plot = df_anon.sample(n=5000, random_state=42)
                else:
                    df_anon_plot = df_anon
                
                masked_x = df_anon_plot['home_x'].values
                masked_y = df_anon_plot['home_y'].values
                
                # Remove NaN values
                mask_valid = ~(pd.isna(masked_x) | pd.isna(masked_y))
                masked_x = masked_x[mask_valid]
                masked_y = masked_y[mask_valid]
                
                if len(masked_x) > 0:
                    sns.scatterplot(x=masked_x, y=masked_y, alpha=0.5, s=1, 
                                  color=colors[i], ax=axes[i+1])
                    axes[i+1].set_title(f'{method_name.replace("_", " ").title()}', 
                                       fontsize=14, fontweight='bold')
                    axes[i+1].set_xlabel('X Coordinate')
                    axes[i+1].set_ylabel('Y Coordinate')
                
            except Exception as e:
                print(f"Error plotting {method_name}: {e}")
        
        # Hide unused subplots
        for i in range(len(anonymized_files) + 1, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/anonymization_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create displacement histogram with seaborn
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Use the same bright colors as overlay plots
        bright_colors = ['#FF1744', '#00E676', '#2979FF', '#FFD600']
        
        for i, (method_name, file_path) in enumerate(anonymized_files.items()):
            if i >= 4:
                break
                
            try:
                df_anon = pd.read_csv(file_path)
                
                if len(df_anon) != len(df_orig):
                    print(f"Warning: Size mismatch for displacement calculation: {method_name}")
                    continue
                
                # Use full dataset for displacement (not just plot sample)
                masked_x = df_anon['home_x'].values
                masked_y = df_anon['home_y'].values
                orig_x_full = df_orig['home_x'].values
                orig_y_full = df_orig['home_y'].values
                
                # Remove NaN values
                mask_valid = (~pd.isna(masked_x) & ~pd.isna(masked_y) & 
                              ~pd.isna(orig_x_full) & ~pd.isna(orig_y_full))
                masked_x = masked_x[mask_valid]
                masked_y = masked_y[mask_valid]
                orig_x_full = orig_x_full[mask_valid]
                orig_y_full = orig_y_full[mask_valid]
                
                if len(masked_x) > 0:
                    # Calculate displacements
                    orig_coords = self.project_to_meters(orig_x_full, orig_y_full)
                    masked_coords = self.project_to_meters(masked_x, masked_y)
                    distances = np.sqrt(np.sum((orig_coords - masked_coords)**2, axis=1))
                    
                    sns.histplot(distances, bins=50, alpha=0.75, kde=True, 
                               color=bright_colors[i], ax=axes[i], linewidth=1.5)
                    axes[i].set_title(f'{method_name.replace("_", " ").title()}\nMean: {np.mean(distances):.1f}m', 
                                    fontsize=12, fontweight='bold')
                    axes[i].set_xlabel('Displacement (meters)', fontsize=11)
                    axes[i].set_ylabel('Frequency', fontsize=11)
                
            except Exception as e:
                print(f"Error creating histogram for {method_name}: {e}")
                import traceback
                traceback.print_exc()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/displacement_histograms.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {output_dir}/")


def run_comprehensive_analysis():
    """Run comprehensive analysis of anonymization results."""
    
    # File paths - using the fixed 10k sample
    original_file = "statpop_sample_10k.csv"
    anonymized_files = {
        'Donut Geomask': 'anonymized_10k_donut_geomask.csv',
        'K-Anonymity': 'anonymized_10k_k_anonymity.csv',
        'Differential Privacy': 'anonymized_10k_differential_privacy.csv',
        'Voronoi Mask': 'anonymized_10k_voronoi_mask.csv'
    }
    
    # Check if files exist
    import os
    available_files = {k: v for k, v in anonymized_files.items() if os.path.exists(v)}
    
    if not available_files:
        print("No anonymized files found. Please run the anonymization methods first.")
        return
    
    print(f"Found {len(available_files)} anonymized datasets to analyze")
    
    # Initialize analyzer
    analyzer = AnonymizationAnalyzer()
    
    # Perform analysis
    results_df = analyzer.analyze_all_methods(original_file, available_files)
    
    # Display results
    if not results_df.empty:
        print("\n" + "="*80)
        print("ANONYMIZATION METHODS COMPARISON - COMPREHENSIVE RESULTS")
        print("="*80)
        
        print("\n1. DISPLACEMENT METRICS:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['method'].upper()}:")
            print(f"  Mean displacement: {row['mean_displacement_m']:.1f} m")
            print(f"  Median displacement: {row['median_displacement_m']:.1f} m")
            print(f"  Max displacement: {row['max_displacement_m']:.1f} m")
            print(f"  Std displacement: {row['std_displacement_m']:.1f} m")
        
        print("\n" + "-" * 80)
        print("2. K-ANONYMITY PRESERVATION RATES:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['method'].upper()}:")
            for k in [3, 5, 10]:
                col = f'k{k}_anonymity_preservation'
                if col in row:
                    print(f"  k={k}: {row[col]:.3f}")
        
        print("\n" + "-" * 80)
        print("3. RE-IDENTIFICATION RISK:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['method'].upper()}:")
            if 'reidentification_reidentification_rate' in row:
                print(f"  Re-identification rate: {row['reidentification_reidentification_rate']:.3f}")
            if 'reidentification_unique_points_ratio' in row:
                print(f"  Unique points ratio: {row['reidentification_unique_points_ratio']:.3f}")
            if 'reidentification_avg_confusion_distance' in row:
                print(f"  Avg confusion distance: {row['reidentification_avg_confusion_distance']:.1f} m")
        
        print("\n" + "-" * 80)
        print("4. SPATIAL DISTRIBUTION PRESERVATION:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['method'].upper()}:")
            if 'entropy_preservation' in row:
                print(f"  Entropy preservation: {row['entropy_preservation']:.3f}")
            if 'ripleys_k_correlation' in row:
                print(f"  Ripley's K correlation: {row['ripleys_k_correlation']:.3f}")
        
        print("\n" + "-" * 80)
        print("5. NEIGHBOR PRESERVATION:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['method'].upper()}:")
            if 'neighbor_preservation_mean_neighbor_preservation' in row:
                print(f"  Mean neighbor preservation: {row['neighbor_preservation_mean_neighbor_preservation']:.3f}")
            if 'neighbor_preservation_mean_rank_correlation' in row:
                print(f"  Mean rank correlation: {row['neighbor_preservation_mean_rank_correlation']:.3f}")
        
        print("\n" + "-" * 80)
        print("6. DIRECTIONAL BIAS:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['method'].upper()}:")
            if 'direction_bias_directional_bias' in row:
                print(f"  Directional bias: {row['direction_bias_directional_bias']:.3f}")
            if 'direction_bias_circular_variance' in row:
                print(f"  Circular variance: {row['direction_bias_circular_variance']:.3f}")
            if 'direction_bias_is_biased' in row:
                print(f"  Is biased: {bool(row['direction_bias_is_biased'])}")
        
        print("\n" + "-" * 80)
        print("7. DENSITY CORRELATION:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['method'].upper()}:")
            if 'density_correlation_density_pearson_correlation' in row:
                print(f"  Pearson correlation: {row['density_correlation_density_pearson_correlation']:.3f}")
            if 'density_correlation_density_similarity' in row:
                print(f"  Density similarity: {row['density_correlation_density_similarity']:.3f}")
            if 'density_correlation_density_js_divergence' in row:
                print(f"  JS divergence: {row['density_correlation_density_js_divergence']:.3f}")
        
        print("\n" + "-" * 80)
        print("8. QUERY ACCURACY:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['method'].upper()}:")
            if 'query_accuracy_query_accuracy' in row:
                print(f"  Query accuracy: {row['query_accuracy_query_accuracy']:.3f}")
            if 'query_accuracy_mean_count_error' in row:
                print(f"  Mean count error: {row['query_accuracy_mean_count_error']:.1f}")
            if 'query_accuracy_mean_precision' in row:
                print(f"  Mean precision: {row['query_accuracy_mean_precision']:.3f}")
            if 'query_accuracy_mean_recall' in row:
                print(f"  Mean recall: {row['query_accuracy_mean_recall']:.3f}")
        
        # Save detailed results
        results_df.to_csv('anonymization_analysis_results.csv', index=False)
        print(f"\n" + "="*80)
        print(f"Detailed results saved to: anonymization_analysis_results.csv")
        print("="*80)
        
        # Create plots
        try:
            analyzer.create_comparison_plots(original_file, available_files)
            print("Comparison plots created successfully")
        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("No results to display")


if __name__ == "__main__":
    run_comprehensive_analysis()