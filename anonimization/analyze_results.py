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
                metrics = self.calculate_privacy_metrics(orig_x, orig_y, masked_x, masked_y)
                metrics['method'] = method_name
                results.append(metrics)
                
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
                    
                    sns.histplot(distances, bins=50, alpha=0.7, kde=True, 
                               color=colors[i], ax=axes[i])
                    axes[i].set_title(f'{method_name.replace("_", " ").title()}\nMean: {np.mean(distances):.1f}m', 
                                    fontsize=12, fontweight='bold')
                    axes[i].set_xlabel('Displacement (meters)')
                    axes[i].set_ylabel('Frequency')
                
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
        print("ANONYMIZATION METHODS COMPARISON")
        print("="*80)
        
        print("\nDisplacement Metrics:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['method'].upper()}:")
            print(f"  Mean displacement: {row['mean_displacement_m']:.1f} m")
            print(f"  Median displacement: {row['median_displacement_m']:.1f} m")
            print(f"  Max displacement: {row['max_displacement_m']:.1f} m")
            print(f"  Std displacement: {row['std_displacement_m']:.1f} m")
        
        print("\n" + "-" * 80)
        print("K-Anonymity Preservation Rates:")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"\n{row['method'].upper()}:")
            for k in [3, 5, 10]:
                col = f'k{k}_anonymity_preservation'
                if col in row:
                    print(f"  k={k}: {row[col]:.3f}")
        
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