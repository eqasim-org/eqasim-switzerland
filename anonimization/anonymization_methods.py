"""
Implementation of four anonymization methods for geographical data.
Methods implemented:
1. Density-aware donut geomask
2. Spatial k-anonymity
3. Differential privacy with planar Laplace noise
4. Adaptive Voronoi mask
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.stats import gamma
import pyproj
import warnings
import math
from typing import Tuple, List, Optional, Union

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class GeographicalAnonymizer:
    """
    A class for applying various geographical anonymization methods to coordinate data.
    """
    
    def __init__(self, crs_from="EPSG:2056", crs_to="EPSG:3857"):
        """
        Initialize the anonymizer with coordinate reference systems.
        
        Args:
            crs_from: Source CRS (default: Swiss LV95 - EPSG:2056)
            crs_to: Target metric CRS for calculations (default: Web Mercator - EPSG:3857)
        """
        self.crs_from = crs_from
        self.crs_to = crs_to
        self.transformer_to_meters = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
        self.transformer_from_meters = pyproj.Transformer.from_crs(crs_to, crs_from, always_xy=True)
    
    def project_to_meters(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """
        Project coordinates to metric CRS for distance calculations.
        
        Args:
            x_coords: Array of x coordinates
            y_coords: Array of y coordinates
            
        Returns:
            Array of projected coordinates in meters
        """
        x_proj, y_proj = self.transformer_to_meters.transform(x_coords, y_coords)
        return np.column_stack([x_proj, y_proj])
    
    def inverse_project(self, coords_meters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse project coordinates back to original CRS.
        
        Args:
            coords_meters: Array of coordinates in meters
            
        Returns:
            Tuple of (x_coords, y_coords) in original CRS
        """
        x_orig, y_orig = self.transformer_from_meters.transform(
            coords_meters[:, 0], coords_meters[:, 1]
        )
        return x_orig, y_orig
    
    def is_valid_location(self, point: np.ndarray, bounds: Optional[Tuple] = None) -> bool:
        """
        Check if a point is within valid bounds (simplified version).
        
        Args:
            point: [x, y] coordinate
            bounds: Optional bounds as (min_x, min_y, max_x, max_y)
            
        Returns:
            True if point is valid
        """
        if bounds is None:
            return True
        min_x, min_y, max_x, max_y = bounds
        return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y
    
    def fallback_location(self, original_point: np.ndarray, tree: KDTree, k_target: int) -> np.ndarray:
        """
        Generate a fallback location when sampling fails.
        
        Args:
            original_point: Original point coordinates
            tree: KDTree for neighbor queries
            k_target: Number of neighbors to consider
            
        Returns:
            Fallback coordinate
        """
        # Use centroid of k nearest neighbors as fallback
        dists, idxs = tree.query(original_point.reshape(1, -1), k=k_target+1)
        neighbors = tree.data[idxs[0]]
        return np.mean(neighbors, axis=0)
    
    def density_aware_donut_geomask(self, 
                                  x_coords: np.ndarray, 
                                  y_coords: np.ndarray,
                                  k_target: int = 5,
                                  r_min: float = 100.0,
                                  beta: float = 1.5,
                                  r_global_max: float = 2000.0,
                                  max_iter: int = 50,
                                  bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply density-aware donut geomask anonymization.
        
        Args:
            x_coords: Array of x coordinates
            y_coords: Array of y coordinates
            k_target: Target number of neighbors for density estimation
            r_min: Minimum radius for donut sampling (meters)
            beta: Multiplier for k-th neighbor distance
            r_global_max: Maximum allowed radius (meters)
            max_iter: Maximum sampling attempts
            bounds: Optional bounds for validity checking
            
        Returns:
            Tuple of (masked_x, masked_y) coordinates
        """
        print(f"Applying density-aware donut geomask with k={k_target}, r_min={r_min}m, beta={beta}")
        
        # Project to metric coordinates
        coords = self.project_to_meters(x_coords, y_coords)
        tree = KDTree(coords)
        
        if bounds is None:
            bounds = (coords[:, 0].min(), coords[:, 1].min(), 
                     coords[:, 0].max(), coords[:, 1].max())
        
        masked_coords = np.zeros_like(coords)
        
        for i in range(len(coords)):
            p = coords[i]
            
            # Find distance to k-th nearest neighbor
            dists, idxs = tree.query(p.reshape(1, -1), k=k_target+1)
            R_k = dists[0][k_target]  # distance to k-th neighbor (excluding self)
            
            # Compute upper radius
            r_upper = min(max(beta * R_k, r_min), r_global_max)
            
            # Sample in donut
            success = False
            for attempt in range(max_iter):
                # Sample distance uniformly in [r_min, r_upper]
                d = np.random.uniform(r_min, r_upper)
                # Sample angle uniformly
                theta = np.random.uniform(0, 2 * np.pi)
                
                new_point = np.array([
                    p[0] + d * np.cos(theta),
                    p[1] + d * np.sin(theta)
                ])
                
                if self.is_valid_location(new_point, bounds):
                    masked_coords[i] = new_point
                    success = True
                    break
            
            if not success:
                # Fallback to centroid of k neighbors
                masked_coords[i] = self.fallback_location(p, tree, k_target)
        
        # Inverse project back to original CRS
        return self.inverse_project(masked_coords)
    
    def spatial_k_anonymity(self,
                          x_coords: np.ndarray,
                          y_coords: np.ndarray,
                          k_target: int = 5,
                          strategy: str = "random_in_circle",
                          bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply spatial k-anonymity anonymization.
        
        Args:
            x_coords: Array of x coordinates
            y_coords: Array of y coordinates
            k_target: Target k for k-anonymity
            strategy: "random_in_circle", "centroid", or "region_id"
            bounds: Optional bounds for validity checking
            
        Returns:
            Tuple of (masked_x, masked_y) coordinates
        """
        print(f"Applying spatial k-anonymity with k={k_target}, strategy={strategy}")
        
        coords = self.project_to_meters(x_coords, y_coords)
        tree = KDTree(coords)
        
        if bounds is None:
            bounds = (coords[:, 0].min(), coords[:, 1].min(), 
                     coords[:, 0].max(), coords[:, 1].max())
        
        masked_coords = np.zeros_like(coords)
        
        for i in range(len(coords)):
            p = coords[i]
            
            # Find k nearest neighbors (including self)
            dists, idxs = tree.query(p.reshape(1, -1), k=k_target+1)
            R_k = dists[0][k_target]  # radius to include k neighbors
            neighbors = coords[idxs[0]]
            
            if strategy == "centroid":
                # Use centroid of k neighbors
                masked_coords[i] = np.mean(neighbors, axis=0)
            
            elif strategy == "random_in_circle":
                # Sample uniformly inside circle with radius R_k
                success = False
                for attempt in range(50):
                    u = np.random.uniform(0, 1)
                    r = np.sqrt(u) * R_k
                    theta = np.random.uniform(0, 2 * np.pi)
                    
                    candidate = np.array([
                        p[0] + r * np.cos(theta),
                        p[1] + r * np.sin(theta)
                    ])
                    
                    if self.is_valid_location(candidate, bounds):
                        masked_coords[i] = candidate
                        success = True
                        break
                
                if not success:
                    masked_coords[i] = np.mean(neighbors, axis=0)
            
            elif strategy == "region_id":
                # For region_id, we'll use the centroid as a simplified implementation
                masked_coords[i] = np.mean(neighbors, axis=0)
        
        return self.inverse_project(masked_coords)
    
    def sample_planar_laplace(self, epsilon: float) -> Tuple[float, float]:
        """
        Sample from planar Laplace distribution.
        
        Args:
            epsilon: Privacy parameter
            
        Returns:
            Tuple of (dx, dy) displacement
        """
        theta = np.random.uniform(0, 2 * np.pi)
        # Radius follows Gamma(k=2, scale=1/epsilon)
        r = np.random.gamma(shape=2, scale=1/epsilon)
        return r * np.cos(theta), r * np.sin(theta)
    
    def geo_dp_mask(self,
                   x_coords: np.ndarray,
                   y_coords: np.ndarray,
                   epsilon: float = 0.1,
                   max_resample: int = 10,
                   bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply differential privacy with planar Laplace noise.
        
        Args:
            x_coords: Array of x coordinates
            y_coords: Array of y coordinates
            epsilon: Privacy parameter (smaller = more privacy)
            max_resample: Maximum resampling attempts
            bounds: Optional bounds for validity checking
            
        Returns:
            Tuple of (masked_x, masked_y) coordinates
        """
        print(f"Applying differential privacy with epsilon={epsilon}")
        
        coords = self.project_to_meters(x_coords, y_coords)
        
        if bounds is None:
            bounds = (coords[:, 0].min(), coords[:, 1].min(), 
                     coords[:, 0].max(), coords[:, 1].max())
        
        masked_coords = np.zeros_like(coords)
        
        for i in range(len(coords)):
            p = coords[i]
            
            success = False
            for attempt in range(max_resample):
                dx, dy = self.sample_planar_laplace(epsilon)
                candidate = np.array([p[0] + dx, p[1] + dy])
                
                if self.is_valid_location(candidate, bounds):
                    masked_coords[i] = candidate
                    success = True
                    break
            
            if not success:
                # If all resampling fails, use the last candidate or original point
                masked_coords[i] = candidate if 'candidate' in locals() else p
        
        return self.inverse_project(masked_coords)
    
    def sample_uniform_in_polygon(self, hull: ConvexHull, max_attempts: int = 100) -> Optional[np.ndarray]:
        """
        Sample a point uniformly inside a convex polygon.
        
        Args:
            hull: ConvexHull object
            max_attempts: Maximum sampling attempts
            
        Returns:
            Random point inside polygon or None if failed
        """
        if len(hull.vertices) < 3:
            return None
        
        points = hull.points[hull.vertices]
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        
        for _ in range(max_attempts):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = np.array([x, y])
            
            # Simple point-in-polygon test using cross products
            inside = True
            n_vertices = len(hull.vertices)
            for i in range(n_vertices):
                j = (i + 1) % n_vertices
                v1 = points[i]
                v2 = points[j]
                
                # Cross product to determine side
                cross = (v2[0] - v1[0]) * (point[1] - v1[1]) - (v2[1] - v1[1]) * (point[0] - v1[0])
                if cross < 0:  # Point is on wrong side
                    inside = False
                    break
            
            if inside:
                return point
        
        return None
    
    def adaptive_voronoi_mask(self,
                            x_coords: np.ndarray,
                            y_coords: np.ndarray,
                            k_target: int = 5,
                            bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply adaptive Voronoi mask anonymization.
        
        Args:
            x_coords: Array of x coordinates
            y_coords: Array of y coordinates
            k_target: Target number of points per cluster
            bounds: Optional bounds for validity checking
            
        Returns:
            Tuple of (masked_x, masked_y) coordinates
        """
        print(f"Applying adaptive Voronoi mask with k_target={k_target}")
        
        coords = self.project_to_meters(x_coords, y_coords)
        n_regions = max(1, int(np.ceil(len(coords) / k_target)))
        
        # Cluster points into regions
        kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        if bounds is None:
            bounds = (coords[:, 0].min(), coords[:, 1].min(), 
                     coords[:, 0].max(), coords[:, 1].max())
        
        # Compute polygons for each cluster
        polygons = {}
        centroids = {}
        
        for r in range(n_regions):
            pts_r = coords[labels == r]
            centroids[r] = np.mean(pts_r, axis=0)
            
            if len(pts_r) < 3:
                polygons[r] = None
            else:
                try:
                    hull = ConvexHull(pts_r)
                    polygons[r] = hull
                except:
                    polygons[r] = None
        
        # Sample points for each cluster
        masked_coords = np.zeros_like(coords)
        
        for i in range(len(coords)):
            r = labels[i]
            poly = polygons[r]
            
            if poly is None:
                # Use centroid if no valid polygon
                masked_coords[i] = centroids[r]
            else:
                # Try to sample inside polygon
                candidate = self.sample_uniform_in_polygon(poly)
                if candidate is None:
                    # Fallback to centroid
                    masked_coords[i] = centroids[r]
                else:
                    masked_coords[i] = candidate
        
        return self.inverse_project(masked_coords)


def apply_anonymization_methods(input_file: str, output_prefix: str = "anonymized"):
    """
    Apply all four anonymization methods to a CSV file with geographical data.
    
    Args:
        input_file: Path to input CSV file
        output_prefix: Prefix for output files
    """
    print(f"Loading data from {input_file}")
    
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract coordinates
    x_coords = df['home_x'].values
    y_coords = df['home_y'].values
    
    # Remove any NaN values
    valid_mask = ~(pd.isna(x_coords) | pd.isna(y_coords))
    df_clean = df[valid_mask].copy()
    x_coords = df_clean['home_x'].values
    y_coords = df_clean['home_y'].values
    
    print(f"Processing {len(df_clean)} valid coordinate pairs")
    
    # Initialize anonymizer (assuming Swiss LV95 coordinates)
    anonymizer = GeographicalAnonymizer(crs_from="EPSG:2056", crs_to="EPSG:3857")
    
    # Apply methods
    methods = {
        'donut_geomask': {
            'func': anonymizer.density_aware_donut_geomask,
            'params': {'k_target': 5, 'r_min': 100.0, 'beta': 1.5, 'r_global_max': 2000.0}
        },
        'k_anonymity': {
            'func': anonymizer.spatial_k_anonymity,
            'params': {'k_target': 5, 'strategy': 'random_in_circle'}
        },
        'differential_privacy': {
            'func': anonymizer.geo_dp_mask,
            'params': {'epsilon': 0.1}
        },
        'voronoi_mask': {
            'func': anonymizer.adaptive_voronoi_mask,
            'params': {'k_target': 5}
        }
    }
    
    for method_name, method_info in methods.items():
        print(f"\n--- Applying {method_name} ---")
        
        try:
            # Apply anonymization
            masked_x, masked_y = method_info['func'](x_coords, y_coords, **method_info['params'])
            
            # Create output dataframe
            df_output = df_clean.copy()
            df_output['home_x'] = masked_x
            df_output['home_y'] = masked_y
            
            # Save to file
            output_file = f"{output_prefix}_{method_name}.csv"
            df_output.to_csv(output_file, index=False)
            print(f"Saved {method_name} results to {output_file}")
            
            # Calculate basic statistics
            orig_coords = anonymizer.project_to_meters(x_coords, y_coords)
            masked_coords = anonymizer.project_to_meters(masked_x, masked_y)
            distances = np.sqrt(np.sum((orig_coords - masked_coords)**2, axis=1))
            
            print(f"Distance statistics (meters):")
            print(f"  Mean: {np.mean(distances):.1f}")
            print(f"  Median: {np.median(distances):.1f}")
            print(f"  Min: {np.min(distances):.1f}")
            print(f"  Max: {np.max(distances):.1f}")
            print(f"  Std: {np.std(distances):.1f}")
            
        except Exception as e:
            print(f"Error applying {method_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Use the fixed sample file
    input_file = "statpop_sample_10k.csv"
    
    print("Applying all anonymization methods to fixed 10k sample...")
    apply_anonymization_methods(input_file, "anonymized_10k")
    
    # Uncomment to process full dataset
    # print("\nProcessing full dataset...")
    # apply_anonymization_methods(input_file, "full_anonymized")