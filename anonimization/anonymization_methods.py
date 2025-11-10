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
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import ConvexHull
from scipy.stats import gamma
import warnings
import math
import gc
from typing import Tuple, List, Optional, Union
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class GeographicalAnonymizer:
    """
    A class for applying various geographical anonymization methods to coordinate data.
    """
    
    def __init__(self, house_coords=None):
        """
        Initialize the anonymizer with house coordinates for snapping.
        
        Args:
            house_coords: Array of valid house coordinates (N x 2) in Swiss LV95 meters
        """
        # Set up house snapping - coordinates are already in meters (Swiss LV95)
        if house_coords is not None:
            self.house_coords = house_coords
            self.house_tree = KDTree(house_coords)
        else:
            self.house_coords = None
            self.house_tree = None
    
    def snap_to_nearest_house(self, point: np.ndarray) -> np.ndarray:
        """
        Snap a point to the nearest valid house coordinate.
        
        Args:
            point: Point in Swiss LV95 coordinates
            
        Returns:
            Nearest house coordinate in Swiss LV95
        """
        if self.house_tree is None:
            return point
        
        # Find nearest house
        _, idx = self.house_tree.query(point.reshape(1, -1), k=1)
        return self.house_coords[idx[0]]
    
    def batch_snap_to_houses(self, points: np.ndarray) -> np.ndarray:
        """
        Batch snap multiple points to nearest houses with memory management.
        
        Args:
            points: Points in Swiss LV95 coordinates (N x 2)
            
        Returns:
            Array of nearest house coordinates in Swiss LV95
        """
        if self.house_tree is None:
            return points
        
        # Process in batches to avoid memory issues for large datasets
        batch_size = 10000
        n_points = len(points)
        result = np.zeros_like(points)
        
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            batch_points = points[i:end_idx]
            
            distances, indices = self.house_tree.query(batch_points, k=1)
            result[i:end_idx] = self.house_coords[indices]
            
            # Clean up memory periodically
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        return result
    
    def validate_house_snapping(self, masked_x: np.ndarray, masked_y: np.ndarray, tolerance: float = 1.0) -> dict:
        """Validate that all masked coordinates are actually valid house locations."""
        if self.house_coords is None:
            return {"valid_houses": 0, "total": len(masked_x), "validation_rate": 0.0}
        
        masked_coords = np.column_stack([masked_x, masked_y])
        
        min_distances = []
        for masked_coord in masked_coords:
            distances = np.sqrt(np.sum((self.house_coords - masked_coord)**2, axis=1))
            min_distances.append(np.min(distances))
        
        min_distances = np.array(min_distances)
        valid_count = np.sum(min_distances <= tolerance)
        
        return {
            "valid_houses": valid_count,
            "total": len(masked_x),
            "validation_rate": valid_count / len(masked_x),
            "mean_min_distance": np.mean(min_distances),
            "max_min_distance": np.max(min_distances),
            "distances_under_1m": np.sum(min_distances < 1.0),
            "distances_under_10m": np.sum(min_distances < 10.0)
        }
    def is_valid_location(self, point: np.ndarray, bounds: Optional[Tuple] = None) -> bool:
        """Check if a point is within valid bounds."""
        if bounds is None:
            return True
        min_x, min_y, max_x, max_y = bounds
        return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y

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
            x_coords: Array of x coordinates in Swiss LV95
            y_coords: Array of y coordinates in Swiss LV95
            k_target: Target number of neighbors for density estimation
            r_min: Minimum radius for donut sampling (meters)
            beta: Multiplier for k-th neighbor distance
            r_global_max: Maximum allowed radius (meters)
            max_iter: Maximum sampling attempts
            bounds: Optional bounds for validity checking
            
        Returns:
            Tuple of (masked_x, masked_y) coordinates
        """
        print("Computing Density Aware Donut Masking Anonymization....")

        # Work directly with Swiss LV95 coordinates (already in meters)
        coords = np.column_stack([x_coords, y_coords])
        tree = KDTree(coords)
        
        if bounds is None:
            bounds = (coords[:, 0].min(), coords[:, 1].min(), 
                     coords[:, 0].max(), coords[:, 1].max())
        
        candidate_coords = np.zeros_like(coords)
        
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
                d = np.random.uniform(r_min, r_upper)
                theta = np.random.uniform(0, 2 * np.pi)
                
                new_point = np.array([
                    p[0] + d * np.cos(theta),
                    p[1] + d * np.sin(theta)
                ])
                
                if self.is_valid_location(new_point, bounds):
                    candidate_coords[i] = new_point
                    success = True
                    break
            
            if not success:
                # Fallback to centroid of k neighbors
                dists, idxs = tree.query(p.reshape(1, -1), k=k_target+1)
                neighbors = tree.data[idxs[0]]
                candidate_coords[i] = np.mean(neighbors, axis=0)
        
        # Snap all candidates to nearest houses
        masked_coords = self.batch_snap_to_houses(candidate_coords)
        
        return masked_coords[:, 0], masked_coords[:, 1]
    
    def spatial_k_anonymity(self,
                          x_coords: np.ndarray,
                          y_coords: np.ndarray,
                          k_target: int = 5,
                          strategy: str = "random_in_circle",
                          bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply spatial k-anonymity anonymization.
        
        Args:
            x_coords: Array of x coordinates in Swiss LV95
            y_coords: Array of y coordinates in Swiss LV95
            k_target: Target k for k-anonymity
            strategy: "random_in_circle", "centroid", or "region_id"
            bounds: Optional bounds for validity checking
            
        Returns:
            Tuple of (masked_x, masked_y) coordinates
        """
        print("Computing Spatial K Anonymization....")

        coords = np.column_stack([x_coords, y_coords])
        tree = KDTree(coords)
        
        if bounds is None:
            bounds = (coords[:, 0].min(), coords[:, 1].min(), 
                     coords[:, 0].max(), coords[:, 1].max())
        
        candidate_coords = np.zeros_like(coords)
        
        for i in range(len(coords)):
            p = coords[i]
            
            # Find k nearest neighbors (including self)
            dists, idxs = tree.query(p.reshape(1, -1), k=k_target+1)
            R_k = dists[0][k_target]  # radius to include k neighbors
            neighbors = coords[idxs[0]]
            
            if strategy == "centroid":
                candidate_coords[i] = np.mean(neighbors, axis=0)
            elif strategy == "random_in_circle":
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
                        candidate_coords[i] = candidate
                        success = True
                        break
                
                if not success:
                    candidate_coords[i] = np.mean(neighbors, axis=0)
            else:  # region_id
                candidate_coords[i] = np.mean(neighbors, axis=0)
        
        masked_coords = self.batch_snap_to_houses(candidate_coords)
        return masked_coords[:, 0], masked_coords[:, 1]
    
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
            x_coords: Array of x coordinates in Swiss LV95
            y_coords: Array of y coordinates in Swiss LV95
            epsilon: Privacy parameter (smaller = more privacy)
            max_resample: Maximum resampling attempts
            bounds: Optional bounds for validity checking
            
        Returns:
            Tuple of (masked_x, masked_y) coordinates
        """
        print("Computing Geo DP Anonymization....")
        coords = np.column_stack([x_coords, y_coords])
        
        if bounds is None:
            bounds = (coords[:, 0].min(), coords[:, 1].min(), 
                     coords[:, 0].max(), coords[:, 1].max())
        
        candidate_coords = np.zeros_like(coords)
        
        for i in range(len(coords)):
            p = coords[i]
            
            success = False
            for attempt in range(max_resample):
                dx, dy = self.sample_planar_laplace(epsilon)
                candidate = np.array([p[0] + dx, p[1] + dy])
                
                if self.is_valid_location(candidate, bounds):
                    candidate_coords[i] = candidate
                    success = True
                    break
            
            if not success:
                candidate_coords[i] = candidate if 'candidate' in locals() else p
        
        masked_coords = self.batch_snap_to_houses(candidate_coords)
        return masked_coords[:, 0], masked_coords[:, 1]
    
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
                            k_target: int = 50,
                            bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply memory-efficient adaptive Voronoi mask anonymization.
        
        Args:
            x_coords: Array of x coordinates in Swiss LV95
            y_coords: Array of y coordinates in Swiss LV95
            k_target: Target number of points per cluster (larger for memory efficiency)
            bounds: Optional bounds for validity checking
            
        Returns:
            Tuple of (masked_x, masked_y) coordinates
        """
        print("Computing Memory-Efficient Adaptive Voronoi Anonymization....")
        coords = np.column_stack([x_coords, y_coords])
        n_points = len(coords)
        
        print(f"Dataset size: {n_points:,} points")
        
        # Calculate number of clusters - use larger k_target for memory efficiency
        n_regions = max(1, int(np.ceil(n_points / k_target)))
        print(f"Creating {n_regions:,} clusters (target size: {k_target})")
        
        # Use MiniBatchKMeans for very large datasets - much more memory efficient
        if n_points > 100000:
            print("Using MiniBatchKMeans for memory efficiency...")
            # Batch size should be much smaller than dataset size
            batch_size = min(10000, n_points // 20)
            kmeans = MiniBatchKMeans(
                n_clusters=n_regions, 
                random_state=42,
                batch_size=batch_size,
                max_iter=100,
                n_init=3,
                verbose=1
            )
        else:
            print("Using regular KMeans...")
            kmeans = KMeans(
                n_clusters=n_regions, 
                random_state=42, 
                n_init=3, 
                max_iter=50
            )
        
        print("Performing clustering...")
        labels = kmeans.fit_predict(coords)
        print(f"Clustering complete.")
        
        # Clear kmeans object to free memory
        del kmeans
        gc.collect()
        
        if bounds is None:
            bounds = (coords[:, 0].min(), coords[:, 1].min(), 
                     coords[:, 0].max(), coords[:, 1].max())
        
        # Process clusters in batches to avoid memory issues
        candidate_coords = self._process_clusters_in_batches(coords, labels, n_regions)
        
        print("Cluster processing complete.")
        
        # Batch snap to houses with progress
        print("Snapping to nearest houses...")
        masked_coords = self.batch_snap_to_houses(candidate_coords)
        
        return masked_coords[:, 0], masked_coords[:, 1]

    def _process_clusters_in_batches(self, coords, labels, n_regions, batch_size=1000):
        """Process clusters in batches to manage memory usage."""
        candidate_coords = np.zeros_like(coords)
        
        # Process clusters in batches
        cluster_batch_size = batch_size
        
        for batch_start in range(0, n_regions, cluster_batch_size):
            batch_end = min(batch_start + cluster_batch_size, n_regions)
            
            print(f"Processing cluster batch {batch_start//cluster_batch_size + 1}/{(n_regions-1)//cluster_batch_size + 1}")
            print(f"  Clusters {batch_start} to {batch_end-1}")
            
            for r in tqdm(range(batch_start, batch_end), 
                         desc=f"Batch {batch_start//cluster_batch_size + 1}", 
                         unit="cluster"):
                cluster_mask = labels == r
                if not cluster_mask.any():
                    continue
                    
                cluster_points = coords[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_points) < 3:
                    # Small cluster: use centroid
                    centroid = np.mean(cluster_points, axis=0)
                    candidate_coords[cluster_indices] = centroid
                else:
                    # Large cluster: use convex hull
                    try:
                        hull = ConvexHull(cluster_points)
                        
                        # Process points in this cluster in sub-batches for very large clusters
                        point_batch_size = 5000
                        for i in range(0, len(cluster_indices), point_batch_size):
                            end_i = min(i + point_batch_size, len(cluster_indices))
                            batch_indices = cluster_indices[i:end_i]
                            
                            for idx in batch_indices:
                                candidate = self.sample_uniform_in_polygon(hull)
                                candidate_coords[idx] = candidate if candidate is not None else np.mean(cluster_points, axis=0)
                                
                    except Exception as e:
                        # Fallback to centroid if convex hull fails
                        print(f"  Convex hull failed for cluster {r}: {e}")
                        centroid = np.mean(cluster_points, axis=0)
                        candidate_coords[cluster_indices] = centroid
            
            # Force garbage collection after each batch
            gc.collect()
        
        return candidate_coords


def apply_anonymization_methods(input_file: str, output_prefix: str = "anonymized"):
    """
    Apply all four anonymization methods to a CSV file with geographical data.
    
    Args:
        input_file: Path to input CSV file
        output_prefix: Prefix for output files
    """
    print("====== Started ======")
    try:
        print("Loading in the data")
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("===== Loaded in all data =====")

    # Extract coordinates
    x_coords = df['home_x'].values
    y_coords = df['home_y'].values
    
    # Remove any NaN values
    valid_mask = ~(pd.isna(x_coords) | pd.isna(y_coords))
    df_clean = df[valid_mask].copy()
    x_coords = df_clean['home_x'].values
    y_coords = df_clean['home_y'].values
    
    # Create array of all house coordinates for snapping
    house_coords = np.column_stack([x_coords, y_coords])

    print("===== Starting to create KD tree =====")
    # Create KDTree for original coordinates (for neighbor counting)
    original_tree = KDTree(house_coords)

    print("===== Initializing geographical anonymizer =====")
    # Initialize anonymizer with house snapping (Swiss LV95 coordinates already in meters)
    anonymizer = GeographicalAnonymizer(house_coords=house_coords)
    
    # Apply methods
    methods = {
        # 'donut_geomask': {
        #     'func': anonymizer.density_aware_donut_geomask,
        #     'params': {'k_target': 5, 'r_min': 100.0, 'beta': 1.5, 'r_global_max': 2000.0}
        # },
        # 'k_anonymity': {
        #     'func': anonymizer.spatial_k_anonymity,
        #     'params': {'k_target': 5, 'strategy': 'random_in_circle'}
        # },
        # 'differential_privacy': {
        #     'func': anonymizer.geo_dp_mask,
        #     'params': {'epsilon': 0.1}
        # },
        'voronoi_mask': {
            'func': anonymizer.adaptive_voronoi_mask,
            'params': {'k_target': 50}
        }
    }

    print("created the original tree")
    
    for method_name, method_info in methods.items():
        try:
            masked_x, masked_y = method_info['func'](x_coords, y_coords, **method_info['params'])
            
            # Calculate distance from original to anonymized location (in meters)
            distances = np.sqrt((x_coords - masked_x)**2 + (y_coords - masked_y)**2)
            
            # For each original house, count how many original houses are within its anonymization distance
            neighbors_within_displacement = []
            
            for i in range(len(x_coords)):
                original_point = [x_coords[i], y_coords[i]]
                displacement_distance = distances[i]
                
                # Count original houses within the displacement distance of this original house
                neighbor_indices = original_tree.query_ball_point(original_point, displacement_distance)
                # Subtract 1 to exclude the house itself
                neighbor_count = len(neighbor_indices) - 1
                neighbors_within_displacement.append(neighbor_count)
            
            df_output = df_clean.copy()
            df_output['home_x'] = masked_x
            df_output['home_y'] = masked_y
            df_output['anonymization_distance_m'] = distances
            df_output['original_neighbors_within_displacement'] = neighbors_within_displacement
            
            output_file = f"/cluster/home/chaoch/new_{output_prefix}_{method_name}.csv"
            df_output.to_csv(output_file, index=False)
            
            # Print summary statistics
            print(f"\n{method_name.upper()} Results:")
            print(f"  Mean displacement: {np.mean(distances):.1f}m")
            print(f"  Median displacement: {np.median(distances):.1f}m")
            print(f"  Mean neighbors within displacement: {np.mean(neighbors_within_displacement):.1f}")
            print(f"  Median neighbors within displacement: {np.median(neighbors_within_displacement):.1f}")
            print(f"  Houses with 0 neighbors within displacement: {np.sum(np.array(neighbors_within_displacement) == 0)}")
            print(f"  Houses with 5+ neighbors within displacement: {np.sum(np.array(neighbors_within_displacement) >= 5)}")
            
        except Exception as e:
            print(f"Error applying {method_name}: {e}")
            import traceback
            traceback.print_exc()


def check_anonymization_displacement(original_coords: np.ndarray, anonymized_coords: np.ndarray, method_name: str):
    """Check if anonymized coordinates actually moved to different locations."""
    same_location = np.all(np.abs(original_coords - anonymized_coords) < 1e-6, axis=1)
    displacements = np.sqrt(np.sum((original_coords - anonymized_coords)**2, axis=1))
    
    moved_to_existing = 0
    for i in range(len(anonymized_coords)):
        if not same_location[i]:
            matches = np.all(np.abs(original_coords - anonymized_coords[i]) < 0.1, axis=1)
            if np.any(matches):
                moved_to_existing += 1
    
    return {
        'method': method_name,
        'n_same_location': np.sum(same_location),
        'n_moved': len(original_coords) - np.sum(same_location),
        'n_moved_to_existing': moved_to_existing,
        'mean_displacement': np.mean(displacements),
        'median_displacement': np.median(displacements),
        'min_displacement': np.min(displacements),
        'max_displacement': np.max(displacements)
    }


if __name__ == "__main__":
    input_file = "statpop_original_zurich.csv"
    apply_anonymization_methods(input_file, "zurich_full")
