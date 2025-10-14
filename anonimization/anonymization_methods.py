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
import warnings
import math
from typing import Tuple, List, Optional, Union

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
        Batch snap multiple points to nearest houses for better performance.
        
        Args:
            points: Points in Swiss LV95 coordinates (N x 2)
            
        Returns:
            Array of nearest house coordinates in Swiss LV95
        """
        if self.house_tree is None:
            return points
        
        distances, indices = self.house_tree.query(points, k=1)
        return self.house_coords[indices]
    
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
                            k_target: int = 5,
                            bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply adaptive Voronoi mask anonymization.
        
        Args:
            x_coords: Array of x coordinates in Swiss LV95
            y_coords: Array of y coordinates in Swiss LV95
            k_target: Target number of points per cluster
            bounds: Optional bounds for validity checking
            
        Returns:
            Tuple of (masked_x, masked_y) coordinates
        """
        print("Computing Adaptive Voronoi Anonymization....")
        coords = np.column_stack([x_coords, y_coords])
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
        
        candidate_coords = np.zeros_like(coords)
        
        for i in range(len(coords)):
            r = labels[i]
            poly = polygons[r]
            
            if poly is None:
                candidate_coords[i] = centroids[r]
            else:
                candidate = self.sample_uniform_in_polygon(poly)
                if candidate is None:
                    candidate_coords[i] = centroids[r]
                else:
                    candidate_coords[i] = candidate
        
        masked_coords = self.batch_snap_to_houses(candidate_coords)
        return masked_coords[:, 0], masked_coords[:, 1]


def apply_anonymization_methods(input_file: str, output_prefix: str = "anonymized"):
    """
    Apply all four anonymization methods to a CSV file with geographical data.
    
    Args:
        input_file: Path to input CSV file
        output_prefix: Prefix for output files
    """
    try:
        df = pd.read_csv(input_file)
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
    
    # Create array of all house coordinates for snapping
    house_coords = np.column_stack([x_coords, y_coords])
    
    # Create KDTree for original coordinates (for neighbor counting)
    original_tree = KDTree(house_coords)
    
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
            'params': {'k_target': 5}
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
            
            output_file = f"{output_prefix}_{method_name}.csv"
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
    apply_anonymization_methods(input_file, "full")
