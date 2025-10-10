"""
Enhanced metrics for evaluating anonymization quality.
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree, distance_matrix
from scipy.stats import entropy, ks_2samp, spearmanr
from sklearn.metrics import pairwise_distances
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class EnhancedMetrics:
    """Calculate enhanced privacy and utility metrics for anonymization evaluation."""
    
    @staticmethod
    def calculate_reidentification_risk(original_coords: np.ndarray, 
                                       masked_coords: np.ndarray,
                                       threshold_meters: float = 50.0) -> Dict[str, float]:
        """
        Calculate re-identification risk based on how many points can be uniquely matched.
        
        Args:
            original_coords: Original coordinates (N, 2)
            masked_coords: Masked coordinates (N, 2)
            threshold_meters: Distance threshold for considering a match
            
        Returns:
            Dictionary with re-identification metrics
        """
        n_points = len(original_coords)
        
        # Build KD-tree for masked coordinates
        tree = KDTree(masked_coords)
        
        # For each original point, find closest masked point
        distances, indices = tree.query(original_coords, k=1)
        
        # Count correct matches (within threshold)
        correct_matches = np.sum(distances <= threshold_meters)
        
        # Calculate uniqueness - how many masked points have only one original point nearby
        unique_masked = 0
        for i in range(len(masked_coords)):
            nearby = np.sum(np.linalg.norm(original_coords - masked_coords[i], axis=1) <= threshold_meters)
            if nearby == 1:
                unique_masked += 1
        
        return {
            'reidentification_rate': correct_matches / n_points,
            'unique_points_ratio': unique_masked / n_points,
            'avg_confusion_distance': np.mean(distances),
            'median_confusion_distance': np.median(distances)
        }
    
    @staticmethod
    def calculate_location_entropy(coords: np.ndarray, 
                                  grid_size: float = 100.0) -> float:
        """
        Calculate Shannon entropy of spatial distribution using grid cells.
        Higher entropy = more uniform distribution.
        
        Args:
            coords: Coordinates (N, 2)
            grid_size: Size of grid cells in meters
            
        Returns:
            Shannon entropy value
        """
        # Create grid
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        
        x_bins = int((max_x - min_x) / grid_size) + 1
        y_bins = int((max_y - min_y) / grid_size) + 1
        
        # Count points in each cell
        x_idx = ((coords[:, 0] - min_x) / grid_size).astype(int)
        y_idx = ((coords[:, 1] - min_y) / grid_size).astype(int)
        
        # Create 2D histogram
        hist = np.zeros((x_bins, y_bins))
        for xi, yi in zip(x_idx, y_idx):
            hist[xi, yi] += 1
        
        # Calculate entropy
        hist_flat = hist.flatten()
        hist_flat = hist_flat[hist_flat > 0]  # Remove empty cells
        probabilities = hist_flat / hist_flat.sum()
        
        return entropy(probabilities)
    
    @staticmethod
    def calculate_ripleys_k(coords: np.ndarray, 
                           radii: List[float] = [100, 250, 500, 1000]) -> np.ndarray:
        """
        Calculate Ripley's K function for point pattern analysis.
        
        Args:
            coords: Coordinates (N, 2)
            radii: List of radii to evaluate
            
        Returns:
            Array of K values for each radius
        """
        n = len(coords)
        
        # Calculate area
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
        
        # Build distance matrix (use sample for large datasets)
        if n > 1000:
            sample_idx = np.random.choice(n, 1000, replace=False)
            coords_sample = coords[sample_idx]
            dist_matrix = distance_matrix(coords_sample, coords_sample)
            n_sample = len(coords_sample)
        else:
            dist_matrix = distance_matrix(coords, coords)
            n_sample = n
        
        k_values = []
        for r in radii:
            # Count pairs within distance r
            count = np.sum(dist_matrix <= r) - n_sample  # Subtract diagonal
            # Ripley's K
            k = (area * count) / (n_sample * (n_sample - 1))
            k_values.append(k)
        
        return np.array(k_values)
    
    @staticmethod
    def calculate_nearest_neighbor_preservation(original_coords: np.ndarray,
                                               masked_coords: np.ndarray,
                                               k: int = 5) -> Dict[str, float]:
        """
        Calculate how well nearest neighbor relationships are preserved.
        
        Args:
            original_coords: Original coordinates (N, 2)
            masked_coords: Masked coordinates (N, 2)
            k: Number of neighbors to consider
            
        Returns:
            Dictionary with preservation metrics
        """
        n_points = len(original_coords)
        
        # Build KD-trees
        orig_tree = KDTree(original_coords)
        mask_tree = KDTree(masked_coords)
        
        # Sample for efficiency
        sample_size = min(1000, n_points)
        sample_idx = np.random.choice(n_points, sample_size, replace=False)
        
        # Find k nearest neighbors in original space
        _, orig_neighbors = orig_tree.query(original_coords[sample_idx], k=k+1)  # +1 for self
        orig_neighbors = orig_neighbors[:, 1:]  # Remove self
        
        # Find k nearest neighbors in masked space
        _, mask_neighbors = mask_tree.query(masked_coords[sample_idx], k=k+1)
        mask_neighbors = mask_neighbors[:, 1:]
        
        # Calculate preservation metrics
        preservation_scores = []
        rank_correlations = []
        
        for i in range(sample_size):
            orig_set = set(orig_neighbors[i])
            mask_set = set(mask_neighbors[i])
            
            # Jaccard similarity
            intersection = len(orig_set & mask_set)
            union = len(orig_set | mask_set)
            preservation_scores.append(intersection / union if union > 0 else 0)
            
            # Rank correlation (Spearman-like)
            common = orig_set & mask_set
            if len(common) > 1:
                orig_ranks = {v: idx for idx, v in enumerate(orig_neighbors[i])}
                mask_ranks = {v: idx for idx, v in enumerate(mask_neighbors[i])}
                rank_diff = sum(abs(orig_ranks[v] - mask_ranks[v]) for v in common)
                rank_correlations.append(1 - (rank_diff / (k * len(common))))
            else:
                rank_correlations.append(0)
        
        return {
            'mean_neighbor_preservation': np.mean(preservation_scores),
            'median_neighbor_preservation': np.median(preservation_scores),
            'mean_rank_correlation': np.mean(rank_correlations),
            'neighbor_preservation_std': np.std(preservation_scores)
        }
    
    @staticmethod
    def calculate_displacement_direction_bias(original_coords: np.ndarray,
                                             masked_coords: np.ndarray) -> Dict[str, float]:
        """
        Calculate if there's systematic directional bias in displacement.
        
        Args:
            original_coords: Original coordinates (N, 2)
            masked_coords: Masked coordinates (N, 2)
            
        Returns:
            Dictionary with directional bias metrics
        """
        # Calculate displacement vectors
        displacements = masked_coords - original_coords
        
        # Calculate angles
        angles = np.arctan2(displacements[:, 1], displacements[:, 0])
        
        # Calculate mean resultant length (0 = uniform, 1 = all same direction)
        mean_x = np.mean(np.cos(angles))
        mean_y = np.mean(np.sin(angles))
        mean_resultant_length = np.sqrt(mean_x**2 + mean_y**2)
        
        # Calculate mean direction
        mean_direction = np.arctan2(mean_y, mean_x)
        mean_direction_degrees = np.degrees(mean_direction)
        
        # Calculate circular variance
        circular_variance = 1 - mean_resultant_length
        
        return {
            'directional_bias': mean_resultant_length,
            'mean_direction_degrees': mean_direction_degrees,
            'circular_variance': circular_variance,
            'is_biased': float(mean_resultant_length > 0.3)  # Threshold for bias
        }
    
    @staticmethod
    def calculate_density_correlation(original_coords: np.ndarray,
                                    masked_coords: np.ndarray,
                                    grid_size: float = 200.0) -> Dict[str, float]:
        """
        Calculate correlation between original and masked density distributions.
        
        Args:
            original_coords: Original coordinates (N, 2)
            masked_coords: Masked coordinates (N, 2)
            grid_size: Size of grid cells for density calculation
            
        Returns:
            Dictionary with density correlation metrics
        """
        # Determine common bounds
        all_coords = np.vstack([original_coords, masked_coords])
        min_x, min_y = all_coords.min(axis=0)
        max_x, max_y = all_coords.max(axis=0)
        
        x_bins = int((max_x - min_x) / grid_size) + 1
        y_bins = int((max_y - min_y) / grid_size) + 1
        
        # Create density grids
        def create_density_grid(coords):
            x_idx = ((coords[:, 0] - min_x) / grid_size).astype(int)
            y_idx = ((coords[:, 1] - min_y) / grid_size).astype(int)
            hist = np.zeros((x_bins, y_bins))
            for xi, yi in zip(x_idx, y_idx):
                if 0 <= xi < x_bins and 0 <= yi < y_bins:
                    hist[xi, yi] += 1
            return hist
        
        orig_density = create_density_grid(original_coords)
        mask_density = create_density_grid(masked_coords)
        
        # Flatten and calculate correlation
        orig_flat = orig_density.flatten()
        mask_flat = mask_density.flatten()
        
        # Pearson correlation
        pearson_corr = np.corrcoef(orig_flat, mask_flat)[0, 1]
        
        # Spearman rank correlation
        spearman_corr, _ = spearmanr(orig_flat, mask_flat)
        
        # Jensen-Shannon divergence
        orig_prob = orig_flat / orig_flat.sum()
        mask_prob = mask_flat / mask_flat.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        orig_prob = orig_prob + epsilon
        mask_prob = mask_prob + epsilon
        orig_prob = orig_prob / orig_prob.sum()
        mask_prob = mask_prob / mask_prob.sum()
        
        m = (orig_prob + mask_prob) / 2
        js_divergence = (entropy(orig_prob, m) + entropy(mask_prob, m)) / 2
        
        return {
            'density_pearson_correlation': pearson_corr,
            'density_spearman_correlation': spearman_corr,
            'density_js_divergence': js_divergence,
            'density_similarity': 1 - js_divergence  # 1 = identical, 0 = completely different
        }
    
    @staticmethod
    def calculate_query_accuracy(original_coords: np.ndarray,
                                masked_coords: np.ndarray,
                                n_queries: int = 100,
                                query_radius: float = 500.0) -> Dict[str, float]:
        """
        Calculate accuracy of range queries on anonymized data.
        
        Args:
            original_coords: Original coordinates (N, 2)
            masked_coords: Masked coordinates (N, 2)
            n_queries: Number of random query points to test
            query_radius: Radius for range queries
            
        Returns:
            Dictionary with query accuracy metrics
        """
        # Generate random query points within data bounds
        min_coords = original_coords.min(axis=0)
        max_coords = original_coords.max(axis=0)
        
        query_points = np.random.uniform(
            low=min_coords,
            high=max_coords,
            size=(n_queries, 2)
        )
        
        # Build trees
        orig_tree = KDTree(original_coords)
        mask_tree = KDTree(masked_coords)
        
        count_errors = []
        precision_scores = []
        recall_scores = []
        
        for query_point in query_points:
            # Range query on original data
            orig_indices = orig_tree.query_ball_point(query_point, query_radius)
            orig_count = len(orig_indices)
            
            # Range query on masked data
            mask_indices = mask_tree.query_ball_point(query_point, query_radius)
            mask_count = len(mask_indices)
            
            # Count error
            count_error = abs(orig_count - mask_count)
            count_errors.append(count_error)
            
            # Calculate precision and recall
            if mask_count > 0:
                # True positives: points that are in both results
                # (This is an approximation since we don't have 1-1 correspondence)
                precision = min(orig_count, mask_count) / mask_count
                precision_scores.append(precision)
            
            if orig_count > 0:
                recall = min(orig_count, mask_count) / orig_count
                recall_scores.append(recall)
        
        return {
            'mean_count_error': np.mean(count_errors),
            'median_count_error': np.median(count_errors),
            'mean_precision': np.mean(precision_scores) if precision_scores else 0,
            'mean_recall': np.mean(recall_scores) if recall_scores else 0,
            'query_accuracy': 1 - (np.mean(count_errors) / len(original_coords))
        }


def calculate_all_enhanced_metrics(original_coords: np.ndarray,
                                   masked_coords: np.ndarray) -> Dict[str, any]:
    """
    Calculate all enhanced metrics for anonymization evaluation.
    
    Args:
        original_coords: Original coordinates (N, 2)
        masked_coords: Masked coordinates (N, 2)
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = EnhancedMetrics()
    
    results = {}
    
    print("  - Calculating re-identification risk...")
    results['reidentification'] = metrics.calculate_reidentification_risk(
        original_coords, masked_coords
    )
    
    print("  - Calculating location entropy...")
    results['location_entropy_original'] = metrics.calculate_location_entropy(original_coords)
    results['location_entropy_masked'] = metrics.calculate_location_entropy(masked_coords)
    results['entropy_preservation'] = results['location_entropy_masked'] / results['location_entropy_original']
    
    print("  - Calculating Ripley's K function...")
    radii = [100, 250, 500, 1000]
    results['ripleys_k_original'] = metrics.calculate_ripleys_k(original_coords, radii)
    results['ripleys_k_masked'] = metrics.calculate_ripleys_k(masked_coords, radii)
    results['ripleys_k_correlation'] = np.corrcoef(
        results['ripleys_k_original'],
        results['ripleys_k_masked']
    )[0, 1]
    
    print("  - Calculating nearest neighbor preservation...")
    results['neighbor_preservation'] = metrics.calculate_nearest_neighbor_preservation(
        original_coords, masked_coords, k=5
    )
    
    print("  - Calculating displacement direction bias...")
    results['direction_bias'] = metrics.calculate_displacement_direction_bias(
        original_coords, masked_coords
    )
    
    print("  - Calculating density correlation...")
    results['density_correlation'] = metrics.calculate_density_correlation(
        original_coords, masked_coords
    )
    
    print("  - Calculating query accuracy...")
    results['query_accuracy'] = metrics.calculate_query_accuracy(
        original_coords, masked_coords
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Enhanced Metrics Module")
    print("Import this module and use calculate_all_enhanced_metrics() function")
