# Anonymization Methods

## 1. Density-Aware Geomasking
> finds a new random location that is within a min and max radius depending on the distance to the k-nearest neighbour. 

1. Create KD tree to find nearest neighbour
2. Compute maximum radius `r_max` of the perturbation region based on point's distance to kth nearest neighbour. 
3. Samples a random new location within a donut (`[r_min, r_max]`). Moves to this new location if it's a valid location. Otherwise, move it to neighbour's location

## 2. Spatial k-Anonymity
> finds a radius that encapsulates k neighbours and samples a random location within that radius. Lacks formal guarantees.
1. Using KD tree determines the distance needed to contain k nearest neighbours
2. Can project to a region, centroid of the k-neighbours, or random location within the circle
(not sure how to deal with region identifiers)

## 3. Geo-Differential Privacy
> Given a privacy budget, moves coordinates according to a Laplace distribution
1. Samples `dx` and `dy` from Laplace distribution based on privacy budget `epsilon`. `epsilon` is the inverse of the variance of the Laplace distribution.
(not density aware)

## 4. Adaptive Voronoi
> finds a point within a polygon of its cluster 

1. Cluster points into regions. The downside here is that the number of households per cluster are pre-determined and do not depend on the density. The size of the region (polygon) where they are then randomly placed depend on the density, where the region is bigger when region is more sparsely populated. 
2. For each cluster, compute a region (polygon) that can encapsulate all points in that region
3. For each point in the polygon, sample a random point inside it. If the new location is invalid, use the location of the centroid. 

1. DISPLACEMENT METRICS:
--------------------------------------------------------------------------------
How far points move from their original location.
- Mean/Median: Average distance moved (lower = more accurate, higher = more private)
- Max: Worst-case displacement
- Std: Consistency of displacement (lower = more predictable)

--------------------------------------------------------------------------------
2. K-ANONYMITY PRESERVATION RATES:
--------------------------------------------------------------------------------
Higher values = better privacy (harder to isolate individuals by their neighbors).
- k=3: At least 3 neighbors preserved (basic privacy)
- k=5: At least 5 neighbors preserved (moderate privacy)  
- k=10: At least 10 neighbors preserved (strong privacy)
Values closer to 1.0 mean neighborhoods are well-preserved.

--------------------------------------------------------------------------------
3. RE-IDENTIFICATION RISK:
--------------------------------------------------------------------------------
Measures how easily original individuals can be identified in anonymized data.
- Re-identification rate: % of points correctly matched within 50m (lower = more private)
- Unique points ratio: % of points that are uniquely identifiable (lower = more private)
- Avg confusion distance: Average distance to closest point (higher = more confusion = more private)
Good anonymization should have LOW re-identification rate and unique points ratio.

--------------------------------------------------------------------------------
4. SPATIAL DISTRIBUTION PRESERVATION:
--------------------------------------------------------------------------------
Measures how well the overall spatial pattern is maintained.
- Entropy preservation: Ratio of spatial uniformity (1.0 = perfect, >1.0 = more spread, <1.0 = more clustered)
- Ripley's K correlation: Correlation of clustering patterns at multiple scales (1.0 = perfect preservation)
Higher values = better utility for spatial analysis.

--------------------------------------------------------------------------------
5. NEIGHBOR PRESERVATION:
--------------------------------------------------------------------------------
Measures how well relationships between nearby points are maintained.
- Mean neighbor preservation: % of original neighbors still nearby (higher = better utility)
- Mean rank correlation: How well neighbor ordering is preserved (higher = better utility)
Values closer to 1.0 mean point relationships are well-preserved for analysis.

--------------------------------------------------------------------------------
6. DIRECTIONAL BIAS:
--------------------------------------------------------------------------------
Checks if displacement has systematic directional patterns (e.g., all points move north).
- Directional bias: Strength of directional trend (0 = no bias, 1 = all same direction)
- Circular variance: Measure of direction spread (higher = more random directions)
- Is biased: True if bias > 0.3 threshold
Good anonymization should have LOW directional bias (more unpredictable).

--------------------------------------------------------------------------------
7. DENSITY CORRELATION:
--------------------------------------------------------------------------------
Measures how well population density patterns are preserved.
- Pearson correlation: Linear correlation of density grids (higher = better utility)
- Density similarity: Overall similarity of distributions (1.0 = identical, 0 = completely different)
- JS divergence: Statistical distance between distributions (lower = more similar)
Higher correlation/similarity = better for demographic/urban analysis.


--------------------------------------------------------------------------------
8. QUERY ACCURACY:
--------------------------------------------------------------------------------
Measures accuracy of spatial queries on anonymized data (e.g., "how many people in area X?").
- Query accuracy: Overall accuracy (closer to 1.0 = better)
- Mean count error: Average error in counting points within radius
- Mean precision: Correctness of returned results (higher = better)
- Mean recall: Completeness of returned results (higher = better)
Higher values = anonymized data gives more accurate answers to spatial queries.
