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

1. Cluster points into regions 
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


### 1. Displacement Metrics (meters)
| Method | Mean | Median | Max | Std Dev |
|--------|------|--------|-----|---------|
| **Donut Geomask** | 323.4 | 248.0 | 1,997.1 | 263.6 |
| **K-Anonymity** | 257.3 | 187.8 | 5,111.7 | 290.6 |
| **Differential Privacy** | 19.9 | 16.7 | 123.3 | 14.0 |
| **Voronoi Mask** | 180.3 | 162.3 | 808.6 | 116.7 |

### 2. K-Anonymity Preservation Rates
| Method | k=3 | k=5 | k=10 |
|--------|-----|-----|------|
| **Donut Geomask** | 0.669 | 0.699 | 0.805 |
| **K-Anonymity** | 0.836 | 0.872 | 0.929 |
| **Differential Privacy** | 0.999 | 1.000 | 1.000 |
| **Voronoi Mask** | 0.832 | 0.896 | 0.932 |

### 3. Re-identification Risk
| Method | Re-ID Rate | Unique Points | Avg Confusion Dist (m) |
|--------|------------|---------------|------------------------|
| **Donut Geomask** | 0.144 | 0.107 | 147.1 |
| **K-Anonymity** | 0.208 | 0.151 | 144.9 |
| **Differential Privacy** | 0.970 | 0.733 | 18.7 |
| **Voronoi Mask** | 0.363 | 0.281 | 80.8 |

### 4. Spatial Distribution Preservation
| Method | Entropy Preservation | Ripley's K Correlation |
|--------|---------------------|----------------------|
| **Donut Geomask** | 1.011 | 1.000 |
| **K-Anonymity** | 1.009 | 1.000 |
| **Differential Privacy** | 1.003 | 1.000 |
| **Voronoi Mask** | 0.973 | 1.000 |

### 5. Neighbor Preservation
| Method | Mean Neighbor Preservation | Mean Rank Correlation |
|--------|---------------------------|---------------------|
| **Donut Geomask** | 0.298 | 0.489 |
| **K-Anonymity** | 0.416 | 0.602 |
| **Differential Privacy** | 0.879 | 0.919 |
| **Voronoi Mask** | 0.465 | 0.631 |

### 6. Directional Bias
| Method | Directional Bias | Circular Variance | Is Biased |
|--------|-----------------|------------------|-----------|
| **Donut Geomask** | 0.017 | 0.983 | No |
| **K-Anonymity** | 0.007 | 0.993 | No |
| **Differential Privacy** | 0.014 | 0.986 | No |
| **Voronoi Mask** | 0.035 | 0.965 | No |

### 7. Density Correlation
| Method | Pearson Correlation | Density Similarity | JS Divergence |
|--------|-------------------|------------------|---------------|
| **Donut Geomask** | 0.477 | 0.624 | 0.376 |
| **K-Anonymity** | 0.561 | 0.662 | 0.338 |
| **Differential Privacy** | 0.940 | 0.955 | 0.045 |
| **Voronoi Mask** | 0.672 | 0.785 | 0.215 |

### 8. Query Accuracy
| Method | Query Accuracy | Mean Count Error | Mean Precision | Mean Recall |
|--------|---------------|-----------------|----------------|-------------|
| **Donut Geomask** | 1.000 | 0.5 | 0.629 | 0.704 |
| **K-Anonymity** | 1.000 | 0.3 | 0.667 | 0.783 |
| **Differential Privacy** | 1.000 | 0.1 | 0.997 | 0.971 |
| **Voronoi Mask** | 1.000 | 0.2 | 0.969 | 0.744 |

