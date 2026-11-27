"""Smooth continuous trip surfaces into genuine polygons ordered by intensity."""

import contourpy
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from ..activity_chain_features import FEATURES, CLUSTER_COLUMN, effective_sample


def sample_fields(x, y, fields, coordinates):
    positions = np.array(
        [
            (coordinates[:, 1] - y[0]) / (y[1] - y[0]),
            (coordinates[:, 0] - x[0]) / (x[1] - x[0]),
        ]
    )
    return np.column_stack(
        [
            map_coordinates(fields[:, :, i], positions, order=1, mode="nearest")
            for i in range(fields.shape[2])
        ]
    )


def weighted_quantile(values, weights, probability):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(
        values[
            order[
                min(
                    np.searchsorted(cumulative, probability * cumulative[-1]),
                    len(order) - 1,
                )
            ]
        ]
    )


def filled_geometry(x, y, field, low, high, border):
    # A common triangle subdivision keeps triple junctions consistent across scores.
    generator = contourpy.contour_generator(
        x=x, y=y, z=field, fill_type="OuterOffset", corner_mask=False, quad_as_tri=True
    )
    point_sets, offset_sets = generator.filled(float(low), float(high))
    polygons = []
    for points, offsets in zip(point_sets, offset_sets):
        rings = [points[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        if rings:
            polygons.append(shapely.make_valid(shapely.Polygon(rings[0], rings[1:])))
    return shapely.intersection(shapely.union_all(polygons), shapely.force_2d(border))


def check_partition(regions, border):
    area = border.area
    overlap = regions.geometry.area.sum() - regions.geometry.union_all().area
    missing = (
        regions.geometry.union_all().symmetric_difference(shapely.force_2d(border)).area
    )
    if abs(overlap) > area * 1e-9 or missing > area * 1e-9:
        raise RuntimeError("Contour polygons have gaps or overlapping area")


def behavioural_contours(x, y, fields, cutoff, centres, mean, scale, border):
    """Use total-intensity and centroid-discriminant contours as shared interfaces."""
    total = fields[:, :, 3]
    high = filled_geometry(x, y, total, cutoff, float(total.max()) + 1, border)
    lower = filled_geometry(x, y, total, float(total.min()) - 1, cutoff, border)
    standardized = (fields - mean) / scale
    # Pairwise score differences are linear in the four continuous input fields.
    scores = 2 * np.einsum("ijk,lk->ijl", standardized, centres) - (centres**2).sum(
        axis=1
    )
    rows = [{CLUSTER_COLUMN: 1, "geometry": high}]
    order = np.argsort((centres * scale + mean)[:, 3])[::-1]
    for label, j in enumerate(order, start=2):
        geometry = lower
        for k in range(3):
            if k == j:
                continue
            delta = scores[:, :, j] - scores[:, :, k]
            halfspace = filled_geometry(
                x, y, delta, 0.0, max(float(delta.max()) + 1, 1.0), border
            )
            geometry = geometry.intersection(halfspace)
        if geometry.is_empty or geometry.area <= 0:
            raise ValueError("Behavioural contour region is empty")
        rows.append({CLUSTER_COLUMN: label, "geometry": geometry})
    regions = gpd.GeoDataFrame(rows, crs=2056)
    check_partition(regions, border)
    return regions


def assign_points(regions, xy):
    labels = np.zeros(len(xy), dtype=np.int8)
    for row in regions.sort_values(CLUSTER_COLUMN).itertuples():
        hit = shapely.intersects_xy(row.geometry, xy[:, 0], xy[:, 1])
        labels[(labels == 0) & hit] = getattr(row, CLUSTER_COLUMN)
    if (labels == 0).any():
        raise ValueError(
            f"{int((labels == 0).sum())} valid home points outside contour polygons"
        )
    return labels


def smooth_contours(
    cells, population, diaries, border, step=1000.0, sigma=0.0, high_share=0.16, seed=0
):
    if not np.isfinite(step) or step <= 0 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Geometry step must be positive and smoothing non-negative")
    if not 0 < high_share < 0.5:
        raise ValueError("High-activity population share must be between 0 and 0.5")
    bounds = border.bounds
    x = np.arange(
        np.floor(bounds[0] / step) * step - step,
        np.ceil(bounds[2] / step) * step + 2 * step,
        step,
    )
    y = np.arange(
        np.floor(bounds[1] / step) * step - step,
        np.ceil(bounds[3] / step) * step + 2 * step,
        step,
    )
    xx, yy = np.meshgrid(x, y)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    coordinates = cells[["x", "y"]].to_numpy()
    rates = cells[list(FEATURES)].to_numpy()
    linear = LinearNDInterpolator(coordinates, rates)(query)
    missing = ~np.isfinite(linear).all(axis=1)
    linear[missing] = NearestNDInterpolator(coordinates, rates)(query[missing])
    fields = linear.reshape(len(y), len(x), len(FEATURES))
    # Filter field values, never the categorical labels; preserve shared boundaries.
    for i in range(len(FEATURES)):
        fields[:, :, i] = gaussian_filter(
            fields[:, :, i], sigma=sigma / step, mode="nearest"
        )
    keep = (
        population.population_type.eq(1)
        & population.type_of_residence.eq(1)
        & population.age.ge(6)
    )
    px = population.home_x.to_numpy(float)
    py = population.home_y.to_numpy(float)
    valid = keep.to_numpy() & np.isfinite(px) & np.isfinite(py)
    valid &= (
        (px >= bounds[0]) & (px <= bounds[2]) & (py >= bounds[1]) & (py <= bounds[3])
    )
    ids = np.flatnonzero(valid)
    ids = ids[shapely.intersects_xy(border, px[ids], py[ids])]
    homes = np.column_stack([px[ids], py[ids]])
    predicted = sample_fields(x, y, fields, homes)
    intensity = predicted[:, 3]
    # Reserve the highest total-intensity population tail for group 1.
    values, weights = np.unique(np.round(intensity, 4), return_counts=True)
    cutoff = weighted_quantile(values, weights, 1 - high_share)
    below = intensity < cutoff
    if len(np.unique(values[values < cutoff])) < 3:
        raise ValueError(
            "Trip surface needs at least three distinct intensities below the high-activity tail"
        )
    # Compress occupied geometry-grid bins for fitting, preserving resident weights.
    # All four features enter equally after population-weighted standardization.
    lower_rates = predicted[below]
    mean = lower_rates.mean(axis=0)
    scale = lower_rates.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    bins = np.floor((homes[below] - [x[0], y[0]]) / step).astype(np.int64)
    _, inverse = np.unique(bins[:, 1] * len(x) + bins[:, 0], return_inverse=True)
    mass = np.bincount(inverse)
    support = np.column_stack(
        [
            np.bincount(inverse, weights=lower_rates[:, j]) / mass
            for j in range(len(FEATURES))
        ]
    )
    with threadpool_limits(limits=1):
        model = KMeans(
            n_clusters=3, n_init=50, random_state=seed, tol=0.0, max_iter=1000
        ).fit((support - mean) / scale, sample_weight=mass)
    if model.n_iter_ >= model.max_iter or len(np.unique(model.labels_)) != 3:
        raise RuntimeError(
            "Four-feature clustering did not converge to three distinct lower-intensity groups"
        )
    regions = behavioural_contours(
        x, y, fields, cutoff, model.cluster_centers_, mean, scale, border
    )
    labels = assign_points(regions, homes)
    survey_xy = diaries[["home_x", "home_y"]].to_numpy()
    survey_labels = assign_points(regions, survey_xy)
    profiles = []
    for group in range(1, 5):
        mask = labels == group
        observed = diaries.loc[survey_labels == group]
        if not mask.any() or observed.empty:
            raise ValueError("Every class requires residents and survey support")
        row = {
            CLUSTER_COLUMN: group,
            "population": int(mask.sum()),
            "population_share": float(mask.mean()),
            "survey_persons": len(observed),
            "survey_effective_sample": effective_sample(
                observed.person_weight.to_numpy()
            ),
        }
        for j, f in enumerate(FEATURES):
            row[f] = float(predicted[mask, j].mean())
            row["survey_" + f] = float(
                np.average(observed[f], weights=observed.person_weight)
            )
        profiles.append(row)
    profiles = pd.DataFrame(profiles)
    regions = regions.merge(profiles, on=CLUSTER_COLUMN, validate="one_to_one")
    metadata = {
        "seed": int(seed),
        "geometry_step_m": float(step),
        "smoothing_m": float(sigma),
        "high_population_target": float(high_share),
        "high_total_trips_cutoff": cutoff,
        "population": len(homes),
        "fitting_diaries": len(diaries),
        "clustering_iterations": int(model.n_iter_),
        "clustering_converged": True,
        "clustering_restarts": int(model.n_init),
        "clustering_inertia": float(model.inertia_),
        "clustering_support_bins": len(mass),
        "clustering_features": list(FEATURES),
        "clustering_feature_means": mean.tolist(),
        "clustering_feature_scales": scale.tolist(),
        "lower_group_centres_standardized": model.cluster_centers_.tolist(),
        "ordering": "Group 1 = highest total-trip population tail; groups 2–4 ordered by total-trip centroid",
        "method": "Adaptive four-feature fields with optional Gaussian smoothing; high total-intensity population tail plus three weighted k-means groups; continuous discriminant contours",
        "observed_vs_modelled": "Survey means are observed weighted diary rates in the exported polygons; model means integrate smoothed fields over actual STATPOP home points.",
        "target_note": "No rate was rescaled to obtain four trips/day. Top population share is a descriptive design setting, not a fitted predictive optimum.",
        "validation_note": "Postprocessing of a surface fitted on the full survey; observed class means are in-sample, not independent validation.",
        "geometry_note": f"{step:g} m is contour tessellation precision, not source-cell size or MZ evidence resolution; original adaptive cells remain 2–32 km.",
    }
    return {
        "regions": regions,
        "profiles": profiles,
        "metadata": metadata,
        "x": x,
        "y": y,
        "fields": fields,
    }
