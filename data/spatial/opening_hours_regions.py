"""Cluster municipalities by observed weekday shop/leisure opening profiles."""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from data.utils import coerce_boolean_series

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.osm.businesses")
    context.stage("data.spatial.municipalities")
    context.config("random_seed")
    context.config("opening_hours_region_clusters", default=3)
    context.config("opening_hours_region_min_businesses", default=50)
    context.config("opening_hours_region_buffer_m", default=5000.)
    context.config("opening_hours_region_buffer_step_m", default=2500.)


def execute(context):

    # Use observed business schedules, independent of synthetic destinations.

    businesses = gpd.read_parquet(context.stage("data.osm.businesses")["businesses"])
    municipalities, _ = context.stage("data.spatial.municipalities")
    count = context.config("opening_hours_region_clusters")
    regions, assignments, profiles = cluster_regions(businesses, municipalities, count, context.config("random_seed"),
        context.config("opening_hours_region_min_businesses"), context.config("opening_hours_region_buffer_m"),
        context.config("opening_hours_region_buffer_step_m"))

    # Export the dissolved regions and municipality-level evidence for review.

    output = Path(context.path())
    regions.to_parquet(output / "regions.parquet", index=False)
    assignments.to_parquet(output / "municipalities.parquet", index=False)
    profiles.to_parquet(output / "profiles.parquet", index=False)
    (output / "summary.json").write_text(json.dumps({
        "unit": "municipality", "random_seed": int(context.config("random_seed")),
        "buffer_m": float(context.config("opening_hours_region_buffer_m")),
        "buffer_step_m": float(context.config("opening_hours_region_buffer_step_m")),
        "adaptive_buffers": True,
        "expanded_municipalities": int(profiles.buffer_expansions.gt(0).sum()),
        "maximum_buffer_m": float(profiles.buffer_m.max()),
        "median_buffer_m": float(profiles.buffer_m.median()),
        "targets_not_met": int((~profiles.shop_target_met | ~profiles.leisure_target_met).sum()),
        "min_businesses": int(context.config("opening_hours_region_min_businesses")),
        "regions": len(regions), "municipalities": len(assignments),
        "municipalities_per_region": assignments.opening_hours_region.value_counts().sort_index().to_dict(),
        "shop_imputed": int(profiles.shop_imputed.sum()),
        "leisure_imputed": int(profiles.leisure_imputed.sum()),
        "shop_sparse_without_buffer": int((profiles.shop_local_businesses < context.config("opening_hours_region_min_businesses")).sum()),
        "leisure_sparse_without_buffer": int((profiles.leisure_local_businesses < context.config("opening_hours_region_min_businesses")).sum()),
    }, indent=2), encoding="utf-8")
    return regions


def cluster_regions(businesses, municipalities, count=3, seed=0, min_businesses=50, buffer_m=5000., buffer_step_m=2500.):
    municipalities = municipalities[["municipality_id", "geometry"]].to_crs(2056).dissolve(by="municipality_id", as_index=False).sort_values("municipality_id").reset_index(drop=True)
    swiss = municipalities.municipality_id.gt(0)
    # The configured external extent can include Swiss territory. Swiss municipal
    # polygons take precedence; dissolved cluster regions must not overlap.
    if swiss.any() and (~swiss).any():
        municipalities.loc[~swiss, "geometry"] = municipalities.loc[~swiss, "geometry"].difference(municipalities.loc[swiss].geometry.union_all())
        municipalities = municipalities.loc[~municipalities.geometry.is_empty].reset_index(drop=True)
    if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or not 1 <= count <= len(municipalities):
        raise ValueError("opening_hours_region_clusters must be an integer between 1 and the number of municipalities")
    if not isinstance(min_businesses, (int, np.integer)) or isinstance(min_businesses, bool) or min_businesses < 1:
        raise ValueError("opening_hours_region_min_businesses must be a positive integer")
    if isinstance(buffer_m, bool) or not isinstance(buffer_m, (int, float, np.number)) or not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("opening_hours_region_buffer_m must be finite and non-negative")
    if isinstance(buffer_step_m, bool) or not isinstance(buffer_step_m, (int, float, np.number)) or not np.isfinite(buffer_step_m) or buffer_step_m <= 0:
        raise ValueError("opening_hours_region_buffer_step_m must be finite and positive")
    businesses = businesses[["offers_shop", "offers_leisure", "typical_intervals", "geometry"]].to_crs(2056).reset_index(drop=True)
    local = gpd.sjoin(businesses, municipalities, how="inner", predicate="intersects")
    local = local.sort_values("municipality_id").loc[lambda df: ~df.index.duplicated()]
    joined, radii, expansions = adaptive_neighbourhoods(
        businesses, municipalities, min_businesses, buffer_m, buffer_step_m)
    # A business intentionally contributes to every municipality whose buffer
    # includes it, once per municipality. Original polygons remain unchanged.
    sample_minutes = np.arange(15, 1440, 30)
    decoded = {}
    for value in joined.typical_intervals.dropna().unique():
        available = np.zeros(48, dtype=float)
        for start, end in json.loads(value):
            for offset in (-1440, 0, 1440):
                available[(sample_minutes >= start + offset) & (sample_minutes < end + offset)] = 1.
        decoded[value] = available
    profile_columns = {"municipality_id": municipalities.municipality_id.to_numpy(),
                       "buffer_m": radii, "buffer_expansions": expansions}
    for purpose in ("shop", "leisure"):
        selected = joined.loc[coerce_boolean_series(joined["offers_" + purpose], name="offers_" + purpose)
                              & joined.typical_intervals.notna()]
        if len(selected):
            # Overlapping buffers must not multiply a business's national vote.
            unique = selected.loc[~selected.index.duplicated()]
            all_values = np.stack([decoded[value] for value in unique.typical_intervals])
            national = all_values.mean(axis=0)
        else:
            national = np.zeros(48)
        groups = {key: group for key, group in selected.groupby("municipality_id")}
        rows, counts = [], []
        for municipality in municipalities.municipality_id:
            group = groups.get(municipality)
            counts.append(0 if group is None else len(group))
            rows.append(national if group is None or len(group) < min_businesses else np.stack([decoded[v] for v in group.typical_intervals]).mean(axis=0))
        profile_columns[purpose + "_businesses"] = counts
        local_counts = local.loc[coerce_boolean_series(local["offers_" + purpose], name="offers_" + purpose)
                                 & local.typical_intervals.notna()].groupby("municipality_id").size()
        profile_columns[purpose + "_local_businesses"] = municipalities.municipality_id.map(local_counts).fillna(0).to_numpy(dtype=int)
        profile_columns[purpose + "_imputed"] = np.asarray(counts) < min_businesses
        profile_columns[purpose + "_target_met"] = np.asarray(counts) >= min_businesses
        for slot in range(48):
            profile_columns[f"{purpose}_{slot:02d}"] = np.asarray(rows)[:, slot]
    profiles = pd.DataFrame(profile_columns)
    columns = [f"{purpose}_{slot:02d}" for purpose in ("shop", "leisure") for slot in range(48)]
    features = profiles[columns].to_numpy()
    if len(np.unique(features, axis=0)) < count:
        raise ValueError("Too few distinct observed opening-hours profiles to form the requested regions")
    fitted = KMeans(n_clusters=count, n_init=20, random_state=seed).fit(features)
    # Stable labels, roughly ordered by evening availability, not geography.
    order = sorted(range(count), key=lambda k: (fitted.cluster_centers_[k, 36:48].mean() +
                  fitted.cluster_centers_[k, 84:96].mean(), tuple(fitted.cluster_centers_[k])))
    labels = {label: position for position, label in enumerate(order)}
    municipalities["opening_hours_region"] = [labels[label] for label in fitted.labels_]
    profiles["opening_hours_region"] = municipalities.opening_hours_region.to_numpy()
    regions = municipalities.dissolve(by="opening_hours_region", as_index=False)[["opening_hours_region", "geometry"]]
    members = municipalities.groupby("opening_hours_region").municipality_id.apply(lambda ids: json.dumps(ids.tolist()))
    regions["municipality_ids"] = regions.opening_hours_region.map(members)
    return regions, municipalities, profiles


def adaptive_neighbourhoods(businesses, municipalities, minimum, initial, step):
    """Expand one common radius until both purpose counts meet their targets.

    Spatial-index distance queries implement exact polygon buffers without
    repeatedly materializing complex buffered geometries. Original polygons
    and the seeded clustering remain unchanged.
    """
    known = businesses.typical_intervals.notna().to_numpy()
    flags = np.column_stack([coerce_boolean_series(businesses["offers_" + purpose],
        name="offers_" + purpose).to_numpy() & known for purpose in ("shop", "leisure")])
    totals = flags.sum(axis=0)
    required = np.minimum(minimum, totals)
    if np.any(totals < minimum):
        logger.warning("Opening-hours region target %d cannot be met globally for every purpose; "
                       "available shop/leisure businesses: %s. Shortfalls remain flagged.", minimum, totals.tolist())
    if businesses.empty:
        raise ValueError("No businesses available for opening-hours region inference")
    tree = businesses.sindex
    bounds = businesses.total_bounds
    groups, ids, radii, expansions = [], [], [], []
    for row in municipalities.itertuples(index=False):
        geometry = row.geometry
        point = geometry.representative_point()
        # This upper bound includes every input business and guarantees a
        # finite stop even when the requested sample is globally unavailable.
        limit = np.hypot(max(abs(bounds[0]-point.x), abs(bounds[2]-point.x)),
                         max(abs(bounds[1]-point.y), abs(bounds[3]-point.y))) + 1.
        radius, iterations = float(initial), 0
        while True:
            indices = np.sort(tree.query(geometry, predicate="dwithin", distance=radius))
            if (flags[indices].sum(axis=0) >= required).all():
                break
            if radius >= max(initial, limit):
                logger.warning("Opening-hours region neighbourhood exhausted available geometry for municipality %s", row.municipality_id)
                break
            radius = min(radius + step, max(initial, limit))
            iterations += 1
        groups.append(indices)
        ids.append(np.full(len(indices), row.municipality_id))
        radii.append(radius)
        expansions.append(iterations)
    joined = businesses.iloc[np.concatenate(groups)].copy()
    joined["municipality_id"] = np.concatenate(ids)
    return joined, np.asarray(radii), np.asarray(expansions)
