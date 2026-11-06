"""Cluster cantons by observed typical weekday shop/leisure opening profiles."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from data.utils import coerce_boolean_series


def configure(context):
    context.stage("data.osm.businesses")
    context.stage("data.spatial.cantons")
    context.config("random_seed")
    context.config("opening_hours_region_clusters", default=3)
    context.config("opening_hours_region_min_businesses", default=20)


def execute(context):

    # Use observed business schedules, independent of synthetic destinations.

    businesses = gpd.read_parquet(context.stage("data.osm.businesses")["businesses"])
    cantons = context.stage("data.spatial.cantons")
    count = context.config("opening_hours_region_clusters")
    regions, assignments, profiles = cluster_regions(businesses, cantons, count, context.config("random_seed"),
        context.config("opening_hours_region_min_businesses"))

    # Export the dissolved regions and canton-level evidence for review.

    output = Path(context.path())
    regions.to_parquet(output / "regions.parquet", index=False)
    assignments.to_parquet(output / "cantons.parquet", index=False)
    profiles.to_parquet(output / "profiles.parquet", index=False)
    return regions


def cluster_regions(businesses, cantons, count=3, seed=0, min_businesses=20):
    cantons = cantons[["canton_id", "geometry"]].to_crs(2056).dissolve(by="canton_id", as_index=False).sort_values("canton_id").reset_index(drop=True)
    swiss = cantons.canton_id.between(1, 26)
    # The configured external extent can include Swiss territory. Swiss canton
    # polygons take precedence; dissolved cluster regions must not overlap.
    if swiss.any() and (~swiss).any():
        cantons.loc[~swiss, "geometry"] = cantons.loc[~swiss, "geometry"].difference(cantons.loc[swiss].geometry.union_all())
        cantons = cantons.loc[~cantons.geometry.is_empty].reset_index(drop=True)
    if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or not 1 <= count <= len(cantons):
        raise ValueError("opening_hours_region_clusters must be an integer between 1 and the number of cantons")
    if not isinstance(min_businesses, (int, np.integer)) or isinstance(min_businesses, bool) or min_businesses < 1:
        raise ValueError("opening_hours_region_min_businesses must be a positive integer")
    businesses = businesses.to_crs(2056).reset_index(drop=True)
    joined = gpd.sjoin(businesses, cantons, how="inner", predicate="intersects")
    # One vote per OSM object; deterministic treatment of boundary points.
    joined = joined.sort_values("canton_id").loc[lambda df: ~df.index.duplicated()]
    sample_minutes = np.arange(15, 1440, 30)
    decoded = {}
    for value in joined.typical_intervals.dropna().unique():
        available = np.zeros(48, dtype=float)
        for start, end in json.loads(value):
            for offset in (-1440, 0, 1440):
                available[(sample_minutes >= start + offset) & (sample_minutes < end + offset)] = 1.
        decoded[value] = available
    profile_columns = {"canton_id": cantons.canton_id.to_numpy()}
    for purpose in ("shop", "leisure"):
        selected = joined.loc[coerce_boolean_series(joined["offers_" + purpose], name="offers_" + purpose)
                              & joined.typical_intervals.notna()]
        if len(selected):
            all_values = np.stack([decoded[value] for value in selected.typical_intervals])
            national = all_values.mean(axis=0)
        else:
            national = np.zeros(48)
        groups = {key: group for key, group in selected.groupby("canton_id")}
        rows, counts = [], []
        for canton in cantons.canton_id:
            group = groups.get(canton)
            counts.append(0 if group is None else len(group))
            rows.append(national if group is None or len(group) < min_businesses else np.stack([decoded[v] for v in group.typical_intervals]).mean(axis=0))
        profile_columns[purpose + "_businesses"] = counts
        profile_columns[purpose + "_imputed"] = np.asarray(counts) < min_businesses
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
    cantons["opening_hours_region"] = [labels[label] for label in fitted.labels_]
    profiles["opening_hours_region"] = cantons.opening_hours_region.to_numpy()
    regions = cantons.dissolve(by="opening_hours_region", as_index=False)[["opening_hours_region", "geometry"]]
    members = cantons.groupby("opening_hours_region").canton_id.apply(lambda ids: json.dumps(ids.tolist()))
    regions["canton_ids"] = regions.opening_hours_region.map(members)
    return regions, cantons, profiles
