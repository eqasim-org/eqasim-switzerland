"""Prepare enterprise, centroid and border coordinates for used NPVM freight zones."""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import nearest_points

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("synthesis.freight.npvm.od")
    context.stage("data.statent.statent")
    context.stage("data.spatial.swiss_border")
    context.config("data_path")
    context.config("random_seed")
    context.config("npvm_freight_swiss_zones_file", default="npvm/1_Verkehrszonen_Schweiz_NPVM_2023.gpkg")
    context.config("npvm_freight_foreign_zones_file", default="npvm/Verkehrszonen_Ausland_NPVM_2023.gpkg")
    context.config("npvm_freight_additional_centroids_file", default="npvm/NPVM_2023_additional_zone_centroids.csv")
    context.config("npvm_freight_use_border_crossings", default=True)


def execute(context):

    # Load the original NPVM zone IDs required by the sampled OD pairs.

    od = context.stage("synthesis.freight.npvm.od")
    used_ids = set(od.origin_zone) | set(od.destination_zone)
    base = Path(context.config("data_path"))
    swiss = read_zones(base / context.config("npvm_freight_swiss_zones_file"), "No", "N_Gem", True)
    swiss = swiss.loc[swiss.zone_id.isin(used_ids)].copy()
    foreign_ids = used_ids - set(swiss.zone_id)
    parts = [swiss]
    if foreign_ids:
        foreign = read_zones(base / context.config("npvm_freight_foreign_zones_file"), "ID_Zone", "Name", False)
        parts.append(foreign.loc[foreign.zone_id.isin(foreign_ids)].copy())
    zones = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:2056")
    missing = used_ids - set(zones.zone_id)
    if missing:
        extra = read_additional_centroids(base / context.config("npvm_freight_additional_centroids_file"))
        zones = gpd.GeoDataFrame(pd.concat([zones, extra.loc[extra.zone_id.isin(missing)]], ignore_index=True),
                                geometry="geometry", crs="EPSG:2056")
        missing = used_ids - set(zones.zone_id)
    if missing:
        raise ValueError(f"NPVM freight zone polygons missing for IDs: {sorted(missing)}")

    # Group valid STATENT establishments by Swiss NPVM zone without modifying STATENT.

    enterprises = context.stage("data.statent.statent")
    points = enterprise_candidates(enterprises, swiss)
    fallback = []

    # Retain demand in Swiss zones without establishments using the zone centroid.

    for row in swiss.itertuples():
        if row.zone_id not in points:
            center = row.geometry.centroid
            points[row.zone_id] = np.array([[center.x, center.y]], dtype=float)
            fallback.append(row.zone_id)
    logger.info("NPVM freight: %d used Swiss zones; %d have centroid fallback", len(swiss), len(fallback))

    # Keep the old geometric fallback only when road crossings are disabled.

    use_crossings = context.config("npvm_freight_use_border_crossings")
    foreign = zones.loc[~zones.is_swiss]
    if not use_crossings and not foreign.empty:
        border = context.stage("data.spatial.swiss_border").to_crs(2056).union_all().boundary
        for row in foreign.itertuples():
            point = nearest_points(border, row.geometry.centroid)[0]
            points[row.zone_id] = np.array([[point.x, point.y]], dtype=float)
    for zone_id, candidates in points.items():
        if not np.isfinite(candidates).all():
            raise ValueError(f"Invalid NPVM sampling coordinates in zone {zone_id}")

    # Return sampling points and original polygons for trip generation and inspection.

    return {"points": points, "zones": zones, "centroid_fallback_zones": fallback,
            "use_border_crossings": use_crossings}


def validate(context):
    """Track zone-layer changes without requiring optional foreign inputs for domestic demand."""
    tokens = []
    for option in ("npvm_freight_swiss_zones_file", "npvm_freight_foreign_zones_file",
                   "npvm_freight_additional_centroids_file"):
        path = Path(context.config("data_path")) / context.config(option)
        stat = path.stat() if path.is_file() else None
        tokens.append([str(path.resolve()), stat.st_size if stat else None, stat.st_mtime_ns if stat else None])
    return tokens


def read_zones(path, id_column, name_column, is_swiss):
    """Normalize zone schemas, preserving their identifiers and original geometry."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing NPVM freight zone layer: {path}")
    zones = gpd.read_file(path)
    if zones.crs is None:
        raise ValueError(f"NPVM zone layer has no CRS: {path}")
    ids = pd.to_numeric(zones[id_column], errors="raise")
    if ids.isna().any() or not np.isfinite(ids).all() or (ids % 1 != 0).any():
        raise ValueError(f"NPVM zone IDs must be finite integers in {path}")
    zones = zones[[id_column, name_column, "geometry"]].rename(columns={id_column: "zone_id", name_column: "name"})
    zones["zone_id"] = ids.astype(np.int64)
    zones["name"] = zones.name.fillna(zones.zone_id.astype(str)).astype(str)
    zones["is_swiss"] = is_swiss
    if zones.zone_id.duplicated().any():
        raise ValueError(f"Duplicate NPVM zone IDs in {path}")
    if zones.geometry.isna().any() or zones.geometry.is_empty.any():
        raise ValueError(f"NPVM zone layer has missing or empty polygons: {path}")
    invalid = ~zones.geometry.is_valid
    if invalid.any():
        logger.warning("Repairing %d invalid NPVM geometries in memory from %s", invalid.sum(), path)
        zones.loc[invalid, "geometry"] = zones.loc[invalid, "geometry"].make_valid()
    if not zones.geometry.is_valid.all() or zones.geometry.is_empty.any():
        raise ValueError(f"NPVM zone layer still has invalid geometry after repair: {path}")
    return zones.to_crs(2056)


def enterprise_candidates(enterprises, zones):
    """Sample establishments equally; row labels and employee counts do not weight them."""
    if zones.empty:
        return {}
    coordinates = enterprises[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(coordinates.to_numpy(dtype=float)).all(axis=1)
    candidates = gpd.GeoDataFrame(coordinates.loc[valid].copy(),
        geometry=gpd.points_from_xy(coordinates.loc[valid, "x"], coordinates.loc[valid, "y"]),
        crs="EPSG:2056")
    joined = gpd.sjoin(candidates, zones[["zone_id", "geometry"]], how="inner", predicate="within")
    return {int(zone_id): group[["x", "y"]].to_numpy(dtype=float)
            for zone_id, group in joined.groupby("zone_id", sort=True)}


def read_additional_centroids(path):
    """Read exact VISUM centroids for additional zones absent from the polygon release."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"NPVM has zones without published polygons; supply their source centroids: {path}")
    frame = pd.read_csv(path)
    if not {"zone_id", "x", "y", "name"}.issubset(frame):
        raise ValueError(f"Additional NPVM centroids require zone_id,x,y,name columns: {path}")
    numeric = frame[["zone_id", "x", "y"]].apply(pd.to_numeric, errors="raise")
    if (not np.isfinite(numeric.to_numpy()).all() or (numeric.zone_id % 1 != 0).any()
            or frame.zone_id.duplicated().any()):
        raise ValueError(f"Invalid or duplicate additional NPVM centroids: {path}")
    frame["zone_id"] = numeric.zone_id.astype(np.int64)
    frame["is_swiss"] = False
    return gpd.GeoDataFrame(frame[["zone_id", "name", "is_swiss"]],
        geometry=gpd.points_from_xy(numeric.x, numeric.y), crs="EPSG:2056")
