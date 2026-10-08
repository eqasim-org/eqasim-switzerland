""" Data provided by Myriam:

Read supplied Geneva construction flags and exclude affected comparisons.

Metadata locations are affected counting stations, not construction polygons.
Exact station IDs take precedence; proximity is a fallback for unknown IDs.
"""

import copy
import json
import logging
import os
import geopandas as gpd
import numpy as np
import pandas as pd


ID_COLUMNS = ["id", "point_de_mesure", "id_capteur_2025", "dfs_numero"]
logger = logging.getLogger("synpp")

def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"), "traffic_counts"))
    scope = context.config("analysis.counts.construction_scope", default="current_or_past")
    if scope not in ("current", "past", "current_or_past"):
        raise ValueError(f"{"analysis.counts.construction_scope"} must be current, past, or current_or_past")


def configure_filter(context):
    if context.config("analysis.counts.exclude_construction_sites", default=False):
        radius = context.config("analysis.counts.construction_radius_m", default=15.0)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError(f"{"analysis.counts.construction_radius_m"} must be a finite non-negative distance in metres")
        context.stage("analysis.counts.construction_sites")

def execute(context):
    sites = read_sites(os.path.join(context.config("counts_path"), "Geneva",
                                   "construction_sites", "metadonnees_comptages.csv"),
                       context.config("analysis.counts.construction_scope"))
    sites.to_file(os.path.join(context.path(), "construction_stations.gpkg"), driver="GPKG")
    logger.info("Construction metadata: %d/%d rows flagged (%s)",
                sites.affected.sum(), len(sites), context.config("analysis.counts.construction_scope"))
    return sites


######################## FUNCTIONS #######################

def read_sites(path, scope="current_or_past"):
    data = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    for column in data:
        data[column] = data[column].str.strip()
    required = {*ID_COLUMNS, "travaux_en_cours", "travaux_passes",
                "x_lv95", "y_lv95", "lon_wgs84", "lat_wgs84"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Construction metadata missing columns: {sorted(missing)}")
    
    flags = {}
    for column in ("travaux_en_cours", "travaux_passes"):
        values = data[column].str.lower()
        if not values.isin(["oui", "non", ""]).all():
            raise ValueError(f"Unexpected construction flag in {column}")
        flags[column] = values.eq("oui")

    if scope not in ("current", "past", "current_or_past"):
        raise ValueError("Unknown construction scope")
    
    data["affected"] = (flags["travaux_en_cours"] if scope == "current" else
                        flags["travaux_passes"] if scope == "past" else
                        flags["travaux_en_cours"] | flags["travaux_passes"])
    
    x, y = (pd.to_numeric(data[c], errors="coerce") for c in ("x_lv95", "y_lv95"))
    lon, lat = (pd.to_numeric(data[c], errors="coerce") for c in ("lon_wgs84", "lat_wgs84"))

    geometry = gpd.GeoSeries(None, index=data.index, crs=2056)
    local = np.isfinite(x) & np.isfinite(y)
    geometry.loc[local] = gpd.points_from_xy(x[local], y[local], crs=2056)
    fallback = ~local & lon.between(-180, 180) & lat.between(-90, 90)
    geometry.loc[fallback] = gpd.GeoSeries(
        gpd.points_from_xy(lon[fallback], lat[fallback]), index=data.index[fallback], crs=4326
    ).to_crs(2056)

    # Retain records without coordinates: their IDs can still be matched.
    return gpd.GeoDataFrame(data, geometry=geometry, crs=2056)


def identify_affected(counts, sites, city, radius):
    """Return exclusion evidence; known unaffected IDs do not use proximity."""
    lookup = {}
    for index, site in sites.iterrows():
        for column in ID_COLUMNS:
            value = site[column]
            if value:
                lookup.setdefault(value, set()).add(index)
    metric = counts.to_crs(2056)
    affected = sites[sites.affected & sites.geometry.notna()].to_crs(2056)
    records = []
    for _, row in metric.iterrows():
        if city.startswith("transcality"):
            tokens = [str(row["detector_ids"])]
        else:
            # Geneva OBJECTID is counter_id + '_' + directional denomination.
            tokens = str(row["id"]).rsplit("_", 1)
        candidates = set().union(*(lookup.get(token, set()) for token in tokens))
        impacted = sorted(index for index in candidates if sites.loc[index, "affected"])
        reason, distance = "id", np.nan
        if not candidates and radius > 0 and not affected.empty and row.geometry is not None:
            distances = affected.geometry.distance(row.geometry)
            impacted = distances.index[distances <= radius].tolist()
            if impacted:
                distance = float(distances.loc[impacted].min())
                reason = "proximity"
        if impacted:
            records.append({"id": row["id"], "reason": reason, "distance_m": distance,
                            "construction_ids": json.dumps(sorted(set(sites.loc[impacted, "id"])))})
            
    return pd.DataFrame(records, columns=["id", "reason", "distance_m", "construction_ids"])


def filter_counts(context, counts, matched, network, city):
    if not context.config("analysis.counts.exclude_construction_sites"):
        return counts, matched
    
    sites = context.stage("analysis.counts.construction_sites")
    evidence = identify_affected(counts.counts, sites, city, context.config("analysis.counts.construction_radius_m"))
    excluded = set(evidence.id)

    if city.startswith("transcality") and not matched.empty:
        links = pd.Series(network.get_in_simulation_links(matched.link_id), index=matched.index)
        affected_links = set(links[matched.id.isin(excluded)])
        shared_ids = set(matched.loc[links.isin(affected_links), "id"]) - excluded
        if shared_ids:
            evidence = pd.concat([evidence, pd.DataFrame({
                "id": sorted(shared_ids), "reason": "shared_simulation_link",
                "distance_m": np.nan, "construction_ids": "[]",
            })], ignore_index=True)
            excluded.update(shared_ids)

    evidence.to_csv(os.path.join(context.path(), "construction_exclusions.csv"), index=False)
    logger.info("%s: excluded %d/%d count observations due to construction", city, len(excluded), len(counts.counts))
    result = copy.copy(counts)
    result.count_data = counts.counts[~counts.counts.id.isin(excluded)].copy()
    result.count_stations = counts.count_stations[~counts.count_stations.id.isin(excluded)].copy()
    
    return result, matched[~matched.id.isin(excluded)].copy()
