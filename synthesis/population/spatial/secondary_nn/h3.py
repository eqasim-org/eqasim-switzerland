from .tools import aggregate_osm_features
from functools import lru_cache

import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon
import logging
import numpy as np
from shapely import contains_xy
from shapely.ops import transform
from pyproj import Transformer
logger = logging.getLogger("synpp")

"""Destination features and a spatially nested hierarchy shared by all NN models.

Each point is indexed at every resolution. Nodes below the root are identified
by their coordinate path, and their polygons are clipped to all ancestors.
"""

H3_LEVELS = [5, 7, 10]
H3_LEVEL_NAMES = [f"level_{i}" for i in range(len(H3_LEVELS))]
OVGK_CATEGORIES = ['B', 'A', 'C', 'None', 'D']
H3_DEST_FEATURE_COLUMNS = ["num_statent", "employees", "urban_core", "urban", "education", "shop", "leisure", "sport", "gastronomy", "accommodation", "cultural", "ovgk_share_a", "ovgk_share_b", "ovgk_share_c", "ovgk_share_d", "ovgk_share_none"]

def _aggregate_destination_features_by_level(destinations_with_levels, level_col, all_hex):
    assert level_col in destinations_with_levels.columns
    assert destinations_with_levels[level_col].notna().all()
    assert set(destinations_with_levels["ovgk"].unique()).issubset(OVGK_CATEGORIES)

    cols = [level_col, "destination_id", "number_employees", "ovgk", "offers_education_secondary", "offers_shop", "offers_leisure", 
            "offers_sport", "offers_gastronomy", "offers_accommodation", "offers_cultural", "municipality_type"] 
    df = destinations_with_levels[cols].copy()

    # municipality_type indicators
    df["urban_core"] = (df["municipality_type"] == "urbancore").astype(np.float32)
    df["urban"] = (df["municipality_type"] == "urban").astype(np.float32)

    # OVGK shares
    ovgk_dummies = pd.get_dummies(df["ovgk"], prefix="ovgk_share", dtype=np.float32)

    # ensure all expected categories exist
    expected_cols = [f"ovgk_share_{c.lower()}" for c in OVGK_CATEGORIES]
    ovgk_dummies.columns = [c.lower() for c in ovgk_dummies.columns]
    ovgk_dummies = ovgk_dummies.reindex(columns=expected_cols, fill_value=0.0)
    df = pd.concat([df, ovgk_dummies[expected_cols]], axis=1)

    out = df.groupby(level_col, sort=False).agg(
                        num_statent=("number_employees", "size"),
                        employees=("number_employees", "sum"),
                        education=("offers_education_secondary", "sum"),
                        shop=("offers_shop", "sum"),
                        leisure=("offers_leisure", "sum"),
                        sport=("offers_sport", "sum"),
                        gastronomy=("offers_gastronomy", "sum"),
                        accommodation=("offers_accommodation", "sum"),
                        cultural=("offers_cultural", "sum"),
                        urban_core=("urban_core", "mean"),
                        urban=("urban", "mean"),
                        ovgk_share_a=("ovgk_share_a", "mean"),
                        ovgk_share_b=("ovgk_share_b", "mean"),
                        ovgk_share_c=("ovgk_share_c", "mean"),
                        ovgk_share_d=("ovgk_share_d", "mean"),
                        ovgk_share_none=("ovgk_share_none", "mean"),            
                        )

    out.index = out.index.astype(str)
    out.index.name = "h3_index"
    out = out.reindex(all_hex, fill_value=0.0)

    return out.reset_index()


def build_h3_tree(merged_level_geoms):
    """Build the shared tree from coordinate paths, never H3 logical ancestry."""
    valid = {name: set(frame["h3_index"]) for name, frame in merged_level_geoms.items()}
    tree = {}
    for leaf in sorted(valid[H3_LEVEL_NAMES[-1]]):
        root, middle, _ = leaf.split("/")
        branch = f"{root}/{middle}"
        if root not in valid[H3_LEVEL_NAMES[0]] or branch not in valid[H3_LEVEL_NAMES[1]]:
            raise ValueError(f"Missing coordinate ancestors for {leaf}")
        tree.setdefault(root, {}).setdefault(branch, []).append(leaf)
    return tree


@lru_cache(maxsize=16)
def _coordinate_transformer(crs):
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


@lru_cache(maxsize=4096)
def _cell_polygon(cell, crs):
    polygon = Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])
    if crs != "EPSG:4326":
        polygon = transform(_coordinate_transformer(crs).transform, polygon)
    return polygon


def coordinate_cell_geometry(path, crs="EPSG:4326"):
    """Intersect independently indexed cells so each branch lies inside its ancestors.

    A physical H3 cell may cross a coarser boundary. The full coordinate path
    distinguishes its pieces instead of assigning the whole cell to one parent.
    """
    geometry = None
    for cell in path.split("/"):
        polygon = _cell_polygon(cell, crs)
        geometry = polygon if geometry is None else geometry.intersection(polygon)
    return geometry


def to_geo_levels(df: gpd.GeoDataFrame, geometry_col: str = 'geometry', levels=H3_LEVELS) -> tuple[pd.DataFrame, dict]:
    """Index each coordinate independently, retaining the full path at each level."""
    points = gpd.GeoSeries(df[geometry_col], crs=df.crs)
    if points.crs is None:
        raise ValueError("A CRS is required to index locations")
    points = points.to_crs(epsg=4326)
    valid = np.isfinite(points.x) & np.isfinite(points.y)
    result = pd.DataFrame(index=df.index)
    unique = {}
    path = pd.Series("", index=df.index, dtype=object)
    for i, resolution in enumerate(levels):
        cells = points.loc[valid].apply(lambda point: h3.latlng_to_cell(point.y, point.x, resolution))
        path.loc[valid] = cells if i == 0 else path.loc[valid] + "/" + cells
        name = f"level_{i}"
        result[name] = path.where(valid, None)
        unique[name] = set(result[name].dropna())
    return result, unique

def within_ch(context, df, cols1=["origin_x", "origin_y"], cols2=None):
    df_switzerland = context.stage("data.spatial.swiss_border")
    ch_polygon = df_switzerland.buffer(0).iloc[0] 
    inside_ch = contains_xy(ch_polygon, df[cols1[0]].values, df[cols1[1]].values)
    if cols2 is not None:
        destination_inside_ch = contains_xy(ch_polygon, df[cols2[0]].values, df[cols2[1]].values)
        inside_ch = inside_ch & destination_inside_ch
    return inside_ch

def configure(context):
    context.stage("data.osm.secondary_features")
    context.stage("data.statent.statent")
    context.stage("synthesis.population.destinations")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.spatial.swiss_border")

def execute(context):
    logger.info("H3: Processing spatial data and converting to H3 indices...")
    data_collections = {}
    level_unique_hex_list = []  # To collect all level_unique_hex for merging later
    
    # Destination/company data (exclude non-offering rows such as remote work locations)
    logger.info("H3: \t Processing destination companies data...")
    destinations = context.stage("synthesis.population.destinations").copy()
    offer_cols = [f"offers_{p}" for p in ["shop", "leisure", "other", "work_secondary", "education_secondary", "home_secondary"] if f"offers_{p}" in destinations.columns]    
    assert len(offer_cols) > 0, "No offers_* columns found in synthesis.population.destinations"
    assert destinations[offer_cols].isna().sum().sum() == 0, "Found NaNs in offers_* columns of destinations data, please check the data preparation steps."

    # enrich destinations
    df_statent = context.stage("data.statent.statent")[["enterprise_id", "noga"]].copy()
    df_statent.columns = ["destination_id", "noga"]
    df_statent["noga"] = df_statent["noga"].fillna("").astype(str)
    df_statent["offers_cultural"] = df_statent["noga"].str.startswith("912") | df_statent["noga"].str.startswith("914")
    df_statent["offers_sport"] = df_statent["noga"].str.startswith("93")
    df_statent["offers_gastronomy"] = df_statent["noga"].str.startswith("56")
    df_statent["offers_accommodation"] = df_statent["noga"].str.startswith("55")
    additional_offers = ["offers_cultural", "offers_sport", "offers_gastronomy", "offers_accommodation"]
    offers = df_statent.groupby("destination_id")[additional_offers].any()
    destinations = destinations.merge(offers, on="destination_id", how="left", validate="one_to_one")
    destinations[additional_offers] = destinations[additional_offers].fillna(False)
    destinations["ovgk"] = destinations["ovgk"].fillna("None")
    employees = pd.to_numeric(destinations["number_employees"], errors="coerce")
    destinations["number_employees"] = employees.where(np.isfinite(employees), 1.0).clip(lower=0.0)

    # keep only rows that have at least one offer (this filter would remove remote work locations for example)
    destinations = destinations[destinations[offer_cols].any(axis=1)].reset_index(drop=True)
    offer_cols = offer_cols + additional_offers
    if destinations.empty:
        raise RuntimeError("No destinations offering secondary activities")

    destinations_attrs_cols = ["destination_id", "number_employees", "ovgk", "municipality_type"] + offer_cols
    destinations_attributes = destinations[destinations_attrs_cols].copy()
    destinations = gpd.GeoDataFrame(destinations[["destination_id", "geometry"]], geometry="geometry", crs=destinations.crs)

    destinations_levels, destinations_unique_hex = to_geo_levels(destinations, geometry_col="geometry")
    destinations_levels = destinations[["destination_id"]].join(destinations_levels)
    data_collections["destinations"] = destinations_levels

    destinations_with_levels = destinations_levels.merge(destinations_attributes, on="destination_id", how="left")

    level_unique_hex_list.append(destinations_unique_hex)
    del destinations

    # Microcensus trips data
    logger.info("H3: \t Processing Microcensus trips data...")
    mz_trips,_, _, _ = context.stage("data.microcensus.trips")
    mz_trips = mz_trips[["person_id","trip_id","origin_x","origin_y","destination_x", "destination_y"]]
    inside_ch = within_ch(context, mz_trips, cols1=["origin_x", "origin_y"], cols2=["destination_x", "destination_y"])
    mz_trips = mz_trips[inside_ch].reset_index(drop=True)
    mz_trips["origin_geometry"] = gpd.points_from_xy(mz_trips.origin_x, mz_trips.origin_y)
    mz_trips["destination_geometry"] = gpd.points_from_xy(mz_trips.destination_x, mz_trips.destination_y)
    mz_trips_origin_levels, mz_trips_origin_unique_hex = to_geo_levels(gpd.GeoDataFrame(mz_trips, 
                                                                                          geometry="origin_geometry", 
                                                                                          crs="EPSG:2056"), 
                                                                         geometry_col="origin_geometry")
    mz_trips_origin_levels.columns = ["origin_"+col for col in mz_trips_origin_levels.columns]
    mz_trips_destination_levels, mz_trips_destination_unique_hex = to_geo_levels(gpd.GeoDataFrame(mz_trips, 
                                                                                                    geometry="destination_geometry", 
                                                                                                    crs="EPSG:2056"), 
                                                                                   geometry_col="destination_geometry")
    mz_trips_destination_levels.columns = ["destination_"+col for col in mz_trips_destination_levels.columns]

    mz_trips_levels = mz_trips[["person_id","trip_id"]].join(mz_trips_origin_levels).join(mz_trips_destination_levels)
    data_collections["microcensus_trips"] = mz_trips_levels
    level_unique_hex_list.extend([mz_trips_origin_unique_hex, mz_trips_destination_unique_hex])
    del mz_trips

    # microcensus persons data
    logger.info("H3: \t Processing Microcensus persons data...")
    mz_persons = context.stage("data.microcensus.persons")[["person_id", "home_x", "home_y","work_x","work_y"]].reset_index(drop=True)
    
    work_within_ch = within_ch(context, mz_persons, cols1=["work_x", "work_y"])
    mz_persons.loc[~work_within_ch, ["work_x", "work_y"]] = np.inf  # Mark work locations outside Switzerland as inf to handle later with non employed people

    mz_persons["home_geometry"] = gpd.points_from_xy(mz_persons.home_x, mz_persons.home_y)
    mz_persons["work_geometry"] = gpd.points_from_xy(mz_persons.work_x, mz_persons.work_y)

    mz_persons_home_levels, mz_persons_home_unique_hex = to_geo_levels(gpd.GeoDataFrame(mz_persons, 
                                                                                         geometry="home_geometry", 
                                                                                         crs="EPSG:2056"),
                                                                        geometry_col="home_geometry")
    mz_persons_home_levels.columns = ["home_"+col for col in mz_persons_home_levels.columns]
    mz_persons_work_levels, mz_persons_work_unique_hex = to_geo_levels(gpd.GeoDataFrame(mz_persons, 
                                                                                         geometry="work_geometry", 
                                                                                         crs="EPSG:2056"),
                                                                        geometry_col="work_geometry")
    mz_persons_work_levels.columns = ["work_"+col for col in mz_persons_work_levels.columns]
    data_collections["microcensus_persons"] = mz_persons[["person_id"]].join(mz_persons_home_levels).join(mz_persons_work_levels)
    level_unique_hex_list.extend([mz_persons_home_unique_hex, mz_persons_work_unique_hex])
    del mz_persons

    # Merge level geometries across all datasets
    logger.info("H3: \t Merging level geometries across all datasets...")
    swiss_border = context.stage("data.spatial.swiss_border").to_crs("EPSG:2056").geometry.union_all()
    merged_level_geoms = {}
    for i in range(len(H3_LEVELS)):        
        all_hex = set()
        for level_unique_hex in level_unique_hex_list:
            if f'level_{i}' in level_unique_hex:
                all_hex.update(level_unique_hex[f'level_{i}'])
        
        logger.info(f"\t\t Processing level {H3_LEVELS[i]} with {len(all_hex)} unique hexagons...")

        all_hex = sorted(all_hex)
        # Project before clipping: projected straight polygon edges must also nest.
        polygons = [coordinate_cell_geometry(path, crs="EPSG:2056") for path in all_hex]
        merged_gdf = gpd.GeoDataFrame({"h3_index": all_hex}, geometry=polygons, crs="EPSG:2056")

        merged_gdf["centroid"] = merged_gdf.geometry.representative_point()
        intersection_area = merged_gdf.geometry.intersection(swiss_border).area
        total_area = merged_gdf.geometry.area
        merged_gdf["outside_fraction"] = np.clip(1 - (intersection_area / total_area), 0, 1)

        # Attach destination-derived H3 features so downstream models can consume one centralized source.
        level_col = f"level_{i}"
        feature_df = _aggregate_destination_features_by_level(destinations_with_levels, level_col, all_hex)
        merged_gdf = merged_gdf.merge(feature_df, on="h3_index", how="left")
        assert all(col in merged_gdf.columns for col in H3_DEST_FEATURE_COLUMNS), f"Missing expected destination feature columns in merged_gdf for level {i}. Expected at least: {H3_DEST_FEATURE_COLUMNS}, but got {merged_gdf.columns.tolist()}"
        assert merged_gdf[H3_DEST_FEATURE_COLUMNS].isna().sum().sum() == 0, f"Found NaNs in destination feature columns of merged_gdf for level {i}, please check the merging process. Columns with NaNs: {merged_gdf[H3_DEST_FEATURE_COLUMNS].isna().sum()}"
        merged_gdf[H3_DEST_FEATURE_COLUMNS] = merged_gdf[H3_DEST_FEATURE_COLUMNS]

        # Environmental context follows clipped cell geometry at every level.
        osm = aggregate_osm_features(merged_gdf, context.stage("data.osm.secondary_features"))
        merged_gdf[osm.columns] = osm
        merged_level_geoms[level_col] = merged_gdf

    # Filter hexagons to only keep those with companies (num_statent > 0).
    logger.info("H3: \t Filtering hexagons to only keep those with companies (num_statent > 0)...")
    valid_hex = {}
    for name in H3_LEVEL_NAMES:
        valid_hex[name] = set(merged_level_geoms[name].loc[merged_level_geoms[name]["num_statent"] > 0, "h3_index"])
        logger.info(f"H3: \t\t Valid {name} hexagons with companies: {len(valid_hex[name])}")

    for name in H3_LEVEL_NAMES:
        merged_level_geoms[name] = merged_level_geoms[name][merged_level_geoms[name]["h3_index"].isin(valid_hex[name])].reset_index(drop=True)

    # Filter MZ trips to only those whose destination falls in a company-having finest-level hexagon.
    finest_dest_col = f"destination_{H3_LEVEL_NAMES[-1]}"
    mz_trips_levels = data_collections["microcensus_trips"]
    n_before = len(mz_trips_levels)
    mz_trips_levels = mz_trips_levels[mz_trips_levels[finest_dest_col].isin(valid_hex[H3_LEVEL_NAMES[-1]])].reset_index(drop=True)
    n_after = len(mz_trips_levels)
    logger.info(f"H3: \t\t MZ trips filtered from {n_before} to {n_after} (dropped {n_before - n_after} trips to hexagons without companies)")
    data_collections["microcensus_trips"] = mz_trips_levels

    h3_tree = build_h3_tree(merged_level_geoms)

    return (data_collections, merged_level_geoms, h3_tree)
