import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import numpy as np
import os
from shapely import vectorized

logger = logging.getLogger("synpp")

"""
This stage processes the spatial data from various sources (Statent, Microcensus trips, and synthetic population locations)
and converts their geometries into H3 hexagonal indices at three resolutions. 
It also merges the unique hexagons across all datasets for each resolution level to create a comprehensive set of 
hexagonal geometries that can be used for further analysis or visualization.
"""



H3_LEVELS = [5, 7, 9]
OVGK_CATEGORIES = ["A", "B", "C", "D", "none"]
H3_DEST_FEATURE_COLUMNS = [
    "num_statent",
    "employees",
    "urban_core",
    "urban",
    "education",
    "shop",
    "leisure",
    "ovgk_share_a",
    "ovgk_share_b",
    "ovgk_share_c",
    "ovgk_share_d",
    "ovgk_share_none",
]


def _aggregate_destination_features_by_level(destinations_with_levels, level_col):
    cols = [level_col, "destination_id", "number_employees", "ovgk", "offers_education_secondary", "offers_shop", "offers_leisure", "municipality_type"]
    available_cols = [c for c in cols if c in destinations_with_levels.columns]
    df = destinations_with_levels[available_cols].copy()

    if level_col not in df.columns:
        return pd.DataFrame(columns=["h3_index"] + H3_DEST_FEATURE_COLUMNS)

    df = df.dropna(subset=[level_col]).copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["h3_index"] + H3_DEST_FEATURE_COLUMNS)

    if "number_employees" not in df.columns:
        df["number_employees"] = 0.0
    if "ovgk" not in df.columns:
        df["ovgk"] = "none"
    if "offers_education_secondary" not in df.columns:
        df["offers_education_secondary"] = False
    if "offers_shop" not in df.columns:
        df["offers_shop"] = False
    if "offers_leisure" not in df.columns:
        df["offers_leisure"] = False

    grouped = df.groupby(level_col)

    out = pd.DataFrame({"h3_index": grouped.size().index.astype(str)})
    out = out.set_index("h3_index")
    out["num_statent"] = grouped.size().astype(float)
    out["employees"] = grouped["number_employees"].sum(min_count=1).fillna(0.0).astype(float)
    out["education"] = grouped["offers_education_secondary"].sum(min_count=1).fillna(0.0).astype(float)
    out["shop"] = grouped["offers_shop"].sum(min_count=1).fillna(0.0).astype(float)
    out["leisure"] = grouped["offers_leisure"].sum(min_count=1).fillna(0.0).astype(float)

    if "municipality_type" in df.columns:
        muni = df[[level_col, "municipality_type"]].copy()
        muni["municipality_type"] = muni["municipality_type"].astype(str).str.lower().fillna("")
        out["urban_core"] = muni[muni["municipality_type"].str.contains("core")].groupby(level_col).size().reindex(out.index, fill_value=0).astype(float)
        out["urban"] = muni[muni["municipality_type"].str.contains("urban")].groupby(level_col).size().reindex(out.index, fill_value=0).astype(float)
    else:
        out["urban_core"] = 0.0
        out["urban"] = 0.0

    ovgk = df[[level_col, "ovgk"]].copy()
    ovgk_counts = ovgk.groupby([level_col, "ovgk"]).size().unstack(fill_value=0)
    for category in OVGK_CATEGORIES:
        col = f"ovgk_share_{category.lower()}"
        num = ovgk_counts[category] if category in ovgk_counts.columns else 0.0
        out[col] = (num / out["num_statent"]).astype(float)

    for col in H3_DEST_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0

    return out.reset_index()


def build_h3_tree(merged_level_geoms):
    logger.info("Building H3 hierarchy tree from merged level geometries...")
    level0_cells = set(merged_level_geoms.get("level_0", pd.DataFrame(columns=["h3_index"]))["h3_index"].tolist())
    level1_cells = merged_level_geoms.get("level_1", pd.DataFrame(columns=["h3_index"]))["h3_index"].tolist()
    level2_cells = merged_level_geoms.get("level_2", pd.DataFrame(columns=["h3_index"]))["h3_index"].tolist()

    tree = {l0: {} for l0 in level0_cells}
    logger.info("\t Adding level 1 cells to tree...")
    for l1 in level1_cells:
        try:
            l0 = h3.cell_to_parent(l1, H3_LEVELS[0])
        except Exception:
            continue
        tree.setdefault(l0, {})
        tree[l0].setdefault(l1, [])
    
    logger.info("\t Adding level 2 cells to tree...")
    for l2 in level2_cells:
        try:
            l1 = h3.cell_to_parent(l2, H3_LEVELS[1])
            l0 = h3.cell_to_parent(l2, H3_LEVELS[0])
        except Exception:
            continue
        tree.setdefault(l0, {})
        tree[l0].setdefault(l1, []).append(l2)

    for l0 in tree:
        for l1 in tree[l0]:
            tree[l0][l1] = sorted(set(tree[l0][l1]))

    return tree

def to_h3(df:gpd.GeoDataFrame, resolution:int=7, geometry_col:str='geometry') -> tuple:
   
    # Create a copy of the input DataFrame to avoid modifying the original
    df_copy = df[[geometry_col]].copy()

    # check that geometries are points
    if not all(df_copy.geometry.geom_type == 'Point'):
        raise ValueError("All geometries must be of type 'Point' to convert to H3.")

    # convert to WGS84 coordinate system if not already in that CRS
    if df_copy.crs is not None and df_copy.crs.to_string() != 'EPSG:4326':
        df_copy = df_copy.to_crs(epsg=4326)
    
    # Calculate H3 index for each row
    df_copy['h3_index'] = df_copy.geometry.apply(lambda geom: h3.latlng_to_cell(geom.y, geom.x, resolution))
    
    # Get unique hex indices
    unique_hex = set(df_copy['h3_index'].unique())

    return df_copy['h3_index'].values, unique_hex

def compute_level(level, df, geom_col):
    indices, unique_hex = to_h3(df, resolution=level, geometry_col=geom_col)
    return level, indices, unique_hex
    
def to_geo_levels_parallel(df: gpd.GeoDataFrame, geometry_col: str = 'geometry', levels=H3_LEVELS) -> tuple[gpd.GeoDataFrame, dict]:
    df_copy = df[[geometry_col]].copy()
    level_cols = [f'level_{i}' for i in range(len(levels))]
    df_copy[level_cols] = None
    level_unique_hex = {}

    # do not process infinities, nans, or nulls
    sel = (df_copy[geometry_col].x.notnull() & (~df_copy[geometry_col].x.isin([np.inf, -np.inf])))
    
    # Run in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_level, level, df_copy[sel], geometry_col) for level in levels]
        for future in as_completed(futures):
            level, indices, unique_hex = future.result()
            col = f'level_{levels.index(level)}'
            df_copy.loc[sel, col] = indices
            level_unique_hex[col] = unique_hex

    return (df_copy[level_cols], level_unique_hex)

def within_ch(context, df, cols1=["origin_x", "origin_y"], cols2=None):
    df_switzerland = context.stage("data.spatial.swiss_border")
    ch_polygon = df_switzerland.buffer(0).iloc[0] 
    inside_ch = vectorized.contains(ch_polygon, df[cols1[0]].values, df[cols1[1]].values)
    if cols2 is not None:
        destination_inside_ch = vectorized.contains(ch_polygon, df[cols2[0]].values, df[cols2[1]].values)
        inside_ch = inside_ch & destination_inside_ch
    return inside_ch

def configure(context):
    context.stage("synthesis.population.destinations")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("synthesis.population.spatial.primary.locations")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("data.spatial.swiss_border")

def execute(context):
    logger.info("H3: Processing spatial data and converting to H3 indices...")
    data_collections = {}
    level_unique_hex_list = []  # To collect all level_unique_hex for merging later
    
    # Destination/company data (exclude non-offering rows such as remote work locations)
    logger.info("H3: \t Processing destination companies data...")
    destinations = context.stage("synthesis.population.destinations").copy()
    offer_cols = [f"offers_{p}" for p in ["shop", "leisure", "other", "work_secondary", "education_secondary", "home_secondary"] if f"offers_{p}" in destinations.columns]
    if len(offer_cols) == 0:
        raise RuntimeError("No offers_* columns found in synthesis.population.destinations")

    destinations = destinations[destinations[offer_cols].fillna(False).any(axis=1)].reset_index(drop=True)
    destinations_attrs_cols = ["destination_id", "number_employees", "ovgk", "municipality_type"] + offer_cols
    destinations_attrs_cols = [c for c in destinations_attrs_cols if c in destinations.columns]
    destinations_attributes = destinations[destinations_attrs_cols].copy()

    destinations = gpd.GeoDataFrame(destinations[["destination_id", "geometry"]], geometry="geometry", crs=getattr(destinations, "crs", None) or "EPSG:2056")

    destinations_levels, destinations_unique_hex = to_geo_levels_parallel(destinations, geometry_col="geometry")
    destinations_levels = destinations[["destination_id"]].join(destinations_levels)
    data_collections["destinations"] = destinations_levels

    destinations_with_levels = destinations_levels.merge(destinations_attributes, on="destination_id", how="left")

    level_unique_hex_list.append(destinations_unique_hex)
    del destinations

    # Microcensus trips data
    logger.info("H3: \t Processing Microcensus trips data...")
    mz_trips,_ = context.stage("data.microcensus.trips")
    mz_trips = mz_trips[["person_id","trip_id","origin_x","origin_y","destination_x", "destination_y"]]
    inside_ch = within_ch(context, mz_trips, cols1=["origin_x", "origin_y"], cols2=["destination_x", "destination_y"])
    mz_trips = mz_trips[inside_ch].reset_index(drop=True)
    mz_trips["origin_geometry"] = gpd.points_from_xy(mz_trips.origin_x, mz_trips.origin_y)
    mz_trips["destination_geometry"] = gpd.points_from_xy(mz_trips.destination_x, mz_trips.destination_y)
    mz_trips_origin_levels, mz_trips_origin_unique_hex = to_geo_levels_parallel(gpd.GeoDataFrame(mz_trips, 
                                                                                                  geometry="origin_geometry", 
                                                                                                  crs="EPSG:2056"), 
                                                                                 geometry_col="origin_geometry")
    mz_trips_origin_levels.columns = ["origin_"+col for col in mz_trips_origin_levels.columns]
    mz_trips_destination_levels, mz_trips_destination_unique_hex = to_geo_levels_parallel(gpd.GeoDataFrame(mz_trips, 
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

    mz_persons_home_levels, mz_persons_home_unique_hex = to_geo_levels_parallel(gpd.GeoDataFrame(mz_persons, 
                                                                                                 geometry="home_geometry", 
                                                                                                 crs="EPSG:2056"),
                                                                                geometry_col="home_geometry")
    mz_persons_home_levels.columns = ["home_"+col for col in mz_persons_home_levels.columns]
    mz_persons_work_levels, mz_persons_work_unique_hex = to_geo_levels_parallel(gpd.GeoDataFrame(mz_persons, 
                                                                                                 geometry="work_geometry", 
                                                                                                 crs="EPSG:2056"),
                                                                                geometry_col="work_geometry")
    mz_persons_work_levels.columns = ["work_"+col for col in mz_persons_work_levels.columns]
    data_collections["microcensus_persons"] = mz_persons[["person_id"]].join(mz_persons_home_levels).join(mz_persons_work_levels)
    level_unique_hex_list.extend([mz_persons_home_unique_hex, mz_persons_work_unique_hex])
    del mz_persons

    # synthetic population
    logger.info("H3: \t Processing synthetic population data...")
    df_work, df_education = context.stage("synthesis.population.spatial.primary.locations")
    df_homes = context.stage("synthesis.population.spatial.home.locations")
    df_work = gpd.GeoDataFrame(df_work, geometry=gpd.points_from_xy(df_work["geometry"].x, df_work["geometry"].y), crs="EPSG:2056")
    df_education = gpd.GeoDataFrame(df_education, geometry=gpd.points_from_xy(df_education["geometry"].x, df_education["geometry"].y), crs="EPSG:2056")
    df_homes = gpd.GeoDataFrame(df_homes, geometry="geometry", crs="EPSG:2056")
    work_levels, work_unique_hex = to_geo_levels_parallel(df_work, geometry_col="geometry")
    education_levels, education_unique_hex = to_geo_levels_parallel(df_education, geometry_col="geometry")
    home_levels, home_unique_hex = to_geo_levels_parallel(df_homes, geometry_col="geometry")
    data_collections["synthetic_population_work"] = work_levels
    data_collections["synthetic_population_education"] = education_levels
    data_collections["synthetic_population_home"] = home_levels
    level_unique_hex_list.extend([work_unique_hex, education_unique_hex, home_unique_hex])
    del df_work, df_education, df_homes

    # Merge level geometries across all datasets
    logger.info("H3: \t Merging level geometries across all datasets...")
    swiss_border = context.stage("data.spatial.swiss_border").to_crs("EPSG:2056").unary_union
    merged_level_geoms = {}
    for i in range(len(H3_LEVELS)):
        all_hex = set()
        for level_unique_hex in level_unique_hex_list:
            if f'level_{i}' in level_unique_hex:
                all_hex.update(level_unique_hex[f'level_{i}'])
        polygons = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h)]) for h in all_hex]
        merged_gdf = gpd.GeoDataFrame({"h3_index": list(all_hex)}, geometry=polygons, crs="EPSG:4326")
        # We need to merge to EPSG:2056 for compatibility with our pipeline later on
        merged_gdf = merged_gdf.to_crs("EPSG:2056")

        merged_gdf["centroid"] = merged_gdf.geometry.centroid
        intersection_area = merged_gdf.geometry.intersection(swiss_border).area
        total_area = merged_gdf.geometry.area
        merged_gdf["outside_fraction"] = np.clip(1 - (intersection_area / total_area), 0, 1)

        # Attach destination-derived H3 features so downstream models can consume one centralized source.
        level_col = f"level_{i}"
        feature_df = _aggregate_destination_features_by_level(destinations_with_levels, level_col)
        merged_gdf = merged_gdf.merge(feature_df, on="h3_index", how="left")
        for col in H3_DEST_FEATURE_COLUMNS:
            if col not in merged_gdf.columns:
                merged_gdf[col] = 0.0
        merged_gdf[H3_DEST_FEATURE_COLUMNS] = merged_gdf[H3_DEST_FEATURE_COLUMNS].fillna(0.0)

        merged_level_geoms[level_col] = merged_gdf

    h3_tree = build_h3_tree(merged_level_geoms)

    return (data_collections, merged_level_geoms, h3_tree)








# def plot_geom(levels):
#     # --- Plotting ---
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     for i, gdf_lvl in enumerate(levels.values()):
#         gdf_lvl.plot(ax=axes[i], edgecolor="black", facecolor="none")
#         axes[i].set_title(f"Level {i}")

#     for ax in axes:
#         ax.set_axis_off()

#     plt.tight_layout()
#     plt.show()