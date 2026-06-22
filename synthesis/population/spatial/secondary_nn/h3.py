import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon
import logging
import numpy as np
from shapely import vectorized
logger = logging.getLogger("synpp")

"""
This stage processes the spatial data from various sources (Statent, Microcensus trips, and synthetic population locations)
and converts their geometries into H3 hexagonal indices at three resolutions. 
It also merges the unique hexagons across all datasets for each resolution level to create a comprehensive set of 
hexagonal geometries that can be used for further analysis or visualization.
"""

H3_LEVELS = [5, 7, 10]
H3_LEVEL_NAMES = [f"level_{i}" for i in range(len(H3_LEVELS))]
OVGK_CATEGORIES = ['B', 'A', 'C', 'None', 'D']
H3_DEST_FEATURE_COLUMNS = ["num_statent", "employees", "urban_core", "urban", "education", "shop", "leisure", "sport", "gastronomy", "accommodation", "cultural", "ovgk_share_a", "ovgk_share_b", "ovgk_share_c", "ovgk_share_d", "ovgk_share_none"]

def _aggregate_destination_features_by_level(destinations_with_levels, level_col, all_hex):
    assert level_col in destinations_with_levels.columns
    assert destinations_with_levels[level_col].notna().all()
    assert set(destinations_with_levels["ovgk"].unique()) == set(OVGK_CATEGORIES)

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
    assert all(col in ovgk_dummies.columns for col in expected_cols), f"Missing expected OVGK category columns after get_dummies. Expected: {expected_cols}, but got: {ovgk_dummies.columns.tolist()}"
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
    logger.info("Building H3 hierarchy tree from merged level geometries...")
    level0_cells = set(merged_level_geoms.get(H3_LEVEL_NAMES[0], pd.DataFrame(columns=["h3_index"]))["h3_index"].tolist())
    level1_cells = merged_level_geoms.get(H3_LEVEL_NAMES[1], pd.DataFrame(columns=["h3_index"]))["h3_index"].tolist()
    level2_cells = merged_level_geoms.get(H3_LEVEL_NAMES[-1], pd.DataFrame(columns=["h3_index"]))["h3_index"].tolist()

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

def to_geo_levels(df: gpd.GeoDataFrame, geometry_col: str = 'geometry', levels=H3_LEVELS) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Convert point geometries to H3 indices at multiple resolutions.

    Only the finest resolution is computed directly from geo coordinates via latlng_to_cell.
    All coarser levels are then derived using cell_to_parent, which guarantees strict H3
    hierarchy consistency: the same latlng_to_cell call cannot land a point in two different
    parent chains at different resolutions.
    """
    df_copy = df[[geometry_col]].copy()
    level_cols = [f'level_{i}' for i in range(len(levels))]
    df_copy[level_cols] = None
    level_unique_hex = {}

    # do not process infinities, nans, or nulls
    sel = df_copy[geometry_col].x.notnull() & (~df_copy[geometry_col].x.isin([np.inf, -np.inf]))

    # convert to WGS84 once — a single shared conversion avoids cross-resolution floating-point drift
    if df_copy.crs is not None and df_copy.crs.to_string() != 'EPSG:4326':
        df_copy = df_copy.to_crs(epsg=4326)

    # Compute the finest level directly from coordinates
    finest_idx = len(levels) - 1
    finest_col = f'level_{finest_idx}'
    finest_res = levels[finest_idx]
    df_copy.loc[sel, finest_col] = df_copy.loc[sel, geometry_col].apply(
        lambda geom: h3.latlng_to_cell(geom.y, geom.x, finest_res)
    )
    level_unique_hex[finest_col] = set(df_copy.loc[sel, finest_col].dropna().unique())

    # Derive all coarser levels as parents of the finest level — guaranteed consistent by H3
    for i, res in enumerate(levels[:-1]):
        col = f'level_{i}'
        df_copy.loc[sel, col] = df_copy.loc[sel, finest_col].apply(
            lambda l: h3.cell_to_parent(l, res) if pd.notna(l) else None
        )
        level_unique_hex[col] = set(df_copy.loc[sel, col].dropna().unique())

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
    # context.stage("synthesis.population.destinations") # maybe it should be this one instead of destinations_statent?
    context.stage("data.statent.statent")
    context.stage("synthesis.population.destinations_statent")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.spatial.swiss_border")

def execute(context):
    logger.info("H3: Processing spatial data and converting to H3 indices...")
    data_collections = {}
    level_unique_hex_list = []  # To collect all level_unique_hex for merging later
    
    # Destination/company data (exclude non-offering rows such as remote work locations)
    logger.info("H3: \t Processing destination companies data...")
    destinations = context.stage("synthesis.population.destinations_statent").copy()
    offer_cols = [f"offers_{p}" for p in ["shop", "leisure", "other", "work_secondary", "education_secondary", "home_secondary"] if f"offers_{p}" in destinations.columns]    
    assert len(offer_cols) > 0, "No offers_* columns found in synthesis.population.destinations_statent"
    assert destinations[offer_cols].isna().sum().sum() == 0, "Found NaNs in offers_* columns of destinations data, please check the data preparation steps."

    # enrich destinations
    df_statent = context.stage("data.statent.statent")[["enterprise_id", "noga"]]
    df_statent.columns = ["destination_id", "noga"]
    df_statent["offers_cultural"] = df_statent["noga"].str.startswith("912") | df_statent["noga"].str.startswith("914")
    df_statent["offers_sport"] = df_statent["noga"].str.startswith("93")
    df_statent["offers_gastronomy"] = df_statent["noga"].str.startswith("56")
    df_statent["offers_accommodation"] = df_statent["noga"].str.startswith("55")
    additional_offers = ["offers_cultural", "offers_sport", "offers_gastronomy", "offers_accommodation"]
    destinations = destinations.merge(df_statent[["destination_id"] + additional_offers], on="destination_id", how="left")
    destinations[additional_offers] = destinations[additional_offers].fillna(0)

    # keep only rows that have at least one offer (this filter would remove remote work locations for example)
    offer_cols = offer_cols + additional_offers
    destinations = destinations[destinations[offer_cols].any(axis=1)].reset_index(drop=True)

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

    # synthetic population
    # logger.info("H3: \t Processing synthetic population data...")
    # df_work, df_education = context.stage("synthesis.population.spatial.primary.locations")
    # df_homes = context.stage("synthesis.population.spatial.home.locations")
    # df_work = gpd.GeoDataFrame(df_work, geometry=gpd.points_from_xy(df_work["geometry"].x, df_work["geometry"].y), crs="EPSG:2056")
    # df_education = gpd.GeoDataFrame(df_education, geometry=gpd.points_from_xy(df_education["geometry"].x, df_education["geometry"].y), crs="EPSG:2056")
    # df_homes = gpd.GeoDataFrame(df_homes, geometry="geometry", crs="EPSG:2056")
    # work_levels, work_unique_hex = to_geo_levels_parallel(df_work, geometry_col="geometry")
    # education_levels, education_unique_hex = to_geo_levels_parallel(df_education, geometry_col="geometry")
    # home_levels, home_unique_hex = to_geo_levels_parallel(df_homes, geometry_col="geometry")
    # data_collections["synthetic_population_work"] = work_levels
    # data_collections["synthetic_population_education"] = education_levels
    # data_collections["synthetic_population_home"] = home_levels
    # level_unique_hex_list.extend([work_unique_hex, education_unique_hex, home_unique_hex])
    # del df_work, df_education, df_homes

    # Merge level geometries across all datasets
    logger.info("H3: \t Merging level geometries across all datasets...")
    swiss_border = context.stage("data.spatial.swiss_border").to_crs("EPSG:2056").unary_union
    merged_level_geoms = {}
    for i in range(len(H3_LEVELS)):        
        all_hex = set()
        for level_unique_hex in level_unique_hex_list:
            if f'level_{i}' in level_unique_hex:
                all_hex.update(level_unique_hex[f'level_{i}'])
        
        logger.info(f"\t\t Processing level {H3_LEVELS[i]} with {len(all_hex)} unique hexagons...")

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
        feature_df = _aggregate_destination_features_by_level(destinations_with_levels, level_col, all_hex)
        merged_gdf = merged_gdf.merge(feature_df, on="h3_index", how="left")
        assert all(col in merged_gdf.columns for col in H3_DEST_FEATURE_COLUMNS), f"Missing expected destination feature columns in merged_gdf for level {i}. Expected at least: {H3_DEST_FEATURE_COLUMNS}, but got {merged_gdf.columns.tolist()}"
        assert merged_gdf[H3_DEST_FEATURE_COLUMNS].isna().sum().sum() == 0, f"Found NaNs in destination feature columns of merged_gdf for level {i}, please check the merging process. Columns with NaNs: {merged_gdf[H3_DEST_FEATURE_COLUMNS].isna().sum()}"
        merged_gdf[H3_DEST_FEATURE_COLUMNS] = merged_gdf[H3_DEST_FEATURE_COLUMNS]

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