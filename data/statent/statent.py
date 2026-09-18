import logging
import os
import re

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.ops

import data.spatial.utils
from data.spatial.zones import impute

logger = logging.getLogger("synpp")


def configure(context):
    context.config("data_path")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.stage("data.spatial.nuts")
    context.stage("data.spatial.postal_codes")
    context.stage("data.spatial.cantons")

    context.config("include_international_organizations", default = False)
    if context.config("include_international_organizations"):
        context.stage("data.statent.international_organizations")


def load_raw_statent(context):
    """The raw STATENT extract, before international-organization rows are
    appended. Factored out of execute() so
    find_statent_matches_for_international_organizations can reuse it.
    """
    data_path = context.config("data_path")

    df = pd.DataFrame(pd.read_csv(
        "%s/statent/250221_STATENT_2022_LOC_17042025.csv" % data_path,
        encoding = "latin1", sep = ","))

    df = pd.DataFrame(df[["METER_X", "METER_Y", "NOGA08_CD", "EMPTOT", "ANONYM_LOCAL_ID"]])
    df.columns = ["x", "y", "noga", "number_employees", "anonym_local_id"]

    df["noga"] = df["noga"].astype(str)

    # Canonical id from BFS's stable per-establishment id, base62-encoded.
    df["enterprise_id"] = "CH_STATENT_" + df["anonym_local_id"].astype(int).astype(str)#.apply(data.utils.to_base62)
    del df["anonym_local_id"]

    df.loc[df["noga"].str.startswith("851"), "education_type"] = "kindergarten"
    df.loc[df["noga"].str.startswith("852"), "education_type"] = "primary"
    df.loc[df["noga"].str.startswith("853"), "education_type"] = "secondary"
    df.loc[df["noga"].str.startswith("854"), "education_type"] = "tertiary"
    df["education_type"] = df["education_type"].astype("category")

    # For now we don't do anything with the NOGA category.
    return df


def execute(context):
    df = load_raw_statent(context)

    if context.config("include_international_organizations"):
        df = pd.concat([df, get_international_organizations(context)], ignore_index = True)

    # Impute zones
    df_zones          = context.stage("data.spatial.zones")
    df_quarters       = context.stage("data.spatial.quarters")
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_nuts           = context.stage("data.spatial.nuts")
    df_postal_codes   = context.stage("data.spatial.postal_codes")
    df_cantons        = context.stage("data.spatial.cantons")

    df_spatial = pd.DataFrame(df[["enterprise_id", "x", "y"]])

    df_spatial = data.spatial.utils.to_gpd(
        context, df_spatial, "x", "y", 
        coord_type="enterprise")    
    df_spatial = df_spatial.drop(["x", "y"], axis = 1)

    columns = ["enterprise_id"]

    # Impute municipalities
    df_spatial = data.spatial.utils.impute(
        context, 
        df_spatial, df_municipalities, 
        "enterprise_id", "municipality_id",
        zone_type = "municipality", point_type = "enterprise")
    columns.extend(["municipality_id"])

    # Impute cantons
    df_spatial = data.spatial.utils.impute(
        context,
        df_spatial, df_cantons,
        "enterprise_id", "canton_name",
        zone_type = "canton", point_type = "enterprise"
    )
    columns.extend(["canton_name"])

    # Impute quarters
    df_spatial = data.spatial.utils.impute(
        context, 
        df_spatial, df_quarters, 
        "enterprise_id", "quarter_id", 
        fix_by_distance = False,
        zone_type = "quarter", point_type = "enterprise")
    columns.extend(["quarter_id"])

    # Impute NUTS
    df_nuts = df_nuts[df_nuts["nuts_id"].str.contains("CH")].reset_index(drop=True)

    for level in df_nuts["nuts_level"].unique():
        f_level    = df_nuts["nuts_level"] == level
        df_spatial = (data.spatial.utils.impute(
            context,
            df_spatial, df_nuts[f_level].reset_index(drop=True),
            "enterprise_id", "nuts_id", 
            fix_by_distance = False,
            zone_type = "NUTS", point_type = "enterprise"
            )
        .rename({"nuts_id": ("nuts_id_level_" + str(level))}, axis=1)
        )
        columns.extend([("nuts_id_level_" + str(level))])

    # Impute postal codes
    df_spatial = data.spatial.utils.impute(
        context, 
        df_spatial, df_postal_codes, 
        "enterprise_id", "postal_code", 
        fix_by_distance = False,
        zone_type = "postal code", point_type = "enterprise"
        )
    columns.extend(["postal_code"])

    # clean up columns
    df_spatial = df_spatial[columns]

    # Impute zones
    df2 = df_spatial.drop("quarter_id", axis=1)
    df2 = impute(df2, df_zones)
    df2.rename(columns={"zone_id": "zone_municipality_id"}, inplace=True)
    df2 = df2[["enterprise_id", "zone_municipality_id"]]
    df_spatial = impute(df_spatial, df_zones)

    df_spatial = df_spatial.merge(
        df2,   
        on="enterprise_id",         
        how="left"       
    )
    assert(len(df) == len(df_spatial))
    assert(len(df_spatial) == len(df_spatial["zone_id"].dropna()))

    columns.extend(["zone_id"])
    columns.extend(["zone_municipality_id"])

    df = pd.merge(
        df, df_spatial[columns],
        on = "enterprise_id"
    )
    df["zone_id"] = df["zone_id"].astype(int)
    df["zone_municipality_id"] = df["zone_municipality_id"].astype(int)
    return df


# NOGA section U / class 99 = "Activities of extraterritorial organisations
# and bodies" - closest real NOGA code for these entries.
INTERNATIONAL_ORGANIZATION_NOGA = "990000"

# Organizations below this staff figure are excluded - see
# get_international_organizations.
MIN_STATENT_STAFF = 100


def get_international_organizations(context):
    """
    Builds STATENT-shaped rows for international organizations not covered
    by STATENT itself. Excludes organizations with a plausible STATENT
    match (see find_statent_matches_for_international_organizations).
    """
    df = context.stage("data.statent.international_organizations")
    df = df[
        df["lat"].notna() & df["lon"].notna()
        & df["staff"].notna() & (df["staff"] >= MIN_STATENT_STAFF)
    ].reset_index(drop = True)

    matches = find_statent_matches_for_international_organizations(context)
    if len(matches) > 0:
        already_counted = matches["name"].unique()
        df = df[~df["name"].isin(already_counted)].reset_index(drop = True)

    # Reproject WGS84 lat/lon to the same epsg:2056 coordinates STATENT uses.
    df_gpd = data.spatial.utils.to_gpd(
        context, df.rename(columns = {"lon": "x", "lat": "y"}), "x", "y",
        crs = "epsg:4326", coord_type = "international organization")

    x, y = snap_to_switzerland(context, df_gpd.geometry.x.values, df_gpd.geometry.y.values)

    slug = df["name"].str.upper().apply(lambda name: re.sub(r"[^A-Z0-9]+", "_", name).strip("_"))

    return pd.DataFrame({
        "x": x,
        "y": y,
        "noga": INTERNATIONAL_ORGANIZATION_NOGA,
        "number_employees": df["staff"],
        "enterprise_id": "CH_INTORG_" + slug,
        "education_type": pd.NA,
    })


def snap_to_switzerland(context, x, y):
    """
    A few organizations' coordinates sit right on the border (e.g. CERN) and
    can fall just outside Switzerland, which breaks execute()'s zone
    imputation. Snaps any such point onto the nearest municipality edge.
    """
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_municipalities = df_municipalities[df_municipalities["municipality_id"] >= 0]
    switzerland       = shapely.ops.unary_union(df_municipalities.geometry.values)

    points  = gpd.points_from_xy(x, y)
    outside = ~shapely.contains(switzerland, points)

    if outside.any():
        logger.warning(
            "%d international organization coordinate(s) fell outside Switzerland "
            "and were snapped to the nearest point inside it.",
            outside.sum(),
        )
        interior = switzerland.buffer(-10)
        snapped  = [shapely.ops.nearest_points(interior, p)[0] for p in points[outside]]
        x = x.copy()
        y = y.copy()
        x[outside] = [p.x for p in snapped]
        y[outside] = [p.y for p in snapped]

    return x, y


MATCH_MAX_DISTANCE_M = 30

MATCH_MIN_EMPLOYEE_RATIO = 0.2
MATCH_MAX_EMPLOYEE_RATIO = 3.0

EXCLUDED_SERVICE_NOGA_PREFIXES = (
    "56",   # food and beverage service activities (catering, cafeterias)
    "811",  # combined facilities support activities
    "812",  # cleaning activities
    "80",   # security and investigation activities
    "68",   # real estate activities (property/facility management)
    "78",   # employment activities (staffing/temp agencies)
    "82",   # office administrative/support and other business support activities
    "96",   # other personal service activities
)

# Free, no-key WMTS basemap, same as data.statent.international_organizations'.
_TILE_URL = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"

# The review map/CSV is restricted to these cantons to keep it legible - see
# build_statent_duplicate_check_outputs.
REVIEW_CANTON_NAMES_EN = ("geneva", "vaud")


def find_statent_matches_for_international_organizations(context):
    """
    Checks whether a STATENT establishment already covers an international
    organization's jobs (distance/NOGA/employee-ratio criteria above), so
    get_international_organizations can exclude it from the STATENT merge.
    Also writes a CSV/map for review - see build_statent_duplicate_check_outputs.
    """
    df_statent  = load_raw_statent(context)
    gdf_statent = data.spatial.utils.to_gpd(
        context, df_statent, "x", "y", coord_type = "enterprise")

    df_orgs = context.stage("data.statent.international_organizations")
    df_orgs = df_orgs[
        df_orgs["lat"].notna() & df_orgs["lon"].notna()
        & df_orgs["staff"].notna() & (df_orgs["staff"] >= MIN_STATENT_STAFF)
    ].reset_index(drop = True)

    gdf_orgs = data.spatial.utils.to_gpd(
        context, df_orgs.rename(columns = {"lon": "x", "lat": "y"}), "x", "y",
        crs = "epsg:4326", coord_type = "international organization")
    # Keep the original WGS84 lat/lon too, for the folium map markers.
    gdf_orgs["org_lat"] = df_orgs["lat"].values
    gdf_orgs["org_lon"] = df_orgs["lon"].values

    # "dwithin" (not sjoin_nearest) gets every STATENT establishment within
    # range, not just the single closest one.
    candidates = gpd.sjoin(
        gdf_orgs, gdf_statent, how = "inner",
        predicate = "dwithin", distance = MATCH_MAX_DISTANCE_M,
    )
    right_geometry = gpd.GeoSeries(
        gdf_statent.geometry.values[candidates["index_right"].values],
        index = candidates.index, crs = gdf_statent.crs,
    )
    candidates["distance_m"] = candidates.geometry.distance(right_geometry)

    is_service_noga = candidates["noga"].str.startswith(EXCLUDED_SERVICE_NOGA_PREFIXES)
    ratio = candidates["number_employees"] / candidates["staff"]
    is_plausible_size = (ratio >= MATCH_MIN_EMPLOYEE_RATIO) & (ratio <= MATCH_MAX_EMPLOYEE_RATIO)
    candidates["is_plausible_match"] = ~is_service_noga & is_plausible_size

    matches = candidates[candidates["is_plausible_match"]]
    matches = matches[[
        "name", "staff", "enterprise_id", "number_employees", "noga", "distance_m",
    ]].sort_values(["name", "distance_m"]).reset_index(drop = True)

    if len(matches) > 0:
        logger.warning(
            "%d international organization(s) have a plausible STATENT match "
            "(same site, comparable headcount) and may already be counted by "
            "STATENT: %s",
            matches["name"].nunique(), matches["name"].unique().tolist(),
        )

    kept_mask = ~gdf_orgs["name"].isin(matches["name"].unique())
    build_statent_duplicate_check_outputs(context, gdf_orgs, candidates, kept_mask)

    return matches


def _review_cantons(context):
    """The REVIEW_CANTON_NAMES_EN rows of data.spatial.cantons."""
    df_cantons = context.stage("data.spatial.cantons")
    return df_cantons[df_cantons["canton_name_en"].isin(REVIEW_CANTON_NAMES_EN)]


def _assign_canton_names(gdf, df_cantons):
    """Per-row canton_name for gdf's points, via point-in-polygon (only a
    couple of cantons - a simple loop beats a spatial join here since gdf's
    index may have duplicate labels from an earlier sjoin)."""
    canton = pd.Series(pd.NA, index = gdf.index, dtype = "object")
    for _, row in df_cantons.iterrows():
        canton[gdf.geometry.within(row.geometry).values] = row["canton_name"]
    return canton


# How far apart (meters) to nudge STATENT markers sharing a coordinate on
# the review map - see _spread_overlapping_points.
MARKER_SPREAD_OFFSET_M = 10

# Marker style for STATENT candidates on the review map - larger/darker for
# an actual plausible duplicate, smaller/lighter for a rejected one.
STATENT_CANDIDATE_RADIUS = 3
STATENT_CANDIDATE_COLOR  = "#9ecae1"
STATENT_MATCH_RADIUS = 7
STATENT_MATCH_COLOR  = "#08306b"


def _spread_overlapping_points(gdf, offset_m = MARKER_SPREAD_OFFSET_M):
    """
    Nudges points sharing the same (rounded) coordinate into a small circle
    around it, so each gets its own visible marker instead of one hiding
    the rest. Purely a display adjustment.
    """
    # Reset the index first: duplicate labels (from an earlier sjoin) make
    # index.get_loc() return a slice/array instead of a plain position.
    gdf = gdf.reset_index(drop = True)
    keys = pd.Series(
        list(zip(gdf.geometry.x.round().astype(int), gdf.geometry.y.round().astype(int))),
        index = gdf.index,
    )

    offset_x = np.zeros(len(gdf))
    offset_y = np.zeros(len(gdf))
    for _, group_index in gdf.groupby(keys).groups.items():
        positions = np.asarray(group_index)
        n = len(positions)
        if n <= 1:
            continue
        angles = 2 * np.pi * np.arange(n) / n
        offset_x[positions] = offset_m * np.cos(angles)
        offset_y[positions] = offset_m * np.sin(angles)

    gdf["geometry"] = gpd.points_from_xy(
        gdf.geometry.x.values + offset_x, gdf.geometry.y.values + offset_y, crs = gdf.crs,
    )
    return gdf


def build_statent_duplicate_check_outputs(context, gdf_orgs, candidates, kept_mask):
    """
    Writes a review CSV and an interactive folium map (STATENT candidates in
    blue, kept organizations green, removed ones red) to this stage's cache
    folder, restricted to REVIEW_CANTON_NAMES_EN for legibility.
    """
    df_cantons = _review_cantons(context)
    region = shapely.ops.unary_union(df_cantons.geometry.values)

    gdf_orgs = gdf_orgs.copy()
    gdf_orgs["kept"] = kept_mask.values
    df_orgs_region = gdf_orgs[gdf_orgs.geometry.within(region)]

    candidates_region = candidates[candidates.geometry.within(region)].copy()
    candidates_region["canton"] = _assign_canton_names(candidates_region, df_cantons)

    output_dir = context.path()
    os.makedirs(output_dir, exist_ok = True)

    csv_path = os.path.join(output_dir, "statent_international_organizations_duplicates.csv")
    candidates_region[[
        "name", "staff", "canton", "enterprise_id", "number_employees", "noga",
        "distance_m", "is_plausible_match",
    ]].sort_values(["name", "distance_m"]).to_csv(csv_path, index = False)

    if len(df_orgs_region) == 0:
        logger.warning("No international organizations found in %s - skipping review map.", REVIEW_CANTON_NAMES_EN)
        return

    center = [df_orgs_region["org_lat"].mean(), df_orgs_region["org_lon"].mean()]
    m = folium.Map(location = center, zoom_start = 11, tiles = None)
    folium.TileLayer(
        tiles = _TILE_URL, attr = "© swisstopo", name = "swisstopo (grayscale)", opacity = 0.6, control = False,
    ).add_to(m)

    # One marker per distinct STATENT establishment.
    statent_points       = candidates_region.drop_duplicates("enterprise_id")
    statent_points       = _spread_overlapping_points(statent_points)
    statent_layer        = folium.FeatureGroup(
        name = "STATENT candidates (dark/large = actual duplicate, light/small = rejected)", show = True,
    )
    statent_points_wgs84 = statent_points.to_crs("epsg:4326")

    org_layer = folium.FeatureGroup(name = "International organizations (green=kept, red=removed)", show = True)
    for _, row in df_orgs_region.iterrows():
        color = "#2ca02c" if row["kept"] else "#d62728"
        status = "kept" if row["kept"] else "removed (plausible STATENT duplicate)"
        folium.CircleMarker(
            location = (row["org_lat"], row["org_lon"]),
            radius = 10, color = color, weight = 1.5, fill = True,
            fill_color = color, fill_opacity = 0.85,
            popup = folium.Popup(
                f"<b>{row['name']}</b><br>Staff: {row['staff']:.0f}<br>Status: {status}",
                max_width = 300,
            ),
            tooltip = f"{row['name']} ({status})",
        ).add_to(org_layer)
    org_layer.add_to(m)

    for (_, row), (_, row_wgs84) in zip(statent_points.iterrows(), statent_points_wgs84.iterrows()):
        is_match = row["is_plausible_match"]
        color = STATENT_MATCH_COLOR if is_match else STATENT_CANDIDATE_COLOR
        radius = STATENT_MATCH_RADIUS if is_match else STATENT_CANDIDATE_RADIUS
        folium.CircleMarker(
            location = (row_wgs84.geometry.y, row_wgs84.geometry.x),
            radius = radius, color = color, weight = 1, fill = True,
            fill_color = color, fill_opacity = 0.85 if is_match else 0.7,
            popup = folium.Popup(
                f"<b>{row['enterprise_id']}</b><br>NOGA: {row['noga']}<br>"
                f"Employees: {row['number_employees']:.0f}<br>"
                f"Distance to {row['name']}: {row['distance_m']:.0f}m<br>"
                f"Plausible match: {is_match}",
                max_width = 300,
            ),
            tooltip = row["enterprise_id"],
        ).add_to(statent_layer)
    statent_layer.add_to(m)

    folium.LayerControl(collapsed = False).add_to(m)
    m.save(os.path.join(output_dir, "statent_international_organizations_duplicates_map.html"))
