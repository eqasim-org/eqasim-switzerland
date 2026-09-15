import logging
import re

import geopandas as gpd
import pandas as pd
import shapely
import shapely.ops

import data.spatial.utils
from data.spatial.zones import impute
import data.utils

logger = logging.getLogger("synpp")


def configure(context):
    context.config("data_path")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.stage("data.spatial.nuts")
    context.stage("data.spatial.postal_codes")

    context.config("include_international_organizations", default = False)
    if context.config("include_international_organizations"):
        context.stage("data.statent.international_organizations")

def execute(context):
    data_path = context.config("data_path")

    df = pd.DataFrame(pd.read_csv(
        "%s/statent/250221_STATENT_2022_LOC_17042025.csv" % data_path,
        encoding = "latin1", sep = ","))

    df = pd.DataFrame(df[["METER_X", "METER_Y", "NOGA08_CD", "EMPTOT", "ANONYM_LOCAL_ID"]])
    df.columns = ["x", "y", "noga", "number_employees", "anonym_local_id"]

    df["noga"] = df["noga"].astype(str)

    # Canonical id from BFS's stable per-establishment id, base62-encoded.
    # Keep in sync with eqasim-france's data/locations_CH/statent/statent.py.
    df["enterprise_id"] = "CH_STATENT_" + df["anonym_local_id"].astype(int).apply(data.utils.to_base62)
    del df["anonym_local_id"]

    df.loc[df["noga"].str.startswith("851"), "education_type"] = "kindergarten"
    df.loc[df["noga"].str.startswith("852"), "education_type"] = "primary"
    df.loc[df["noga"].str.startswith("853"), "education_type"] = "secondary"
    df.loc[df["noga"].str.startswith("854"), "education_type"] = "tertiary"
    df["education_type"] = df["education_type"].astype("category")

    # For now we don't do anything with the NOGA category.
    # (but need to do later for the education locations)

    if context.config("include_international_organizations"):
        df = pd.concat([df, get_international_organizations(context)], ignore_index = True)

    # Impute zones
    df_zones = context.stage("data.spatial.zones")
    df_quarters = context.stage("data.spatial.quarters")
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_nuts = context.stage("data.spatial.nuts")
    df_postal_codes = context.stage("data.spatial.postal_codes")

    df_spatial = pd.DataFrame(df[["enterprise_id", "x", "y"]])

    df_spatial = data.spatial.utils.to_gpd(
        context, df_spatial, "x", "y", 
        coord_type="enterprise")    
    df_spatial = df_spatial.drop(["x", "y"], axis=1)

    columns = ["enterprise_id"]

    # impute municipalities
    df_spatial = data.spatial.utils.impute(
        context, 
        df_spatial, df_municipalities, 
        "enterprise_id", "municipality_id",
        zone_type="municipality", point_type="enterprise")
    columns.extend(["municipality_id"])

    # impute quarters
    df_spatial = data.spatial.utils.impute(
        context, 
        df_spatial, df_quarters, 
        "enterprise_id", "quarter_id", 
        fix_by_distance = False,
        zone_type="quarter", point_type="enterprise")
    columns.extend(["quarter_id"])

    # impute NUTS
    df_nuts = df_nuts[df_nuts["nuts_id"].str.contains("CH")].reset_index(drop=True)
    for level in df_nuts["nuts_level"].unique():
        f_level = df_nuts["nuts_level"] == level
        df_spatial = (data.spatial.utils.impute(
            context,
            df_spatial, df_nuts[f_level].reset_index(drop=True),
            "enterprise_id", "nuts_id", 
            fix_by_distance=False,
            zone_type="NUTS", point_type="enterprise"
            )
        .rename({"nuts_id": ("nuts_id_level_" + str(level))}, axis=1)
        )
        columns.extend([("nuts_id_level_" + str(level))])

    # impute postal codes
    df_spatial = data.spatial.utils.impute(
        context, 
        df_spatial, df_postal_codes, 
        "enterprise_id", "postal_code", 
        fix_by_distance=False,
        zone_type="postal code", point_type="enterprise"
        )
    columns.extend(["postal_code"])

    # clean up columns
    df_spatial = df_spatial[columns]

    # impute zones
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
# and bodies" - the closest real NOGA code to what these entries are.
INTERNATIONAL_ORGANIZATION_NOGA = "990000"

# Organizations below this staff figure (or with no staff figure at all) are
# excluded - see get_international_organizations. Also keeps out entries
# whose staff estimate is too small/uncertain to be worth the risk of
# double-counting against real STATENT establishments.
MIN_STATENT_STAFF = 100


def get_international_organizations(context):
    """
    STATENT (Swiss business census) does not cover organizations with
    extraterritorial/diplomatic status - see data.statent.international_organizations
    for why. This builds STATENT-shaped rows (same x/y/noga/number_employees/
    enterprise_id/education_type columns as the main STATENT extract) for the
    subset of those organizations that have known coordinates and a staff
    figure of at least MIN_STATENT_STAFF, so they can be appended to the
    STATENT data before spatial imputation runs.
    """
    df = context.stage("data.statent.international_organizations")
    df = df[
        df["lat"].notna() & df["lon"].notna()
        & df["staff"].notna() & (df["staff"] >= MIN_STATENT_STAFF)
        & ~df["exclude_from_statent"]
    ].reset_index(drop = True)

    # Reproject WGS84 lat/lon to the same CH1903+/LV95 (epsg:2056) meter
    # coordinates STATENT's METER_X/METER_Y are already in.
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
    A few of these organizations' hand-picked coordinates genuinely sit right
    on the France-Switzerland border (e.g. CERN's campus straddles it) and
    can end up just outside Swiss territory. execute()'s zone-imputation
    chain has no fallback for that: a point outside every real municipality
    resolves to the pipeline's synthetic "external population" region
    (a negative municipality_id - see data.spatial.utils.keep_one_zone_per_point),
    which was deliberately excluded from data.spatial.zones' own catalog, so
    it can never resolve a zone_id and trips execute()'s assertion.

    Snap any such point onto the nearest edge of the real Swiss municipality
    union instead, so every international organization is guaranteed to land
    inside an actual Swiss municipality regardless of how approximate its
    source coordinate was.
    """
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_municipalities = df_municipalities[df_municipalities["municipality_id"] >= 0]
    switzerland = shapely.ops.unary_union(df_municipalities.geometry.values)

    points = gpd.points_from_xy(x, y)
    outside = ~shapely.contains(switzerland, points)

    if outside.any():
        logger.warning(
            "%d international organization coordinate(s) fell outside Switzerland "
            "and were snapped to the nearest point inside it.",
            outside.sum(),
        )
        # Snapping onto switzerland's own boundary would land exactly on the
        # edge, and polygon.contains() is strict about boundary points (it
        # would report the snapped point as still outside). Snap onto a
        # slightly eroded copy instead, so the result sits a few meters
        # inside the real border.
        interior = switzerland.buffer(-10)
        snapped = [shapely.ops.nearest_points(interior, p)[0] for p in points[outside]]
        x = x.copy()
        y = y.copy()
        x[outside] = [p.x for p in snapped]
        y[outside] = [p.y for p in snapped]

    return x, y
