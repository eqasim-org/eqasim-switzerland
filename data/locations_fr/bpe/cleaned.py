import numpy as np
import pandas as pd
import data.utils
import geopandas as gpd
import logging

logger = logging.getLogger("synpp")

"""
This stage cleans the enterprise census:
  - Filter out enterprises that do not have a valid municipality or IRIS
  - Assign coordinates randomly to enterprises that do not have coordinates
  - Simplify activity types for all enterprises
"""


def configure(context):
    context.stage("data.locations_fr.bpe.raw")

    context.config("bpe_random_seed", 0)


ACTIVITY_TYPE_MAP = [
    ("A", "other"),         # Police, post office, etc ...
    ("A504", "leisure"),    # Restaurant
    ("B", "shop"),          # Shopping
    ("C", "education"),     # Education
    ("C701","other"),       # CROUS Residence
    ("C702","other"),       # CROUS Restaurant
    ("D", "other"),         # Health
    ("E", "other"),         # Transport
    ("F", "leisure"),       # Sports & Culture
    ("G", "other"),         # Tourism, hotels, etc. (Hôtel = G102)
]



def execute(context):
    df = context.stage("data.locations_fr.bpe.raw")

    # Coordinates are needed early for the canonical id built below.
    df["x"] = df["LAMBERT_X"].astype(str).str.replace(",", ".").astype(float)
    df["y"] = df["LAMBERT_Y"].astype(str).str.replace(",", ".").astype(float)

    # Canonical id from SIRET+coordinates (base62), falling back to
    # IRIS+type+coordinates when SIRET is missing. Keep in sync with
    # eqasim-france's data/bpe/cleaned.py.
    siret  = df["SIRET"].astype(str).str.strip()
    x_key  = df["x"].round().astype("Int64").apply(lambda v: data.utils.to_base62(v) if pd.notna(v) else "")
    y_key  = df["y"].round().astype("Int64").apply(lambda v: data.utils.to_base62(v) if pd.notna(v) else "")
    siret_key = siret.apply(lambda s: data.utils.to_base62(int(s)) if s.isdigit() else "")

    has_siret = siret.str.len() > 0
    df["enterprise_id"] = "FR_BPE_" + siret_key + "_" + x_key + "_" + y_key
    df.loc[~has_siret, "enterprise_id"] = (
        "FR_BPE_NOSIRET_" + df["DCIRIS"] + "_" + df["TYPEQU"] + "_" + x_key + "_" + y_key
    )

    # SIRET+coordinates (or IRIS+type+coordinates) is not always unique: a
    # single address/SIRET can host several distinct BPE equipment records.
    # Disambiguate with a suffix so enterprise_id stays one-to-one with rows.
    duplicate_index = df.groupby("enterprise_id").cumcount()
    df.loc[duplicate_index > 0, "enterprise_id"] += "_" + duplicate_index[duplicate_index > 0].astype(str)

    # Clean activity type
    df["activity_type"] = "other"
    for prefix, activity_type in ACTIVITY_TYPE_MAP:
        df.loc[df["TYPEQU"].str.startswith(prefix), "activity_type"] = activity_type

    df["activity_type"] = df["activity_type"].astype("category")

    #Add 
    df = df.rename(columns={"TYPEQU":"education_type"})
    df["weight"] = df["CAPACITE"].fillna(500)

    # Clean IRIS and commune
    df["iris_id"] = df["DCIRIS"].str.replace("_", "")
    df["iris_id"] = df["iris_id"].str.replace("IND", "")

    df.loc[df["DEPCOM"] == df["iris_id"], "iris_id"] = "undefined"

    df["iris_id"] = df["iris_id"].astype("category")

    if not "undefined" in df["iris_id"].cat.categories:
        df["iris_id"] = df["iris_id"].cat.add_categories("undefined")

    df["commune_id"] = df["DEPCOM"].astype("category")

    logger.info("Found %d/%d (%.2f%%) observations without IRIS" % (
        (df["iris_id"] == "undefined").sum(), len(df), 100 * (df["iris_id"] == "undefined").mean()
    ))

    # Impute missing coordinates for known IRIS
    random = np.random.default_rng(context.config("bpe_random_seed"))

    f_undefined = df["iris_id"] == "undefined"
    f_missing   = df["x"].isna()

    logger.info("Found %d/%d (%.2f%%) observations without coordinate" % (
        ((f_missing & ~f_undefined).sum(), len(df), 100 * (f_missing & ~f_undefined).mean()
    )))

    # In the eqasim-france pipeline, some functions assign random coordinates in IRIS or municipality
    # for locations without coordinates. Here, to keep it simple, we will remove completely those locations.

    df = df[~f_missing & ~f_undefined]
    df["imputed"] = False

    # Consolidate
    assert not df["x"].isna().any()

    # Package up data set
    df = df[["enterprise_id", "activity_type","education_type", "commune_id", "imputed", "x", "y","weight"]]

    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y),crs="EPSG:2154")
    df = df.to_crs("EPSG:2056")

    return df
