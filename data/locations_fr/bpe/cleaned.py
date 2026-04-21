import numpy as np
import shapely.geometry as geo
import data.spatial.utils as spatial_utils
import geopandas as gpd

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

    # Clean IDs
    df["enterprise_id"] = np.arange(len(df))

    # Clean activity type
    df["activity_type"] = "other"
    for prefix, activity_type in ACTIVITY_TYPE_MAP:
        df.loc[df["TYPEQU"].str.startswith(prefix), "activity_type"] = activity_type

    df["activity_type"] = df["activity_type"].astype("category")

    #Add 
    df = df.rename(columns={"TYPEQU":"education_type"})
    df["weight"] = df["CAPACITE"].fillna(500) 

    # Clean coordinates
    df["x"] = df["LAMBERT_X"].astype(str).str.replace(",", ".").astype(float)
    df["y"] = df["LAMBERT_Y"].astype(str).str.replace(",", ".").astype(float)

    # Clean IRIS and commune
    df["iris_id"] = df["DCIRIS"].str.replace("_", "")
    df["iris_id"] = df["iris_id"].str.replace("IND", "")

    df.loc[df["DEPCOM"] == df["iris_id"], "iris_id"] = "undefined"

    df["iris_id"] = df["iris_id"].astype("category")

    if not "undefined" in df["iris_id"].cat.categories:
        df["iris_id"] = df["iris_id"].cat.add_categories("undefined")

    df["commune_id"] = df["DEPCOM"].astype("category")

    print("Found %d/%d (%.2f%%) observations without IRIS" % (
        (df["iris_id"] == "undefined").sum(), len(df), 100 * (df["iris_id"] == "undefined").mean()
    ))

    # Impute missing coordinates for known IRIS
    random = np.random.default_rng(context.config("bpe_random_seed"))

    f_undefined = df["iris_id"] == "undefined"
    f_missing   = df["x"].isna()

    print("Found %d/%d (%.2f%%) observations without coordinate" % (
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
