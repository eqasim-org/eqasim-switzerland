import pandas as pd
import geopandas as gpd


def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")

    df = gpd.read_file(
        "%s/spatial/districts/swissBOUNDARIES3D_1_5_TLM_BEZIRKSGEBIET.shp" % data_path,
        encoding="latin1"
    ).to_crs("epsg:2056")

    df.crs = "epsg:2056"

    df = df.rename({"BEZIRKSNUM": "district_id", "NAME": "district_name"}, axis=1)
    df = df[["district_id", "district_name", "geometry"]]

    # Some cantons do not have districts / Bezirke in the district shapefile.
    # For these, add the canton itself as a district with ID canton_id * 100,
    # i.e. canton_id + "00".
    cantons = gpd.read_file(
        "%s/spatial/canton/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp" % data_path,
        encoding="latin1"
    ).to_crs("epsg:2056")

    cantons.crs = "epsg:2056"

    cantons = cantons.rename({"KANTONSNUM": "canton_id", "NAME": "canton_name"}, axis=1)
    cantons = cantons[["canton_id", "canton_name", "geometry"]]

    df["district_id"] = df["district_id"].astype(int)
    cantons["canton_id"] = cantons["canton_id"].astype(int)

    canton_ids_with_districts = set((df["district_id"] // 100).unique())

    missing_districts = cantons[
        ~cantons["canton_id"].isin(canton_ids_with_districts)
    ].copy()

    missing_districts["district_id"] = missing_districts["canton_id"] * 100
    missing_districts = missing_districts.rename(
        {"canton_name": "district_name"},
        axis=1
    )
    missing_districts = missing_districts[
        ["district_id", "district_name", "geometry"]
    ]

    df = gpd.GeoDataFrame(
        pd.concat([df, missing_districts], ignore_index=True),
        crs="epsg:2056"
    )

    df = df.sort_values("district_id").reset_index(drop=True)

    return df