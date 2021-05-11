import geopandas as gpd


def configure(context):
    context.config("data_path")


def execute(context):
    # Load data
    data_path = context.config("data_path")

    df = gpd.read_file(
        "%s/municipality_borders/gd-b-00.03-875-gg18/ggg_2018-LV95/shp/g1k18.shp" % data_path,
        encoding="latin1"
    ).to_crs("epsg:2056")

    df.crs = "epsg:2056"

    df = df.rename({"KTNR": "canton_id", "KTNAME": "canton_name"}, axis=1)
    df = df[["canton_id", "canton_name", "geometry"]]

    return df


SP_REGION_1 = [25, 12, 13, 1, 2, 14, 9]
SP_REGION_2 = [21, 26, 15, 16, 22, 11, 24, 3, 6, 7]
SP_REGION_3 = [17, 19, 10, 23, 20, 5, 18, 4, 8]


def impute_sp_region(df):
    assert ("canton_id" in df.columns)
    assert ("sp_region" not in df.columns)

    df["sp_region"] = 0
    df.loc[df["canton_id"].isin(SP_REGION_1), "sp_region"] = 1
    df.loc[df["canton_id"].isin(SP_REGION_2), "sp_region"] = 2
    df.loc[df["canton_id"].isin(SP_REGION_3), "sp_region"] = 3

    # TODO: There are some municipalities that are not included in the shape
    # file above. Hence, they get region 0. Should be fixed in the future.
    # Especially, we need a consistent spatial system. It probably would make
    # more sense to impute the SP region in another way

    # assert(not np.any(df["sp_region"] == 0))
    return df
