import geopandas as gpd

def configure(context):
    context.config("data_path")


def execute(context):
    # Load data
    data_path = context.config("data_path")

    df = gpd.read_file(
        "%s/spatial/districts/swissBOUNDARIES3D_1_5_TLM_BEZIRKSGEBIET.shp" % data_path,
        encoding="latin1"
    ).to_crs("epsg:2056")
    df.crs = "epsg:2056"

    df = df.rename({"BEZIRKSNUM": "district_id", "NAME": "district_name"}, axis=1)
    df = df[["district_id", "district_name", "geometry"]]

    return df
