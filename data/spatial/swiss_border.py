import geopandas as gpd

def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")

    df = gpd.read_file(
        "%s/municipality_borders/gd-b-00.03-875-gg18/ggg_2018-LV95/shp/g1l18.shp" % data_path,
        encoding = "latin1"
    ).to_crs("epsg:2056")

    df.crs = "epsg:2056"

    return df["geometry"]
