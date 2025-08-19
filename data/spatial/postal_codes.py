import geopandas as gpd

def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")

    df = gpd.read_file(
        "%s/spatial/postal_codes/AMTOVZ_ZIP.shp" % data_path,
        encoding = "latin1"
    ).to_crs("epsg:2056")
    df.crs = "epsg:2056"

    df["postal_code"] = df["ZIP4"]
    df = df.sort_values(by="postal_code").reset_index()
    df = df[["postal_code", "geometry"]]

    return df
