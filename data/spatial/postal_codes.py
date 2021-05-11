import geopandas as gpd

def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")

    df = gpd.read_file(
        "%s/postal_codes/PLZO_SHP_LV95/PLZO_PLZ.shp" % data_path,
        encoding = "latin1"
    ).to_crs("epsg:2056")

    df.crs = "epsg:2056"

    df["postal_code"] = df["PLZ"]
    df = df.sort_values(by="postal_code").reset_index()
    df = df[["postal_code", "geometry"]]

    return df
