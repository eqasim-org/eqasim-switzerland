import geopandas as gpd

def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")

    df = gpd.read_file(
        "%s/spatial/country/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp" % data_path,
        encoding = "latin1"
    ).to_crs("epsg:2056")

    df.crs = "epsg:2056"

    return df[df["NAME"]=="Schweiz"]["geometry"]
