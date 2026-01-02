import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import contextily as ctx
from shapely import wkb

def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"),"traffic_counts"))

def execute(context):
    # Define paths
    data_path = context.config("counts_path")    
    input_path = os.path.join(data_path, "Saint-gallen","verkehrszahlung-miv-stadt-stgallen.parquet")  
    locations_data = os.path.join(data_path, "Saint-gallen","detailplane-und-geokoordinaten-miv-zahlstellen-stadt-stgallen.parquet")  
    output_path = os.path.join(context.path(),"processed_data.gpkg")
    
    # Read the files
    df = pd.read_parquet(input_path)
    loc = pd.read_parquet(locations_data)

    # filters
    df["datum"] = pd.to_datetime(df["datum"])
    df = df[df.datum.dt.year.isin([2023,2024])]
    df = df[df.standort.notna()]
    df = df[df.tagestotal.notna()]
    df = df[df.tagestotal>0]

    # keep only two directional data
    df = df[df['ri'].isin(['1','2'])]
    df = df[df.groupby("ort_id").ri.transform("nunique")==2]

    # get the average by roads (two directional)
    df = (df.groupby(["ort_id","ri"])
            .agg({"tagestotal":"mean"})
            .groupby(["ort_id"])
            .agg({"tagestotal":"sum"})
            ).reset_index()

    # build geometry
    loc["standort"] = loc["standort"].apply(wkb.loads)
    loc = gpd.GeoDataFrame(loc,geometry="standort", crs="EPSG:4326")
    loc = loc.to_crs("EPSG:2056")

    df = df.merge(loc[['ort_id','standort']], how="left", on="ort_id")

    df = df.rename(columns={"ort_id":"objectid",
                            "standort":"geometry",
                            "tagestotal":"flow",
                            })
    # convert to geopandas
    df = gpd.GeoDataFrame(df, geometry = "geometry", crs="EPSG:2056")                        

    # plot
    fig, ax = plt.subplots(figsize=(10,10))
    df.plot(ax=ax, color='red', markersize=5)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=df.crs)
    _ = plt.axis("off")
    plt.savefig(os.path.join(context.path(),"fig.png"), dpi=300)
    plt.close()

    # save file    
    df.to_file(output_path)
    
    return output_path