# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:53:54 2025

@author: dabdelkader
"""

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import contextily as ctx

def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"),"traffic_counts"))

def execute(context):
    # Define paths
    data_path = context.config("counts_path")    

    input_path  = os.path.join(data_path,"Aargau","AGIS","avk_vkzsmeas_20250812.csv")
    output_path = os.path.join(context.path(),"processed_data.gpkg")

    # read data
    df = pd.read_csv(input_path, sep=";")

    # process data
    df["flow"] = df["DTV"] # this is already processed to represent an average yearly day
    df['flow24'] = df["DTV24"] # this is raw 24h count
    df["flow_w"] = df["DWV24"] # this is raw 24h count for weekdays
    # df["x2"] = df["MESS_E"] #E-Koordinate (Ost) LV95 des tatsächlichen Standortes der Zählung
    # df["y2"] = df["MESS_N"]
    df["x"] = df["ZST_E"] # E-Koordinate (Ost) LV95 der Zählstelle
    df["y"] = df["ZST_N"]
    df["direction"] = df['R']
    # df["direction2"] = df['Richtung']
    df["objectid"] = df["ZSTID"]#35425+np.arange(len(df))
    
    # filter
    df = df[df["direction"]=="beide Richtungen"] # keep only both directions
    df = df[df.flow.notna()].reset_index(drop=True)
    df = df[df.x.notna() & df.y.notna()]    
    
    # keep only the most recent year
    df = df.groupby("objectid").apply(lambda x: x[x["JAHR"]==x["JAHR"].max()]).reset_index(drop=True)
    assert (len(df)==len(df.objectid.unique())), "there are duplicate objectid, please check!"

    # keep only relevant columns
    df = df[["objectid","x","y","flow","flow_w"]]    

    # converto to geodf
    df["geometry"] = gpd.points_from_xy(df.x, df.y)

    df = gpd.GeoDataFrame(df, 
                        geometry = gpd.points_from_xy(df.x, df.y),
                        crs = "EPSG:2056")

    # plot
    fig, ax = plt.subplots(figsize=(10,10))
    df.plot(ax=ax, color='red', markersize=5)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=df.crs)
    _ = plt.axis("off")
    plt.savefig(os.path.join(context.path(),"fig.png"), dpi=300)
    plt.close()

    # save
    df.to_file(output_path)
    
    return output_path









