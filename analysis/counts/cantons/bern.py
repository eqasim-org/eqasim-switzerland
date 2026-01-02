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
from shapely import wkb

def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"),"traffic_counts"))

def execute(context):
    # Define paths
    data_path = context.config("counts_path")
    counts_data1  = os.path.join(data_path,"Bern","ksvd","KSVD.gpkg")    
    counts_data2  = os.path.join(data_path,"Bern","ksvd_ksvdvb.parquet")
    output_path = os.path.join(context.path(),"processed_data.gpkg")
    
    # read data
    df1 = gpd.read_file(counts_data1)
    df2 = pd.read_parquet(counts_data2)

    # identify the flow
    df2["flow"] = df2["dtv"]

    # merge the dataframes
    df2["geometry"] = df2["geometry"].apply(wkb.loads)
    df2 = df2[['objectid','flow','geometry']].rename(
            columns = {"geometry":"link_geometry"})

    # turn into geopandas and merge (I need both link geometry and the point geometry)
    df2 = gpd.GeoDataFrame(df2, geometry="link_geometry", crs="EPSG:2056")
    df1 = gpd.GeoDataFrame(df1, geometry="geometry", crs="EPSG:2056")

    df = df1[["geometry","link_blatt"]].sjoin_nearest(df2, how="left")
    df = df.merge(df2[["objectid","link_geometry"]], on="objectid", how="left")    

    # only keep unique objectid
    df = df.drop_duplicates(subset=["objectid"]).reset_index(drop=True)

    # projection (it appears that some points are not exactly on the link, so we project them)
    df = df[["objectid","flow","geometry","link_geometry"]]    
    df["projected_point"] = df.apply(
        lambda row: row.link_geometry.interpolate(
            row.link_geometry.project(row.geometry)
        ),
        axis=1
    )

    # only keep relevant columns
    df = df[["objectid","flow","projected_point"]]
    df = df.rename(columns = {"projected_point":"geometry"})
    df = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:2056")

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









