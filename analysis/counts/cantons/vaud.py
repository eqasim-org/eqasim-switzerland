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
    input_path = os.path.join(data_path, "Lausanne", "TR_ Demande d’informations complémentaires sur les données de comptage routier")  
    output_path = os.path.join(context.path(),"processed_data.gpkg")
    
    # File definitions
    files = {
        "quinquennaux": {
            "data": "master_quinquennaux_2022_indicateurs_comptages.csv",
            "locations": "master_quinquennaux_2022_postes_comptages.csv"
        },
        "lr21": {
            "data": "master_LR21_indicateurs_comptages.csv",
            "locations": "master_LR21_postes_comptages.csv"
        }
    }

    # Column mappings
    data_cols = {
        "rMeasurementLocation": "id",
        "Direction": "direction",
        "Year": "year",
        "DTV": "TJM",
        "DWV": "TJOM"
    }

    location_cols = {
        "MLocNr": "id",
        "X": "x",
        "Y": "y"
    }

    
    def load_counts(data_file, loc_file):
        df = pd.read_csv(os.path.join(input_path, data_file), sep='|', usecols=data_cols.keys()).rename(columns=data_cols)
        
        idx = df.groupby('id').apply(lambda x: len(x)==3)
        idx = idx[idx].reset_index()["id"]    
        df = df[(df["direction"] == 0)&(df["id"].isin(idx))]

        loc = pd.read_csv(os.path.join(input_path, loc_file), sep='|', usecols=location_cols.keys()).rename(columns=location_cols)
        return df.merge(loc, on="id", how="left")

    # Load both datasets
    df1 = load_counts(files["quinquennaux"]["data"], files["quinquennaux"]["locations"])
    df2 = load_counts(files["lr21"]["data"], files["lr21"]["locations"])

    # merge
    df = pd.concat([df1, df2], ignore_index=True)

    # filter to recent years
    df = df[df.year>=2022].reset_index(drop=True)

    # turn into geodf
    geometry = gpd.points_from_xy(df["x"], df["y"])
    df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:2056")

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









