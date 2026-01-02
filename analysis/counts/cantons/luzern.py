# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:53:54 2025

@author: dabdelkader
"""

import matplotlib.pyplot as plt
import geopandas as gpd
import os
import contextily as ctx

def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"),"traffic_counts"))

def execute(context):
    # Define paths
    data_path = context.config("counts_path")
    # input_path1 = os.path.join(data_path, "Luzern","daten","LICHTSGN_DS_V2_20251006.gpkg")  
    input_path = os.path.join(data_path, "Luzern","daten","VERKZAEL_DS_V2_20251009.gpkg")  
    output_path = os.path.join(context.path(),"processed_data.gpkg")
    
    # Read and process    
    # df1 = gpd.read_file(input_path1)
    df2 = gpd.read_file(input_path)

    df2 = df2[(df2['VERKEHR_TYP']==1)&(df2['DTV'].notna())] # transport privé

    df2 = df2.rename(columns={"NUMMER_VERKZAEHLER":"objectid", 
                            "DTV":"flow"}).reset_index(drop=True)

    df = df2[["objectid","flow","geometry"]]
    
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