import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import numpy as np

def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"),"traffic_counts"))

def execute(context):
    data_path = context.config("counts_path")    
    output_path = os.path.join(context.path(),"processed_data.gpkg")
    input_data_path  = os.path.join(data_path,"Annemasse","counts_annemasse_2025")
    
    # Read the data and keep relevant columns
    df = gpd.read_file(input_data_path).to_crs("EPSG:2056")
    df = df[["N_ROUTE","T_COMPT","ID", "TMJA_VL_24", "TMJA_PL_24","geometry"]].copy()

    # define an object ID
    df["OBJECTID"] = df["N_ROUTE"].astype(str)+'_'+df["T_COMPT"].astype(str)+'_'+df["ID"].astype(str)

    # remove nans
    sel = df[["TMJA_VL_24", "TMJA_PL_24","geometry"]].isna().sum(axis=1)==0
    df = df[sel]

    # compute TJM (léger + poinds lourds)
    df["TJM"] = df["TMJA_VL_24"] + df["TMJA_PL_24"]
    df = df.loc[df.TJM>0, ["OBJECTID","TJM","geometry"]].reset_index(drop=True)

    #TODO: this needs to be changes, but for now we take three points per line
    df = sample_points_from_lines(df, 3)

    # finale dataframe
    df = gpd.GeoDataFrame(df.reset_index(drop=True), geometry="geometry", crs="EPSG:2056")
    df.to_file(output_path)

    return output_path







############# helper functions ####################
def sample_points_from_lines(gdf, n_points=3):
    """
    Sample n_points randomly from each LineString in the GeoDataFrame.
    Returns a new GeoDataFrame where each original row is expanded into n_points rows.
    """
    all_geometries = []
    all_indices = []
    
    for idx, row in gdf.iterrows():
        line = row.geometry
        
        # Generate n_points random distances along the line
        distances = np.random.uniform(0, line.length, n_points)
        
        # Get points at those distances
        points = [line.interpolate(dist) for dist in distances]
        
        all_geometries.extend(points)
        all_indices.extend([idx] * n_points)
    
    # Create new GeoDataFrame with sampled points
    result_gdf = gpd.GeoDataFrame(
        geometry=all_geometries,
        index=all_indices,
        crs=gdf.crs
    )
    
    # Add other columns from original dataframe by reindexing
    for col in gdf.columns:
        if col != 'geometry':
            result_gdf[col] = gdf.loc[all_indices, col].values
    
    # Reset index to have a clean integer index
    result_gdf = result_gdf.reset_index(drop=True)
    
    return result_gdf