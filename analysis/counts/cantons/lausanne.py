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
    input_path = os.path.join(data_path, "Lausanne", "downTownLausanne.geojson")  
    output_path = os.path.join(context.path(),"processed_data.gpkg")
    
    # read the file
    df = gpd.read_file(input_file)

    
    return output_path









