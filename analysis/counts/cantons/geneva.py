import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os

def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"),"traffic_counts"))

def execute(context):
    data_path = context.config("counts_path")    
    output_path = os.path.join(context.path(),"processed_data.gpkg")
    geneva_counts_data  = os.path.join(data_path,"Geneva","counts_geneva_2025.csv")

    # read the dataset
    df = pd.read_csv(geneva_counts_data, sep=";", encoding='latin-1')

    # change types
    df.loc[:,["mean_flow_2025","median_flow_2025"]] = df.loc[:,["mean_flow_2025","median_flow_2025"]
                                                             ].applymap(lambda x: x.replace(",",".").replace(" ","") 
                                                                        if isinstance(x,str) else x)

    df = df.astype({"mean_flow_2025":float, "median_flow_2025":float, "denomination":str, "counter_id":str})
    
    # convert the flow into daily flow
    df["mean_flow_2025"] = df["mean_flow_2025"]*24
    df["median_flow_2025"] = df["median_flow_2025"]*24

    # convert to geodataframe
    df["geometry"] = gpd.points_from_xy(df.lon.str.replace(',','.').astype(float), df.lat.str.replace(',','.').astype(float))
    df = gpd.GeoDataFrame(pd.DataFrame(df), crs="EPSG:4326").to_crs("EPSG:2056")

    # keep only bidirectional links
    df["denomination"] = df["denomination"].str.split('.').str[0]
    df["num_directions"] = df.groupby("denomination")["denomination"].transform("count")
    df = df[df.num_directions==2].reset_index(drop=True)

    # build final dataframe (aggregate by denomination, to make it bidirectional)
    df = df[['counter_id', 'denomination', 'mean_flow_2025', 'median_flow_2025', 'geometry']]

    df = df.groupby("denomination").agg({"counter_id": "first",
                                         "denomination": "first",
                                         "mean_flow_2025": "sum",
                                         "median_flow_2025": "sum",
                                         "geometry": "first"
                                        }).reset_index(drop=True)
    
    df["OBJECTID"] = df["counter_id"].astype(str)+'_'+df["denomination"].astype(str)
    df = df.drop(columns=["counter_id", "denomination"])
    df = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:2056")
    
    # save
    df.to_file(output_path)
    
    return output_path