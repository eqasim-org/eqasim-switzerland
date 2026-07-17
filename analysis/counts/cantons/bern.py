import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import contextily as ctx

TO_BE_REMOVED = ["3072-2061"]

def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"),"traffic_counts"))

def execute(context):
    # Define paths
    data_path = context.config("counts_path")
    data_path  = os.path.join(data_path,"Bern","traffic_counts_bern_2026")  
    output_path = os.path.join(context.path(),"processed_data.gpkg")
    
    # find the files
    locations = "Liste_Points_de_comptage.xlsx"
    files = [f for f in os.listdir(data_path) if f.endswith(".xlsx") and f!=locations]
    
    # read file by file and aggregate
    df = []
    for f in files:
        flow, week_day_flow, station_name = process_file(data_path,f)
        if flow is not None and week_day_flow is not None and station_name is not None:
            df.append({"flow":flow,"flow_w":week_day_flow,"objectid":station_name})
    df = pd.DataFrame(df)

    # read stations
    stations = pd.read_excel(os.path.join(data_path,locations), sheet_name=None, skiprows=2)
    stations = stations[list(stations.keys())[0]]
    stations.columns = ["objectid","description","x","y"]

    # merge the data
    df = pd.merge(df, stations[["objectid","x","y"]], on="objectid", how="inner")
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:2056")

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









def process_file(data_path, file_name):
    file_path = os.path.join(data_path,file_name)
    df_temp = pd.read_excel(file_path, sheet_name=None, skiprows=9)
    keys = list(df_temp.keys())

    # assertions (two directions)
    if len(df_temp)!=2:
        return None, None, None

    # process files
    flows = []
    weekday_flows = []
    for key in keys:
        df_temp[key]["Datum"] = pd.to_datetime(df_temp[key]["Datum"], format="%d.%m.%Y")
        dfi = df_temp[key].groupby("Datum").agg({"Total":"sum"}).reset_index()
        # assert whether we have enought data (6 months minimum)
        if len(dfi)<180:
            return None, None, None
        # get day of the week
        dfi["day_of_week"] = dfi["Datum"].dt.day_name()
        # average flow
        avg_flow = dfi.Total.mean()
        # get the average flow per weekday
        dfi = dfi[dfi.day_of_week.isin(["Monday","Tuesday","Wednesday","Thursday","Friday"])]
        avg_weekday_flow = dfi.Total.mean()
        flows.append(avg_flow)
        weekday_flows.append(avg_weekday_flow)
    
    # sum bidirectional flow
    flow = sum(flows)
    week_day_flow = sum(weekday_flows)

    # get station name from file name
    name_parts = file_name.split("-")
    station_name = name_parts[2].strip()+'-'+name_parts[3].strip()
    for item in TO_BE_REMOVED:
        if item in station_name:
            return None, None, None

    return flow, week_day_flow, station_name