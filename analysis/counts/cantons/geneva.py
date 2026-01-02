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
import numpy as np

def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"),"traffic_counts"))

def execute(context):
    # Define paths
    data_path = context.config("counts_path")    
    output_path = os.path.join(context.path(),"processed_data.gpkg")
    geneva_counts_data  = os.path.join(data_path,"Geneva","comptages_routiers_WGS84.zip")
    geneva_locations    = os.path.join(data_path,"Geneva","Localisation.csv")
    
    # read the file    
    veh_types = ["VL", "tous-vehicules"]
    cols = ['OBJECTID', 'dates', 'heure', 'comptages_ts_vehicules', 
            'veh', 'type_veh', 'X_LV95', 'Y_LV95', 'ANGLE']

    dtypes = {
        'OBJECTID':str,
        'dates': str,
        'heure': 'category',
        'comptages_ts_vehicules': int,
        'veh': 'category',
        'type_veh': 'category',
        'X_LV95': float,
        'Y_LV95': float,
        'ANGLE': "float32"
    }    
    df = pd.read_csv(geneva_counts_data, usecols=cols, dtype=dtypes)
    df = df[df.type_veh.isin(veh_types)].reset_index(drop=True)
    df = df[df.comptages_ts_vehicules<10000] # remove outliers

    # get week days and weekends
    df['dates'] = pd.to_datetime(df['dates'], format='%d.%m.%Y')

    #Only keep recent data
    df = df[df.dates.dt.year>=2023].reset_index(drop=True)        

    # identify weekends
    df['day'] = df['dates'].dt.day_name()
    df['is_weekend'] = df['day'].isin(['Saturday', 'Sunday'])

    # locations
    locations = df.loc[:,['OBJECTID', 'X_LV95', 'Y_LV95', 'ANGLE']].drop_duplicates('OBJECTID')
    
    # compute average flow
    def median_average(group, lower=0.1, upper=0.9):  
        valid_group = lambda g: len(g) > 20 and (g > 0).sum() > 12 # at leat 20 observations per day and 12 positif values
        
        valid = group.groupby("dates")['comptages_ts_vehicules'].filter(valid_group)
        mean  = group.loc[valid.index].groupby("dates")['comptages_ts_vehicules'].mean()    
        
        # filter out values that are higher than the estimated capacity (impossible to have an average flow during the whole day higher than capacity)
        limit_value = group.loc[valid.index].groupby("dates")['comptages_ts_vehicules'].max().median()    
        mean = mean[(mean > 0)&(mean < limit_value)]
        
        number_of_counts_limit = 4 if group.is_weekend.iloc[0] else 10
        if len(mean)<number_of_counts_limit: #at least two weeks of correct observations
            low, high, median = np.nan, np.nan, np.nan
        else:
            low = mean.quantile(lower)
            high = mean.quantile(upper)
            median = mean.median()
        
        return pd.Series({'median_flow': median*24,
                        'quantile_lower_flow': low*24,
                        'quantile_upper_flow': high*24})

    avg_flows = df.groupby('OBJECTID').apply(median_average)
    avg_flows = avg_flows[avg_flows.median_flow.notna()].reset_index()

    avg_flows_w = df[~df['is_weekend']].groupby('OBJECTID').apply(median_average)
    avg_flows_w = avg_flows_w[avg_flows_w.median_flow.notna()].reset_index()

    # compute maximum flow
    def get_maximum_flow(group):
        valid_group = lambda g: len(g) > 20 and (g > 0).sum() > 12    
        valid = group.groupby("dates")['comptages_ts_vehicules'].filter(valid_group)
        capa  = group.loc[valid.index].groupby("dates")['comptages_ts_vehicules'].max()    
        return pd.Series({'max_flow': capa[capa > 0].median()*24})
        
    flow_99 = df[~df['is_weekend']].groupby('OBJECTID').apply(get_maximum_flow).reset_index()

    # merge results    
    df = (avg_flows.merge(avg_flows_w, on='OBJECTID', suffixes=('','_weekday'))
                   .merge(flow_99, on="OBJECTID", how="left"))
    df = df.rename(columns={'median_flow':"flow", 'median_flow_weekday':"flow_w"})
    df = df[df.flow.notna() & df.flow_w.notna()].reset_index(drop=True)

    # Add geometry    
    df = df.merge(locations, on="OBJECTID", how="left")

    df['geometry'] = gpd.points_from_xy(df["X_LV95"], df["Y_LV95"])
    df             = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:2056')        

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