#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 09:31:06 2025

@author: dabdelkader
"""


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging
from .utils.osm import Osm
from pathlib import Path
from typing import Union, List

logger = logging.getLogger("synpp")
        
        
class Counts:
    
    def __init__(self, 
                 counts_data_file: str = None, 
                 count_stations_file: str = None,
                 include_incomplete_data: bool = True, 
                 minimum_months: int = 6,
                 year: int = 2023,
                 file_path: Union[str, Path, List[Union[str, Path]]] = None,
                 context: str = None,
                 **kwargs):

        # Mode 1: ASTRA format
        if count_stations_file and counts_data_file:
            self.count_stations = self.read_astra_counts_stations_locations(count_stations_file)
            self.count_data = self.read_astra_counts_data(
                counts_data_file=counts_data_file,
                count_stations=self.count_stations,
                minimum_months=minimum_months,
                include_incomplete_data=include_incomplete_data,
                year=year
            )

        # Mode 2: Shapefile or GPKG
        elif file_path:
            if isinstance(file_path, (list, tuple)) and len(file_path) == 2:
                # Recurse manually with unpacked args
                self.__init__(
                    counts_data_file=file_path[0],
                    count_stations_file=file_path[1],
                    include_incomplete_data=include_incomplete_data,
                    minimum_months=minimum_months,
                    year=year,
                    **kwargs
                )
                return
            else:
                self.count_stations, self.count_data = self.from_file(file_path, **kwargs)

        else:
            raise ValueError("You must provide either count station/data files or a (GPKG/SHP) file.")

        if not hasattr(self, "count_stations") or not hasattr(self, "count_data"):
            raise RuntimeError("Initialization failed: `count_stations` or `count_data` is missing.")   

        ### Border handling
        if context:
            self.ch_border = context.stage("data.spatial.swiss_border")
        else:
            self.ch_border = Osm().get_border("switzerland")   
            
    def from_file(self, file_path, id_column="OBJECTID", columns_to_keep=['TJM', 'TJOM'], projection="EPSG:2056"):
        # Load the file
        df = gpd.read_file(file_path)
        
        if isinstance(columns_to_keep,list):
            df = df[[id_column,*columns_to_keep,'geometry']]
            df = df.rename(columns={
                    col: ('id' if col == id_column else
                          'flow' if 'tjm' in col.lower() else 
                          'flow_w' if 'tjom' in col.lower() else 
                          'flow_heavy' if 'lourd' in col.lower() else   
                          
                          col.lower())
                    for col in df.columns })
        elif isinstance(columns_to_keep,dict):
            df = df[[id_column,*columns_to_keep.keys(),'geometry']]
            df = df.rename(columns={id_column:"id",**columns_to_keep})
        else:
            raise ValueError("columns_to_keep must be a list or a dict.")
            
        df = df[df.flow.notna()].reset_index(drop=True)
        df.set_crs(projection, inplace=True)
        
        cols = ["id","geometry"]
        if "angle" in  columns_to_keep:
            cols.append("angle")
        stations = df[cols]        
        return stations, df                    
                
    def get_id(self, _id:int):
        if _id not in self.count_data["id"]:
            return None
        station = self.count_data[self.count_data["id"]==_id].iloc[0]
        return station
    
    @property
    def counts(self):
        return self.get_counts_data()
            
    def get_counts_data(self):
        return self.count_data

    def get_count_stations(self):      
        return self.count_stations
                   
    def read_astra_counts_stations_locations(self, count_stations_file:str=None, projection="EPSG:2056")-> gpd.GeoDataFrame:
            
        count_stations = pd.read_excel(count_stations_file, skiprows=10, dtype={0: int})
        count_stations.columns = [col.split()[1] for col in count_stations.columns]
    
        count_stations = count_stations.rename(columns={
                                'Numéro': 'id',
                                'Désignation': 'poste',
                                'Canton': 'canton',
                                '1': 'direction1', 
                                '2': 'direction2', 
                                'Route': 'route',
                                'Ost': 'coor_est',   
                                'Nord': 'coor_nord' 
                            })
    
        geometry = [Point(xy) for xy in zip(count_stations['coor_est'], count_stations['coor_nord'])]
        count_stations = gpd.GeoDataFrame(count_stations, geometry=geometry)
        count_stations.set_crs(projection, inplace=True)
        
        count_stations = count_stations.drop(columns=["coor_est","coor_nord"])
        return count_stations
        
    def read_astra_counts_data(self, counts_data_file:str=None, 
                               count_stations:gpd.GeoDataFrame=None, 
                               include_incomplete_data:bool=None,
                               minimum_months:int=6,                                
                               year:int=2023)-> gpd.GeoDataFrame:        
        
        assert minimum_months>0, "minimum number of months should be higher than 0"
        
        df = pd.read_excel(counts_data_file, sheet_name='TJM', skiprows=6, dtype={0: str})
        df = df.drop(columns=["Unnamed: 2"])
        cols = ['Nr.', 'Station de mesure', 'Ct','Route']
        df[cols] = df[cols].ffill()     
  
        df.rename(columns={    
            year:"flow",
            "Unnamed: 5":"TJM",
            'Nr.': 'id',
            'Station de mesure': 'poste',
            'Ct': 'canton',            
            }, inplace=True)
        
        # Only keep TJMO
        df = df[df["TJM"].isin(['TJM', 'TJMO'])]
        
        # If we include incomplete data, we add it to the flow column
        if include_incomplete_data:            
            months = ['01', '02', '03', '04', '05','06', '07', '08', '09', '10', '11', '12']
            incomplete = df[months].notna().sum(axis=1)
            to_carry_on = (incomplete>minimum_months)
            to_modify_flow = to_carry_on & df.flow.isna()
            df.loc[to_modify_flow,"flow"] = df.loc[to_modify_flow,months].mean(axis=1)
            
        # Last cleaning
        df["id"] = df["id"].astype(int)
        df = df.loc[:,["id","flow","TJM"]]   
        
        df = df.pivot(index='id', columns='TJM', values='flow').reset_index()
        df = df.rename(columns={"TJM":"flow","TJMO":"flow_w"})
        
        df = df[df["flow"].notna()].reset_index(drop=True)        
        
        df = df.merge(count_stations, on="id", how="inner")
        
        df = gpd.GeoDataFrame(df, 
                              geometry='geometry', 
                              crs=count_stations.crs)
        return df
    
    def get_near_border_ids(self, distance_to_border:float):
        border = self.ch_border.copy()
        stations = self.count_stations[["id","geometry"]].copy()
        border = border.to_crs(stations.crs)
        
        assert border.crs==stations.crs, "There is projection incompatibility between border and stations"
        
        swiss_poly = border.geometry.iloc[0]
        stations["distance_to_border"] = stations.geometry.distance(swiss_poly.boundary)
        
        points_near_border = stations[stations.distance_to_border<distance_to_border]
        return points_near_border[["id","distance_to_border"]]
    
    
    
    
    
    
    
    
    
    
    