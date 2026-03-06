#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:53:12 2025

@author: dabdelkader
"""

import geopandas as gpd
import pandas as pd
import logging
from shapely.geometry import LineString
from .matcher_utils import MatcherUtils
from .network import Network
from .counts import Counts
from .utils.osm import Osm
from .matching_functions import GeometryOrientation, GeometryDistanceMetrics

logger = logging.getLogger(__name__)

class PointMatcher:

    def match(self, network:Network, counts:Counts, osm:Osm=None,
              search_radius:float=10, by_highway_order:bool=False, 
              get_pairs:bool=True, **kwargs):
        
        if by_highway_order:
            return self.match_points_by_highway_order(network, counts, osm, search_radius, get_pairs, **kwargs)
        else:
            return self.match_points(network, counts, osm, search_radius, get_pairs, **kwargs)
    
    def match_points_by_highway_order(self, network: Network, 
                                            counts: Counts, 
                                            osm:Osm = None,
                                            search_radius: float = 30,
                                            get_pairs:bool=True,
                                            direction_from_osm:bool = False,
                                            **kwargs) -> gpd.GeoDataFrame:
        link_types_iter = MatcherUtils.get_link_types()
        
        all_matched = False
        _search_radius = search_radius
        count_station_ids = set(counts.counts["id"].unique().tolist())        
        columns_to_keep = ['id', 'geometry', 'link_id', 'road_geometry','distance']
        res = pd.DataFrame(columns=columns_to_keep)
        
        logger.info("Matching count stations to MATSim network")        
        while not all_matched:
            count_data = counts.counts.copy()
            road_type  = next(link_types_iter)
            roads      = MatcherUtils.get_roads_category(road_type, network)
            assert count_data.crs == roads.crs, "CRS mismatch"                        
            
            logger.info(f"    Matching {road_type} ...")
            # Ensure dataset are GeoDataFrames
            count_data = gpd.GeoDataFrame(count_data.loc[:, ["id", "geometry"]], geometry='geometry', crs=count_data.crs)
            count_data = count_data[~count_data["id"].isin(res["id"])] # Do not match same id multiple times
            
            # Match
            joined = MatcherUtils.find_point_match(count_data = count_data, 
                                                   roads = roads, 
                                                   search_radius = _search_radius, 
                                                   get_pairs=get_pairs)
            
            if not joined.empty and not joined.isna().all(axis=1).all():
                res = pd.concat([res, joined]) if not res.empty else joined
            
            matched_ids = set(res["id"])
            if count_station_ids==matched_ids:
                all_matched = True
                
            if road_type=="primary_link":
                logger.info("    Breaking the while loop at 'primary ramps'.")
                logger.info(f"    {len(count_station_ids)-len(matched_ids)} stations are unfound!")
                break
            
            _search_radius = max(search_radius/4,_search_radius*0.7) #descrease the distance by 30% each iteration

        if osm is not None and direction_from_osm:
            res["bearing"] = res["link_id"].apply(lambda x: network.get_bearing(x, in_simulation_link=True))
            res = self.include_directions_from_names(res, counts, osm)
            columns_to_keep.append("direction")
            
        return res[columns_to_keep]
    
    
    def match_points(self, network: Network, 
                           counts: Counts, 
                           osm:Osm = None,
                           search_radius: float = 10,
                           get_pairs:bool=False, 
                           **kwargs):
        logger.info("Matching count stations to MATSim network")
        # Get count data
        count_data = counts.counts.copy()                
        count_data = gpd.GeoDataFrame(count_data.loc[:, ["id", "geometry"] + (["angle"] if "angle" in count_data else [])],
                                      geometry='geometry', 
                                      crs=count_data.crs)
        number_of_stations = set(count_data["id"])
        
        # Get local network
        assert network.crs == count_data.crs, "CRS Mismatch"
        west,south,east,north = count_data.total_bounds  
        buffer = 100+search_radius
        roads = network.get_geometry().cx[west-buffer:east+buffer, south-buffer:north+buffer]        
        roads = roads[roads.link_id.str.isnumeric()].reset_index(drop=True)
        
        # Matching        
        joined = MatcherUtils.find_point_match(count_data = count_data, 
                                               roads = roads, 
                                               search_radius = search_radius, 
                                               get_pairs=get_pairs)
        
        matched_stations =  set(joined["id"])  
        logger.info(f"    {len(number_of_stations-matched_stations)} stations are unfound!")
        
        # Only keep relevant rows
        joined = joined[joined.link_id.notna()].reset_index(drop=True)
        
        columns_to_keep = ['id', 'geometry', 'link_id', 'road_geometry','distance']
        return joined[columns_to_keep]


    def include_directions_from_names(self, res, counts, osm):
        #TODO: change the logic when names are provided
        places = osm.places.copy()
        df = res.copy()
        places = places.to_crs(df.crs)
        
        df = df.merge(counts.counts[['id','direction1', 'direction2']],
                      on="id",
                      how="left")
        
        places["city_geometry_direction1"] = places["geometry"]
        places["city_geometry_direction2"] = places["geometry"]
        
        df = df.merge(places[['old_name', 'city_geometry_direction1']], left_on="direction1", right_on="old_name", how="left")
        df = df.merge(places[['old_name', 'city_geometry_direction2']], left_on="direction2", right_on="old_name", how="left")
        
        cond = lambda x: x['city_geometry_direction1'] is not None and not x['city_geometry_direction1'].is_empty
        df['city_geometry_direction1'] = df.apply(
                            lambda row: LineString([row['geometry'], row['city_geometry_direction1']]) if cond(row) else None,
                            axis=1
                        )
        cond = lambda x: x['city_geometry_direction2'] is not None and not x['city_geometry_direction2'].is_empty
        df['city_geometry_direction2'] = df.apply(
                            lambda row: LineString([row['geometry'], row['city_geometry_direction2']]) if cond(row) else None,
                            axis=1
                        )
        df["bearing_direction1"]  = df["city_geometry_direction1"].apply(GeometryOrientation.calculate_bearing)
        df["bearing_direction2"]  = df["city_geometry_direction2"].apply(GeometryOrientation.calculate_bearing)
        
        # Here I just need to assign directions.
        directions = GeometryOrientation.assign_bearing_direction(df)
        return directions
