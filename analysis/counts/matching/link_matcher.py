#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:53:29 2025

@author: dabdelkader
"""

import geopandas as gpd
import logging
from .matcher_utils import MatcherUtils
from .counts import Counts
from .network import RoadNetwork
import sys

logger = logging.getLogger("synpp")

class LinkMatcher:
    def match(self, network: RoadNetwork, counts: Counts, search_radius:float=30, **kwargs):
        # Start matching: Here I will not use a lopp because most the links have same importance I think
        logger.info("Matching counts (links) to MATSim network")
        logger.info("Using Links matcher: matching all links within 2m distance")
        
        # Get count data
        count_data = counts.counts.copy()                
        count_data = gpd.GeoDataFrame(count_data.loc[:, ["id", "geometry"]], geometry='geometry', crs=count_data.crs)
        number_of_stations = set(count_data["id"])
        
        # Get local network
        assert network.crs == count_data.crs, "CRS Mismatch"
        west,south,east,north = count_data.total_bounds  
        buffer = 100+search_radius # Only keep the regional roads: Faster
        roads = network.get_geometry().cx[west-buffer:east+buffer, south-buffer:north+buffer]        
        roads = roads[roads.link_id.str.isnumeric()].reset_index(drop=True) # This would exlude pt links
        
        # Match
        count_data = self._match(roads, count_data, search_radius, **kwargs)
        matched_stations =  set(count_data["id"])        
        logger.info(f"    {len(number_of_stations-matched_stations)} stations are unfound!")
        
        # Only keep relevant rows
        count_data = count_data[count_data.link_id.notna()].reset_index(drop=True)
        
        columns_to_keep = ['id', 'geometry', 'link_id', 'road_geometry','distance', "direction"]
        return count_data[columns_to_keep]


    def _match(self, roads:gpd.GeoDataFrame, 
                     count_data:gpd.GeoDataFrame, 
                     search_radius:float, 
                     write_progression:bool=True, **kwargs): 
        
        self.total = len(count_data)        
        
        sindex = roads.sindex        
        matches = count_data.geometry.apply(lambda g: self._match_one_station(g, roads, sindex, search_radius, write_progression))        
        
        matched_data = count_data[["id"]].copy()
        matched_data['link_id']   = matches.apply(lambda x: x[0])
        matched_data['distance']  = matches.apply(lambda x: x[1])
        matched_data['direction'] = matches.apply(lambda x: x[2])
        matched_data = matched_data.explode(column = ['link_id', 'distance', 'direction'])
        matched_data = matched_data[matched_data.link_id.notna()]
        
        count_data = matched_data.merge(count_data, on="id", how="left")
        
        count_data = count_data.merge(roads[["link_id", "geometry"]].rename(columns={"geometry": "road_geometry"}),
                              on="link_id",how="left")
                
        return count_data.reset_index(drop=True)

    def _match_one_station(self, g, roads, sindex, search_radius, write_progression=True):
        if write_progression:            
            self.print_progress()                
        return MatcherUtils.find_link_match(g, roads, sindex, search_radius)

    def print_progress(self):
        if not hasattr(self, "_counter"):
            self._counter = 1
            
        percent = (self._counter / self.total) * 100
        end_char = '\n' if self._counter == self.total else ''
        sys.stdout.write(f'\rMatching: {self._counter}/{self.total} ({percent:.2f})%{end_char}')
        sys.stdout.flush()
        self._counter += 1
