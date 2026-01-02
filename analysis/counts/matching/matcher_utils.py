#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:54:25 2025

@author: dabdelkader
"""

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from typing import Union
from .matching_functions import GeometryOrientation, GeometryDistanceMetrics


class MatcherUtils:
    @staticmethod
    def get_link_types():
        return iter([
            'motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link',
            'primary_link', 'secondary', 'secondary_link', 'tertiary',
            'tertiary_link', 'unclassified', 'residential', 'living_street',
            'service', 'track', 'busway', 'pedestrian', 'footway', 'platform',
            'bus_stop', 'construction', 'nan'
        ])
    
    @staticmethod
    def get_roads_category(cat, network, clipped_geo=None):
        # Only get motorways
        road_type = [cat]
        motorways = network.get_ways(road_types=road_type, geo_df = clipped_geo)
        return motorways
    
    @staticmethod
    def spatial_join_nearby(count_data:gpd.GeoDataFrame, roads:gpd.GeoDataFrame, search_radius:float):
        joined = gpd.sjoin(count_data, roads, how="left", predicate="dwithin", distance=search_radius)
        joined = joined.drop(columns=["index_left", "index_right"], errors="ignore") 
        joined = joined[joined["link_id"].notna()]
        return joined
    
    @staticmethod
    def find_point_match(count_data: gpd.GeoDataFrame, 
                         roads: gpd.GeoDataFrame, 
                         search_radius: float,
                         get_pairs:bool=True) -> gpd.GeoDataFrame:
        # Perform spatial join with distance constraint
        joined = MatcherUtils.spatial_join_nearby(count_data, roads, search_radius)
        
        # Add road geometry for distance calculation
        joined = joined.merge(roads[["link_id", "geometry"]].rename(columns={"geometry": "road_geometry"}),
                              on="link_id",how="left")
        # Compute actual distances between count point and joined road geometry
        joined["distance"] = joined.geometry.distance(joined["road_geometry"])
        joined["bearing"]  = joined["road_geometry"].apply(GeometryOrientation.calculate_bearing)
        
        # Keep only the two closest road links for each count point
        if get_pairs:
            joined = joined.groupby("id", group_keys=False)[joined.columns].apply(GeometryOrientation.get_best_opposite_pair).reset_index(drop=True)
            #joined = joined[joined.groupby("id")["id"].transform("count") == 2]
        else:
            joined = joined.groupby("id", group_keys=False)[joined.columns].apply(GeometryDistanceMetrics.get_minimum_distance_match).reset_index(drop=True)
            # TODO: I don't remember why this line ->
            # joined = joined[joined.groupby("id")["id"].transform("count") == 2]
            
        joined = joined.drop(columns=['bearing'])
        return joined
        
                
        
    @staticmethod
    def find_link_match(geom:Union[LineString, MultiLineString], roads:gpd.GeoDataFrame, 
                        sindex=None, search_radius:float=20):    
        """
        In this function, matching is based on a custom hausdorff distance. If this distance
        is less than 3meters, the link is considered a match. In addiction, if there are not
        links that respect this condition, keep the closest one, if it is not more that 10 meters
        from the reference geometry.
        """
        # Only keep close links
        if sindex:
            bounds = geom.buffer(search_radius).bounds
            possible_idx = list(sindex.intersection(bounds))
            candidates = roads.iloc[possible_idx].copy()
            if not len(candidates):
                return [None], [None], [None]
        else:
            candidates = roads    
        #only parallel links
        candidates.loc[:, "direction"] = candidates.geometry.apply(lambda g: GeometryOrientation.is_parallel(geom, g))
        candidates = candidates[candidates.direction.isin(["same","opposite"])]
        if not len(candidates):
            return [None], [None], [None]
        
        # Compute distance to all candidates
        candidates.loc[:,"distances"] = candidates.geometry.apply(lambda g: 
                              GeometryDistanceMetrics.asymmetric_min_hausdorff(geom, g))
        sel = candidates["distances"]<3 #3 meters distance as threshold
        
        if sel.sum() > 2 and set(candidates.loc[sel, "direction"].unique()) == {"opposite", "same"}:
            min_idx = candidates[sel].index
        elif len(candidates):
            min_idx = []
            for direction in ["same", "opposite"]:
                sub = candidates[candidates["direction"] == direction]
                if not sub.empty:
                    idx = sub["distances"].idxmin()
                    if candidates.loc[idx, "distances"] < 10:
                        min_idx.append(idx)
        
            if not len(min_idx):
                return [None], [None], [None]
        else:
            return [None], [None], [None]
        
        return (candidates.loc[min_idx, 'link_id'].tolist(), 
                candidates.loc[min_idx,"distances"].tolist(), 
                candidates.loc[min_idx,"direction"].tolist())  







