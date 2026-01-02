# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:27:07 2025

@author: dabdelkader
"""
import os
import pandas as pd

class Merge:
    def __init__(self, city, matched, flows, cache = "cache"):
        self.flows = flows
        self.matched = matched
        self.city = city.lower()
        self.cache = cache
        self.filename = os.path.join(cache, f"results_flows_{city}.pkl")
    
    def run(self, return_it = False, return_path = False):
        grouped =self.matched.groupby("id").agg({
                                "geometry": "first",
                                "link_id": list,
                                "road_geometry": list,
                                "distance": list
                            }).reset_index()
        
        df = self.flows.merge(grouped, on="id", how="left")
        df["city"] = self.city
        
        df.to_pickle(self.filename)
        
        if return_it:
            if return_path:
                return df, self.filename
            return df
        if return_path:
            return self.filename
