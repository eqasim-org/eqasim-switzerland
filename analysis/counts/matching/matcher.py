#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:33:36 2025

@author: dabdelkader
"""

import logging
import time
from .point_matcher import PointMatcher
from .link_matcher import LinkMatcher
import os 
import pandas as pd

logger = logging.getLogger("synpp")

class TrafficDataMatcher:
    def __init__(self, name:str, cache:str="cache", overwrite:bool = True):
        self.point_matcher = PointMatcher()
        self.link_matcher = LinkMatcher()
        
        self.cache_file = os.path.join(cache,name.lower()+"_matching_cache.pkl")
        self.matches = None
        self.overwrite = overwrite
        
    def match(self, **kwargs):
        if not self.overwrite and os.path.exists(self.cache_file):
            self.load()
            if self.matches is not None:
                return self.matches
        
        return self._match(**kwargs)
        
    def _match(self, **kwargs):
        start_time = time.time()
        counts = kwargs["counts"]
        if counts.counts.geom_type.iloc[0] == "Point":
            result = self.point_matcher.match(**kwargs)
        else:
            result = self.link_matcher.match(**kwargs)
        logger.info(f"Matching counts took {(time.time()-start_time):.1f} seconds.\n")
        
        if "only_two_link_ids" in kwargs and kwargs["only_two_link_ids"]:
            result = self.keep_only_two_directional_links(result) 
            
        self.matches = result
        self.save()
        return result
    
    def keep_only_two_directional_links(self, matched):    
        # Remove one direction links
        occurance = matched.groupby("id")["id"].transform("count")        
        matched   = matched[occurance==2].reset_index(drop=True)
        return matched
    
    def load(self):
        if os.path.exists(self.cache_file):
            self.matches = pd.read_pickle(self.cache_file)
        else:
            logger.warn("No file in the cache to lead!")
    
    def save(self):
        if self.matches is not None:
            pd.to_pickle(self.matches, self.cache_file)
        else:
            logger.warn("No matched stations to save!")





