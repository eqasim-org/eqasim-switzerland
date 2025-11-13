#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:22:46 2025

@author: dabdelkader
"""

from .BaseUtility import BaseUtility
from .BikeUtility import BikeUtility
from .CarUtility import CarUtility
from .PtUtility import PtUtility
from .WalkUtility import WalkUtility
from .ZeroUtility import ZeroUtility
from .CpUtility import CpUtility
import pandas as pd
import numpy as np
import polars as pl
import glob
import os

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TourUtility(BaseUtility):
    utility_estimators = {
        "car": CarUtility,
        "pt": PtUtility,
        "walk": WalkUtility,
        "bike": BikeUtility,
        "car_passenger": CpUtility
    }

    # Shared class-level variables for dataframes
    variables_by_mode = {}
    tours = None
    exploded_tours = None
    persons = []
    num_persons = 0
    
    @staticmethod
    def init(tours=None, persons=None, trips = None, variables=None):
        # add row_id to tours (like index in pandas)
        tours = tours.with_row_index(name="tour_row_id")

        # Unpack travel times
        TourUtility.variables_by_mode = {
            "car": variables['car'].lazy(),
            "pt": variables['pt'].lazy(),
            "bike": variables['bike'].lazy(),
            "walk": variables['walk'].lazy(),
            "car_passenger": variables['car_passenger'].lazy()
        }
        
        # make tours lazy
        TourUtility.tours = tours.lazy()
                  
        # for better efficiency, we explode tours here
        exploded_tours = TourUtility.get_exploded_tours_for_utilities()
        TourUtility.exploded_tours = {k:v.lazy() for k,v in exploded_tours.items()}
        
        # store persons info
        TourUtility.persons = tours["person_id"].unique()
        TourUtility.num_persons = len(TourUtility.persons)

        # if persons are provided, meaning their attributes are not in the variables
        if persons is not None:
            TourUtility.add_persons_attributes_to_variables(persons.lazy())
        
        # if trips are provided, meaning their attributes are not in the variables
        if trips is not None:
            TourUtility.add_trips_attributes_to_variables(trips.lazy())

    @staticmethod
    def get_exploded_tours_for_utilities():
        if TourUtility.tours is None:
            raise RuntimeError("Tours are not initialized.")
    
        cols = ["tour_row_id", "trip_id", "mode_candidates"]

        # Explode trips and candidate modes
        exploded_lazy = (
            TourUtility.tours.select(cols)
            .explode(["trip_id", "mode_candidates"])
            .with_columns([
            pl.col("mode_candidates").cast(pl.Categorical)
            ])
        ).collect()

        exploded_lazy = {mode: exploded_lazy.filter(pl.col("mode_candidates") == mode)
                         for mode in TourUtility.utility_estimators}
        return exploded_lazy

    @staticmethod
    def add_persons_attributes_to_variables(persons: pl.LazyFrame):
        for mode, variables_lazy in TourUtility.variables_by_mode.items():
            # Join persons attributes with the variables
            variables_lazy = variables_lazy.join(persons, on="person_id", how="left").collect() #do this now, I don't want a do it each time we compute utilities
            TourUtility.variables_by_mode[mode] = variables_lazy.lazy()

    @staticmethod
    def add_trips_attributes_to_variables(trips: pl.LazyFrame):
        for mode, variables_lazy in TourUtility.variables_by_mode.items():
            # Join trips attributes with the variables
            variables_lazy = variables_lazy.join(trips, on="trip_id", how="left").collect() #do this now, I don't want a do it each time we compute utilities
            TourUtility.variables_by_mode[mode] = variables_lazy.lazy()

    @staticmethod
    def compute_mode_utilities(mode: str) -> pl.LazyFrame:
        estimator = TourUtility.utility_estimators.get(mode)
        variables_lazy = TourUtility.variables_by_mode.get(mode)
        exploded_lazy = TourUtility.exploded_tours[mode]

        return (
            exploded_lazy
            .join(variables_lazy, on="trip_id", how="left")
            .with_columns([
                estimator.compute_lazy().cast(pl.Float32)
                .alias("utility")
            ])
            .select(["tour_row_id","utility"])
        )
             
    @staticmethod
    def get_all_utilities():       
        # Compute utilities per mode
        results = (pl.concat([ TourUtility.compute_mode_utilities(mode)
                              for mode in TourUtility.utility_estimators])
                   .group_by("tour_row_id")
                   .agg(pl.col("utility").sum().alias("utility")))
                   
        
        ### join with tours and return results
        #select data      
        cols = ["tour_row_id", "person_id", "trip_id", "tour_id"]                 
        results = (TourUtility.tours.select(cols)
                    .join(results, on="tour_row_id", how="left")
                    .select(cols + ["utility"]))        
        return results
        










    

    