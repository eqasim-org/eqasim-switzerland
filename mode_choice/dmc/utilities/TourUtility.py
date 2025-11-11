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
from modeShares.ModeShares import ModeShares
from utils.utils import stable_hash
import pandas as pd
import numpy as np
from scipy.stats import qmc
import polars as pl
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import time
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
    sample = None
    cols_to_return_with_utilities = [] 
    
    @staticmethod
    def init_data(car, pt, bike, walk, cp, tours=None, population_sample = None, 
                  eqasim_cache_dir=None, optimizer_cache_dir = None, mode_shares_provider:ModeShares=None):
        """
        Initializes mode-specific input data once.
        """
        TourUtility.sample = population_sample
        TourUtility.eqasim_cache_dir = eqasim_cache_dir
        TourUtility.optimizer_cache_dir = optimizer_cache_dir

        # if population_sample is lower then the number of agents in the dataframes, sample randomly this number of agents
        num_agents = len(tours["person_id"].unique()) if tours is not None else 0
        if (population_sample is not None) and (tours is not None) and (population_sample < num_agents):
            population = tours["person_id"].unique().sample(n=population_sample)
            car, pt, bike, walk, cp, tours = TourUtility.sample_dataframes(population, car, pt, bike, walk, cp, tours)

        TourUtility.variables_by_mode = {
            "car": car.lazy(),
            "pt": pt.lazy(),
            "bike": bike.lazy(),
            "walk": walk.lazy(),
            "car_passenger":cp.lazy()
        }

        if tours is not None:
            TourUtility.tours = tours.lazy()
            # Here we include Euclidean distance in the tours dataframe in order to get mode shares distribution
            TourUtility.create_distance_column_in_tours(mode_shares_provider)
            # Here, we include other attributes (age, income, sex, canton) for distributions
            if eqasim_cache_dir is not None:
                TourUtility.add_person_attributes_to_tours()            
            # for better efficiency, we explode tours here
            exploded_tours = TourUtility.get_exploded_tours_for_utilities()
            TourUtility.exploded_tours = {k:v.lazy() for k,v in exploded_tours.items()}
            
            TourUtility.persons = tours["person_id"].unique()
            TourUtility.num_persons = len(TourUtility.persons)
            
        TourUtility.cols_to_return_with_utilities = ['tour_row_id', 'person_id', 
         'trip_key', 'selection_id', 'candidate_mode', 'euclidean_distance',
         'age_class','sex','income_class','canton_id', 'sp_region'] 
        if mode_shares_provider is not None:
            TourUtility.cols_to_return_with_utilities.append("distance_class")
                
    @staticmethod
    def sample_dataframes(population, *args):
        filtered_dfs = []
        for df in args:
            if df is not None:
                if "person_id" in df.columns:
                    df = df.filter(pl.col("person_id").is_in(population))
            filtered_dfs.append(df)
        return filtered_dfs

    @staticmethod
    def set_population_sample(population_sample):
        TourUtility.sample = population_sample
            
    @staticmethod
    def get_utility_of(person_id, trip_index, mode):
        if mode=="car_passenger":
            return 0.0
        estimator = TourUtility.utility_estimators[mode]        
        variables = TourUtility.variables_by_mode[mode].loc[f"{person_id}_{trip_index}"]
        return estimator.compute(variables)

    @staticmethod
    def compute(tour):
        """
        Computes utilities for each trip in the tour using pre-initialized data.

        Parameters:
        - tour: an object with attributes `person_id`, `trips_index`, and `candidate_mode`

        Returns:
        - List[float]: computed utilities
        """
        if not TourUtility.variables_by_mode:
            raise RuntimeError("TourUtility data not initialized. Call init_data() first.")

        person_id = tour.person_id
        trips = zip(tour.trips_index, tour.candidate_mode)

        return [TourUtility.get_utility_of(person_id, trip_index, mode) for trip_index, mode in trips]
    
    @staticmethod
    def get_population_sample():
        sample_population = np.random.choice(TourUtility.persons, size=TourUtility.sample)  
        return sample_population
    
    
    @staticmethod
    def get_exploded_tours_for_utilities():
        if TourUtility.tours is None:
            raise RuntimeError("Tours are not initialized.")
    
        cols = ["tour_row_id", "trip_key", "candidate_mode"]

        # Explode trips and candidate modes
        exploded_lazy = (
            TourUtility.tours.select(cols)
            .explode(["trip_key", "candidate_mode"])
            .with_columns([
            pl.col("candidate_mode").cast(pl.Categorical)
            ])
        ).collect()
        
        exploded_lazy = {mode: exploded_lazy.filter(pl.col("candidate_mode") == mode)
                         for mode in TourUtility.utility_estimators}
        return exploded_lazy
    
    @staticmethod
    def compute_mode_utilities(mode: str) -> pl.LazyFrame:
        estimator = TourUtility.utility_estimators.get(mode)
        variables_lazy = TourUtility.variables_by_mode.get(mode)
        exploded_lazy = TourUtility.exploded_tours[mode]

        return (
            exploded_lazy
            .join(variables_lazy, on="trip_key", how="left")
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
        results = (TourUtility.tours.select(TourUtility.cols_to_return_with_utilities)
                    .join(results, on="tour_row_id", how="left")
                    .select([*TourUtility.cols_to_return_with_utilities, "utility"]))        
        return results
        

    @staticmethod
    def read_csv(file_path):        
        df = (
            pl.read_csv(file_path, separator=";")
            .filter(
                ~pl.col("candidate_mode").str.contains("loop")
                )
            .with_columns(
                # Split strings into lists
                pl.col("trips_index").str.split(","),
                pl.col("candidate_mode").str.split(","),
                
                # Split utilities and cast to float list
                pl.col("utilities").str.split(",")
                .list.eval(pl.element().cast(pl.Float32))
            )
            .rename({
                "utilities": "eqasim_utilities",
                "utility": "eqasim_utility",
                "selected": "eqasim_selected"
            })
            .with_row_index(name="tour_row_id")
        )
        return df
    
    @staticmethod
    def create_distance_column_in_tours(mode_shares_provider:ModeShares=None):
        # Add a stable row index to preserve original tour rows
        tours = TourUtility.tours.select(
                ['tour_row_id', 'person_id', 'trips_index', 'candidate_mode']).collect()
    
        # Explode tours into individual trips
        exploded = (tours.explode(["trips_index", "candidate_mode"])
                    .with_columns(
                        (pl.col("person_id").cast(pl.Utf8) + "_" + pl.col("trips_index").cast(pl.Utf8))
                        .alias("trip_key"))
                    .with_columns(pl.lit(None).cast(pl.Float64).alias("euclidean_distance"))
                    )
    
        for mode in ["car", "pt", "walk", "bike", "car_passenger"]:
            variables_df = TourUtility.variables_by_mode.get(mode).collect()
            if variables_df is None:
                raise RuntimeError(f"Missing variables dataframe for mode {mode}.")
    
            variables_df = variables_df.select(["trip_key", "euclideanDistance_km"]
                            ).rename({"euclideanDistance_km": "euclidean_distance"})            
    
            # Join distances on trip_key
            exploded = exploded.join(variables_df, on="trip_key", how="left")
    
            # Update only rows where candidate_mode == mode
            exploded = exploded.with_columns(
                pl.when(pl.col("candidate_mode") == mode)
                  .then(pl.col("euclidean_distance_right"))  # from join
                  .otherwise(pl.col("euclidean_distance"))   # keep existing
                  .alias("euclidean_distance")
            ).drop("euclidean_distance_right")

        # Group back by original tour row ID and collect euclidean_distance as list
        cols_to_explode = ["tour_row_id","trip_key","euclidean_distance"]
        if mode_shares_provider is not None:
            distance_bins = np.array(mode_shares_provider.get_distance_bins())*1e-3 #convert to km    
            distance_labels = mode_shares_provider.get_distance_labels()
            exploded      = exploded.with_columns([
                pl.col("euclidean_distance").cut(breaks=distance_bins[1:-1], 
                                                 labels=distance_labels).alias("distance_class")])                            
            cols_to_explode.append("distance_class")

        updated_tours = (exploded.select(cols_to_explode)
                         .group_by("tour_row_id")
                         .agg([pl.col(j) for j in cols_to_explode if j!="tour_row_id"])
                         .sort("tour_row_id"))
        
        tours = TourUtility.tours.collect().with_columns([
                updated_tours["euclidean_distance"].alias("euclidean_distance"),
                updated_tours["trip_key"].alias("trip_key"),
                updated_tours["distance_class"].alias("distance_class") if "distance_class" in updated_tours.columns else None
            ])
        
        TourUtility.tours = tours.lazy()
            
    
    @staticmethod
    def add_person_attributes_to_tours():         
        if TourUtility.eqasim_cache_dir is None:
            return
        persons = TourUtility.get_persons()
        tours = TourUtility.tours.collect()
        tours = tours.join(persons, on="person_id", how="left")
        
        assert tours.select(pl.col("sex").is_nan().sum()).item()==0, "Some agents are not found!"
        assert tours.select(pl.col("sp_region").is_nan().sum()).item()==0, "Some spRegions are not found!"
        
        TourUtility.tours = tours.lazy()

    @staticmethod
    def get_persons(attributes=["age_class","sex","income_class","canton_id", "sp_region"], overwrite = False):
        eqasim_cache_dir = TourUtility.eqasim_cache_dir
        cache_dir = TourUtility.optimizer_cache_dir
        hash_file = stable_hash((eqasim_cache_dir, cache_dir))
        file_name = os.path.join(cache_dir, f"persons_{hash_file}.parquet")

        # Always include person_id              
        if os.path.exists(file_name) and not overwrite:
            persons = pl.read_parquet(file_name)
            return persons.select(["person_id",*attributes])
        else:
            # this is slower, so we save it in the first matsim iteration as parquet file
            # for faster polars reads
            persons_file = glob.glob(os.path.join(eqasim_cache_dir, "**", "*synthesis.population.enriched*.p"), recursive=True)
            persons_file = max(persons_file, key=os.path.getctime)
            persons_pd = pd.read_pickle(persons_file)            

            int_cols = ["age_class","sex","income_class","canton_id", "sp_region"]
            persons_pd = persons_pd.astype({attr: int for attr in int_cols if attr in persons_pd.columns})

            persons_pd.to_parquet(file_name)
            persons = pl.from_pandas(persons_pd[["person_id",*attributes]])
            return persons

    @staticmethod
    def read_and_init(tours, car, pt, bike, walk, car_passenger, 
                      population_sample = None, eqasim_cache_dir = None, 
                      optimizer_cache_dir = None,
                      mode_shares_provider:ModeShares=None):
        
        logger.info("Reading and initializing TourUtility with provided data files.")
        df_tours = TourUtility.read_csv(tours)        
        
        df_bike = BikeUtility.read_csv(bike)
        df_car  = CarUtility.read_csv(car)
        df_pt   = PtUtility.read_csv(pt)
        df_walk = WalkUtility.read_csv(walk)
        df_cp   = ZeroUtility.read_csv(car_passenger)

        TourUtility.init_data(df_car, df_pt, df_bike, df_walk, df_cp, df_tours, 
                              population_sample=population_sample,
                              eqasim_cache_dir = eqasim_cache_dir,
                              optimizer_cache_dir = optimizer_cache_dir,
                              mode_shares_provider = mode_shares_provider)











    

    