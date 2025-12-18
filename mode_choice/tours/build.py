"""
This module takes the trips that were already build in the prepare_trips stage and build the tours needed for mode choice.
We define the tour as a sequence of trips that starts and ends at home.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import polars as pl
import os
import logging 

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("data.constants")
    context.stage("mode_choice.trips.prepare_persons")
    context.stage("mode_choice.tours.core")

    context.config("num_tour_batches", default=0)


def process(context, df_trips, df_persons, tour_id_offset=0) -> pl.DataFrame:
    df_trips = df_trips.copy()  # Ensure df_trips is a proper copy to avoid SettingWithCopyWarning

    # 1. a tour is a sequence of trips that starts and ends at home, thus each home activity defines a tour start point
    ### make sure dataframe is sorted
    df_trips = df_trips.sort_values(['person_id', 'trip_index'])
    ### Detect trip starts a tour (origin == Home)
    df_trips['tour_start'] = (df_trips["preceding_purpose"].eq("home")).astype("int8")
    ### Compute tour index as cumulative sum of tour starts per person
    df_trips['tour_index'] = df_trips.groupby('person_id')['tour_start'].cumsum()

    # 2. aggregate trips into tours
    df_tours = df_trips.groupby(['person_id', 'tour_index']).agg({
        'trip_id': list,
        'preceding_purpose': list,
        'following_purpose': list,
        'euclidean_distance_km': list
    }).reset_index()    

    # 3. merge person attributes into tours dataframe
    df_tours = df_tours.merge(df_persons, on="person_id", how="left")

    # set an id for each tour
    df_tours["tour_id"] = np.arange(len(df_tours)) + tour_id_offset

    # 4. for each tour, get possible mode combinations
    # get the function to find mode candidates for each tour
    tours_finder = context.stage("mode_choice.tours.core")# make sure the stage is executed so that any function changes are taken into account    

    # get mode candidates for each tour
    persons_attributes = ["age","car_availability","driving_license","bike_availability","is_car_passenger"]
    res = tours_finder( df_tours.euclidean_distance_km,
                        df_tours[persons_attributes].to_dict(orient="records"),
                        df_tours.preceding_purpose,
                        df_tours.following_purpose)

    df_tours["mode_candidates"] = res

    # finalize tours dataframe
    df_tours = df_tours[["person_id","trip_id","tour_id","euclidean_distance_km","mode_candidates"]]    
    df_tours = df_tours.explode("mode_candidates")

    # convert to polars
    df_tours = pl.from_pandas(df_tours).with_columns([
                    pl.col("euclidean_distance_km").list.eval(pl.element().cast(pl.Float32))
                ])
    return df_tours 
    
def execute(context):
    # get the trips dataframe
    df_trips = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id","trip_index","trip_id","preceding_purpose","following_purpose","euclidean_distance_km"]].astype({
            "euclidean_distance_km": "float32",}).copy()

    # get person attributes
    df_persons = context.stage("mode_choice.trips.prepare_persons")[
        ["person_id","age","car_availability","driving_license","bike_availability","is_car_passenger"]
        ].astype({"age": "int8"}).copy()    
    
    # build tours dataframe
    num_batches = context.config("num_tour_batches")    

    if num_batches <=1:                              
        df_tours = process(context, df_trips, df_persons)        
        return df_tours
    
    else:
        logger.info("\t Saving tours in batches...")        
        persons_in_each_batch = np.array_split(df_trips['person_id'].unique(), int(num_batches))
        
        path_to_data = context.path()
        list_paths = []    
        tour_id_offset = 0
        for i,p in enumerate(persons_in_each_batch):
            # process the batch
            batch = process(context, 
                            df_trips[df_trips['person_id'].isin(p)], 
                            df_persons[df_persons['person_id'].isin(p)],
                            tour_id_offset=tour_id_offset)
            tour_id_offset = batch['tour_id'].max() + 10
            # save the batch
            path = os.path.join(path_to_data,f"tours_batch_{i}.parquet")
            list_paths.append(path)     
            batch.write_parquet(path)
            logger.info(f"\t\t Saved batch {i+1}/{num_batches} with {len(p)} persons to {path}")

        return list_paths