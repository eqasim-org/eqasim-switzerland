"""
This module takes the trips that were already build in the prepare_trips stage and build the tours needed for mode choice.
We define the tour as a sequence of trips that starts and ends at home.
"""
import pandas as pd
import geopandas as gpd
import numpy as np

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("data.constants")
    context.stage("mode_choice.trips.prepare_persons")
    context.stage("mode_choice.tours.core")

def execute(context):
    df_trips = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id","trip_index","trip_id","preceding_purpose","following_purpose","euclidean_distance_km"]]

    # 1. a tour is a sequence of trips that starts and ends at home, thus each home activity defines a tour start point
    ### make sure dataframe is sorted
    df_trips = df_trips.sort_values(['person_id', 'trip_index'])
    ### Detect trip starts a tour (origin == Home)
    df_trips['tour_start'] = (df_trips["preceding_purpose"].eq("home")).astype(int)
    ### Compute tour index as cumulative sum of tour starts per person
    df_trips['tour_index'] = df_trips.groupby('person_id')['tour_start'].cumsum()

    # 2. aggregate trips into tours
    df_tours = df_trips.groupby(['person_id', 'tour_index']).agg({
        'trip_id': list,
        'preceding_purpose': list,
        'following_purpose': list,
        'euclidean_distance_km': list
    }).reset_index()
    df_tours["tour_id"] = range(len(df_tours))

    # 3. get person attributes
    df_persons = context.stage("mode_choice.trips.prepare_persons")[
        ["person_id","age","car_availability","driving_license","bike_availability","is_car_passenger"]
        ].copy()
    df_tours = df_tours.merge(df_persons, on="person_id", how="left")
    
    # 4. get all possible mode combinations for each tour
    tours_finder = context.stage("mode_choice.tours.core")# make sure the stage is executed so that any function changes are taken into account
    persons_attributes = ["age","car_availability","driving_license","bike_availability","is_car_passenger"]
    res = tours_finder( df_tours.euclidean_distance_km,
                        df_tours[persons_attributes].to_dict(orient="records"),
                        df_tours.preceding_purpose,
                        df_tours.following_purpose)
    
    df_tours["mode_candidates"] = res

    # 5. finalize tours dataframe
    df_tours = df_tours[["person_id","trip_id","tour_id","euclidean_distance_km","mode_candidates"]]    
    df_tours = df_tours.explode("mode_candidates")

    return df_tours