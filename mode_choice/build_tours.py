"""
This module takes the trips that were already build in the prepare_trips stage and build the tours needed for mode choice.
We define the tour as a sequence of trips that starts and ends at home.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import mode_choice.tours.core as tours_core

def configure(context):
    context.stage("mode_choice.prepare_trips")
    context.stage("data.constants")
    context.stage("synthesis.population.enriched")
    context.stage("mode_choice.tours.core")

def execute(context):
    df_trips = context.stage("mode_choice.prepare_trips")[
        ["person_id","trip_index","preceding_purpose","following_purpose","crowfly_distance"]]

    # 1. a tour is and home home activity, thus each home activity defines a tour end point
    ### make sure dataframe is sorted
    df_trips = df_trips.sort_values(['person_id', 'trip_index'])
    ### Detect trip ends a tour (destination == Home)
    df_trips['tour_end'] = (df_trips["following_purpose"].eq("home")).astype(int)
    ### Compute tour index as cumulative sum of tour ends per person
    df_trips['tour_index'] = df_trips.groupby('person_id')['tour_end'].cumsum()

    # 2. aggregate trips into tours
    df_tours = df_trips.groupby(['person_id', 'tour_index']).agg({
        'trip_index': list,
        'preceding_purpose': list,
        'following_purpose': list,
        'crowfly_distance': list
    }).reset_index()
    df_tours["tour_id"] = range(len(df_tours))

    # 3. get person attributes
    c = context.stage("data.constants")
    df_persons = context.stage("synthesis.population.enriched")[
        ["person_id","age","car_availability","driving_license","number_of_bikes_class","is_car_passenger"]]
    df_persons['car_availability'] = (df_persons['car_availability']!=c.CAR_AVAILABILITY_NEVER).astype(bool)
    df_persons['bike_availability'] = (df_persons['number_of_bikes_class']!=c.BIKE_AVAILABILITY_FOR_NONE).astype(bool)
    df_persons['is_car_passenger'] = df_persons['is_car_passenger'].fillna(False).astype(bool)
    df_persons['driving_license'] = df_persons['driving_license'].fillna(False).astype(bool)
    
    df_tours = df_tours.merge(df_persons[
                          ["person_id","age","car_availability","driving_license","bike_availability","is_car_passenger"]], 
                          on="person_id", how="left")
    
    # 4. get all possible mode combinations for each tour
    context.stage("mode_choice.tours.core")# make sure the stage is executed so that any function changes are taken into account
    persons_attributes = ["age","car_availability","driving_license","bike_availability","is_car_passenger"]
    res = tours_core.get_possible_mode_combinations_parallel(df_tours.crowfly_distance,
                                              df_tours[persons_attributes].to_dict(orient="records"),
                                              df_tours.preceding_purpose,
                                              df_tours.following_purpose)
    df_tours["mode_candidates"] = res

    # 5. finalize tours dataframe
    df_tours = df_tours[["person_id","trip_index","mode_candidates"]]
    df_tours = df_tours.explode("mode_candidates")
    return df_tours