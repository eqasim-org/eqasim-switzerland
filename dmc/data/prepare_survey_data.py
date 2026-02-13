import pandas as pd
import numpy as np
import os
import geopandas as gpd
import shapely.geometry as geo
from shapely import vectorized
from matsim.scenario.network.utils.elevation_estimator import ElevationEstimator
import logging
from dmc.constants import constants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MS_REGIONS = constants.MS_REGIONS

def configure(context):
    context.config("data_path")

    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")    
    context.stage("data.spatial.swiss_border")
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")    
    context.stage("data.constants")

def execute(context):
    df_persons = context.stage("data.microcensus.persons")    
    df_trips,filterout_person_ids = context.stage("data.microcensus.trips")
    filterout_person_ids = set(filterout_person_ids)
    logger.info(f"There are {len(df_trips)} trips in total.")
    logger.info(f"There are {len(df_persons)} persons in total.")

    # sort trips by person and by trip
    df_trips = df_trips.sort_values(["person_id","trip_id"])
    df_trips = df_trips[['person_id', 'trip_id', 'departure_time', 'mode', 'purpose','destination_x',
                         'destination_y', 'origin_x', 'origin_y', 'crowfly_distance']].reset_index(drop=True)
    
    # include personal information
    # 1. subscriptions
    df_persons["hasGeneralSubscription"] = df_persons.subscriptions_ga
    df_persons["hasHalbtaxSubscription"] = df_persons.subscriptions_halbtax
    df_persons["hasRegionalSubscription"] = df_persons.subscriptions_verbund | df_persons.subscriptions_strecke
    df_persons["hasJuniorSubscription"] = df_persons.subscriptions_junior
    df_persons["hasGleis7Subscription"] = df_persons.subscriptions_gleis7
    df_persons["statedPreferenceRegion"] = df_persons.sp_region
    # 2. income equivalence scale
    c = context.stage("data.constants")
    df_persons["income"] = df_persons.income_class.map(c.INCOME_CLASS_MAP)
    num_children = df_persons["N_children_under_12"]
    num_adults = np.maximum(1, df_persons['household_size'] - num_children)
    equvalent_size =  1 + 0.5 * (num_adults - 1) + 0.3 * num_children    
    df_persons["income"] = df_persons["income"] / equvalent_size
    df_persons["low_income"] = df_persons["income"] <= c.LOW_INCOME_THRESHOLD

    # 3. sp_region and ms_region
    df_persons["ms_region"] = df_persons.canton_id.map(lambda x: MS_REGIONS.loc[x,"cluster"])
    # 4. car ration
    df_persons["car_ownership_ratio"] = np.clip(1 - df_persons["number_of_cars_class"]/df_persons["N_adults"],0,1)
    # 5. pt quality
    df_persons["good_pt_service"] = (df_persons["ovgk"].isin(["A", "B"])).astype(int)
    df_persons["medium_pt_service"] = (df_persons["ovgk"].isin(["C","D"])).astype(int)
    # 6. age related
    df_persons["is_retired"] = (df_persons["age"]>=65).astype(int)
    df_persons["is_junior"] = (df_persons["age"]<16).astype(int)
    # 7. merge
    cols = ["person_id","home_x","home_y", "hasGeneralSubscription","hasHalbtaxSubscription","hasRegionalSubscription", "hasJuniorSubscription", 
            "hasGleis7Subscription", "statedPreferenceRegion", 'person_weight', 'age', 'sex', 'driving_license', 'sp_region', 'ms_region', "ovgk",
            'is_car_passenger', "income", "weekend", "good_pt_service", "medium_pt_service", "car_ownership_ratio", "is_retired", "is_junior", "low_income", "income_class"]
    df_persons = df_persons[cols]    
    df_trips = df_trips.merge(df_persons, on="person_id", how="left")

    # updates filtered persons to include the weekend trips
    filterout_person_ids.update(df_persons[df_persons["weekend"]==True].person_id.tolist())
    
    # enrich trips with additional information
    df_trips["destination_home"] = df_trips.purpose.isin(["home", "home_secondary"])
    df_trips["origin_home"] = (df_trips.origin_x == df_trips.home_x) & (df_trips.origin_y == df_trips.home_y)    
    df_trips["destination_work"] = df_trips.purpose.isin(["work","work_secondary"])
    df_trips["destination_education"] = df_trips.purpose.isin(["education","education_secondary"])
    df_trips["destination_shopping"] = df_trips.purpose.isin(["shop"])
    df_trips["destination_leisure"] = df_trips.purpose.isin(["leisure"])
    df_trips["destination_other"] = df_trips.purpose.isin(["other"])    
    df_trips["euclidean_distance_km"] = df_trips.crowfly_distance*1e-3
    df_trips["is_first"] = df_trips["person_id"].shift(1) != df_trips["person_id"]
    df_trips["is_last"]  = df_trips["person_id"].shift(-1) != df_trips["person_id"]

    # estimate parking duration without travel time
    parking_duration_min = (np.clip(df_trips["departure_time"].shift(-1), 8*3600, 19*3600) - 
                            np.clip(df_trips["departure_time"], 8*3600, 19*3600)) / 60.0

    parking_duration_min[parking_duration_min<=0] = np.nan  # doesn't pay parking (duration out of bounds)
    parking_duration_min[df_trips["is_last"].values] = np.nan # doesn't pay parking (home parking at night)

    df_trips["parking_duration_wo_travelTime_min"] = parking_duration_min
    
    #When weight is unavailable, I assume that the weight is equal to the average to keep the person.
    df_trips["person_weight"] = df_trips["person_weight"].fillna(round(df_trips["person_weight"].mean(),2))

    # include the municipality
    df_municipality_type = context.stage("data.spatial.municipality_types")
    df_municipalities,_ = context.stage("data.spatial.municipalities")
    df_municipalities = df_municipalities.merge(df_municipality_type)[["municipality_type","geometry"]]

    # Create geometry columns for home, origin, and destination
    df_trips["geometry_home"] = gpd.points_from_xy(df_trips.home_x, df_trips.home_y)
    df_trips["geometry_origin"] = gpd.points_from_xy(df_trips.origin_x, df_trips.origin_y)
    df_trips["geometry_destination"] = gpd.points_from_xy(df_trips.destination_x, df_trips.destination_y)

    # Spatial join for home municipality
    df_home = gpd.GeoDataFrame(df_trips, geometry="geometry_home", crs="EPSG:2056")
    assert df_home.crs == df_municipalities.crs
    df_home = df_home.sjoin(df_municipalities, how="left", predicate="intersects")
    df_trips["home_municipality"] = df_home["municipality_type"]

    # Spatial join for origin municipality
    df_origin = gpd.GeoDataFrame(df_trips, geometry="geometry_origin", crs="EPSG:2056")
    df_origin = df_origin.sjoin(df_municipalities, how="left", predicate="intersects")
    df_trips["origin_municipality"] = df_origin["municipality_type"]

    # Spatial join for destination municipality
    df_dest = gpd.GeoDataFrame(df_trips, geometry="geometry_destination", crs="EPSG:2056")
    df_dest = df_dest.sjoin(df_municipalities, how="left", predicate="intersects")
    df_trips["destination_municipality"] = df_dest["municipality_type"]

    # within switzerland
    df_switzerland = context.stage("data.spatial.swiss_border")
    ch_polygon = df_switzerland.buffer(0).iloc[0] 
    inside_origin = vectorized.contains(ch_polygon, df_trips["origin_x"].values, df_trips["origin_y"].values)
    inside_destination = vectorized.contains(ch_polygon, df_trips["destination_x"].values, df_trips["destination_y"].values)
    df_trips["inside_ch"] = inside_origin&inside_destination    
    
    ### filter
    df_trips = df_trips[df_trips["euclidean_distance_km"]>1e-2] # remove trips with less than 10m
    df_trips = df_trips[~df_trips.person_id.isin(filterout_person_ids)] # persons that need to be removed
    df_trips = df_trips[df_trips["inside_ch"]==True] # only trips within Switzerland
    df_trips = df_trips[df_trips.isna().sum(axis=1)==0] # remove trips with incomplete data
    df_trips = df_trips.reset_index(drop=True)

    logger.info(f"There are {len(df_trips)} trips after cleaning.")

    ### last thing, estimate elevations 'after filters to limit the number of queries)
    origins = [(x,y) for x,y in zip(df_trips["origin_x"], df_trips["origin_y"])]
    origins_elevations = ElevationEstimator(data_path = context.config("data_path"),
                                            coordinates = origins)
    origins_elevations = origins_elevations.run()

    destinations = [(x,y) for x,y in zip(df_trips["destination_x"], df_trips["destination_y"])]
    destination_elevations = ElevationEstimator(data_path = context.config("data_path"),
                                                coordinates = destinations)
    destination_elevations = destination_elevations.run()
    
    df_trips["elevation_origin"] = origins_elevations["z"]
    df_trips["elevation_destination"] = destination_elevations["z"]
    df_trips["elevation_difference"] = df_trips["elevation_destination"] - df_trips["elevation_origin"]

    ### Assertions
    assert all(df_trips["weekend"]==False)     
    assert (set(df_trips['home_municipality'].unique()) ==
            set(df_trips['origin_municipality'].unique()) ==
            set(df_trips['destination_municipality'].unique())== {'rural', 'suburban', 'urban', 'urbancore'})
    assert df_trips["elevation_difference"].notna().all()

    ### return
    cols = ['person_id', 'trip_id', 'departure_time', 'mode', 'purpose',
            'destination_x', 'destination_y', 'origin_x', 'origin_y',
            'home_x', 'home_y', 'hasGeneralSubscription', 'hasJuniorSubscription', 'hasGleis7Subscription',
            'hasHalbtaxSubscription', 'hasRegionalSubscription', 'ovgk', 'car_ownership_ratio', "good_pt_service", "medium_pt_service",
            'statedPreferenceRegion', 'person_weight', 'age', 'sex', 'is_retired','is_junior','low_income',
            'driving_license', 'sp_region', 'ms_region', 'is_car_passenger', 'income', 'income_class',
            'destination_home', 'origin_home', 'destination_work', 'destination_education',
            'destination_shopping', 'destination_leisure', 'destination_other',
            'euclidean_distance_km', 'is_first', 'is_last',
            'parking_duration_wo_travelTime_min', 'home_municipality',
            'origin_municipality', 'destination_municipality', 'inside_ch',
            'elevation_destination','elevation_origin', 'elevation_difference']

    return df_trips[cols]
