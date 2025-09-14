import pandas as pd
import numpy as np
import os
import geopandas as gpd
import shapely.geometry as geo
from shapely import vectorized
from matsim.scenario.network.utils.elevation_estimator import ElevationEstimator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MS_REGIONS = {'canton_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 
              'cluster': [2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 0]}
MS_REGIONS = pd.DataFrame(MS_REGIONS)
MS_REGIONS = MS_REGIONS.set_index("canton_id")

def configure(context):
    context.config("data_path")

    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")    
    context.stage("data.spatial.swiss_border")
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")

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
    df_persons["hasGeneralSubscription"] = df_persons.subscriptions_ga
    df_persons["hasHalbtaxSubscription"] = df_persons.subscriptions_halbtax
    df_persons["hasRegionalSubscription"] = df_persons.subscriptions_verbund | df_persons.subscriptions_strecke
    df_persons["hasJuniorSubscription"] = df_persons.subscriptions_junior
    df_persons["hasGleis7Subscription"] = df_persons.subscriptions_gleis7
    df_persons["statedPreferenceRegion"] = df_persons.sp_region
    

    INCOME_CLASS_MAP = {0: 2000, 1: 3000, 2: 4500, 3: 7000, 4: 9000, 5: 11000,  6: 13000, 7: 15000, 8: 17000}
    df_persons["income"] = df_persons.income_class.map(INCOME_CLASS_MAP)
    df_persons["income"] = df_persons["income"] / df_persons["household_size"].fillna(1).clip(lower=1, upper=7)

    df_persons["ms_region"] = df_persons.canton_id.map(lambda x: MS_REGIONS.loc[x,"cluster"])

    cols = ["person_id","home_x","home_y", "hasGeneralSubscription","hasHalbtaxSubscription","hasRegionalSubscription", "hasJuniorSubscription", 
            "hasGleis7Subscription", "statedPreferenceRegion", 'person_weight', 'age', 'sex', 'driving_license', 'sp_region', 'ms_region',
             'is_car_passenger', "income", "weekend"]
    df_persons = df_persons[cols]    
    df_trips = df_trips.merge(df_persons, on="person_id", how="left")

    # updates filtered persons to include the weekend trips
    filterout_person_ids.update(df_persons[df_persons["weekend"]==True].person_id.tolist())
    
    # correct trip information
    df_trips["destination_home"] = df_trips.purpose == "home"
    df_trips["origin_home"] = (df_trips.origin_x == df_trips.home_x) & (df_trips.origin_y == df_trips.home_y)    
    df_trips["destination_work"] = df_trips.purpose == "work"
    
    df_trips["euclidean_distance_km"] = df_trips.crowfly_distance*1e-3

    df_trips["is_first"] = df_trips["person_id"].shift(1) != df_trips["person_id"]
    df_trips["is_last"]  = df_trips["person_id"].shift(-1) != df_trips["person_id"]

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
    df_trips = df_trips[df_trips["euclidean_distance_km"]>0.01] # remove trips with less than 10m
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
    assert set(df_trips['home_municipality'].unique())==set(df_trips['origin_municipality'].unique())==set(df_trips['destination_municipality'].unique())== {'rural', 'suburban', 'urban'}
    assert df_trips["elevation_difference"].notna().all()

    ### return
    cols = ['person_id', 'trip_id', 'departure_time', 'mode', 'purpose',
            'destination_x', 'destination_y', 'origin_x', 'origin_y',
            'home_x', 'home_y', 'hasGeneralSubscription', 'hasJuniorSubscription', 'hasGleis7Subscription',
            'hasHalbtaxSubscription', 'hasRegionalSubscription',
            'statedPreferenceRegion', 'person_weight', 'age', 'sex',
            'driving_license', 'sp_region', 'ms_region', 'is_car_passenger', 'income', 'weekend',
            'destination_home', 'origin_home', 'destination_work',
            'euclidean_distance_km', 'is_first', 'is_last',
            'parking_duration_wo_travelTime_min', 'home_municipality',
            'origin_municipality', 'destination_municipality', 'inside_ch',
            'elevation_destination','elevation_origin', 'elevation_difference']

    return df_trips[cols]
