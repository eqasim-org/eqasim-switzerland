import numpy as np
import geopandas as gpd
from shapely import vectorized
import logging
from mode_choice.dmc_defaults import Defaults
from .utils import merge_same_trips

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MS_REGIONS = Defaults.MS_REGIONS
INCOME_CLASS_MAP = Defaults.INCOME_CLASS_MAP

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
    # 1. subscriptions
    df_persons["hasGeneralSubscription"] = df_persons.subscriptions_ga
    df_persons["hasHalbtaxSubscription"] = df_persons.subscriptions_halbtax
    df_persons["hasRegionalSubscription"] = df_persons.subscriptions_verbund | df_persons.subscriptions_strecke
    df_persons["hasJuniorSubscription"] = df_persons.subscriptions_junior
    df_persons["hasGleis7Subscription"] = df_persons.subscriptions_gleis7
    df_persons["statedPreferenceRegion"] = df_persons.sp_region
    df_persons["hasVerbundSubscription"] = df_persons.subscriptions_verbund
    df_persons["hasStreckenSubscription"] = df_persons.subscriptions_strecke
    # 2. income
    df_persons["income"] = df_persons.income_class.map(INCOME_CLASS_MAP)
    num_children = df_persons["N_children_under_12"]
    num_adults = np.maximum(1, df_persons['household_size'] - num_children)
    equvalent_size =  1 + 0.5 * (num_adults - 1) + 0.3 * num_children    
    df_persons["income"] = df_persons["income"] / equvalent_size
    # 3. ms_region
    df_persons["region"] = df_persons.canton_id.map(lambda x: MS_REGIONS.loc[x,"cluster"])

    cols = ["person_id","home_x","home_y", "hasGeneralSubscription","hasHalbtaxSubscription","hasRegionalSubscription", "hasJuniorSubscription", 
            "hasGleis7Subscription", "statedPreferenceRegion", 'person_weight', 'age', 'sex', 'driving_license', 'region',
             'is_car_passenger', "income", "number_of_cars","number_of_bikes_class", "weekend", "car_availability"]
    df_persons = df_persons[cols]  
    # 4. merge  
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

    # estimate parking duration without travel time
    parking_duration_min = (np.clip(df_trips["departure_time"].shift(-1), 8*3600, 19*3600) - 
                            np.clip(df_trips["departure_time"], 8*3600, 19*3600)) / 60.0

    parking_duration_min[parking_duration_min<=0] = np.nan  # doesn't pay parking (duration out of bounds)
    parking_duration_min[df_trips["is_last"].values] = np.nan # doesn't pay parking (home parking at night)
    df_trips["parking_duration_wo_travelTime_min"] = parking_duration_min
    
    # When weight is unavailable, I assume that the weight is equal to the average to keep the person.
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

    cols = ['person_id', 'trip_id', 'departure_time', 'mode', 'purpose','destination_x', 'destination_y', 'origin_x', 'origin_y',
            'home_x', 'home_y', 'hasGeneralSubscription', 'hasJuniorSubscription', 'hasGleis7Subscription', 'hasHalbtaxSubscription',
            'hasVerbundSubscription', 'hasStreckenSubscription', 'hasRegionalSubscription', 'person_weight', 'age', 'sex',
            "number_of_cars", "number_of_bikes_class", 'driving_license', 'region', 'is_car_passenger', 'income', 'weekend', 
            "car_availability",'destination_home', 'origin_home', 'destination_work', 'euclidean_distance_km', 'is_first', 
            'is_last', 'parking_duration_wo_travelTime_min', 'home_municipality', 'origin_municipality', 'destination_municipality']
    df_trips = df_trips[cols]

    logger.info(f"There are {len(df_trips)} trips after cleaning.")

    ### merge same trips
    # here I merge the trips that are supposed to be part of the same trip
    # If the arrival time of trip ``i`` is equal to the departure time of trip ``i+1``, and the mode is the same, merge them
    df_trips = merge_same_trips(context, df_trips)

    ### Assertions
    assert (df_trips["weekend"]==False).all(), "Weekend trips are not allowed in the final dataset."   
    assert set(df_trips['home_municipality'].unique())==set(df_trips['origin_municipality'].unique())==set(df_trips['destination_municipality'].unique())== {'rural', 'suburban', 'urban'}
    assert df_trips.isna().sum().sum() == 0, "There should be no missing values in the final dataset." 

    ### return
    return df_trips
