import logging
from mode_choice.dmc_defaults import Defaults

# for routing
from mode_choice.variables.walk import walk_travel_time, walk_distance
from mode_choice.variables.bike import bike_travel_time, bike_distance
from mode_choice.variables.pt import pt_variables
from mode_choice.variables.car import car_variables

# costs
from mode_choice.cost.car import car_cost
from mode_choice.cost.pt import pt_cost
from mode_choice.cost.parking import parking_cost

# penalties
from mode_choice.penalties.parking_search import parking_search_time

logger = logging.getLogger(__name__)

"""
In this stage I compute all the variables needed for the mode choice model.
"""
def configure(context):
    context.stage("mode_choice.estimate_model.data.survey_data")
    context.stage("data.constants")

    context.stage("mode_choice.network.network_processor")
    context.stage("mode_choice.network.car_router")
    context.config("routing_batch_size", default=Defaults.DEFAULT_CAR_ROUTING_BATCH_SIZE)

    context.config("walk_speed_m_per_s", default=Defaults.DEFAULT_WALK_SPEED_M_PER_S)  
    context.config("walk_distance_factor", default=Defaults.DEFAULT_WALK_DISTANCE_FACTOR) 
    context.config("bike_speed_m_per_s", default=Defaults.DEFAULT_BIKE_SPEED_M_PER_S) 
    context.config("bike_distance_factor", default=Defaults.DEFAULT_BIKE_DISTANCE_FACTOR) 
    
    context.config("pt_distance_factor", default=Defaults.DEFAULT_PT_DISTANCE_FACTOR)

    context.config("car_cost_per_km", default = Defaults.CAR_COST_PER_KM) #CHF per km
    context.config("parking_cost_per_hour_CHF_urban", default=Defaults.PARKING_COST_PER_HOUR_URBAN) #CHF per hour
    context.config("parking_cost_per_hour_CHF_suburban", default=Defaults.PARKING_COST_PER_HOUR_SUBURBAN) #CHF per hour

    context.config("urban_parking_search_min", default = Defaults.PARKING_SEARCH_MIN_URBAN)   
    context.config("suburban_parking_search_min", default = Defaults.PARKING_SEARCH_MIN_SUBURBAN)     

    context.config("pt_cost_model", default=Defaults.PT_COST_MODEL)
    context.config("car_cost_model", default = Defaults.CAR_COST_MODEL)
    
def execute(context):
    survey_data = context.stage("mode_choice.estimate_model.data.survey_data")[
        ['person_id', 'trip_id', 'origin_x', 'origin_y', 'destination_x', 'destination_y',
         'departure_time', 'euclidean_distance_km', 'number_of_cars', 'driving_license',
         'number_of_bikes_class','is_car_passenger', 'car_availability']
    ].copy()
    c = context.stage("data.constants")

    ########################## get availabilities ###########################
    # 1. walk
    survey_data["walk_availability"] = (survey_data['euclidean_distance_km'] <= Defaults.DEFAULT_WALK_THRESHOLD_KM)                        
    # 2. bike
    survey_data["bike_availability"] = ((survey_data['euclidean_distance_km'] <= Defaults.DEFAULT_BIKE_THRESHOLD_KM) &
                                        (survey_data["number_of_bikes_class"] != c.BIKE_AVAILABILITY_FOR_NONE))
    # 3. car
    survey_data["car_availability"]  = ((survey_data['car_availability']!=c.CAR_AVAILABILITY_NEVER) & 
                                        (survey_data['driving_license']) &
                                        (survey_data['euclidean_distance_km'] >= Defaults.DEFAULT_CAR_MIN_THRESHOLD_KM))
    # 4. public transport
    survey_data["pt_availability"]   = (survey_data['euclidean_distance_km'] >= Defaults.DEFAULT_PT_MIN_THRESHOLD_KM)    
    # 5. car passenger
    survey_data["car_passenger_availability"] = (survey_data['euclidean_distance_km'] >= Defaults.DEFAULT_CAR_MIN_THRESHOLD_KM)
    
    
    ################### compute travel times for each mode ###################
    # 1. walk trips
    walk_trips = survey_data.loc[survey_data["walk_availability"],
                                 ['person_id', 'trip_id', 'euclidean_distance_km']].reset_index(drop=True).copy()
    walk_trips["walk_distance_km"] = walk_distance(walk_trips["euclidean_distance_km"], context)  
    walk_trips["walk_travel_time_min"] = walk_travel_time(walk_trips["walk_distance_km"], context)
    walk_trips.drop(columns=["euclidean_distance_km"], inplace=True)
    # 2. bike trips
    bike_trips = survey_data.loc[survey_data["bike_availability"],
                                 ['person_id', 'trip_id', 'euclidean_distance_km']].reset_index(drop=True).copy()
    bike_trips["bike_distance_km"] = bike_distance(bike_trips["euclidean_distance_km"], context)
    bike_trips["bike_travel_time_min"] = bike_travel_time(bike_trips["bike_distance_km"], context)
    bike_trips.drop(columns=["euclidean_distance_km"], inplace=True)
    # 3. pt trips
    pt_trips = survey_data.loc[survey_data["pt_availability"],
                               ['person_id', 'trip_id', 'euclidean_distance_km']].reset_index(drop=True).copy()
    pt_trips = pt_variables(pt_trips, context)    
    pt_trips.columns = [c if c in ["person_id", "trip_id", "euclidean_distance_km"] else "pt_" + c for c in pt_trips.columns]    
    # 4. car and car passenger trips at once (to save routing calls)
    car_trips = survey_data.loc[survey_data["car_availability"]|survey_data["car_passenger_availability"],
                                ['person_id', 'trip_id','car_availability', 'car_passenger_availability',
                                 'origin_x', 'origin_y', 'destination_x', 'destination_y',  'departure_time']
                                ].reset_index(drop=True).copy()                
    car_vars = car_variables(car_trips, context)
    car_vars = car_vars[["person_id","trip_id", "travel_time_min","access_egress_time_min", "distance_km"]]
    car_vars.columns = ["person_id","trip_id", "car_travel_time_min","car_access_egress_time_min", "car_distance_km"]    
    car_vars = car_vars.merge(car_trips[["person_id","trip_id","car_availability", "car_passenger_availability"]], on=['person_id','trip_id'], how='left')

    cp_trips = car_vars[car_vars["car_passenger_availability"]].drop(columns=["car_availability","car_passenger_availability"]).reset_index(drop=True)
    cp_trips.columns = [c.replace('car_', 'car_passenger_') for c in cp_trips.columns]
    car_trips = car_vars[car_vars["car_availability"]].drop(columns=["car_availability","car_passenger_availability"]).reset_index(drop=True)


    ###################### compute costs for car and pt ###################
    # 1. car cost
    car_trips["car_cost_CHF"] = car_cost(context, car_trips["car_distance_km"], context.config("car_cost_per_km"))
    # 2. pt cost
    cost_pt = pt_trips[["person_id","trip_id","euclidean_distance_km"]].copy()
    subscriptions =  context.stage("mode_choice.estimate_model.data.survey_data")[
        ["person_id", "trip_id", "hasGeneralSubscription","hasHalbtaxSubscription",  "hasRegionalSubscription", 
         "hasJuniorSubscription","hasGleis7Subscription", "hasStreckenSubscription","hasVerbundSubscription",
         'age','destination_x', 'destination_y', 'origin_x', 'origin_y', 'home_x', 'home_y','departure_time']]
    cost_pt = cost_pt.merge(subscriptions, on=["person_id", "trip_id"], how="left")
    cost_pt["cost_CHF"]  = pt_cost(context, cost_pt)
    pt_trips["pt_cost_CHF"] = cost_pt["cost_CHF"]    
    pt_trips.drop(columns=["euclidean_distance_km"], inplace=True)

    ##################### compute parking variables ######################
    # parking cost
    parking_duration = context.stage("mode_choice.estimate_model.data.survey_data")[
        ["person_id", "trip_id", "parking_duration_wo_travelTime_min", "is_last",
         "destination_municipality", "purpose","departure_time"]
    ].rename(columns={"parking_duration_wo_travelTime_min": "parking_duration_min"}).copy()
    parking_duration = car_trips[["person_id","trip_id","car_travel_time_min","car_access_egress_time_min"]
                                 ].merge(parking_duration, on=["person_id","trip_id"], how="left")
    is_last = parking_duration["is_last"]
    parking_duration.loc[~is_last, "parking_duration_min"] -= (parking_duration.loc[~is_last, "car_travel_time_min"] + 
                                                               parking_duration.loc[~is_last, "car_access_egress_time_min"])
    parking_duration["parking_duration_min"] = parking_duration["parking_duration_min"].fillna(0.0).clip(0.0, 11 * 60.0)  # max 11 hours (from 8am to 7pm) and nans are last activities
    cost_parking = parking_cost(parking_duration, context)
    car_trips["parking_cost_CHF"] = cost_parking

    # parking search time
    parking_duration = parking_duration[["person_id","trip_id","destination_municipality","purpose"]].copy()
    parking_search_time_min = parking_search_time(parking_duration, context)
    car_trips["parking_searching_duration_min"] = parking_search_time_min

    
    ####################### merge all variables ######################
    df = survey_data[['person_id', 'trip_id', "euclidean_distance_km", "walk_availability", "bike_availability",
                      "car_availability", "pt_availability", "car_passenger_availability"]]
    df = df.merge(walk_trips, on=['person_id', 'trip_id'], how='left', suffixes=('', '_walk'))
    df = df.merge(bike_trips, on=['person_id', 'trip_id'], how='left', suffixes=('', '_bike'))
    df = df.merge(pt_trips,   on=['person_id', 'trip_id'], how='left', suffixes=('', '_pt'))
    df = df.merge(car_trips,  on=['person_id', 'trip_id'], how='left', suffixes=('', '_car'))
    df = df.merge(cp_trips,   on=['person_id', 'trip_id'], how='left', suffixes=('', '_car_passenger'))
    

    ####################### adjust availabilities ######################
    # 1. pt
    pt_unavailability = ((df["pt_in_vehicle_time_min"]<1) | 
                         (df["pt_in_vehicle_time_min"]>300) |
                         (df["pt_transfers"]>5) |
                         (df["pt_access_egress_time_min"]>60)|
                         (df["pt_distance_km"]>300))
    df.loc[pt_unavailability, "pt_availability"] = False
    # 2. car 
    car_unavailability = (df["car_travel_time_min"]<1) | (df["car_distance_km"]>300) | (df["car_access_egress_time_min"]>60)
    df.loc[car_unavailability, "car_availability"] = False        
    # 3. car passenger
    cp_unavailability = (df["car_passenger_travel_time_min"]<1) | (df["car_passenger_distance_km"]>300) | (df["car_passenger_access_egress_time_min"]>60)
    df.loc[cp_unavailability, "car_passenger_availability"] = False    

    ###################### return the dataframe ######################
    return df[[
        # Person and trip identifiers
        'person_id', 'trip_id',
        
        # Availabilities
        'walk_availability', 'bike_availability', 'car_availability', 'pt_availability', 'car_passenger_availability',
        
        # General distance
        'euclidean_distance_km',
        
        # Walk variables
        'walk_distance_km', 'walk_travel_time_min',
        
        # Bike variables
        'bike_distance_km', 'bike_travel_time_min',
        
        # PT variables
        'pt_access_egress_time_min', 'pt_waiting_time_min', 'pt_transfers', 'pt_in_vehicle_time_min', 'pt_distance_km', 'pt_cost_CHF',
        
        # Car variables
        'car_travel_time_min', 'car_access_egress_time_min', 'car_distance_km', 'car_cost_CHF', 'parking_cost_CHF', 'parking_searching_duration_min',
        
        # Car passenger variables
        'car_passenger_travel_time_min', 'car_passenger_access_egress_time_min', 'car_passenger_distance_km'
    ]]
