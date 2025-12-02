from mode_choice.dmc_defaults import Defaults

def configure(context):
    context.stage("mode_choice.dmc_defaults")
    context.stage("mode_choice.trips.prepare_trips")
    context.config("bike_speed_m_per_s", default=Defaults.DEFAULT_BIKE_SPEED_M_PER_S)  # average biking speed ~14.4 km/h
    context.config("bike_distance_factor", default=Defaults.DEFAULT_BIKE_DISTANCE_FACTOR)  # factor to account for indirect biking paths


def bike_travel_time(distance_km, context):
    bike_speed_m_per_s = context.config("bike_speed_m_per_s")
    travel_time_min = (distance_km * 1e3 / bike_speed_m_per_s) / 60
    return travel_time_min

def bike_distance(euclidean_distance_km, context):
    bike_distance_factor = context.config("bike_distance_factor")
    adjusted_distance_km = euclidean_distance_km * bike_distance_factor
    return adjusted_distance_km

def execute(context):
    # read prepared trips
    trips = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id","trip_id","euclidean_distance_km"]
    ].copy()
    
    # calculate biking distance    
    trips["distance_km"] = bike_distance(trips["euclidean_distance_km"], context)
    
    # calculate biking travel time in seconds    
    trips["travel_time_min"] = bike_travel_time(trips["distance_km"], context)

    return trips[["person_id","trip_id","travel_time_min","distance_km"]]

    
    
    

    