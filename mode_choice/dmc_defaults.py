import pandas as pd

#TODO: add this class synpp dependency in all the stages where it is used

class Defaults:
    # Mode names
    MODE_CAR = "car"
    MODE_PT = "pt"
    MODE_BIKE = "bike"
    MODE_WALK = "walk"
    MODE_CP = "car_passenger"

    # Modes we consider
    POSSIBLE_MODES = set([MODE_CAR, MODE_PT, MODE_BIKE, MODE_WALK, MODE_CP])
    
    # spatial continuity constraints
    MODE_CONTINUITY = set([MODE_CAR, MODE_BIKE])

    # default thresholds (Euclidean distance) (km)
    DEFAULT_WALK_THRESHOLD_KM = 4.5 # quantile 99% of all walk trips in microcensus (euclidean distance)
    DEFAULT_BIKE_THRESHOLD_KM = 11 # quantile 99% of all bike trips in microcensus (euclidean distance)
    DEFAULT_CAR_MIN_THRESHOLD_KM = 0.15
    DEFAULT_PT_MIN_THRESHOLD_KM = 0.15
    DEFAULT_PASSENGER_MIN_THRESHOLD_KM = 0.15

    # speeds and distance factors
    DEFAULT_WALK_SPEED_M_PER_S = 1.3  
    DEFAULT_WALK_DISTANCE_FACTOR = 1.3  # factor to account for indirect walking paths
    DEFAULT_BIKE_SPEED_M_PER_S = 4.0 
    DEFAULT_BIKE_DISTANCE_FACTOR = 1.4  # factor to account for indirect biking paths
    DEFAULT_PT_DISTANCE_FACTOR = 1.4

    # MS regions and income mapping (population segments)    
    MS_REGIONS = pd.DataFrame(
        {'canton_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 
         'cluster': [2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 0]}
        ).set_index("canton_id")
    
    INCOME_CLASS_MAP = {0: 2000, 1: 3000, 2: 5000, 3: 7000, 4: 9000, 5: 11000,  6: 13000, 7: 15000, 8: 17000}

    WORKING_HOURS = [8 , 17] # start and end of working hours
    # Pt cost parameters 
    PT_COST_DISTANCE_THRESHOLD_KM = 10.0    
    PT_COST_MODEL = "simple"  # "simple" or "detailed"
    
    # car cost parameters
    CAR_COST_PER_KM = 0.26  # CHF per km
    CAR_COST_MODEL = "simple"  # "simple" or "weiss"
    
    # parking
    PARKING_COST_PER_HOUR_URBAN = 1.0  # CHF per hour
    PARKING_COST_PER_HOUR_SUBURBAN = 0.5  # CHF per hour
    PARKING_SEARCH_MIN_URBAN = 2.0  # minutes
    PARKING_SEARCH_MIN_SUBURBAN = 1.0  # minutes

    # routing params
    DEFAULT_CAR_ROUTING_BATCH_SIZE = 4096

    # model related params
    USE_EXPONENTS_IN_MODE_CHOICE = True
    INCOME_COST_INTERACTION = True
    DISTANCE_COST_INTERACTION = True
    SHORT_DISTANCE_LIMIT_KM = 1.5 
    LONG_DISTANCE_LIMIT_KM = 13.0 

    # calibration defaults
    ESTIMATE_DMC_PARAMETERS = True
    CALIBRATE_DMC_PARAMETERS = True

    # PT router and cost
    USE_SKIM_MATRICES = False
    
    # model estimation data
    MERGE_TRIPS_THAT_MIGHT_BE_SAME_TRIP = False

def configure(context):
    pass

def execute(context):
    return Defaults()