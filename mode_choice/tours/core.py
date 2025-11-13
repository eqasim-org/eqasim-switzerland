from typing import List, Dict, Any, Tuple
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
import time
import logging

logger = logging.getLogger(__name__)

# Modes we consider
POSSIBLE_MODES = set(['car', 'public_transport', 'bike', 'walk', 'car_passenger'])

# default thresholds (Euclidean distance) (km)
DEFAULT_WALK_THRESHOLD = 4.0
DEFAULT_BIKE_THRESHOLD = 8.0
DEFAULT_CAR_MIN_THRESHOLD = 0.15
DEFAULT_PT_MIN_THRESHOLD = 0.15
DEFAULT_PASSENGER_MIN_THRESHOLD = 0.15
# spatial continuity constraints
MODE_CONTINUITY = set(["car", "bike"] )


def get_possible_mode_combinations_parallel(distances: List[List[float]],
                                            persons: List[Dict[str, Any]],
                                            preceding_activities: List[List[str]] = None,
                                            following_activities: List[List[str]] = None
                                            ) -> List[Tuple[str, ...]]:
    
    logger.info("Starting parallel computation of possible mode combinations for each tour")
    start_time = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(get_possible_mode_combinations)(d, p, pa, fa)
        for d, p, pa, fa in 
        zip(distances, persons, preceding_activities, following_activities)
    )
    end_time = time.time()
    logger.info(f"\t Parallel computation for {len(results)} tours completed in {end_time - start_time:.2f} seconds")
    return results

def get_possible_mode_combinations(distances: List[float],
                                   person: Dict[str, Any],
                                   preceding_activities: List[str] = None,
                                   following_activities: List[str] = None,
                                   ) -> List[Tuple[str, ...]]:
    """
    Given a list of crowfly distances for the trips in a tour, return all possible mode combinations
    that can be used for the tour based on distance thresholds for walking and biking.
    """    
    possible_modes_per_trip = get_mode_candidates_per_trip(distances, person)

    # Generate all combinations of modes for the trips in the tour    
    all_combinations = list(product(*possible_modes_per_trip))

    # Apply spatial continuity constraints
    filtered_combinations = spatial_continuity_filter(all_combinations, preceding_activities, following_activities) 

    return filtered_combinations

def get_mode_candidates_per_trip(distances: List[float], person: dict) -> List[str]:
    possible_modes_for_agent = get_available_modes_per_agent(person)
    possible_modes_per_trip = []    
    for distance in distances:        
        modes = possible_modes_for_agent.copy()
        if distance > DEFAULT_WALK_THRESHOLD:
            modes.discard('walk')
        if distance > DEFAULT_BIKE_THRESHOLD:
            modes.discard('bike')
        if distance < DEFAULT_PT_MIN_THRESHOLD:
            modes.discard('public_transport')
        if distance < DEFAULT_CAR_MIN_THRESHOLD:
            modes.discard('car')
        if distance < DEFAULT_PASSENGER_MIN_THRESHOLD:
            modes.discard('car_passenger')
        possible_modes_per_trip.append(modes)
    
    return possible_modes_per_trip

def get_available_modes_per_agent(person:dict):
    possible_modes = POSSIBLE_MODES.copy()
    
    # check car availability
    car_availability = (person["driving_license"] and 
                        person["age"] >= 18 and 
                        person["car_availability"])
    if not car_availability:
        possible_modes.discard("car")
    
    # check bike availability
    if not person["bike_availability"]:
        possible_modes.discard("bike")
    
    # check car_passenger availability
    cp_availability = person["car_availability"] or person["is_car_passenger"]
    if not cp_availability:
        possible_modes.discard("car_passenger")
    
    return possible_modes

def spatial_continuity_filter(all_combinations: List[Tuple[str, ...]],
                              preceding_activities: List[str], 
                              following_activities: List[str]) -> List[Tuple[str, ...]]:
    
    if not MODE_CONTINUITY:
        return all_combinations
        
    filtered_combinations = []
    for combination in all_combinations:
        first_activity = preceding_activities[0]
        
        # Initialize mode locations for this combination
        modes_locations = {mode: first_activity for mode in MODE_CONTINUITY}
        
        valid = True
        for i, mode in enumerate(combination):
            if mode in MODE_CONTINUITY:
                if modes_locations[mode] != preceding_activities[i]:
                    valid = False
                    break
                # Update location to next activity
                modes_locations[mode] = following_activities[i]

        # if the person took the modes in the mode_continuity_set, it needs to bring it back home in the last trip        
        for mode in MODE_CONTINUITY:
            mode_back_home = (modes_locations[mode] == first_activity) or (modes_locations[mode] == "home")
            if not mode_back_home:
                valid = False
                break
        
        if valid:
            filtered_combinations.append(combination)
    
    return filtered_combinations




# I include this as a stage so that if we modify some functions, it will be considered in synpp
def configure(context):
    pass    
def execute(context):
    return get_possible_mode_combinations_parallel