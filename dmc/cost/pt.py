import numpy as np


euc_distance = lambda x,y: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

homdistance = lambda x: max(euc_distance((x['home_x'],x['home_y']),(x['origin_x'],x['origin_y'])),
                            euc_distance((x['home_x'],x['home_y']),(x['destination_x'],x['destination_y'])))*1e-3

def get_cost(df, context, distance_threshold_km = 10.0):    

    homeDistance_km = df.apply(homdistance, axis=1)
    in_vehicle_distance_km = df.pt_in_vehicle_distance_km    
    
    cost = np.maximum(2.8, 2*(0.21 * in_vehicle_distance_km - 0.00015 * in_vehicle_distance_km**2)) # cost = np.maximum(2.0, 0.6 * in_vehicle_distance_km)    
    
    #### cases with subscriptions, and age
    cost[df["hasHalbtaxSubscription"].fillna(False)] *= 0.5
    cost[df["hasGeneralSubscription"].fillna(False)] = 0.0
    cost[df["hasRegionalSubscription"].fillna(False) & (homeDistance_km < distance_threshold_km)] = 0.0

    cost[df["age"]<=6] = 0.0 # (source: https://www.sbb.ch/en/travel-information/individual-needs/travelling-with-children/tickets-travelcards.html#:~:text=The%20Junior%20Travelcard%20enables%20children,a%20valid%20ticket%20or%20travelcard.)
    cost[df["age"]<16] *= 0.5
    cost[(df["age"]<16)&df["hasJuniorSubscription"]] = 0.0
    
    between7and5 = (df["departure_time"]>=19*3600) | (df["departure_time"]<5*3600) # (source: https://www.sbb.ch/en/tickets-offers/travelcards/ga-travelcard/night-ga-travelcard.html)
    cost[(df["age"]<25)&df["hasGleis7Subscription"]&between7and5] = 0.0

    ### Limit pt cost per person per day to 50 CHF, split among their trips (daily pass)
    return np.clip(cost,0,50)