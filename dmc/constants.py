import pandas as pd


class constants:
    REF_EUCLIDEAN_DISTANCE_KM = 15.0 # km
    REF_INCOME_CHF    = 6000.0 # CHF per month per capita
    TIME_SCALE_MIN    = 10.0   # minutes
    DISTANCE_SCALE_KM = 10.0   # km

    CAR_COST_PER_KM = 0.26  # CHF per km
    PARKING_COST_PER_HOUR_CHF_URBANCORE = 1.3  # CHF per hour
    PARKING_COST_PER_HOUR_CHF_URBAN = 1.0  # CHF per hour
    PARKING_COST_PER_HOUR_CHF_SUBURBAN = 0.5  # CHF per hour
    PARKING_PRICE_REDUCTION_FOR_WORK = 1.0  # x% of normal parking cost for work trips
    
    URBANCORE_PARKING_SEARCH_MIN = 3.0  # minutes
    URBAN_PARKING_SEARCH_MIN = 2.5  # minutes (source1: https://www.sciencedirect.com/science/article/pii/S0965856424000934)
    #                                         (source2(zurich, see validation data): https://link.springer.com/article/10.1007/s11116-017-9832-9)
    #                                         (source3(page 114): https://www.research-collection.ethz.ch/entities/publication/19cf9faa-55b4-4f01-96ab-fdfb9587032a)
    # ref for VoT: https://www.sciencedirect.com/science/article/pii/S0965856421001658#:~:text=We%20obtain%20median%20VTTS%20for,amounts%20to%2025.2%20CHF%2Fh.
    SUBURBAN_PARKING_SEARCH_MIN = 1.5  # minutes        

    # ms regions (clusters, bu mode shares)
    MS_REGIONS = pd.DataFrame({
        'canton_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 
        'cluster': [2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 0]}
                ).set_index("canton_id")
    
    # car cost model options: "simple" or "weiss"
    CAR_COST_MODEL = "weiss"

    # for old pt cost model
    PT_REGIONAL_RADIUS_KM = 10.0  # km