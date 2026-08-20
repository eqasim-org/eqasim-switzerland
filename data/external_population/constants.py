import pandas as pd
import numpy as np
from data.utils import coerce_boolean_series

class ExternalPopulationConstants:
    canton_name       = "fr"
    ovgk              = "fr"
    municipality_name = "fr"
    municipality_type = "urban" # for now, we suppose it is urban, we will need to impute it later by subregion
    municipality_id   = -1
    sp_region         = -1
    canton_id         = -1

    person_type = "FR"
    is_external = "isExternalFR"

    @staticmethod
    def convert_sex(sex):
        sex = sex.values
        sex_array = np.zeros(len(sex), dtype=int)
        # sex.loc[sex=="male"]   = 0
        sex_array[sex=="female"] = 1
        return sex_array
    
    @staticmethod
    def get_subscriptions(df): 
        df = df.copy()
        subscription_cols = ["subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "subscriptions_strecke", "subscriptions_junior"]
        
        for col in subscription_cols:
            if col not in df.columns:
                df[col] = False
            df[col] = coerce_boolean_series(df[col], name=col)

        df = df[subscription_cols + ["age"]].reset_index(drop = True)
        
        # Combined subscriptions
        df["ga"] = df["subscriptions_ga"] | (df["subscriptions_junior"] & (pd.to_numeric(df["age"], errors = "coerce") < 16))
        df["vb"] = df["subscriptions_verbund"] | df["subscriptions_strecke"]
        
        # This is the list of subscriptions: ["none", "GA", "VA", "HT", "VA+HT"]
        subscriptions = np.zeros(len(df), dtype=int)
        subscriptions[df["ga"]] = 1
        subscriptions[~df["ga"] & df["vb"] & ~df["subscriptions_halbtax"]] = 2
        subscriptions[~df["ga"] & df["subscriptions_halbtax"] & ~df["vb"]] = 3
        subscriptions[~df["ga"] & df["subscriptions_halbtax"] & df["vb"]] = 4
        return subscriptions

    @staticmethod
    def convert_car_availability(car_availability):
        conversion_dict = {"never": 0, "always": 1}
        return car_availability.map(conversion_dict).astype(int).values
    
    @staticmethod
    def convert_bike_availability(bike_availability,cst):        
        return (bike_availability!=cst.BIKE_AVAILABILITY_NEVER).astype(int).values

def configure(context):
    pass

def execute(context):
    return ExternalPopulationConstants()
