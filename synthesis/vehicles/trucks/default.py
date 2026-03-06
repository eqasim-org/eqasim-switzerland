import re
import pandas as pd

"""
Creates a vehicle fleet based on a default vehicle type for the dummy truck mode
"""

def configure(context):
    context.stage("synthesis.freight.trips")

def execute(context):
    df_persons = context.stage("synthesis.freight.trips")

    # the older PCE is 3, I changed it to 2.
    # this needs to be discussed and tested, but I think 2 is more reasonable for the moment
    # it is true that a truck can occupy the size of 3 vehicles or more, but storage capacity in traffic is measured as density, and 
    # trucks affect density because it is a one long vehicle without any gaps, so I think 2 is more reasonable for the moment.
    # the value of 2 can be obtained from HCM, From Chapter 12, Basic Freeway and Multilane Highway Segments, the passenger car equivalent ET
    # for trucks in level terrain is 2.0. 
    # link: https://nap.nationalacademies.org/resource/26432/Highway_Capacity_Manual_Edition_7.1_Chapters.pdf
    # However, here they used 4: https://www.simunto.com/assets/files/matsim/tutorials/2023-osc/slides_day4.pdf?utm_source=chatgpt.com

    # Trucks also reduce the flow capacity on roads.
    # https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2019.00077/full
    # I'll use a small reduction though, this also should be tested and documented further

    df_vehicle_types = pd.DataFrame.from_records([{
        "type_id": "default_truck", "nb_seats": 1, "length": 12.0, "width": 1.0, "pce": 1.8, "mode": "truck",
        "hbefa_cat": "HEAVY_GOODS_VEHICLE", "hbefa_tech": "average", "hbefa_size": "average", "hbefa_emission": "average",
        "maxVelocity": round(130/3.6, 2), "flowEfficiencyFactor": 1.0
    }])

    df_vehicles = df_persons[["agent_id"]].copy()
    df_vehicles = df_vehicles.rename(columns = { "agent_id": "owner_id" })
    
    df_vehicles["mode"] = "truck"

    df_vehicles["vehicle_id"] = df_vehicles["owner_id"].astype(str) + ":truck"
    df_vehicles["type_id"] = "default_truck"
    df_vehicles["age"] = 0
    df_vehicles["euro"] = 6

    return df_vehicle_types, df_vehicles