"""
This stage provides the public transport variables that are used in the mode choice model.
For now, it returns zeros for all variables.

Aurore will implement this stage when she finishes her model of pt travel times.
"""

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")


def pt_variables(df, context):    
    df["access_egress_time_min"] = 5
    df["waiting_time_min"] = 5
    df["transfers"] = 1
    df["in_vehicle_time_min"] = 20
    df["distance_km"] = 4 # this is the network distance traveled by pt
    return df

def execute(context):
    df = context.stage("mode_choice.trips.prepare_trips")[["person_id","trip_id"]].copy() # you can add whatever variables you want here
    df = pt_variables(df, context)
    return df