"""
This stage provides the public transport variables that are used in the mode choice model.
For now, it returns zeros for all variables.

Aurore will implement this stage when she finishes her model of pt travel times.
"""

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")

def execute(context):
    df = context.stage("mode_choice.trips.prepare_trips")[["person_id","trip_id"]].copy()
    df["travel_time_min"] = 0
    df["access_egress_time_min"] = 0
    df["waiting_time_min"] = 0
    df["number_of_transfers"] = 0
    df["in_vehicle_time_min"] = 0
    df["distance_km"] = 0
    return df