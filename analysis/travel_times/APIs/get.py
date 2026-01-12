
from turtle import pd


def configure(context):
    context.config("travel_times_from", default="tomtom")
    assert context.config("travel_times_from").lower() in ["google", "tomtom", "mapbox", "all"], \
        "travel_times_from must be either 'google', 'tomtom', or 'mapbox'"
    
    if context.config("travel_times_from").lower() == "google":
        context.stage("analysis.travel_times.APIs.travel_times_google", alias="travel_times")
    elif context.config("travel_times_from").lower() == "mapbox":
        context.stage("analysis.travel_times.APIs.travel_times_mapbox", alias="travel_times")
    elif context.config("travel_times_from").lower() == "tomtom":
        context.stage("analysis.travel_times.APIs.travel_times_tomtom", alias="travel_times")
    else:
        context.stage("analysis.travel_times.APIs.travel_times_google")
        context.stage("analysis.travel_times.APIs.travel_times_mapbox")
        context.stage("analysis.travel_times.APIs.travel_times_tomtom")

def execute(context):
    travel_times_from = context.config("travel_times_from").lower()
    if travel_times_from == "all":
        df_google = context.stage("analysis.travel_times.APIs.travel_times_google")
        df_mapbox = context.stage("analysis.travel_times.APIs.travel_times_mapbox")
        df_tomtom = context.stage("analysis.travel_times.APIs.travel_times_tomtom")
        df = dict(google=df_google, 
                  mapbox=df_mapbox, 
                  tomtom=df_tomtom)
    else:
        df = {travel_times_from: context.stage("travel_times")}
    
    return df
    