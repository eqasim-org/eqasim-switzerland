from .departure_time_utils import (load_trips, get_variation_window_min, prepare_trips_for_router, run_pt_router,
                                   compute_utilities, find_best_departure_times, 
                                   filter_trips, get_mz_departures)

def configure(context):
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.spatial.locations")
    context.stage("synthesis.population.enriched")
    context.stage("data.microcensus.trips")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.stage("calibration.pt_pricing.generate_config")
    context.stage("dmc.params")

    context.config("threads")
    context.config('random_seed')


def execute(context):
    # load all the trips from the pipeline
    df_trips = load_trips(context)
    original_departures = df_trips[["person_id", "trip_index", "departure_time"]].copy()
    # get the variation window for each trip (if 0, don't route, keep current departure time)
    df_trips["variation_window_min"] = get_variation_window_min(context, df_trips)
    # for those trips that will be routed, we try to use mz departure time instead
    df_trips["departure_time"] = get_mz_departures(context, df_trips)
    # filter the trips
    df_trips = filter_trips(context, df_trips)
    # prepare the trips for the router
    router_trips = prepare_trips_for_router(context, df_trips)
    # routed trips
    routed_trips = run_pt_router(context, router_trips)
    # compute utilities
    routed_trips["utility"] = compute_utilities(context, routed_trips)
    # find best departure times
    best_departure_times = find_best_departure_times(context, routed_trips, original_departures)

    return best_departure_times[["person_id", "trip_index", "best_departure_time"]]
  



def get_best_departue_time(context, df_trips):
    df = df_trips[["person_id","trip_index"]]
    best_departures = context.stage("synthesis.population.departure_times.trips_departures")[["person_id","trip_index","best_departure_time"]]
    df = df.merge(best_departures, on=["person_id","trip_index"], how="left")
    return df["best_departure_time"].values