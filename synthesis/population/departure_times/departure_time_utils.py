import pandas as pd
import numpy as np
import matsim.runtime.eqasim as eqasim
import os
from .dmc_parameters_class import Parameters
import logging
logger = logging.getLogger("synpp")

def load_trips(context):
    # Load only the minimal columns needed for fast processing
    df_trips = context.stage("synthesis.population.trips")[["person_id", "trip_index", "departure_time", "preceding_purpose", "following_purpose"]].copy()

    # Define origin/destination activity indices
    df_trips["preceding_activity_index"] = df_trips["trip_index"]
    df_trips["following_activity_index"] = df_trips["trip_index"] + 1

    # Load only person/activity geometry from locations
    df_locations = context.stage("synthesis.population.spatial.locations")[["person_id", "activity_index", "geometry"]].copy()

    # Merge preceding geometry (origin)
    df = pd.merge(
        df_trips,
        df_locations.rename(columns={"activity_index": "preceding_activity_index", "geometry": "preceding_geometry"})[["person_id", "preceding_activity_index", "preceding_geometry"]],
        how="left",
        on=["person_id", "preceding_activity_index"]
    )

    # Merge following geometry (destination)
    df = pd.merge(
        df,
        df_locations.rename(columns={"activity_index": "following_activity_index", "geometry": "following_geometry"})[["person_id", "following_activity_index", "following_geometry"]],
        how="left",
        on=["person_id", "following_activity_index"]
    )

    # Extract simple (x,y) tuples for origin and destination to keep the result lightweight
    df["origin_x"] = df["preceding_geometry"].apply(lambda g: g.x if g is not None else None)
    df["origin_y"] = df["preceding_geometry"].apply(lambda g: g.y if g is not None else None)

    df["destination_x"] = df["following_geometry"].apply(lambda g: g.x if g is not None else None)
    df["destination_y"] = df["following_geometry"].apply(lambda g: g.y if g is not None else None)

    # Compute euclidean (crow-fly)
    df["euclidean_distance"] = np.hypot(df["destination_x"] - df["origin_x"], df["destination_y"] - df["origin_y"])

    result = df[["person_id", "trip_index", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time", "euclidean_distance", "preceding_purpose", "following_purpose"]].copy()
    return result

def get_variation_window_min(context, df_trips):
    # enrich with car availability
    car_avail = context.stage("synthesis.population.enriched")[["person_id", "car_availability"]]
    df = df_trips[["person_id","euclidean_distance"]].merge(car_avail, on="person_id", how="left")

    euclidean_distances_km = (df["euclidean_distance"] / 1000).values
    car_availability = df["car_availability"].fillna(False).astype(bool).values
    car_availability_additional_range = 8  # if car is available, we add 8 minutes to the variation window

    # Get the variation window for each trip based on the distance
    variation_window = np.zeros(len(euclidean_distances_km), dtype=int)
    variation_window[euclidean_distances_km >= 5]  = 10
    variation_window[euclidean_distances_km >= 10] = 13
    variation_window[euclidean_distances_km >= 15] = 16
    variation_window[euclidean_distances_km >= 50] = 25

    # Add additional range if car is available
    has_variation_window = variation_window > 0
    variation_window[car_availability & has_variation_window] += car_availability_additional_range

    return variation_window

def filter_trips(context, df_trips):
    # filter out trips that do not need their departure time to be optimized
    # 1. trips with variation window = 0
    df = df_trips[df_trips["variation_window_min"] > 0].copy()
    # 2. trips with departure time not between 5:00 and 24:00
    df = df[df['departure_time'].notna()]
    df = df[df['departure_time'].between(5.3*3600, 24*3600)]  # filter trips with departure time between 5:00 and 24:00 (in seconds)
    # 3. crossborder trips (these will not be routed)
    cb = (df["preceding_purpose"] == "border") | (df["following_purpose"] == "border")
    df = df[~cb]
    
    return df.reset_index(drop=True)

def propose_departures(row, min_time=0, max_time=30*3600, interval_s=5*60, rng=np.random):
    departure_time = row["departure_time"]
    var_window = row["variation_window_min"] * 60
    start_time = max(min_time, departure_time - var_window)
    end_time = min(max_time, departure_time + var_window + 10)
    dur = end_time - start_time
    n = max(3, (dur + interval_s - 1) // interval_s + 1)
    a = np.empty(n, dtype=np.int64)
    a[0] = int(start_time)
    a[n-1] = int(end_time)
    a[1:n-1] =  rng.uniform(start_time, end_time, size=n - 2)
    a.sort()
    return a.astype(int).tolist()

def prepare_trips_for_router(context, df_trips):
    df = df_trips.copy()
    base_seed = context.config('random_seed')
    rng = np.random.default_rng(base_seed)

    # propose different departure times (since we are using mz departures here, we allow it to have higher intervals)
    df["departure_time"] = df["departure_time"].astype(int)
    df["departure_time"] = df.apply(lambda x: propose_departures(x, rng=rng), axis=1)
    df = df.explode("departure_time")

    # Router needs an identifier, different for each trip
    df["identifier"] = df["person_id"].astype(str) + "_" + df["trip_index"].astype(str) + "_" + df["departure_time"].astype(str)

    return df[["identifier", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time"]]

def run_pt_router(context, df):
    # output path
    output_path= os.path.join(context.path(), "pt_routed_trips.csv")
    if os.path.exists(output_path):
        logger.info("\t PT routed trips already exist. Checking if they are valid.")
        df_old = pd.read_csv(output_path)
        if set(df_old["identifier"]) == set(df["identifier"]):
            logger.info("\t  - They are valid. Skipping routing.")
            return df_old
        else:
            logger.info("\t  - They are not valid. Performing routing.")
            del df_old

    # save trips as csv file
    input_path = os.path.join(context.path(), "pt_trips_to_be_routerd.csv")
    initial_ids = df[["identifier"]].astype(str).copy()
    initial_length = len(df)
    logger.info("\t Saving %d trips to be routed to %s.", initial_length, input_path)
    df.to_csv(input_path, index=False)
    del df


    # config path
    config_path = context.stage("calibration.pt_pricing.generate_config")  
    
    # run the router
    eqasim.run(context, "org.eqasim.core.tools.routing.RunBatchPublicTransportRouter",
            [
                "--config-path", config_path,
                "--input-path", input_path,
                "--output-trips-path", output_path,
                "--batch-size", "2048",
                "--eqasim-configurator", "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator",
                "--threads", str(2*context.config("threads"))
            ]
        )
    
    # read the routed trips
    df = pd.read_csv(output_path)
    logger.info(f"\t There are {len(df)} trips after PT routing.")
    logger.info(f"\t There are {initial_length - len(df)} trips lost during PT routing.")

    # relevant variables
    df["access_egress_time_min"] = df["access_travel_time_min"] + df["egress_travel_time_min"]
    df["waiting_time_min"] = df["transfer_waiting_time_min"] + df["transfer_travel_time_min"]
    df["in_vehicle_time_min"] = df["in_vehicle_time_total_min"]
    df["distance_km"] = df["in_vehicle_distance_total_km"]

    # we merge to keep all initial trips, and set the nan values to very high number, so they won't be selected anyway
    df = initial_ids.merge(df, on="identifier", how="left")
    df.loc[df["access_egress_time_min"].isna(), ["access_egress_time_min", "in_vehicle_time_min", "transfers", "waiting_time_min", "initial_waiting_time_min", "distance_km"]] = 1e6

    return df[["identifier", "access_egress_time_min", "in_vehicle_time_min", "transfers", "waiting_time_min", "initial_waiting_time_min", "distance_km"]]
    
def compute_utilities(context, df):
    # get and parse the params file
    mode_params_path, _ = context.stage("dmc.params")
    Parameters.from_yaml(mode_params_path)

    # compute the utility for each trip
    utility = (
        Parameters.pt.betaAccessEgressTime_u_min * np.power((df["access_egress_time_min"]/Parameters.cost.timeScale_min), Parameters.pt.accessEgressTimeExponent) +
        Parameters.pt.betaInVehicleTime_u_min * np.power((df["in_vehicle_time_min"]/Parameters.cost.timeScale_min), Parameters.pt.inVehicleTimeExponent) +
        Parameters.pt.betaWaitingTime_u_min * np.power((df["waiting_time_min"]/Parameters.cost.timeScale_min), Parameters.pt.waitingTimeExponent) +
        Parameters.pt.betaWaitingTime_u_min * np.power((df["initial_waiting_time_min"]/Parameters.cost.timeScale_min), Parameters.pt.waitingTimeExponent) +
        Parameters.pt.betaLineSwitch_u * np.power((df["transfers"]), Parameters.pt.lineSwitchExponent)
    )
    # add very low gumbel randomness
    noise = np.random.gumbel(0.0, scale=abs(Parameters.pt.betaWaitingTime_u_min), size=len(utility))

    return utility + noise

def find_best_departure_times(context, df, original_departures):
    # we first filter out the trips with not a valid route (access_egress_time_min = 1e6)
    df = df[df["access_egress_time_min"] < 1e4].copy()

    # split the identifier into person_id, trip_index, and departure_time (this is how it was before routing)
    df[["person_id","trip_index","departure_time"]] = df['identifier'].str.split('_', expand=True)
    df = df.astype({"person_id": str, "trip_index": int, "departure_time": int}).reset_index(drop=True)

    # group par person_id and trip_index, and select the row with the maximum utility
    best_departure_times = df.loc[df.groupby(['person_id', 'trip_index'])['utility'].idxmax()][["person_id", "trip_index", "departure_time", "initial_waiting_time_min"]].copy()

    # For people without car availability, we advance the departure by the initial waiting time - noise
    car_availability = context.stage("synthesis.population.enriched")[["person_id", "car_availability"]]

    best_departure_times["person_id"] = best_departure_times["person_id"].astype(car_availability["person_id"].dtype)
    best_departure_times = best_departure_times.merge(car_availability, on="person_id", how="left")

    selection = (best_departure_times["car_availability"] == False) & (best_departure_times["initial_waiting_time_min"] > 5) 
    random_noise = np.random.uniform(0, 1, size=selection.sum()) * 5 * 60  # random noise between 0 and 5 minutes in seconds
    best_departure_times.loc[selection, "departure_time"] = (best_departure_times.loc[selection, "departure_time"] + 
                                                             (best_departure_times.loc[selection, "initial_waiting_time_min"] * 60 - random_noise).astype(int)
                                                            )

    # Keep trips that were not candidates for routing, or for which no PT route
    # was found, at their original departure time.
    result = original_departures.merge(
        best_departure_times,
        on=["person_id", "trip_index"],
        how="left",
        suffixes=("_original", "_optimized"),
        validate="one_to_one",
    )
    result["best_departure_time"] = result["departure_time_optimized"].fillna(
        result["departure_time_original"]
    )
    return result[["person_id", "trip_index", "best_departure_time"]]

def get_mz_departures(context, df_trips):
    # get the mz departures from mz trips
    mz_departures = context.stage("data.microcensus.trips")[0][["person_id", "trip_id", "departure_time"]].copy()
    mz_departures = mz_departures.rename(columns={"person_id": "mz_person_id", "trip_id": "trip_index", "departure_time": "mz_departure_time"})
    # get mz id of each person
    df_persons = context.stage("synthesis.population.enriched")[["person_id", "mz_person_id"]]
    # merge the mz departures with the trips to get the mz departure time for each trip
    df = df_trips[["person_id","trip_index","departure_time"]].merge(df_persons, on="person_id", how="left")
    df = df.merge(mz_departures, on=["mz_person_id","trip_index"], how="left")
    # if mz departure time is not available, keep the original departure time
    df["departure_time"] = df["mz_departure_time"].fillna(df["departure_time"])

    return df["departure_time"].values



