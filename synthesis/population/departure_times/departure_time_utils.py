import pandas as pd
import numpy as np
from .routing_cache import route_trips
from dmc.parameters import Parameters
import logging
logger = logging.getLogger("synpp")

def load_trips(context):
    # Load only the minimal columns needed for fast processing
    df_trips = context.stage("synthesis.population.trips")[["person_id", "trip_id", "trip_index", "departure_time", "preceding_purpose", "following_purpose"]].copy()

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

    result = df[["person_id", "trip_id", "trip_index", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time", "euclidean_distance", "preceding_purpose", "following_purpose"]].copy()
    return result.sort_values(["person_id", "trip_index"], kind="stable").reset_index(drop=True)

def get_variation_window_min(context, df_trips):
    # enrich with car availability
    car_avail = context.stage("synthesis.population.enriched")[["person_id", "car_availability"]]
    df = df_trips[["person_id","euclidean_distance"]].merge(car_avail, on="person_id", how="left")

    euclidean_distances_km = (df["euclidean_distance"] / 1000).values
    car_availability = df["car_availability"].fillna(False).astype(bool).values
    car_availability_additional_range = 5  # if car is available, we add 5 minutes to the variation window

    # Get the variation window for each trip based on the distance
    variation_window = np.zeros(len(euclidean_distances_km), dtype=int)
    variation_window[euclidean_distances_km >= 10] = 5
    variation_window[euclidean_distances_km >= 15] = 10
    variation_window[euclidean_distances_km >= 30] = 20

    # Add additional range if car is available
    has_variation_window = variation_window > 0
    variation_window[car_availability & has_variation_window] += car_availability_additional_range

    # we do not change departure times of crossborder
    cb = (df_trips["preceding_purpose"] == "border") | (df_trips["following_purpose"] == "border")
    variation_window[cb] = 0
    
    return variation_window

def filter_trips(context, df_trips):
    # filter out trips that do not need their departure time to be optimized
    # 1. trips with variation window = 0
    df = df_trips[df_trips["variation_window_min"] > 0].copy()
    # 2. trips with departure time not between 5:00 and 24:00
    df = df[df['departure_time'].notna()]
    df = df[df['departure_time'].between(5.3*3600, 24*3600)]  # filter trips with departure time between 5:18 and 24:00 (in seconds)
    # 3. crossborder trips (these will not be routed)
    cb = (df["preceding_purpose"] == "border") | (df["following_purpose"] == "border")
    df = df[~cb]
    
    return df.reset_index(drop=True)

def propose_departures(row, min_time=0, max_time=30*3600, interval_s=5*60, *, rng):
    departure_time = row["departure_time"]
    var_window = row["variation_window_min"] * 60
    start_time = max(min_time, departure_time - var_window - 1)
    end_time = min(max_time, departure_time + var_window + 1)
    dur = end_time - start_time
    n = int(max(3, (dur + interval_s) // interval_s))
    a = np.empty(n, dtype=np.int64)
    a[0] = int(start_time)
    a[n-1] = int(end_time)
    a[1:n-1] =  rng.uniform(start_time, end_time, size=n - 2)
    a.sort()
    return a.astype(int).tolist()

def prepare_trips_for_router(context, df_trips):
    df = df_trips.sort_values(["person_id", "trip_index"], kind="stable").copy()
    if df.empty:
        return pd.DataFrame(columns=(
            'identifier',
            'origin_x',
            'origin_y',
            'destination_x',
            'destination_y',
            'departure_time',
        ))
    base_seed = context.config('random_seed')
    rng = np.random.default_rng(base_seed)

    # propose different departure times (since we are using mz departures here, we allow it to have higher intervals)
    df["departure_time"] = df["departure_time"].astype(int)
    df["departure_time"] = df.apply(lambda x: propose_departures(x, rng=rng), axis=1)
    df = df.explode("departure_time")

    # Router needs an identifier, different for each trip
    df["identifier"] = df["person_id"].astype(str) + "__" + df["trip_index"].astype(str) + "__" + df["departure_time"].astype(str)

    # Integer-second proposals can coincide; route each alternative only once.
    return (df.loc[:, (
        'identifier',
        'origin_x',
        'origin_y',
        'destination_x',
        'destination_y',
        'departure_time',
    )].drop_duplicates("identifier")
            .sort_values("identifier", kind="stable").reset_index(drop=True))

def run_pt_router(context, df):
    return route_trips(context, df)

def compute_utilities(context, df):
    # get and parse the params file
    parameters = Parameters.from_context(context)

    # compute the utility for each trip
    utility = (
        parameters.pt.betaAccessEgressTime_u_min * np.power((df["access_egress_time_min"]/parameters.cost.timeScale_min), parameters.pt.accessEgressTimeExponent) +
        parameters.pt.betaInVehicleTime_u_min * np.power((df["in_vehicle_time_min"]/parameters.cost.timeScale_min), parameters.pt.inVehicleTimeExponent) +
        parameters.pt.betaWaitingTime_u_min * np.power((df["waiting_time_min"]/parameters.cost.timeScale_min), parameters.pt.waitingTimeExponent) +
        parameters.pt.betaWaitingTime_u_min * np.power((df["initial_waiting_time_min"]/parameters.cost.timeScale_min), parameters.pt.waitingTimeExponent) +
        parameters.pt.betaLineSwitch_u * np.power((df["transfers"]), parameters.pt.lineSwitchExponent)
    )
    # add very low gumbel randomness
    rng = np.random.default_rng(context.config('random_seed')*2)
    noise = rng.gumbel(0.0, scale=abs(parameters.pt.betaWaitingTime_u_min), size=len(utility))

    # Assign draws by identifier, independently of router/output row order.
    order = np.argsort(df["identifier"].to_numpy(), kind="stable")
    aligned_noise = np.empty(len(noise))
    aligned_noise[order] = noise
    return utility + aligned_noise

def find_best_departure_times(context, df, original_departures):
    # we first filter out the trips with not a valid route or for which the best departuer times wouldn't help much
    df = df[(df["access_egress_time_min"] < 3600) & (df["initial_waiting_time_min"] < 3600)].copy()

    if df.empty:
        return original_departures.rename(columns={"departure_time": "best_departure_time"})[
            ["person_id", "trip_index", "best_departure_time"]].copy()
    df = df.sort_values("identifier", kind="stable")

    # split the identifier into person_id, trip_index, and departure_time (this is how it was before routing)
    df[["person_id","trip_index","departure_time"]] = df['identifier'].str.split('__', expand=True)
    df = df.astype({"person_id": str, "trip_index": int, "departure_time": int}).reset_index(drop=True)

    # group par person_id and trip_index, and select the row with the maximum utility
    best_departure_times = df.loc[df.groupby(['person_id', 'trip_index'])['utility'].idxmax()][["person_id", "trip_index", "departure_time", "initial_waiting_time_min"]].copy()

    # For people without car availability, we advance the departure by the initial waiting time - noise
    car_availability = context.stage("synthesis.population.enriched")[["person_id", "car_availability"]]

    best_departure_times["person_id"] = best_departure_times["person_id"].astype(car_availability["person_id"].dtype)
    best_departure_times = best_departure_times.merge(car_availability, on="person_id", how="left")

    selection = (best_departure_times["car_availability"] == False) & (best_departure_times["initial_waiting_time_min"] > 5.1) 
    rng = np.random.default_rng(context.config('random_seed')*3)
    random_noise = rng.uniform(0, 1, size=selection.sum()) * 5 * 60 # random noise between 0 and 5 minutes
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
    mz_departures = mz_departures.rename(columns={"person_id": "mz_person_id", "departure_time": "mz_departure_time"})
    # get mz id of each person
    df_persons = context.stage("synthesis.population.enriched")[["person_id", "mz_person_id"]]
    # merge the mz departures with the trips to get the mz departure time for each trip
    df = df_trips[["person_id","trip_id","trip_index","departure_time"]].merge(df_persons, on="person_id", how="left")
    df = df.merge(mz_departures, on=["mz_person_id","trip_id"], how="left")
    # if mz departure time is not available, keep the original departure time
    df["departure_time"] = df["mz_departure_time"].fillna(df["departure_time"])

    # Since we get departure times from mz instead of using the ones assigned in synthesis.population.trips, to which we add noise of -30,30 minutes
    # we add a lower noise here, because the rest will come from the departure time selection process
    base_seed = context.config('random_seed')
    rng = np.random.default_rng(base_seed+1)
    noise = rng.uniform(-5*60, 5*60, size=len(df))
    order = df.sort_values(["person_id", "trip_index"], kind="stable").index.to_numpy()
    aligned_noise = np.empty(len(noise))
    aligned_noise[order] = noise
    df["departure_time"] = (df["departure_time"] + aligned_noise).astype(int)
    
    return df["departure_time"].values



