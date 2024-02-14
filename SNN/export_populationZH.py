import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path


shp_path = "/nas/asallard/Switzerland/SNN_shapefile_zurich/zurich_5km.shp"

def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.SNN_population")
    context.stage("SNN.SNN_run_zurich_astra_asc_FB")

    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.config("output_path")


# In this function, we extract information about WFH agents
def wfh(pop, output_file_path):
    pop = pop[pop["employed"]]

    # 1. WFH frequency
    df_nbdays = pop["wfh"].value_counts(dropna = False).reset_index()
    df_nbdays.columns = ["Number of days", "Frequency"]
    df_nbdays["Frequency"] = df_nbdays["Frequency"] / np.sum(df_nbdays["Frequency"]) * 100
    df_nbdays.to_hdf(output_file_path, key = "ho_frequency")

    # 2. WFH ability
    df_wfa = pop.filter(like='wfa_1')
    del pop["wfa_1"]
    df_wfa.columns = ["old", "wfa_1"]
    df_wfa.loc[:, "person_id"] = pop["person_id"]
    pop = pd.merge(pop, df_wfa)
    df_possible = pop["wfa_1"].value_counts(dropna = False).reset_index()


    #print(pop["wfh"].value_counts(dropna = False))
    #print(pop["wfh_the_days"].value_counts(dropna = False))

# In this function, we extract MTO information from the population before and after applying the models.
# The mobility tools we consider are the ones affected by Daniel's models: GA, Halbtax, Verbund subscription, car ownership.
def mto_comparison_wfh_models(pop_before, pop_after, pop_zh, output_file_path):
    pop_before = pop_before[pop_before["person_id"].isin(pop_zh["person"])]
    pop_after = pop_after[pop_after["person_id"].isin(pop_zh["person"])]

    for modeT in ["subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "car_availability"]:
        s1 = pop_before[pop_before["age"]>=18][modeT].value_counts(dropna = False)
        s2 = pop_after[pop_after["age"]>=18][modeT].value_counts(dropna = False)
        d1 = pd.DataFrame(s1).reset_index()
        d2 = pd.DataFrame(s2).reset_index()

        d = d1.merge(d2, on = "index", suffixes=["_before", "_after"])
        d["index"] = d["index"].astype(str)
        d["index"] = [c.replace("False", "No") for c in d["index"]]
        d["index"] = [c.replace("True", "Yes") for c in d["index"]]
        d["index"] = [c.replace("0.0", "Always") for c in d["index"]]
        d["index"] = [c.replace("1.0", "Sometimes") for c in d["index"]]
        d["index"] = [c.replace("2.0", "Never") for c in d["index"]]

        if modeT.split("_")[0] == "subscriptions":
            suffix = modeT.split("_")[1]
        elif modeT == "car_availability":
            suffix = "car"
        
        d.to_hdf(output_file_path, key = "mto_"+suffix)

    return


# In this function, we load MZ trips starting AND ending in the study area for mode share comparison.
def load_MZ_trips_study_area(mztrips):
    zurich5km = gpd.read_file(shp_path)
    print("INFO shapefile loaded")

    zurich5km = zurich5km["geometry"].values.tolist()[0]

    # load MZ trips and extract those starting AND ending in the shapefile extent
    trips = mztrips

    origins      = gpd.GeoSeries.from_xy(trips["origin_x"], trips["origin_y"])
    destinations = gpd.GeoSeries.from_xy(trips["destination_x"], trips["destination_y"])

    origins_in_shp = origins.within(zurich5km)
    destinations_in_shp = destinations.within(zurich5km)

    trips_zh = trips[origins_in_shp & destinations_in_shp]
    print("INFO MZ trips imported")
    return trips_zh


# Import MATSim trips starting AND ending in the study area for mode share comparison.
def select_zh_trips_MATSim(results):
    zurich5km = gpd.read_file(shp_path)
    zurich5km = zurich5km["geometry"].values.tolist()[0]

    origins      = gpd.GeoSeries.from_xy(results["origin_x"], results["origin_y"])
    destinations = gpd.GeoSeries.from_xy(results["destination_x"], results["destination_y"])

    origins_in_shp = origins.within(zurich5km)
    destinations_in_shp = destinations.within(zurich5km)

    results_zh = results[origins_in_shp & destinations_in_shp]
    results_zh = results_zh[(results_zh["preceding_purpose"] != "outside") & (results_zh["following_purpose"] != "outside")]
    results_zh = results_zh[results_zh["mode"]!= "truck"]
    print("INFO simulated trips imported")

    return results_zh


# We compute the mode shares by number of trips, travelled distance and trip duration
def mode_shares(mz_trips, sim_trips, output_file_path):
    ## 1. MZ
    # aggregate by number of trips
    g_nb_trips = mz_trips.groupby("mode")["person_weight"].sum().reset_index()
    g_nb_trips.columns = ["mode", "nb_trips"]
    g_nb_trips["nb_trips"] = g_nb_trips["nb_trips"] / np.sum(g_nb_trips["nb_trips"])

    # aggreagte by total distance
    mz_trips.loc[:, "weighted_distance"] = mz_trips["network_distance"] * mz_trips["person_weight"] / np.sum(mz_trips["person_weight"])
    g_distance = mz_trips.groupby("mode")["weighted_distance"].sum().reset_index()
    g_distance.columns = ["mode", "travelled_distance"]
    g_distance["travelled_distance"] = g_distance["travelled_distance"] / np.sum(g_distance["travelled_distance"])

    g = g_nb_trips.merge(g_distance, how = "left", on = "mode")

    # aggreagte by travel duration
    mz_trips.loc[:, "weighted_duration"] = (mz_trips["arrival_time"] - mz_trips["departure_time"]) * mz_trips["person_weight"] / np.sum(mz_trips["person_weight"])
    g_duration = mz_trips.groupby("mode")["weighted_duration"].sum().reset_index()
    g_duration.columns = ["mode", "travel_time"]
    g_duration["travel_time"] = g_duration["travel_time"] / np.sum(g_duration["travel_time"])

    g = g.merge(g_duration, how = "left", on = "mode")

    g.to_hdf(output_file_path, key = "ms_mz")

    ## 2. Simulation output
    sim_trips.loc[:, "def"] = 1
    h_nb_trips = sim_trips.groupby("mode")["def"].sum().reset_index()
    h_nb_trips.columns = ["mode", "nb_trips"]
    h_nb_trips["nb_trips"] = h_nb_trips["nb_trips"] / np.sum(h_nb_trips["nb_trips"])

    # aggreagte by total distance
    h_distance = sim_trips.groupby("mode")["routed_distance"].sum().reset_index()
    h_distance.columns = ["mode", "routed_distance"]
    h_distance["travelled_distance"] = h_distance["routed_distance"] / np.sum(h_distance["routed_distance"])

    h = h_nb_trips.merge(h_distance, how = "left", on = "mode")

    # aggreagte by travel duration
    h_duration = sim_trips.groupby("mode")["travel_time"].sum().reset_index()
    h_duration.columns = ["mode", "travel_time"]
    h_duration["travel_time"] = h_duration["travel_time"] / np.sum(h_duration["travel_time"])

    h = h.merge(h_duration, how = "left", on = "mode")
    del h["routed_distance"]

    h.to_hdf(output_file_path, key = "ms_sim")



def execute(context):
    # Populations before and after applying WFH models
    pop_before = context.stage("synthesis.population.enriched")
    pop_after  = context.stage("synthesis.population.SNN_population")

    # MZ data
    mztrips    = context.stage("data.microcensus.trips")[0]
    mzpersons  = context.stage("data.microcensus.persons")

    # Simulation output
    pop_zh_path = context.path("SNN.SNN_run_zurich_astra_asc_FB") + "/simulation_output/output_persons.csv.gz"
    output_trips_path = context.path("SNN.SNN_run_zurich_astra_asc_FB")  + "/simulation_output/ITERS/it.60/60.trips.csv"    

    pop_zh = pd.read_csv(pop_zh_path, compression='gzip', sep = ";")
    pop_zh["person"] = pop_zh["person"].astype(str)
    pop_zh = pop_zh[~pop_zh["person"].str.contains("freight")]
    pop_zh["person"] = pop_zh["person"].astype(int)

    output_trips = pd.read_csv(output_trips_path, sep = ";")

    # Setting up the output folder
    output_path = context.config("output_path") + "/Results"
    Path(output_path).mkdir(parents = True, exist_ok= True)
    output_file_path = output_path + "/results_data_agg.h5"

    # Extract ZH population
    pop_before = pop_before[pop_before["person_id"].isin(pop_zh["person"])]
    pop_after = pop_after[pop_after["person_id"].isin(pop_zh["person"])]

    # WFH information
    wfh(pop_after, output_file_path)
    exit()

    # MTO comparison
    print("INFO starting to compute MTO")
    mto_comparison_wfh_models(pop_before, pop_after, pop_zh, output_file_path)
    print("  INFO checking car availability")
    data = pd.read_hdf(output_file_path, "mto_car")
    print(data)
    print("INFO MTO computed")

    # Import MZ trips in Zurich and then the simulated trips
    mz_trips = load_MZ_trips_study_area(mztrips)
    sim_trips = select_zh_trips_MATSim(output_trips)    

    # Merge weights to MZ trips
    persons_to_filterout = context.stage("data.microcensus.trips")[1]
    mzpersons = mzpersons[~mzpersons["person_id"].isin(persons_to_filterout)]
    mzpersons = mzpersons[["person_id", "person_weight"]]
    mz_trips = mz_trips.merge(mzpersons, how = "inner", on = "person_id")

    # Mode share
    print("INFO starting to compute the mode shares")
    mode_shares(mz_trips, sim_trips,  output_file_path)
    print("  INFO checking mode shares")
    data = pd.read_hdf(output_file_path, "ms_sim")
    print(data)
    print("INFO mode shares computed")
    
