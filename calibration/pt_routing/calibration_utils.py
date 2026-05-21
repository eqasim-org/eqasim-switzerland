import numpy as np
import pandas as pd
from itertools import product
import requests as reqlib
import time
import logging

logger = logging.getLogger("synpp")

def is_useful_column(column):
    if column == "trip_id":
        return True
    elif column == "day_of_the_week":
        return True
    elif column == "weight":
        return True
    elif "legs" in column or "destination" in column or "origin" in column or "->" in column or "transfers" in column:
        return True
    elif column in ["walk_time", "waiting_time"]:
        return True
    elif column=="departure_time":
        return True
    return False


def read_reference(reference, parameters):

    df_reference = reference.copy()#pd.read_parquet(parameters["reference_path"])
    df_reference = df_reference[[col for col in df_reference.columns if is_useful_column(col)]]

    # Only select relevant days
    select_workdays = parameters["keep_workdays"]
    select_weekends = parameters["keep_weekends"]

    if select_workdays and not select_weekends:
        df_reference = df_reference[df_reference["day_of_the_week"] <= 5]
    elif select_weekends and not select_workdays:
        df_reference = df_reference[df_reference["day_of_the_week"] > 5]
    elif not select_weekends and not select_workdays:
        raise ValueError("The reference dataframe will be empty. Please adjust the day selector.")

    # Only select relevant canton
    if parameters["canton_selector"]:
        canton = parameters["canton_selector"]
        df_reference = df_reference[(df_reference["origin_canton"]==canton) & (df_reference["destination_canton"]==canton)]

    # Identify modes and maximum transfers
    modes             = [c.replace("legs_", "") for c in df_reference.columns if c.startswith("legs_")]
    maximum_transfers = df_reference["transfers"].max()

    ref_modes_to_main = {}
    for mode in modes:
        if mode in parameters["pt_main_modes"]:
            ref_modes_to_main[mode] = mode
        else:
            ref_modes_to_main[mode] = "pt_other"

    parameters["reference_modes"]   = modes
    parameters["max_nb_transfers"]  = maximum_transfers
    parameters["ref_modes_to_main"] = ref_modes_to_main

    all_pairs = [f"{a}->{b}" for a, b in product(modes, repeat=2)]

    rail_to_rail_columns   = [pair for pair in all_pairs if pair == "rail->rail"]
    rail_to_other_columns  = [pair for pair in all_pairs if "rail" in pair and pair != "rail->rail"]
    other_to_other_columns = [pair for pair in all_pairs if "rail" not in pair]

    df_reference["rail_transfers"]       = df_reference[rail_to_rail_columns].sum(axis=1)
    df_reference["other_transfers"]      = df_reference[[c for c in other_to_other_columns if c in df_reference.columns]].sum(axis=1)
    df_reference["intermodal_transfers"] = df_reference[[c for c in rail_to_other_columns if c in df_reference.columns]].sum(axis=1)

    parameters["max_nb_transfers_intermodal"] = df_reference["intermodal_transfers"].max()
    parameters["max_nb_transfers_rail"]       = df_reference["rail_transfers"].max()
    parameters["max_nb_transfers_other"]      = df_reference["other_transfers"].max()

    df_long             = df_reference.melt(id_vars="trip_id", var_name="mode", value_name="count")
    df_long["mode"]     = df_long["mode"].str.replace("legs_", "")
    df_long["category"] = df_long["mode"].map(ref_modes_to_main)
    df_grouped          = df_long.groupby(["trip_id", "category"], as_index=False)["count"].sum()
    df_result           = df_grouped.pivot(index="trip_id", columns="category", values="count").fillna(0).astype(int)
    df_result           = df_result.add_prefix("legs_").reset_index()

    for column in df_reference.columns:
        if "legs_" in column:
            del df_reference[column]

    initial_length = len(df_reference)
    df_reference   = df_reference.merge(df_result, on = "trip_id", how = "inner")
    final_length   = len(df_reference)

    assert initial_length == final_length

    return parameters, df_reference


def create_requests(df_reference):
    requests = []

    df_reference["request_index"] = np.arange(len(df_reference))

    for _, row in df_reference.iterrows():
        requests.append({
            "request_index": int(row["request_index"]),
            "origin_x": row["origin_x"],
            "origin_y": row["origin_y"],
            "destination_x": row["destination_x"],
            "destination_y": row["destination_y"],
            "departure_time_s": row["departure_time"]
        })

    return requests


def count_transfers(modes):
    intermodal = 0
    rail       = 0
    other      = 0
    for i in range(len(modes) - 1):
        if modes[i] == modes[i + 1]:
            if  modes[i] == "rail":
                rail += 1
            else:
                other += 1
        else:
            if modes[i] == "rail" or modes[i+1] == "rail":
                intermodal += 1
            else:
                other += 1
    return intermodal, rail, other


# Prepare querying the routing server
def query_endpoint(parameters, requests, utilities):
    response = reqlib.post(parameters["routing_endpoint"], json = {
        "batch": requests,
        "utilities": utilities
    })

    assert response.status_code == 200

    df_response = { 
        "request_index": [],
        "transfers": [],
        "rail_transfers": [],
        "other_transfers": [],
        "intermodal_transfers": [],
        "walk_time": [],
        "waiting_time": []
    }

    pt_main_modes    = parameters["pt_main_modes"]
    pt_modes         = parameters["pt_modes"]
    pt_modes_to_main = parameters["pt_modes_to_main"]

    for mode in parameters["pt_main_modes"]:
        df_response["legs_{}".format(mode)] = []

    for row in response.json():
        df_response["request_index"].append(row["request_index"])
        df_response["transfers"].append(row["transfers"])

        df_response["walk_time"].append(row["access_walk_time_min"] + row["egress_walk_time_min"])
        df_response["waiting_time"].append(row["initial_wait_time_min"] + row["transfer_wait_time_min"])

        count_legs = {}
        
        for mode in pt_modes:
            if mode in row["vehicle_legs_by_mode"]:
                main_mode = pt_modes_to_main[mode]
                if main_mode in count_legs.keys():
                    count_legs[main_mode] += row["vehicle_legs_by_mode"][mode]
                else:
                    count_legs[main_mode] = row["vehicle_legs_by_mode"][mode]

        for main_mode in pt_main_modes:
            if main_mode in count_legs.keys():
                df_response["legs_{}".format(main_mode)].append(count_legs[main_mode])
            else:
                df_response["legs_{}".format(main_mode)].append(0)

        transfers_intermodal, transfers_rail, transfers_other = count_transfers(row["modes_sequence"])
        df_response["intermodal_transfers"].append(transfers_intermodal)
        df_response["rail_transfers"].append(transfers_rail)
        df_response["other_transfers"].append(transfers_other)

    df_response = pd.DataFrame(df_response)

    return df_response


# Send the requests to the server
def query_endpoint_batched(parameters, requests, utilities):
    df_response = []
    batch_index = 0

    maximum_batch_size = parameters["maximum_batch_size"]

    while batch_index * maximum_batch_size < len(requests):
        df_response.append(query_endpoint(parameters, 
            requests[batch_index * maximum_batch_size : (batch_index + 1) * maximum_batch_size],
            utilities))
        
        batch_index += 1
    
    return pd.concat(df_response)


# Set up before the server runs
def wait_for_server(url, requests_sample, utilities, timeout=600, interval=10):
    """Wait until the server at `url` responds successfully."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        logger.info("Waiting for server...")
        try:
            response = reqlib.post(url, json = {
                    "batch": requests_sample,
                    "utilities": utilities
                })
            if response.status_code == 200:
                logger.info("The server is running!")
                return True
        except reqlib.ConnectionError:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Server didn't start within {timeout} seconds.")


# Define the optimization objective: observation comparison based
def calculate_objective_observation_based(parameters, df_reference, df_evaluation):
    transfers_weight = parameters["transfers_weight"]
    modes_weight     = parameters["modes_weight"]
    modes            = parameters["pt_main_modes"]

    df_evaluation = pd.merge(df_reference, df_evaluation, on = "request_index", 
        suffixes = ["_reference", "_evaluation"])
    
    df_evaluation["offset"] = transfers_weight * np.abs(
        df_evaluation["transfers_reference"] - df_evaluation["transfers_evaluation"])

    for mode in modes:
        df_evaluation["offset"] += modes_weight * np.abs(
            df_evaluation["legs_{}_reference".format(mode)] - df_evaluation["legs_{}_evaluation".format(mode)]
        )

    return np.sum(df_evaluation["offset"] * df_evaluation["weight"]) / df_evaluation["weight"].sum()


# Define the optimization objective: distribution comparison based
def calculate_objective_distribution_based(parameters, df_reference, df_evaluation):
    transfers_weight = parameters["transfers_weight"]
    modes_weight     = parameters["modes_weight"]
    walk_wait_weight = parameters["walk_wait_weight"]
    modes            = parameters["pt_main_modes"]
    max_transfers    = parameters["max_nb_transfers"] 

    df_evaluation = pd.merge(df_reference, df_evaluation, on = "request_index", 
         suffixes = ["_reference", "_evaluation"])
    
    reference_mode_distribution  = []
    evaluation_mode_distribution = []

    for mode in modes:
        reference_mode_distribution.append((df_evaluation["legs_{}_reference".format(mode)] * df_evaluation["weight"]).sum())
        evaluation_mode_distribution.append((df_evaluation["legs_{}_evaluation".format(mode)] * df_evaluation["weight"]).sum())

    reference_mode_distribution  = np.array(reference_mode_distribution)
    evaluation_mode_distribution = np.array(evaluation_mode_distribution)

    if parameters["evaluate_intermodality"]:

        max_intermodal_transfers = parameters["max_nb_transfers_intermodal"]
        max_rail_transfers       = parameters["max_nb_transfers_rail"]
        max_other_transfers      = parameters["max_nb_transfers_other"]

        max_nb = {"intermodal": max_intermodal_transfers,
                  "rail":       max_rail_transfers,
                  "other":      max_other_transfers}
        
        transfers_offsets = {}

        for transfer_category in ["intermodal", "rail", "other"]:

            reference_transfers  = []
            evaluation_transfers = []

            for transfers in range(max_nb[transfer_category] + 1):
                f_reference = df_evaluation[transfer_category + "_transfers_reference"] == transfers
                reference_transfers.append(df_evaluation.loc[f_reference, "weight"].sum())

                f_evaluation = df_evaluation[transfer_category + "_transfers_evaluation"] == transfers
                evaluation_transfers.append(df_evaluation.loc[f_evaluation, "weight"].sum())

            reference_transfers  = np.array(reference_transfers) #/ np.sum(reference_transfers)
            evaluation_transfers =  np.array(evaluation_transfers) #/ np.sum(evaluation_transfers)

            transfers_offsets[transfer_category] = np.abs(reference_transfers - evaluation_transfers)**2

        transfer_distribution_offset = 0
        for _, offset in transfers_offsets.items():
            transfer_distribution_offset += np.sum(offset)

        transfer_distribution_offset = transfer_distribution_offset * transfers_weight

        mode_distribution_offset = np.abs(reference_mode_distribution - evaluation_mode_distribution)**2
        mode_distribution_offset = np.sum(mode_distribution_offset) * modes_weight

        # Walk and wait time - total time difference
        bins     = [0, 5, 10, 20, 30, np.inf]
        time_bins = ["0–5 min", "5–10 min", "10–20 min", "20–30 min", "> 30 min"]

        time_offset = 0

        for time_evaluation in []:#["walk_time"]:#, "waiting_time"]:

            df_evaluation[f"{time_evaluation}_bin_reference"] = df_evaluation[f"{time_evaluation}_reference"] / 60 # first convert in minutes
            df_evaluation[f"{time_evaluation}_bin_reference"] = pd.cut(df_evaluation[f"{time_evaluation}_bin_reference"], bins = bins, labels = time_bins, right=False )
    
            df_evaluation[f"{time_evaluation}_bin_evaluation"] = pd.cut(df_evaluation[f"{time_evaluation}_evaluation"], bins = bins, labels = time_bins, right = False)
    
            reference_time   = []
            evaluation_time  = []
            for time_bin in time_bins:
                f_reference  = df_evaluation[f"{time_evaluation}_bin_reference"] == time_bin
                f_evaluation = df_evaluation[f"{time_evaluation}_bin_evaluation"] == time_bin
    
                reference_time.append(df_evaluation.loc[f_reference, "weight"].sum())
                evaluation_time.append(df_evaluation.loc[f_evaluation, "weight"].sum())
    
            reference_time_distribution  = np.array(reference_time)
            evaluation_time_distribution = np.array(evaluation_time)
    
            time_offset = np.sum((reference_time_distribution - evaluation_time_distribution) ** 2)
            time_offset +=  np.sum(time_offset) * walk_wait_weight
        

        den = np.sum(df_evaluation["weight"]) * (transfers_weight + modes_weight + walk_wait_weight)

        return (transfer_distribution_offset + mode_distribution_offset + time_offset) / den
    
    else:
         
        reference_transfer_distribution = []
        evaluation_transfer_distribution = []

        for transfers in range(max_transfers + 1):
            f_reference = df_evaluation["transfers_reference"] == transfers
            reference_transfer_distribution.append(df_evaluation.loc[f_reference, "weight"].sum())

            f_evaluation = df_evaluation["transfers_evaluation"] == transfers
            evaluation_transfer_distribution.append(df_evaluation.loc[f_evaluation, "weight"].sum())

        reference_transfer_distribution = np.array(reference_transfer_distribution) / np.sum(reference_transfer_distribution)
        evaluation_transfer_distribution = np.array(evaluation_transfer_distribution) / np.sum(evaluation_transfer_distribution)

        mode_distribution_offset = np.abs(reference_mode_distribution - evaluation_mode_distribution)**2
        transfer_distribution_offset = np.abs(reference_transfer_distribution - evaluation_transfer_distribution)**2

        return transfers_weight * np.sum(transfer_distribution_offset) + modes_weight * np.sum(mode_distribution_offset)