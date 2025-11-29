import os
import pandas as pd
import subprocess
from mode_choice.dmc_defaults import Defaults
import matsim.runtime.eqasim as eqasim
import logging
import threading

logger = logging.getLogger(__name__)

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    
    context.config("data_path")
    context.config("dmc_simulation_data_path", default = os.path.join(context.config("data_path"), "simulation_data"))
    context.config("dmc_jar_file_path", default=os.path.join(context.config("dmc_simulation_data_path"), "matsim_switzerland.jar"))
    context.config("dmc_matsim_config_file", default=os.path.join(context.config("dmc_simulation_data_path"), "matsim_config.xml"))
    
    context.config("walk_speed_m_per_s", default=Defaults.DEFAULT_WALK_SPEED_M_PER_S)
    context.config("walk_distance_factor", default=Defaults.DEFAULT_WALK_DISTANCE_FACTOR)     

def run_pt_router(context, input_path, output_path):    
    # cwd = os.getcwd()
    # os.chdir(context.config("dmc_simulation_data_path"))
    eqasim.run(context, "org.eqasim.core.tools.routing.RunBatchPublicTransportRouter",
            [
                "--config-path", context.config("dmc_matsim_config_file"),
                "--input-path", input_path,
                "--output-trips-path", output_path,
                "--walk-speed", str(context.config("walk_speed_m_per_s")),
                "--walk-factor", str(context.config("walk_distance_factor")),
                "--batch-size", "1000",
                "--eqasim-configurator", "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator"
            ]
        )
    # os.chdir(cwd)

def pt_variables(context, df):
    # ensure identifier column 
    if "identifier" not in df.columns:
        if "trip_id" in df.columns:
            df = df.rename(columns={"trip_id":"identifier"}).reset_index(drop=True) 
        else:
            raise ValueError("Input dataframe must have either 'identifier' or 'trip_id' column.")
    df = df[["identifier", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time"]].copy()
    
    # save file to be read by the router        
    input_path = os.path.join(context.path(), "pt_trips_to_be_routerd.csv")
    initial_length = len(df)
    logger.info("\t Saving %d trips to be routed to %s.", initial_length, input_path)
    df.to_csv(input_path, index=False)

    # get the output path
    output_path= os.path.join(context.path(), "pt_routed_trips.csv")

    # run the router
    run_pt_router(context, input_path, output_path)

    # read routed trips
    df = pd.read_csv(output_path)
    logger.info(f"\t There are {len(df)} trips after PT routing.")
    logger.info(f"\t There are {initial_length - len(df)} trips lost during PT routing.")

    # rename back to trip_id
    df = df.rename(columns={"identifier":"trip_id"})
    df["trip_id"] = df["trip_id"].astype(str)
    df["person_id"] = df["trip_id"].str.split("_").str[0].astype(int) # to keep consistency between stages (person_id and trip_id always)
 
    # only keep relevant columns
    df["access_egress_time_min"] = df["access_travel_time_min"] + df["egress_travel_time_min"]
    df["waiting_time_min"] = df["transfer_waiting_time_min"] + df["transfer_travel_time_min"]
    df["in_vehicle_time_min"] = df["in_vehicle_time_total_min"]
    df["distance_km"] = df["in_vehicle_distance_total_km"]
    
    return df[
         ["trip_id", "person_id", "access_egress_time_min", "in_vehicle_time_min", "transfers", 
          "waiting_time_min", "distance_km"]
         ]

def execute(context):    
    df = context.stage("mode_choice.trips.prepare_trips")[
        ["trip_id","origin_x", "origin_y", "destination_x", "destination_y", "departure_time"]
        ].copy()
    
    # run the router    
    df = pt_variables(context, df)

    return df