from matsim.runtime.eqasim import run as run_eqasim
import os
import pandas as pd
import logging 
from shapely import vectorized

logger = logging.getLogger("synpp")

def configure(context):    
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.stage("analysis.travel_times.APIs.get")

    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("output_prefix", "switzerland_")
    context.config("threads")


def execute(context):
    logger.info("Routing car car trips using MATSim")
    # Load trips
    df = context.stage("analysis.travel_times.APIs.get")
    df = df[['identifier', 'origin_x', 'origin_y', 'destination_x', 'destination_y', 'departure_time']]

    # save trips to temporary csv
    logger.info("\t - Saving car trips to temporary file")   
    path_to_trips = os.path.join(context.path(), "car_trips.csv")
    df.to_csv(path_to_trips, index=False)
    logger.info(f"\t - Car trips saved to {path_to_trips}")

    # Path to events file
    logger.info("\t - Locating simulation output files")
    path_to_dir = os.path.join(context.config("output_path"), 
                               context.config("output_id"))
    
    path_to_events = os.path.join(path_to_dir,
                                  context.config("simulation_directory"),
                                  "output_events.xml.gz")
    
    path_to_config = os.path.join(path_to_dir,
                                  "%sconfig.xml" % context.config("output_prefix"))
    
    assert os.path.exists(path_to_events), f"Events file not found at {path_to_events}"
    assert os.path.exists(path_to_config), f"Config file not found at {path_to_config}"

    # Route trips using MATSim
    logger.info("\t - Routing car trips using MATSim")
    output_path = os.path.join(context.path(), "routed_car_trips.csv")
    
    cwd = os.getcwd()
    os.chdir(path_to_dir)  # eqasim requires to be run from the simulation directory
    run_eqasim(
        context,
        "org.eqasim.core.tools.routing.TripsRouter",
        [
            "--config-path", path_to_config,
            "--trips-path", path_to_trips,
            "--events-path", path_to_events,
            "--output-path", output_path,
            "--threads", str(context.config("threads"))
        ]
    )
    os.chdir(cwd)

    return (path_to_trips, output_path)