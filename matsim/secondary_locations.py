import gzip
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import numpy.linalg as la
import os

def configure(context, require):
    require.stage("matsim.population")
    require.stage("matsim.facilities")
    require.stage("data.microcensus.trips")
    require.stage("matsim.java.eqasim")
    require.stage("utils.java")

def execute(context):
    df_mz_trips = context.stage("data.microcensus.trips")
    df_mz_trips["travel_time"] = df_mz_trips["arrival_time"] - df_mz_trips["departure_time"]

    number_of_obs_per_quantile = 400
    modes = ["car", "pt", "bike", "walk", "car_passenger"]

    quantiles_path = "%s/quantiles.dat" % context.cache_path
    distributions_path = "%s/distributions.dat" % context.cache_path

    with open(quantiles_path, "w+") as quantiles_writer:
        with open(distributions_path, "w+") as distributions_writer:
            for mode in modes:
                df_mode = df_mz_trips[df_mz_trips["mode"] == mode]
                df_mode = df_mode[df_mode["travel_time"] > 0.0]

                number_of_obs = len(df_mode)
                number_of_quantiles = int(np.floor(number_of_obs / number_of_obs_per_quantile))
                quantiles = np.unique([
                    np.percentile(df_mode["travel_time"], 100.0 * i / number_of_quantiles) for i in range(number_of_quantiles)
                ])
                quantiles[-1] = np.max(df_mode["travel_time"])
                quantiles = quantiles[quantiles > 0]

                quantiles_writer.write("%s;%s\n" % (mode, ";".join(map(str, quantiles))))

                lower_bound = -np.inf
                for index, upper_bound in enumerate(quantiles):
                    distances = df_mode[
                        (df_mode["travel_time"] > lower_bound) & (df_mode["travel_time"] <= upper_bound)
                    ]["crowfly_distance"].values

                    distributions_writer.write("%s;%d;%s\n" % (mode, index,
                        ";".join(map(str, distances))
                    ))

                    lower_bound = upper_bound

    java = context.stage("utils.java")
    input_population_path = context.stage("matsim.population")
    input_facilities_path = context.stage("matsim.facilities")

    output_population_path = "%s/population_with_locations.xml.gz" % context.cache_path
    number_of_threads = context.config["threads"]

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.core.scenario.location_assignment.RunLocationAssignment", [
        "--population-path", input_population_path,
        "--facilities-path", input_facilities_path,
        "--quantiles-path", quantiles_path,
        "--distributions-path", distributions_path,
        "--output-path", output_population_path,
        "--threads", str(number_of_threads),
        "--random-seed", str(0)
    ],cwd = context.cache_path)

    assert(os.path.exists(output_population_path))

    return output_population_path
