import numpy as np
import pandas as pd

from SNN.adapt_work_locations import work_to_homeoffice

import data.spatial.utils as spatial_utils


def configure(context):
    #context.stage("data.statent.statent")
    #context.stage("data.spatial.zones")
    #context.stage("data.spatial.zone_shapes")
    context.stage("synthesis.population.spatial.primary.work.locations")

    if context.config("run_snn"):
        context.config("snn_heuristic")
        context.stage("synthesis.population.spatial.home.locations")
        context.stage("synthesis.population.SNN_population")


def execute(context):
    df = context.stage("synthesis.population.spatial.primary.work.locations")
    
    if context.config("run_snn"):
        if context.config("snn_heuristic") in [1, 3, 4]:
            df_persons = context.stage("synthesis.population.SNN_population")
            df_homes = context.stage("synthesis.population.spatial.home.locations")
            df = work_to_homeoffice(df, df_homes, df_persons)

            df = spatial_utils.to_gpd(context, df, coord_type="work_from_home")

            return df[["person_id", "destination_id", "geometry"]]
