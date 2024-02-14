import geopandas as gpd
import pandas as pd
import numpy as np


def configure(context):
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")
    context.stage("synthesis.population.spatial.secondary.locations")

    context.stage("synthesis.population.SNN_mobility")
    context.stage("synthesis.population.sampled")

    context.config("run_snn")
    if context.config("run_snn"):
        context.config("snn_heuristic")
        context.stage("synthesis.population.SNN_population")


def execute(context):
    df_home = context.stage("synthesis.population.spatial.home.locations")
    if (not context.config("run_snn")) or (context.config("run_snn") and context.config("snn_heuristic") == 0):
        df_work, df_education = context.stage("synthesis.population.spatial.primary.locations")

    else:
        df_work, df_education, df_work_from_home = context.stage("synthesis.population.spatial.primary.locations")

    df_secondary = context.stage("synthesis.population.spatial.secondary.locations")[0]

    df_persons = context.stage("synthesis.population.sampled")[["person_id", "household_id"]]
    df_locations = context.stage("synthesis.population.SNN_mobility")[0][["person_id", "activity_index", "purpose"]]  

    # Home locations
    df_home_locations = df_locations[df_locations["purpose"] == "home"]
    df_home_locations = pd.merge(df_home_locations, df_persons, on="person_id")
    df_home_locations = pd.merge(df_home_locations, df_home[["household_id", "geometry"]], on="household_id")
    df_home_locations["destination_id"] = -1
    df_home_locations = df_home_locations[["person_id", "activity_index", "destination_id", "geometry"]]
    df_home_locations = gpd.GeoDataFrame(df_home_locations)
    print(df_home_locations.crs)

    # Work locations
    df_work_locations = df_locations[df_locations["purpose"] == "work"]
    df_work_locations = pd.merge(df_work_locations,
                                 df_work[["person_id", "destination_id", "geometry"]],
                                 on="person_id")
    df_work_locations = df_work_locations[["person_id", "activity_index", "destination_id", "geometry"]]
    df_work_locations = gpd.GeoDataFrame(df_work_locations)
    print(df_work_locations.crs)

    # Work from home
    if context.config("run_snn") and context.config("snn_heuristic") in [1,3, 4]:
        df_wfh_locations = df_locations[df_locations["purpose"] == "work_from_home"]
        df_wfh_locations = pd.merge(df_wfh_locations,
                                    df_work[["person_id", "destination_id", "geometry"]],
                                    on="person_id")
        df_wfh_locations = df_wfh_locations[["person_id", "activity_index", "destination_id", "geometry"]]
        df_wfh_locations = gpd.GeoDataFrame(df_wfh_locations)
        print(df_wfh_locations.crs)

    # SNN strategy number 2: the work location was NOT changed previously to the home location so let's do that now
    if context.config("run_snn") and context.config("snn_heuristic") == 2:
        df_persons_enriched = context.stage("synthesis.population.SNN_population")
        
        wfh_ids = df_persons_enriched[df_persons_enriched["wfh_today"]]["person_id"].values.tolist()

        df_wfh_locations = df_locations[(df_locations["purpose"] == "work_from_home") & (df_locations["person_id"].isin(wfh_ids))]
        df_wfh_locations = pd.merge(df_wfh_locations, df_home_locations[["person_id", "destination_id", "geometry"]].drop_duplicates(), on = "person_id", how = "left")
        df_wfh_locations = df_wfh_locations[["person_id", "activity_index", "destination_id", "geometry"]]
        df_wfh_locations = gpd.GeoDataFrame(df_wfh_locations)
        print(df_wfh_locations.crs)

    # Education locations
    df_education_locations = df_locations[df_locations["purpose"] == "education"]
    df_education_locations = pd.merge(df_education_locations,
                                      df_education[["person_id", "destination_id", "geometry"]],
                                      on="person_id")
    df_education_locations = df_education_locations[["person_id", "activity_index", "destination_id", "geometry"]]
    df_education_locations = gpd.GeoDataFrame(df_education_locations)
    print(df_education_locations.crs)

    # Secondary locations
    df_secondary_locations = df_locations[~df_locations["purpose"].isin(("home", "work", "education", "work_from_home"))].copy()
    df_secondary_locations = pd.merge(df_secondary_locations,
                                      df_secondary[["person_id", "activity_index", "destination_id", "geometry"]],
                                      on=["person_id", "activity_index"], how="left")
    df_secondary_locations = df_secondary_locations[["person_id", "activity_index", "destination_id", "geometry"]]
    assert not df_secondary_locations["geometry"].isna().any()
    df_secondary_locations = gpd.GeoDataFrame(df_secondary_locations, crs = "epsg:2056")
    df_secondary_locations = df_secondary_locations.to_crs("epsg:2056")
    print(df_secondary_locations.crs)

    # Validation
    initial_count = len(df_locations)

    df_locations = pd.concat([df_home_locations, df_work_locations, df_education_locations, df_secondary_locations])

    if context.config("run_snn")and context.config("snn_heuristic") != 0:
        df_locations = pd.concat([df_locations, df_wfh_locations])

    df_locations = df_locations.sort_values(by=["person_id", "activity_index"])
    final_count = len(df_locations)

    assert initial_count == final_count

    df_locations = gpd.GeoDataFrame(df_locations, crs="epsg:2056")

    print(df_locations.head(20))

    return df_locations
