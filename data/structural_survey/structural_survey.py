import numpy as np
import pandas as pd

from data.spatial import countries, municipalities, quarters, zones, utils
from data.spatial.municipality_types import impute as impute_municipality_type
import logging

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.structural_survey.raw")
    context.stage("data.spatial.municipality_types")
    context.stage("data.statpop.statpop")
    # context.stage("data.statent.statent")
    
    context.stage("data.spatial.countries")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.stage("data.spatial.zones")

    context.config("random_seed")


def execute(context):
    df_se = context.stage("data.structural_survey.raw")

    df_se["sex"] -= 1

    df_zones = context.stage("data.spatial.zones")
    df_countries = context.stage("data.spatial.countries")
    df_municipalities, df_municipality_mapping = context.stage("data.spatial.municipalities")
    df_quarters = context.stage("data.spatial.quarters")
    
    # Set up RNG
    random_seed = context.config("random_seed")
    rng = np.random.RandomState(random_seed)

    # Find the correct modes
    mode_numeric = df_se["mode"].astype(int)
    mode_map = {
        -10: "unknown",
        -9: "unknown",
        -8: "unknown",
        1: "walk",    # walking
        2: "walk",    # skateboard
        4: "car",     # Mofa / Moped / light motor bike
        5: "car",     # Car as driver or passenger
        6: "car",     # company bus
        7: "pt",      # Train
        8: "pt",      # Tram / Metro
        9: "pt",      # Bus
        10: "other",  # Ship, cable car, ...
        11: "bike",   # bike
        12: "bike",   # e-bike
    }

    df_se["mode"] = mode_numeric.map(mode_map).astype("object")
    del mode_numeric

    # Impute the home zone
    df_se = df_se.reset_index(drop=True)
    df_se = df_se.rename({"home_municipality": "municipality_id",
                          "home_quarter": "quarter_id"}, axis=1)
    df_se = quarters.update_quarter_ids(df_se, df_quarters)
    df_se = municipalities.update_municipality_ids(df_se, df_municipality_mapping)
    df_se = df_se.rename({"municipality_id": "home_municipality_id",
                          "quarter_id": "home_quarter_id"}, axis=1)
    
    # Impute the work zone
    df_se = df_se.rename({"work_country": "country_id",
                          "work_municipality": "municipality_id",
                          "work_quarter": "quarter_id"}, axis=1)
    df_se = quarters.update_quarter_ids(df_se, df_quarters)
    df_se = municipalities.update_municipality_ids(df_se, df_municipality_mapping)
    df_se = countries.update_country_ids(df_se, df_countries)
    df_se = df_se.rename({"municipality_id": "work_municipality_id",
                          "quarter_id": "work_quarter_id",
                          "country_id": "work_country_id"}, axis=1)

    # Impute the education zone
    df_se = df_se.reset_index(drop=True)
    df_se = df_se.rename({"education_municipality": "municipality_id",
                          "education_quarter": "quarter_id"}, axis=1)
    df_se = quarters.update_quarter_ids(df_se, df_quarters)
    df_se = municipalities.update_municipality_ids(df_se, df_municipality_mapping)
    df_se = df_se.rename({"municipality_id": "education_municipality_id",
                          "quarter_id": "education_quarter_id"}, axis=1)


    f_no_home = ((np.isnan(df_se["home_municipality_id"])) & 
                 (np.isnan(df_se["home_quarter_id"])))
    f_no_work = ((np.isnan(df_se["work_country_id"])) & 
                 (np.isnan(df_se["work_municipality_id"])) & 
                 (np.isnan(df_se["work_quarter_id"])))
    f_no_education = ((np.isnan(df_se["education_municipality_id"])) & 
                      (np.isnan(df_se["education_quarter_id"])))
    logger.info("Found %d observations without home location information", np.count_nonzero(f_no_home))
    logger.info("Found %d observations without work location information", np.count_nonzero(f_no_work))
    logger.info("Found %d observations without education location information", np.count_nonzero(f_no_education))

    # in certain cases it can happen that quarter is not reported for the work municipality
    # even though this municipality is divided into quarters.
    # This is relevant when assigning exact locations later in the synthesis.population.spatial.primary.work.locations

    #!!! what is here below seems not to be needed anymore as these situations do not happen with the new data

    # Now we face the problem that in an old structural survey we may assign
    # a municipality that has now quarters, or we find quarters that have been
    # resolved and now we just have a municipality in our zoning system.
    # Therefore the approach is as follows: Since we know the quarter and municipality
    # for all the STATPOP persons, we can sample one of them for each structural survey
    # observation. This has two purposes: First, we attach a physical location to each
    # structural survey observation, and second this physical location is also
    # consistent with the overall population density in the area. (Imagine there is
    # a municiaplity that has been divided in quarters: We would still like to
    # reproduce the same population density for the structural survey, but just by
    # knowing which municipality the people are from does not give us that information.)
    # Then, we use the new physical location to get the zone id of the overall zoning
    # system in the pipeline.

    # df_statpop = context.stage("data.statpop.statpop")[[
    #     "person_id", "home_x", "home_y", "home_municipality_id", "home_quarter_id", "home_zone_id"
    # ]]

    # # Assign coordinates in the home municipalities

    # se_municipality_ids = np.unique(df_se["home_municipality_id"].dropna()).astype(np.int)
    # for municipality_id in context.progress(se_municipality_ids,
    #                                         total=len(se_municipality_ids),
    #                                         label="Imputing home locations by municipality from STATPOP"):
    #     indices = np.where(df_statpop["home_municipality_id"] == municipality_id)[0]

    #     if len(indices) > 0:
    #         f = df_se["home_municipality_id"] == municipality_id
    #         indices = indices[rng.randint(len(indices), size=(np.count_nonzero(f),))]
    #         df_se.loc[f, "home_municipality_x"] = df_statpop.iloc[indices]["home_x"].values
    #         df_se.loc[f, "home_municipality_y"] = df_statpop.iloc[indices]["home_y"].values

    # unassigned_municipality_ids = np.unique(
    #     df_se[np.isnan(df_se["home_municipality_x"])]["home_municipality_id"].dropna())
    # print("A number of %d municipalities could not be assigned from STATPOP" % len(unassigned_municipality_ids))

    # for municipality_id in context.progress(unassigned_municipality_ids,
    #                                         total=len(unassigned_municipality_ids),
    #                                         label="Sampling home locations for municipalities"):
    #     f = np.isnan(df_se["home_municipality_x"]) & (df_se["home_municipality_id"] == municipality_id)
    #     row = df_municipalities[df_municipalities["municipality_id"] == municipality_id].iloc[0]
    #     coordinates = utils.sample_coordinates(row=row, count=np.count_nonzero(f), random_seed=random_seed)
    #     df_se.loc[f, "home_municipality_x"], df_se.loc[f, "home_municipality_y"] = coordinates[:, 0], coordinates[:, 1]

    # assert (~np.any(np.isnan(df_se["home_municipality_x"])))

    # # Assign coordinates in the home quarters

    # se_quarter_ids = np.unique(df_se["home_quarter_id"].dropna()).astype(np.int)
    # for quarter_id in context.progress(se_quarter_ids, 
    #                                    total=len(se_quarter_ids),
    #                                    label="Imputing home locations by quarter from STATPOP"):
    #     indices = np.where(df_statpop["home_quarter_id"] == quarter_id)[0]

    #     if len(indices) > 0:
    #         f = df_se["home_quarter_id"] == quarter_id
    #         indices = indices[rng.randint(len(indices), size=(np.count_nonzero(f),))]
    #         df_se.loc[f, "home_quarter_x"] = df_statpop.iloc[indices]["home_x"].values
    #         df_se.loc[f, "home_quarter_y"] = df_statpop.iloc[indices]["home_y"].values

    # unassigned_quarter_ids = np.unique(df_se[np.isnan(df_se["home_quarter_x"])]["home_quarter_id"].dropna())
    # print("A number of %d quarters could not be assigned from STATPOP" % len(unassigned_quarter_ids))

    # for quarter_id in context.progress(unassigned_quarter_ids, 
    #                                    total=len(unassigned_quarter_ids),
    #                                    label="Sampling home locations for municipalities"):
    #     f = np.isnan(df_se["home_quarter_x"]) & (df_se["home_quarter_id"] == quarter_id)
    #     row = df_quarters[df_quarters["quarter_id"] == quarter_id].iloc[0]
    #     coordinates = utils.sample_coordinates(row=row, count=np.count_nonzero(f), random_seed=random_seed)
    #     df_se.loc[f, "home_quarter_x"], df_se.loc[f, "home_quarter_y"] = coordinates[:, 0], coordinates[:, 1]

    # quarter_count = np.count_nonzero(~np.isnan(df_se["home_quarter_x"]))
    # municipality_count = np.count_nonzero(~np.isnan(df_se["home_municipality_x"]))

    # print("Homes assigned by municipality:", municipality_count - quarter_count)
    # print("Homes assigned by quarter:", quarter_count)

    # df_se.loc[:, "home_x"] = df_se.loc[:, "home_municipality_x"]
    # df_se.loc[:, "home_y"] = df_se.loc[:, "home_municipality_y"]

    # f_quarter = ~np.isnan(df_se["home_quarter_x"])
    # df_se.loc[f_quarter, "home_x"] = df_se.loc[f_quarter, "home_quarter_x"]
    # df_se.loc[f_quarter, "home_y"] = df_se.loc[f_quarter, "home_quarter_y"]

    # # Cleanup
    # df_se = df_se.drop(["home_municipality_x", 
    #                     "home_municipality_y", 
    #                     "home_quarter_x", 
    #                     "home_quarter_y"], axis=1)

    # assert (~np.any(np.isnan(df_se["home_x"])))

    # # The same we have to do with work places, except we can use STATENT here.

    # df_statent = context.stage("data.statent.statent")[[
    #     "enterprise_id", "x", "y", "municipality_id", "quarter_id", "zone_id"
    # ]]

    # # Assign coordinates in the work municipalities

    # se_municipality_ids = np.unique(df_se["work_municipality_id"].dropna()).astype(np.int)
    # for municipality_id in context.progress(se_municipality_ids,
    #                                         total=len(se_municipality_ids),
    #                                         label="Imputing work locations by municipality from STATENT"):
    #     indices = np.where(df_statent["municipality_id"] == municipality_id)[0]

    #     if len(indices) > 0:
    #         f = df_se["work_municipality_id"] == municipality_id
    #         indices = indices[rng.randint(len(indices), size=(np.count_nonzero(f),))]
    #         df_se.loc[f, "work_municipality_x"] = df_statent.iloc[indices]["x"].values
    #         df_se.loc[f, "work_municipality_y"] = df_statent.iloc[indices]["y"].values

    # unassigned_municipality_ids = np.unique(
    #     df_se[np.isnan(df_se["work_municipality_x"])]["work_municipality_id"].dropna())
    # print("A number of %d municipalities could not be assigned from STATENT" % len(unassigned_municipality_ids))

    # for municipality_id in context.progress(unassigned_municipality_ids,
    #                                         total=len(unassigned_municipality_ids),
    #                                         label="Sampling work locations for municipalities"):
    #     f = np.isnan(df_se["work_municipality_x"]) & (df_se["work_municipality_id"] == municipality_id)
    #     row = df_municipalities[df_municipalities["municipality_id"] == municipality_id].iloc[0]
    #     coordinates = utils.sample_coordinates(row=row, count=np.count_nonzero(f), random_seed=random_seed)
    #     df_se.loc[f, "work_municipality_x"], df_se.loc[f, "work_municipality_y"] = coordinates[:, 0], coordinates[:, 1]

    # # Assign coordinates in the work quarters

    # se_quarter_ids = np.unique(df_se["work_quarter_id"].dropna()).astype(np.int)
    # for quarter_id in context.progress(se_quarter_ids, 
    #                                    total=len(se_quarter_ids),
    #                                    label="Imputing work locations by quarter from STATENT"):
    #     indices = np.where(df_statent["quarter_id"] == quarter_id)[0]

    #     if len(indices) > 0:
    #         f = df_se["work_quarter_id"] == quarter_id
    #         indices = indices[rng.randint(len(indices), size=(np.count_nonzero(f),))]
    #         df_se.loc[f, "work_quarter_x"] = df_statent.iloc[indices]["x"].values
    #         df_se.loc[f, "work_quarter_y"] = df_statent.iloc[indices]["y"].values

    # unassigned_quarter_ids = np.unique(df_se[np.isnan(df_se["work_quarter_x"])]["work_quarter_id"].dropna())
    # print("A number of %d quarters could not be assigned from STATENT" % len(unassigned_quarter_ids))

    # for quarter_id in context.progress(unassigned_quarter_ids,
    #                                    total=len(unassigned_quarter_ids) ,
    #                                    label="Sampling work locations for municipalities"):
    #     f = np.isnan(df_se["work_quarter_x"]) & (df_se["work_quarter_id"] == quarter_id)
    #     row = df_quarters[df_quarters["quarter_id"] == quarter_id].iloc[0]
    #     coordinates = utils.sample_coordinates(row=row, count=np.count_nonzero(f), random_seed=random_seed)
    #     df_se.loc[f, "work_quarter_x"], df_se.loc[f, "work_quarter_y"] = coordinates[:, 0], coordinates[:, 1]

    # quarter_count = np.count_nonzero(~np.isnan(df_se["work_quarter_x"]))
    # municipality_count = np.count_nonzero(~np.isnan(df_se["work_municipality_x"]))

    # print("Work places assigned by municipality:", municipality_count - quarter_count)
    # print("Work places assigned by quarter:", quarter_count)

    # df_se.loc[:, "work_x"] = df_se.loc[:, "work_municipality_x"]
    # df_se.loc[:, "work_y"] = df_se.loc[:, "work_municipality_y"]

    # f_quarter = ~np.isnan(df_se["work_quarter_x"])
    # df_se.loc[f_quarter, "work_x"] = df_se.loc[f_quarter, "work_quarter_x"]
    # df_se.loc[f_quarter, "work_y"] = df_se.loc[f_quarter, "work_quarter_y"]

    # # Cleanup
    # df_se = df_se.drop(["work_municipality_x", 
    #                     "work_municipality_y", 
    #                     "work_quarter_x", 
    #                     "work_quarter_y"], axis=1)

    #!!! End of the ignored part
    
    logger.info("Imputing home zones ...")
    df_se = zones.impute(df_se, df_zones, 
                                      zone_id_prefix="home_",
                                      zone_fields={"quarter": "home_quarter_id",
                                                    "municipality": "home_municipality_id",
                                                    "country": "country_id",
                                                    "nuts": "nuts_id",
                                                    "postal_code": "postal_code"})

    logger.info("Imputing work zones ...")
    df_se = zones.impute(df_se, df_zones, zone_id_prefix="work_", 
                                      zone_fields={"quarter": "work_quarter_id",
                                                    "municipality": "work_municipality_id",
                                                    "country": "work_country_id",
                                                    "nuts": "nuts_id",
                                                    "postal_code": "postal_code"})
    logger.info("Imputing education zones ...")
    df_se = zones.impute(df_se, df_zones, zone_id_prefix="education_", 
                                      zone_fields={"quarter": "education_quarter_id",
                                                    "municipality": "education_municipality_id",
                                                    "country": "country_id",
                                                    "nuts": "nuts_id",
                                                    "postal_code": "postal_code"})
    logger.info("Imputing municipality type at home location ...")
    df_municipality_types = context.stage("data.spatial.municipality_types")
    df_se = df_se.reset_index(drop=True)
    df_se.insert(0, "id", np.arange(1, len(df_se) + 1, dtype=int))

    df_spatial = pd.DataFrame(df_se[["id", "home_municipality_id"]])
    df_spatial = df_spatial.rename(columns={"home_municipality_id" : "municipality_id"})
    df_spatial = impute_municipality_type(df_spatial, df_municipality_types)

    df_se = pd.merge(
        df_se, df_spatial[["id", "municipality_type"]],
        on="id"
    )

    # for some individuals in Berufschule it is not recreded that they are employed
    # we correct this here for is_student, employed and job_position variables
    df_se.loc[df_se["current_education"].eq(6), "employed"] = 1
    df_se.loc[df_se["current_education"].eq(6), "job_position"] = 50
    #recode nationality to 0: swiss 1: others
    col = "nationality"
    assert df_se[col].isin([1, 2]).all(), f"{col} has values other than 1/2: {df_se.loc[~df_se[col].isin([1,2]), col].unique()}"
    
    df_se["nationality"]-=1
    return df_se[[
        "home_municipality_id", "home_quarter_id", "home_zone_id", "home_zone_level",
        "work_country_id", "work_municipality_id", "work_quarter_id", "work_zone_id", "work_zone_level",
        "mode", "weight",  "age", "sex", "employed", "job_position", "canton_id", "district_id", "is_student",
        "education_municipality_id", "education_quarter_id", "education_zone_id", "education_zone_level", "municipality_type",
        "nationality", "crowfly_distance_to_work", "crowfly_distance_to_school", "freq_per_week", "commute_to_work", "start_work", 
        "highest_completed_education", "isco_code",       
    ]]


def get_filtered_data(context, filter_type = "all"):
    df_od = context.stage("data.structural_survey.structural_survey")
    df_od = df_od[~pd.isnull(df_od["home_zone_id"])]
    df_od = df_od[~pd.isnull(df_od["work_zone_id"])]
    df_od = df_od[~(df_od["work_zone_level"] == "country")]
    df_od = df_od[~(df_od["home_zone_level"] == "country")]
    df_od = df_od[df_od["employed"] == 1]

    match filter_type:
        case "all":
            return df_od
        case "fixed":
            return df_od[df_od["start_work"] > 2]
        case "moving":
            return df_od[df_od["start_work"] == 2]
        case "remote":
            return df_od[df_od["start_work"] == 1]
        case _:
            raise ValueError("Unknown filter type: %s" % filter_type)