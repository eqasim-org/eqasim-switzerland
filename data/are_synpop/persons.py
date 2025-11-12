import pandas as pd
import numpy as np

import data.spatial.cantons
from data.spatial.municipality_types import impute as impute_municipality_types
import data.spatial.ovgk
import data.spatial.utils
import data.spatial.zones

def configure(context):
    context.config("data_path")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.municipality_types")
    context.stage("data.statpop.density")
    context.stage("data.spatial.cantons")
    context.stage("data.spatial.ovgk")
    context.stage("data.constants")

def execute(context):
    data_path = context.config("data_path")
    c         = context.stage("data.constants")

    synpop_path = "%s/SynpopAre/data/2754_SynPop2022_Data_v1.0.zip" % data_path
    df   = pd.read_csv(synpop_path, sep = ";")

    df.loc[df["sex"]=="F", "sex"] = c.SEX_FEMALE
    df.loc[df["sex"]=="M", "sex"] = c.SEX_MALE

    df.loc[df["nation"]=="swiss", "nationality"]     = 0
    df.loc[df["nation"]=="non-swiss", "nationality"] = 1

    df["home_x"] = df["xcoord"]
    df["home_y"] = df["ycoord"]

    df["age_class"] = df["age"]

    df["child_in_household"]   = df["child_in_household"].astype(int)
    df["driving_license"]      = df["driving_licence"].astype(int)
    df["household_size_class"] = df["household_size"]
    df["N_children_under_18"]  = df["child_in_household"]

    del df["nation"]
    del df["xcoord"]
    del df["ycoord"]
    del df["age"]
    del df["driving_licence"]
    del df["zone_id"]
    del df["household_size"]
    del df["child_in_household"]

    # Identifying children under the age of 6 from the education variables
    df.loc[(df["age_class"]=="0-17") & (df["education"]=="primary") & (df["position_in_edu"].isna()) & (df["position_in_bus"].isna()), "age_class"] = "0-5"
    df.loc[ df["age_class"]=="0-17", "age_class"]                                                                                                    = "6-17"

    # Age class to int
    df["age_class"] = df["age_class"].replace({"0-5":0, "6-17":1, "18-24": 2, "25-44":3, "45-64":4, "65-74":5, "75+":6})

    # Cleaning employment and education status
    df.loc[(df["position_in_edu"]=="pupil"), "employment_status"]                                     = "student"
    df.loc[(df["position_in_edu"]=="apprentice"), "employment_status"]                                = "working student"
    df.loc[(df["position_in_edu"]=="student") & (df["position_in_bus"].isna()), "employment_status"]  = "student"
    df.loc[(df["position_in_edu"]=="student") & (~df["position_in_bus"].isna()), "employment_status"] = "working student"
    df.loc[(df["position_in_edu"].isna()) & (~df["position_in_bus"].isna()), "employment_status"]     = "active"
    df.loc[(df["position_in_edu"].isna()) & (df["position_in_bus"].isna()), "employment_status"]      = "inactive"

    # Maybe there are too many different employment categories now. Andwe are losing information on socio-professional class, which is used for matching in IdF.
    # TODO adjust classification if needed.

    #filter = (df["employment_status"]=="active") | (df["employment_status"]=="working student")| (df["employment_status"]=="apprentice")
    #df.loc[filter, "employment_status"] = df[filter]["employment_status"] + "_" + df[filter]["level_of_employment"]

    # Number of cars and bikes
    df["number_of_cars_class"]  = df["cars_in_hh"]
    df["number_of_bikes_class"] = df["bike_ownership"]

    # Make number of cars class compatible with MZ
    df.loc[df["number_of_cars_class"]=="3+", "number_of_cars_class"] = 3
    df["number_of_cars_class"] = df["number_of_cars_class"].astype(int)

    # Number of bikes class cannot be used for matching as it is given at the household level in MZ
    df.loc[df["number_of_bikes_class"] == "null", "number_of_bikes_class"]                                                            = c.BIKE_AVAILABILITY_FOR_NONE
    df.loc[df["number_of_bikes_class"].isin(["bike", "eBike25", "eBike45", "bike+eBike25", "bike+eBike45"]), "number_of_bikes_class"] = c.BIKE_AVAILABILITY_FOR_ALL

    del df["cars_in_hh"]
    del df["bike_ownership"]

    # PT subscription ownership
    df["subscriptions"] = df["public_transport"]
    del df["public_transport"]

    # Impute spatial information
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_zones = context.stage("data.spatial.zones")
    df_municipality_types = context.stage("data.spatial.municipality_types")
    df_quarters = context.stage("data.spatial.quarters")
    df_cantons = context.stage("data.spatial.cantons")

    df_spatial = pd.DataFrame(df[["person_id", "home_x", "home_y"]])
    df_spatial = data.spatial.utils.to_gpd(context, df_spatial, "home_x", "home_y", coord_type="home")

    # Impute municipalities
    df_spatial = (data.spatial.utils.impute(context, df_spatial, df_municipalities, "person_id", "municipality_id",
                                            zone_type="municipality", point_type="home")[
        ["person_id", "municipality_id", "geometry"]])
    df_spatial["municipality_id"] = df_spatial["municipality_id"].astype(np.int)

    # Impute quarters
    df_spatial = (data.spatial.utils.impute(context, df_spatial, df_quarters, "person_id", "quarter_id",
                                            fix_by_distance=False, zone_type="quarter", point_type="home")[
        ["person_id", "municipality_id", "quarter_id", "geometry"]])

    # Impute cantons
    df_spatial = (data.spatial.utils.impute(context, df_spatial, df_cantons, "person_id", "canton_id",
                                            zone_type="canton", point_type="home")[
        ["person_id", "municipality_id", "quarter_id", "canton_id", "geometry"]])

    # Impute municipality types
    df_spatial = impute_municipality_types(df_spatial, df_municipality_types)

    # Impute zones
    df_spatial = data.spatial.zones.impute(df_spatial, df_zones)

    assert (len(df) == len(df_spatial))

    df = pd.merge(
        df, df_spatial[["person_id", "zone_id", "municipality_type", "municipality_id", "quarter_id", "canton_id"]],
        on="person_id"
    )

    df["home_zone_id"]         = df["zone_id"]
    df["home_municipality_id"] = df["municipality_id"]
    df["home_quarter_id"]      = df["quarter_id"]
    df["canton_id"]            = df["canton_id"].astype(int)

    # Impute SP region
    df = data.spatial.cantons.impute_sp_region(df)

    # Impute OV Guteklasse
    df_ovgk = context.stage("data.spatial.ovgk")
    df_spatial = data.spatial.ovgk.impute(context, df_ovgk, df_spatial, ["person_id"], chunk_size=1e3, point_type="home")
    df = pd.merge(df, df_spatial[["person_id", "ovgk"]], on=["person_id"], how="left")

    # Save original statpop person and household ids
    df["synpop_person_id"]    = df["person_id"].astype(int)

    # Wrap everything up
    df = df[[
        "person_id", "sex",
        "home_x", "home_y",
        "nationality",
        "household_income",
        "N_children_under_18",
        "number_of_cars_class",
        "employment_status",
        "driving_license",
        "subscriptions",
        "level_of_employment", "education",
        "position_in_edu", "position_in_bus",
        "age_class", "household_size_class",
        "home_zone_id", "municipality_type",
        "home_municipality_id", "home_quarter_id", "canton_id", 
        "sp_region", "ovgk",
        "synpop_person_id"]]

    return df



