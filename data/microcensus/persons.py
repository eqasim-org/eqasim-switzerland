import numpy as np
import pandas as pd

import data.constants as c
import data.microcensus.income
import data.utils


def configure(context):
    context.config("data_path")
    context.stage("data.microcensus.households")
    context.stage("data.microcensus.trips")

def execute(context):
    data_path = context.config("data_path")

    df_mz_persons = pd.read_csv(
        "%s/microcensus/zielpersonen.csv" % data_path,
        sep = ",", encoding = "latin1", parse_dates = ["USTag"]
    )

    df_mz_persons["age"] = df_mz_persons["alter"]
    df_mz_persons["sex"] = df_mz_persons["gesl"] - 1 # Make zero-based
    df_mz_persons["person_id"] = df_mz_persons["HHNR"]
    df_mz_persons["person_weight"] = df_mz_persons["WP"]
    df_mz_persons["date"] = df_mz_persons["USTag"]

    # Marital status
    df_mz_persons.loc[df_mz_persons["zivil"] == 1, "marital_status"] = c.MARITAL_STATUS_SINGLE
    df_mz_persons.loc[df_mz_persons["zivil"] == 2, "marital_status"] = c.MARITAL_STATUS_MARRIED
    df_mz_persons.loc[df_mz_persons["zivil"] == 3, "marital_status"] = c.MARITAL_STATUS_SEPARATE
    df_mz_persons.loc[df_mz_persons["zivil"] == 4, "marital_status"] = c.MARITAL_STATUS_SEPARATE
    df_mz_persons.loc[df_mz_persons["zivil"] == 5, "marital_status"] = c.MARITAL_STATUS_SINGLE
    df_mz_persons.loc[df_mz_persons["zivil"] == 6, "marital_status"] = c.MARITAL_STATUS_MARRIED
    df_mz_persons.loc[df_mz_persons["zivil"] == 7, "marital_status"] = c.MARITAL_STATUS_SEPARATE

    # Driving license
    df_mz_persons["driving_license"] = df_mz_persons["f20400a"] == 1

    # Car availability
    df_mz_persons["car_availability"] = c.CAR_AVAILABILITY_NEVER
    df_mz_persons.loc[df_mz_persons["f42100e"] == 1, "car_availability"] = c.CAR_AVAILABILITY_ALWAYS
    df_mz_persons.loc[df_mz_persons["f42100e"] == 2, "car_availability"] = c.CAR_AVAILABILITY_SOMETIMES
    df_mz_persons.loc[df_mz_persons["f42100e"] == 3, "car_availability"] = c.CAR_AVAILABILITY_NEVER

    # Employment (TODO: I know that LIMA uses a more fine-grained category here)
    df_mz_persons["employed"] = df_mz_persons["f40800_01"] != -99

    # Infer age class
    df_mz_persons["age_class"] = np.digitize(df_mz_persons["age"], c.AGE_CLASS_UPPER_BOUNDS)

    # Fix marital status
    data.utils.fix_marital_status(df_mz_persons)

    # Day of the observation
    df_mz_persons["weekend"] = False
    df_mz_persons.loc[df_mz_persons["tag"] == 6, "weekend"] = True
    df_mz_persons.loc[df_mz_persons["tag"] == 7, "weekend"] = True

    # Here we extract a bit more than Kirill, but most likely it will be useful later

    df_mz_persons["subscriptions_ga"] = df_mz_persons["f41610a"] == 1
    df_mz_persons["subscriptions_halbtax"] = df_mz_persons["f41610b"] == 1
    df_mz_persons["subscriptions_verbund"] = df_mz_persons["f41610c"] == 1
    df_mz_persons["subscriptions_strecke"] = df_mz_persons["f41610d"] == 1
    df_mz_persons["subscriptions_gleis7"] = df_mz_persons["f41610e"] == 1
    df_mz_persons["subscriptions_junior"] = df_mz_persons["f41610f"] == 1
    df_mz_persons["subscriptions_other"] = df_mz_persons["f41610g"] == 1

    df_mz_persons["subscriptions_ga_class"] = df_mz_persons["f41651"] == 1
    df_mz_persons["subscriptions_verbund_class"] = df_mz_persons["f41653"] == 1
    df_mz_persons["subscriptions_strecke_class"] = df_mz_persons["f41654"] == 1

    # Education
    df_mz_persons["highest_education"] = np.nan
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([1, 2, 3, 4]), "highest_education"] = "primary"
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([5, 6, 7, 8, 9, 10, 11, 12]), "highest_education"] = "secondary"
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([13, 14, 15, 16]), "highest_education"] = "tertiary_professional"
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([17, 18, 19]), "highest_education"] = "tertiary_academic"
    df_mz_persons["highest_education"] = df_mz_persons["highest_education"].astype("category")

    # Parking
    df_mz_persons["parking_work"] = "unknown"
    df_mz_persons.loc[df_mz_persons["f41300"] == 1, "parking_work"] = "free"
    df_mz_persons.loc[df_mz_persons["f41300"] == 2, "parking_work"] = "paid"
    df_mz_persons.loc[df_mz_persons["f41300"] == 3, "parking_work"] = "no"
    df_mz_persons["parking_work"] = df_mz_persons["parking_work"].astype("category")

    df_mz_persons["parking_education"] = "unknown"
    df_mz_persons.loc[df_mz_persons["f41301"] == 1, "parking_education"] = "free"
    df_mz_persons.loc[df_mz_persons["f41301"] == 2, "parking_education"] = "paid"
    df_mz_persons.loc[df_mz_persons["f41301"] == 3, "parking_education"] = "no"
    df_mz_persons["parking_education"] = df_mz_persons["parking_education"].astype("category")

    df_mz_persons["parking_cost_work"] = np.maximum(0, df_mz_persons["f41400"].astype(np.float))
    df_mz_persons["parking_cost_education"] = np.maximum(0, df_mz_persons["f41401"].astype(np.float))

    # Wrap up
    df_mz_persons = df_mz_persons[[
        "person_id",
        "age", "sex",
        "marital_status",
        "driving_license",
        "car_availability",
        "employed",
        "highest_education",
        "parking_work", "parking_cost_work",
        "parking_education", "parking_cost_education",
        "subscriptions_ga",
        "subscriptions_halbtax",
        "subscriptions_verbund",
        "subscriptions_strecke",
        "subscriptions_gleis7",
        "subscriptions_junior",
        "subscriptions_other",
        "subscriptions_ga_class",
        "subscriptions_verbund_class",
        "subscriptions_strecke_class",
        "age_class", "person_weight",
        "weekend", "date"
    ]]

    # Merge in the other data sets
    df_mz_households = context.stage("data.microcensus.households")
    df_mz_trips = context.stage("data.microcensus.trips")

    df_mz_persons = pd.merge(df_mz_persons, df_mz_households)
    df_mz_persons = data.microcensus.income.impute(df_mz_persons)

    remove_ids = set(df_mz_persons["person_id"]) - set(df_mz_trips["person_id"])
    initial_size = len(df_mz_persons)

    df_mz_persons = df_mz_persons[~df_mz_persons["person_id"].isin(remove_ids)]

    # Note: Around 7000 of them are those, which do not even have an activity chain in the first place
    # because they have not been asked.
    print("  Removed %d (%.2f%%) persons from MZ because of insufficient trip data" % (
        len(remove_ids), 100.0 * len(remove_ids) / initial_size
    ))

    # Add car passenger flag
    car_passenger_ids = df_mz_trips.loc[df_mz_trips["mode"] == "car_passenger", "person_id"].unique()
    df_mz_persons["is_car_passenger"] = df_mz_persons["person_id"].isin(car_passenger_ids)

    return df_mz_persons
