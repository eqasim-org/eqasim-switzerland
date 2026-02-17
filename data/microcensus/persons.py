import numpy as np
import pandas as pd

import data.microcensus.income
import data.utils


def configure(context):
    context.config("data_path")
    #context.config("weekend", default = False)

    context.stage("data.microcensus.households")
    context.stage("data.microcensus.trips")
    context.stage("data.constants")

def execute(context):
    data_path = context.config("data_path")
    c         = context.stage("data.constants")

    df_mz_persons = pd.read_csv(
        "%s/microcensus/zielpersonen.csv" % data_path,
        sep = ",", encoding = "latin1", parse_dates = ["USTag"]
    )

    df_mz_persons["age"] = df_mz_persons["alter"]
    df_mz_persons["sex"] = df_mz_persons["gesl"] - 1 # Make zero-based
    df_mz_persons["person_id"] = df_mz_persons["HHNR"]
    df_mz_persons["person_weight"] = df_mz_persons["WP"]
    df_mz_persons["date"] = df_mz_persons["USTag"]

    df_mz_persons["is_swiss"] = df_mz_persons["f43500"]

    columns = ["person_id", "person_weight", "age", "sex", "date", "is_swiss"]

    # Marital status
    df_mz_persons.loc[df_mz_persons["zivil"] == 1, "marital_status"] = c.MARITAL_STATUS_SINGLE
    df_mz_persons.loc[df_mz_persons["zivil"] == 2, "marital_status"] = c.MARITAL_STATUS_MARRIED
    df_mz_persons.loc[df_mz_persons["zivil"] == 3, "marital_status"] = c.MARITAL_STATUS_SEPARATE
    df_mz_persons.loc[df_mz_persons["zivil"] == 4, "marital_status"] = c.MARITAL_STATUS_SEPARATE
    df_mz_persons.loc[df_mz_persons["zivil"] == 5, "marital_status"] = c.MARITAL_STATUS_SINGLE
    df_mz_persons.loc[df_mz_persons["zivil"] == 6, "marital_status"] = c.MARITAL_STATUS_MARRIED
    df_mz_persons.loc[df_mz_persons["zivil"] == 7, "marital_status"] = c.MARITAL_STATUS_SEPARATE

    columns.append("marital_status")

    # Driving license
    df_mz_persons["driving_license"] = df_mz_persons["f20400a"] == 1

    columns.append("driving_license")

    # Learning driving license
    df_mz_persons["learning_driving_license"] = df_mz_persons["f20400c"] == 1

    columns.append("learning_driving_license")

    # Car availability
    df_mz_persons["car_availability"] = c.CAR_AVAILABILITY_NEVER
    df_mz_persons.loc[df_mz_persons["f42100e"] == 1, "car_availability"] = c.CAR_AVAILABILITY_ALWAYS
    df_mz_persons.loc[df_mz_persons["f42100e"] == 2, "car_availability"] = c.CAR_AVAILABILITY_SOMETIMES
    df_mz_persons.loc[df_mz_persons["f42100e"] == 3, "car_availability"] = c.CAR_AVAILABILITY_NEVER

    columns.append("car_availability")

    # bike availability
    df_mz_persons["bike_availability"] = c.BIKE_AVAILABILITY_NEVER
    df_mz_persons.loc[df_mz_persons["f42100a"] == 1, "bike_availability"] = c.BIKE_AVAILABILITY_ALWAYS
    df_mz_persons.loc[df_mz_persons["f42100a"] == 2, "bike_availability"] = c.BIKE_AVAILABILITY_SOMETIMES
    df_mz_persons.loc[df_mz_persons["f42100a"] == 3, "bike_availability"] = c.BIKE_AVAILABILITY_NEVER

    columns.append("bike_availability")

    # Employment (TODO: I know that LIMA uses a more fine-grained category here)
    df_mz_persons["employed"] = df_mz_persons["f40800_01"] != -99

    columns.append("employed")

    # Infer age class
    df_mz_persons["age_class"] = np.digitize(df_mz_persons["age"], c.AGE_CLASS_UPPER_BOUNDS)

    columns.append("age_class")

    # Fix marital status
    data.utils.fix_marital_status(df_mz_persons, c)

    # Day of the observation
    df_mz_persons["weekend"] = False
    df_mz_persons.loc[df_mz_persons["tag"] >= 6, "weekend"] = True

    df_mz_persons["workday"] = False
    df_mz_persons.loc[df_mz_persons["tag"] <= 5, "workday"] = True

    df_mz_persons["day"] = "Monday"
    df_mz_persons.loc[df_mz_persons["tag"] == 2, "day"] = "Tuesday"
    df_mz_persons.loc[df_mz_persons["tag"] == 3, "day"] = "Wednesday"
    df_mz_persons.loc[df_mz_persons["tag"] == 4, "day"] = "Thursday"
    df_mz_persons.loc[df_mz_persons["tag"] == 5, "day"] = "Friday"
    df_mz_persons.loc[df_mz_persons["tag"] == 6, "day"] = "Saturday"
    df_mz_persons.loc[df_mz_persons["tag"] == 7, "day"] = "Sunday"

    columns.append("weekend")
    columns.append("workday")
    columns.append("day")

    # Here we extract a bit more than Kirill, but most likely it will be useful later

    df_mz_persons["subscriptions_ga"]      = df_mz_persons["f41610a"] == 1
    df_mz_persons["subscriptions_halbtax"] = df_mz_persons["f41610b"] == 1
    df_mz_persons["subscriptions_verbund"] = df_mz_persons["f41610c"] == 1
    df_mz_persons["subscriptions_strecke"] = df_mz_persons["f41610d"] == 1
    df_mz_persons["subscriptions_gleis7"]  = df_mz_persons["f41610e"] == 1
    df_mz_persons["subscriptions_junior"]  = df_mz_persons["f41610f"] == 1
    df_mz_persons["subscriptions_other"]   = df_mz_persons["f41610g"] == 1

    df_mz_persons["subscriptions_ga_class"]      = df_mz_persons["f41651"] == 1
    df_mz_persons["subscriptions_verbund_class"] = df_mz_persons["f41653"] == 1
    df_mz_persons["subscriptions_strecke_class"] = df_mz_persons["f41654"] == 1

    columns.extend(["subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "subscriptions_strecke", "subscriptions_gleis7", "subscriptions_junior",
                    "subscriptions_other", "subscriptions_ga_class", "subscriptions_verbund_class", "subscriptions_strecke_class"])
    
    # Summary of PT subscriptions - matching with classification from ARE synpop
    df_mz_persons["subscriptions"] = "null"
    df_mz_persons.loc[df_mz_persons["subscriptions_ga"],            "subscriptions"] = "GA"
    df_mz_persons.loc[df_mz_persons["subscriptions_ga_class"],      "subscriptions"] = "GA"
    df_mz_persons.loc[df_mz_persons["subscriptions_halbtax"],       "subscriptions"] = "HTA"
    df_mz_persons.loc[df_mz_persons["subscriptions_verbund"],       "subscriptions"] = "VA"
    df_mz_persons.loc[df_mz_persons["subscriptions_verbund_class"], "subscriptions"] = "VA"
    df_mz_persons.loc[(df_mz_persons["subscriptions_halbtax"]) & ((df_mz_persons["subscriptions_verbund"]) | (df_mz_persons["subscriptions_verbund_class"])), "subscriptions"] = "HTA+VA"

    columns.append("subscriptions")

    # Education
    df_mz_persons["highest_education"]                                                               = "secondary"
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([1, 2, 3, 4]), "highest_education"]                = "primary"
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([5, 6, 7, 8, 9, 10, 11, 12]), "highest_education"] = "secondary"
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([13, 14, 15, 16]), "highest_education"]            = "tertiary_professional"
    df_mz_persons.loc[df_mz_persons["HAUSB"].isin([17, 18, 19]), "highest_education"]                = "tertiary_academic"
    df_mz_persons["highest_education"] = df_mz_persons["highest_education"].astype("category")

    columns.append("highest_education")

    # Employment status
    df_mz_persons.loc[:, "employment_status"]                                                                   = 0
    df_mz_persons.loc[df_mz_persons["f40800_01"] == 5, "employment_status"]                                     = 3
    df_mz_persons.loc[(df_mz_persons["f40800_01"] < 5) & (df_mz_persons["f40800_01"] > 0), "employment_status"] = 1
    df_mz_persons.loc[df_mz_persons["age"]<15, "employment_status"]                                             = 2
    df_mz_persons.loc[(df_mz_persons["f41001a"]==32) | (df_mz_persons["f41001b"]==32), "employment_status"]     = 2
    df_mz_persons.loc[(df_mz_persons["f41000a"]==32) | (df_mz_persons["f41000b"]==32), "employment_status"]     = 3

    columns.append("employment_status")

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
    df_mz_persons["occupation"] = df_mz_persons["ISCO_08"]
    
    # BSTELL codes
    # -99	-99.Alter der Zielperson < 15 Jahre
    # -98	-98.keine Antwort
    # -97	-97.weiss nicht
    # 11	11.Selbständige/Selbständiger mit Arbeitnehmer(n)
    # 12	12.Selbständige/Selbständiger ohne Arbeitnehmer
    # 20	20.Mitarbeitendes Familienmitglied
    # 31	31.Arbeitnehmerin / Arbeitnehmer in Unternehmensleitung
    # 32	32.Arbeitnehmerin / Arbeitnehmer mit Vorgesetztenfunktion
    # 33	33.Arbeitnehmerin / Arbeitnehmer ohne Vorgesetztenfunktion
    # 40	40.Lehrtochter / Lehrling
    # 50	50.Erwerbslose / Erwerbsloser
    # 60	60.Nichterwerbspersonen (falls >= 15 Jahre alt)

    df_mz_persons["job_position"] = df_mz_persons["BSTELL"]
    columns.extend(["parking_work", "parking_education", "parking_cost_work", "parking_cost_education", "occupation", "job_position"])

    # Wrap up
    df_mz_persons = df_mz_persons[columns]

    # Merge in the other data sets
    df_mz_households = context.stage("data.microcensus.households")
    df_mz_trips, filterout_person_ids = context.stage("data.microcensus.trips")

    df_mz_persons = pd.merge(df_mz_persons, df_mz_households)
    df_mz_persons = data.microcensus.income.impute(df_mz_persons)

    initial_size = len(df_mz_persons)

    # This will only filter out persons that do not have enough information in the trips file
    # it will still keep persons that did not report any trips
    df_mz_persons = df_mz_persons[~df_mz_persons["person_id"].isin(filterout_person_ids)]

    #if context.config("weekend"):
    #    df_mz_persons = df_mz_persons[df_mz_persons["weekend"]]
    #else:
    #    df_mz_persons = df_mz_persons[~df_mz_persons["weekend"]]

    then_size = len(df_mz_persons)
    home_ids = set(df_mz_persons["person_id"]) - set(df_mz_trips["person_id"])

    # Note: Around 7000 of them are those, which do not even have an activity chain in the first place
    # because they have not been asked.
    print("  Removed %d (%.2f%%) persons from MZ because of insufficient trip data" % (
        len(filterout_person_ids), 100.0 * len(filterout_person_ids) / initial_size
    ))
    
    print("  Percentage of agents staying home (not weighted): %d (%.2f%%)" % (
        len(home_ids), 100.0 * len(home_ids) / then_size
    ))

    # Add car passenger flag
    car_passenger_ids = df_mz_trips.loc[df_mz_trips["mode"] == "car_passenger", "person_id"].unique()
    df_mz_persons["is_car_passenger"] = df_mz_persons["person_id"].isin(car_passenger_ids)

    return df_mz_persons
