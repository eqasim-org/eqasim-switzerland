import numpy as np
import pandas as pd

import data.microcensus.income
import data.utils


def configure(context):
    context.config("data_path")
    #context.config("weekend", default = False)

    context.stage("data.microcensus.21.households")
    context.stage("data.constants")

def execute(context):
    data_path = context.config("data_path")
    c         = context.stage("data.constants")

    df_mz_persons = pd.read_csv(
        "%s/microcensus/21/zielpersonen.csv" % data_path,
        sep = ";", encoding = "latin1", parse_dates = ["USTag"]
    )

    df_mz_persons["age"] = df_mz_persons["alter"]
    df_mz_persons["sex"] = df_mz_persons["gesl"] - 1 # Make zero-based
    df_mz_persons["person_id"] = df_mz_persons["HHNR"]
    df_mz_persons["person_weight"] = df_mz_persons["WP"]
    df_mz_persons["date"] = df_mz_persons["USTag"]

    df_mz_persons["is_swiss"] = df_mz_persons["nation"]
    df_mz_persons["is_swiss"] = np.where(df_mz_persons["is_swiss"] == 8100, 0, 1) # 8100 is coded as swiss
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
    df_mz_persons.loc[df_mz_persons["Tag"] >= 6, "weekend"] = True

    df_mz_persons["workday"] = False
    df_mz_persons.loc[df_mz_persons["Tag"] <= 5, "workday"] = True

    df_mz_persons["day"] = "Monday"
    df_mz_persons.loc[df_mz_persons["Tag"] == 2, "day"] = "Tuesday"
    df_mz_persons.loc[df_mz_persons["Tag"] == 3, "day"] = "Wednesday"
    df_mz_persons.loc[df_mz_persons["Tag"] == 4, "day"] = "Thursday"
    df_mz_persons.loc[df_mz_persons["Tag"] == 5, "day"] = "Friday"
    df_mz_persons.loc[df_mz_persons["Tag"] == 6, "day"] = "Saturday"
    df_mz_persons.loc[df_mz_persons["Tag"] == 7, "day"] = "Sunday"

    columns.append("weekend")
    columns.append("workday")
    columns.append("day")

    # Here we extract a bit more than Kirill, but most likely it will be useful later

    df_mz_persons["subscriptions_ga"]      = df_mz_persons["f41600_01a"] == 1
    df_mz_persons["subscriptions_halbtax"] = df_mz_persons["f41600_01b"] == 1
    df_mz_persons["subscriptions_verbund"] = df_mz_persons["f41600_01c"] == 1
    df_mz_persons["subscriptions_strecke"] = df_mz_persons["f41600_01d"] == 1
    df_mz_persons["subscriptions_gleis7"]  = df_mz_persons["f41600_01e"] == 1
    df_mz_persons["subscriptions_junior"]  = df_mz_persons["f41600_01f"] == 1
    df_mz_persons["subscriptions_other"]   = df_mz_persons["f41600_01g"] == 1

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
    df_mz_persons["highest_education"]                                                               = "secondary" # those that did not report
    df_mz_persons.loc[df_mz_persons["HAUSB3"].isin([1]), "highest_education"]                = "primary"
    df_mz_persons.loc[df_mz_persons["HAUSB3"].isin([2]), "highest_education"] = "secondary"
    df_mz_persons.loc[df_mz_persons["HAUSB3"].isin([3]), "highest_education"]            = "tertiary"
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
    columns.extend(["parking_work", "parking_education", "parking_cost_work", "parking_cost_education", "job_position"])

    # Wrap up
    df_mz_persons = df_mz_persons[columns]

    # Merge in the other data sets
    df_mz_households = context.stage("data.microcensus.21.households")

    df_mz_persons = pd.merge(df_mz_persons, df_mz_households)
    df_mz_persons = data.microcensus.income.impute(df_mz_persons)

    initial_size = len(df_mz_persons)

    return df_mz_persons
