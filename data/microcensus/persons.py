import pandas as pd
import numpy as np
import data.utils
import data.constants as c

def configure(context, require):
    require.config("raw_data_path")
    # require.cache = False

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    df_mz_persons = pd.read_csv(
        "%s/microcensus/zielpersonen.csv" % raw_data_path, sep = ",", encoding = "latin1")

    df_mz_persons["age"] = df_mz_persons["alter"]
    df_mz_persons["sex"] = df_mz_persons["gesl"] - 1 # Make zero-based
    df_mz_persons["person_id"] = df_mz_persons["HHNR"]
    df_mz_persons["person_weight"] = df_mz_persons["WP"]

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

    return df_mz_persons[[
        "person_id",
        "age", "sex",
        "marital_status",
        "driving_license",
        "car_availability",
        "employed",
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
        "weekend"
    ]]
