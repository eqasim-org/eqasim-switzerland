import numpy as np
import pandas as pd

import data.constants as c
import data.utils

def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")

    household_persons = pd.read_csv(
        "%s/microcensus/haushaltspersonen.csv" % data_path, sep=",", encoding="latin1")
    
    household_persons = household_persons[["HHNR", "HPNR", "alter", "gesl"]]

    household_persons.columns = ["household_id", "hhpers_id", "age", "sex"]

    household_persons.loc[:, "young_child"] = (household_persons["age"] <  6) & (household_persons["age"] >= 0)
    household_persons.loc[:, "adult"] = household_persons["age"] >= 18
    household_persons.loc[:, "child"] = (household_persons["age"] <  19) & (household_persons["age"] >= 0)

    nb_small_children = household_persons.groupby(["household_id"])["young_child"].sum().reset_index()
    nb_small_children.columns = ["household_id", "n_small_children"]

    nb_adults = household_persons.groupby(["household_id"])["adult"].sum().reset_index()
    nb_adults.columns = ["household_id", "n_adults"]

    nb_children = household_persons.groupby(["household_id"])["child"].sum().reset_index()
    nb_children.columns = ["household_id", "n_children"]

    household_info = pd.merge(nb_small_children, nb_adults, how = "outer", on = "household_id")
    household_info = pd.merge(household_info, nb_children, how = "outer", on = "household_id")

    return household_persons, household_info