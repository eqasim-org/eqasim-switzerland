import pandas as pd

def configure(context):
    context.config("data_path")

    context.config("output_path")

def execute(context):
    data_path = context.config("data_path")

    household_persons = pd.read_csv(
        "%s/microcensus/haushaltspersonen.csv" % data_path, sep=",", encoding="latin1")
    
    household_persons = household_persons[["HHNR", "HPNR", "alter", "gesl"]]

    household_persons.columns = ["household_id", "hhpers_id", "age", "sex"]

    household_persons.loc[:, "children_under_3"]  = (household_persons["age"] <  3)  & (household_persons["age"] >= 0)
    household_persons.loc[:, "children_under_6"]  = (household_persons["age"] <  6)  & (household_persons["age"] >= 0)
    household_persons.loc[:, "children_under_12"] = (household_persons["age"] <  12) & (household_persons["age"] >= 0)
    household_persons.loc[:, "children_under_18"] = (household_persons["age"] <  18) & (household_persons["age"] >= 0)
    household_persons.loc[:, "adults"]            = (household_persons["age"] >= 18) 

    nb_children_under_3 = household_persons.groupby(["household_id"])["children_under_3"].sum().reset_index()
    nb_children_under_3.columns = ["household_id", "N_children_under_3"]

    nb_children_under_6 = household_persons.groupby(["household_id"])["children_under_6"].sum().reset_index()
    nb_children_under_6.columns = ["household_id", "N_children_under_6"]

    nb_children_under_12 = household_persons.groupby(["household_id"])["children_under_12"].sum().reset_index()
    nb_children_under_12.columns = ["household_id", "N_children_under_12"]

    nb_children_under_18 = household_persons.groupby(["household_id"])["children_under_18"].sum().reset_index()
    nb_children_under_18.columns = ["household_id", "N_children_under_18"]

    nb_adults = household_persons.groupby(["household_id"])["adults"].sum().reset_index()
    nb_adults.columns = ["household_id", "N_adults"]

    household_info = pd.merge(nb_children_under_3, nb_children_under_6, how = "outer", on = "household_id")
    household_info = pd.merge(household_info, nb_children_under_12, how = "outer", on = "household_id")
    household_info = pd.merge(household_info, nb_children_under_18, how = "outer", on = "household_id")
    household_info = pd.merge(household_info, nb_adults, how = "outer", on = "household_id")

    household_composition_columns = ["N_children_under_3", "N_children_under_6", "N_children_under_12", "N_children_under_18", "N_adults"]

    return household_persons, household_info, household_composition_columns
