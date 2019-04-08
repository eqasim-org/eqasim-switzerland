import pandas as pd
import numpy as np
import data.constants as c

CANTON_TO_ID = {
    "Zürich": 1,
    "Bern": 2,
    "Luzern": 3,
    "Uri": 4,
    "Schwyz": 5,
    "Obwalden": 6,
    "Nidwalden": 7,
    "Glarus": 8,
    "Zug": 9,
    "Freiburg": 10,
    "Solothurn": 11,
    "Basel-Stadt": 12,
    "Basel-Landschaft": 13,
    "Schaffhausen": 14,
    "Appenzell A.Rh.": 15,
    "Appenzell I.Rh.": 16,
    "St. Gallen": 17,
    "Graubünden": 18,
    "Aargau": 19,
    "Thurgau": 20,
    "Tessin": 21,
    "Waadt": 22,
    "Wallis": 23,
    "Neuenburg": 24,
    "Genf": 25,
    "Jura": 26
}

def configure(context, require):
    require.config("raw_data_path")
    require.config("scaling_year")

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    # Load excel data
    df_households = pd.read_excel(
        "%s/projections/households/su-d-01.03.03.03.01.xlsx" % raw_data_path,
        header=[0,1], skiprows = 2, nrows = 27, index_col = 0).reset_index().rename({
            "index": "canton_id",
            "Total": "total",
            "1 Person": 1,
            "2 Personen": 2,
            "3 und mehr Personen": 3
        }, axis = 1)

    # Convert to long format
    df_households = df_households.melt(
        id_vars = "canton_id", value_vars = [1, 2, 3],
        value_name = "weight", var_name = ["household_size", "year"]
    )

    # Remove Switzerland total
    df_households = df_households[df_households["canton_id"] != "Schweiz"]

    # Pivot years to columns
    df_households = df_households.pivot_table(
        index = ["canton_id", "household_size"], columns = ["year"]
    )

    # Select year in the future to project to
    scaling_year = np.max([c.BASE_YEAR, context.config["scaling_year"]])

    # Add new interpolated column
    df_households[("weight", scaling_year)] = df_households.apply(
        lambda x: max(0, 1e3 * np.dot(
            np.polyfit(
                [2017, 2045],
                [x[("weight"), 2017], x[("weight"), 2045]],
                1
            ),
            [scaling_year, 1]
        ))
    , axis = 1)

    # Reformat
    df_households = df_households[("weight", scaling_year)].reset_index()
    df_households.columns = ["canton_id", "household_size", "weight"]

    # Make zero-based
    df_households["household_size"] -= 1
    df_households = df_households.rename({"household_size": "household_size_class_projection"}, axis = 1)

    # Replace cantons
    df_households = df_households.replace(CANTON_TO_ID)

    return df_households
