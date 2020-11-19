import numpy as np
import pandas as pd

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

CANTON_TO_ID_MULTILANGUAGE = {"Zürich": 1,
                              "Bern / Berne": 2,
                              "Luzern": 3,
                              "Uri": 4,
                              "Schwyz": 5,
                              "Obwalden": 6,
                              "Nidwalden": 7,
                              "Glarus": 8,
                              "Zug": 9,
                              "Fribourg / Freiburg": 10,
                              "Solothurn": 11,
                              "Basel-Stadt": 12,
                              "Basel-Landschaft": 13,
                              "Schaffhausen": 14,
                              "Appenzell Ausserrhoden": 15,
                              "Appenzell Innerrhoden": 16,
                              "St. Gallen": 17,
                              "Graubünden / Grigioni / Grischun": 18,
                              "Aargau": 19,
                              "Thurgau": 20,
                              "Ticino": 21,
                              "Vaud": 22,
                              "Valais / Wallis": 23,
                              "Neuchâtel": 24,
                              "Genève": 25,
                              "Jura": 26}


def configure(context):
    context.config("data_path")
    context.config("scaling_year")


def execute(context):
    data_path = context.config("data_path")

    # Select year in the future to project to
    scaling_year = np.max([c.BASE_SCALING_YEAR, context.config("scaling_year")])

    if scaling_year < c.BASE_PROJECTED_YEAR:

        # Load csv for historical data
        df_households = (pd.read_csv("%s/projections/households/px-x-0102020000_402.csv" % data_path,
                                     sep=";", encoding="latin1", skiprows=1)
                         .rename({'Kanton (-) / Bezirk (>>) / Gemeinde (......)': "canton_id"}, axis=1)
                         )

        # Convert to long format
        df_households = df_households.melt(
            id_vars="canton_id", value_name="weight"
        )

        # Clean up canton names
        df_households = df_households[df_households['canton_id'].str.contains("- ")]
        df_households["canton_id"] = df_households["canton_id"].str.split("- ", expand=True)[1]

        # Split household size and year
        df_households["household_size"] = df_households["variable"].str.split(" ", expand=True)[0].astype(np.int64)
        df_households["year"] = df_households["variable"].str.split("n ", expand=True)[1].astype(np.int64)

        # Select year
        df_households = df_households[df_households["year"] == scaling_year]

        # Reformat
        df_households = df_households[["canton_id", "household_size", "weight"]].reset_index(drop=True)

        # Replace cantons
        df_households = df_households.replace(CANTON_TO_ID_MULTILANGUAGE)

        # TODO: why do we only use five categories?
        # Limit to 5 categories
        df_households["household_size"] = np.minimum(5, df_households["household_size"])
        df_households = df_households.groupby(["canton_id", "household_size"]).sum().reset_index()

    else:

        # Load excel for projections
        df_households = pd.read_excel(
            "%s/projections/households/su-d-01.03.03.03.01.xlsx" % data_path,
            header=[0, 1], skiprows=2, nrows=27, index_col=0).reset_index().rename({
            "index": "canton_id",
            "Total": "total",
            "1 Person": "1",
            "2 Personen": "2",
            "3 und mehr Personen": "3",
            2017: "2017",
            2045: "2045"
        }, axis=1)

        # Flatten multi-index columns
        df_households.columns = ['_'.join(col).strip("_") for col in df_households.columns.values]

        # Convert to long format
        df_households = df_households.melt(
            id_vars="canton_id", value_vars=["1_2017", "1_2045", "2_2017", "2_2045", "3_2017", "3_2045"],
            value_name="weight", var_name=["household_size_year"]
        )

        # split and rename columns
        temp = df_households["household_size_year"].str.split("_", expand=True)
        df_households["household_size"] = temp[0].astype(int)
        df_households["year"] = temp[1].astype(int)
        df_households = df_households[["canton_id", "household_size", "year", "weight"]]

        # Remove Switzerland total
        df_households = df_households[df_households["canton_id"] != "Schweiz"]

        # Pivot years to columns
        df_households = df_households.pivot_table(
            index=["canton_id", "household_size"], columns=["year"]
        )

        # Add new interpolated column
        df_households[("weight", scaling_year)] = df_households.apply(
            lambda x: max(0, 1e3 * np.dot(
                np.polyfit(
                    [2017, 2045],
                    [x["weight", 2017], x["weight", 2045]],
                    1
                ),
                [scaling_year, 1]
            ))
            , axis=1)

        # Reformat
        df_households = df_households[("weight", scaling_year)].reset_index()
        df_households.columns = ["canton_id", "household_size", "weight"]

        # Replace cantons
        df_households = df_households.replace(CANTON_TO_ID)

    # round weights and convert to integer
    df_households["weight"] = np.round(df_households["weight"])
    df_households["weight"] = df_households["weight"].astype(int)

    # make size class zero-based
    df_households = df_households.rename({"household_size": "household_size_class"}, axis=1)
    df_households["household_size_class"] = df_households["household_size_class"] - 1

    # sort values
    df_households = df_households.sort_values(["canton_id", "household_size_class"])

    print(df_households.head())

    return df_households, scaling_year
