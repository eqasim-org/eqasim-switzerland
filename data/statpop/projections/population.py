import numpy as np
import pandas as pd

import data.constants as c

CANTON_TO_ID = {"Zürich": 1,
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

        # load excel data
        df = pd.read_csv("%s/projections/population/px-x-0102010000_101.csv" % data_path, sep=";",
                         encoding="latin1", skiprows=1).rename({
            "Kanton (-) / Bezirk (>>) / Gemeinde (......)":"canton_id",
            "Jahr":"year",
            "Staatsangehörigkeit (Kategorie)":"nationality",
            "Geschlecht":"sex",
            "Bevölkerungstyp":"population_type"
        }, axis=1)

        # add up permanent and non-permanent residents
        df = df.groupby(["year", "canton_id", "nationality", "sex"]).sum().reset_index()

        # long format
        df = df.melt(id_vars=["year", "canton_id", "nationality", "sex"], var_name="age", value_name="weight")

        # clean canton names and convert to id
        df["canton_id"] = df["canton_id"].str.split("- ", expand=True)[1]
        df = df.replace(CANTON_TO_ID)

    else:

        # load csv projection data
        df = pd.read_csv("%s/projections/population/px-x-0104020000_101.csv" % data_path, sep=";",
                         encoding="latin1", skiprows=1).rename({
            "Kanton": "canton_id",
            "Staatsangehörigkeit (Kategorie)":"nationality",
            "Geschlecht": "sex",
            "Alter": "age",
            "Jahr": "year",
            "Bevölkerungsstand am 1. Januar": "weight"
        }, axis=1)

        # replace canton names with ids
        df = df.replace(CANTON_TO_ID)


    # turn sex and nationality into an actual 0-based class
    df = df.replace({"Mann": 0, "Frau": 1}).replace({"Schweiz": 0, "Ausland": 1})

    # turn age into integer
    df["age"] = df["age"].str.split("Jahr", expand=True)[0].astype(int)

    # Get the age class
    df["age_class"] = np.digitize(df["age"], c.AGE_CLASS_UPPER_BOUNDS)

    # aggregate by age class
    df = df[["canton_id", "sex", "nationality", "age_class", "year", "weight"]]
    df = df.groupby(["canton_id", "sex", "nationality", "age_class", "year"]).sum().reset_index()

    if (scaling_year in list(df["year"].unique())):
        df = df[df["year"] == scaling_year].drop("year", axis=1).reset_index().drop("index", axis=1)
    else:
        # Pivot years to columns
        df = df.pivot_table(
            index=["canton_id", "sex", "nationality", "age_class"], columns=["year"]
        )

        # Add new interpolated column based on last five years
        df[("weight", scaling_year)] = df.apply(
            lambda x: max(0, np.round(
                np.dot(
                    np.polyfit(
                        [2041, 2042, 2043, 2044, 2045],
                        [x[("weight"), 2041], x[("weight"), 2042], x[("weight"), 2043], x[("weight"), 2044],
                         x[("weight"), 2045]],
                        1
                    ),
                    [scaling_year, 1]
                )
            ))
            , axis=1)

        # Reformat
        df = df[("weight", scaling_year)].reset_index()
        df.columns = ["canton_id", "sex", "nationality", "age_class", "weight"]

    # round weights and convert to integer
    df["weight"] = np.round(df["weight"])
    df["weight"] = df["weight"].astype(int)

    print(df.head())

    return df, scaling_year
