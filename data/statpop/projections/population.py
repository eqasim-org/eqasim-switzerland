import numpy as np
import pandas as pd
import logging
logger = logging.getLogger("synpp")

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
    context.config("enable_scaling")
    context.stage("data.constants")

def execute(context):
    if not context.config("enable_scaling"):
        logger.info("Skipping projecting population as scaling is disabled!")
        return
    data_path = context.config("data_path")
    c         = context.stage("data.constants")

    # Select year in the future to project to
    scaling_year = np.max([c.BASE_SCALING_YEAR, context.config("scaling_year")])

    # load csv projection data
    df = pd.read_csv("%s/projections/population/px-x-0104020000_101_20250808-151932.csv" % data_path, sep=";",
                        encoding="latin1", skiprows=0).rename({
        "Kanton": "canton_id",
        "Staatsangehörigkeit (Kategorie)":"nationality",
        "Geschlecht": "sex",
        "Alter": "age",
        "Jahr": "year",
        "Bevölkerungsstand am 1. Januar": "weight"
    }, axis=1)
    # Ensure scaling_year is valid
    if scaling_year not in df["year"].unique():
        raise ValueError(
            f"scaling_year {scaling_year} is not present in the 'year' column. "
            f"Available years: {sorted(df['year'].unique())}"
        )
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

    df = df[df["year"] == scaling_year].drop("year", axis=1).reset_index().drop("index", axis=1)

    # round weights and convert to integer
    df["weight"] = np.round(df["weight"])
    df["weight"] = df["weight"].astype(int)
    return df, scaling_year
