import numpy as np
import pandas as pd
import logging
logger = logging.getLogger("synpp")

# Use completed ages at year end, as in the STATPOP source. January stocks in
# this BFS table use age attained during the coming year and cannot be grouped
# directly with STATPOP ages (their age-zero count is structurally zero).
PROJECTION_FILE = "projections/population/population_year_end_2025.csv"
STOCK_COLUMN = "Bevölkerungsstand am 31. Dezember"

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
    scaling_year = context.config("scaling_year")

    # load csv projection data
    df = pd.read_csv("%s/%s" % (data_path, PROJECTION_FILE), sep=";", encoding="latin1")
    return prepare_controls(df, scaling_year, c.AGE_CLASS_UPPER_BOUNDS), scaling_year


def prepare_controls(df, scaling_year, age_bounds):
    """Read the year-end reference extract without silently accepting January ages."""
    if STOCK_COLUMN not in df:
        raise ValueError("Population projections require '%s'; re-export year-end stocks from BFS" % STOCK_COLUMN)
    df = df.rename(columns={
        "Kanton": "canton_id",
        "Staatsangehörigkeit (Kategorie)":"nationality",
        "Geschlecht": "sex",
        "Alter": "age",
        "Jahr": "year",
        STOCK_COLUMN: "weight"
    }).copy()
    df["year"] = pd.to_numeric(df["year"], errors="raise")
    # Ensure scaling_year is valid
    if scaling_year not in df["year"].unique():
        raise ValueError(
            f"scaling_year {scaling_year} is not present in the 'year' column. "
            f"Available years: {sorted(df['year'].unique())}"
        )
    df = df.loc[df["year"].eq(scaling_year)].copy()
    # Column-specific mappings avoid confusing Swiss nationality with geography.
    df["canton_id"] = df["canton_id"].map(CANTON_TO_ID)
    df["sex"] = df["sex"].map({"Mann": 0, "Frau": 1})
    df["nationality"] = df["nationality"].map({"Schweiz": 0, "Ausland": 1})
    if df[["canton_id", "sex", "nationality"]].isna().any().any():
        raise ValueError("Projection extract contains unknown categories or total rows")

    # '100 Jahre oder mehr' remains in the final, open-ended age class.
    df["age"] = df["age"].str.split("Jahr", expand=True)[0].astype(int)
    keys = ["canton_id", "sex", "nationality", "age"]
    if df.duplicated(keys).any():
        raise ValueError("Population projections contain duplicate single-age cells")
    df["weight"] = pd.to_numeric(df["weight"], errors="raise")
    if not np.isfinite(df["weight"]).all() or (df["weight"] < 0).any():
        raise ValueError("Population projection weights must be finite and non-negative")
    expected = pd.MultiIndex.from_product([range(1, 27), range(2), range(2), range(101)], names=keys)
    if not pd.MultiIndex.from_frame(df[keys]).sort_values().equals(expected.sort_values()):
        raise ValueError("Population projections require all 26 cantons, both sexes/nationalities and ages 0–100+")

    # Get the age class
    df["age_class"] = np.digitize(df["age"], age_bounds)

    # aggregate by age class
    df = df[["canton_id", "sex", "nationality", "age_class", "year", "weight"]]
    df = df.groupby(["canton_id", "sex", "nationality", "age_class", "year"]).sum().reset_index()

    df = df[df["year"] == scaling_year].drop("year", axis=1).reset_index().drop("index", axis=1)

    # round weights and convert to integer
    df["weight"] = np.round(df["weight"])
    df["weight"] = df["weight"].astype(int)
    return df
