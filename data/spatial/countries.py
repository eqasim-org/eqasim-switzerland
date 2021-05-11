import numpy as np
import pandas as pd


def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")

    df = pd.read_excel(
        "%s/country_codes_2018.xlsx" % data_path
    )

    df["country_id"] = df["Ländercode BFS\nCode des pays OFS\nCodice del paese UST"]
    df["country_name"] = df["EN Short form"]
    df = df[["country_id", "country_name"]]

    return df

def update_country_ids(df, df_countries, remove_unknown = False):
    assert("country_id" in df.columns)

    df["deprecated_country_id"] = df["country_id"]
    del df["country_id"]

    df_join = pd.merge(
        df[["deprecated_country_id"]], df_countries,
        left_on = "deprecated_country_id", right_on = "country_id", how = "left"
    )

    df.loc[:, "country_id"] = df_join.loc[:, "country_id"].values

    if remove_unknown:
        return df[~np.isnan(df["country_id"])]
    else:
        return df













#
