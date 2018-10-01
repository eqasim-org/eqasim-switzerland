import pandas as pd
import numpy as np

def configure(context, require):
    require.config("raw_data_path")

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    df = pd.read_excel(
        "%s/country_codes_2018.xlsx" % raw_data_path
    )

    df["country_id"] = df["Ländercode BFS\nCode des pays OFS\nCodice del paese UST"]
    df["country_name"] = df["EN Short form"]
    df = df[["country_id", "country_name"]]

    return df















#
