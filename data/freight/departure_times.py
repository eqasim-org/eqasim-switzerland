import pandas as pd


def configure(context, require):
    require.config("raw_data_path")


def execute(context):
    raw_data_path = context.config["raw_data_path"]
    df = pd.read_csv("%s/freight/departure_times.csv" % raw_data_path, sep=";")

    return df