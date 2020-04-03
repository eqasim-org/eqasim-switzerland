import pandas as pd


def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")
    df = pd.read_csv("%s/freight/departure_times.csv" % data_path, sep=";")

    return df