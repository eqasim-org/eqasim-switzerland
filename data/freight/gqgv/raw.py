import pandas as pd

def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")
    df = pd.read_csv("%s/freight/gqgv/GQGV_2014/GQGV_2014_Mikrodaten.csv" % data_path, sep=";")

    return df


