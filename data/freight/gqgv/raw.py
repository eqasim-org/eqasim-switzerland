import pandas as pd

def configure(context, require):
    require.config("raw_data_path")

def execute(context):
    raw_data_path = context.config["raw_data_path"]
    df = pd.read_csv("%s/freight/gqgv/GQGV_2014/GQGV_2014_Mikrodaten.csv" % raw_data_path, sep=";")

    return df


