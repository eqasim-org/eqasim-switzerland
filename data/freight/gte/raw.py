import pandas as pd

def configure(context):
    context.config("raw_data_path")

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    df_transport = pd.read_csv("%s/freight/gte/GTE_2017/Donnees/transport.csv" % raw_data_path, sep=";", low_memory=False)
    df_journey = pd.read_csv("%s/freight/gte/GTE_2017/Donnees/journeych.csv" % raw_data_path, sep=";", low_memory=False)
    df_week = pd.read_csv("%s/freight/gte/GTE_2017/Donnees/week.csv" % raw_data_path, sep=";", low_memory=False)

    return df_transport, df_journey, df_week


