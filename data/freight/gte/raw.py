import pandas as pd

def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")

    df_transport = pd.read_csv("%s/freight/gte/GTE_2017/Donnees/transport.csv" % data_path, sep=";", low_memory=False)
    df_journey = pd.read_csv("%s/freight/gte/GTE_2017/Donnees/journeych.csv" % data_path, sep=";", low_memory=False)
    df_week = pd.read_csv("%s/freight/gte/GTE_2017/Donnees/week.csv" % data_path, sep=";", low_memory=False)

    return df_transport, df_journey, df_week


