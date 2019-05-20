import pandas as pd
import numpy as np

def configure(context, require):
    require.stage("freight.gte.trips")
    require.stage("freight.gqgv.trips")

def execute(context):
    df_gte_trips = context.stage("freight.gte.trips")
    df_gqgv_trips = context.stage("freight.gqgv.trips")

    df_trips = pd.concat([df_gte_trips, df_gqgv_trips], ignore_index=True)

    # reset agent ids
    df_trips["agent_id"] = np.arange(len(df_trips))

    return df_trips
