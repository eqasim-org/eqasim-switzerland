import numpy as np
import pandas as pd


def configure(context):
    context.stage("synthesis.freight.gte.trips")
    context.stage("synthesis.freight.gqgv.trips")
    context.stage("data.freight.departure_times")

def execute(context):
    df_gte_trips = context.stage("synthesis.freight.gte.trips")
    df_gqgv_trips = context.stage("synthesis.freight.gqgv.trips")
    df_departure_times = context.stage("data.freight.departure_times")

    df_trips = pd.concat([df_gte_trips, df_gqgv_trips], ignore_index=True)

    # generate hour from departure time distribution
    departure_times = 3600 * np.random.choice(df_departure_times["hour"].values,
                                       len(df_trips),
                                       p=df_departure_times["probability"].values)

    # add seconds with hour randomly from uniform distribution
    departure_times += (3600 * np.random.rand(len(df_trips))).astype(int)

    # add departure times to trips
    df_trips["departure_time"] = departure_times

    # add agent ids
    df_trips["agent_id"] = np.arange(len(df_trips))

    # package up
    df_trips = df_trips[["agent_id",
                         "origin_x", "origin_y",
                         "destination_x", "destination_y",
                         "departure_time",
                         "vehicle_type"]
    ]

    return df_trips
