import numpy as np
import pandas as pd


def configure(context):
    if context.config("use_npvm_freight", default=True):
        context.stage("synthesis.freight.npvm.trips")
    else:
        context.stage("synthesis.freight.gte.trips")
        context.stage("synthesis.freight.gqgv.trips")
        context.stage("data.freight.departure_times")
        context.config("random_seed")


def execute(context):

    # Return NPVM demand directly when the default freight source is selected.

    if context.config("use_npvm_freight"):
        return context.stage("synthesis.freight.npvm.trips")

    # Load and combine the original GTE and GQGV freight trips.

    df_gte_trips = context.stage("synthesis.freight.gte.trips")
    df_gqgv_trips = context.stage("synthesis.freight.gqgv.trips")
    df_departure_times = context.stage("data.freight.departure_times")

    df_trips = pd.concat([df_gte_trips, df_gqgv_trips], ignore_index=True)
    
    # Preserve the original seeded departure-time sampling for the survey method.

    rng = np.random.RandomState(context.config("random_seed"))

    departure_times = 3600 * rng.choice(df_departure_times["hour"].values,
                                       len(df_trips),
                                       p=df_departure_times["probability"].values)

    departure_times += rng.uniform(low=0.0, high=3600.0, size=len(df_trips)).astype(int)

    # Assign departure times and stable consecutive agent identifiers.

    df_trips["departure_time"] = departure_times

    df_trips["agent_id"] = np.arange(len(df_trips))

    # Return the shared seven-column freight trip contract.

    df_trips = df_trips[["agent_id",
                         "origin_x", "origin_y",
                         "destination_x", "destination_y",
                         "departure_time",
                         "vehicle_type"]
    ]

    return df_trips
