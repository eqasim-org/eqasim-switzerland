import numpy as np


def configure(context):
    context.stage("data.freight.gte.cleaned")
    context.stage("data.freight.gqgv.cleaned")
    context.stage("data.freight.projections")


def execute(context):
    df_gte_legs = context.stage("data.freight.gte.cleaned")
    df_gqgv_legs = context.stage("data.freight.gqgv.cleaned")
    df_projections = context.stage("data.freight.projections")

    # compute total yearly distance from trips data
    total_distance = np.sum(df_gte_legs["distance_km"] * df_gte_legs["weight"]) \
                     + np.sum(df_gqgv_legs["distance_km"] * df_gqgv_legs["weight"])

    # get projected yearly distance
    projected_distance = df_projections[df_projections["type"] == "truck"]["vehicle_km"].values[0]

    # compute scaling factor
    scaling_factor = projected_distance / total_distance

    return scaling_factor
