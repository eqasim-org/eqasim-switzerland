import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.freight.gte.cleaned")
    context.stage("data.freight.scaling_factor")
    context.config("enable_scaling")


def execute(context):
    df_trips = context.stage("data.freight.gte.cleaned")

    # filter weekdays and trips in CH
    weekdays_to_consider = [1,2,3,4,5]
    df_od = df_trips[df_trips["weekday"].isin(weekdays_to_consider)]

    number_of_weeks = len(df_od["week"].unique())
    number_of_weekdays = len(weekdays_to_consider)

    # create OD matrix per vehicle type
    demands = {}
    origin_pdf_matrices = {}
    od_pdf_matrices = {}

    for vehicle_type in list(df_od["vehicle_type"].unique()):

        df_vehicle_od = df_od[df_od["vehicle_type"] == vehicle_type]

        # create matrix
        matrix = pd.crosstab(
            df_vehicle_od["origin_nuts_id"], df_vehicle_od["destination_nuts_id"],
            df_vehicle_od["weight"], aggfunc=sum, dropna=False).fillna(0)
        matrix_values = matrix.values

        # compute demand
        demands[vehicle_type] =  int(np.round(np.sum(matrix_values) / number_of_weeks / number_of_weekdays))

        # scale demand
        if context.config("enable_scaling"):
            demands[vehicle_type] *= context.stage("data.freight.scaling_factor")

        # make sure each from sums up to one
        f_zero = np.sum(matrix_values, axis = 1) == 0.0
        for index in np.where(f_zero)[0]:
            matrix_values[index,:] = 0.0
            matrix_values[index,index] = 1.0

        # compute pdfs
        origin_pdf_matrix = np.sum(matrix_values, axis=1) / np.sum(matrix_values)
        od_pdf_matrix = matrix_values / np.sum(matrix_values, axis=1)[:, np.newaxis]

        origin_pdf_matrices[vehicle_type] = pd.DataFrame(index=list(matrix.index), columns=["probability"], data=origin_pdf_matrix)
        od_pdf_matrices[vehicle_type] = pd.DataFrame(index=matrix.index,
                                  columns=matrix.columns,
                                  data=od_pdf_matrix)

    return demands, origin_pdf_matrices, od_pdf_matrices
