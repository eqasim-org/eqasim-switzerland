import numpy as np
import pandas as pd
from data.structural_survey.structural_survey import get_filtered_data
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.spatial.zones")


def execute(context):
    df_zones = context.stage("data.spatial.zones")
    df_od = get_filtered_data(context, "moving")[[
        "home_municipality_id", "home_quarter_id", "home_zone_id", "home_zone_level",
        "work_municipality_id", "work_quarter_id", "work_zone_id", "work_zone_level",
        "mode", "weight", "crowfly_distance_to_work", "start_work"
    ]]

    # Create the matrices
    zone_ids = list(df_zones["zone_id"])
    municipality_ids = list(df_zones[df_zones["zone_level"] == "municipality"]["zone_level_id"])
    quarter_ids = list(df_zones[df_zones["zone_level"] == "quarter"]["zone_level_id"])

    # here we had a loop over modes before
    municipality_matrix = pd.crosstab(
        df_od["home_municipality_id"], df_od["work_zone_id"],
        df_od["weight"], aggfunc = sum).reindex(
            index = pd.Index(municipality_ids, name = "municipality_id"), columns = pd.Index(zone_ids, name = "destination_zone_id")
        ).fillna(0).reset_index()

    quarter_matrix = pd.crosstab(
        df_od["home_quarter_id"], df_od["work_zone_id"],
        df_od["weight"], aggfunc = sum).reindex(
            index = pd.Index(quarter_ids, name = "quarter_id"), columns = pd.Index(zone_ids, name = "destination_zone_id")
        ).fillna(0).reset_index()

    municipality_matrix = pd.merge(
        municipality_matrix, df_zones[df_zones["zone_level"] == "municipality"],
        left_on = "municipality_id", right_on = "zone_level_id"
    )
    del municipality_matrix["municipality_id"]

    quarter_matrix = pd.merge(
        quarter_matrix, df_zones[df_zones["zone_level"] == "quarter"],
        left_on = "quarter_id", right_on = "zone_level_id"
    )
    del quarter_matrix["quarter_id"]

    matrix = pd.concat((municipality_matrix, quarter_matrix))
    for column in ("zone_name", "zone_level", "zone_level_id"): del matrix[column]

    matrix = matrix.set_index("zone_id")
    matrix = matrix.reindex(index = pd.Index(zone_ids))
    matrix = matrix.values

    f_origin = df_zones["zone_level"].isin(("municipality", "quarter"))
    f_zero = np.sum(matrix, axis = 1) == 0.0

    for index in np.where(f_origin & f_zero)[0]:
        matrix[index,:] = 0.0
        matrix[index,index] = 1.0

    matrix[~f_origin & f_zero] = 1.0

    pdf_matrix = matrix / np.sum(matrix, axis = 1)[:, np.newaxis]
    pdf_matrix[~f_origin & f_zero,:] = np.nan

    cdf_matrix = np.cumsum(matrix, axis = 1)
    cdf_matrix /= cdf_matrix[:, -1][:, np.newaxis]

    logger.info("  - Finished (%d fixed municipalities)" % (np.count_nonzero(f_origin & f_zero)))

    return pdf_matrix, cdf_matrix
