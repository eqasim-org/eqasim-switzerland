import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.spatial.zones")


def execute(context):
    df_zones = context.stage("data.spatial.zones")
    df_od = context.stage("data.structural_survey.structural_survey")[[
        "home_municipality_id", "home_quarter_id", "home_zone_id", "home_zone_level",
        "education_municipality_id", "education_quarter_id", "education_zone_id", "education_zone_level",
        "mode", "weight"
    ]]

    # There are many people for which we don't have a valid OD pair
    before_count = len(df_od)
    df_od = df_od[~np.isnan(df_od["home_zone_id"])]
    df_od = df_od[~np.isnan(df_od["education_zone_id"])]

    unknown_count = before_count - len(df_od)
    logger.info("Removed %d (%.2f%%) observations from structural survey for which no education or home location is known" % (unknown_count, 100 * unknown_count / before_count))

    # Filter out people who are not in a neighboring country (currently, don't exist any)
    before_count = len(df_od)
    df_od = df_od[~(df_od["education_zone_level"] == "country")]
    df_od = df_od[~(df_od["home_zone_level"] == "country")]

    outside_count = before_count - len(df_od)
    logger.info("Removed %d (%.2f%%) observations from structural survey which live or study abroad" % (outside_count, 100 * outside_count / before_count))

    #######################################################
    # Create the matrices
    zone_ids = list(df_zones["zone_id"])
    municipality_ids = list(df_zones[df_zones["zone_level"] == "municipality"]["zone_level_id"])
    quarter_ids = list(df_zones[df_zones["zone_level"] == "quarter"]["zone_level_id"])

    municipality_matrix = pd.crosstab(
        df_od["home_municipality_id"], df_od["education_zone_id"],
        df_od["weight"], aggfunc=sum).reindex(
            index=pd.Index(municipality_ids, name="municipality_id"), columns=pd.Index(zone_ids, name="destination_zone_id")
        ).fillna(0).reset_index()

    quarter_matrix = pd.crosstab(
        df_od["home_quarter_id"], df_od["education_zone_id"],
        df_od["weight"], aggfunc=sum).reindex(
            index=pd.Index(quarter_ids, name="quarter_id"), columns=pd.Index(zone_ids, name="destination_zone_id")
        ).fillna(0).reset_index()

    municipality_matrix = pd.merge(
        municipality_matrix, df_zones[df_zones["zone_level"] == "municipality"],
        left_on="municipality_id", right_on="zone_level_id"
    )
    del municipality_matrix["municipality_id"]

    quarter_matrix = pd.merge(
        quarter_matrix, df_zones[df_zones["zone_level"] == "quarter"],
        left_on="quarter_id", right_on="zone_level_id"
    )
    del quarter_matrix["quarter_id"]

    matrix = pd.concat((municipality_matrix, quarter_matrix))
    for column in ("zone_name", "zone_level", "zone_level_id"): del matrix[column]

    matrix = matrix.set_index("zone_id")
    matrix = matrix.reindex(index=pd.Index(zone_ids))
    matrix = matrix.values

    f_origin = df_zones["zone_level"].isin(("municipality", "quarter"))
    f_zero = np.sum(matrix, axis=1) == 0.0

    # Handle origins (quarters or municipalities) with zero observed trips:
    # - These are zones for which no survey respondents have education locations, resulting in zero rows in the matrix.
    # - To avoid issues with downstream processing, we assign all individuals from these zones to stay within their own zone.
    # - Note: This is a placeholder solution; we should be aware that these zones lack empirical data and results may be biased.
    logger.info("Fixing %d municipalities/quarters with zero education destinations", np.count_nonzero(f_origin & f_zero))
    for index in np.where(f_origin & f_zero)[0]:
        matrix[index,:] = 0.0
        matrix[index,index] = 1.0

    pdf_matrix = matrix / np.sum(matrix, axis=1)[:, np.newaxis]

    cdf_matrix = np.cumsum(matrix, axis=1)
    cdf_matrix /= cdf_matrix[:, -1][:, np.newaxis]

    logger.info("Finished %d fixed municipalities", np.count_nonzero(f_origin & f_zero))

    return pdf_matrix, cdf_matrix
