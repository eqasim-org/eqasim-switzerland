import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.spatial.zones")

# TODO: Right now we only produce OD matrices for WORK. We have the information
# from statpop on where the schools are, so we can use this in the future. Also,
# we have commute information for school already prepared (see population.commute).

def execute(context):
    df_zones = context.stage("data.spatial.zones")
    df_od = context.stage("data.structural_survey.structural_survey")[[
        "home_municipality_id", "home_quarter_id", "home_zone_id", "home_zone_level",
        "work_municipality_id", "work_quarter_id", "work_zone_id", "work_zone_level",
        "mode", "weight"
    ]]

    # There are some people for which we don't have a valid OD pair
    before_count = len(df_od)
    df_od = df_od[~np.isnan(df_od["home_zone_id"])]
    df_od = df_od[~np.isnan(df_od["work_zone_id"])]

    unknown_count = before_count - len(df_od)
    print("Removed %d (%.2f%%) observations from structural survey for which no work or home location is known" % (unknown_count, 100 * unknown_count / before_count))
    #assert(len(df_od) == len(df_od.dropna())) Commented this, because home_quarter_id may be NaN deliberately

    # Filter out people who are not working in a neighboring country
    # TODO: Eventually, we want to have commuters back in the population!
    # But this involves adjustments at several points:
    # - We want them to get activity chains for commuters
    # - We want them to have consistent work / education locations at the border
    #   at the right crossing.
    before_count = len(df_od)
    df_od = df_od[~(df_od["work_zone_level"] == "country")]
    df_od = df_od[~(df_od["home_zone_level"] == "country")]

    outside_count = before_count - len(df_od)
    print("Removed %d (%.2f%%) observations from structural survey which live or work abroad (TODO: eventually we want them back in!)" % (outside_count, 100 * outside_count / before_count))
    #assert(len(df_od) == len(df_od.dropna())) Commented this, because home_quarter_id may be NaN deliberately

    # Filter unknonwn modes
    before_count = len(df_od)
    df_od = df_od[~((df_od["mode"] == "unknown") | (df_od["mode"] == "other"))]
    unknown_mode_count = before_count - len(df_od)
    print("Removed %d (%.2f%%) observations from structural survey with unknown mode" % (unknown_mode_count, 100 * unknown_mode_count / before_count))

    # Create the matrices
    zone_ids = list(df_zones["zone_id"])
    municipality_ids = list(df_zones[df_zones["zone_level"] == "municipality"]["zone_level_id"])
    quarter_ids = list(df_zones[df_zones["zone_level"] == "quarter"]["zone_level_id"])

    pdf_matrices = {}
    cdf_matrices = {}

    for mode in ["car", "pt", "bike", "walk"]:
        df_mode_od = df_od[df_od["mode"] == mode]

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

        # There are two types of origins with zero observations:
        # - Quarters or municipalities for which we simply don't have data
        # - Countries which we do not want to handle for now
        #
        # The latter ones will simply be set to NaN after avoiding a division
        # by zero error. The former ones will be set for now in a way that all
        # people stay inside this zone. (TODO: Probably it would be better to
        # attach them to adjacent zones.)

        for index in np.where(f_origin & f_zero)[0]:
            matrix[index,:] = 0.0
            matrix[index,index] = 1.0

        matrix[~f_origin & f_zero] = 1.0

        pdf_matrix = matrix / np.sum(matrix, axis = 1)[:, np.newaxis]
        pdf_matrix[~f_origin & f_zero,:] = np.nan

        cdf_matrix = np.cumsum(matrix, axis = 1)
        cdf_matrix /= cdf_matrix[:, -1][:, np.newaxis]

        pdf_matrices[mode] = pdf_matrix
        cdf_matrices[mode] = cdf_matrix

        print("  - Finished %s (%d fixed municipalities)" % (mode, np.count_nonzero(f_origin & f_zero)))

    # A final note on the structure of these OD matrices:
    # - The origin counts for municipalities contain all originating trips, also
    #   those which are actually assigned to quarters within this zone
    # - The destination counts target the assigned top-level zone. So arrivals
    #   in a quarter are NOT included in the arrivals for the municipality. This
    #   way, arrivals in municipalities with quarters can only happen if the
    #   municipality is not covered 1:1 by the quarters, which is usually the case
    #   in our zoning system. This way the municipality itself will only have little
    #   arrivals, while the quarters will have more.

    return pdf_matrices, cdf_matrices
