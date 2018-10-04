import pandas as pd
import numpy as np
import data.constants as c
from tqdm import tqdm
import data.spatial.zones
import data.spatial.countries
import data.spatial.municipalities
import data.spatial.quarters

def configure(context, require):
    require.stage("data.od.raw")
    require.stage("data.spatial.countries")
    require.stage("data.spatial.municipalities")
    require.stage("data.spatial.quarters")
    require.stage("data.spatial.zones")

# TODO: Right now we only produce OD matrices for WORK. We have the information
# from statpop on where the schools are, so we can use this in the future. Also,
# we have commute information for school already prepared (see population.commute).

def execute(context):
    df_od = context.stage("data.od.raw")
    df_zones = context.stage("data.spatial.zones")
    df_countries = context.stage("data.spatial.countries")
    df_municipality_mapping = context.stage("data.spatial.municipalities")[1]
    df_quarters = context.stage("data.spatial.quarters")

    # Find the correct modes
    df_od.loc[:, "mode_numeric"] = df_od.loc[:, "mode"].astype(np.int)
    df_od.loc[df_od["mode_numeric"] == -10, "mode"] = "unknown"
    df_od.loc[df_od["mode_numeric"] == -9, "mode"] = "unknown"
    df_od.loc[df_od["mode_numeric"] == -8, "mode"] = "unknown"
    df_od.loc[df_od["mode_numeric"] == 1, "mode"] = "walk" # walking
    df_od.loc[df_od["mode_numeric"] == 2, "mode"] = "walk" # skateboard
    df_od.loc[df_od["mode_numeric"] == 3, "mode"] = "bike" # bike / elec. bike
    df_od.loc[df_od["mode_numeric"] == 4, "mode"] = "car" # Mofa / Moped / light motor bike
    df_od.loc[df_od["mode_numeric"] == 5, "mode"] = "car" # Car as driver or passenger
    df_od.loc[df_od["mode_numeric"] == 6, "mode"] = "car" # company bus
    df_od.loc[df_od["mode_numeric"] == 7, "mode"] = "pt" # Train
    df_od.loc[df_od["mode_numeric"] == 8, "mode"] = "pt" # Tram / Metro
    df_od.loc[df_od["mode_numeric"] == 9, "mode"] = "pt" # Bus
    df_od.loc[df_od["mode_numeric"] == 10, "mode"] = "other" # Ship, cable car, ...
    del df_od["mode_numeric"]

    # First impute the home zone
    df_od.loc[:, "municipality_id"] = df_od["home_municipality"]
    df_od.loc[:, "quarter_id"] = df_od["home_quarter"]
    df_od = data.spatial.quarters.update_quarter_ids(df_od, df_quarters)
    df_od = data.spatial.municipalities.update_municipality_ids(df_od, df_municipality_mapping)

    df_od = data.spatial.zones.impute(df_od, df_zones)
    df_od.loc[:, "home_zone_id"] = df_od.loc[:, "zone_id"]
    df_od.loc[:, "home_zone_level"] = df_od.loc[:, "zone_level"]

    # Second impute the work zone
    df_od.loc[:, "country_id"] = df_od["work_country"]
    df_od.loc[:, "municipality_id"] = df_od["work_municipality"]
    df_od.loc[:, "quarter_id"] = df_od["work_quarter"]
    df_od = data.spatial.quarters.update_quarter_ids(df_od, df_quarters)
    df_od = data.spatial.municipalities.update_municipality_ids(df_od, df_municipality_mapping)
    df_od = data.spatial.countries.update_country_ids(df_od, df_countries)

    df_od = data.spatial.zones.impute(df_od, df_zones)
    df_od.loc[:, "work_zone_id"] = df_od.loc[:, "zone_id"]
    df_od.loc[:, "work_zone_level"] = df_od.loc[:, "zone_level"]
    del df_od["country_id"]

    # Third impute the education zone (TODO: not used right now)
    #df_od.loc[:, "municipality_id"] = df_od["education_municipality"]
    #df_od.loc[:, "quarter_id"] = df_od["education_quarter"]
    #df_od = data.spatial.quarters.update_quarter_ids(df_od, df_quarters)
    #df_od = data.spatial.municipalities.update_municipality_ids(df_od, df_municipality_mapping)

    #df_od = data.spatial.zones.impute(df_od, df_zones)
    #df_od.loc[:, "education_zone_id"] = df_od.loc[:, "zone_id"]
    #df_od.loc[:, "education_zone_level"] = df_od.loc[:, "zone_level"]

    del df_od["quarter_id"]
    del df_od["municipality_id"]

    # There are some people for which we don't have a valid OD pair
    before_count = len(df_od)
    df_od = df_od[~np.isnan(df_od["home_zone_id"])]
    df_od = df_od[~np.isnan(df_od["work_zone_id"])]

    unknown_count = before_count - len(df_od)
    print("Removed %d (%.2f%%) observations from structural survey for which no work or home location is known" % (unknown_count, 100 * unknown_count / before_count))
    assert(len(df_od) == len(df_od.dropna()))

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
    assert(len(df_od) == len(df_od.dropna()))

    # Filter unknonwn modes
    df_od = df_od[~((df_od["mode"] == "unknown") | (df_od["mode"] == "other"))]

    # Create the matrices
    zone_ids = list(df_zones["zone_id"])

    pdf_matrices = {}
    cdf_matrices = {}

    for mode in ["car", "pt", "bike", "walk"]:
        df_mode_od = df_od[df_od["mode"] == mode]

        matrix = pd.crosstab(
            df_od["home_zone_id"], df_od["work_zone_id"],
            df_od["weight"], aggfunc = sum).reindex(
                index = pd.Index(zone_ids), columns = pd.Index(zone_ids)
            ).fillna(0).values

        # Find the origins which have no observations and make sure we don't divide by zero
        zero_filter = np.sum(matrix, axis = 1) == 0.0
        matrix[zero_filter,:] = 1

        pdf_matrix = matrix / np.sum(matrix, axis = 1)[:, np.newaxis]

        # However, we actually want a NaN here, because later on we will know
        # that something is wrong if we ever want to sample a destination for
        # an origin for which we do not have any observations (this may be
        # municipalities with quarters, or countries for now)
        pdf_matrix[zero_filter,:] = np.nan

        cdf_matrix = np.cumsum(matrix, axis = 1)
        cdf_matrix /= cdf_matrix[:, -1][:, np.newaxis]

        pdf_matrices[mode] = pdf_matrix
        cdf_matrices[mode] = cdf_matrix

    # One final note: The way the OD matrix is measured here is such that the
    # observations that fall into quarters are *not* included in the counts for
    # the municipalities. Let's say there is a municipality which is mostly covered
    # by quarters, but not entirely. Strictly speaking, the count for the municipality
    # is then valid for the area of [municipality area] - [cumulative quarter area].
    # For the origin it makes sense, because we will have a coordinate, we will
    # assign the zone. If we don't hit a quarter, we will hit the municipality, so
    # we use indeed the remaining area. For the destination it is tricky, because
    # if we do not sample a quarter, strictly speaking we should put the destinaton
    # coordinates into the difference area. However, we don't really know how
    # this looks like, so we say that we ignore this fact here.
    # Still, one *could* improve this.

    return pdf_matrices, cdf_matrices
