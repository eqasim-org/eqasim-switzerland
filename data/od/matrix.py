import pandas as pd
import numpy as np
import data.constants as c
from tqdm import tqdm

def configure(context, require):
    require.stage("data.od.raw")
    require.stage("data.od.municipality_shapes")
    require.stage("data.od.quarter_shapes")

def execute(context):
    df_od = context.stage("data.od.raw")
    df_municipalities, df_municipality_mapping = context.stage("data.od.municipality_shapes")
    df_quarters = context.stage("data.od.quarter_shapes")

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

    # Find the lowest level of aggregation for home and work:
    # - Municipalities have the format {MUN_ID} with MUN_ID being 5 or 6 digits
    # - Quarters have the format {MUN_ID,QUARTER_ID} with QUARTER_ID being max. 3 digits
    # - Countries have the format {COUNTRY_ID} starting at around 8000, while the
    #   municipalities have IDs less than 8000
    # So just by setting the lowest level of aggregation as the "actual" ID does
    #   work and we do not introduce any collisions.
    df_od.loc[:, "home"] = df_od["home_municipality"]
    df_od.loc[:, "home_type"] = "municipality"

    f = df_od["home_quarter"].isin(df_quarters["zone"]) # Make sure we have a shape!
    df_od.loc[f, "home"] = df_od[f]["home_quarter"]
    df_od.loc[f, "home_type"] = "quarter"

    df_od.loc[:, "work"] = df_od["work_country"]
    df_od.loc[:, "work_type"] = "country"

    df_od.loc[df_od["work_municipality"] > 0, "work"] = df_od[df_od["work_municipality"] > 0]["work_municipality"]
    df_od.loc[df_od["work_municipality"] > 0, "work_type"] = "municipality"

    f = df_od["work_quarter"].isin(df_quarters["zone"]) # Make sure we have a shape!
    df_od.loc[f, "work"] = df_od[f]["work_quarter"]
    df_od.loc[f, "work_type"] = "quarter"

    # We know now that every observation that *is* on the quarter level *has*
    # a valid quarter. However, the municipalities don't need to exist in 2018.
    # However, we have a mapping of old quarters to new ones from the previous
    # stage. We can apply this here to remove invalid municipalities.

    f_home = df_od["home_type"] == "municipality"
    f_work = df_od["work_type"] == "municipality"

    df_changed = df_municipality_mapping[df_municipality_mapping["zone"] != df_municipality_mapping["zone_previously"]]
    for _, row in tqdm(df_changed.iterrows(), total = len(df_changed), desc = "Rewriting municipality ids"):
        updated_id, previous_id = row["zone"], row["zone_previously"]
        df_od.loc[f_home & (df_od["home"] == previous_id), "home"] = updated_id
        df_od.loc[f_work & (df_od["work"] == previous_id), "work"] = updated_id

    # There are people about which we do not know anything. This is when there aggregation
    # level is "country", but they don't have a country id.

    before_count = len(df_od)

    df_od = df_od[~((df_od["home_type"] == "country") & (df_od["home"] <= 0))]
    df_od = df_od[~((df_od["work_type"] == "country") & (df_od["work"] <= 0))]

    unknown_count = before_count - len(df_od)
    print("Removed %d (%.2f%%) observations for which no work location is known" % (unknown_count, 100 * unknown_count / before_count))

    # I'm leaving this piece of code here which allows to plot where those people
    # with unknown work places are. They are distributed quite uniformly over Switzerland.
    # At first I was skeptical, but of course those are all the students, unemployed
    # people and retired.
    #
    #    pd.merge(df_municipalities, df_od[
    #        (df_od["work_type"] == "country") &
    #        (df_od["work"] <= 0) &
    #        (df_od["home_type"] == "municipality")
    #    ].groupby("home").size().reset_index(name = "count"),
    #    left_on = "zone", right_on = "home").to_file("/home/sebastian/output.shp")
    #

    # Now we should only have valid ids
    assert(np.all(df_od[df_od["home_type"] == "municipality"]["home"].isin(df_municipalities["zone"])))
    assert(np.all(df_od[df_od["work_type"] == "municipality"]["work"].isin(df_municipalities["zone"])))
    assert(np.all(df_od[df_od["home_type"] == "quarter"]["home"].isin(df_quarters["zone"])))
    assert(np.all(df_od[df_od["work_type"] == "quarter"]["work"].isin(df_quarters["zone"])))
    assert(np.all(df_od[df_od["home_type"] == "country"]["home"] > 0))
    assert(np.all(df_od[df_od["work_type"] == "country"]["work"] > 0))

    df_od["work_type"] = df_od["work_type"].astype("category")
    df_od["home_type"] = df_od["home_type"].astype("category")
    df_od["mode"] = df_od["mode"].astype("category")

    df_od = df_od[["home", "home_type", "work", "work_type", "mode", "weight"]]
    assert(len(df_od) == len(df_od.dropna()))

    # Filter unknonwn modes
    df_od = df_od[~((df_od["mode"] == "unknown") | (df_od["mode"] == "other"))]

    # Create the matrices
    unique_zone_ids = set(np.unique(df_od["home"]))
    unique_zone_ids |= set(np.unique(df_od["home"]))
    unique_zone_ids = list(unique_zone_ids)

    pdf_matrices = {}
    cdf_matrices = {}

    for mode in ["car", "pt", "bike", "walk"]:
        df_mode_od = df_od[df_od["mode"] == mode]

        matrix = pd.crosstab(
            df_od["home"], df_od["work"],
            df_od["weight"], aggfunc = sum).reindex(
                index = pd.Index(unique_zone_ids), columns = pd.Index(unique_zone_ids)
            ).fillna(0).values

        zero_filter = np.sum(matrix, axis = 1) > 0.0
        matrix = matrix[zero_filter,:]

        pdf_matrix = matrix / np.sum(matrix, axis = 1)[:, np.newaxis]

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

    return pdf_matrices, cdf_matrices, unique_zone_ids
