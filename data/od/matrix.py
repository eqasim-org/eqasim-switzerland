import pandas as pd
import numpy as np
import data.constants as c

def configure(context, require):
    require.stage("data.od.raw")

def execute(context):
    df_od = context.stage("data.od.raw")

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

    df_od.loc[df_od["home_quarter"] > 0, "home"] = df_od[df_od["home_quarter"] > 0]["home_quarter"]
    df_od.loc[df_od["home_quarter"] > 0, "home_type"] = "quarter"

    df_od.loc[:, "work"] = df_od["work_country"]
    df_od.loc[:, "work_type"] = "country"

    df_od.loc[df_od["work_municipality"] > 0, "work"] = df_od[df_od["work_municipality"] > 0]["work_municipality"]
    df_od.loc[df_od["work_municipality"] > 0, "work_type"] = "municipality"

    df_od.loc[df_od["work_quarter"] > 0, "work"] = df_od[df_od["work_quarter"] > 0]["work_quarter"]
    df_od.loc[df_od["work_quarter"] > 0, "work_type"] = "quarter"

    df_od["work_type"] = df_od["work_type"].astype("category")
    df_od["home_type"] = df_od["home_type"].astype("category")
    df_od["mode"] = df_od["mode"].astype("category")

    df_od = df_od[["home", "home_type", "work", "work_type", "mode", "weight"]]
    assert(len(df_od) == len(df_od.dropna()))

    # Filter unknonwn information
    df_od = df_od[
        (df_od["home"] > 0) &
        (df_od["work"] > 0) &
        ~((df_od["mode"] == "unknown") | (df_od["mode"] == "other"))
    ]

    # Find information about the zones
    df_home_zones = df_od[["home", "home_type"]].drop_duplicates("home")
    df_home_zones.columns = ["zone", "type"]
    df_work_zones = df_od[["work", "work_type"]].drop_duplicates("work")
    df_work_zones.columns = ["zone", "type"]
    df_zones = pd.concat([df_home_zones, df_work_zones]).drop_duplicates("zone")

    # Create the matrices
    unique_zone_ids = df_zones["zone"].values

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
        matrix = matrix[zero_filter,:][:, zero_filter]

        pdf_matrix = matrix / np.sum(matrix, axis = 1)[:, np.newaxis]

        cdf_matrix = np.cumsum(matrix, axis = 1)
        cdf_matrix /= cdf_matrix[:, -1][:, np.newaxis]

        pdf_matrices[mode] = pdf_matrix
        cdf_matrices[mode] = cdf_matrix

    return pdf_matrices, cdf_matrices, df_zones
