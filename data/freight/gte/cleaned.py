import pandas as pd

RENAMES = {"ernr":"agent_id",
           "journeyId":"journey_id",
           "vehicleKind":"vehicle_type",
           "grossingFactor":"weight"
           }

FIELDS = ["week", "weekday",
          "origin_nuts_id","destination_nuts_id",
          "origin_postal_code", "destination_postal_code",
          "origin_country", "destination_country",
          "vehicle_type", "distance_km", "weight"
          ]


# VEHICLE_TYPES = {
#         35:"truck",
#         37:"semi-trailer truck",
#         38:"tractor unit"
#     }

VEHICLE_TYPES = {
    35:"truck",
    37:"truck",
    38:"truck"
}


def configure(context):
    context.stage("data.freight.gte.raw")
    context.stage("data.spatial.nuts")


def execute(context):
    df_transport, df_journey, df_week = context.stage("data.freight.gte.raw")

    df_merge = pd.merge(df_transport, df_journey, on=["ernr", "journeyId"], suffixes=("_leg", "_trip"))
    df_merge = pd.merge(df_merge, df_week, on="ernr", suffixes=("", "_week"))

    # get all unique legs within trips
    df_merge = df_merge.drop_duplicates(["ernr", "journeyId", "fromPlz_leg", "toPlz_leg"])

    # fix OD pairs and distances for loading trips
    f_loading = (df_merge["journeyLoading"] == 1) & (df_merge["journeyUnloading"] > 1)
    index_first_loading = df_merge.loc[f_loading, ["ernr", "journeyId", "fromPlz_leg"]].drop_duplicates(keep="first").index

    df_forward_differences = df_merge["transportKmCH"].diff()
    df_unloading_points = df_merge.loc[:, ["toLand_leg", "toNuts_leg", "toPlz_leg"]].shift(periods=1)

    df_merge.loc[f_loading, "origin_country"] = df_unloading_points[f_loading]["toLand_leg"]
    df_merge.loc[f_loading, "origin_nuts_id"] = df_unloading_points[f_loading]["toNuts_leg"]
    df_merge.loc[f_loading, "origin_postal_code"] = df_unloading_points[f_loading]["toPlz_leg"]
    df_merge.loc[f_loading, "destination_country"] = df_merge.loc[f_loading, "toLand_leg"]
    df_merge.loc[f_loading, "destination_nuts_id"] = df_merge.loc[f_loading, "toNuts_leg"]
    df_merge.loc[f_loading, "destination_postal_code"] = df_merge.loc[f_loading, "toPlz_leg"]
    df_merge.loc[index_first_loading, "origin_country"] = df_merge.loc[index_first_loading, "fromLand_leg"]
    df_merge.loc[index_first_loading, "origin_nuts_id"] = df_merge.loc[index_first_loading, "fromNuts_leg"]
    df_merge.loc[index_first_loading, "origin_postal_code"] = df_merge.loc[index_first_loading, "fromPlz_leg"]

    df_merge.loc[f_loading, "distance_km"] = df_forward_differences[f_loading]
    df_merge.loc[index_first_loading, "distance_km"] = df_merge.loc[index_first_loading, "transportKmCH"]

    # fix OD pairs and distances for unloading trips
    f_unloading = (df_merge["journeyLoading"] > 1) & (df_merge["journeyUnloading"] == 1)
    index_last_unloading = df_merge.loc[f_unloading, ["ernr", "journeyId", "toPlz_leg"]].drop_duplicates(keep="last").index

    df_backward_differences = df_merge["transportKmCH"].diff(periods=-1)
    df_unloading_points = df_merge.loc[:, ["fromLand_leg", "fromNuts_leg", "fromPlz_leg"]].shift(periods=-1)

    df_merge.loc[f_unloading, "origin_country"] = df_merge.loc[f_unloading, "fromLand_leg"]
    df_merge.loc[f_unloading, "origin_nuts_id"] = df_merge.loc[f_unloading, "fromNuts_leg"]
    df_merge.loc[f_unloading, "origin_postal_code"] = df_merge.loc[f_unloading, "fromPlz_leg"]
    df_merge.loc[f_unloading, "destination_country"] = df_unloading_points[f_unloading]["fromLand_leg"]
    df_merge.loc[f_unloading, "destination_nuts_id"] = df_unloading_points[f_unloading]["fromNuts_leg"]
    df_merge.loc[f_unloading, "destination_postal_code"] = df_unloading_points[f_unloading]["fromPlz_leg"]
    df_merge.loc[index_last_unloading, "destination_country"] = df_merge.loc[index_last_unloading, "toLand_leg"]
    df_merge.loc[index_last_unloading, "destination_nuts_id"] = df_merge.loc[index_last_unloading, "toNuts_leg"]
    df_merge.loc[index_last_unloading, "destination_postal_code"] = df_merge.loc[index_last_unloading, "toPlz_leg"]

    df_merge.loc[f_unloading, "distance_km"] = df_backward_differences[f_unloading]
    df_merge.loc[index_last_unloading, "distance_km"] = df_merge.loc[index_last_unloading, "transportKmCH"]

    # copy OD pairs and distances for single-leg trips
    f_single = (df_merge["journeyLoading"] <= 1) & (df_merge["journeyUnloading"] <= 1)
    df_merge.loc[f_single, "origin_country"] = df_merge.loc[f_single, "fromLand_leg"]
    df_merge.loc[f_single, "origin_nuts_id"] = df_merge.loc[f_single, "fromNuts_leg"]
    df_merge.loc[f_single, "origin_postal_code"] = df_merge.loc[f_single, "fromPlz_leg"]
    df_merge.loc[f_single, "destination_country"] = df_merge.loc[f_single, "toLand_leg"]
    df_merge.loc[f_single, "destination_nuts_id"] = df_merge.loc[f_single, "toNuts_leg"]
    df_merge.loc[f_single, "destination_postal_code"] = df_merge.loc[f_single, "toPlz_leg"]
    df_merge.loc[f_single, "distance_km"] = df_merge.loc[f_single, "transportKmCH"]

    # rename columns
    df_merge = df_merge.rename(RENAMES, axis=1)

    # rename vehicle types
    df_merge["vehicle_type"] = df_merge["vehicle_type"].replace(VEHICLE_TYPES)

    # There are some NUTS ids that do not exist in our NUTS data (maybe old ids)
    # for now, drop all trips where NUTS not in NUTS data
    print("Dropping all OD pairs where NUTS id not contained in NUTS data ...")
    number_trips = len(df_merge)
    df_nuts = context.stage("data.spatial.nuts")
    nuts_ids = list(df_nuts["nuts_id"].unique())
    df_merge = df_merge[(df_merge["origin_nuts_id"].isin(nuts_ids)) & (df_merge["destination_nuts_id"].isin(nuts_ids))]
    number_trips_dropped = number_trips - len(df_merge)
    print("Dropped %s of %s OD pairs" % (number_trips_dropped, number_trips))

    # package
    df_merge = df_merge[FIELDS]

    return df_merge
