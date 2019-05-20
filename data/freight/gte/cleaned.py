import pandas as pd

RENAMES = {"ernr":"agent_id",
           "journeyId":"journey_id",
           "fromNuts":"origin_nuts_id",
           "toNuts":"destination_nuts_id",
           "fromPlz": "origin_postal_code",
           "toPlz": "destination_postal_code",
           "fromLand": "origin_country",
           "toLand": "destination_country",
           "vehicleKind":"vehicle_type",
           "grossingFactor":"weight"
           }

FIELDS = ["week", "weekday",
          "origin_nuts_id","destination_nuts_id",
          "origin_postal_code", "destination_postal_code",
          "origin_country", "destination_country",
          "vehicle_type", "weight"
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


def configure(context, require):
    require.stage("data.freight.gte.raw")
    require.stage("data.spatial.nuts")


def execute(context):
    df_transport, df_journey, df_week = context.stage("data.freight.gte.raw")

    # select transport columns of interest
    df_transport = df_transport[["ernr", "journeyId", "weekday",
                                 "fromNuts", "toNuts",
                                 "fromLand", "toLand",
                                 "fromPlz", "toPlz",
                                 "transportKm", "transportKmCH"]]

    # get start location of each agent
    df_start_locations = df_transport[~df_transport["ernr"].duplicated()][["fromPlz", "fromNuts", "fromLand"]]

    # get all unique stop along trips
    df_transport = df_transport.drop_duplicates()

    # separate into different trip legs
    df_transport.loc[1:, "fromPlz"] = df_transport.loc[:, "toPlz"].shift(periods=1)
    df_transport.loc[1:, "fromNuts"] = df_transport.loc[:, "toNuts"].shift(periods=1)
    df_transport.loc[1:, "fromLand"] = df_transport.loc[:, "toLand"].shift(periods=1)

    # reset initial start locations of each agent
    df_transport.loc[df_start_locations.index, ["fromPlz", "fromNuts", "fromLand"]] = df_start_locations

    # # get distances of first journeys
    # df_distances = df_transport.drop_duplicates(["ernr", "journeyId"], keep="first")[["transportKm", "transportKmCH"]]
    #
    # # calculate distance of each trip leg
    # df_transport.loc[1:, ["transportKm", "transportKmCH"]] = df_transport[["transportKm", "transportKmCH"]].diff().loc[1:,:]
    #
    # # reset first leg distance in each journey for each agent
    # df_transport.loc[df_distances.index, ["transportKm", "transportKmCH"]] = df_distances

    # select week columns of interest
    df_week = df_week[["ernr", "week", "vehicleKind", "grossingFactor"]]

    # merge
    df_merge = pd.merge(df_transport, df_week, on="ernr")

    # remove all trips not at least partially in CH
    df_merge = df_merge[df_merge["transportKmCH"] > 0]

    # rename columns
    df_merge = df_merge.rename(RENAMES, axis=1)

    # rename vehicle types
    df_merge["vehicle_type"] = df_merge["vehicle_type"].replace(VEHICLE_TYPES)

    # There are some NUTS ids that do not exist in our NUTS data (maybe old ids)
    # for now, drop all trips where NUTS not in NUTS data
    print("Dropping all trips where NUTS id not contained in NUTS data ...")
    number_trips = len(df_merge)
    df_nuts = context.stage("data.spatial.nuts")
    nuts_ids = list(df_nuts["nuts_id"].unique())
    df_merge = df_merge[(df_merge["origin_nuts_id"].isin(nuts_ids)) & (df_merge["destination_nuts_id"].isin(nuts_ids))]
    number_trips_dropped = number_trips - len(df_merge)
    print("Dropped %s of %s trips" % (number_trips_dropped, number_trips))

    # package
    df_merge = df_merge[FIELDS]

    return df_merge


