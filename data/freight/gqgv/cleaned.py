RENAMES = {"ORIGIN":"origin_nuts_id",
           "DESTINATION":"destination_nuts_id",
           "CH_MUNICIPALITY_ORIGIN": "origin_municipality",
           "CH_MUNICIPALITY_DESTINATION": "destination_municipality",
           "COUNTRY_OF_LOADING": "origin_country",
           "COUNTRY_OF_UNLOADING": "destination_country",
           "VEHICLE_TYPE": "vehicle_type",
           "TYPE_OF_GOOD": "good_type",
           "KM_PERFORMANCE": "distance_km",
           "WEIGHTING_FACTOR": "weight",
           "DIVISOR": "divisor"
           }

FIELDS = ["origin_nuts_id","destination_nuts_id",
          "origin_municipality", "destination_municipality",
          "origin_country", "destination_country",
          "vehicle_type", "good_type", "distance_km", "weight"
          ]

# VEHICLE_TYPES = {
#         1:"truck",
#         2:"road train",
#         3:"semi-trailer truck"
#     }

VEHICLE_TYPES = {
    1: "truck",
    2: "truck",
    3: "truck"
}


def configure(context):
    context.stage("data.freight.gqgv.raw")
    context.stage("data.spatial.nuts")


def execute(context):
    df = context.stage("data.freight.gqgv.raw")

    # rename columns
    df = df.rename(RENAMES, axis=1)

    # apply divisor to weight
    df["weight"] /= df["divisor"]

    # rename vehicle types
    df["vehicle_type"] = df["vehicle_type"].replace(VEHICLE_TYPES)

    # There are some NUTS ids that do not exist in our NUTS data (maybe old ids)
    # for now, drop all trips where NUTS not in NUTS data
    print("Dropping all OD pairs where NUTS id not contained in NUTS data ...")
    number_trips = len(df)
    df_nuts = context.stage("data.spatial.nuts")
    nuts_ids = list(df_nuts["nuts_id"].unique())
    df = df[(df["origin_nuts_id"].isin(nuts_ids)) & (df["destination_nuts_id"].isin(nuts_ids))]
    number_trips_dropped = number_trips - len(df)
    print("Dropped %s of %s OD pairs" % (number_trips_dropped, number_trips))

    # package
    df = df[FIELDS]

    return df


