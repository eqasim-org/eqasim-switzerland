import pandas as pd
import numpy.linalg as la
import pandas as pd


def configure(context):
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.commute")

# TODO: Merge this into data.microcensus.commute

def execute(context):
    df_trips = context.stage("data.microcensus.trips")[[
        "person_id", "trip_id", "destination_x", "destination_y", "purpose"
    ]]

    df_trips = df_trips[df_trips["purpose"].isin(["work", "education"])]

    df_persons = context.stage("data.microcensus.persons")[[
        "person_id", "home_x", "home_y"
    ]]

    df_commute = context.stage("data.microcensus.commute")[[
        "person_id", "commute_trip_id", "commute_x", "commute_y"
    ]]

    df_trips = pd.merge(df_trips, df_commute, on = "person_id")
    df_trips = pd.merge(df_trips, df_persons, on = "person_id")

    df_trips = df_trips[[
        "person_id", "trip_id", "commute_trip_id", "commute_x", "commute_y", "home_x", "home_y", "destination_x", "destination_y"
    ]]

    data = df_trips[["home_x", "home_y", "commute_x", "commute_y", "destination_x", "destination_y"]].values

    home_coordinates = data[:, 0:2]
    primary_coordinates = data[:, 2:4]
    secondary_coordinates = data[:, 4:6]

    primary_distance = la.norm(primary_coordinates - home_coordinates, axis = 1)
    primary_direction = (primary_coordinates - home_coordinates) / primary_distance[:, np.newaxis]

    secondary_distance = la.norm(secondary_coordinates - home_coordinates, axis = 1)
    secondary_direction = (secondary_coordinates - home_coordinates) / secondary_distance[:, np.newaxis]

    tangential_distance = np.sum(primary_direction * secondary_direction * secondary_distance[:, np.newaxis], axis = 1)
    tangential_factor = tangential_distance / primary_distance

    normal_direction = np.dot(primary_direction, np.array([[0.0, -1.0], [1.0, 0.0]]))
    center_direction = secondary_coordinates - (home_coordinates + primary_direction * tangential_distance[:, np.newaxis])
    normal_distance = np.sum(normal_direction * center_direction, axis = 1)

    df_trips.loc[:, "commute_tangential_ratio"] = tangential_distance / primary_distance
    df_trips.loc[:, "commute_normal_ratio"] = normal_distance / primary_distance
    df_trips.loc[:, "commute_direct_distance"] = secondary_distance

    df_trips = df_trips[[
        "person_id", "trip_id", "commute_tangential_ratio", "commute_normal_ratio", "commute_direct_distance"
    ]]
    df_trips.columns = ["mz_person_id", "trip_id", "commute_tangential_ratio", "commute_normal_ratio", "commute_direct_distance"]

    return df_trips
