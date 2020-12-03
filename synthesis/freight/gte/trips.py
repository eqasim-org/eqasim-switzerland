import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


def configure(context):
    context.stage("data.freight.gte.od")
    context.stage("data.spatial.zones")
    context.stage("data.statent.statent")
    context.stage("data.spatial.nuts")
    context.stage("data.spatial.swiss_border")
    context.config("input_downsampling")


def execute(context):
    demands, origin_pdf_matrices, od_pdf_matrices = context.stage("data.freight.gte.od")
    input_downsampling = context.config("input_downsampling")

    trips_frames = []

    print("Computing freight origin-destination counts ...")
    for vehicle_type in demands.keys():

        demand = np.round(input_downsampling * demands[vehicle_type])

        # compute origin counts
        origin_counts = np.random.multinomial(demand, origin_pdf_matrices[vehicle_type].values[:, 0])
        counts = np.zeros(od_pdf_matrices[vehicle_type].shape, dtype=np.int)

        # compute origin-destination counts
        for i in range(len(origin_counts)):
            if origin_counts[i] > 0:
                assert (~np.any(np.isnan(od_pdf_matrices[vehicle_type].values[i])))
                counts[i, :] = np.random.multinomial(origin_counts[i], od_pdf_matrices[vehicle_type].values[i, :])

        assert (len(counts) == len(origin_counts))

        # generate single trips
        with context.progress(label="Creating %i %s trips ..." % (int(demand), vehicle_type),
                              total=np.sum(counts)) as progress:
            for origin_index in range(counts.shape[0]):
                for destination_index in range(counts.shape[1]):

                    number_of_trips = counts[origin_index, destination_index]

                    if number_of_trips > 0:
                        origin_id = od_pdf_matrices[vehicle_type].index[origin_index]
                        destination_id = od_pdf_matrices[vehicle_type].columns[destination_index]

                        trips = np.repeat(np.array([[origin_id, destination_id, vehicle_type]], dtype=np.object),
                                          number_of_trips, axis=0)

                        trips_frames.append(pd.DataFrame(columns=["origin_id", "destination_id", "vehicle_type"],
                                                         data=trips))

                        progress.update(number_of_trips)

    # concatenate into single data frame
    df_trips = pd.concat(trips_frames)

    # get coordinates of Swiss border
    df_nuts = context.stage("data.spatial.nuts")
    swiss_border = context.stage("data.spatial.swiss_border")

    swiss_border = np.vstack(
        [swiss_border.values[0].boundary.coords.xy[0], swiss_border.values[0].boundary.coords.xy[1]]).T
    kd_tree = KDTree(swiss_border)

    coordinates = np.vstack([df_nuts["geometry"].centroid.x, df_nuts["geometry"].centroid.y]).T
    indices = kd_tree.query(coordinates, return_distance=False).flatten()

    df_nuts["border_coordinate_x"] = swiss_border[indices, 0]
    df_nuts["border_coordinate_y"] = swiss_border[indices, 1]

    # add origin and destination locations
    df_statent = context.stage("data.statent.statent")

    df_trips["origin_x"] = 0.0
    df_trips["origin_y"] = 0.0

    origin_ids = df_trips["origin_id"].unique()
    destination_ids = df_trips["destination_id"].unique()

    with context.progress(label="Setting origin locations ...", total=len(origin_ids)) as progress:

        for origin_id in origin_ids:

            nuts_level = len(origin_id) - 2

            if "CH" not in origin_id:

                # find closest border coordinate
                df_trips.loc[df_trips["origin_id"] == origin_id, "origin_x"] = df_nuts.loc[
                    df_nuts["nuts_id"] == origin_id, "border_coordinate_x"].values[0]
                df_trips.loc[df_trips["origin_id"] == origin_id, "origin_y"] = df_nuts.loc[
                    df_nuts["nuts_id"] == origin_id, "border_coordinate_y"].values[0]

            else:

                number_trips = np.sum(df_trips["origin_id"] == origin_id)

                # generate probability for each enterprise based on number of employees (proxy for size)
                indices = df_statent[df_statent[("nuts_id_level_" + str(nuts_level))] == origin_id].index.values

                # select random enterprise within NUTS zone
                choices = np.random.choice(indices, number_trips)
                enterprises = df_statent.values[choices, 0:2]

                # assign coordinates
                df_trips.loc[df_trips["origin_id"] == origin_id, "origin_x"] = enterprises[:, 0]
                df_trips.loc[df_trips["origin_id"] == origin_id, "origin_y"] = enterprises[:, 1]

            progress.update()

    with context.progress(label="Setting destination locations ...", total=len(destination_ids)) as progress:

        for destination_id in destination_ids:

            nuts_level = len(destination_id) - 2

            if "CH" not in destination_id:

                # find closest border coordinate
                df_trips.loc[df_trips["destination_id"] == destination_id, "destination_x"] = df_nuts.loc[
                    df_nuts["nuts_id"] == destination_id, "border_coordinate_x"].values[0]

                df_trips.loc[df_trips["destination_id"] == destination_id, "destination_y"] = df_nuts.loc[
                    df_nuts["nuts_id"] == destination_id, "border_coordinate_y"].values[0]

            else:

                number_trips = np.sum(df_trips["destination_id"] == destination_id)

                # generate probability for each enterprise based on number of employees (proxy for size)
                indices = df_statent[df_statent[("nuts_id_level_" + str(nuts_level))] == destination_id].index.values

                # select random enterprise within NUTS zone
                choices = np.random.choice(indices, number_trips)
                enterprises = df_statent.values[choices, 0:2]

                # assign coordinates
                df_trips.loc[df_trips["destination_id"] == destination_id, "destination_x"] = enterprises[:, 0]
                df_trips.loc[df_trips["destination_id"] == destination_id, "destination_y"] = enterprises[:, 1]

            progress.update()

    # package up
    df_trips = df_trips[["origin_x", "origin_y",
                         "destination_x", "destination_y",
                         "vehicle_type"]
    ]

    return df_trips
