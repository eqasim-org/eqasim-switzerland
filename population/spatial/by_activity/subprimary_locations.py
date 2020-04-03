import pandas as pd
import numpy.linalg as la
import pandas as pd
from sklearn.neighbors import KDTree


def configure(context):
    context.stage("population.sociodemographics")
    context.stage("population.activities")
    context.stage("population.spatial.by_activity.primary_locations")
    context.stage("data.microcensus.commute_extrapolation")
    context.stage("population.opportunities")

def execute(context):
    df_activities = context.stage("population.activities")

    # Attach the location of the primary activity to any activity
    df_primary_locations = context.stage("population.spatial.by_activity.primary_locations")
    df_primary_locations = pd.merge(
        df_activities, df_primary_locations,
        how = "inner", on = ["person_id", "activity_id"]
    )[["person_id", "is_commute", "location_x", "location_y", "location_id"]]
    df_primary_locations = df_primary_locations[df_primary_locations["is_commute"]]
    df_primary_locations = df_primary_locations[["person_id", "location_x", "location_y", "location_id"]]
    df_primary_locations.columns = ["person_id", "primary_x", "primary_y", "primary_id"]

    df_activities = pd.merge(
        df_activities, df_primary_locations,
        on = "person_id", how = "left"
    )

    # Find all the activities that are subprimary, ie. they are work or education,
    # but not the major commute activity in the plan
    df_activities = df_activities[~df_activities["is_commute"]]
    df_activities = df_activities[df_activities["purpose"].isin(["work", "education"])]
    assert(len(df_activities) == len(df_activities.dropna()))

    df_subprimary = df_activities[[
        "person_id", "activity_id", "primary_x", "primary_y", "purpose"
    ]]

    # Impute home location and MZ ID
    df_persons = context.stage("population.sociodemographics")[[
        "person_id", "mz_person_id", "home_x", "home_y"
    ]]

    df_subprimary = pd.merge(df_subprimary, df_persons, on = "person_id")

    # Impute commute extrapolation information
    df_extrapolation = context.stage("data.microcensus.commute_extrapolation")
    df_extrapolation.loc[:, "activity_id"] = df_extrapolation.loc[:, "trip_id"] + 1

    df_subprimary = pd.merge(
        df_subprimary, df_extrapolation,
        on = ["mz_person_id", "activity_id"], how = "left"
    )
    assert(not np.isnan(df_subprimary["commute_direct_distance"]).any())

    # Compute the distance between home and primary
    home_location = df_subprimary[["home_x", "home_y"]].values
    primary_location = df_subprimary[["primary_x", "primary_y"]].values
    primary_distance = la.norm(home_location - primary_location, axis = 1)

    # A) Fix the cases where the commute is collapsed to one point. Just put the
    # subprimary activities also to this position.
    f = primary_distance == 0.0
    df_subprimary.loc[f, "location_x"] = primary_location[f,0]
    df_subprimary.loc[f, "location_y"] = primary_location[f,1]

    # B) Extrapolate the locations for all other cases

    # B1) First, the cases where home and primary were originally collapsed
    f = (primary_distance > 0.0) & np.isnan(df_subprimary["commute_tangential_ratio"])

    angles = np.random.random(np.count_nonzero(f))
    df_subprimary.loc[f, "location_x"] = home_location[f,0] + np.cos(angles) * df_subprimary.loc[f, "commute_direct_distance"]
    df_subprimary.loc[f, "location_y"] = home_location[f,1] + np.sin(angles) * df_subprimary.loc[f, "commute_direct_distance"]

    # B2) Second, the cases where we do an actual extrapolation
    f = (primary_distance > 0.0) & ~np.isnan(df_subprimary["commute_tangential_ratio"])

    tangent = (primary_location[f,:] - home_location[f,:]) / primary_distance[f, np.newaxis]
    normal = np.dot(tangent, np.array([[0.0, -1.0], [1.0, 0.0]]))

    tangential_ratio = df_subprimary.loc[f, "commute_tangential_ratio"].values.reshape(-1, 1)
    normal_ratio = df_subprimary.loc[f, "commute_normal_ratio"].values.reshape(-1, 1)

    extrapolation = home_location[f,:]
    extrapolation += tangent * tangential_ratio * primary_distance[f, np.newaxis]
    extrapolation += normal * normal_ratio * primary_distance[f, np.newaxis]

    df_subprimary.loc[f, "location_x"] = extrapolation[:,0]
    df_subprimary.loc[f, "location_y"] = extrapolation[:,1]
    assert(not np.isnan(df_subprimary["location_x"]).any())

    # Attach the coordinates to actual locations
    df_opportunities = context.stage("population.opportunities")

    for purpose in ["work", "education"]:
        print("Building KD tree for opportunities: %s" % purpose)
        f_opportunities = df_opportunities["offers_%s" % purpose]
        coordinates = np.vstack([df_opportunities.loc[f_opportunities, "location_x"], df_opportunities.loc[f_opportunities, "location_y"]]).T
        indices = np.arange(len(df_opportunities))[f_opportunities]
        kd_tree = KDTree(coordinates)

        print("Assigning opportunities: %s" % purpose)
        f_subprimary = df_subprimary["purpose"] == purpose
        coordinates = np.vstack([df_subprimary.loc[f_subprimary, "location_x"], df_subprimary.loc[f_subprimary, "location_y"]]).T
        selection = kd_tree.query(coordinates, return_distance = False).flatten()
        indices = indices[selection]

        df_subprimary.loc[f_subprimary, "location_x"] = df_opportunities.iloc[indices]["location_x"].values
        df_subprimary.loc[f_subprimary, "location_y"] = df_opportunities.iloc[indices]["location_y"].values
        df_subprimary.loc[f_subprimary, "location_id"] = df_opportunities.iloc[indices]["location_id"].values

    df_subprimary = df_subprimary[[
        "person_id", "activity_id", "location_x", "location_y", "location_id"
    ]]

    return df_subprimary
