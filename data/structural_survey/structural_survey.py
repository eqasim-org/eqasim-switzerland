import numpy as np

import data.spatial.countries
import data.spatial.countries
import data.spatial.municipalities
import data.spatial.municipalities
import data.spatial.quarters
import data.spatial.quarters
import data.spatial.utils
import data.spatial.utils
import data.spatial.zones
import data.spatial.zones


def configure(context):
    context.stage("data.structural_survey.raw")
    context.stage("data.statpop.statpop")
    context.stage("data.statent.statent")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.countries")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")


def execute(context):
    df_se = context.stage("data.structural_survey.raw")

    df_zones = context.stage("data.spatial.zones")
    df_countries = context.stage("data.spatial.countries")
    df_municipalities, df_municipality_mapping = context.stage("data.spatial.municipalities")
    df_quarters = context.stage("data.spatial.quarters")

    # Find the correct modes
    df_se.loc[:, "mode_numeric"] = df_se.loc[:, "mode"].astype(np.int)
    df_se.loc[df_se["mode_numeric"] == -10, "mode"] = "unknown"
    df_se.loc[df_se["mode_numeric"] == -9, "mode"] = "unknown"
    df_se.loc[df_se["mode_numeric"] == -8, "mode"] = "unknown"
    df_se.loc[df_se["mode_numeric"] == 1, "mode"] = "walk"  # walking
    df_se.loc[df_se["mode_numeric"] == 2, "mode"] = "walk"  # skateboard
    df_se.loc[df_se["mode_numeric"] == 3, "mode"] = "bike"  # bike / elec. bike
    df_se.loc[df_se["mode_numeric"] == 4, "mode"] = "car"  # Mofa / Moped / light motor bike
    df_se.loc[df_se["mode_numeric"] == 5, "mode"] = "car"  # Car as driver or passenger
    df_se.loc[df_se["mode_numeric"] == 6, "mode"] = "car"  # company bus
    df_se.loc[df_se["mode_numeric"] == 7, "mode"] = "pt"  # Train
    df_se.loc[df_se["mode_numeric"] == 8, "mode"] = "pt"  # Tram / Metro
    df_se.loc[df_se["mode_numeric"] == 9, "mode"] = "pt"  # Bus
    df_se.loc[df_se["mode_numeric"] == 10, "mode"] = "other"  # Ship, cable car, ...
    del df_se["mode_numeric"]

    # Impute the home zone
    df_se.loc[:, "municipality_id"] = df_se["home_municipality"]
    df_se.loc[:, "quarter_id"] = df_se["home_quarter"]
    df_se = data.spatial.quarters.update_quarter_ids(df_se, df_quarters)
    df_se = data.spatial.municipalities.update_municipality_ids(df_se, df_municipality_mapping)
    df_se.loc[:, "home_municipality_id"] = df_se.loc[:, "municipality_id"]
    df_se.loc[:, "home_quarter_id"] = df_se.loc[:, "quarter_id"]

    # Impute the work zone
    df_se.loc[:, "country_id"] = df_se["work_country"]
    df_se.loc[:, "municipality_id"] = df_se["work_municipality"]
    df_se.loc[:, "quarter_id"] = df_se["work_quarter"]
    df_se = data.spatial.quarters.update_quarter_ids(df_se, df_quarters)
    df_se = data.spatial.municipalities.update_municipality_ids(df_se, df_municipality_mapping)
    df_se = data.spatial.countries.update_country_ids(df_se, df_countries)
    df_se.loc[:, "work_municipality_id"] = df_se.loc[:, "municipality_id"]
    df_se.loc[:, "work_quarter_id"] = df_se.loc[:, "quarter_id"]
    df_se.loc[:, "work_country_id"] = df_se.loc[:, "country_id"]

    f_no_home = np.isnan(df_se["home_municipality_id"]) & np.isnan(df_se["home_quarter_id"])
    f_no_work = np.isnan(df_se["work_country_id"]) & np.isnan(df_se["work_municipality_id"]) & np.isnan(
        df_se["work_quarter_id"])

    print("Found %d observations without home location information" % np.count_nonzero(f_no_home))
    print("Found %d observations without work location information" % np.count_nonzero(f_no_work))

    # Cleanup
    for variable in ["quarter_id", "municipality_id", "country_id", "home_quarter", "home_municipality",
                     "work_municipality", "work_country"]:
        del df_se[variable]

    # Now we face the problem that in an old structural survey we may assign
    # a municipality that has now quarters, or we find quarters that have been
    # resolved and now we just have a municipality in our zoning system.
    # Therefore the approach is as follows: Since we know the quarter and municipality
    # for all the STATPOP persons, we can sample one of them for each structural survey
    # observation. This has two purposes: First, we attach a physical location to each
    # structural survey observation, and second this physical location is also
    # consistent with the overall population density in the area. (Imagine there is
    # a municiaplity that has been divided in quarters: We would still like to
    # reproduce the same population density for the structural survey, but just by
    # knowing which municipality the people are from does not give us that information.)
    # Then, we use the new physical location to get the zone id of the overall zoning
    # system in the pipeline.

    df_statpop = context.stage("data.statpop.statpop")[[
        "person_id", "home_x", "home_y", "home_municipality_id", "home_quarter_id", "home_zone_id"
    ]]

    # Assign coordinates in the home municipalities

    se_municipality_ids = np.unique(df_se["home_municipality_id"].dropna()).astype(np.int)
    for municipality_id in context.progress(se_municipality_ids,
                                            label="Imputing home locations by municipality from STATPOP"):
        indices = np.where(df_statpop["home_municipality_id"] == municipality_id)[0]

        if len(indices) > 0:
            f = df_se["home_municipality_id"] == municipality_id
            indices = indices[np.random.randint(len(indices), size=(np.count_nonzero(f),))]
            df_se.loc[f, "home_municipality_x"] = df_statpop.iloc[indices]["home_x"].values
            df_se.loc[f, "home_municipality_y"] = df_statpop.iloc[indices]["home_y"].values

    unassigned_municipality_ids = np.unique(
        df_se[np.isnan(df_se["home_municipality_x"])]["home_municipality_id"].dropna())
    print("A number of %d municipalities could not be assigned from STATPOP" % len(unassigned_municipality_ids))

    for municipality_id in context.progress(unassigned_municipality_ids,
                                            label="Sampling home locations for municipalities"):
        f = np.isnan(df_se["home_municipality_x"]) & (df_se["home_municipality_id"] == municipality_id)
        row = df_municipalities[df_municipalities["municipality_id"] == municipality_id].iloc[0]
        coordinates = data.spatial.utils.sample_coordinates(row, np.count_nonzero(f))
        df_se.loc[f, "home_municipality_x"], df_se.loc[f, "home_municipality_y"] = coordinates[:, 0], coordinates[:, 1]

    assert (~np.any(np.isnan(df_se["home_municipality_x"])))

    # Assign coordinates in the home quarters

    se_quarter_ids = np.unique(df_se["home_quarter_id"].dropna()).astype(np.int)
    for quarter_id in context.progress(se_quarter_ids, label="Imputing home locations by quarter from STATPOP"):
        indices = np.where(df_statpop["home_quarter_id"] == quarter_id)[0]

        if len(indices) > 0:
            f = df_se["home_quarter_id"] == quarter_id
            indices = indices[np.random.randint(len(indices), size=(np.count_nonzero(f),))]
            df_se.loc[f, "home_quarter_x"] = df_statpop.iloc[indices]["home_x"].values
            df_se.loc[f, "home_quarter_y"] = df_statpop.iloc[indices]["home_y"].values

    unassigned_quarter_ids = np.unique(df_se[np.isnan(df_se["home_quarter_x"])]["home_quarter_id"].dropna())
    print("A number of %d quarters could not be assigned from STATPOP" % len(unassigned_quarter_ids))

    for quarter_id in context.progress(unassigned_quarter_ids, label="Sampling home locations for municipalities"):
        f = np.isnan(df_se["home_quarter_x"]) & (df_se["home_quarter_id"] == quarter_id)
        row = df_quarters[df_quarters["quarter_id"] == quarter_id].iloc[0]
        coordinates = data.spatial.utils.sample_coordinates(row, np.count_nonzero(f))
        df_se.loc[f, "home_quarter_x"], df_se.loc[f, "home_quarter_y"] = coordinates[:, 0], coordinates[:, 1]

    quarter_count = np.count_nonzero(~np.isnan(df_se["home_quarter_x"]))
    municipality_count = np.count_nonzero(~np.isnan(df_se["home_municipality_x"]))

    print("Homes assigned by municipality:", municipality_count - quarter_count)
    print("Homes assigned by quarter:", quarter_count)

    df_se.loc[:, "home_x"] = df_se.loc[:, "home_municipality_x"]
    df_se.loc[:, "home_y"] = df_se.loc[:, "home_municipality_y"]

    f_quarter = ~np.isnan(df_se["home_quarter_x"])
    df_se.loc[f_quarter, "home_x"] = df_se.loc[f_quarter, "home_quarter_x"]
    df_se.loc[f_quarter, "home_y"] = df_se.loc[f_quarter, "home_quarter_y"]

    del df_se["home_municipality_x"]
    del df_se["home_municipality_y"]
    del df_se["home_quarter_x"]
    del df_se["home_quarter_y"]

    assert (~np.any(np.isnan(df_se["home_x"])))

    # The same we have to do with work places, except we can use STATENT here.

    df_statent = context.stage("data.statent.statent")[[
        "enterprise_id", "x", "y", "municipality_id", "quarter_id", "zone_id"
    ]]

    # Assign coordinates in the work municipalities

    se_municipality_ids = np.unique(df_se["work_municipality_id"].dropna()).astype(np.int)
    for municipality_id in context.progress(se_municipality_ids,
                                            label="Imputing work locations by municipality from STATENT"):
        indices = np.where(df_statent["municipality_id"] == municipality_id)[0]

        if len(indices) > 0:
            f = df_se["work_municipality_id"] == municipality_id
            indices = indices[np.random.randint(len(indices), size=(np.count_nonzero(f),))]
            df_se.loc[f, "work_municipality_x"] = df_statent.iloc[indices]["x"].values
            df_se.loc[f, "work_municipality_y"] = df_statent.iloc[indices]["y"].values

    unassigned_municipality_ids = np.unique(
        df_se[np.isnan(df_se["work_municipality_x"])]["work_municipality_id"].dropna())
    print("A number of %d municipalities could not be assigned from STATENT" % len(unassigned_municipality_ids))

    for municipality_id in context.progress(unassigned_municipality_ids,
                                            label="Sampling work locations for municipalities"):
        f = np.isnan(df_se["work_municipality_x"]) & (df_se["work_municipality_id"] == municipality_id)
        row = df_municipalities[df_municipalities["municipality_id"] == municipality_id].iloc[0]
        coordinates = data.spatial.utils.sample_coordinates(row, np.count_nonzero(f))
        df_se.loc[f, "work_municipality_x"], df_se.loc[f, "work_municipality_y"] = coordinates[:, 0], coordinates[:, 1]

    # Assign coordinates in the work quarters

    se_quarter_ids = np.unique(df_se["work_quarter_id"].dropna()).astype(np.int)
    for quarter_id in context.progress(se_quarter_ids, label="Imputing work locations by quarter from STATENT"):
        indices = np.where(df_statent["quarter_id"] == quarter_id)[0]

        if len(indices) > 0:
            f = df_se["work_quarter_id"] == quarter_id
            indices = indices[np.random.randint(len(indices), size=(np.count_nonzero(f),))]
            df_se.loc[f, "work_quarter_x"] = df_statent.iloc[indices]["x"].values
            df_se.loc[f, "work_quarter_y"] = df_statent.iloc[indices]["y"].values

    unassigned_quarter_ids = np.unique(df_se[np.isnan(df_se["work_quarter_x"])]["work_quarter_id"].dropna())
    print("A number of %d quarters could not be assigned from STATENT" % len(unassigned_quarter_ids))

    for quarter_id in context.progress(unassigned_quarter_ids, label="Sampling work locations for municipalities"):
        f = np.isnan(df_se["work_quarter_x"]) & (df_se["work_quarter_id"] == quarter_id)
        row = df_quarters[df_quarters["quarter_id"] == quarter_id].iloc[0]
        coordinates = data.spatial.utils.sample_coordinates(row, np.count_nonzero(f))
        df_se.loc[f, "work_quarter_x"], df_se.loc[f, "work_quarter_y"] = coordinates[:, 0], coordinates[:, 1]

    quarter_count = np.count_nonzero(~np.isnan(df_se["work_quarter_x"]))
    municipality_count = np.count_nonzero(~np.isnan(df_se["work_municipality_x"]))

    print("Work places assigned by municipality:", municipality_count - quarter_count)
    print("Work places assigned by quarter:", quarter_count)

    df_se.loc[:, "work_x"] = df_se.loc[:, "work_municipality_x"]
    df_se.loc[:, "work_y"] = df_se.loc[:, "work_municipality_y"]

    f_quarter = ~np.isnan(df_se["work_quarter_x"])
    df_se.loc[f_quarter, "work_x"] = df_se.loc[f_quarter, "work_quarter_x"]
    df_se.loc[f_quarter, "work_y"] = df_se.loc[f_quarter, "work_quarter_y"]

    del df_se["work_municipality_x"]
    del df_se["work_municipality_y"]
    del df_se["work_quarter_x"]
    del df_se["work_quarter_y"]

    # Now that all the coordinates are available, a zone can be assigned
    print("Imputing home zones ...")
    df_se = data.spatial.zones.impute(df_se, df_zones, zone_id_prefix="home_",
                                      municipality_id_field="home_municipality_id", quarter_id_field="home_quarter_id")

    print("Imputing work zones ...")
    df_se = data.spatial.zones.impute(df_se, df_zones, zone_id_prefix="work_", country_id_field="work_country_id",
                                      municipality_id_field="work_municipality_id", quarter_id_field="work_quarter_id")

    return df_se[[
        "home_municipality_id", "home_quarter_id", "home_zone_id", "home_zone_level",
        "work_country_id", "work_municipality_id", "work_quarter_id", "work_zone_id", "work_zone_level",
        "mode", "weight"
    ]]
