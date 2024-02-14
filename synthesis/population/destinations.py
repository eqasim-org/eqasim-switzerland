import pandas as pd
import numpy as np
import data.spatial.utils as spatial_utils
import geopandas as gpd


def configure(context):
    context.stage("data.statent.statent")
    context.config("use_detailed_activities")
    context.config("output_path")


def execute(context):
    det_activities = context.config("use_detailed_activities")
    df = pd.DataFrame(context.stage("data.statent.statent")[["enterprise_id", "x", "y", "noga", "number_employees"]],
                                    copy=True)
    df.columns = ["destination_id", "destination_x", "destination_y", "noga", "number_employees"]

    df.loc[:, "offers_work"] = True
    df.loc[:, "offers_other"] = True

    # 85 = education
    df.loc[:, "offers_education"] = df["noga"].str.startswith("85")
    df.loc[df["noga"].str.startswith("85"), "education_type"] = "diverse"
    df.loc[df["noga"].str.startswith("851"), "education_type"] = "kindergarten"
    df.loc[df["noga"].str.startswith("852"), "education_type"] = "primary"
    df.loc[df["noga"] == "853101", "education_type"] = "secondary1"
    df.loc[df["noga"] == "853102", "education_type"] = "secondary2"
    df.loc[df["noga"] == "853103", "education_type"] = "secondary2"
    df.loc[df["noga"] == "853200", "education_type"] = "secondary2"
    df.loc[df["noga"].str.startswith("854"), "education_type"] = "tertiary"
    df["education_type"] = df["education_type"].astype("category")

    # 90 = arts, entertainment, leisure; 56 = gastronomy
    df.loc[:, "offers_leisure"] = df["noga"].str.startswith("90") | df["noga"].str.startswith("56") | df["noga"].str.startswith("91") |     df["noga"].str.startswith("55") | df["noga"].str.startswith("94") | (df["noga"] == "591400") | (df["noga"] == "920000") | (df["noga"] == "855200") | (df["noga"] == "932100") | (df["noga"] == "932900") | (df["noga"] == "855100") | df["noga"].str.startswith("931") 

    if det_activities:
        df.loc[:, "offers_culture"] = df["noga"].str.startswith("90") | df["noga"].str.startswith("91") | (df["noga"] == "591400") | (df["noga"] == "920000") | (df["noga"] == "855200") | (df["noga"] == "932100") | (df["noga"] == "932900")

        df.loc[:, "offers_religion"] = df["noga"].str.startswith("94")
        df.loc[:, "offers_gastronomy"] = df["noga"].str.startswith("55") | df["noga"].str.startswith("56")
        df.loc[:, "offers_sport"] = (df["noga"] == "855100") | df["noga"].str.startswith("931")
        df.loc[:, "offers_other(L)"] = df["offers_leisure"]

    # 47 = retail
    df.loc[:, "offers_shop"] = df["noga"].str.startswith("47")

    if det_activities:
        df.loc[:, "offers_grocery"] = df["noga"].str.startswith("471") | df["noga"].str.startswith("472") | df["noga"].str.startswith("473") | (df["noga"] == "478100")
        df.loc[:, "offers_other(S)"] = df["offers_shop"] & np.logical_not(df["offers_grocery"])

        df.loc[:, "offers_visits"] = False
        df.loc[:, "offers_volunteer"] = True
        df.loc[:, "offers_outdoor"] = False

    #del df["noga"]

    df = spatial_utils.to_gpd(context, df, x="destination_x", y="destination_y", coord_type="facility")
    max_id = np.max(df["destination_id"].values.tolist())

    # Services
    if det_activities:
        df.loc[:, "offers_services"] = df["noga"].str.startswith("8690") | df["noga"].str.startswith("95") | df["noga"].str.startswith("33") | df["noga"].str.startswith("6419") | df["noga"].str.startswith("6512") |  (df["noga"] == "750000") |  (df["noga"] == "861001") |  (df["noga"] == "861002") |  (df["noga"] == "862100") |  (df["noga"] == "862200") |  (df["noga"] == "862300") |  (df["noga"] == "772200") |  (df["noga"] == "960101") |  (df["noga"] == "960102") |  (df["noga"] == "960201") |  (df["noga"] == "960202") |  (df["noga"] == "960401") |  (df["noga"] == "960402") |  (df["noga"] == "855300") |  (df["noga"] == "531000") |  (df["noga"] == "532000") |  (df["noga"] == "452001") |  (df["noga"] == "452002") |  (df["noga"] == "649201") |  (df["noga"] == "651100") |  (df["noga"] == "681000") |  (df["noga"] == "682001") |  (df["noga"] == "682002") |  (df["noga"] == "683100") |  (df["noga"] == "683200") |  (df["noga"] == "691001") |  (df["noga"] == "692000") |  (df["noga"] == "771100") |  (df["noga"] == "772100") |  (df["noga"] == "772900") |  (df["noga"] == "791100") |  (df["noga"] == "791200") |  (df["noga"] == "841100") |  (df["noga"] == "842400") |  (df["noga"] == "842301") |  (df["noga"] == "843000") 

    df.loc[:, "offers_outdoor"] = False

    # Outdoor points
    if det_activities:
        outdoor_path = "/nas/asallard/Switzerland/Outdoor_points"
        outdoor = gpd.read_file("%s/sample_points.shp"% outdoor_path)
        initial_crs = outdoor.crs

        print("OUTDOOR POINTS LOADED")

        outdoor["destination_id"] = np.arange(max_id + 1, max_id + 1 + len(outdoor), 1)
        outdoor["destination_x"] = [p.x for p in outdoor.geometry.values]
        outdoor["destination_y"] = [p.y for p in outdoor.geometry.values]
        outdoor["offers_work"] = False
        outdoor["offers_leisure"] = True
        outdoor["offers_grocery"] = False
        outdoor["offers_other(S)"] = False
        outdoor["offers_culture"] = False
        outdoor["education_type"] = None
        outdoor["offers_religion"] = False
        outdoor["offers_gastronomy"] = False
        outdoor["offers_sport"] = False
        outdoor["offers_other(L)"] = True
        outdoor["offers_other"] = False
        outdoor["offers_visits"] = False
        outdoor["offers_volunteer"] = False
        outdoor["offers_outdoor"] = True
        outdoor["offers_services"] = True
        outdoor["nb_employees"] = 1
        print(initial_crs)

        outdoor = spatial_utils.to_gpd(context, outdoor, x="destination_x", y="destination_y", crs=initial_crs, coord_type="outdoor")

        outdoor["destination_x"] = [p.x for p in outdoor.geometry.values]
        outdoor["destination_y"] = [p.y for p in outdoor.geometry.values]

        print(outdoor.crs)

        df = pd.concat([df, outdoor])

        df_gastronomy = df[df["offers_gastronomy"]][["destination_id", "destination_x", "destination_y"]]
        df_grocery = df[df["offers_grocery"]][["destination_id", "destination_x", "destination_y"]]
        df_shopping = df[df["offers_other(S)"]][["destination_id", "destination_x", "destination_y"]]
        df_culture = df[df["offers_culture"]][["destination_id", "destination_x", "destination_y"]]

        output_path = context.config("output_path")
        #df_gastronomy.to_csv("%s/destinations_gastronomy.csv" % output_path, index = False)
        #df_grocery.to_csv("%s/destinations_grocery.csv" % output_path, index = False)
        #df_shopping.to_csv("%s/destinations_shopping.csv" % output_path, index = False)
        #df_culture.to_csv("%s/destinations_culture.csv" % output_path, index = False)

        return df[["destination_id", "destination_x", "destination_y",
               "offers_work", "offers_education", "offers_leisure", "offers_grocery", "offers_other(S)", "offers_culture", "education_type", "offers_religion", "offers_gastronomy", "offers_sport", "offers_other(L)", "offers_other", "offers_visits", "offers_volunteer",
"offers_outdoor", "offers_services", "offers_shop", "geometry", "number_employees", "noga"]]

    else:
        return df[["destination_id", "destination_x", "destination_y",
               "offers_work", "offers_education", "offers_leisure", "education_type", "offers_shop", "offers_other", "geometry", "number_employees"]]
