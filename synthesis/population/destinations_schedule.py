import pandas as pd
import numpy as np
import data.spatial.utils as spatial_utils
import geopandas as gpd


def configure(context):
    context.stage("synthesis.population.destinations")
    context.config("use_detailed_activities")


def execute(context):
    det_activities = context.config("use_detailed_activities")
    df = pd.DataFrame(context.stage("synthesis.population.destinations"), copy=True)

    if not det_activities:
        return df

    df["open_0-3"] = 0
    df["open_3-6"] = 0
    df["open_6-9"] = 0
    df["open_9-12"] = 0
    df["open_12-15"] = 0
    df["open_15-18"] = 0
    df["open_18-21"] = 0
    df["open_21-24"] = 0

    df.loc[:, "always_open"] = df["noga"].str.startswith("94") | (df["noga"] == "932900") | df["noga"].str.startswith("931") |  (df["noga"] == "861001") |  (df["noga"] == "861002")

    df.loc[:, "office_hours"] = (df["noga"] == "791100") |  (df["noga"] == "791200") |  (df["noga"] == "841100") |  (df["noga"] == "842400") |  (df["noga"] == "842301") |  (df["noga"] == "843000") | (df["noga"] == "649201") |  (df["noga"] == "651100") |  (df["noga"] == "681000") |  (df["noga"] == "682001") |  (df["noga"] == "682002") |  (df["noga"] == "683100") |  (df["noga"] == "683200") |  (df["noga"] == "691001") |  (df["noga"] == "692000") | df["noga"].str.startswith("6419") | df["noga"].str.startswith("6512") | (df["noga"] == "772200")

    df.loc[:, "office++"] = df["noga"].str.startswith("472") | (df["offers_shop"] & np.logical_not(df["offers_grocery"])) | df["noga"].str.startswith("8690") | df["noga"].str.startswith("95") | df["noga"].str.startswith("33") | (df["noga"] == "750000") | (df["noga"] == "960101") |  (df["noga"] == "960102") |  (df["noga"] == "960201") |  (df["noga"] == "960202") | (df["noga"] == "531000") |  (df["noga"] == "532000") |  (df["noga"] == "452001") |  (df["noga"] == "452002") |  (df["noga"] == "855300")

    df.loc[:, "office+-"] = (df["noga"] == "771100") |  (df["noga"] == "772100") |  (df["noga"] == "772900")

    df.loc[:, "office-++"] = (df["noga"] == "862100") |  (df["noga"] == "862200") |  (df["noga"] == "862300") 

    df.loc[:, "always-"] = df["noga"].str.startswith("55") | (df["noga"].str.startswith("56") & df["noga"] != "563002") | df["noga"].str.startswith("473")

    df.loc[:, "extended_night"] =  (df["noga"] == "932100") | df["noga"].str.startswith("471") |  (df["noga"] == "960401") |  (df["noga"] == "960402") 

    df.loc[:, "markets"] = (df["noga"] == "478100")

    df.loc[:, "theaters"] =  df["noga"].str.startswith("90")
    df.loc[:, "museums"] = df["noga"].str.startswith("91")
    df.loc[:, "cinemas"] = (df["noga"] == "591400")    
    df.loc[:, "casinos"] = (df["noga"] == "920000")
    df.loc[:, "night"] = (df["noga"] == "563002")
    df.loc[:, "culture_sport_hobby"] = (df["noga"] == "855200") | (df["noga"] == "855100")


    df.loc[df["always_open"] | df["casinos"]| df["night"], "open_0-3"] = 1
    df.loc[df["always-"] | df["theaters"], "open_0-3"] = 0.5
    
    df.loc[df["always_open"]| df["night"], "open_3-6"] = 1
    df.loc[df["always-"] | df["casinos"] | df["markets"], "open_3-6"] = 0.5   

    df.loc[df["always_open"]| df["markets"], "open_6-9"] = 1
    df.loc[df["always-"] | df["museums"] | df["extended_night"] | df["office++"] | df["office+-"] | df["office-++"] , "open_6-9"] = 0.5   

    df.loc[df["always_open"]| df["markets"] | df["office_hours"] | df["always-"] | df["office++"] | df["office+-"] | df["office-++"]| df["extended_night"] | df["culture_sport_hobby"] | df["museums"], "open_9-12"] = 1
    df.loc[df["theaters"] | df["cinemas"] , "open_9-12"] = 0.5   

    df.loc[df["always_open"]| df["always-"] | df["office_hours"] | df["office++"] | df["office+-"]| df["office-++"]| df["extended_night"] | df["culture_sport_hobby"] | df["museums"] | df["cinemas"], "open_12-15"] = 1
    df.loc[df["theaters"] | df["casinos"] | df["markets"] , "open_12-15"] = 0.5 

    df.loc[df["always_open"]| df["always-"] | df["office_hours"] | df["office++"] | df["office+-"]|  df["office-++"]| df["extended_night"] | df["culture_sport_hobby"] | df["museums"] | df["cinemas"], "open_15-18"] = 1
    df.loc[df["theaters"] | df["casinos"], "open_15-18"] = 0.5 

    df.loc[df["always_open"]| df["always-"] | df["office-++"] | df["extended_night"] | df["culture_sport_hobby"] | df["cinemas"]|df["theaters"]| df["casinos"] , "open_18-21"] = 1
    df.loc[df["office++"] | df["museums"], "open_18-21"] = 0.5 

    df.loc[df["always_open"]|df["theaters"]|df["cinemas"]|df["casinos"]| df["night"], "open_21-24"] = 1
    df.loc[df["always-"] | df["museums"] | df["extended_night"], "open_21-24"] = 0.5  

    del df["always_open"]
    del df["office_hours"]
    del df["office++"]
    del df["office+-"]
    del df["office-++"]
    del df["always-"]
    del df["extended_night"]
    del df["markets"]
    del df["theaters"]
    del df["museums"]
    del df["cinemas"]
    del df["casinos"]
    del  df["culture_sport_hobby"]

    df.loc[df["offers_outdoor"], "open_0-3"] = 1 
    df.loc[df["offers_outdoor"], "open_3-6"] = 1
    df.loc[df["offers_outdoor"], "open_6-9"] = 1
    df.loc[df["offers_outdoor"], "open_9-12"] = 1
    df.loc[df["offers_outdoor"], "open_12-15"] = 1
    df.loc[df["offers_outdoor"], "open_15-18"] = 1
    df.loc[df["offers_outdoor"], "open_18-21"] = 1
    df.loc[df["offers_outdoor"], "open_21-24"] = 1
    
    return df
     
