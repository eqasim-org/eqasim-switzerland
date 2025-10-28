import numpy as np
import pandas as pd
from tqdm import tqdm


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")


def execute(context):
    df_persons    = context.stage("data.microcensus.persons")
    df_trips      = context.stage("data.microcensus.trips")[0]
    
    nb_max_activities  = np.max(df_trips.groupby(["person_id"])["trip_id"].count())
    activities_columns = ["activity" + str(i) for i in range(nb_max_activities + 1)]
    trips_columns      = ["trip_mode" + str(i) for i in range(1, nb_max_activities + 1)]

    act_df = df_trips.groupby(["person_id"], as_index=False).nth(0).copy()[["person_id", "origin_purpose"]]
    
    act_df.loc[:, "activity0"] = act_df["origin_purpose"]

    del act_df["origin_purpose"]
            
    for i in tqdm(range(nb_max_activities+1)):  
        colname   = "activity" + str(i+1)  
        trips_act = df_trips.groupby(["person_id"], as_index=False).nth(i).copy()[
            ["person_id", "purpose", "mode"]
        ].rename(columns={"purpose": colname, "mode": "trip_mode"+str(i+1)}) 
        act_df    = pd.merge(act_df, trips_act, on="person_id", how="left")

    act_df = act_df.fillna("stop")

    act_df["activity_chain"] = act_df[activities_columns].agg("-".join, axis=1)
    act_df["activity_chain"] = [ca.replace("-stop", "") for ca in act_df["activity_chain"]]

    act_df["mode_chain"] = act_df[trips_columns].agg("-".join, axis=1)
    act_df["mode_chain"] = [ca.replace("-stop", "") for ca in act_df["mode_chain"]]
    
    act_df = act_df[["person_id", "activity_chain", "mode_chain"]]
    
    df_persons = pd.merge(df_persons, act_df, on = "person_id", how = "left")
    df_persons = df_persons.fillna({"activity_chain": "home", "mode_chain": "no trip"})
    
    return df_persons[["person_id", "person_weight", "weekend", "workday", "day", "activity_chain", "mode_chain"]]
