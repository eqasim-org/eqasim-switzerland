import numpy as np
import pandas as pd
from tqdm import tqdm


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")


def execute(context):
    df_persons    = context.stage("data.microcensus.persons")
    df_trips      = context.stage("data.microcensus.trips")
    
    nb_max_activities  = np.max(df_trips.groupby(["person_id"])["trip_id"].count())
    activities_columns = ["activity"+str(i) for i in range(nb_max_activities + 1)]

    act_df = df_trips.groupby(['person_id'], as_index=False).nth(0).copy()[
            ['person_id']    ]
    
    act_df.loc[:, "activity0"] = ["home" for k in range(len(act_df))]
            
    for i in tqdm(range(nb_max_activities+1)):  
        colname   = 'activity' + str(i+1)  
        trips_act = df_trips.groupby(['person_id'], as_index=False).nth(i).copy()[
            ['person_id', 'purpose']
        ].rename(columns={'purpose': colname}) 
        act_df    = pd.merge(act_df, trips_act, on='person_id', how='left')

    act_df = act_df.fillna("stop")

    act_df["activity_chain"] = act_df[activities_columns].agg('-'.join, axis=1)
    act_df["activity_chain"] = [ca.replace('-stop', '') for ca in act_df["activity_chain"]]
    act_df                   = act_df[act_df["activity"+str(nb_max_activities + 1)] == "stop"]
    
    act_df.loc[:, "category_activities"] = 2
    act_df.loc[act_df.activity_chain == "home", "category_activities"] = 0
    act_df.loc[:, "has_educ"] = [
        np.isin("education",s.split("-")) for s in
        act_df["activity_chain"]]
    act_df.loc[
        act_df.has_educ == True, 
            "category_activities"] = 1
    
    del act_df["has_educ"]
    
    act_df = act_df[["person_id", "category_activities", "activity_chain"]]
    
    df_persons = pd.merge(df_persons, act_df, on = "person_id", how = "left")
    df_persons = df_persons.fillna({"category_activities": 0,
                                    "activity_chain": "home"})
    

    df_persons =  df_persons[(df_persons["age"] == 6) |  (df_persons["age"] == 7)]  
    return df_persons
    
    
    
    
    
    
    
