import numpy as np
import pandas as pd
from numpy import random
from tqdm import tqdm

import SNN.adapt_schedule_strategies as ass
from SNN.SNNmatching import SNNmatching_match

import data.constants as c

"""
This stage gathers cache from the activities and trips stages, and adapts them as much as possible according to the various SNN strategies
"""


def configure(context):
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.trips")
    
    context.config("run_snn")
    if context.config("run_snn"):
        context.config("snn_heuristic")
        context.config("snn_day")
        context.stage("synthesis.population.SNN_population")


def processtimeusetrips(dftrips, timeuseplus_days):

    dftrips = dftrips.rename(columns = {"trip_id": "trip_index", "start_time": "departure_time",
                                                            "end_time": "arrival_time"})

    cleaned = {}

    for pid in list(set(dftrips["person_day_id"].values.tolist())):
        df = dftrips[dftrips["person_day_id"]==pid]
        df.loc[df["mode"] == "mpt", "mode"] = "car"
        df.loc[df["mode"] == "other", "mode"] = "car"

        cleaned[pid] = df

        # mode for home-home trips should be walk... 
        df.loc[(df["preceding_purpose"]=="home") & (df["following_purpose"]=="home"), "mode"] = "walk"

        # activities
        df.loc[:, "previous_trip_index"] = df.loc[:, "trip_index"] - 1

        act = pd.merge(
            df, df, left_on=["person_day_id", "previous_trip_index"], right_on=["person_day_id", "trip_index"],
            suffixes=["_following_trip", "_previous_trip"], how="left"
        )

        act.loc[:, "start_time"] = act.loc[:, "arrival_time_previous_trip"]
        act.loc[:, "end_time"] = act.loc[:, "departure_time_following_trip"]
        act.loc[:, "purpose"] = act.loc[:, "following_purpose_previous_trip"]
        act["start_time"] = act["start_time"].fillna(0)
        act["purpose"] = act["purpose"].fillna(df.iloc[0]["preceding_purpose"])
        act = act.sort_values(by = "start_time", ascending=True)
        act.loc[:, "activity_index"] = range(len(act))
        act = act[["person_day_id", "activity_index", "start_time", "end_time", "purpose"]]

        df_last = act.sort_values(by=["person_day_id", "activity_index"]).groupby("person_day_id").last().reset_index()
        df_last.loc[:, "purpose"] = df.tail(1)["following_purpose"].values.tolist()[0]
        df_last.loc[:, "start_time"] = df.tail(1)["arrival_time"].values.tolist()[0]
        df_last.loc[:, "end_time"] = 24*3600
        df_last.loc[:, "activity_index"] += 1
        act = pd.concat([act, df_last])

        act.loc[:, "duration"] = act["end_time"] - act["start_time"]
        act["duration"] = act["duration"] / 3600

        durations = act.groupby(["purpose"])["duration"].sum().reset_index().set_index("purpose").T.to_dict('list')

        # If more than 2 hours of "work" activity: the person spent quite some time in the office -> no WFH day
        if "work" in durations.keys():
            if durations["work"][0] > 2:
                timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
                #print("  This day will be deleted because too much time is spent in the office")

        # Similarly, days with more than 3 hours of education shouldn't correspond to the profile of peope we are targeting.
        if "education" in durations.keys():
            if durations["education"][0] > 3:
                timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
                #print("  This day will be deleted because too much time is spent at school")

        if len(df) > 10:
            timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
            #print("  This day will be deleted because there are unrealistically many trips")

        countacts = act.groupby(["purpose"])["duration"].count().reset_index().set_index("purpose").T.to_dict('list')
        #print(countacts)

        if act.head(1)["purpose"].values.tolist()[0] != "home":
            timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
            #print("  zut")

        if act.tail(1)["purpose"].values.tolist()[0] != "home":
            timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
            #print("  zut")

        if "other" in countacts.keys():
            if countacts["other"][0]  >= 3:
                timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
                #print("  This day will be deleted because there are too many errands")

        if "home" in countacts.keys():
            if countacts["home"][0]  >= 4:
                timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
                #print("  This day will be deleted because there are too many loop trips")

        if "education" in countacts.keys():
            if countacts["education"][0]  >= 3:
                timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
                #print("  This day will be deleted because there are too many education activities")

        if "work" in countacts.keys():
            if countacts["work"][0]  >= 3:
                timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
                #print("  This day will be deleted because there are too many work activities")

        if "shop" in countacts.keys():
            if countacts["shop"][0]  >= 3:
                timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
                #print("  This day will be deleted because there are too many shop activities")

        if "leisure" in countacts.keys():
            if countacts["leisure"][0]  >= 3:
                timeuseplus_days = timeuseplus_days[timeuseplus_days["person_day_id"] != pid]
                #print("  This day will be deleted because there are too many leisure activities")

    return pd.concat(cleaned.values()), timeuseplus_days


def trips_to_activites(df_trips, df_persons):
    df_trips.loc[:, "previous_trip_index"] = df_trips.loc[:, "trip_index"] - 1

    df_activities = pd.merge(
        df_trips, df_trips, left_on=["person_id", "previous_trip_index"], right_on=["person_id", "trip_index"],
        suffixes=["_following_trip", "_previous_trip"], how="left"
    )

    df_activities.loc[:, "start_time"] = df_activities.loc[:, "arrival_time_previous_trip"]
    df_activities.loc[:, "end_time"] = df_activities.loc[:, "departure_time_following_trip"]
    df_activities.loc[:, "purpose"] = df_activities.loc[:, "following_purpose_previous_trip"]
    df_activities.loc[:, "activity_index"] = df_activities.loc[:, "trip_index_following_trip"]
    df_activities.loc[:, "is_last"] = False

    # We assume that the plans start at home
    df_activities.loc[:, "purpose"] = df_activities.loc[:, "purpose"].fillna("home")

    # We're still missing the last activity in the chain.
    df_last = df_activities.sort_values(by=["person_id", "activity_index"]).groupby("person_id").last().reset_index()
    df_last.loc[:, "purpose"] = df_last.loc[:, "following_purpose_following_trip"]
    df_last.loc[:, "start_time"] = df_last.loc[:, "arrival_time_following_trip"]
    df_last.loc[:, "end_time"] = np.nan
    df_last.loc[:, "activity_index"] += 1
    df_last.loc[:, "is_last"] = True

    df_activities = pd.concat([df_activities, df_last])

    # We're still missing activities for people who don't have a any trips
    missing_ids = set(np.unique(df_persons["person_id"])) - set(np.unique(df_activities["person_id"]))
    print("Found %d persons without activities" % len(missing_ids))

    df_missing = pd.DataFrame.from_records([
        (person_id, 1, "home", True) for person_id in missing_ids
    ], columns=["person_id", "activity_index", "purpose", "is_last"])

    df_activities = pd.concat([df_activities, df_missing], sort=True)
    assert (len(np.unique(df_persons["person_id"])) == len(np.unique(df_activities["person_id"])))

    # Some cleanup
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"])
    df_activities.loc[:, "duration"] = df_activities.loc[:, "end_time"] - df_activities.loc[:, "start_time"]

    df_activities = df_activities[[
        "person_id", "activity_index", "start_time", "end_time", "duration", "purpose", "is_last"
    ]]

    return df_activities


def execute(context):

    df_activities = context.stage("synthesis.population.activities")
    df_trips      = context.stage("synthesis.population.trips")

    if context.config("run_snn"):
        df_persons    = context.stage("synthesis.population.SNN_population")
        strategy = context.config("snn_heuristic")

        if strategy in [0,1,2,3]:

            # For each strategy, get the corresponding function
            if strategy == 0:
                strategy_adapt = ass.strategy0

            if strategy == 1:
                strategy_adapt = ass.strategy1

            if strategy == 3:
                strategy_adapt = ass.strategy3

            if strategy == 2:
                strategy_adapt = ass.strategy3

            agents_wfh = df_persons[df_persons["wfh_today"]]["person_id"].values.tolist()

            # Isolate agents whose plans must be adapted
            trips_no_wfh = df_trips[~df_trips["person_id"].isin(agents_wfh)]
            acts_no_wfh  = df_activities[~df_activities["person_id"].isin(agents_wfh)]

            trips_wfh = df_trips[df_trips["person_id"].isin(agents_wfh)]
            acts_wfh  = df_activities[df_activities["person_id"].isin(agents_wfh)]

            new_trips_wfh = {}
            new_acts_wfh  = {}

            trips_wfh_by_person_id = trips_wfh.groupby("person_id")
            acts_wfh_by_person_id  = acts_wfh.groupby("person_id")

            # For each agent working from home, adapt their activity and trip plan.
            # TODO could be parallelized but I don't know how :)

            for agent_id in tqdm(agents_wfh, desc = "Adapting plans for WFH agents"):

                activities_indiv = acts_wfh_by_person_id.get_group(agent_id)   

                if agent_id in trips_wfh_by_person_id.groups:
                    trips_indiv                   = trips_wfh_by_person_id.get_group(agent_id)  
                    trips_indiv, activities_indiv = strategy_adapt(trips_indiv, activities_indiv)

                    if len(activities_indiv) == 1:
                        trips_indiv = trips_indiv[0:0] 

                    new_trips_wfh[agent_id] = trips_indiv
                    new_acts_wfh[agent_id]  = activities_indiv

            new_trips_wfh = pd.concat(list(new_trips_wfh.values())).drop_duplicates()
            new_acts_wfh  = pd.concat(list(new_acts_wfh.values())).drop_duplicates()

            df_trips       = pd.concat([trips_no_wfh, new_trips_wfh])
            df_activities  = pd.concat([acts_no_wfh, new_acts_wfh])

            missing_ids = set(np.unique(df_persons["person_id"])) - set(np.unique(df_activities["person_id"]))

            df_missing = pd.DataFrame.from_records([
                (person_id, 1, "home", True) for person_id in missing_ids
                ], columns=["person_id", "activity_index", "purpose", "is_last"])
        
            df_activities = pd.concat([df_activities, df_missing], sort=True)    

        elif strategy == 4:
            print("Strategy 4 activated")
            agents_wfh = df_persons[df_persons["wfh_today"]]["person_id"].values.tolist()

            trips_no_wfh = df_trips[~df_trips["person_id"].isin(agents_wfh)]
            acts_no_wfh  = df_activities[~df_activities["person_id"].isin(agents_wfh)]

            trips_wfh = df_trips[df_trips["person_id"].isin(agents_wfh)]
            acts_wfh  = df_activities[df_activities["person_id"].isin(agents_wfh)]

            timeuseplus_persons = pd.read_csv("SNN/timeuse+_participants.csv")
            timeuseplus_days = pd.read_csv("SNN/timeuse+_link_ids.csv")

            timeuseplus_trips =  pd.read_csv("SNN/timeuse+_trips.csv")
            timeuseplus_trips, timeuseplus_days = processtimeusetrips(timeuseplus_trips, timeuseplus_days)

            days = context.config("snn_day").split("-")
            timeuseplus_days = timeuseplus_days[timeuseplus_days["date_d"].str.lower().isin(days)]
            timeuseplus_persons = timeuseplus_persons[timeuseplus_persons["participant_id"].isin(timeuseplus_days["participant_id"])]

            statmatch = SNNmatching_match(context, df_persons, timeuseplus_persons, timeuseplus_days, agents_wfh)
            print(statmatch)

            df_persons = statmatch[["person_id", "person_day_id"]]

            new_trips_wfh = pd.merge(df_persons, timeuseplus_trips, on = "person_day_id")
            new_acts_wfh  = trips_to_activites(new_trips_wfh, df_persons)    

            df_trips       = pd.concat([trips_no_wfh, new_trips_wfh])
            df_activities  = pd.concat([acts_no_wfh, new_acts_wfh])

            print(df_trips[df_trips["person_id"] == 8129363][["preceding_purpose", "following_purpose", "mode"]])

            missing_ids = set(np.unique(df_persons["person_id"])) - set(np.unique(df_activities["person_id"]))

            df_missing = pd.DataFrame.from_records([
                (person_id, 1, "home", True) for person_id in missing_ids
                ], columns=["person_id", "activity_index", "purpose", "is_last"])
        
            df_activities = pd.concat([df_activities, df_missing], sort=True)  

        else:
            exit()

    return df_activities, df_trips


