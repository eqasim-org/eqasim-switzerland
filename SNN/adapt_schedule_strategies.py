import geopandas as gpd
import pandas as pd
import numpy as np
from numpy.random import choice
from tqdm import tqdm
from sklearn.neighbors import KDTree
import shapely.geometry as geo

def detect_tours(trips):
    trips.loc[:, "tour_id"] = -1

    for indexi in trips.index:
        if trips.loc[indexi,"preceding_purpose"] in ["home"]:
           for j in range(indexi, max(trips.index)+1):
               trips.loc[j, "tour_id"] = int(trips.loc[j, "tour_id"] +1)

    return trips

def activities_to_trips(act, trips):

    act.loc[:, "preceding_activity_index"] = act["activity_index"] - 1
    prec = act[["person_id", "activity_index", "end_time", "purpose"]].rename(columns = {"end_time": "departure_time", "purpose": "preceding_purpose"})
    foll = act[["preceding_activity_index", "start_time", "purpose", "duration"]].rename(columns = {"start_time": "arrival_time", "purpose": "following_purpose", "duration": "activity_duration"})

    trips_new = prec.merge(foll, left_on = "activity_index", right_on = "preceding_activity_index", how = "inner")
    trips_new["mode"]    = trips["mode"].values.tolist()
    #trips_new["tour_id"] = trips["tour_id"].values.tolist()
    trips_new = trips_new.rename(columns = {"activity_index": "trip_index"})
    trips_new.loc[:, "trip_duration"] = trips_new["arrival_time"] - trips_new["departure_time"]
    trips_new.loc[:, "following_trip_index"] = trips_new["trip_index"] + 1

    return trips_new


def trips_to_activities(trips, act):

    act_initial = act.copy()

    trips.loc[:, "following_trip_index"] = trips["trip_index"] + 1

    act = pd.merge(trips[["person_id", "following_trip_index", "arrival_time", "following_purpose"]],
                   trips[["trip_index", "departure_time"]], 
                   left_on = "following_trip_index", right_on = "trip_index", 
                   how = "inner")
    act.columns = ["person_id", "following_trip_index", "start_time", "purpose", "trip_index", "end_time"]
    del act["following_trip_index"]

    act = act.rename(columns = {"trip_index": "activity_index"})

    row0 = act.head(1).copy()
    row0.loc[:, "end_time"] = trips.head(1)["departure_time"].values[0]
    row0.loc[:, "start_time"] = 0
    row0.loc[:, "purpose"] = trips.head(1)["preceding_purpose"].values[0]
    row0.loc[:, "activity_index"] = 0
    act = pd.concat([row0, act])

    rowEnd = row0
    rowEnd["start_time"] = trips.tail(1)["arrival_time"].values[0]
    rowEnd["end_time"] = 30*3600
    rowEnd["purpose"] = trips.tail(1)["following_purpose"].values[0]
    rowEnd["activity_index"] = len(act)

    act = pd.concat([act, rowEnd])

    act = act.sort_values(by = "start_time")
    act.loc[:, "duration"] = act["end_time"] - act["start_time"]
    act.loc[:, "is_last"] = act["activity_index"] == np.max(act["activity_index"])

    act = act[["person_id", "activity_index", "start_time", "end_time", "duration", "purpose", "is_last"]]

    if len(act) == 0:
        print("Only one trip during that day ie loop trip around home")
        act = act_initial

    assert len(act) != 0

    return act


def fix_act_before(act):
    act["start_time"] = act["start_time"].fillna(0)
    act["end_time"]   = act["end_time"].fillna(30*3600)
    act["duration"]   = act["end_time"] - act["start_time"]
    return act


def strategy0(trips, act):

    return trips, act


def strategy1(trips, act):
    act = fix_act_before(act)
    
    # First, change all work activities to wfh
    act.loc[act["purpose"]== "work", "purpose"] = "work_from_home"
    trips.loc[trips["preceding_purpose"]== "work", "preceding_purpose"] = "work_from_home"
    trips.loc[trips["following_purpose"]== "work", "following_purpose"] = "work_from_home"

    # Identify tours containing work-from-home
    trips = detect_tours(trips)
    trips["tour_id"] = trips["tour_id"].astype(int)
    tours_wfh = np.unique(trips[trips["preceding_purpose"]=="work_from_home"]["tour_id"].values)
    for tour_id in range(0, np.max(trips["tour_id"])+1):
        trips_current_tour = trips[trips["tour_id"]==tour_id]
        if tour_id in tours_wfh:
            first_trip_index = trips_current_tour.head(1)["trip_index"].values[0]
            last_trip_index = trips_current_tour.tail(1)["trip_index"].values[0] 

            trips_current_tour = trips_current_tour[~trips_current_tour["trip_index"].isin(range(first_trip_index+1,
                                                                                                last_trip_index))]
            trips_current_tour.head(1)["following_purpose"] = "work_from_home"
            trips_current_tour.tail(1)["preceding_purpose"] = "work_from_home"
        
        if tour_id == 0:
            new_trips = trips_current_tour
        else:
            new_trips = pd.concat([new_trips, trips_current_tour])

    trips = new_trips.sort_values(by = "departure_time")
    trips.loc[:, "trip_index"] = range(len(trips))  

    act = trips_to_activities(trips, act)

    wfh_acts_ind = act[act["purpose"]=="work_from_home"]["activity_index"].values.tolist()

    # For the first heuristics, we want to delete
    if len(wfh_acts_ind) >= 2:

        # merge consecutive wfh activities
        wfh_acts_ind = act[act["purpose"]=="work_from_home"]["activity_index"].values.tolist()
        consecutive = [i for i in wfh_acts_ind if i+1 in wfh_acts_ind]

        for act_ind in consecutive:
            # merge activity i with i+1
            act.loc[act["activity_index"]==act_ind+1, "start_time"] = act[act["activity_index"]==act_ind]["start_time"].values[0]
            act = act[act["activity_index"]!=act_ind]
            trips = trips[trips["trip_index"]!=act_ind]

        nb_act = len(act)
        act["activity_index"] = list(range(nb_act))
        act["duration"]   = act["end_time"] - act["start_time"]

        nb_trips = len(trips)
        trips["trip_index"] = list(range(nb_trips))

    if np.all([act["purpose"].values.tolist()[i] in ["home", "work_from_home"] for i in range(len(act))]) :
        act = act[act["activity_index"]==0]
        act["end_time"] = 30*3600
        act["duration"] = 30*3600
        act["purpose"]  = "home"
        act["is_last"]  = True

    elif np.any([act["purpose"].values.tolist()[i] in ["work_from_home"] for i in range(len(act))]) :

        while np.any([act["purpose"].values.tolist()[i] in ["work_from_home"] for i in range(len(act))]) :
            id_first_wfh = np.min(act[act["purpose"]== "work_from_home"]["activity_index"])
            id_home_before = id_first_wfh - 1
            id_home_after = id_first_wfh + 1

            act.loc[act["activity_index"]==id_home_before, "end_time"] = act[act["activity_index"]==id_home_after]["end_time"].values.tolist()[0]
            act = act[act["activity_index"] != id_first_wfh]
            act = act[act["activity_index"] != id_home_after]
            trips = trips[trips["trip_index"]!= id_home_before]
            trips = trips[trips["trip_index"]!= id_first_wfh]

            act.loc[:, "duration"] = act["end_time"] - act["start_time"]
            act.loc[:, "activity_index"] = range(len(act))
            act.loc[:, "is_last"] = act["activity_index"] == len(act)-1

            trips.loc[:, "trip_index"] = range(len(trips))
        
    trips = activities_to_trips(act, trips)

    act = act.reset_index()
    del act["index"]

    return trips, act


def strategy3(trips, act):
    act = fix_act_before(act)
    # First, change all work activities to wfh
    act.loc[act["purpose"]== "work", "purpose"] = "work_from_home"
    trips.loc[trips["preceding_purpose"]== "work", "preceding_purpose"] = "work_from_home"
    trips.loc[trips["following_purpose"]== "work", "following_purpose"] = "work_from_home"

    wfh_acts_ind = act[act["purpose"]=="work_from_home"]["activity_index"].values.tolist()

    # For the first heuristics, we want to delete
    if len(wfh_acts_ind) >= 2:

        # merge consecutive wfh activities
        wfh_acts_ind = act[act["purpose"]=="work_from_home"]["activity_index"].values.tolist()
        consecutive = [i for i in wfh_acts_ind if i+1 in wfh_acts_ind]

        for act_ind in consecutive:
            # merge activity i with i+1
            act.loc[act["activity_index"]==act_ind+1, "start_time"] = act[act["activity_index"]==act_ind]["start_time"].values[0]
            act = act[act["activity_index"]!=act_ind]
            trips = trips[trips["trip_index"]!=act_ind]

        nb_act = len(act)
        act["activity_index"] = list(range(nb_act))
        act["duration"]   = act["end_time"] - act["start_time"]

        nb_trips = len(trips)
        trips["trip_index"] = list(range(nb_trips))

    trips = activities_to_trips(act, trips)

    return trips, act