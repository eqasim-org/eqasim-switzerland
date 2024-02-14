import numpy as np
import pandas as pd
from tqdm import tqdm

def configure(context):
    context.config("uk_data_path")
    context.config("weekend_scenario", False)
    context.config("specific_weekend_scenario", "all") # options are "all", "saturday", "sunday"

def execute(context):
    data_path = context.config("uk_data_path")
    
    is_weekend_scenario             = context.config("weekend_scenario")
    specific_weekend_scenario       = context.config("specific_weekend_scenario")
    
    ## Days 
    days             = pd.read_csv(data_path + "day_special_2002-2019_protect.tab", delimiter = '\t')
    days             = days[["DayID", "IndividualID", "TravelWeekDay_B02ID", "TravelMonth_B01ID", "TravelDay", "TravelYear", "TravelDayType_B01ID", "TravelDayTypeOld_B01ID"]]
    days.columns     = ["person_day_id", "person_id", "weekend_or_not", "month", "day", "year", "holidays_from_2008", "holidays_before_2008"]
    
    dic_from_2008 = {1:"Week-end", 
                  2: "bank holiday", 
                  3: "school time", 
                  4: "school holiday", 
                  -8: "NA", 
                  -10: "NA"}

    dic_to_2008 = {1:"Week-end",
                2: "school time",
                3: "school holiday", 
                -8: "NA", 
                -10: "NA"}

    days["holidays_from_2008"] = [dic_from_2008[k] for k in days["holidays_from_2008"]]
    days["holidays_before_2008"] = [dic_to_2008[k] for k in days["holidays_before_2008"]]
    
    days = days[((days["year"]<2008) &  (days["holidays_before_2008"] == "school time")) |     ((days["year"]>=2008) & (days["holidays_from_2008"] == "school time"))]

    days["weekend"]  = days["weekend_or_not"] != 1
    days["saturday"] = days["weekend_or_not"] == 2
    days["sunday"]   = days["weekend_or_not"] == 3 
    days["date"]     = ["-".join([str(y), str(m), str(d)]) 
                        for y,m,d in list(zip(days["year"].values.tolist(), 
                                              days["month"].values.tolist(), 
                                              days["day"].values.tolist(),
                                              )
                                          )
                        ]
    
    if not is_weekend_scenario:
        days = days[~ days["weekend"]]
    else:
        if specific_weekend_scenario =="all":
            days = days[days["weekend"]]
        elif specific_weekend_scenario == "saturday":
            days = days[days["saturday"]]
        elif specific_weekend_scenario == "sunday":
            days = days[days["sunday"]]
    
    del days["weekend_or_not"]
    del days["month"]
    del days["day"]
    del days["year"]
    
    
    ## Households
    households = pd.read_csv(data_path + "household_special_2002-2019_protect.tab", 
                             delimiter="\t",  encoding = "unicode_escape")

    households = households[["HouseholdID", "HHIncome2002_B01ID", 
                             "HHoldAreaType1_B02ID",  "HHoldNumPeople", 
                             "NumBike", "NumCarVan",
                             "W2", "W3"]]

    households.columns = ["hhl_id", "hhl_income_26cat",
                          "settlement_category", "nb_hhl_members", 
                          "number_bikes", "number_cars", 
                          "weight_diary_sample", "weight_interview_sample"]


    ## Persons
    persons = pd.read_csv(data_path + "individual_special_2002-2019_protect.tab",
                          delimiter="\t",  encoding = "unicode_escape")

    persons = persons[["HouseholdID", "IndividualID", 
                       "Age", "Sex_B01ID", "MaritalS_B01ID"]]

    persons.columns = ["hhl_id", "person_id",
                       "age", "sex", "Marital_status"]
    
    persons = persons[persons["age"] <= 7]
    
    
    ## Trips
    trips = pd.read_csv(data_path + "trip_special_2002-2019_protect.tab", 
                        delimiter="\t",  encoding = "unicode_escape")

    trips = trips[["HouseholdID", "IndividualID", 
                   "DayID", "TripID", 
                   "MainMode_B04ID", 
                   "TripPurpFrom_B01ID", "TripPurpTo_B01ID", 
                   "TripStart", "TripEnd", 
                   "TripTotalTime", "TripDisIncSW", 
                   "W5"]]

    trips.columns = ["hhl_id", "person_id", 
                     "person_day_id", "trip_id", 
                     "mode", 
                     "preceding_purpose", "following_purpose",
                     "start_time", "end_time",
                     "trip_duration", "trip_distance", 
                     "weight_trip"]
    
    trips = pd.merge(trips, persons[["person_id", "age"]], on = "person_id", how = "left")
    trips = trips[trips["age"] <= 7]

    trips["is_car_passenger"] = trips["mode"] == 4
    car_passengers = np.unique(trips[trips["is_car_passenger"]]["person_id"].values.tolist())
    print("Number of car passengers: " + str(len(car_passengers)))
    print("Total number: " + str(len(np.unique(trips["person_id"]))))
    
    dic_purpose            = {1: "work", #"Work", 
                              2: "work", #"Business", 
                              3: "education", #"Education", 
                              4: "shop", #"Food shopping", 
                              5: "shop", #"Other shopping", 
                              6: "other", #"Medical", 
                              7: "leisure", #"Restaurant/bar alone", 
                              8: "other", #"Other personal", 
                              9: "leisure", #"Restaurant-bar with friends", 
                              10: "leisure", #"Visit friends", 
                              11: "other", #"Other social", 
                              12: "leisure", #"Entertainment / public", 
                              13: "leisure", #"Active sport", 
                              14: "other", #"holiday", 
                              15: "other", #"day trip/just walk", 
                              16: "other", #"Other non escort", 
                              17: "other", #"Escort home", 
                              18: "other", #"Escort work", 
                              19: "other", #"Escort business", 
                              20: "other", #"Escort education", 
                              21: "shop", #"Escort shopping/personal", 
                              22: "other", #"Other escort", 
                              23: "home", #"Home",
                              -8: "other"}

    L = trips["following_purpose"].values.tolist()
    L = [l[:2] if type(l)!= int else l for l in L]
    L = [int(l) for l in L]
    L = [min(l, 23) for l in L]

    M = trips["preceding_purpose"].values.tolist()
    M = [l[:2] if type(l)!= int else l for l in M]
    M = [int(l) for l in M]
    M = [min(l, 23) for l in M]

    trips["following_purpose"] = [dic_purpose[k] for k in L]
    trips["preceding_purpose"] = [dic_purpose[k] for k in M]
    
    trips["start_time"] = trips["start_time"].replace({' ': -10})
    trips["end_time"]   = trips["end_time"].replace({' ': -10})
    trips["start_time"] = trips["start_time"].astype(int)
    trips["end_time"]   = trips["end_time"].astype(int)
    
    empty_starts = set(trips[trips["start_time"] < 0]["person_day_id"].values.tolist())
    empty_ends   = set(trips[trips["end_time"] < 0]["person_day_id"].values.tolist())
    empty_starts_ends = empty_starts.union(empty_ends)
    empty_starts_ends = list(set(empty_starts_ends))
    
    dic_mode  = {1: "walk", #"Walk", 
                 2: "bike", #"Bicycle", 
                 3: "car", #"Car driver", 
                 4: "car", #"Car passenger",
                 5: "car", #"Motorcycle", 
                 6: "car", #"Other private", 
                 7: "pt", 
                 8:"pt", 
                 9:"pt", 
                 10:"pt", 
                 11: "pt", 
                 12: "pt", #"taxi", 
                 13: "pt", #"other pt", 
                 -8: "unknown", #"NA" 
                 }
    trips["mode"] = [dic_mode[k] for k in trips["mode"]]
    
    nb_max_activities  = np.max(trips.groupby(["person_day_id"])["trip_id"].count())
    activities_columns = ["activity"+str(i) for i in range(nb_max_activities + 1)]
    
    trips = trips.sort_values(["person_day_id", "start_time"])
    trips.loc[:, "new_trip_id"] = 1
    print(trips.columns)
    for i in tqdm(range(len(trips))):
        if i>0 and trips.iloc[i, 2] == trips.iloc[i - 1, 2]:
            trips.iloc[i,-1] = trips.iloc[i - 1, -1]+1
            
    trips.loc[:, "trip_id"] = trips["new_trip_id"]    
    trips = trips.sort_values(["person_day_id", "trip_id"])


    act_df = trips.groupby(['person_day_id'], as_index=False).nth(0).copy()[
        ['person_day_id', 'preceding_purpose', 
         "person_id", "hhl_id", "weight_trip"]
        ]
        
    act_df.rename(columns = {"preceding_purpose": "activity0"}, inplace = True)

    for i in tqdm(range(nb_max_activities+1)):  
        colname   = 'activity' + str(i+1)  
        trips_act = trips.groupby(['person_day_id'], as_index=False).nth(i).copy()[
            ['person_day_id', 'following_purpose']
            ].rename(columns={'following_purpose': colname}) 
        act_df    = pd.merge(act_df, trips_act, on='person_day_id', how='left')

    act_df = act_df.fillna("stop")

    act_df["activity_chain"] = act_df[activities_columns].agg('-'.join, axis=1)
    act_df["activity_chain"] = [c.replace('-stop', '') for c in act_df["activity_chain"]]
    act_df                   = act_df[act_df["activity"+str(nb_max_activities + 1)] == "stop"]
    
    act_df.loc[:, "category_activities"] = 2
    act_df.loc[act_df.activity_chain == "home", "category_activities"] = 0
    
    act_df.loc[:, "has_educ"] = [
        np.isin("education",s.split("-")) for s in
        act_df["activity_chain"]]
    act_df.loc[
        act_df.has_educ == True, 
            "category_activities"] = 1
    
    act_df.loc[:, "has_work"] = [
        np.isin("work",s.split("-")) for s in
        act_df["activity_chain"]]
    
    act_df.loc[:, "starts_from_home"] = [
        s.split("-")[0] == "home" for s in
        act_df["activity_chain"]]
    
    act_df.loc[:, "returns_home"] = [
        s.split("-")[-1] == "home" for s in
        act_df["activity_chain"]]
    
    listin_time = act_df["person_day_id"].isin(empty_starts_ends)
    act_df.loc[:, "start_end_issues"] = listin_time
    
    print(act_df.head(10))
    
    del act_df["has_educ"]
    
    act_df = act_df[["person_day_id", "category_activities", "activity_chain",
                     "starts_from_home", "returns_home", "start_end_issues", "has_work"]]
    
    ac_exp =  pd.merge(days, act_df, on = "person_day_id", how = "left")
    
    ac_exp["start_end_issues"] = ac_exp["start_end_issues"].fillna(False)
    ac_exp["has_work"]         = ac_exp["has_work"].fillna(False)
    
    print(ac_exp.columns)
    #print(list(set(ac_exp["has_work"].values.tolist())))
    
    print("Before: ", len(ac_exp))
    ac_exp = ac_exp[(ac_exp["starts_from_home"]) & (ac_exp["returns_home"]) 
                    & ~(ac_exp["start_end_issues"]) & ~(ac_exp["has_work"])]
    del ac_exp["starts_from_home"]
    del ac_exp["returns_home"]
    del ac_exp["start_end_issues"]    
    del ac_exp["has_work"]
    
    print("After: ", len(ac_exp))
    
    ac_exp = ac_exp.fillna({"category_activities": 0, "activity_chain": "home"})
    print("New: " + str(len(ac_exp)))
    
    
    days_exp = pd.merge(ac_exp, persons, on = "person_id")
    print("New: " + str(len(days_exp)))

    days_exp2 = pd.merge(days_exp, households, on = "hhl_id")
    print("New: " + str(len(days_exp2)))

    print(days_exp2.columns)

    days_exp2["is_car_passenger"] = days_exp2["person_id"].isin(car_passengers)

    source = days_exp2[days_exp2["age"] <= 6]

    source["person_id_source"] = ["UK_" + str(k) for k in range(len(source))]
    
    ## Now process trips
    trips = trips[trips["person_day_id"].isin(ac_exp["person_day_id"])]
    
    #print(trips[["person_day_id", "start_time", "end_time"]].head(10))
    
    
    trips = trips[["person_day_id", "trip_id", "start_time",
                   "end_time", "mode", "preceding_purpose", "following_purpose",
                   ]]
    
    source_ids = source[["person_day_id", "person_id_source"]]
    trips = pd.merge(trips, source_ids, on = "person_day_id", how = "inner")
    
    trips.loc[:, "previous_trip_id"] = trips["trip_id"] -1
    df_durations = pd.merge(
        trips[["person_day_id", "trip_id", "end_time"]],
        trips[["person_day_id", "previous_trip_id", "start_time"]],
        left_on = ["person_day_id", "trip_id"],
        right_on = ["person_day_id", "previous_trip_id"])
    
    df_durations.loc[:, "activity_duration"] = df_durations["start_time"] - df_durations["end_time"]
    
    trips = pd.merge(
        trips, df_durations[["person_day_id", "trip_id", "activity_duration"]],
        on = ["person_day_id", "trip_id"], how = "left"
    )
    
    trips = trips.rename(columns = {"person_id_source": "person_id",
                                    "start_time"      : "departure_time",
                                    "end_time"        : "arrival_time",
                                    })  
    
    trips = trips[["person_id", "person_day_id", "trip_id", "departure_time", "arrival_time",
                   "mode", "preceding_purpose", "following_purpose", "activity_duration"]]
    
    print(trips[trips["person_id"] == "UK_44807"])
    #exit()


    source["sex"] = [s-1 for s in source["sex"].values]
    source["marital_status"] = 0
    source.loc[source["Marital_status"] == 2, "marital_status"] = 1
    source.loc[source["Marital_status"] == 3, "marital_status"] = 2
    source.loc[source["Marital_status"] == 4, "marital_status"] = 2
    source.loc[source["Marital_status"] == 5, "marital_status"] = 2
    source.loc[source["Marital_status"] == 6, "marital_status"] = 1
    source.loc[source["Marital_status"] == 7, "marital_status"] = 2
    source.loc[source["Marital_status"] == 8, "marital_status"] = 2
    source.loc[source["Marital_status"] == 9, "marital_status"] = 2
    del source["Marital_status"]

    source["municipality_type"] = "urban"
    source.loc[source["settlement_category"] == 5, "municipality_type"] = "suburban"
    source.loc[source["settlement_category"] == 6, "municipality_type"] = "suburban"
    source.loc[source["settlement_category"] == 7, "municipality_type"] = "rural"
    del source["settlement_category"]

    source["number_of_cars_class"] = 0
    source["number_cars"] = source["number_cars"].replace(' ', 0)
    source["number_cars"] = source["number_cars"].astype(int)
    source.loc[source["number_cars"] > 0, "number_of_cars_class"] = np.minimum(3, source["number_cars"])
    source["number_of_cars"] = source["number_cars"]
    del source["number_cars"]

    source["number_of_bikes_class"] = 0
    source["number_bikes"] = source["number_bikes"].astype(int)
    source.loc[source["number_bikes"] > 0, "number_of_bikes_class"] = 1
    source.loc[source["number_bikes"] >= source["nb_hhl_members"], "number_of_bikes_class"] = 2
    source["number_of_bikes"] = source["number_bikes"]
    del source["number_bikes"]

    source["household_size_class"] = np.minimum(5, source["nb_hhl_members"]) - 1
    source["household_size"] = source["nb_hhl_members"]
    del source["nb_hhl_members"]

    source["marital_status"]    = 0
    source["driving_license"]  = False
    source["employed"]         = False
    source["car_availability"] = 2

    source["parking_work"]           = "free"
    source["parking_cost_work"]      = 0
    source["parking_education"]      = "free"
    source["parking_cost_education"] = 0

    source["subscriptions_ga"]            = 1
    source["subscriptions_halbtax"]       = 1
    source["subscriptions_verbund"]       = 1
    source["subscriptions_strecke"]       = 1
    source["subscriptions_gleis7"]        = 1
    source["subscriptions_other"]         = 1
    source["subscriptions_junior"]        = 1
    source["subscriptions_verbund_class"] = 1
    source["subscriptions_ga_class"]      = 1
    source["subscriptions_strecke_class"] = 1

    age_classes_upper_bounds = [6, 15, 18, 24, 30, 45, 65, 80]
    source["age_class"] = np.digitize(source["age"], age_classes_upper_bounds)

    source["person_id"]        = source["person_id_source"]
    source["person_weight"]    = source["weight_diary_sample"]
    source["person_weight"]    = [(w if w != " " else 0) for w in source["person_weight"]]
    source["person_weight"]    = source["person_weight"].astype(float)

    del source["person_day_id"]
    del source["person_id_source"]
    del source["hhl_id"]
    del source["hhl_income_26cat"]
    del source["weight_interview_sample"]
    del source["weight_diary_sample"]

    return source, trips
