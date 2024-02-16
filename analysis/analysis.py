import pandas as pd
import numpy as np
import geopandas as gpd
import analysis.myutils as myutils
import analysis.myplottools as myplottools
import matplotlib.pyplot as plt
import data.constants as c
import pyproj
import data.utils
import data.spatial.utils
from tqdm import tqdm

def configure(context):
    context.config("output_path")
    context.config("data_path")
    context.config("analysis_path")

    context.stage("data.microcensus.trips")

    context.config("weekend_scenario", False)
    context.config("specific_weekend_scenario", "all") # options are "all", "saturday", "sunday"
    context.config("specific_day_scenario", "avgworkday") #options can be any of the days of the week or "avgworkday"
    context.config("include_children")

    if context.config("include_children"):
        context.stage("data.uk_data.trips")
        context.stage("data.uk_data.persons")

    #context.stage("analysis.import_syn_trips")
    context.config("generate_senior_homes")
    
    
def import_data_synthetic(context, population_selector = None):
    filepath = "%s/trips_with_distance.csv" % context.config("output_path")
    df_trips = pd.read_csv(filepath, encoding = "latin1", sep = ";")

    filepath = "%s/persons.csv" % context.config("output_path")
    df_persons = pd.read_csv(filepath, encoding = "latin1", sep = ";")

    filepath = "%s/households.csv" % context.config("output_path")
    df_hhl = pd.read_csv(filepath, encoding = "latin1", sep = ";")

    df_syn = df_persons.merge(df_hhl, left_on="person_id", right_on="household_id")
    df_syn = df_persons.merge(df_trips, left_on="person_id", right_on="person_id")
    
    t_id = df_syn["person_id"].values.tolist()
    df_persons_no_trip = df_persons[np.logical_not(df_persons["person_id"].isin(t_id))]
    df_persons_no_trip = df_persons_no_trip.set_index(["person_id"])

    df_persons_no_trip = df_persons_no_trip[df_persons_no_trip["age"] >= 6]

    if population_selector:
        if "age_selector" in population_selector.keys():
            age_min = population_selector["age_selector"][0]
            age_max = population_selector["age_selector"][1]
            df_syn = df_syn[(df_syn["age"] <= age_max) & (df_syn["age"] >= age_min)]
            df_persons_no_trip = df_persons_no_trip[(df_persons_no_trip["age"] <= age_max) & (df_persons_no_trip["age"] >= age_min)]
            print("INFO excluding agents NOT between the age of ", age_min, " and ", age_max)
        if "gender_selector" in population_selector.keys():
            gender = population_selector["gender_selector"]
            if gender == "male":
                g = 0
            else:
                g = 1
            df_syn = df_syn[df_syn["sex"] == g]
            df_persons_no_trip = df_persons_no_trip[df_persons_no_trip["sex"] == g]
            print("INFO only considering ", gender, " agents.")
        if "canton_selector" in population_selector.keys():
            cantons = population_selector["canton_selector"]
            df_syn = df_syn[df_syn["canton_id"].isin(cantons)]
            df_persons_no_trip = df_persons_no_trip[df_persons_no_trip["canton_id"].isin(cantons)]
            print("INFO only considering agents living in cantons n° ", cantons)
        if "senior_homes_selector" in population_selector.keys():
            sel = population_selector["senior_homes_selector"]
            if sel == "yes" and context.config("generate_senior_homes"):
                print("INFO selecting agents living in retirement homes")
                df_syn = df_syn[df_syn["is_resident"]==1]
                df_persons_no_trip = df_persons_no_trip[df_persons_no_trip["is_resident"]==1]
            elif sel == "yes":
                raise Warning("Trying to select senior home residents while they were not synthesized.")


    return df_syn, df_persons_no_trip   


def import_data_actual(context, population_selector = None):
    df_act_persons = pd.read_csv(
        "%s/microcensus/zielpersonen.csv" % context.config("data_path"),
        sep = ",", encoding = "latin1", parse_dates = ["USTag"]
    )

    df_act_trips, persons_to_be_removed = context.stage("data.microcensus.trips")
    df_act_trips.loc[:, "following_purpose"] = df_act_trips["purpose"]
    df_act_trips["preceding_purpose"] = df_act_trips["origin_purpose"]

    # Merging with person information, correcting trips with erroneous purpose
    df_act_persons["age"] = df_act_persons["alter"]
    df_act_persons["sex"] = df_act_persons["gesl"] - 1 # Make zero-based
    df_act_persons["person_id"] = df_act_persons["HHNR"]
    df_act_persons["weight_person"] = df_act_persons["WP"]
    df_act_persons["date"] = df_act_persons["USTag"]

    df_act_persons["weekend"] = False
    df_act_persons.loc[df_act_persons["tag"] == 6, "weekend"] = True
    df_act_persons.loc[df_act_persons["tag"] == 7, "weekend"] = True

    # Select correct day(s)
    if context.config("weekend_scenario"):
        if context.config("specific_weekend_scenario") == "all":
            print("INFO selecting data for an average week-end day")
            df_act_persons = df_act_persons[df_act_persons["weekend"]]
        elif context.config("specific_weekend_scenario") == "saturday":
            print("INFO selecting data for an average Saturday")
            df_act_persons = df_act_persons[df_act_persons["tag"] == 6]
        elif context.config("specific_weekend_scenario") == "sunday":
            print("INFO selecting data for an average Sunday")
            df_act_persons = df_act_persons[df_act_persons["tag"] == 7]
        else:
            raise ValueError("Please provide a correct weekend day")

    else:
        day_to_number = {"monday": [1], "tuesday":[2], "wednesday":[3], "thursday": [4], "friday":[5], "avgworkday":[1,2,3,4,5]}
        for k,v in day_to_number.items():
            if context.config("specific_day_scenario") == k:
                    if k!= "avgworkday":
                        print("INFO: selecting data for an average ", k)
                    elif k== "avgworkday":
                        print("INFO: selecting data for an average work day")
                    df_act_persons = df_act_persons[df_act_persons["tag"].isin(v)]

    # Driving license
    df_act_persons["driving_license"] = df_act_persons["f20400a"] == 1

    # Car availability
    df_act_persons["car_availability"] = c.CAR_AVAILABILITY_NEVER
    df_act_persons.loc[df_act_persons["f42100e"] == 1, "car_availability"] = c.CAR_AVAILABILITY_ALWAYS
    df_act_persons.loc[df_act_persons["f42100e"] == 2, "car_availability"] = c.CAR_AVAILABILITY_SOMETIMES
    df_act_persons.loc[df_act_persons["f42100e"] == 3, "car_availability"] = c.CAR_AVAILABILITY_NEVER

    # Employment (TODO: I know that LIMA uses a more fine-grained category here)
    df_act_persons["employed"] = df_act_persons["f40800_01"] != -99

    # Infer age class
    df_act_persons["age_class"] = np.digitize(df_act_persons["age"], c.AGE_CLASS_UPPER_BOUNDS)

    df_act_persons.rename(columns = {"binary_car_availability":"car_availability"}, inplace = True)
    df_px = df_act_persons[["person_id", "weight_person", "employed", 
                                                "age", "sex", "car_availability"]]
    df_act = df_act_trips.merge(df_px, on=["person_id"], how='left')
    df_act.loc[(df_act["purpose"]=='work') & (df_act["age"] < 16), "purpose"]="other"
    df_act.loc[(df_act["purpose"]=='work') & (df_act["age"] < 16), "following_purpose"]="other"

    # Only keep the persons that could have been used in activity chain matching
    df_act = df_act[~df_act["weight_person"].isna()]
    df_act = df_act[~df_act["person_id"].isin(persons_to_be_removed)]
    df_act = df_act.set_index(["person_id"])
    df_act.sort_index(inplace=True)
    
    t_id = df_act_trips["person_id"].values.tolist()
    df_persons_no_trip = df_act_persons[np.logical_not(df_act_persons["person_id"].isin(t_id))]
    df_persons_no_trip = df_persons_no_trip.set_index(["person_id"])

    if population_selector:
        if "age_selector" in population_selector.keys():
            age_min = population_selector["age_selector"][0]
            age_max = population_selector["age_selector"][1]
            df_act = df_act[(df_act["age"] <= age_max) & (df_act["age"] >= age_min)]
            df_persons_no_trip = df_persons_no_trip[(df_persons_no_trip["age"] <= age_max) & (df_persons_no_trip["age"] >= age_min)]
            print("INFO excluding agents NOT between the age of ", age_min, " and ", age_max)
        if "gender_selector" in population_selector.keys():
            gender = population_selector["gender_selector"]
            if gender == "male":
                g = 0
            else:
                g = 1
            df_act = df_act[df_act["sex"] == g]
            df_persons_no_trip = df_persons_no_trip[df_persons_no_trip["sex"] == g]
            print("INFO only considering ", gender, " agents.")

    if context.config("include_children"):
        uktrips = context.stage("data.uk_data.trips")
        ukpers  = context.stage("data.uk_data.persons")
        print(uktrips.columns)
        print(ukpers.columns)

    #print(np.sum(df_act["weight_person"]))
    #print(np.sum())

    return df_act, df_persons_no_trip


def aux_data_frame(df_act, df_syn, population_selector = None):
    if population_selector:
        if "age_selector" in population_selector.keys():
            age_min = population_selector["age_selector"][0]
            age_max = population_selector["age_selector"][1]
            df_act = df_act[(df_act["age"] <= age_max) & (df_act["age"] >= age_min)]
            df_syn = df_syn[(df_syn["age"] <= age_max) & (df_syn["age"] >= age_min)]
            print("INFO excluding agents NOT between the age of ", age_min, " and ", age_max)
        if "gender_selector" in population_selector.keys():
            gender = population_selector["gender_selector"]
            df_act = df_act[df_act["sex"] == gender]
            df_syn = df_syn[df_syn["sex"] == gender]
            print("INFO only considering ", gender, " agents.")

    df_act["person_id"] = df_act.index
    pers_ids = list(set(df_act["person_id"].values.tolist()))

    ids = []
    weights = []
    chains = []

    for pid in tqdm(pers_ids, desc = "Building activity chains"):
        df_thisperson = df_act[df_act["person_id"] == pid]
        weight = np.mean(df_thisperson["weight_person"].values.tolist())
        purposes = df_thisperson["purpose"].values.tolist()
        first_activity = df_thisperson["preceding_purpose"].values.tolist()[0]
        chain = "home" + "-" + "-".join([purpose for purpose in purposes])
        ids.append(pid)
        weights.append(weight)
        chains.append(chain)

    df_aux_act = pd.DataFrame.from_dict({"person_id": ids, "weight_person":weights, "chain": chains})   

    pers_ids = list(set(df_syn["person_id"].values.tolist()))

    ids = []
    chains = []
    
    nb_max_activities = 23
    activities_columns = ["activity"+str(i) for i in range(nb_max_activities + 1)]
    act_syn = df_syn[df_syn["trip_id"] == 0][["person_id", "preceding_purpose"]]
    act_syn.rename(columns = {"preceding_purpose": "activity0"}, inplace = True)
    for i in tqdm(range(nb_max_activities+1)):  
        colname = 'activity' + str(i+1)  
        trips_act = df_syn.groupby(['person_id'], as_index=False).nth(i).copy()[
            ['person_id', 'following_purpose']
        ].rename(columns={'following_purpose': colname}) 
        act_syn = pd.merge(act_syn, trips_act, on='person_id', how='left')
    act_syn = act_syn.fillna("stop")
    act_syn["activity_chain"] = act_syn[activities_columns].agg('-'.join, axis=1)
    act_syn["activity_chain"] = [c.replace('-stop', '') for c in act_syn["activity_chain"]]
    act_syn = act_syn[act_syn["activity"+str(nb_max_activities + 1)] == "stop"]
    print("Done")
    
    ids = act_syn["person_id"].values.tolist()
    chains = act_syn["activity_chain"].values.tolist()

    df_aux_syn = pd.DataFrame.from_dict({"person_id": ids, "weights": [1 for i in range(len(ids))], "chain": chains})      

    return df_aux_act, df_aux_syn


def activity_chains_comparison(context, all_CC, suffix = None):
    # Get percentages, prepare for plotting
    all_CC["synthetic Count"] = all_CC ["synthetic Count"] / all_CC["synthetic Count"].sum() *100
    all_CC["actual Count"] = all_CC["actual Count"] / all_CC["actual Count"].sum() *100
    all_CC = all_CC.sort_values(by=['actual Count'], ascending=False)
    all_CC.to_csv("%s/actchains_DF.csv" % context.config("analysis_path"), index = False)

    # First step done: plot activity chain counts
    title_plot = "Synthetic and HTS activity chain comparison"
    title_figure = "activitychains"
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
        
    title_figure += ".png"
    
    myplottools.plot_comparison_bar(context, imtitle = title_figure, plottitle = title_plot, ylabel = "Percentage", xlabel = "Activity chain", lab = all_CC["Chain"], actual = all_CC["actual Count"], synthetic = all_CC["synthetic Count"], xticksrot=True)


def activity_counts_comparison(context, all_CC, suffix = None):
    all_CC_dic = all_CC.to_dict('records')
    counts_dic = {}
    for actchain in all_CC_dic:
        chain = actchain["Chain"]
        s = actchain["synthetic Count"]
        a = actchain["actual Count"]
        if np.isnan(s):
            s = 0
        if np.isnan(a):
            a = 0
        if chain == "-" or chain == "h":
            x = 0
        else:
            act = chain.split("-")
            x = len(act) - 2
        x = min(x, 7)
        if x not in counts_dic.keys():
            counts_dic[x] = [s, a]
        else:
            counts_dic[x][0] += s
            counts_dic[x][1] += a
    
    counts = pd.DataFrame(columns = ["number", "synthetic Count", "actual Count"])
    for k in range(min(8, np.max(list(counts_dic.keys())))):
        v = counts_dic[k]
        if k == 7:
            l = "7+"
        else:
            l = str(int(k))
        counts.loc[k] = pd.Series({"number": l, 
                                      "synthetic Count": v[0],
                                      "actual Count": v[1]
                                          })
    
    # Get percentages, prepare for plotting
    counts["synthetic Count"] = counts["synthetic Count"] / counts["synthetic Count"].sum() *100
    counts["actual Count"] = counts["actual Count"] / counts["actual Count"].sum() *100
    #counts = counts.sort_values(by=['actual Count'], ascending=False)

    # First step done: plot activity chain counts
    title_plot = "Synthetic and HTS activity counts comparison"
    title_figure = "activitycounts"
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
        
    title_figure += ".png"
    
    myplottools.plot_comparison_bar(context, imtitle = title_figure, plottitle = title_plot, 
                                    ylabel = "Percentage", xlabel = "Number of activities in the activity chain",
                                    lab = counts["number"], actual = counts["actual Count"], 
                                    synthetic = counts["synthetic Count"], xticksrot=True)
    
    
def activity_counts_per_purpose(context, all_CC, suffix = None):
    all_CC_dic = all_CC.to_dict('records')
    purposes = ['home', 'work', 'education', 'shop', 'leisure', 'other', "start_out_of_home"]
    counts_dic = {}
    for actchain in all_CC_dic:
        chain = actchain["Chain"]
        s = actchain["synthetic Count"]
        a = actchain["actual Count"]
        if np.isnan(s):
            s = 0
        if np.isnan(a):
            a = 0
        if chain == "-" or chain == "h":
            pass
        else:
            acts = chain.split("-")
            for act in acts:
                if act not in purposes:
                    purposes.append(act)
            for p in purposes:
                cpt_purpose = acts.count(p)
                if cpt_purpose > 0 :
                    identifier = p + " - " + str(cpt_purpose) 
                    if cpt_purpose > 1:
                        identifier += " times"
                    else:
                        identifier += " time"
                    if identifier not in counts_dic.keys():
                        counts_dic[identifier] = [s, a]
                    else:
                        counts_dic[identifier][0] += s
                        counts_dic[identifier][1] += a
    
    counts = pd.DataFrame(columns = ["number", "synthetic Count", "actual Count"])

    for k, v in counts_dic.items():
        counts.loc[k] = pd.Series({"number": k, 
                                      "synthetic Count": v[0],
                                      "actual Count": v[1]
                                          })
            

    # Get percentages, prepare for plotting
    counts["synthetic Count"] = counts["synthetic Count"] / counts["synthetic Count"].sum() *100
    counts["actual Count"] = counts["actual Count"] / counts["actual Count"].sum() *100
    counts = counts.sort_values(by=['actual Count'], ascending=False)
    
    idx = counts.index.tolist() 
    counts = counts.reindex(idx)

    # First step done: plot activity chain counts
    title_plot = "Activity counts per purpose comparison"
    title_figure = "activitycountspurpose"
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
        
    title_figure += ".png"
    
    myplottools.plot_comparison_bar(context, imtitle = title_figure, plottitle = title_plot, 
                                    ylabel = "Percentage", xlabel = "Activities with the same purpose in the activity chain",
                                    lab = counts["number"], actual = counts["actual Count"], 
                                    synthetic = counts["synthetic Count"], t = 20, xticksrot=True)
    


def compute_distances_synthetic(df_syn, threshold = 25):
    df_syn["crowfly_distance"] = 0.001 * np.array(df_syn["crowfly_distance"])

    # Only consider crowfly distances shorter than <threshold> km
    df_syn_dist = df_syn[df_syn["crowfly_distance"] < threshold]
    df_syn_dist = df_syn_dist[df_syn_dist["crowfly_distance"] > 0]
    return df_syn_dist


def compute_distances_actual(df_act, threshold = 25):
    # Compute the distances
    df_act["crowfly_distance"] = 0.001 * np.sqrt(
        (df_act["origin_x"] - df_act["destination_x"])**2 + 
        (df_act["origin_y"] - df_act["destination_y"])**2
    )
    
    df_act_dist = df_act[df_act["crowfly_distance"] < threshold]
    df_act_dist = df_act_dist[df_act_dist["crowfly_distance"] > 0]
    return df_act_dist


def compare_dist_from_home(context, df_syn, df_act, target_purpose = "education", suffix = None):
    if not "origin_purpose" in df_act.columns:
        df_act.loc[:, "origin_purpose"] = df_act["preceding_purpose"]

    # select candidates
    filter_home_prim_syn = (df_syn["following_purpose"] == target_purpose) & (df_syn["preceding_purpose"] == "home")
    filter_home_prim_act = (df_act["following_purpose"] == target_purpose) & (df_act["origin_purpose"] == "home")
    filter_prim_home_syn = (df_syn["following_purpose"] == "home") & (df_syn["preceding_purpose"] == target_purpose)
    filter_prim_home_act = (df_act["following_purpose"] == "home") & (df_act["origin_purpose"] == target_purpose)


    df_syn_educ = df_syn[filter_home_prim_syn | filter_prim_home_syn].drop_duplicates(subset = ["person_id"])
    df_act_educ = df_act[filter_prim_home_act | filter_home_prim_act].drop_duplicates(subset = ["person_id"])

    pers_educ_syn = list(set(df_syn_educ["person_id"].values))
    pers_educ_act = list(set(df_act_educ["person_id"].values))

    dic_syn = {"person_id": pers_educ_syn, "dist_home_educ": [0 for i in range(len(pers_educ_syn))]}
    dic_act = {"person_id": pers_educ_act, "weight_person": [0 for i in range(len(pers_educ_act))], "dist_home_educ": [0 for i in range(len(pers_educ_act))]}

    for i in range(len(pers_educ_syn)):
        pid = pers_educ_syn[i]
        df_pers = df_syn_educ[df_syn_educ["person_id"] == pid]
        for _, row in df_pers.iterrows():
             dist = row["crowfly_distance"]
        dic_syn["dist_home_educ"][i] = dist
            
    for i in range(len(pers_educ_act)):
        pid = pers_educ_act[i]
       
        df_pers = df_act_educ[df_act_educ["person_id"] == pid]
        home_x = None
        educ_y = None
        for index, row in df_pers.iterrows():
            if row["origin_purpose"] != target_purpose:
                home_x = row["origin_x"]
                home_y = row["origin_y"]
            elif row["following_purpose"] != target_purpose:
                home_x = row["destination_x"]
                home_y = row["destination_y"]
            if row["origin_purpose"] == target_purpose:
                educ_x = row["origin_x"]
                educ_y = row["origin_y"]
            elif row["following_purpose"] == target_purpose:
                educ_x = row["destination_x"]
                educ_y = row["destination_y"]
            if educ_y is not None and home_y is not None:
                break
        dic_act["dist_home_educ"][i] = 0.001 * np.sqrt(((home_x - educ_x) ** 2 + (home_y - educ_y) ** 2))
        dic_act["weight_person"][i] = row["weight_person"]

    dist_df_syn = pd.DataFrame.from_dict(dic_syn)
    dist_df_act = pd.DataFrame.from_dict(dic_act)

    syn = dist_df_syn["dist_home_educ"].values
    act = dist_df_act["dist_home_educ"].values
    act_w = dist_df_act["weight_person"].values

    fig, ax = plt.subplots(1,1)
    x_data = np.array(syn, dtype=np.float64)
    x_sorted = np.argsort(x_data)
    x_weights = np.array([1.0 for i in range(len(syn))], dtype=np.float64)
    x_cdf = np.cumsum(x_weights[x_sorted])
    if len(x_cdf) >= 1:
        x_cdf /= x_cdf[-1]

    y_data = np.array(act, dtype=np.float64)
    y_sorted = np.argsort(y_data)
    y_weights = np.array(act_w, dtype=np.float64)
    y_cdf = np.cumsum(y_weights[y_sorted])
    if len(y_cdf) >= 1:
        y_cdf /= y_cdf[-1]

    ax.plot(y_data[y_sorted], y_cdf, label="Actual", color = "#A3A3A3")
    ax.plot(x_data[x_sorted], x_cdf, label="Synthetic", color="#00205B")  

    imtitle = "dist_home_"+target_purpose
    plottitle = "Distance from home to " + target_purpose
    if suffix:
        imtitle += "_" + suffix
        plottitle  += " - " + suffix 
    imtitle += ".png"

    ax.set_ylabel("Probability")
    ax.set_xlabel("Crowfly Distance [km]")
    ax.legend(loc="best")
    ax.set_title(plottitle)
    plt.savefig("%s/" % context.config("analysis_path") + imtitle)
    return syn, act, act_w


def mode_purpose_comparison(context, df_syn, df_act, suffix = None):
    # first in the synthetic data
    types = df_syn.groupby(["mode","following_purpose"]).count()["person_id"]
    syn = types / types.sum()

    # then in the actual data
    df_act.loc[df_act["mode"]=='car_passanger', "mode"] = 'car_passenger'
    which = ["car","car_passenger","pt", "taxi","walk"]
    atypes = df_act.groupby(["mode","destination_purpose"]).sum().loc[which,"weight_person"].reindex(index=which, level=0)
    act = atypes / atypes.sum()
    
    lista = [item for item in list(types.index.levels[0]) for i in range(len(types.index.levels[1]))]
    listb = list(types.index.levels[1]) * len(types.index.levels[0])
    labels = [a + " " + b for a, b in zip(lista,listb)]

    # already ready to plot!
    title_plot = "Synthetic and HTS Mode-Purpose Distribution"
    title_figure = "modepurpose"
    
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
        
    title_figure += ".png"
    
    myplottools.plot_comparison_bar(context, imtitle = title_figure, plottitle = title_plot,
                                    ylabel = "Percentage", xlabel = "", lab = labels, 
                                    actual = act.values.tolist(), synthetic = syn.values.tolist(), 
                                    t = 10, xticksrot = True )



def all_the_plot_distances(context, df_act_dist, df_syn_dist, suffix = None):
    dph_title = "distance_purpose_hist"
    dmh_title = "distance_mode_hist"
    dpc_title = "distance_purpose_cdf"
    dmc_title = "distance_mode_cdf"
    
    if suffix:
        dph_title += "_" + suffix
        dmh_title += "_" + suffix
        dpc_title += "_" + suffix
        dmc_title += "_" + suffix
        
    dph_title += ".png"
    dph_title += ".png"
    dpc_title += ".png"
    dmc_title += ".png"
    
    myplottools.plot_comparison_hist_purpose(context, dph_title, df_act_dist, df_syn_dist, bins = np.linspace(0,25,120), dpi = 300, cols = 3, rows = 2)
    #myplottools.plot_comparison_hist_mode(context, dmh_title, df_act_dist, df_syn_dist, bins = np.linspace(0,25,120), dpi = 300, cols = 3, rows = 2)

    myplottools.plot_comparison_cdf_purpose(context, dpc_title, df_act_dist, df_syn_dist, dpi = 300, cols = 3, rows = 2)
    #myplottools.plot_comparison_cdf_mode(context, dmc_title, df_act_dist, df_syn_dist, dpi = 300, cols = 3, rows = 2)


def generate_plots(context, df_aux_act, df_aux_syn, df_act, df_syn, df_syn_no_trip, df_act_no_trip, suffix):
    syn_CC = df_aux_syn.groupby("chain").size().reset_index(name='count')
    act_CC = df_aux_act.groupby("chain").size().reset_index(name='count')
    act_CC.columns = ["Chain", "actual Count"]
    syn_CC.columns = ["Chain", "synthetic Count"]

     # 1. ACTIVITY CHAINS
    
    # Creating the new dataframes with activity chain counts
    #syn_CC = myutils.process_synthetic_activity_chain_counts(df_syn)
    syn_CC.loc[len(syn_CC) + 1] = pd.Series({"Chain": "home", "synthetic Count": df_syn_no_trip.shape[0] })
   
    #act_CC = myutils.process_actual_activity_chain_counts(df_act, df_aux)
    act_CC.loc[len(act_CC) + 1] = pd.Series({"Chain": "home", "actual Count": np.sum(df_act_no_trip["weight_person"].values.tolist())})

    # Merging together, comparing
    all_CC = pd.merge(syn_CC, act_CC, on = "Chain", how = "outer")
    activity_chains_comparison(context, all_CC, suffix = suffix)
    
    # Number of activities    
    activity_counts_comparison(context, all_CC, suffix = suffix)
    
    # Number of activities per purposes
    activity_counts_per_purpose(context, all_CC, suffix = suffix)

    # 2. CROWFLY DISTANCES
    
    # 2.1. Compute the distances
    df_syn_dist = compute_distances_synthetic(df_syn)
    df_act_dist = compute_distances_actual(df_act) 
    
    # 2.2 Prepare for plotting
    df_act_dist["x"] = df_act_dist["weight_person"] * df_act_dist["crowfly_distance"]

    act = df_act_dist.groupby(["purpose"]).sum()["x"] / df_act_dist.groupby(["purpose"]).sum()["weight_person"]
    syn = df_syn_dist.groupby(["following_purpose"]).mean()["crowfly_distance"] 

    act_purposes = list(set(act.reset_index()["purpose"]))
    syn = syn.reset_index()
    for p in act_purposes:
        if p not in list(set(syn["following_purpose"])):
            syn.loc[len(syn)] = [p, 0]

    syn = syn.groupby(["following_purpose"]).mean()["crowfly_distance"] 

    # 2.3 Ready to plot!
    myplottools.plot_comparison_bar(context, imtitle = "distancepurpose.png", plottitle = "Crowfly distance " + suffix, ylabel = "Mean crowfly distance [km]", xlabel = "", lab = syn.index, actual = act, synthetic = syn, t = None, xticksrot = True )
    all_the_plot_distances(context, df_act_dist, df_syn_dist, suffix)

    # 2.4 Distance from home to education
    for primary_purpose in ["work", "education"]:
        print("INFO computing distances between home and ", primary_purpose)
        syn_0, act_0, act_w0 = compare_dist_from_home(context, df_syn, df_act,primary_purpose, suffix = suffix)


    
def execute(context):
    pop_all = None
    pop_men_1840 = {"age_selector": [18, 40], "gender_selector": "male", "canton_selector":[1,2,5,7]}
    pop_wom_1840 = {"age_selector": [18, 100], "gender_selector": "female", "senior_homes_selector": "no"}

    suff_all = ""
    suff_men_1840 = "men aged 18 to 40 living in cantons 1,2, 5, 7"
    suff_women_1840 = "women aged 18 to 100 living in retirement homes"

    pop_selectors = [pop_all, pop_wom_1840, pop_men_1840]
    suffixes      = [suff_all, suff_women_1840,  suff_men_1840]

    for population_selector, suffix in list(zip(pop_selectors, suffixes)):
        df_syn, df_syn_no_trip = import_data_synthetic(context, population_selector)
        df_act, df_act_no_trip = import_data_actual(context, population_selector)
        df_aux_act, df_aux_syn = aux_data_frame(df_act, df_syn)

        generate_plots(context, df_aux_act, df_aux_syn, df_act, df_syn, df_syn_no_trip, df_act_no_trip, suffix)

    exit()
    
    # 4. Do the same for men and women separated, aged 18 to 40
    
    # 4.1 Create the dataframes
    df_syn_men = df_syn[df_syn["sex"] == "male"]
    df_syn_men = df_syn_men[np.logical_and(df_syn_men["age"] >= 18,
                                           df_syn_men["age"] <= 40)]
    df_syn_no_trip_men = df_syn_no_trip[df_syn_no_trip["sex"] == "male"]
    df_syn_no_trip_men = df_syn_no_trip_men[np.logical_and(df_syn_no_trip_men["age"] >= 18,
                                           df_syn_no_trip_men["age"] <= 40)]
    
    df_act_no_trip_men = df_act_no_trip[df_act_no_trip["sex"] == "male"]
    df_act_no_trip_men = df_act_no_trip_men[np.logical_and(df_act_no_trip_men["age"] >= 18,
                                           df_act_no_trip_men["age"] <= 40)]
        
    df_syn_women = df_syn[df_syn["sex"] == "female"]
    df_syn_women = df_syn_women[np.logical_and(df_syn_women["age"] >= 18,
                                           df_syn_women["age"] <= 40)]
    df_syn_no_trip_women = df_syn_no_trip[df_syn_no_trip["sex"] == "female"]
    df_syn_no_trip_women = df_syn_no_trip_women[np.logical_and(df_syn_no_trip_women["age"] >= 18,
                                           df_syn_no_trip_women["age"] <= 40)]
        
    df_act_men = df_act[df_act["sex"] == "male"]
    df_act_men = df_act_men[np.logical_and(df_act_men["age"] >= 18,
                                           df_act_men["age"] <= 40)]
    df_aux_men, df_aux_men_syn = aux_data_frame(df_act_men, df_syn_men)
    syn_CC = df_aux_men_syn.groupby("chain").size().reset_index(name='count')
    act_CC = df_aux_men.groupby("chain").size().reset_index(name='count')
    act_CC.columns = ["Chain", "actual Count"]
    syn_CC.columns = ["Chain", "synthetic Count"]
    syn_CC.loc[len(syn_CC) + 1] = pd.Series({"Chain": "home", "synthetic Count": df_syn_no_trip_men.shape[0] })    
    act_CC.loc[len(act_CC) + 1] = pd.Series({"Chain": "home", "actual Count": np.sum(df_act_no_trip_men["weight_person"].values.tolist())})
    all_CC_men = pd.merge(syn_CC, act_CC, on = "Chain", how = "outer")   
        
    df_act_women = df_act[df_act["sex"] == "female"]
    df_act_women = df_act_women[np.logical_and(df_act_women["age"] >= 18,
                                           df_act_women["age"] <= 40)]
    df_aux_women = aux_data_frame(df_act_women)
    df_act_no_trip_women = df_act_no_trip[df_act_no_trip["sex"] == "female"]
    df_act_no_trip_women = df_act_no_trip_women[np.logical_and(df_act_no_trip_women["age"] >= 18,
                                           df_act_no_trip_women["age"] <= 40)]
        
    # 4.2 Activity chains
    # Creating the new dataframes with activity chain counts
    M_syn_CC = myutils.process_synthetic_activity_chain_counts(df_syn_men)
    M_syn_CC.loc[len(M_syn_CC) + 1] = pd.Series({"Chain": "h", 
                                          "synthetic Count": df_syn_no_trip_men.shape[0]
                                          })
    M_act_CC = myutils.process_actual_activity_chain_counts(df_act_men, df_aux_men)
    M_act_CC.loc[len(M_act_CC) + 1] = pd.Series({"Chain": "h", 
                                          "actual Count": np.sum(df_act_no_trip_men["weight_person"].values.tolist())
                                          })
    
    W_syn_CC = myutils.process_synthetic_activity_chain_counts(df_syn_women)
    W_syn_CC.loc[len(W_syn_CC) + 1] = pd.Series({"Chain": "h", 
                                          "synthetic Count": df_syn_no_trip_women.shape[0]
                                          })
    W_act_CC = myutils.process_actual_activity_chain_counts(df_act_women, df_aux_women)
    W_act_CC.loc[len(W_act_CC) + 1] = pd.Series({"Chain": "h", 
                                          "actual Count": np.sum(df_act_no_trip_women["weight_person"].values.tolist())
                                          })
    
    # Merging together, comparing
    M_all_CC = pd.merge(M_syn_CC, M_act_CC, on = "Chain", how = "left")
    activity_chains_comparison(context, M_all_CC, "men")
    
    W_all_CC = pd.merge(W_syn_CC, W_act_CC, on = "Chain", how = "left")
    activity_chains_comparison(context, W_all_CC, "women")
    
    activity_counts_comparison(context, M_all_CC, "men")
    activity_counts_comparison(context, W_all_CC, "women")
    
    activity_counts_per_purpose(context, M_all_CC, "men")
    activity_counts_per_purpose(context, W_all_CC, "women")

    # 4.3 Mode-purpose comparison
    mode_purpose_comparison(context, df_syn_men, df_act_men, "men")
    mode_purpose_comparison(context, df_syn_women, df_act_women, "women")
    
    # 4.4 Distance-purpose comparison
    df_syn_distM = compute_distances_synthetic(df_syn_men)
    df_act_distM = compute_distances_actual(df_act_men) 
    df_act_distM["x"] = df_act_distM["weight_person"] * df_act_distM["crowfly_distance"]
    actM = df_act_distM.groupby(["purpose"]).sum()["x"] / df_act_distM.groupby(["purpose"]).sum()["weight_person"]
    synM = df_syn_distM.groupby(["following_purpose"]).mean()["crowfly_distance"] 
    myplottools.plot_comparison_bar(context, imtitle = "distancepurpose_men.png", 
                                    plottitle = "Crowfly distances - men", 
                                    ylabel = "Mean crowfly distance [km]", xlabel = "", 
                                    lab = synM.index, actual = actM, synthetic = synM, t = None, xticksrot = True )
    all_the_plot_distances(context, df_act_distM, df_syn_distM, suffix = "men")
    
    df_syn_distW = compute_distances_synthetic(df_syn_women)
    df_act_distW = compute_distances_actual(df_act_women) 
    df_act_distW["x"] = df_act_distW["weight_person"] * df_act_distW["crowfly_distance"]
    actW = df_act_distW.groupby(["purpose"]).sum()["x"] / df_act_distW.groupby(["purpose"]).sum()["weight_person"]
    synW = df_syn_distW.groupby(["following_purpose"]).mean()["crowfly_distance"] 
    myplottools.plot_comparison_bar(context, imtitle = "distancepurpose_women.png", 
                                    plottitle = "Crowfly distances - women", 
                                    ylabel = "Mean crowfly distance [km]", xlabel = "", 
                                    lab = synM.index, actual = actW, synthetic = synW, 
                                    t = None, xticksrot = True )
    all_the_plot_distances(context, df_act_distW, df_syn_distW, suffix = "women")


    # 5 Distance from home to education according to age
    ages = [[0, 14], [15, 18], [19, 24], [25, 1000]]

    syn_means = [np.mean(syn_0)]
    act_means = [np.average(act_0, weights = act_w0)]
    labels = ["All"]
    for age in ages:
        df_syn_age = df_syn[np.logical_and(df_syn["age"] >= age[0],
                                           df_syn["age"] <= age[1] )]
        df_act_age = df_act[np.logical_and(df_act["age"] >= age[0],
                                           df_act["age"] <= age[1] )]
        suf = "aged " + str(age[0]) + " to " + str(age[1])
        lab = str(age[0]) + " to " + str(age[1]) + " y. o."
        if age[1] == 1000:
            lab = "25 +"
        syn, act, act_w = compare_dist_educ(context, df_syn_age, df_act_age, suffix = suf)

        syn_means.append(np.average(syn))
        act_means.append(np.average(act, weights = act_w))
        labels.append(lab)

    myplottools.plot_comparison_bar(context,"avdisthomeeduc - age.png", "Average distances from home to education", "Average distance [km]", "Population group", labels, act_means, syn_means)

    # 6. Distance from home to education according to residence area

    areas = [1,2,3]

    syn_means = [np.mean(syn_0)]
    act_means = [np.average(act_0, weights = act_w0)]
    labels = ["All"]
    for area in areas:
        df_syn_area = df_syn[df_syn["residence_area_index"] == area]
        df_act_area = df_act[df_act["residence_area_index"] == area]
        suf = "agents living in "
        if area == 1:
            suf += " state"
            lab = "state"
        if area == 2 :
            suf += " city"
            lab = "city"
        if area == 3 :
            suf += " downtown"
            lab = "downtown"

        syn, act, act_w = compare_dist_educ(context, df_syn_area, df_act_area, suffix = suf)
        syn_means.append(np.average(syn))
        act_means.append(np.average(act, weights = act_w))
        labels.append(lab)

    myplottools.plot_comparison_bar(context,"avdisthomeeduc - area.png", "Average distances from home to education", "Average distance [km]", "Population group", labels, act_means, syn_means)


    # 7. Distance from home to education according to gender

    genders = ["male","female"]

    syn_means = [np.mean(syn_0)]
    act_means = [np.average(act_0, weights = act_w0)]
    labels = ["All"]
    for gender in genders:
        df_syn_gender = df_syn[df_syn["sex"] == gender]
        df_act_gender = df_act[df_act["sex"] == gender]
        suf = gender
        lab = gender

        syn, act, act_w = compare_dist_educ(context, df_syn_gender, df_act_gender, suffix = suf)
        syn_means.append(np.average(syn))
        act_means.append(np.average(act, weights = act_w))
        labels.append(lab)

    myplottools.plot_comparison_bar(context,"avdisthomeeduc - gender.png", "Average distances from home to education", "Average distance [km]", "Population group", labels, act_means, syn_means)

    # Zipping modes in correct order
    #modes = zip(np.sort(df_syn["mode"].unique()),["car","car_passenger","pt", "taxi", "walk"])


    # Fourth step: mode share
    #plot_mode_share(df_syn, amdf2, dpi = 300)

    ## TODO: overall percentage of mode share by share
    
    #df_syn, amdf = add_geo_location_to_origin_and_destination(df_syn, amdf)
   


    
    












