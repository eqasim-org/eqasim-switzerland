import pandas as pd
import numpy as np
import analysis.myutils as myutils
import analysis.myplottools as myplottools
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def configure(context):
    context.config("output_path")
    context.config("data_path")
    context.config("analysis_path")
    context.config("weekend_scenario", False)
    context.config("specific_day_scenario", "avgworkday")
    context.config("specific_weekend_scenario", "all")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("population.output")
    #context.stage("analysis.import_syn_trips")
    
    
def import_data_synthetic(context):
    #Run the stage that generates the synthetic data needed
    filepath = "%s/trips_with_distance.csv" % context.config("output_path")
    if ~(os.path.isfile(filepath)):
        context.stage("population.output")
    df_trips = pd.read_csv(filepath, encoding = "latin1", sep = ";")

    filepath = "%s/persons.csv" % context.config("output_path")
    df_persons = pd.read_csv(filepath, encoding = "latin1", sep = ";")

    df_syn = df_persons.merge(df_trips, left_on="person_id", right_on="person_id")

    df_syn = df_syn[df_syn["age"] >= 6]

    df_syn["od"] = df_syn["preceding_purpose"] + "_" + df_syn["following_purpose"]

    t_id = df_syn["person_id"].values.tolist()
    df_persons_no_trip = df_persons[np.logical_not(df_persons["person_id"].isin(t_id))]
    df_persons_no_trip = df_persons_no_trip.set_index(["person_id"])

    df_persons_no_trip = df_persons_no_trip[df_persons_no_trip["age"] >= 6]

    print("Synthetic: ", len(list(set(df_syn["person_id"].values.tolist()))), ", ", len(df_persons_no_trip))

    return df_syn, df_persons_no_trip   


def import_data_actual(context):
    df_act_persons = context.stage("data.microcensus.persons")

    is_weekend_scenario = context.config("weekend_scenario")
    specific_weekend_scenario = context.config("specific_weekend_scenario")
    specific_day = context.config("specific_day_scenario")
    is_specific_day_scenario = (specific_day != "avgworkday") & (specific_day is not None)
    df_act_persons = pd.DataFrame(df_act_persons[
                                 (is_weekend_scenario & df_act_persons[
                                     "weekend"])  # use only weekend samples for a weekend scenario - maybe not needed
                                 |
                                 (~is_weekend_scenario & ~is_specific_day_scenario & (
                                     ~df_act_persons["weekend"]))  # and only weekday samples for a weekday
                                 |
                                 # add options for different days of the week scenarios that are not weekend
                                 (~is_weekend_scenario & (specific_day != "avgworkday") & (
                                             df_act_persons["day"] == specific_day))
                                 ])

    # If specific weekend context is needed for saturday or sunday
    if (is_weekend_scenario & (specific_weekend_scenario != "all")):
        df_act_persons = pd.DataFrame(df_act_persons[((specific_weekend_scenario == "saturday") & df_act_persons["saturday"]) |
                                           ((specific_weekend_scenario == "sunday") & df_act_persons["sunday"])])

    print("INFO: The scenario will be generated for: ", df_act_persons["day"].unique(), " day (s) of the week")

    df_act_trips = context.stage("data.microcensus.trips")

    # Merging with person information, correcting trips with erroneous purpose

    df_act_persons.rename(columns = {"person_weight": "weight_person"}, inplace = True)
    df_px = df_act_persons[["person_id", "weight_person", "employed",
                                                "age", "sex", "car_availability"]]
    df_act = df_act_trips.merge(df_px, on=["person_id"], how='left')
    df_act.loc[(df_act["purpose"]=='work') & (df_act["age"] < 16), "purpose"]="other"

    #separate trip purposes into O-D
    df_act["following_purpose"] = df_act["purpose"]
    df_act["preceding_purpose"] = df_act["following_purpose"].shift(1)
    df_act.loc[df_act["trip_id"] == 1, "preceding_purpose"] = "home"
    df_act["od"] = df_act["preceding_purpose"] + "_" + df_act["following_purpose"]

    # Only keep the persons that could have been used in activity chain matching
    df_act = df_act[~df_act["weight_person"].isna()]
    df_act = df_act.set_index(["person_id"])
    df_act.sort_index(inplace=True)
    
    t_id = df_act_trips["person_id"].values.tolist()
    df_persons_no_trip = df_act_persons[np.logical_not(df_act_persons["person_id"].isin(t_id))]
    df_persons_no_trip = df_persons_no_trip.set_index(["person_id"])

    return df_act, df_persons_no_trip


def aux_data_frame(df_act, df_syn):
    df_act["person_id"] = df_act.index
    pers_ids = list(set(df_act["person_id"].values.tolist()))

    ids = []
    weights = []
    chains = []

    for pid in pers_ids:
        df = df_act[df_act["person_id"] == pid]
        weight = np.mean(df["weight_person"].values.tolist())
        purposes = df["purpose"].values.tolist()
        chain = "home-" + "-".join([purpose for purpose in purposes])
        ids.append(pid)
        weights.append(weight)
        chains.append(chain)

    df_aux_act = pd.DataFrame.from_dict({"person_id": ids, "weight_person":weights, "chain": chains})   

    pers_ids = list(set(df_syn["person_id"].values.tolist()))

    ids = []
    chains = []

    for pid in tqdm(pers_ids):
        df = df_syn[df_syn["person_id"] == pid]
        purposes = df["following_purpose"].values.tolist()
        chain = "home-" + "-".join([purpose for purpose in purposes])
        ids.append(pid)
        chains.append(chain)

    df_aux_syn = pd.DataFrame.from_dict({"person_id": ids, "weights": [1 for i in range(len(ids))], "chain": chains})      

    return df_aux_act, df_aux_syn


def activity_chains_comparison(context, all_CC, suffix = None):
    # Get percentages, prepare for plotting
    all_CC["synthetic Count"] = all_CC ["synthetic Count"] / all_CC["synthetic Count"].sum() *100
    all_CC["actual Count"] = all_CC["actual Count"] / all_CC["actual Count"].sum() *100
    all_CC = all_CC.sort_values(by=['actual Count'], ascending=False)
    all_CC.to_csv("%s/actchains_DF.csv" % context.config("analysis_path"))

    # First step done: plot activity chain counts
    title_plot = "Synthetic and HTS activity chain comparison"
    title_figure = "activitychains"
    if suffix:
        title_plot += " - " + suffix
        title_figure += "_" + suffix
        
    title_figure += ".png"
    
    myplottools.plot_comparison_bar(context, imtitle = title_figure, plottitle = title_plot, ylabel = "Percentage", xlabel = "Activity chain", lab = all_CC["Chain"], actual = all_CC["actual Count"], synthetic = all_CC["synthetic Count"])


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
    for k in range(8):
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
                                    synthetic = counts["synthetic Count"])
    
    
def activity_counts_per_purpose(context, all_CC, suffix = None):
    all_CC_dic = all_CC.to_dict('records')
    purposes = ['home', 'work', 'education', 'shop', 'leisure', 'other']
    counts_dic = {}
    cpt = 0
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
            act = chain.split("-")
            act = act[1:-1]
            for p in purposes:
                cpt_purpose = act.count(p)
                if cpt_purpose > 0 :
                    identifier = p + " - " + str(cpt_purpose) 
                    if cpt_purpose > 1:
                        identifier += " times"
                    else:
                        identifier += " time"
                    if cpt_purpose >= 3 or (cpt_purpose == 2 and p not in ['h', 'w', 'e']):
                        identifier = "other"
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
    val = "other"
    idx = counts.index.drop(val).tolist() + [val]
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
                                    synthetic = counts["synthetic Count"], t = 20)
    


def compute_distances_synthetic(df_syn, threshold = 25):
    df_syn["crowfly_distance"] = 0.001 * np.array(df_syn["crowfly_distance"])

    # Only consider crowfly distances shorter than <threshold> km
    df_syn_dist = df_syn[df_syn["crowfly_distance"] < threshold]
    return df_syn_dist


def compute_distances_actual(df_act, threshold = 25):
    # Compute the distances
    df_act["crowfly_distance"] = 0.001 * np.sqrt(
        (df_act["origin_x"] - df_act["destination_x"])**2 +
        (df_act["origin_y"] - df_act["destination_y"])**2
    )

    df_act_dist = df_act[df_act["crowfly_distance"] < threshold]
    return df_act_dist


def compare_dist_educ(context, df_syn, df_act, suffix = None):
    pers_educ_syn = list(set(df_syn[df_syn["following_purpose"] == "education"]["person_id"].values))
    pers_educ_act = list(set(df_act[df_act["purpose"] == "education"].index.values))

    df_syn_educ = df_syn[np.isin(df_syn["person_id"], pers_educ_syn)]
    df_act_educ = df_act[np.isin(df_act.index, pers_educ_act)]

    print(df_act_educ.columns)

    df_syn_h_e = df_syn_educ[np.logical_or( np.logical_and( df_syn_educ["preceding_purpose"] == "home",  df_syn_educ["following_purpose"] == "education" ), np.logical_and(df_syn_educ["following_purpose"] == "home",  df_syn_educ["preceding_purpose"] == "education")     )]
    pers_he_syn = list(set(df_syn_h_e["person_id"].values))

    df_act_h_e = df_act_educ[np.logical_or( np.logical_and( df_act_educ["preceding_purpose"] == "home",  df_act_educ["purpose"] == "education" ), np.logical_and(df_act_educ["purpose"] == "home",  df_act_educ["preceding_purpose"] == "education")     )]
    pers_he_act = list(set(df_syn_h_e.index.values))

    dic_syn = {"person_id": pers_educ_syn, "dist_home_educ": [0 for i in range(len(pers_educ_syn))]}
    dic_act = {"person_id": pers_educ_act, "weight_person": [0 for i in range(len(pers_educ_act))], "dist_home_educ": [0 for i in range(len(pers_educ_act))]}

    for i in range(len(pers_educ_syn)):
        pid = pers_educ_syn[i]
        df_pers = df_syn_educ[df_syn_educ["person_id"] == pid]
        home_coord = None
        educ_coord = None
        for index, row in df_pers.iterrows():
             dist = row["crowfly_distance"]
        #    if row["preceding_purpose"] == "home":
        #        home_coord = row["geometry"].coords[0]
        #    if row["following_purpose"] == "home":
        #        home_coord = row["geometry"].coords[1]
        #    if row["preceding_purpose"] == "education":
        #        educ_coord = row["geometry"].coords[0]
        #    if row["following_purpose"] == "education":
        #        educ_coord = row["geometry"].coords[1]
        #    if home_coord is not None and educ_coord is not None:
        #        break
        #dic_syn["dist_home_educ"][i] = 0.001 * np.sqrt(((home_coord[0] - educ_coord[0]) ** 2 + (home_coord[1] - educ_coord[1]) ** 2))
        dic_syn["dist_home_educ"][i] = dist
            

    for i in range(len(pers_educ_act)):
        pid = pers_educ_act[i]
       
        df_pers = df_act_educ[df_act_educ.index == pid]
        home_x = None
        educ_y = None
        for index, row in df_pers.iterrows():
            if row["preceding_purpose"] == "home":
                home_x = row["origin_x"]
                home_y = row["origin_y"]
            elif row["purpose"] == "home":
                home_x = row["destination_x"]
                home_y = row["destination_y"]
            if row["preceding_purpose"] == "education":
                educ_x = row["origin_x"]
                educ_y = row["origin_y"]
            elif row["purpose"] == "education":
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
    x_cdf /= x_cdf[-1]

    y_data = np.array(act, dtype=np.float64)
    y_sorted = np.argsort(y_data)
    y_weights = np.array(act_w, dtype=np.float64)
    y_cdf = np.cumsum(y_weights[y_sorted])
    y_cdf /= y_cdf[-1]

    ax.plot(y_data[y_sorted], y_cdf, label="Actual", color = "#A3A3A3")
    ax.plot(x_data[x_sorted], x_cdf, label="Synthetic", color="#00205B")  

    imtitle = "dist_home_educ"
    plottitle = "Distance from home to education"
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
    myplottools.plot_comparison_hist_mode(context, dmh_title, df_act_dist, df_syn_dist, bins = np.linspace(0,25,120), dpi = 300, cols = 3, rows = 2)

    myplottools.plot_comparison_cdf_purpose(context, dpc_title, df_act_dist, df_syn_dist, dpi = 300, cols = 3, rows = 2)
    myplottools.plot_comparison_cdf_mode(context, dmc_title, df_act_dist, df_syn_dist, dpi = 300, cols = 3, rows = 2)

def compute_detailed_plots(context, df_act_dist, df_syn_dist):
    df_act_dist["weighted_dist"] = df_act_dist["weight_person"] * df_act_dist["crowfly_distance"]
    # by o-d pairs to see the w-w issue
    df_act_od = (df_act_dist.groupby(["od"]).sum()["weighted_dist"] / df_act_dist.groupby(["od"]).sum()[
        "weight_person"]).reset_index(name="act_distance")
    df_syn_od = (df_syn_dist.groupby(["od"]).mean()["crowfly_distance"]).reset_index(name="syn_distance")

    # ensure there are comparable OD pairs
    df_od_compare = df_act_od.merge(df_syn_od, on="od", how="outer").fillna(0)
    df_od_compare = df_od_compare.set_index("od")
    act_od = df_od_compare["act_distance"]
    syn_od = df_od_compare["syn_distance"]

    myplottools.plot_comparison_bar(context, imtitle="distancepurpose_od.png",
                                    plottitle="Crowfly distance for O-D activity pairs",
                                    ylabel="Mean crowfly distance [km]", xlabel="", lab=syn_od.index, actual=act_od,
                                    synthetic=syn_od, t=None, xticksrot=True)

    # plot average distance without the w-w or e-e or h-h situations
    f_act = df_act_dist["od"].isin(['home_home', 'work_work', 'education_education'])
    f_syn = df_syn_dist["od"].isin(['home_home', 'work_work', 'education_education'])

    df_act_cleaned = df_act_dist[~(f_act)].copy()
    df_syn_cleaned = df_syn_dist[~(f_syn)].copy()

    # by purpose only
    act = df_act_cleaned.groupby(["purpose"]).sum()["weighted_dist"] / df_act_cleaned.groupby(["purpose"]).sum()[
        "weight_person"]
    syn = df_syn_cleaned.groupby(["following_purpose"]).mean()["crowfly_distance"]

    # make the plots
    myplottools.plot_comparison_bar(context, imtitle="distancepurpose_cleaned.png",
                                    plottitle="Crowfly distance for individual activities without w-w, e-e or h-h",
                                    ylabel="Mean crowfly distance [km]", xlabel="", lab=syn.index, actual=act,
                                    synthetic=syn, t=None, xticksrot=True)
    # by o-d pairs to see the w-w issue
    df_act_od_clean = (df_act_cleaned.groupby(["od"]).sum()["weighted_dist"] / df_act_cleaned.groupby(["od"]).sum()[
        "weight_person"]).reset_index(name="act_distance")
    df_syn_od_clean = (df_syn_cleaned.groupby(["od"]).mean()["crowfly_distance"]).reset_index(name="syn_distance")

    # ensure there are comparable OD pairs
    df_od_compare_clean = df_act_od_clean.merge(df_syn_od_clean, on="od", how="outer").fillna(0)
    df_od_compare_clean = df_od_compare_clean.set_index("od")
    act_od = df_od_compare_clean["act_distance"]
    syn_od = df_od_compare_clean["syn_distance"]

    myplottools.plot_comparison_bar(context, imtitle="distancepurpose_cleaned_od.png",
                                    plottitle="Crowfly distance for O-D activity pairs without w-w, e-e or h-h",
                                    ylabel="Mean crowfly distance [km]", xlabel="", lab=syn_od.index, actual=act_od,
                                    synthetic=syn_od, t=None, xticksrot=True)

    #plot cleaned distances
    cols = 3
    row_val = len(df_act_cleaned["purpose"].unique()) / cols
    row_mod = len(df_act_cleaned["purpose"].unique()) % cols
    if row_mod > 0:
        rows = row_val + 1
    else:
        rows = row_val

    #plot without h-h, w-w, e-e
    myplottools.plot_comparison_hist_purpose(context, "distance_purpose_hist_cleaned.png", df_act_cleaned, df_syn_cleaned, bins=np.linspace(0, 25, 120),
                                             dpi=300, cols=cols, rows=rows)

    myplottools.plot_comparison_cdf_purpose(context, "distance_purpose_cdf_cleaned.png", df_act_cleaned, df_syn_cleaned, dpi=300, cols=cols, rows=rows)

    #plot for od_distances - change following purpose to od to use matplottools
    df_syn_cleaned_od = df_syn_cleaned.copy()
    df_syn_cleaned_od["following_purpose"] = df_syn_cleaned_od["od"]
    df_act_cleaned_od = df_act_cleaned.copy()
    df_act_cleaned_od["purpose"] = df_syn_cleaned_od["od"]
    row_val = len(df_act_cleaned["od"].unique()) / cols
    row_mod = len(df_act_cleaned["od"].unique()) % cols
    if row_mod > 0:
        rows = row_val + 1
    else:
        rows = row_val
    myplottools.plot_comparison_hist_purpose(context, "distance_purpose_hist_cleaned_od.png", df_act_cleaned_od,
                                             df_syn_cleaned_od, bins=np.linspace(0, 25, 120),
                                             dpi=300, cols=cols, rows=rows)

    myplottools.plot_comparison_cdf_purpose(context, "distance_purpose_cdf_cleaned_od.png", df_act_cleaned_od, df_syn_cleaned_od,
                                            dpi=300, cols=cols, rows=rows)


def execute(context):
    # Import data, merging
    df_syn, df_syn_no_trip = import_data_synthetic(context)
    df_act, df_act_no_trip = import_data_actual(context)

    df_aux_act, df_aux_syn = aux_data_frame(df_act, df_syn)    

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
    activity_chains_comparison(context, all_CC)
    
    # Number of activities    
    activity_counts_comparison(context, all_CC)
    
    # Number of activities per purposes
    activity_counts_per_purpose(context, all_CC)

    # 2. MODE AND DESTINATION PURPOSE
    #mode_purpose_comparison(context, df_syn, df_act)

    # 3. CROWFLY DISTANCES
    
    # 3.1. Compute the distances
    df_syn_dist = compute_distances_synthetic(df_syn)
    df_act_dist = compute_distances_actual(df_act) 
    
    print(list(set(df_syn["following_purpose"].values.tolist())))

    # 3.2 Prepare for plotting
    df_act_dist["x"] = df_act_dist["weight_person"] * df_act_dist["crowfly_distance"]

    act = df_act_dist.groupby(["purpose"]).sum()["x"] / df_act_dist.groupby(["purpose"]).sum()["weight_person"]
    syn = df_syn_dist.groupby(["following_purpose"]).mean()["crowfly_distance"] 
    print(syn)

    # 3.3 Ready to plot!
    myplottools.plot_comparison_bar(context, imtitle = "distancepurpose.png", plottitle = "Crowfly distance", ylabel = "Mean crowfly distance [km]", xlabel = "", lab = syn.index, actual = act, synthetic = syn, t = None, xticksrot = True )
    all_the_plot_distances(context, df_act_dist, df_syn_dist)
    compute_detailed_plots(context, df_act_dist, df_syn_dist)
    # 3.4 Distance from home to education
    syn_0, act_0, act_w0 = compare_dist_educ(context, df_syn, df_act)
    
    return
    
    
    # 4. Do the same for men and women separated, aged 18 to 40
    
    # 4.1 Create the dataframes
    df_syn_men = df_syn[df_syn["sex"] == "male"]
    df_syn_men = df_syn_men[np.logical_and(df_syn_men["age"] >= 18,
                                           df_syn_men["age"] <= 40)]
    df_syn_no_trip_men = df_syn_no_trip[df_syn_no_trip["sex"] == "male"]
    df_syn_no_trip_men = df_syn_no_trip_men[np.logical_and(df_syn_no_trip_men["age"] >= 18,
                                           df_syn_no_trip_men["age"] <= 40)]
        
    df_syn_women = df_syn[df_syn["sex"] == "female"]
    df_syn_women = df_syn_women[np.logical_and(df_syn_women["age"] >= 18,
                                           df_syn_women["age"] <= 40)]
    df_syn_no_trip_women = df_syn_no_trip[df_syn_no_trip["sex"] == "female"]
    df_syn_no_trip_women = df_syn_no_trip_women[np.logical_and(df_syn_no_trip_women["age"] >= 18,
                                           df_syn_no_trip_women["age"] <= 40)]
        
    df_act_men = df_act[df_act["sex"] == "male"]
    df_act_men = df_act_men[np.logical_and(df_act_men["age"] >= 18,
                                           df_act_men["age"] <= 40)]
    df_aux_men = aux_data_frame(df_act_men)
    df_act_no_trip_men = df_act_no_trip[df_act_no_trip["sex"] == "male"]
    df_act_no_trip_men = df_act_no_trip_men[np.logical_and(df_act_no_trip_men["age"] >= 18,
                                           df_act_no_trip_men["age"] <= 40)]
    
        
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
   


    
    












