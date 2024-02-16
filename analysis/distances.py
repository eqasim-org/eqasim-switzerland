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
    context.config("raw_data_path")
    context.config("analysis_path")
    context.stage("analysis.import_syn_trips")
    
    
def import_data_synthetic(context):
    filepath = "%s/syn_trips.csv" % context.config["output_path"]
    df_trips = pd.read_csv(filepath, encoding = "latin1")

    filepath = "%s/persons.csv" % context.config["output_path"]
    df_persons = pd.read_csv(filepath, encoding = "latin1", sep = ";")

    df_syn = df_trips.merge(df_persons, left_on="person_id", right_on="person_id")
    t_id = df_syn["person_id"].values.tolist()
    df_persons_no_trip = df_persons[np.logical_not(df_persons["person_id"].isin(t_id))]
    df_persons_no_trip = df_persons_no_trip.set_index(["person_id"])

    print("Synthetic: ", len(df_syn), ",  ", len(df_persons_no_trip))

    return df_syn, df_persons_no_trip   


def import_data_actual(context):
    df_act_persons = pd.read_csv(
        "%s/microcensus/zielpersonen.csv" % context.config["raw_data_path"],
        sep = ",", encoding = "latin1", parse_dates = ["USTag"]
    )

    filepath = "%s/microcensus_trips.csv" % context.config["output_path"]
    df_act_trips = pd.read_csv(filepath, encoding = "latin1")

    # Merging with person information, correcting trips with erroneous purpose
    df_act_persons["age"] = df_act_persons["alter"]
    df_act_persons["sex"] = df_act_persons["gesl"] - 1 # Make zero-based
    df_act_persons["person_id"] = df_act_persons["HHNR"]
    df_act_persons["weight_person"] = df_act_persons["WP"]
    df_act_persons["date"] = df_act_persons["USTag"]

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
    df_act = df_act_trips.merge(df_act_persons[["person_id", "weight_person", "employed", 
                                                "age", "sex", "car_availability"]],
                                on=["person_id"], how='left')
    df_act.loc[(df_act["purpose"]=='work') & (df_act["age"] < 16), "purpose"]="other"

    # Only keep the persons that could have been used in activity chain matching
    df_act = df_act[~df_act["weight_person"].isna()]
    df_act = df_act.set_index(["person_id"])
    df_act.sort_index(inplace=True)
    
    t_id = df_act_trips["person_id"].values.tolist()
    df_persons_no_trip = df_act_persons[np.logical_not(df_act_persons["person_id"].isin(t_id))]
    df_persons_no_trip = df_persons_no_trip.set_index(["person_id"])

    print(df_act.columns)
    return df_act, df_persons_no_trip


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

    df_syn_h_e = df_syn_educ[np.logical_or( np.logical_and( df_syn_educ["preceeding_purpose"] == "home",  df_syn_educ["following_purpose"] == "education" ), np.logical_and(df_syn_educ["following_purpose"] == "home",  df_syn_educ["preceeding_purpose"] == "education")     )]
    pers_he_syn = list(set(df_syn_h_e["person_id"].values))

    df_act_h_e = df_act_educ[np.logical_or( np.logical_and( df_act_educ["origin_purpose"] == "home",  df_act_educ["purpose"] == "education" ), np.logical_and(df_act_educ["purpose"] == "home",  df_act_educ["origin_purpose"] == "education")     )]
    pers_he_act = list(set(df_syn_h_e.index.values))

    dic_syn = {"person_id": pers_educ_syn, "dist_home_educ": [0 for i in range(len(pers_educ_syn))]}
    dic_act = {"person_id": pers_educ_act, "weight_person": [0 for i in range(len(pers_educ_act))], "dist_home_educ": [0 for i in range(len(pers_educ_act))]}

    for i in range(len(pers_educ_syn)):
        pid = pers_educ_syn[i]
        df_pers = df_syn_educ[df_syn_educ["person_id"] == pid]
        home_coord = None
        educ_coord = None
        for index, row in df_pers.iterrows():
            if row["preceeding_purpose"] == "home":
                home_coord = [int(float(row["origin_x"])), int(float(row["origin_y"]))]
            if row["following_purpose"] == "home":
                home_coord = [int(float(row["destination_x"])), int(float(row["destination_y"]))]
            if row["preceeding_purpose"] == "education":
                educ_coord = [int(float(row["origin_x"])), int(float(row["origin_y"]))]
            if row["following_purpose"] == "education":
                educ_coord = [int(float(row["destination_x"])), int(float(row["destination_y"]))]
            if home_coord is not None and educ_coord is not None:
                break
        dic_syn["dist_home_educ"][i] = 0.001 * np.sqrt(((home_coord[0] - educ_coord[0]) ** 2 + (home_coord[1] - educ_coord[1]) ** 2))
            

    for i in range(len(pers_educ_act)):
        pid = pers_educ_act[i]
       
        df_pers = df_act_educ[df_act_educ.index == pid]
        home_x = None
        educ_y = None
        for index, row in df_pers.iterrows():
            if row["origin_purpose"] == "home":
                home_x = row["origin_x"]
                home_y = row["origin_y"]
            elif row["purpose"] == "home":
                home_x = row["destination_x"]
                home_y = row["destination_y"]
            if row["origin_purpose"] == "education":
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
    plt.savefig("%s/" % context.config["analysis_path"] + imtitle)
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



def execute(context):
    # Import data, merging
    df_syn, df_syn_no_trip = import_data_synthetic(context)
    df_act, df_act_no_trip = import_data_actual(context)

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

    # 3.4 Distance from home to education
    syn_0, act_0, act_w0 = compare_dist_educ(context, df_syn, df_act)

    # 5 Distance from home to education according to age
    ages = [[0, 6], [6, 12], [12,15], [15,19], [19, 24], [25, 1000]]

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





