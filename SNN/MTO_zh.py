import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

shp_path = "/nas/asallard/Switzerland/SNN_shapefile_zurich/zurich_5km.shp"

def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.SNN_population")

    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.households")
    context.config("output_path")

def mto_comparison_wfh_models(pop_before, pop_after, pop_mz, output_file_path, output_path):

    for modeT in ["subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "car_availability"]:
        s1 = pop_before[pop_before["age"]>=6][modeT].value_counts(dropna = True)
        s2 = pop_after[pop_after["age"]>=6][modeT].value_counts(dropna = True)
        s3 = pop_mz.groupby([modeT])["person_weight"].sum()
        d1 = pd.DataFrame(s1).reset_index()
        d2 = pd.DataFrame(s2).reset_index()
        d3 = pd.DataFrame(s3).reset_index()
        d3.columns = ["index", modeT + "_mz"]

        d = d1.merge(d2, on = "index", suffixes=["_before", "_after"], how = "outer")
        d =  d.merge(d3, on = "index", how = "outer")
        d["index"] = d["index"].astype(str)
        d["index"] = [c.replace("False", "No") for c in d["index"]]
        d["index"] = [c.replace("True", "Yes") for c in d["index"]]
        d["index"] = [c.replace("0.0", "Always") for c in d["index"]]
        d["index"] = [c.replace("1.0", "Sometimes") for c in d["index"]]
        d["index"] = [c.replace("2.0", "Never") for c in d["index"]]

        if modeT.split("_")[0] == "subscriptions":
            suffix = modeT.split("_")[1]
        elif modeT == "car_availability":
            suffix = "car"
            d = d.fillna(0)
            print(d)
        
        d.to_hdf(output_file_path, key = "mto_"+suffix)

        fig, ax = plt.subplots()
    
        if modeT == "car_availability":
            dfsum = d.iloc[[0,2]].sum()
            d.loc[3] = dfsum
            d = d[~d["index"].isin(["Always", "Sometimes"])]
            d["index"] = [c.replace("AlwaysSometimes", "Always/sometimes") for c in d["index"]]
            print(d)
        
        x_labels = d["index"]

        bar0 = d[modeT + "_mz"].values.tolist() / np.sum(d[modeT + "_mz"]) * 100
        bar1 = d[modeT + "_before"].values.tolist() / np.sum(d[modeT + "_before"]) * 100
        bar2 = d[modeT + "_after"].values.tolist() / np.sum(d[modeT + "_after"]) * 100
        
        color_0 = "#DFC27D"
        color_1 = "#80CDC1"
        color_2 = "#01665E"

        bar_width = 0.2

        x0 = range(len(x_labels))
        x1 = [x + bar_width for x in x0]
        x2 = [x + bar_width for x in x1]

        ax.bar(x0, bar0, color = color_0, label = "MZ", width = bar_width)
        ax.bar(x1, bar1, color = color_1, label = "Before applying models", width = bar_width)
        ax.bar(x2, bar2, color = color_2, label = "After applying models", width = bar_width)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(labels=x_labels)

        modetotitle = {"car_availability": "Car availability",
                       "subscriptions_ga": "GA ownership",
                       "subscriptions_halbtax": "Half-fare ownership",
                       "subscriptions_verbund": "Regional PT subscription ownership"}

        plt.title(modetotitle[modeT] + " before and after applying WFH and MTO models")
        plt.xlabel(modetotitle[modeT])
        plt.ylabel("Percentage")
        plt.legend()

        plt.savefig(output_path + "/MTO_" + modeT + ".png", dpi = 300)
        plt.close()

    return


def execute(context):
    # Load population before models
    pop_before = context.stage("synthesis.population.enriched")

    # Load population after models
    pop_after = context.stage("synthesis.population.SNN_population")

    # Select those living within the shapefile
    zurich5km = gpd.read_file(shp_path)
    zurich5km = zurich5km["geometry"].values.tolist()[0]

    homes     = gpd.GeoSeries.from_xy(pop_before["home_x"], pop_before["home_y"])
    homes_in_shp = homes.within(zurich5km)
    pop_before = pop_before[homes_in_shp]
    pop_after  = pop_after[homes_in_shp]

    # Load microcensus population
    #hhl = context.stage("data.microcensus.households")[["person_id", "home_x", "home_y"]]
    pop_mz = context.stage("data.microcensus.persons")
    #pop_mz = pop_mz.merge(hhl, on = "person_id")
    print(pop_mz.columns)
    homes     = gpd.GeoSeries.from_xy(pop_mz["home_x"], pop_mz["home_y"])
    homes_in_shp = homes.within(zurich5km)
    pop_mz = pop_mz[homes_in_shp]

    # Setting up the output folder
    output_path = context.config("output_path") + "/MTO_zurich5km"
    Path(output_path).mkdir(parents = True, exist_ok= True)
    output_file_path = output_path + "/results_data_agg_onlyMTO.h5"

    mto_comparison_wfh_models(pop_before, pop_after, pop_mz, output_file_path, output_path)







