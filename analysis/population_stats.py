import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.enriched")

    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.households")
    context.config("output_path")
    context.config("analysis_path")
    context.config("cutout_path", False)

def mto_comparison_wfh_models(pop_before, pop_after, pop_mz, output_file_path, output_path):
    for modeT in ["subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "car_availability", "employed", "income_class", "number_of_cars_class"]:
        s1 = pop_before[pop_before["age"]>=6][modeT].value_counts(dropna = True)
        s2 = pop_after[pop_after["age"]>=6][modeT].value_counts(dropna = True)
        if modeT == "income_class":
            pop_mz = pop_mz[pop_mz["income_imputed"]==False]
        s3 = pop_mz.groupby([modeT])["person_weight"].sum()
        d1 = pd.DataFrame(s1).reset_index()
        d2 = pd.DataFrame(s2).reset_index()
        d3 = pd.DataFrame(s3).reset_index()
        d3.columns = ["index", modeT + "_mz"]

        d = d1.merge(d2, on = "index", suffixes=["_before", "_after"])
        d =  d.merge(d3, on = "index")
        # ---------- LABELING BY VARIABLE TYPE ----------
        # make a copy to work with
        idx = d["index"]

        # Boolean subscription/employment variables
        if modeT in ["subscriptions_ga", "subscriptions_halbtax",
                     "subscriptions_verbund", "employed"]:
            # map True/False to Yes/No, keep anything else as string
            mapped = idx.map({True: "Yes", False: "No"})
            d["index"] = mapped.fillna(idx.astype(str))

        # Car availability: 0/1/2 -> Always/Sometimes/Never
        elif modeT == "car_availability":
            # ensure numeric for mapping
            idx_num = pd.to_numeric(idx, errors="coerce")
            mapped = idx_num.map({0: "Always", 1: "Sometimes", 2: "Never"})
            # if something doesn’t map, just keep original string
            d["index"] = mapped.fillna(idx.astype(str))

        # Income: keep numeric categories, but as strings (or add your own mapping)
        elif modeT == "income_class":
            # make sure integers show up nicely, e.g. "1", "2", ...
            # (you can plug your own dict here if you have labels for each code)
            d["index"] = pd.to_numeric(idx, errors="ignore")
            d["index"] = d["index"].astype(str)
        elif modeT == "number_of_cars_class":
            d["index"] = pd.to_numeric(idx, errors="ignore")
            d["index"] = d["index"].astype(str)
        # fallback (shouldn't really be used here)
        else:
            d["index"] = idx.astype(str)
        # ---------- END LABELING ----------

        # suffix for HDF key and filenames
        if modeT.split("_")[0] == "subscriptions":
            suffix = modeT.split("_")[1]
        elif modeT == "car_availability":
            suffix = "car"
        elif modeT == "income":
            suffix = "income"
        elif modeT == "number_of_cars_class":
            suffix = "number_cars"
        else:
            suffix = modeT

        
        
        d.to_hdf(output_file_path, key = "mto_"+suffix)

        fig, ax = plt.subplots()
    
        if modeT == "car_availability":
            dfsum = d.iloc[[0,2]].sum()
            d.loc[3] = dfsum
            d = d[~d["index"].isin(["Always", "Sometimes"])]
            d["index"] = [c.replace("AlwaysSometimes", "Always/sometimes") for c in d["index"]]
        
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
                       "subscriptions_verbund": "Regional PT subscription ownership",
                       "employed": "Employment",
                       "income_class": "Income class",
                       "number_of_cars_class": "number_of_cars_class"}

        plt.title(modetotitle[modeT] + " before and after applying WFH and MTO models")
        plt.xlabel(modetotitle[modeT])
        plt.ylabel("Percentage")
        plt.legend()

        plt.savefig(output_path + "/MTO_" + modeT + ".png", dpi = 300)
        plt.close()

    return


def execute(context):
    output_path = context.config("analysis_path")
    # Load population before models
    pop_before = context.stage("synthesis.population.enriched")
    #activate for synpop_are
    #pop_before["subscriptions_halbtax"] = pop_before["subscriptions"].isin(["HTA", "HTA+VA"])

    # Load population after models
    pop_after = context.stage("synthesis.population.enriched")
    #activate for synpop_are
    #pop_after["subscriptions_halbtax"] = pop_after["subscriptions"].isin(["HTA", "HTA+VA"])


    # Select those living within the shapefile
    if context.config("cutout_path"):
        print("cutting out the interested population")
        zurich5km = gpd.read_file(context.config("cutout_path"))
        zurich5km = zurich5km["geometry"].values.tolist()[0]

        homes = gpd.GeoSeries.from_xy(pop_before["home_x"], pop_before["home_y"])
        homes_in_shp = homes.within(zurich5km)
    
        pop_before = pop_before[homes_in_shp]
        pop_after  = pop_after[homes_in_shp]

    #pop_before = pop_before[pop_before["canton_id"]==22]
    pop_after = pop_after[pop_after["canton_id"]==1]
    
    # Load microcensus population
    #hhl = context.stage("data.microcensus.households")[["person_id", "home_x", "home_y"]]
    pop_mz = context.stage("data.microcensus.persons")
    print(pop_mz.columns)
    pop_mz = pop_mz[pop_mz["weekend"]== False]
    #pop_mz = pop_mz.merge(hhl, on = "person_id")
    if context.config("cutout_path"):
        homes     = gpd.GeoSeries.from_xy(pop_mz["home_x"], pop_mz["home_y"])
        homes_in_shp = homes.within(zurich5km)
        pop_mz = pop_mz[homes_in_shp]
    pop_mz = pop_mz[pop_mz["canton_id"]==1]
    # Setting up the output folder
    
    Path(output_path).mkdir(parents = True, exist_ok= True)
    output_file_path = output_path + "/results_data_agg_onlyMTO.h5"

    mto_comparison_wfh_models(pop_before, pop_after, pop_mz, output_file_path, output_path)



