import numpy as np
from tqdm import tqdm
import requests
import pandas as pd
import itertools
from random import choices
import pyreadr
from collections import Counter

from MATSimAPI.consume import RApiConsumer
import MATSimAPI.utils as utils

from typing import Literal

import warnings
warnings.filterwarnings("ignore")


def read_prob_dist_from_file(file_path = "SNN/weekly_distribution_ho.csv"):
    df = pd.read_csv(file_path)

    df = df[["wfh_n_now", "full_time", "weekday", "perc"]]

    df.loc[:, "cat"] = [str(e) + ", " + str(int(f)) for f,e in list(zip(df["wfh_n_now"], df["full_time"]))]

    table = {}

    for thecat in list(set(df["cat"].values.tolist())):
        print("  INFO processing data for ", thecat)
        thedf = df[df["cat"] == thecat]
        days = []
        ndays = int(thecat.split(", ")[1])
        for weekday in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:

            if weekday in thedf["weekday"].values.tolist():
                p = thedf[thedf["weekday"] == weekday]["perc"].values.tolist()[0]
            else:
                p = 0

            days.append(list(np.random.choice([weekday, "no"], p = [p, 1-p], size = 1000000)))

        wfhdays = ["-".join([days[i][j] for i in range(7)]) for j in range(1000000)]
        wfhdays = [w.replace("-no", "") for w in wfhdays]
        wfhdays = [w.replace("no-", "") for w in wfhdays]

        wfhdays = dict(Counter(wfhdays))
        clean_wfhdays = {}

        for key, value in wfhdays.items():
            if len(key.split("-")) == ndays and key != "no":
                clean_wfhdays[key] = value

        wfhdays = clean_wfhdays

        wfhdf = pd.DataFrame(wfhdays.items())
        wfhdf.columns = ["days", "perc"]
        wfhdf["perc"] = wfhdf["perc"] / np.sum(wfhdf["perc"])

        table[thecat]= wfhdf

    return table


def predictor_wfh_specific_day(data, prob_dist):

    data.loc[:, "cat"] = [str(e) + ", " + str(int(f)) for f,e in list(zip(data["wfh"], data["emp"]))]
    data.loc[:, "wfh_days"] = "no"

    new_data = pd.DataFrame(columns = data.columns)

    for cat in list(set(data["cat"])):
        print("  INFO predicting HO willingness for the category ", cat)
        dat = data[data["cat"] == cat]
        if int(cat.split(", ")[1]) == 0:
            dat.loc[:, "wfh_days"] = "no"
        else:
            df_prob = prob_dist[cat]
            wfhdays = df_prob["days"]
            prob = df_prob["perc"]
            dat.loc[:, "wfh_days"] = [np.random.choice(wfhdays, p = prob) for _ in range(len(dat))]

        if len(new_data) == 0:
            new_data = dat
        else:
            new_data = pd.concat([new_data, dat])

    for days in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        new_data.loc[:, "wfh_" + days] = new_data.wfh_days.apply(lambda x : days in x)

    print(len(new_data[new_data["wfh_friday"]]) / len(new_data))

    return new_data


def configure(context):
    context.config("output_path")


def execute(context):
    test_population_path = "%s/employed_population_canton_zh.csv" % context.config("output_path")

    df = pd.read_csv(test_population_path)
    df = df[["ID", "wfh", "emp", "weekday"]]

    df = predictor_wfh_specific_day(df, prob_dist=read_prob_dist_from_file())

    

