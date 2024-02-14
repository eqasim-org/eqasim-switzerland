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

N_SAMPLES = 10000
DAYS      = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY_NB    = {"monday":1, "tuesday":2, "wednesday":3, "thursday":4, "friday":5, "saturday":6, "sunday":7}
EMP_CAT   = ["full_time", "part_time", "mult_part_time"]
WFH_FREQ  = range(1,6)


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


class RApiException(Exception):
    pass


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
        new_data.loc[new_data["weekday"]== days, "wfh_today"] = new_data[new_data["weekday"]==days].wfh_days.apply(lambda x: days in x)

    data = new_data
    data.loc[:, "wfh_the_days"] = data["wfh_days"]
    return data


def weekday_probability(df_persons):

    assert "wfh" in df_persons.columns # WFH frequency in number of days / week
    #assert "emp" in df_persons.columns # Employment status between "full_time", "mult_part_time", "part_time"

    data_wfh = df_persons.copy()
    prob_dist = read_prob_dist_from_file()
    
    try:
        data_wfh = predictor_wfh_specific_day(data_wfh, prob_dist)

    except RApiException as e:
        print(e) 

    return data_wfh


def predictor_wfh(data: pd.DataFrame, port: int = 8000) -> pd.DataFrame:
    """
    Wraps the API in a simple-to-use function
    
    Check out `MATSimAPI start` and then inspect the documentation endpoint `/doc`
    to understand what variables the predictor expects.
    """
    url = f"http://localhost:{port}/predict/wfh"
    headers = {"Content-Type": "application/json"}

    data = utils.from_df(data)  # use helper to cast to json-like format as expected by API

    with RApiConsumer(port=port):
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return pd.DataFrame(result)
        
        else:
            raise RApiException(f"Request failed with status code: {response.status_code}")


def wfh_status(df_persons):
    data_wfh = df_persons.copy()


    try:
        # Ability to WFH
        probs = predictor_wfh(data_wfh)
        data_wfh.loc[:, "proba_able_wfh"] = [x["p1"] for x in probs["selection"]]
        del data_wfh["wfh"]
        data_wfh.loc[:, "able_wfh"] = [np.random.random()<float(p) for p in data_wfh["proba_able_wfh"]]
        del data_wfh["proba_able_wfh"]

        # Desired frequency
        for i in range(6):
            probs.loc[:, "p_"+str(i)] = [x["p"+str(i)] for x in probs["frequency"]]

        probs.loc[:, "weights"] = probs[["p_"+str(i) for i in range(6)]].values.tolist()
        
        data_wfh.loc[:, "desired_wfh_freq"] = [choices(list(range(6)), weights = w, k = 1)[0] for w in probs["weights"]]
        data_wfh.loc[data_wfh["able_wfh"]==0, "desired_wfh_freq"] = 0

        data_wfh = data_wfh.rename(columns = {"able_wfh": "wfa_1", "desired_wfh_freq": "wfh"})

        return data_wfh
        
            
    except RApiException as e:
        print(e)        

    return data_wfh


def predictor_mode(data: pd.DataFrame, mode: Literal["ga", "ca", "ht", "re", "bi", "cs"], port: int = 8000) -> pd.DataFrame:
    """
    Wraps the API in a simple-to-use function
    
    Check out `MATSimAPI start` and then inspect the documentation endpoint `/doc`
    to understand what variables the predictor expects.
    """
    url = f"http://localhost:{port}/predict/{mode}"
    headers = {"Content-Type": "application/json"}

    data = utils.from_df(data)  # use helper to cast to json-like format as expected by API

    with RApiConsumer(port=port):
        response = requests.post(url, json=data, headers=headers)
        print("  ", response.status_code)

        if response.status_code == 200:
            result = response.json()
            return pd.DataFrame(result)
        else:
            raise RApiException(f"Request failed with status code: {response.status_code}")
        

def mob_tool_ownership(df_persons):
    data_wfh = df_persons.copy()
    for mobility_tool in ["ga", "ca", "ht", "re", "bi", "cs"]:
        print("  INFO: computing mobility tool ownership for " + mobility_tool +".")
        try:
            probs = predictor_mode(data_wfh, mode = mobility_tool)
            data_wfh.loc[:, "has_"+mobility_tool] = [np.random.random()<float(p) for p in probs["p1"]]

        except RApiException as e:
            print(e)        

    return data_wfh