import geopandas as gpd
import pandas as pd
import numpy as np
from numpy.random import choice
from tqdm import tqdm
from sklearn.neighbors import KDTree
import shapely.geometry as geo
import shapely.geometry as geo
import warnings
#from pandas.core.common import SettingWithCopyWarning

#warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def work_to_homeoffice(df_work, df_home, df_persons):
    df_home = df_home.copy()

    df_home.loc[:, "home_x"] = [p.x for p in df_home["geometry"]]
    df_home.loc[:, "home_y"] = [p.y for p in df_home["geometry"]]

    agents_wfh = df_persons[df_persons["wfh"]>0]["person_id"].values.tolist()
    hhl_wfh    = df_persons[df_persons["wfh"]>0]["household_id"].values.tolist()

    work_wfh = df_work[df_work["person_id"].isin(agents_wfh)]
    home_wfh = df_home[df_home["household_id"].isin(hhl_wfh)]

    work_no_wfh = df_work[~df_work["person_id"].isin(agents_wfh)]

    # Join with household_id
    work_wfh = pd.merge(work_wfh, df_persons[["person_id", "household_id"]], how = "left", on = "person_id")

    # Join with home coordinates
    work_wfh = pd.merge(work_wfh, home_wfh[["household_id", "home_x", "home_y"]], how = "outer", on = "household_id")

    # Change work locations to home locations
    work_wfh.loc[:, "x"] = work_wfh["home_x"]
    work_wfh.loc[:, "y"] = work_wfh["home_y"]
    work_wfh.loc[:, "destination_id"] = -1

    # Delete columns
    del work_wfh["household_id"]
    del work_wfh["home_x"]
    del work_wfh["home_y"]

    df_work = pd.concat([work_no_wfh, work_wfh])

    return df_work