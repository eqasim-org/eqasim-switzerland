import data.utils
import pandas as pd
import numpy as np
import data.constants as c
import scipy.spatial
from tqdm import tqdm

def impute(df_statpop, df_mz):
    # TODO: HEre, and for the MZ we could use the open data "Gemeindetypen" from ARE!

    # Get all structural observations from MZ
    kd_data = df_mz[
        df_mz["home_structure"] != -1
    ][["home_x", "home_y", "home_structure"]].values

    # Build a KD tree structure for spatial search
    kd_tree = scipy.spatial.KDTree(kd_data[:,:2])

    # Find the heads of houeshold for matching
    df_head = pd.DataFrame(df_statpop[
        df_statpop["is_head"]][["household_id", "home_x", "home_y"]])

    distances = np.zeros((len(df_head), ))
    indices = np.zeros((len(df_head), ), dtype = np.int)

    # Match every STATPOP observation coordinate to an MZ coordinate and assign structural calss from this coordinate
    for i, coord in enumerate(tqdm(df_head[["home_x", "home_y"]].values)):
        distances[i], indices[i] = kd_tree.query(coord)

    # Assign the structural class
    df_head.loc[:, "home_structure"] = kd_data[indices, 2]

    # Merge into statpop population
    return pd.merge(df_statpop, df_head[["household_id", "home_structure"]])
