import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.config("municipality_type_source", default="model")
    assert context.config("municipality_type_source") in ["data", "model"], "municipality_type_source must be either 'data' or 'model'"
    
    if context.config("municipality_type_source") == "model":
        context.stage("data.spatial.municipality_type_model", alias="municipality_type")
    else:
        context.stage("data.spatial.municipality_type_data", alias="municipality_type")


def execute(context):
    return context.stage("municipality_type")



def impute(df, df_municipality_types, remove_unknown=False):
    assert ("municipality_id" in df.columns)
    df = pd.merge(df, df_municipality_types, on="municipality_id", how="left")

    if remove_unknown:
        return df[~np.isnan(df["municipality_type"])]
    else:
        return df