import geopandas as gpd
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.config("municipality_type_source", default="model")
    assert context.config("municipality_type_source") in ["data", "model"], "municipality_type_source must be either 'data' or 'model'"
    
    if context.config("municipality_type_source") == "model":
        context.stage("data.spatial.municipality_type_model", alias="municipality_type")
    else:
        context.stage("data.spatial.municipality_type_data", alias="municipality_type")

    # we include the network of this region, i don't know if this is the right config param to use, to check later!
    context.config("cross_border_exclude_shapefiles", default=None)
    context.config("include_external_population", default = False)
    context.stage("data.external_population.constants")


def execute(context):
    df = context.stage("municipality_type")
    # outside CH region
    out_region_file = context.config("cross_border_exclude_shapefiles")
    include_external_population = context.config("include_external_population")
    if out_region_file is not None and include_external_population:
        cst = context.stage("data.external_population.constants")
        df = df.append({"municipality_id":cst.municipality_id,	
                        "municipality_type":cst.municipality_type,	
                        "imputed_municipality_type":True}, ignore_index=True)
        
    df = df.astype({"municipality_type": "category", "municipality_id":int})
    return df



def impute(df, df_municipality_types, remove_unknown=False):
    assert ("municipality_id" in df.columns)
    df = pd.merge(df, df_municipality_types, on="municipality_id", how="left")

    if remove_unknown:
        return df[~df["municipality_type"].isna()]
    else:
        return df