import pandas as pd
import numpy as np
import data.constants as c
import geopandas as gpd
from tqdm import tqdm
from sklearn.neighbors import KDTree

def configure(context, require):
    require.stage("data.od.raw")
    require.config("raw_data_path")

def execute(context):
    raw_data_path = context.config["raw_data_path"]
    df_od = context.stage("data.od.raw")

    df_shapes = gpd.read_file(
        "%s/statistical_quarter_borders/shp/quart17.shp" % raw_data_path,
        encoding = "utf-8"
    ).to_crs({'init': 'EPSG:2056'})

    df_shapes["zone"] = df_shapes["GMDEQNR"]
    df_shapes = df_shapes[["zone", "geometry"]]

    requested_quarter_ids = set(np.unique(df_od[df_od["home_quarter"] > 0]["home_quarter"]))
    requested_quarter_ids |= set(np.unique(df_od[df_od["work_quarter"] > 0]["work_quarter"]))
    available_quarter_ids = set(np.unique(df_shapes["zone"]))
    remaining_quarter_ids = requested_quarter_ids - available_quarter_ids

    print("Found %d quarter ids which do not exist anymore" % len(remaining_quarter_ids))

    # Note: The code below was an attempt to find the closest id and thus hopefully
    # keep them in the city. However, there is no observation for which it really
    # works. This means that the missing ids are mostly quarters for cities which
    # do not provide quarters anymore! Hence, we just give the list of existing
    # quarters. The code down-stream needs to cope with that.

    # We do not have historical data about the quarters. The best thing we can do
    # is to choose the id that is closest.

    #available_quarter_ids = np.array(list(available_quarter_ids))
    #remaining_quarter_ids = np.array(list(remaining_quarter_ids))
    #mapping = [np.argmin(np.abs(available_quarter_ids - id)) for id in remaining_quarter_ids]
    #mapped_quarter_ids = available_quarter_ids[mapping]

    #for original, mapped in zip(remaining_quarter_ids, mapped_quarter_ids):
    #    print(original, mapped, np.abs(original-mapped))

    return df_shapes















#
