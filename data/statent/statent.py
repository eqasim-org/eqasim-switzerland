import pandas as pd
import numpy as np
import data.constants as c
from tqdm import tqdm

def configure(context, require):
    require.config("raw_data_path")

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    df = pd.read_csv(
        "%s/statent/QUERY_FOR_2014_DEC_STATENT_LOC.csv" % raw_data_path,
        encoding = "latin1", sep = ";")

    df = pd.DataFrame(df[["METER_X", "METER_Y", "NOGA08", "EMPTOT"]])
    df.columns = ["x", "y", "noga", "number_employees"]
    df.loc[:, "noga"] = df["noga"].astype(np.str)
    df.loc[:, "enterprise_id"] = np.arange(len(df))

    # For now we don't do anything with the NOGA category.
    # (but need to do later for the education locations)

    return df
