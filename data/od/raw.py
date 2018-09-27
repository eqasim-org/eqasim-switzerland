import pandas as pd
import numpy as np
import data.constants as c

def configure(context, require):
    require.config("raw_data_path")

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    import lzma as xz
    import data.utils

    data_frames = []

    for path, weight_column, total, sep in [
        ("%s/BfS.SE.raw/data-raw/se_zpers_2012_CH.csv.xz" % raw_data_path, "WEIGHT2012", 286016, ","),
        ("%s/BfS.SE.raw/data-raw/se_zpers_2011_CH.csv.xz" % raw_data_path, "WEIGHT2011", 282750, ";"),
        ("%s/BfS.SE.raw/data-raw/se_zpers_2010_CH.csv.xz" % raw_data_path, "WEIGHT2010", 317222, ","),
    ]:
        with xz.open(path) as f:
            fields = {
                "RES_MUN" : int,
                "RES_QUARTER" : int,
                "COMPANY_MUN" : int,
                "COMPANY_QUARTER" : int,
                "COMPANY_CTRY" : int,
                "MAINMODETRANSPWORK" : int,
                weight_column : float,
            }

            renames = {
                "RES_MUN" : "home_municipality",
                "RES_QUARTER" : "home_quarter",
                "COMPANY_MUN" : "work_municipality",
                "COMPANY_QUARTER" : "work_quarter",
                "COMPANY_CTRY" : "work_country",
                "MAINMODETRANSPWORK" : "mode",
                weight_column : "weight",
            }

            data_frames.append(data.utils.read_csv(f, fields, renames, total = total, sep = sep))

    return pd.concat(data_frames)
