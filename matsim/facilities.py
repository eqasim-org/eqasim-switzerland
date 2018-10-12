import gzip
from tqdm import tqdm
import data.constants as c
import numpy as np
import io
import matsim.writers

def configure(context, require):
    require.stage("data.statent.statent")

FIELDS = ["enterprise_id", "x", "y"]

def execute(context):
    cache_path = context.cache_path

    df_statent = context.stage("data.statent.statent")
    df_statent = df_statent[FIELDS]

    with gzip.open("%s/facilities.xml.gz" % cache_path, "w+") as f:
        with io.BufferedWriter(f, buffer_size = 1024  * 1024 * 1024 * 2) as raw_writer:
            writer = matsim.writers.FacilitiesWriter(raw_writer)
            writer.start_facilities()

            for item in tqdm(df_statent.itertuples(), total = len(df_statent)):
                writer.add_facility(item[1], item[2], item[3])

            writer.end_facilities()

    return "%s/facilities.xml.gz" % cache_path
