import os
import polars as pl

"""
This stage loads the raw data from the French service registry.
"""

def configure(context):
    context.config("data_path")


def execute(context):

    dpts_str = ["01", "25", "39", "73", "74"] #[str(dep) for dep in requested_departements]

    with context.progress(label = "Reading BPE ...") as progress:
        parquet = pl.read_parquet("{}/{}".format(context.config("data_path"), "other_locations/FR/bpe_2024/BPE24.parquet"),
         columns = [ "CAPACITE",
                        "DCIRIS", "LAMBERT_X", "LAMBERT_Y",
                        "TYPEQU", "DEPCOM", "DEP", "SIRET"
                    ],
                )

        parquet  = parquet.cast( dict(DEPCOM = str, DEP = str, DCIRIS = str, SIRET = str))
        parquet  = parquet.filter(pl.col("DEP").cast(pl.Utf8).is_in(dpts_str))

        progress.update(len(parquet))

    return parquet.to_pandas()


def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), "other_locations/FR/bpe_2024/BPE24.parquet")):
        raise RuntimeError("BPE data is not available")

    return os.path.getsize("%s/%s" % (context.config("data_path"), "other_locations/FR/bpe_2024/BPE24.parquet"))
