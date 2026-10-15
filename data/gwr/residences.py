"""Public GWR residence candidates, including explicit names and FMEL addresses."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_REFERENCE = Path(__file__).with_name("residence_candidates.csv")


def configure(context):
    context.config("gwr_residence_candidates", str(DEFAULT_REFERENCE))


def validate(context):
    # Content hashing detects updates to a reference at the same configured path.
    return hashlib.sha256(Path(context.config("gwr_residence_candidates")).read_bytes()).hexdigest()


def execute(context):
    df = pd.read_csv(context.config("gwr_residence_candidates"))
    required = {"EGID", "GKODE", "GKODN", "GSTAT", "GKLAS", "evidence", "GEXPDAT"}
    if not required.issubset(df):
        raise ValueError("GWR residence reference is missing required fields")
    xy = df[["GKODE", "GKODN"]].to_numpy(dtype=float)
    if df.EGID.duplicated().any() or not df.GSTAT.eq(1004).all() or not (
            np.isfinite(xy).all() and (xy > 0).all()):
        raise ValueError("GWR residence candidates require unique existing geocoded buildings")
    return df
