"""Compare peak totals using hourly linkstats, never daily count files."""

import glob
import logging
import os

import numpy as np
import pandas as pd

from .compare import Compare
from ..cantons.transcality_peak_hour import configure_peak_hours, get_peak_hours
from ..paths import configure_simulation_path, get_simulation_path


def configure(context):
    configure_simulation_path(context)
    configure_peak_hours(context)


def read_peak_linkstats(path, start, end):
    # MATSim linkstats are usually tab-delimited .txt.gz; also accept CSV.
    header = pd.read_csv(path, sep=None, engine="python", nrows=0)
    columns = [f"HRS{hour}-{hour + 1}avg" for hour in range(start, end)]
    missing = {"LINK", *columns}.difference(header.columns)
    if missing:
        raise ValueError(f"Peak linkstats is missing columns: {sorted(missing)}")
    # Detect delimiter once; use the fast parser for the full network.
    import csv
    import gzip
    opener = gzip.open if os.fspath(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        delimiter = csv.Sniffer().sniff(stream.readline(), delimiters="\t,;").delimiter
    data = pd.read_csv(path, sep=delimiter, usecols=["LINK", *columns], dtype={"LINK": str})
    values = data[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError("Peak linkstats must contain finite, non-negative hourly flows.")
    if not data["LINK"].is_unique:
        raise ValueError("Peak linkstats contains duplicate link IDs.")
    return pd.DataFrame({"link_id": data["LINK"], "flow": values.sum(axis=1)})


def execute(context):
    start, end = get_peak_hours(context)
    root = get_simulation_path(context)
    files = []
    for pattern in ("*.linkstats.txt.gz", "*.linkstats.txt", "*.linkstats.csv.gz", "*.linkstats.csv"):
        files.extend(glob.glob(os.path.join(root, "ITERS", "it.*", pattern)))
        files.extend(glob.glob(os.path.join(root, pattern)))
    if not files:
        raise FileNotFoundError(f"No hourly linkstats files found in {root}")
    path = max(files, key=os.path.getmtime)
    logging.getLogger("synpp").info("Using peak-hour linkstats %s (%02d:00–%02d:00)", path, start, end)
    comparison = Compare()
    comparison.link_stats = read_peak_linkstats(path, start, end)
    return comparison
