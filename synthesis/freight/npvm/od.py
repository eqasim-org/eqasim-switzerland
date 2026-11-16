"""Read NPVM daily heavy-goods matrices and sample integer OD trip counts."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("synpp")
MATRIX_OPTIONS = {"LW": ("npvm_freight_lw_file", "npvm/LW_NPVM_2023.mtx"),
                  "LZ": ("npvm_freight_lz_file", "npvm/LZ_NPVM_2023.mtx")}


def configure(context):
    context.config("data_path")
    context.config("random_seed")
    context.config("input_downsampling")
    context.config("npvm_freight_demand_factor", default=1.0)
    for option, default in MATRIX_OPTIONS.values():
        context.config(option, default=default)


def execute(context):

    # Load the population fraction and an explicit adjustment of the supplied daily demand.

    fraction = float(context.config("input_downsampling"))
    factor = float(context.config("npvm_freight_demand_factor"))
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("NPVM freight requires 0 < input_downsampling <= 1")
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("npvm_freight_demand_factor must be finite and nonnegative")
    rng = np.random.default_rng(context.config("random_seed"))
    parts = []

    # Read LW and LZ separately and round fractional OD counts reproducibly.

    for source, (option, _) in MATRIX_OPTIONS.items():
        path = Path(context.config("data_path")) / context.config(option)
        od = read_matrix(path)
        scaled = od["demand"].to_numpy() * fraction * factor
        if not np.isfinite(scaled).all() or (scaled >= np.iinfo(np.int64).max).any():
            raise ValueError(f"Invalid or overflowing scaled demand in {path}")
        floors = np.floor(scaled)
        od["n_trips"] = floors.astype(np.int64) + (rng.random(len(od)) < scaled - floors)
        od["source"] = source
        logger.info("NPVM %s: %.2f daily vehicles, %.2f after scaling, %d sampled trips",
                    source, od.demand.sum(), scaled.sum(), od.n_trips.sum())
        parts.append(od.loc[od.n_trips.gt(0)].copy())

    # Return one row per sampled OD pair, preserving the source vehicle category.

    return pd.concat(parts, ignore_index=True)


def validate(context):
    """Invalidate cached counts if either source matrix changes in place."""
    tokens = []
    for option, _ in MATRIX_OPTIONS.values():
        path = Path(context.config("data_path")) / context.config(option)
        stat = path.stat()
        tokens.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    return tokens


def read_matrix(path):
    """Read VISUM $O text, applying its factor and excluding the $NAMES footer."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing NPVM freight matrix: {path}")
    with path.open(encoding="utf-8-sig") as stream:
        header = next(stream, "").strip()
        if header not in {"$O", "$O;D4"}:
            raise ValueError(f"Expected a VISUM $O matrix, got {header!r} in {path}")
        metadata = []
        first_row = None
        count = 0
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text or text.startswith("*"):
                continue
            if text.startswith("$"):
                if text != "$NAMES":
                    raise ValueError(f"Unsupported VISUM section {text!r} in {path}")
                break
            if len(metadata) < 2:
                metadata.append(text)
                continue
            if first_row is None:
                first_row = line_number
            count += 1
    if len(metadata) != 2:
        raise ValueError(f"Missing time window or matrix factor in {path}")
    if [float(x) for x in metadata[0].split()] != [0.0, 24.0]:
        raise ValueError(f"Expected daily 00:00–24:00 demand in {path}")
    factor = float(metadata[1])
    if not np.isfinite(factor) or factor < 0:
        raise ValueError(f"Invalid matrix factor in {path}")
    if count == 0:
        return pd.DataFrame({"origin_zone": pd.Series(dtype="int64"),
                             "destination_zone": pd.Series(dtype="int64"),
                             "demand": pd.Series(dtype="float64")})
    od = pd.read_csv(path, encoding="utf-8-sig", sep=r"\s+", skiprows=first_row,
                     nrows=count, comment="*", header=None,
                     names=["origin_zone", "destination_zone", "demand"],
                     dtype={"origin_zone": "int64", "destination_zone": "int64", "demand": "float64"})
    if (od[["origin_zone", "destination_zone"]] <= 0).any().any():
        raise ValueError(f"NPVM zone IDs must be positive integers in {path}")
    if not np.isfinite(od.demand).all() or od.demand.lt(0).any():
        raise ValueError(f"Negative or nonfinite NPVM demand in {path}")
    od["demand"] *= factor
    if not np.isfinite(od.demand).all():
        raise ValueError(f"Nonfinite NPVM demand after applying the matrix factor in {path}")
    # Sum duplicate pairs before rounding; sort keys to make row order stable.
    return od.groupby(["origin_zone", "destination_zone"], as_index=False, sort=True).demand.sum()
