"""Private-household controls from the same 2025 reference vintage as population."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("synpp")
PROJECTION_FILE = "projections/households/households_2025.csv"


def configure(context):
    context.config("data_path")
    context.config("scaling_year")
    context.config("enable_scaling")


def prepare_controls(df, scaling_year):
    """Select private-household counts, excluding person counts and national totals.

    Keep the official SDMX columns in the input so scenario, measure and scope
    remain reviewable. The previous AM-00-2020 workbook is not a fallback: mixing
    vintages silently would reintroduce incompatible household/population targets.
    """
    required = ["SCENARIO", "CANTON", "SIZE_HOUSEHOLD", "NUMBER_HOUSEHOLD_PERSON",
                "FREQ", "TIME_HORIZ", "TIME_PERIOD", "OBS_VALUE"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError("Household projections require BFS DF_SCEN_MENAGES columns: %s" % missing)
    year = pd.to_numeric(df["TIME_PERIOD"], errors="raise")
    selected = (df["SCENARIO"].eq("AM-00-2025") & df["SIZE_HOUSEHOLD"].eq("_T")
                & df["NUMBER_HOUSEHOLD_PERSON"].eq("MEN") & df["FREQ"].eq("A")
                & df["TIME_HORIZ"].eq("P") & year.eq(scaling_year))
    out = df.loc[selected, ["CANTON", "OBS_VALUE"]].copy()
    out = out.loc[out["CANTON"].astype(str).ne("_T")]
    out.columns = ["canton_id", "weight"]
    out["canton_id"] = pd.to_numeric(out["canton_id"], errors="raise").astype(int)
    out["weight"] = pd.to_numeric(out["weight"], errors="raise")
    if out["canton_id"].duplicated().any() or set(out["canton_id"]) != set(range(1, 27)):
        raise ValueError("AM-00-2025 requires one private-household target per canton for year %s" % scaling_year)
    if (not np.isfinite(out["weight"]).all() or (out["weight"] < 0).any()
            or out["weight"].ne(out["weight"].round()).any()):
        raise ValueError("Household projection weights must be finite non-negative integers")
    out["weight"] = out["weight"].astype("int64")
    return out.sort_values("canton_id").reset_index(drop=True)


def execute(context):
    if not context.config("enable_scaling"):
        logger.info("Skipping projecting households as scaling is disabled!")
        return
    year = context.config("scaling_year")
    df = pd.read_csv("%s/%s" % (context.config("data_path"), PROJECTION_FILE))
    return prepare_controls(df, year), year
