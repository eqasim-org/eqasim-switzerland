import numpy as np
import pandas as pd

import lemanis

CONFIDENCE_Z = 1.96  # ~95% CI assuming a normal distribution


def _with_totals(tpg_mofr):
    df = tpg_mofr.copy()
    df["tpg_total_mean"] = df["boardings_raw_mean"] + df["alightings_raw_mean"]
    df["tpg_total_var"]  = df["boardings_raw_std"] ** 2 + df["alightings_raw_std"] ** 2
    df["matsim_total"]   = df["boardings_matsim"] + df["alightings_matsim"]
    return df


def _add_confidence_interval(df, mean_col = "tpg_mean", var_col = "tpg_var"):
    df["tpg_std"] = np.sqrt(df[var_col].clip(lower = 0))
    df["tpg_lo"]  = (df[mean_col] - CONFIDENCE_Z * df["tpg_std"]).clip(lower = 0)
    df["tpg_hi"]  = df[mean_col] + CONFIDENCE_Z * df["tpg_std"]
    return df


def compute_matsim_pct_of_mean(matsim, mean):
    matsim = np.asarray(matsim, dtype = float)
    mean   = np.asarray(mean, dtype = float)
    denom  = np.maximum(mean, 1.0)

    return 100 * matsim / denom


def filter_active_stops(tpg_mofr, min_avg_events = 10, min_active_hours = 6):
    df = _with_totals(tpg_mofr)

    stop_hour_mean = df.groupby(["gtfs_code", "hour"])["tpg_total_mean"].sum()
    active_hours   = (stop_hour_mean > min_avg_events).groupby("gtfs_code").sum()
    active_stops   = active_hours[active_hours >= min_active_hours].index

    return tpg_mofr[tpg_mofr["gtfs_code"].isin(active_stops)]


def build_stop_hour_table(tpg_mofr, gtfs_stops, stops_in_perimeter):
    df = _with_totals(tpg_mofr)
    df = df[df["gtfs_code"].isin(stops_in_perimeter)]

    grouped = df.groupby(["gtfs_code", "stop_name", "hour"], as_index = False).agg(
        tpg_mean     = ("tpg_total_mean", "sum"),
        tpg_var      = ("tpg_total_var", "sum"),
        matsim_total = ("matsim_total", "sum"),
    )

    coords           = gtfs_stops[["stop_id", "stop_lat", "stop_lon"]].drop_duplicates("stop_id").copy()
    coords["stop_x"] = gtfs_stops.geometry.x  
    coords["stop_y"] = gtfs_stops.geometry.y

    stop_info = grouped[["gtfs_code", "stop_name"]].drop_duplicates("gtfs_code").merge(
        coords, left_on = "gtfs_code", right_on = "stop_id", how = "left"
    )

    full_index = pd.MultiIndex.from_product(
        [stop_info["gtfs_code"], range(24)], names = ["gtfs_code", "hour"]
    )

    grouped = grouped.set_index(["gtfs_code", "hour"]).reindex(full_index).drop(columns = "stop_name").reset_index()
    grouped = grouped.fillna({"tpg_mean": 0, "tpg_var": 0, "matsim_total": 0})
    grouped = grouped.merge(stop_info, on = "gtfs_code", how = "left")

    grouped = _add_confidence_interval(grouped)
    grouped["pct_of_mean"] = compute_matsim_pct_of_mean(grouped["matsim_total"], grouped["tpg_mean"])

    return grouped


def build_full_day_table(stop_hour_df):
    grouped = stop_hour_df.groupby(["gtfs_code", "stop_name", "stop_lat", "stop_lon", "stop_x", "stop_y"], as_index = False).agg(
        tpg_mean     = ("tpg_mean", "sum"),
        tpg_var      = ("tpg_var", "sum"),
        matsim_total = ("matsim_total", "sum"),
    )

    grouped = _add_confidence_interval(grouped)
    grouped["pct_of_mean"] = compute_matsim_pct_of_mean(grouped["matsim_total"], grouped["tpg_mean"])

    return grouped


def build_global_hourly_table(stop_hour_df):
    grouped = stop_hour_df.groupby("hour", as_index = False).agg(
        tpg_mean     = ("tpg_mean", "sum"),
        tpg_var      = ("tpg_var", "sum"),
        matsim_total = ("matsim_total", "sum"),
    )

    grouped = _add_confidence_interval(grouped)

    return grouped


def _split_line_direction(line_direction):
    s         = line_direction.astype(str)
    match     = s.str.extract(r"^(?P<line_base>.+)_(?P<direction_letter>[HR])$")
    line_base = match["line_base"].fillna(s)

    return line_base, match["direction_letter"]


def build_line_hour_table(tpg_mofr):
    df = _with_totals(tpg_mofr)

    grouped = df.groupby(["line_direction", "hour"], as_index = False).agg(
        tpg_mean     = ("tpg_total_mean", "sum"),
        tpg_var      = ("tpg_total_var", "sum"),
        matsim_total = ("matsim_total", "sum"),
    )

    full_index = pd.MultiIndex.from_product(
        [grouped["line_direction"].unique(), range(24)], names=["line_direction", "hour"]
    )
    grouped = grouped.set_index(["line_direction", "hour"]).reindex(full_index).reset_index()
    grouped = grouped.fillna({"tpg_mean": 0, "tpg_var": 0, "matsim_total": 0})

    grouped["line_base"], grouped["direction_letter"] = _split_line_direction(grouped["line_direction"])

    grouped = _add_confidence_interval(grouped)
    grouped["pct_of_mean"] = compute_matsim_pct_of_mean(grouped["matsim_total"], grouped["tpg_mean"])

    return grouped


def build_line_full_day_table(line_hour_df):
    grouped = line_hour_df.groupby(["line_direction", "line_base", "direction_letter"], as_index = False, dropna = False).agg(
        tpg_mean    = ("tpg_mean", "sum"),
        tpg_var     = ("tpg_var", "sum"),
        matsim_total= ("matsim_total", "sum"),
    )

    grouped = _add_confidence_interval(grouped)
    grouped["pct_of_mean"] = compute_matsim_pct_of_mean(grouped["matsim_total"], grouped["tpg_mean"])

    return grouped


def build_lemanis_stop_period_table(lemanis_mofr, gtfs_stops):
    """Same idea as build_stop_hour_table, but keyed by Lemanis' own period
    (see lemanis.to_period_shape) instead of hour, and without the
    perimeter-stop restriction: lemanis_mofr is already limited to
    Lemanis' own matched stops (see stages._build_lemanis_period_mofr)."""
    df = _with_totals(lemanis_mofr)

    grouped = df.groupby(["gtfs_code", "stop_name", "period"], as_index = False).agg(
        tpg_mean     = ("tpg_total_mean", "sum"),
        tpg_var      = ("tpg_total_var", "sum"),
        matsim_total = ("matsim_total", "sum"),
    )

    coords           = gtfs_stops[["stop_id", "stop_lat", "stop_lon"]].drop_duplicates("stop_id").copy()
    coords["stop_x"] = gtfs_stops.geometry.x
    coords["stop_y"] = gtfs_stops.geometry.y

    stop_info = grouped[["gtfs_code", "stop_name"]].drop_duplicates("gtfs_code").merge(
        coords, left_on = "gtfs_code", right_on = "stop_id", how = "left"
    )

    full_index = pd.MultiIndex.from_product(
        [stop_info["gtfs_code"], lemanis.PERIOD_ORDER], names = ["gtfs_code", "period"]
    )

    grouped = grouped.set_index(["gtfs_code", "period"]).reindex(full_index).drop(columns = "stop_name").reset_index()
    grouped = grouped.fillna({"tpg_mean": 0, "tpg_var": 0, "matsim_total": 0})
    grouped = grouped.merge(stop_info, on = "gtfs_code", how = "left")

    grouped = _add_confidence_interval(grouped)
    grouped["pct_of_mean"] = compute_matsim_pct_of_mean(grouped["matsim_total"], grouped["tpg_mean"])

    return grouped


def build_lemanis_stop_full_day_table(stop_period_df):
    grouped = stop_period_df.groupby(
        ["gtfs_code", "stop_name", "stop_lat", "stop_lon", "stop_x", "stop_y"], as_index = False
    ).agg(
        tpg_mean     = ("tpg_mean", "sum"),
        tpg_var      = ("tpg_var", "sum"),
        matsim_total = ("matsim_total", "sum"),
    )

    grouped = _add_confidence_interval(grouped)
    grouped["pct_of_mean"] = compute_matsim_pct_of_mean(grouped["matsim_total"], grouped["tpg_mean"])

    return grouped


def build_lemanis_line_period_table(lemanis_mofr):
    """Same shape/purpose as build_line_hour_table, but keyed by Lemanis'
    own reported period (see lemanis.to_period_shape) instead of hour -
    MATSim is aggregated into those same periods rather than Lemanis being
    split down to an hourly resolution it doesn't have (see
    stages._build_lemanis_period_mofr)."""
    df = _with_totals(lemanis_mofr)

    grouped = df.groupby(["line_direction", "period"], as_index = False).agg(
        tpg_mean     = ("tpg_total_mean", "sum"),
        tpg_var      = ("tpg_total_var", "sum"),
        matsim_total = ("matsim_total", "sum"),
    )

    full_index = pd.MultiIndex.from_product(
        [grouped["line_direction"].unique(), lemanis.PERIOD_ORDER], names = ["line_direction", "period"]
    )
    grouped = grouped.set_index(["line_direction", "period"]).reindex(full_index).reset_index()
    grouped = grouped.fillna({"tpg_mean": 0, "tpg_var": 0, "matsim_total": 0})

    grouped["line_base"], grouped["direction_letter"] = _split_line_direction(grouped["line_direction"])

    grouped = _add_confidence_interval(grouped)
    grouped["pct_of_mean"] = compute_matsim_pct_of_mean(grouped["matsim_total"], grouped["tpg_mean"])

    return grouped


def build_lemanis_line_full_day_table(line_period_df):
    grouped = line_period_df.groupby(["line_direction", "line_base", "direction_letter"], as_index = False, dropna = False).agg(
        tpg_mean     = ("tpg_mean", "sum"),
        tpg_var      = ("tpg_var", "sum"),
        matsim_total = ("matsim_total", "sum"),
    )

    grouped = _add_confidence_interval(grouped)
    grouped["pct_of_mean"] = compute_matsim_pct_of_mean(grouped["matsim_total"], grouped["tpg_mean"])

    return grouped


EXCLUDED_LINE_BASES = {f"C{i}" for i in range(1, 10)} | {"E+", "G+"}


def filter_line_map_lines(line_hour_df, max_zero_hours = 8):
    df = line_hour_df[~line_hour_df["line_base"].isin(EXCLUDED_LINE_BASES)]

    zero_hours    = df.groupby("line_direction")["tpg_mean"].apply(lambda s: (s == 0).sum())
    sparse_lines = zero_hours[zero_hours > max_zero_hours].index
    df           = df[~df["line_direction"].isin(sparse_lines)]

    matsim_totals   = df.groupby("line_direction")["matsim_total"].sum()
    unmatched_lines = matsim_totals[matsim_totals == 0].index

    return df[~df["line_direction"].isin(unmatched_lines)]


def build_line_base_table(line_full_day_df):
    grouped = line_full_day_df.groupby("line_base", as_index=False).agg(
        tpg_mean     = ("tpg_mean", "sum"),
        matsim_total = ("matsim_total", "sum"),
    )

    grouped["pct_of_mean"] = compute_matsim_pct_of_mean(grouped["matsim_total"], grouped["tpg_mean"])

    return grouped
