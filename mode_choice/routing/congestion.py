"""Optional congestion inputs and legacy hourly linkstats travel-time rules."""
from pathlib import Path

import polars as pl

from mode_choice.defaults import (
    ROUTING_DEFAULTS, HOURS_PER_DAY, SECONDS_PER_HOUR,
    LINKSTATS_TIME_COLUMNS, LINKSTATS_MIN_SPEED_M_PER_S,
)


def input_path(context, backend):
    key = "mode_choice_car_linkstats_path" if backend == "pandana" else "mode_choice_car_events_path"
    value = context.config(key, ROUTING_DEFAULTS[key])
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{key}: file does not exist: {path}")
    return path


def departure_hours(trips):
    times = trips["departure_time"]
    if times.null_count() or not times.is_finite().all() or (times < 0).any():
        raise ValueError("Departure times must be finite, nonnegative seconds")
    return ((times // SECONDS_PER_HOUR) % HOURS_PER_DAY).cast(pl.Int8).alias("departure_hour")


def read_linkstats(path, hours):
    """Read tab-separated MATSim linkstats (.gz supported), preserving link IDs."""
    columns = pl.read_csv(path, separator="\t", n_rows=0, infer_schema_length=0).columns
    if "LINK" not in columns or not any(name in columns for name in LINKSTATS_TIME_COLUMNS):
        raise ValueError("Linkstats must contain LINK and TRAVELTIME<h>-<h+1>avg columns")
    requested = [LINKSTATS_TIME_COLUMNS[hour] for hour in sorted(set(hours))
                 if LINKSTATS_TIME_COLUMNS[hour] in columns]
    frame = pl.read_csv(path, separator="\t", columns=["LINK", *requested],
                        schema_overrides={"LINK": pl.String, **{name: pl.Float64 for name in requested}},
                        null_values=["NaN", "nan", "NA", "null"]).rename({"LINK": "link_id"})
    if frame["link_id"].null_count() or frame["link_id"].n_unique() != frame.height:
        raise ValueError("Linkstats contains null or duplicate LINK IDs")
    return frame


def hourly_links(links, stats, hour):
    """Use observed seconds, falling back to free flow for missing/invalid data.

    Preserve the old branch's free-flow lower bound and 1 km/h upper bound,
    without rounding away precision. Parallel links are retained until the
    router selects the fastest link for this particular hour.
    """
    free = pl.col("length") / pl.col("freespeed")
    column = LINKSTATS_TIME_COLUMNS[hour]
    if column not in stats.columns:
        return links.with_columns(free.alias("travel_time"))
    result = links.join(stats.select("link_id", pl.col(column).alias("observed_time")),
                        on="link_id", how="left", validate="m:1")
    observed = pl.col("observed_time")
    time = pl.when(observed.is_finite() & (observed > 0)).then(observed).otherwise(free)
    return result.with_columns(
        time.clip(lower_bound=free,
                  upper_bound=pl.max_horizontal(free, pl.col("length") / LINKSTATS_MIN_SPEED_M_PER_S))
        .alias("travel_time")
    ).drop("observed_time")
