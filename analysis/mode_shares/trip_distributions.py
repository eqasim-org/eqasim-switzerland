"""Per-mode distributions from the simulation trip export, without survey weighting."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.counts.paths import configure_simulation_path, get_simulation_path
from matsim.config.tools import find_simulation_output


logger = logging.getLogger("synpp")
MODES = ["car_passenger", "car", "walk", "pt", "bike"]
METRICS = {
    "travel_time_min": ("Travel time", "min"),
    "euclidean_distance_km": ("Euclidean distance", "km"),
    "routed_distance_km": ("Routed distance", "km"),
}


def configure(context):
    configure_simulation_path(context)
    context.config("analysis.mode_shares.travel_time_bin_min", default=1.0)
    context.config("analysis.mode_shares.distance_bin_km", default=1.0)


def execute(context):

    # Load only the trip columns needed for these distributions.

    simulation = Path(get_simulation_path(context))
    path = find_simulation_output(simulation, "output_trips.csv.gz")
    trips = read_trips(path)
    time_bin = context.config("analysis.mode_shares.travel_time_bin_min")
    distance_bin = context.config("analysis.mode_shares.distance_bin_km")
    for value in (time_bin, distance_bin):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Trip distribution bin widths must be finite and positive.")

    # Save plots alongside the other mode-share figures from this simulation.

    output = simulation / "mode_shares"
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for column, (label, unit) in METRICS.items():
        if column not in trips:
            logger.info("Skipping %s: not present in %s", label, path)
            continue
        figure = plot_distribution(trips, column, label, unit,
                                   time_bin if column == "travel_time_min" else distance_bin)
        if figure is None:
            logger.warning("Skipping %s: no valid non-negative observations", label)
            continue
        destination = output / f"trip_{column}_by_mode.png"
        figure.savefig(destination, dpi=200, bbox_inches="tight")
        plt.close(figure)
        files.append(str(destination))

    # Return generated figure paths.

    return {"path": str(output), "figures": files}


def read_trips(path):
    columns = pd.read_csv(path, sep=";", nrows=0).columns
    required = {"main_mode", "trav_time", "euclidean_distance"}
    if not required.issubset(columns):
        raise ValueError(f"Trip export is missing columns: {sorted(required - set(columns))}")
    selected = [*sorted(required), *(["traveled_distance"] if "traveled_distance" in columns else [])]
    trips = pd.read_csv(path, sep=";", usecols=selected)
    trips = trips.loc[trips.main_mode.isin(MODES)].copy()
    # Durations, not clock times: preserve trips lasting more than 24 hours.
    trips["travel_time_min"] = pd.to_timedelta(trips.trav_time, errors="coerce").dt.total_seconds() / 60
    trips["euclidean_distance_km"] = pd.to_numeric(trips.euclidean_distance, errors="coerce") / 1000
    if "traveled_distance" in trips:
        trips["routed_distance_km"] = pd.to_numeric(trips.traveled_distance, errors="coerce") / 1000
    for column in METRICS:
        if column in trips:
            valid = np.isfinite(trips[column]) & trips[column].ge(0)
            logger.info("%s: excluding %d invalid values", column, (~valid).sum())
            trips.loc[~valid, column] = np.nan
    return trips


def plot_distribution(trips, column, label, unit, bin_width):
    values = trips[column].dropna()
    if values.empty:
        return None
    xmax = max(bin_width, np.ceil(values.quantile(.99) / bin_width) * bin_width)
    bins = np.arange(0, xmax + bin_width * .5, bin_width)
    palette = plt.cm.Set2(np.linspace(0, 1, len(MODES)))
    with plt.rc_context({"font.family": "sans-serif", "axes.spines.top": False,
                         "axes.spines.right": False}):
        figure, axes = plt.subplots(1, len(MODES), figsize=(20, 3.8),
                                    sharex=True, sharey=True, constrained_layout=True)
        for ax, mode, color in zip(axes, MODES, palette):
            data = trips.loc[trips.main_mode.eq(mode), column].dropna()
            counts, edges = np.histogram(data, bins=bins)
            ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
                   color=color, alpha=.35, edgecolor="white", linewidth=.8)
            if len(data):
                mean = data.mean()  # Include the tail in the mean, unlike the display range.
                if mean <= xmax:
                    solid = ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
                                   color=color, edgecolor="white", linewidth=.8)
                    clip = plt.Rectangle((mean, 0), xmax - mean, max(counts.max(), 1) * 2,
                                         transform=ax.transData)
                    for bar in solid:
                        bar.set_clip_path(clip)
                    ax.axvline(mean, color="#222", lw=1.6, ls="--")
                ax.text(.98, 1.02, f"mean {mean:.1f} {unit}" + (" (outside view)" if mean > xmax else ""),
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
            else:
                ax.text(.5, .5, "No valid trips", transform=ax.transAxes, ha="center")
            ax.set_xlim(0, xmax)
            ax.set_title(mode, fontsize=12, fontweight="bold", pad=22, loc="left")
            ax.set_xlabel(f"{label} ({unit})")
            ax.yaxis.grid(True, color="#e6e6e6", lw=.8)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            ax.spines[["left", "bottom"]].set_color("#bbb")
        axes[0].set_ylabel("Number of trips")
        figure.supxlabel("Display limited to pooled 99th percentile; means include all valid trips.", fontsize=9)
    return figure
