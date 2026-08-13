"""
Stage: analysis.cross_border.plans

Reads back the population written by matsim.scenario.population and shows what
the plans of the border-crossing agents actually look like. Everyone flagged
with the isCrossBorder attribute is kept -- that covers both the foreign
population from data.cross_border.generate_cross_border_traffic and the Swiss
residents that synthesis.population.models.cross_border sent across the border.

Reading the written XML rather than the upstream data frames is deliberate: it
is the file MATSim will be given, so anything lost or mangled on the way out
shows up here.

Produces, in <analysis_path>/cross_border_plans:
  - plans_map.png       the sampled plans drawn on the Swiss border, by mode
  - plans_summary.png   chain lengths, modes, departure times, chains, ODs
  - plans.gpkg          the same sample as legs (lines) and activities (points)
                        for inspection in QGIS

Pipeline config keys
--------------------
cross_border_plans_sample_size (default 300)
    How many agents are drawn on the map and written to the gpkg. The summary
    figures always use every border-crossing agent.
"""

import os
import logging
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point
import xopen

logger = logging.getLogger("synpp")

MODE_COLORS = {
    "car":           "#1f77b4",
    "car_passenger": "#9467bd",
    "pt":            "#ff7f0e",
    "walk":          "#2ca02c",
    "bike":          "#d62728",
}
OTHER_MODE_COLOR = "#7f7f7f"

CRS = "EPSG:2056"


def configure(context):
    context.config("analysis_path")
    context.config("random_seed")
    context.config("cross_border_plans_sample_size", default = 300)

    context.stage("matsim.scenario.population")
    context.stage("data.spatial.swiss_border")


def execute(context):
    population_path = context.stage("matsim.scenario.population")

    output_path = os.path.join(context.config("analysis_path"), "cross_border_plans")
    os.makedirs(output_path, exist_ok = True)

    df_persons, df_activities, df_legs = read_cross_border_plans(context, population_path)

    if len(df_persons) == 0:
        logger.warning("No agent carries isCrossBorder=true in %s; nothing to plot.", population_path)
        return output_path

    logger.info(
        "Read %d border-crossing agents: %d activities, %d legs.",
        len(df_persons), len(df_activities), len(df_legs),
    )

    df_trips = build_trips(df_activities, df_legs)

    # The map and the gpkg would be unreadable (and huge) with every agent, so
    # they show a reproducible random sample; the figures below use them all.
    sample_size = min(int(context.config("cross_border_plans_sample_size")), len(df_persons))
    rng         = np.random.RandomState(context.config("random_seed"))
    sampled_ids = rng.choice(df_persons["person_id"].values, size = sample_size, replace = False)

    df_trips_sample      = df_trips[df_trips["person_id"].isin(sampled_ids)]
    df_activities_sample = df_activities[df_activities["person_id"].isin(sampled_ids)]

    plot_map(context, df_trips_sample, df_activities_sample, len(df_persons),
             os.path.join(output_path, "plans_map.png"))

    plot_summary(df_persons, df_activities, df_legs,
                 os.path.join(output_path, "plans_summary.png"))

    write_gpkg(df_trips_sample, df_activities_sample, os.path.join(output_path, "plans.gpkg"))

    logger.info("Wrote the border-crossing plan analysis to %s", output_path)

    return output_path


# ---------------------------------------------------------------------------
# Reading the written population
# ---------------------------------------------------------------------------

def parse_time(value):
    """MATSim writes HH:MM:SS, with hours running past 24 for the last activity."""

    if value is None:
        return np.nan

    hours, minutes, seconds = value.split(":")

    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def read_cross_border_plans(context, population_path):
    """
    Streams the population file and keeps the plan of every person whose
    isCrossBorder attribute is true. The file holds the whole synthetic
    population, so it is parsed incrementally and the tree is dropped as it
    goes -- building it in memory is not an option at this scale.
    """

    persons, activities, legs = [], [], []

    person_id       = None
    attributes      = {}
    is_crossborder  = False
    in_plan         = False
    activity_index  = 0
    leg_index       = 0

    with xopen.xopen(population_path, "r") as f:
        parser   = ET.iterparse(f, events = ["start", "end"])
        _, root  = next(parser)  # <population>, kept to release finished persons

        for event, elem in parser:
            tag = elem.tag.rpartition("}")[2]

            if tag == "person":
                if event == "start":
                    person_id      = elem.attrib["id"]
                    attributes     = {}
                    is_crossborder = False
                    in_plan        = False
                    activity_index = 0
                    leg_index      = 0
                else:
                    if is_crossborder:
                        persons.append(dict(person_id = person_id, **attributes))

                    root.clear()  # otherwise the finished persons pile up

            elif tag == "plan" and event == "start":
                in_plan = True

            elif tag == "attribute" and event == "end" and not in_plan:
                # Activities carry attributes of their own; only the ones
                # before the plan describe the person.
                attributes[elem.attrib["name"]] = elem.text

                if elem.attrib["name"] == "isCrossBorder":
                    is_crossborder = elem.text == "true"

            elif tag == "activity" and event == "start" and is_crossborder:
                activities.append(dict(
                    person_id      = person_id,
                    activity_index = activity_index,
                    purpose        = elem.attrib.get("type"),
                    x              = float(elem.attrib["x"]),
                    y              = float(elem.attrib["y"]),
                    facility_id    = elem.attrib.get("facility"),
                    start_time     = parse_time(elem.attrib.get("start_time")),
                    end_time       = parse_time(elem.attrib.get("end_time")),
                ))
                activity_index += 1

            elif tag == "leg" and event == "start" and is_crossborder:
                legs.append(dict(
                    person_id      = person_id,
                    leg_index      = leg_index,
                    mode           = elem.attrib.get("mode"),
                    departure_time = parse_time(elem.attrib.get("dep_time")),
                    travel_time    = parse_time(elem.attrib.get("trav_time")),
                ))
                leg_index += 1

    df_persons    = pd.DataFrame(persons)
    df_activities = pd.DataFrame(activities)
    df_legs       = pd.DataFrame(legs)

    if len(df_activities) > 0:
        df_activities = df_activities.sort_values(["person_id", "activity_index"])
    if len(df_legs) > 0:
        df_legs = df_legs.sort_values(["person_id", "leg_index"])

    return df_persons, df_activities, df_legs


def build_trips(df_activities, df_legs):
    """
    Pairs every activity with the following one, which is the trip the leg of
    the same index describes.
    """

    df = df_activities.copy()

    df["destination_x"]       = df.groupby("person_id")["x"].shift(-1)
    df["destination_y"]       = df.groupby("person_id")["y"].shift(-1)
    df["following_purpose"]   = df.groupby("person_id")["purpose"].shift(-1)

    df = df[df["destination_x"].notna()].copy()
    df = df.rename(columns = {"x": "origin_x", "y": "origin_y",
                              "purpose": "preceding_purpose",
                              "activity_index": "leg_index"})

    df = df.merge(df_legs[["person_id", "leg_index", "mode", "departure_time", "travel_time"]],
                  on = ["person_id", "leg_index"], how = "left")

    return df[["person_id", "leg_index", "mode", "preceding_purpose", "following_purpose",
               "origin_x", "origin_y", "destination_x", "destination_y",
               "departure_time", "travel_time"]]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def mode_color(mode):
    return MODE_COLORS.get(mode, OTHER_MODE_COLOR)


def plot_map(context, df_trips, df_activities, total_persons, output_file):
    """Draws the sampled plans over the Swiss border, one line per leg."""

    swiss_border = gpd.GeoSeries(context.stage("data.spatial.swiss_border"))

    if swiss_border.crs is None:
        swiss_border = swiss_border.set_crs(CRS)

    fig, ax = plt.subplots(figsize = (11, 8))

    swiss_border.boundary.plot(ax = ax, color = "#4A4A4A", linewidth = 0.6, zorder = 1)

    for mode, group in df_trips.groupby(df_trips["mode"].fillna("unknown")):
        segments = np.stack([
            group[["origin_x", "origin_y"]].values,
            group[["destination_x", "destination_y"]].values,
        ], axis = 1)

        ax.add_collection(LineCollection(
            segments, colors = mode_color(mode), linewidths = 0.7, alpha = 0.5, zorder = 2))

    # The border activities are the crossing points themselves, so they are
    # worth telling apart from the ordinary ends of a trip.
    is_border = df_activities["purpose"] == "border"

    ax.scatter(df_activities.loc[~is_border, "x"], df_activities.loc[~is_border, "y"],
               s = 3, color = "#333333", alpha = 0.5, zorder = 3)
    ax.scatter(df_activities.loc[is_border, "x"], df_activities.loc[is_border, "y"],
               s = 14, color = "#D1495B", marker = "x", linewidths = 0.8, zorder = 4)

    handles = [Line2D([], [], color = mode_color(m), lw = 1.5, label = m)
               for m in sorted(df_trips["mode"].fillna("unknown").unique())]
    handles.append(Line2D([], [], color = "#D1495B", marker = "x", lw = 0, label = "border activity"))
    ax.legend(handles = handles, loc = "upper left", frameon = False)

    ax.set_title("Plans of border-crossing agents (%d of %d agents shown)"
                 % (df_trips["person_id"].nunique(), total_persons))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_file, dpi = 150)
    plt.close(fig)


def plot_summary(df_persons, df_activities, df_legs, output_file):
    fig, axes = plt.subplots(2, 3, figsize = (16, 9))

    # 1. Activities per plan
    counts = df_activities.groupby("person_id").size().value_counts().sort_index()
    axes[0, 0].bar(counts.index, counts.values, color = "#1D4E89")
    axes[0, 0].set_title("Activities per plan")
    axes[0, 0].set_xlabel("number of activities")
    axes[0, 0].set_ylabel("agents")

    # 2. Modes
    modes = df_legs["mode"].fillna("unknown").value_counts()
    axes[0, 1].bar(modes.index, modes.values, color = [mode_color(m) for m in modes.index])
    axes[0, 1].set_title("Legs by mode")
    axes[0, 1].set_ylabel("legs")
    axes[0, 1].tick_params(axis = "x", rotation = 30)

    # 3. Departure times
    departures = df_legs["departure_time"].dropna() / 3600.0
    axes[0, 2].hist(departures, bins = np.arange(0, 31, 0.5), color = "#1D4E89")
    axes[0, 2].set_title("Leg departure times")
    axes[0, 2].set_xlabel("hour of day")
    axes[0, 2].set_ylabel("legs")

    # 4. Activity chains
    chains = (df_activities.groupby("person_id")["purpose"]
              .apply(lambda p: "-".join(p.astype(str)))
              .value_counts().head(10).sort_values())
    axes[1, 0].barh(range(len(chains)), chains.values, color = "#1D4E89")
    axes[1, 0].set_yticks(range(len(chains)))
    axes[1, 0].set_yticklabels(chains.index, fontsize = 8)
    axes[1, 0].set_title("Most frequent activity chains")
    axes[1, 0].set_xlabel("agents")

    # 5. OD relations
    if "crossBorderOD" in df_persons.columns:
        ods = df_persons["crossBorderOD"].fillna("unknown").value_counts().head(10).sort_values()
        axes[1, 1].barh(range(len(ods)), ods.values, color = "#D1495B")
        axes[1, 1].set_yticks(range(len(ods)))
        axes[1, 1].set_yticklabels(ods.index, fontsize = 8)
        axes[1, 1].set_xlabel("agents")
    else:
        axes[1, 1].text(0.5, 0.5, "no crossBorderOD attribute", ha = "center", va = "center")
    axes[1, 1].set_title("Most frequent OD relations")

    # 6. Activity purposes
    purposes = df_activities["purpose"].fillna("unknown").value_counts()
    axes[1, 2].bar(purposes.index, purposes.values, color = "#1D4E89")
    axes[1, 2].set_title("Activity purposes")
    axes[1, 2].set_ylabel("activities")
    axes[1, 2].tick_params(axis = "x", rotation = 30)

    fig.suptitle("Border-crossing agents in the written population (%d agents)" % len(df_persons))
    fig.tight_layout()
    fig.savefig(output_file, dpi = 150)
    plt.close(fig)


def write_gpkg(df_trips, df_activities, output_file):
    """The sampled plans as geometries, to be stepped through in QGIS."""

    legs = gpd.GeoDataFrame(
        df_trips.copy(),
        geometry = [LineString([(o_x, o_y), (d_x, d_y)]) for o_x, o_y, d_x, d_y in zip(
            df_trips["origin_x"], df_trips["origin_y"],
            df_trips["destination_x"], df_trips["destination_y"])],
        crs = CRS,
    )

    activities = gpd.GeoDataFrame(
        df_activities.copy(),
        geometry = [Point(x, y) for x, y in zip(df_activities["x"], df_activities["y"])],
        crs = CRS,
    )

    if os.path.exists(output_file):
        os.remove(output_file)  # layers would otherwise be appended to the old file

    legs.to_file(output_file, layer = "legs", driver = "GPKG")
    activities.to_file(output_file, layer = "activities", driver = "GPKG")
