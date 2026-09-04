"""
Stage: data.cross_border.cross_border_points_counts

Counts, per border crossing point, how many border-purpose activities are
attributed to it, split into three categories of person:

  - swiss_resident: Swiss residents whom synthesis.population.models.cross_border
    sent across the border (one trip per person, from synthesis.population.trips).
  - cross_border_not_teleported: foreign cross-border agents (data.cross_border.*)
    whose home (or, for "Through" trips, either end) lies close enough to the
    border (<20km) that it was used as-is.
  - cross_border_teleported: foreign cross-border agents whose home (or,
    for "Through" trips, either end) was too far from the border (>20km,
    see data.cross_border.generate_od.project_point_series_close_to_border)
    and got snapped onto the nearest matching-mode interview place instead -
    i.e. the survey respondent's real, distant origin is not represented;
    their trip effectively teleports them to the border.

This intentionally reads the pre-write data.cross_border.* /
synthesis.population.trips frames rather than the written MATSim population
(unlike analysis.cross_border.plans): those frames already carry every
column this needs, which avoids re-deriving the person_id renumbering that
data.cross_border.generate_cross_border_traffic applies before MATSim I/O.

Produces, in the stage's cache folder (context.path()):
  - cross_border_points_map.html   interactive map with one togglable layer
                                    per category (folium checkboxes); only
                                    "cross_border_teleported" is checked by
                                    default.

Returns a long-format DataFrame: one row per
(border_crossing_point_id, category) with the activity count.

Counts are scaled by 1 / input_downsampling, so both the returned frame and
the map represent the full population regardless of how much the run was
downsampled - the same convention analysis/pt/stages.py uses for MATSim
boardings/alightings. This is valid here because input_downsampling drives
both populations' sampling: synthesis.population.sampled for Swiss
residents (synthesis.population.trips is downstream of it) and
data.cross_border.sample for foreign cross-border agents.
"""

import logging
import os

import folium
import pandas as pd

logger = logging.getLogger("synpp")

CATEGORIES = ["swiss_resident", "cross_border_not_teleported", "cross_border_teleported"]

CATEGORY_LABELS = {
    "swiss_resident": "Swiss residents",
    "cross_border_not_teleported": "Cross-border people (not teleported)",
    "cross_border_teleported": "Cross-border people (teleported)",
}

CATEGORY_COLORS = {
    "swiss_resident": "#1f77b4",
    "cross_border_not_teleported": "#2ca02c",
    "cross_border_teleported": "#d62728",
}

# Only this category's layer is checked by default on the map.
DEFAULT_VISIBLE_CATEGORY = "cross_border_teleported"

# Free, no-key WMTS basemap - see analysis/pt/interactive_map.py's module
# docstring for why plain OSM/cartodbpositron tiles are not used here.
_TILE_URL = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"


def configure(context):
    context.config("input_downsampling")

    context.stage("data.cross_border.interview_places")
    context.stage("data.cross_border.destinations")
    context.stage("data.cross_border.activities")
    context.stage("synthesis.population.trips")


def execute(context):
    df_points      = context.stage("data.cross_border.interview_places").copy()
    df_trips       = context.stage("synthesis.population.trips")
    df_projected   = context.stage("data.cross_border.destinations")[
        ["cross_border_person_id", "is_border_point_projected"]
    ].drop_duplicates("cross_border_person_id")
    df_activities  = context.stage("data.cross_border.activities")

    df_counts = pd.concat([
        count_swiss_residents(df_trips),
        count_cross_border_people(df_activities, df_projected),
    ], ignore_index = True)

    df_counts = fill_missing_combinations(df_counts, df_points["border_crossing_point_id"])

    # Scale sampled counts back up to the full population - see module docstring.
    df_counts["count"] = df_counts["count"] / context.config("input_downsampling")

    logger.info(
        "Border crossing activity counts (scaled to full population): %s",
        df_counts.groupby("category")["count"].sum().to_dict(),
    )

    build_map(df_points, df_counts, os.path.join(context.path(), "cross_border_points_map.html"))

    return df_counts


def count_swiss_residents(df_trips):
    """
    One row per Swiss resident whose trip crosses the border
    (synthesis.population.trips only fills interview_point_id for those),
    grouped by crossing point.
    """

    df = df_trips[df_trips["interview_point_id"].notna()]

    counts = df.groupby("interview_point_id").size().reset_index(name = "count")
    counts = counts.rename(columns = {"interview_point_id": "border_crossing_point_id"})
    counts["category"] = "swiss_resident"

    return counts[["border_crossing_point_id", "category", "count"]]


def count_cross_border_people(df_activities, df_projected):
    """
    One row per border-purpose activity of a foreign cross-border agent
    (data.cross_border.activities), grouped by crossing point and by
    whether the person's own origin/destination was teleported onto the
    border (see module docstring) - a person-level flag from
    data.cross_border.destinations, joined in by cross_border_person_id
    (== activities' person_id, before
    data.cross_border.generate_cross_border_traffic renumbers it for MATSim).
    """

    df = df_activities[df_activities["purpose"] == "border"]
    df = df.merge(df_projected, left_on = "person_id", right_on = "cross_border_person_id", how = "left")

    assert df["is_border_point_projected"].notna().all(), (
        "Some cross-border border-activities belong to a person missing from "
        "data.cross_border.destinations."
    )

    df["category"] = df["is_border_point_projected"].map({
        True: "cross_border_teleported", False: "cross_border_not_teleported",
    })

    counts = df.groupby(["destination_id", "category"]).size().reset_index(name = "count")
    counts = counts.rename(columns = {"destination_id": "border_crossing_point_id"})

    return counts[["border_crossing_point_id", "category", "count"]]


def fill_missing_combinations(df_counts, point_ids):
    """Every point gets a zero row per category too, so the map/legend can tell "no crossings recorded" apart from "point does not exist"."""

    full_index = pd.MultiIndex.from_product([point_ids, CATEGORIES], names = ["border_crossing_point_id", "category"])

    filled = df_counts.set_index(["border_crossing_point_id", "category"]).reindex(full_index).fillna(0)

    return filled.reset_index()


def build_map(df_points, df_counts, output_path):
    """
    One folium map, one CircleMarker layer per category (see CATEGORIES) -
    each togglable independently through folium's layer control (rendered
    as checkboxes, since these are overlay FeatureGroups, not base layers),
    with only DEFAULT_VISIBLE_CATEGORY checked on load.
    """

    df_points = df_points.to_crs("EPSG:4326")

    wide_counts = df_counts.pivot(index = "border_crossing_point_id", columns = "category", values = "count").reset_index()
    df_points   = df_points.merge(wide_counts, on = "border_crossing_point_id", how = "left")
    for category in CATEGORIES:
        df_points[category] = df_points[category].fillna(0.0)

    center = [df_points.geometry.y.mean(), df_points.geometry.x.mean()]
    m = folium.Map(location = center, zoom_start = 8, tiles = None)
    folium.TileLayer(
        tiles = _TILE_URL, attr = "© swisstopo", name = "swisstopo (grayscale)", opacity = 0.6, control = False,
    ).add_to(m)

    for category in CATEGORIES:
        global_max = max(df_points[category].max(), 1)
        group = folium.FeatureGroup(name = CATEGORY_LABELS[category], show = category == DEFAULT_VISIBLE_CATEGORY)

        for _, row in df_points[df_points[category] > 0].iterrows():
            count  = row[category]
            radius = 5 + 20 * (count ** 0.5) / (global_max ** 0.5)

            popup = (
                f"<b>{row['interview_place']}</b> ({row['border_crossing_point_id']}, {row['label']})<br>"
                + "<br>".join(f"{CATEGORY_LABELS[c]}: {row[c]:.0f}" for c in CATEGORIES)
            )

            folium.CircleMarker(
                location = (row.geometry.y, row.geometry.x),
                radius = radius,
                color = CATEGORY_COLORS[category], weight = 1.2, fill = True,
                fill_color = CATEGORY_COLORS[category], fill_opacity = 0.75,
                popup = folium.Popup(popup, max_width = 320),
                tooltip = f"{row['interview_place']}: {count:.0f} {CATEGORY_LABELS[category].lower()}",
            ).add_to(group)

        group.add_to(m)

    folium.LayerControl(collapsed = False).add_to(m)

    m.save(output_path)
