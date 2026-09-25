"""
Stage: analysis.statent.international_organizations_impact

Standalone comparison of data.statent.statent run twice - once with
data.statent.international_organizations folded in (see that stage's
docstring for why STATENT alone misses these), once without - to show how
many jobs the international organizations add, and where.

This stage is intentionally independent of the rest of the pipeline: it only
depends on two differently-configured instances of data.statent.statent
(requested via synpp's per-call config override, so both live side by side
in the cache under their own config hash) plus data.spatial.municipalities
for names/geometries. It is not wired into any config's `run:` list; invoke
it directly, e.g.:

    python -m synpp config.yml --run analysis.statent.international_organizations_impact

Produces, in the stage's cache folder (context.path()):
  - international_organizations_jobs_impact_map.html   choropleth (sequential
                                                          blue ramp) of jobs
                                                          added per
                                                          municipality, with a
                                                          popup per
                                                          municipality showing
                                                          the before/after
                                                          employee counts.
                                                          Only municipalities
                                                          that actually gain
                                                          jobs are drawn -
                                                          omitting the ~2,100
                                                          untouched ones keeps
                                                          the file a
                                                          reasonable size.

Returns one row per municipality: municipality_id, municipality_name,
jobs_without_international_organizations, jobs_with_international_organizations,
jobs_added, relative_change.
"""

import logging
import os

import folium
import numpy as np
import pandas as pd

logger = logging.getLogger("synpp")

# Free, no-key WMTS basemap - see analysis/pt/interactive_map.py's module
# docstring for why plain OSM/cartodbpositron tiles are not used here.
_TILE_URL = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"

# Sequential blue ramp (near-zero -> high), from the dataviz skill's default
# palette (references/palette.md, steps 150-700). Untouched municipalities
# aren't drawn at all (see build_map), so this ramp only needs to span the
# affected ones.
SEQUENTIAL_BLUE_RAMP = ["#b7d3f6", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]


def configure(context):
    # Two instances of the same stage, distinguished only by the config
    # override - synpp caches/keys them independently. Aliases let execute()
    # fetch each without repeating the config dict (context.stage() requires
    # an exact match against what configure() registered).
    context.stage(
        "data.statent.statent",
        config = {"include_international_organizations": False},
        alias = "statent_without_international_organizations",
    )
    context.stage(
        "data.statent.statent",
        config = {"include_international_organizations": True},
        alias = "statent_with_international_organizations",
    )
    context.stage("data.spatial.municipalities")


def execute(context):
    df_without = context.stage("statent_without_international_organizations")
    df_with = context.stage("statent_with_international_organizations")
    df_municipalities = context.stage("data.spatial.municipalities")[0]

    df_jobs = pd.merge(
        aggregate_jobs(df_without, "jobs_without_international_organizations"),
        aggregate_jobs(df_with, "jobs_with_international_organizations"),
        on = "municipality_id", how = "outer",
    ).fillna(0.0)

    df_jobs["jobs_added"] = (
        df_jobs["jobs_with_international_organizations"] - df_jobs["jobs_without_international_organizations"]
    )
    df_jobs["relative_change"] = df_jobs["jobs_added"] / df_jobs["jobs_without_international_organizations"].replace(0.0, np.nan)

    total_without = df_jobs["jobs_without_international_organizations"].sum()
    total_with = df_jobs["jobs_with_international_organizations"].sum()
    n_affected = int((df_jobs["jobs_added"] > 0).sum())

    logger.info(
        "Total jobs without international organizations: %.0f; with: %.0f (+%.0f, +%.1f%%) "
        "across %d municipalities.",
        total_without, total_with, total_with - total_without,
        100 * (total_with - total_without) / total_without if total_without > 0 else float("nan"),
        n_affected,
    )

    df_jobs = df_jobs.merge(
        df_municipalities[["municipality_id", "municipality_name"]],
        on = "municipality_id", how = "left",
    )

    build_map(
        df_jobs, df_municipalities,
        os.path.join(context.path(), "international_organizations_jobs_impact_map.html"),
    )

    columns = [
        "municipality_id", "municipality_name",
        "jobs_without_international_organizations", "jobs_with_international_organizations",
        "jobs_added", "relative_change",
    ]
    return df_jobs[columns].sort_values("jobs_added", ascending = False).reset_index(drop = True)


def aggregate_jobs(df, column_name):
    """Total number_employees per municipality. min_count=1 so an all-NaN group (e.g. a
    not_found-staff international organization with no other establishment in that
    municipality) sums to NaN, not a misleading 0.0, before the outer-merge fillna."""

    counts = df.groupby("municipality_id")["number_employees"].sum(min_count = 1)
    return counts.reset_index(name = column_name)


def build_map(df_jobs, df_municipalities, output_path):
    """
    One choropleth layer, one polygon per municipality that actually gains
    jobs from international organizations - filled using a sequential blue
    ramp (quantile-bucketed, since the distribution is extremely skewed: a
    handful of municipalities gain hundreds of jobs, everyone else gains
    none). Untouched municipalities are omitted entirely rather than drawn
    with a "no change" fill - swissBOUNDARIES3D's full-resolution polygons
    for all ~2,100 Swiss municipalities inflated the map to 70+ MB for the
    ~16 that actually matter here, which made it too slow to open.
    """

    df_map = df_municipalities.merge(
        df_jobs.drop(columns = ["municipality_name"], errors = "ignore"),
        on = "municipality_id", how = "inner",
    )
    df_map = df_map[df_map["jobs_added"] > 0].to_crs("EPSG:4326")

    bucket_edges = quantile_edges(df_map["jobs_added"], len(SEQUENTIAL_BLUE_RAMP))

    m = folium.Map(location = [0, 0], tiles = None)
    folium.TileLayer(
        tiles = _TILE_URL, attr = "© swisstopo", name = "swisstopo (grayscale)", opacity = 0.6, control = False,
    ).add_to(m)

    layer = folium.FeatureGroup(name = "Jobs added by international organizations", show = True)

    for _, row in df_map.iterrows():
        jobs_added = row["jobs_added"]
        fill_color = SEQUENTIAL_BLUE_RAMP[bucket_index(jobs_added, bucket_edges)]

        relative_change = row["relative_change"]
        relative_change_text = f"{relative_change:+.1%}" if pd.notna(relative_change) else "n/a"

        popup = (
            f"<b>{row['municipality_name']}</b><br>"
            f"Jobs without int'l organizations: {row['jobs_without_international_organizations']:,.0f}<br>"
            f"Jobs with int'l organizations: {row['jobs_with_international_organizations']:,.0f}<br>"
            f"Added: {jobs_added:+,.0f} ({relative_change_text})"
        )

        folium.GeoJson(
            row["geometry"],
            style_function = lambda feature, fill_color = fill_color: {
                "fillColor": fill_color, "color": "#666666", "weight": 0.5,
                "fillOpacity": 0.75,
            },
            popup = folium.Popup(popup, max_width = 320),
            tooltip = f"{row['municipality_name']}: {jobs_added:+,.0f} jobs",
        ).add_to(layer)

    layer.add_to(m)
    folium.LayerControl(collapsed = False).add_to(m)

    minx, miny, maxx, maxy = df_map.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    m.save(output_path)


def quantile_edges(values, bucket_count):
    """Interior quantile edges splitting `values` into `bucket_count` buckets, deduplicated
    (falls back to fewer, wider buckets when the data doesn't have enough distinct values
    to fill every quantile)."""

    quantiles = np.linspace(0, 1, bucket_count + 1)[1:-1]
    edges = sorted(set(np.quantile(values, quantiles)))
    return edges


def bucket_index(value, edges):
    index = 0
    for edge in edges:
        if value > edge:
            index += 1
    return min(index, len(SEQUENTIAL_BLUE_RAMP) - 1)
