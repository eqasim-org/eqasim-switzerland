"""Report the default whole-municipality activity zones and their evidence."""

from base64 import b64encode
from html import escape
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from data.microcensus.activity_chain_features import CLUSTER_COLUMN, FEATURES
from data.microcensus.activity_zones.municipalities import CITY_IDS

COLORS = ["#b73537", "#eb9955", "#79b9b5", "#456b9c"]
LABELS = {
    "work_trips": "Work / business",
    "shopping_trips": "Shopping",
    "leisure_trips": "Leisure",
    "total_trips": "All trips",
}


def configure(context):
    context.stage("data.microcensus.activity_chains_clusters")
    context.config("analysis_path")


def execute(context):

    # Load the default zoning, continuous fields and municipality diagnostics.

    result = context.stage("data.microcensus.activity_chains_clusters")

    # Export the map, evidence and profiles without modifying cached inputs.

    output = Path(context.config("analysis_path")) / "activity_chains_clusters"
    return str(write_report(result, output))


def map_regions(ax, regions, references=None, outline=None, bounds=None):
    regions.plot(
        column=CLUSTER_COLUMN,
        cmap=ListedColormap(COLORS),
        vmin=0.5,
        vmax=4.5,
        ax=ax,
        linewidth=0.5,
        edgecolor="white",
    )
    if outline is not None:
        outline.boundary.plot(ax=ax, color="black", linewidth=1.1, linestyle="--")
    if references is not None:
        for row in references.itertuples():
            point = row.geometry.representative_point()
            ax.scatter(
                point.x,
                point.y,
                color="black",
                edgecolor="white",
                linewidth=0.5,
                s=15,
                zorder=5,
            )
            ax.annotate(
                row.municipality_name,
                (point.x, point.y),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "none",
                    "pad": 1,
                },
                zorder=6,
            )
    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    ax.set_axis_off()


def legend(fig):
    fig.legend(
        handles=[
            Patch(facecolor=c, label=f"Group {i}")
            for i, c in enumerate(COLORS, start=1)
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )


def write_report(result, output):
    source = result["contours"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    units = result["municipalities"]
    profiles = result["profiles"].sort_values(CLUSTER_COLUMN)
    meta = result["metadata"]
    kernel = (
        meta["source_metadata"]
        .get("source_surface_metadata", {})
        .get("selected_neighbours")
    )
    images = []
    for key in ["municipalities", "regions"]:
        result[key].to_parquet(output / (key + ".parquet"), index=False)
    result["regions"].to_file(output / "regions.geojson", driver="GeoJSON")
    for key in [
        "profiles",
        "profiles_comparison",
        "city_comparison",
        "population_flows",
    ]:
        result[key].to_csv(output / (key + ".csv"), index=False)
    units.drop(columns="geometry").to_csv(
        output / "municipality_assignments.csv", index=False
    )
    (output / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cities = units.loc[units.municipality_id.isin(CITY_IDS)]

    fig, ax = plt.subplots(figsize=(12, 8))
    map_regions(ax, result["regions"], references=cities)
    ax.set_title("Four activity groups with whole municipalities")
    legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, output, "regions_municipalities", images)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, regions, title in zip(
        axes,
        [source["regions"], result["regions"]],
        [
            "Original continuous contours",
            "Whole municipalities; largest resident share",
        ],
    ):
        map_regions(
            ax,
            regions,
            references=cities.loc[cities.municipality_id.isin([351, 6621, 5586, 261])],
        )
        ax.set_title(title)
    legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, output, "regions_comparison", images)

    # Include Bern and the most affected municipalities, rather than claiming Bern changed.
    focus_ids = [351] if units.municipality_id.eq(351).any() else []
    focus_ids += [
        int(mid)
        for mid in units.sort_values(
            "changed_population", ascending=False
        ).municipality_id
        if mid not in focus_ids
    ][:3]
    fig, axes = plt.subplots(
        2, len(focus_ids), figsize=(4.5 * len(focus_ids), 9), squeeze=False
    )
    for j, mid in enumerate(focus_ids):
        focus = units.loc[units.municipality_id.eq(mid)]
        row = focus.iloc[0]
        xmin, ymin, xmax, ymax = focus.total_bounds
        pad = max(2000.0, 0.2 * max(xmax - xmin, ymax - ymin))
        bounds = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
        for i, regions in enumerate([source["regions"], result["regions"]]):
            map_regions(
                axes[i, j], regions, references=focus, outline=focus, bounds=bounds
            )
        axes[0, j].set_title(
            f"{row.municipality_name}: continuous\n{row.original_resident_groups} resident group(s)",
            fontsize=10,
        )
        axes[1, j].set_title(
            f"Whole municipality → group {row[CLUSTER_COLUMN]}\n"
            f"{row.winning_share:.1%} original resident agreement",
            fontsize=10,
        )
    legend(fig)
    fig.suptitle(
        "Bern and municipalities most affected by the boundary change; dashed line = municipality"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, output, "city_boundaries_comparison", images)

    fig, ax = plt.subplots(figsize=(12, 7))
    units.plot(
        column="winning_share",
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        legend=True,
        linewidth=0,
        missing_kwds={"color": "#cccccc"},
        rasterized=True,
        legend_kwds={"label": "Resident share in the chosen original contour group"},
    )
    ax.set_axis_off()
    ax.set_title(
        "Agreement with the municipality assignment; grey = no eligible residents"
    )
    save(fig, output, "resident_agreement", images)

    old = source["profiles"].sort_values(CLUSTER_COLUMN)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    positions = np.arange(1, 5)
    for row, prefix in enumerate(["", "survey_"]):
        for ax, feature in zip(axes[row], FEATURES):
            values = profiles[prefix + feature]
            previous = old[prefix + feature]
            ax.bar(
                positions - 0.18,
                previous,
                0.36,
                color=COLORS,
                alpha=0.4,
                label="Continuous",
            )
            ax.bar(
                positions + 0.18, values, 0.36, color=COLORS, label="Whole municipality"
            )
            ax.set_xticks(positions)
            ax.set_xlabel("Group")
            ax.set_ylim(0, max(values.max(), previous.max()) * 1.2)
            ax.set_title(LABELS[feature])
            for i, (before, after) in enumerate(zip(previous, values), start=1):
                ax.text(
                    i - 0.18,
                    before,
                    f"{before:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
                ax.text(
                    i + 0.18,
                    after,
                    f"{after:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    axes[0, 0].set_ylabel("Modelled trips / resident / day")
    axes[1, 0].set_ylabel("Observed weighted MZ trips / day")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Profiles before and after keeping municipalities whole")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output, "region_profiles", images)

    flows = result["population_flows"].pivot(
        index="original_group", columns="municipal_group", values="population"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    heatmap = ax.imshow(flows.to_numpy() / 1e6, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{int(flows.iloc[i, j]):,}",
                ha="center",
                va="center",
                color="white"
                if flows.iloc[i, j] > flows.to_numpy().max() * 0.55
                else "black",
            )
    ax.set_xticks(range(4), range(1, 5))
    ax.set_yticks(range(4), range(1, 5))
    ax.set_xlabel("Whole-municipality group")
    ax.set_ylabel("Original contour group")
    ax.set_title(
        f"{meta['moved_population']:,} reassigned residents ({meta['moved_population_share']:.2%})"
    )
    fig.colorbar(heatmap, ax=ax, label="Residents (millions)")
    fig.tight_layout()
    save(fig, output, "population_flows", images)

    # Show the four modelled rate fields and their adaptive survey support.
    cells = result["cells"]
    cells.to_parquet(output / "cells.parquet", index=False)
    np.savez_compressed(
        output / "smooth_fields.npz",
        x=source["x"],
        y=source["y"],
        fields=source["fields"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, feature in zip(axes.flat, FEATURES):
        cells.plot(
            column=feature,
            ax=ax,
            cmap="viridis",
            legend=True,
            linewidth=0,
            rasterized=True,
        )
        ax.set_axis_off()
        ax.set_title(LABELS[feature] + " trips / resident / day")
    fig.suptitle("Adaptive survey-weighted activity fields before municipal assignment")
    fig.tight_layout()
    save(fig, output, "trip_rate_surfaces", images)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, column, title in zip(
        axes,
        ["bandwidth_m", "effective_sample"],
        ["Adaptive kernel radius (m)", "Effective local survey sample"],
    ):
        cells.plot(
            column=column,
            ax=ax,
            cmap="viridis",
            legend=True,
            linewidth=0,
            rasterized=True,
        )
        ax.set_axis_off()
        ax.set_title(title)
    fig.tight_layout()
    save(fig, output, "survey_support", images)

    city_columns = [
        "municipality_name",
        CLUSTER_COLUMN,
        "population",
        "original_resident_groups",
        "original_area_groups",
        "winning_share",
        "changed_population",
        "final_resident_groups",
    ]
    top = profiles.iloc[0]
    bern = result["city_comparison"].loc[
        result["city_comparison"].municipality_id.eq(351)
    ]
    bern_note = ""
    if not bern.empty:
        row = bern.iloc[0]
        bern_note = (
            f"Bern originally spanned {row.original_resident_groups} resident group(s) and "
            f"{row.original_area_groups} area group(s); its whole-municipality assignment is group {row[CLUSTER_COLUMN]}. "
        )
    text = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Whole-municipality activity groups</title>",
        "<style>body{font:16px system-ui;max-width:1500px;margin:35px auto;padding:0 20px}p{line-height:1.5}"
        "img{max-width:100%}td,th{padding:7px;border-bottom:1px solid #ddd}</style></head><body>",
        "<h1>Keep municipalities whole using the resident distribution</h1>",
        f"<p><b>{meta['previously_split_resident_units']:,} municipalities had residents in multiple contour groups. "
        f"The municipal assignment reclassifies {meta['moved_population']:,} residents ({meta['moved_population_share']:.2%}) "
        f"and {meta['moved_diaries']:,} valid survey diaries.</b> All {meta['population']:,} eligible residents and "
        f"{meta['fitting_diaries']:,} fitting diaries are retained.</p>",
        f"<p>Group 1 contains {top.population_share:.2%} of residents after municipal assignment, with "
        f"{top.survey_total_trips:.2f} observed MZ and {top.total_trips:.2f} modelled trips/day. "
        f"The underlying fields still use {meta['source_metadata']['smoothing_m'] / 1000:g} km smoothing and "
        f"a {meta['source_metadata']['high_population_target']:.0%} continuous high-activity target. "
        f"The initial adaptive kernel uses {kernel} survey neighbours. "
        "Municipal assignment can move the population share away from that target.</p>",
        "<p>Count actual STATPOP permanent main-residence residents aged 6+ in each original contour group, using home coordinates. "
        "Assign the entire municipality to the group with the largest count; this is a plurality rule, not an area or centroid rule. "
        "Repeated home coordinates retain their full resident count. Exact vote ties use the smaller group number. "
        "Municipalities or administrative units without eligible residents use the largest original contour area. "
        "Assignments on shared municipal boundaries use the smallest municipality ID.</p>",
        "<p>Final geometries dissolve entire municipal polygons. Their boundaries follow administrative shapes and are not smoothed afterwards, "
        "because smoothing again could cut a municipality. Modelling and smoothing remain continuous upstream. "
        "This protects each municipality; it does not force a multi-municipality agglomeration to share one group.</p>",
        f"<p>{meta['municipal_units']:,} spatial units, including {meta['empty_units']} without eligible residents; "
        f"uncovered national area {meta['uncovered_area_km2']:.6f} km²; "
        f"residents outside municipalities {meta['population_outside_municipalities']:,}; "
        f"units where the winning resident share is below 50%: {meta['plurality_below_half_units']}.</p>",
        f"<p>{bern_note}The rule guarantees one group per municipality. The detailed examples also show "
        "municipalities with the largest numbers of reassigned residents.</p>",
        "<p>Observed profiles reuse the MZ fitting survey, including valid zero-trip days. These are descriptive comparisons, "
        "not independent validation. The matching stage uses these zones by default: V1/V3 replace density and V2 adds a categorical input while retaining density. Matching fit is evaluated separately.</p>",
        "<h2>City checks</h2>",
        result["city_comparison"][city_columns].to_html(
            index=False, border=0, float_format=lambda v: f"{v:.3f}"
        ),
        "<h2>Final profiles</h2>",
        profiles.to_html(index=False, border=0, float_format=lambda v: f"{v:.3f}"),
        "<h2>Most affected municipalities</h2>",
        units.nlargest(20, "changed_population")
        .drop(columns="geometry")
        .to_html(index=False, border=0, float_format=lambda v: f"{v:.3f}"),
    ]
    for name in images:
        encoded = b64encode((output / (name + ".png")).read_bytes()).decode("ascii")
        text.append(
            f'<h2>{escape(name.replace("_", " ").title())}</h2><img src="data:image/png;base64,{encoded}">'
        )
    text.append("</body></html>")
    path = output / "report.html"
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def save(fig, output, name, images):
    fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)
    images.append(name)
