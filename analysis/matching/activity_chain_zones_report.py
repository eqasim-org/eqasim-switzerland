"""Render the saved paired V1/V2/V3 spatial-preference experiment."""

from base64 import b64encode
from html import escape
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from data.microcensus.activity_chain_features import FEATURES

NAMES = {
    "v1_density": "V1 · density",
    "v1_zones": "V1 · zones",
    "v2_density": "V2 · density",
    "v2_zones": "V2 · density + zones",
    "v3_density": "V3 · density",
    "v3_zones": "V3 · zones",
}
TICK_LABELS = ["V1\nD", "V1\nZ", "V2\nD", "V2\nD+Z", "V3\nD", "V3\nZ"]
COLORS = {
    "v1_density": "#7da1bf",
    "v1_zones": "#204c72",
    "v2_density": "#a3c7a0",
    "v2_zones": "#32784a",
    "v3_density": "#e6af7b",
    "v3_zones": "#bb5728",
}
LABELS = {
    "work_trips": "Work / business",
    "shopping_trips": "Shopping",
    "leisure_trips": "Leisure",
    "total_trips": "All trips",
}


def render(directory):
    output = Path(directory)
    metrics = pd.read_csv(output / "regional_trip_counts.csv")
    changes = pd.read_csv(output / "improvements.csv")
    protocol = json.loads((output / "protocol.json").read_text())
    identity = json.loads((output / "manifest.json").read_text())
    zoning = identity["zones"]["source_metadata"]
    kernel = zoning["source_surface_metadata"]["selected_neighbours"]
    zoning_description = (
        f"Whole municipalities receive the group containing the largest resident count in the continuous activity contours. "
        f"The adaptive kernel uses {kernel} survey neighbours, with {zoning['smoothing_m'] / 1000:g} km additional Gaussian smoothing "
        f"and a {zoning['high_population_target']:.0%} upstream high-activity population target. "
        "Municipal voting can slightly change the final group share."
    )
    coverage = pd.read_csv(output / "coverage.csv")
    exclusions = pd.read_csv(output / "exclusions.csv")
    images = []
    national = metrics.loc[metrics.level.eq("national")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, feature in zip(axes, FEATURES):
        sub = (
            national.loc[national.feature.eq(feature)]
            .set_index("variant")
            .reindex(NAMES)
        )
        ax.bar(range(len(NAMES)), sub.matched_mean, color=[COLORS[k] for k in NAMES])
        ax.axhline(
            sub.survey_mean.iloc[0],
            color="black",
            ls="--",
            label="Weighted MZ reference",
        )
        ax.set_xticks(
            range(len(NAMES)),
            TICK_LABELS,
            fontsize=9,
        )
        ax.set_title(LABELS[feature])
        ax.set_ylabel("Trips per person/day")
        ax.set_ylim(0, max(sub.matched_mean.max(), sub.survey_mean.iloc[0]) * 1.2)
        for i, value in enumerate(sub.matched_mean):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "National matched diary rates · same non-forced-home cohort\nD = density; Z = zones; V2 D+Z retains density"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, output, "national_trip_rates", images)

    cities = ["Lausanne", "Geneva", "Zurich"]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i, city in enumerate(cities):
        for j, feature in enumerate(FEATURES):
            ax = axes[i, j]
            sub = metrics.loc[
                metrics.level.eq("municipality")
                & metrics.region.eq(city)
                & metrics.feature.eq(feature)
            ]
            sub = sub.set_index("variant").reindex(NAMES)
            ax.bar(
                range(len(NAMES)), sub.matched_mean, color=[COLORS[k] for k in NAMES]
            )
            ax.axhline(sub.survey_mean.iloc[0], color="black", ls="--")
            ax.set_xticks(
                range(len(NAMES)),
                TICK_LABELS,
                fontsize=8,
            )
            ax.set_title(f"{city} · {LABELS[feature]}", fontsize=11)
            ax.set_ylim(0, max(sub.matched_mean.max(), sub.survey_mean.iloc[0]) * 1.23)
            ax.grid(axis="y", alpha=0.15)
            for k, value in enumerate(sub.matched_mean):
                ax.text(k, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
            if j == 0:
                ax.set_ylabel("Trips per person/day")
    fig.suptitle(
        "Municipalities: assigned diary rates; dashed line = weighted MZ reference\nD = density; Z = zones; V2 D+Z retains density"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output, "municipality_trip_rates", images)

    totals = metrics.loc[metrics.level.eq("canton") & metrics.feature.eq("total_trips")]
    regions = (
        totals[["region_id", "region"]]
        .drop_duplicates()
        .sort_values("region_id")
        .region.tolist()
    )
    fig, axes = plt.subplots(1, 3, figsize=(19, 11), sharey=True)
    y = np.arange(len(regions))
    for ax, algorithm in zip(axes, ["v1", "v2", "v3"]):
        for shift, suffix in [(-0.18, "density"), (0.18, "zones")]:
            name = algorithm + "_" + suffix
            values = (
                totals.loc[totals.variant.eq(name)].set_index("region").reindex(regions)
            )
            ax.barh(
                y + shift,
                values.relative_error_percent,
                height=0.34,
                color=COLORS[name],
                label=NAMES[name],
            )
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(y, [r.replace("_", " ").title() for r in regions], fontsize=9)
        ax.set_xlabel("Total-trip count error relative to MZ-based expectation (%)")
        ax.set_title(algorithm.upper())
        ax.legend(frameon=False)
        ax.grid(axis="x", alpha=0.15)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Canton total trips · equal regional population denominator in all variants"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, output, "canton_total_trip_errors", images)

    canton_changes = changes.loc[changes.level.eq("canton")]
    matrix = []
    for region in regions:
        row = []
        for algorithm in ["V1", "V2", "V3"]:
            for feature in FEATURES:
                sub = canton_changes.loc[
                    canton_changes.region.eq(region)
                    & canton_changes.algorithm.eq(algorithm)
                    & canton_changes.feature.eq(feature)
                ].iloc[0]
                row.append(
                    abs(sub.density_relative_error_percent)
                    - abs(sub.zones_relative_error_percent)
                )
        matrix.append(row)
    matrix = np.asarray(matrix)
    limit = max(float(np.nanmax(np.abs(matrix))), 1.0)
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    ax.set_xticks(
        range(12),
        [alg + "\n" + LABELS[f] for alg in ["V1", "V2", "V3"] for f in FEATURES],
        fontsize=9,
    )
    ax.set_yticks(
        range(len(regions)), [r.replace("_", " ").title() for r in regions], fontsize=9
    )
    for i in range(len(regions)):
        for j in range(12):
            ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.axvline(3.5, color="black", lw=1)
    ax.axvline(7.5, color="black", lw=1)
    ax.set_title("Do zones improve canton trip counts? Green = smaller absolute error")
    fig.colorbar(
        im,
        ax=ax,
        label="Reduction in absolute relative error (percentage points)",
        shrink=0.65,
    )
    fig.tight_layout()
    save(fig, output, "canton_improvements", images)

    summary = []
    for algorithm in ["V1", "V2", "V3"]:
        for level in ["canton", "municipality"]:
            for feature in FEATURES:
                sub = changes.loc[
                    changes.algorithm.eq(algorithm)
                    & changes.level.eq(level)
                    & changes.feature.eq(feature)
                ]
                summary.append(
                    {
                        "algorithm": algorithm,
                        "level": level,
                        "feature": feature,
                        "regions": len(sub),
                        "regions_improved": int(sub.zones_better.sum()),
                        "density_mean_absolute_rate_error": sub.density_absolute_rate_error.mean(),
                        "zones_mean_absolute_rate_error": sub.zones_absolute_rate_error.mean(),
                        "population_weighted_density_absolute_rate_error": np.average(
                            sub.density_absolute_rate_error,
                            weights=sub.population_persons,
                        ),
                        "population_weighted_zones_absolute_rate_error": np.average(
                            sub.zones_absolute_rate_error,
                            weights=sub.population_persons,
                        ),
                    }
                )
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "error_summary.csv", index=False)
    city_counts = metrics.loc[metrics.level.eq("municipality")].copy()
    city_counts.to_csv(output / "municipality_trip_counts.csv", index=False)
    city_table = city_counts[
        [
            "region",
            "feature",
            "variant",
            "population_persons",
            "survey_persons",
            "survey_effective_sample",
            "expected_trips_full_scale",
            "matched_trips_full_scale",
            "relative_error_percent",
        ]
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    summary_total = summary.loc[
        summary.level.eq("canton") & summary.feature.eq("total_trips")
    ]
    x = np.arange(len(summary_total))
    ax.bar(
        x - 0.18,
        summary_total.density_mean_absolute_rate_error,
        0.35,
        color="#7da1bf",
        label="Density",
    )
    ax.bar(
        x + 0.18,
        summary_total.zones_mean_absolute_rate_error,
        0.35,
        color="#bb5728",
        label="Zones",
    )
    ax.set_xticks(x, summary_total.algorithm)
    ax.set_ylabel("Mean absolute trips/person/day error across 26 cantons")
    ax.set_title("Overall canton total-trip fit")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, output, "canton_error_summary", images)

    # Balance purposes and geographic levels so populous places cannot dominate.
    balanced = balanced_errors(metrics)
    balanced.to_csv(output / "balanced_scores.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(balanced))
    ax.bar(
        x - 0.18, balanced.density_error, 0.36, color="#7da1bf", label="Density only"
    )
    ax.bar(x + 0.18, balanced.zones_error, 0.36, color="#bb5728", label="Zones enabled")
    ax.set_xticks(x, balanced.algorithm)
    ax.set_ylabel("Balanced normalized trip-rate error (lower is better)")
    ax.set_title("Four purposes; equal weights for canton and city levels")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, output, "balanced_error_summary", images)

    # Make the three-city total-trip tradeoffs readable without the purpose panels.
    city_totals = metrics.loc[
        metrics.level.eq("municipality") & metrics.feature.eq("total_trips")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, algorithm in zip(axes, ["v1", "v2", "v3"]):
        for shift, suffix in [(-0.18, "density"), (0.18, "zones")]:
            name = algorithm + "_" + suffix
            sub = (
                city_totals.loc[city_totals.variant.eq(name)]
                .set_index("region")
                .reindex(cities)
            )
            bars = ax.bar(
                np.arange(3) + shift,
                sub.relative_error_percent,
                0.36,
                color=COLORS[name],
                label=NAMES[name],
            )
            ax.bar_label(bars, fmt="%+.2f%%", fontsize=8, padding=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(3), cities)
        ax.set_title(algorithm.upper())
        ax.set_ylabel("Total-trip error relative to MZ (%)")
        ax.legend(frameon=False, fontsize=8)
        ax.margins(y=0.2)
    fig.tight_layout()
    save(fig, output, "city_total_trip_errors", images)

    headlines = []
    for algorithm in ["V1", "V2", "V3"]:
        row = summary.loc[
            summary.algorithm.eq(algorithm)
            & summary.level.eq("canton")
            & summary.feature.eq("total_trips")
        ].iloc[0]
        reduction = 100 * (
            1
            - row.zones_mean_absolute_rate_error / row.density_mean_absolute_rate_error
        )
        weighted = 100 * (
            1
            - row.population_weighted_zones_absolute_rate_error
            / row.population_weighted_density_absolute_rate_error
        )
        national = metrics.loc[
            metrics.level.eq("national") & metrics.feature.eq("total_trips")
        ].set_index("variant")
        density_error = national.loc[
            algorithm.lower() + "_density", "relative_error_percent"
        ]
        zone_error = national.loc[
            algorithm.lower() + "_zones", "relative_error_percent"
        ]
        headlines.append(
            f"<li>{algorithm}: total-trip errors improve in {int(row.regions_improved)}/{int(row.regions)} cantons; "
            f"equal-canton mean absolute rate error reduction is {reduction:+.1f}% and population-weighted reduction is {weighted:+.1f}%. "
            f"National total-trip error changes from {density_error:+.2f}% to {zone_error:+.2f}%.</li>"
        )
    parts = [
        '<!doctype html><html><head><meta charset="utf-8"><title>Zone versus density matching</title>',
        "<style>body{font:15px system-ui;max-width:1300px;margin:35px auto;padding:0 20px}p{line-height:1.5}img{max-width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}table{border-collapse:collapse;font-size:13px}.table{overflow-x:auto}</style></head><body>",
        "<h1>Whole-municipality activity zones: paired V1/V2/V3 matching test</h1>",
        f"<p>Common analysis cohort: <b>{protocol['common_recipients']:,}</b> synthetic residents. "
        f"Weighted survey reference: {protocol['survey_reference_persons']:,} respondents. "
        f"Day: {escape(identity['day'])}; paired seed: {identity['seed']}; synthetic sample: {100 * identity['sample']:.0f}%.</p>",
        f"<p>{zoning_description} The four trip features are work/business, shopping, leisure and total trips.</p>",
        "<h2>What improves and what does not</h2><ul>" + "".join(headlines) + "</ul>",
        "<p>Use the per-purpose figures and tables below to assess tradeoffs: regional total-trip gains need not improve national totals, "
        "work trips or shopping trips. These are comparisons of assigned diary counts before MATSim replanning.</p>",
        "<p><b>Forced home-only people are excluded.</b> This includes explicit bypasses, collective residents "
        "restricted to home-only donors and children below the survey age. Natural zero-trip days remain included. "
        "All six variants are assessed on the same residents with valid assigned diaries.</p>",
        "<p>V1/V3 replace their density preference at the same position with the same relaxation rules. V2 adds the zone as a categorical input and retains its existing density inputs; both V2 models are fitted separately. "
        "Opening-hours masks, all other features, donor weights and seeds remain unchanged. "
        "Regional expected counts = weighted MZ trips/person × regional common-cohort population. "
        "Full-scale counts additionally divide by the sampling fraction; these are not official observed canton totals.</p>",
        "<p><b>This is in-sample descriptive validation.</b> The zones were learned from the same MZ 2015 diaries. "
        "It does not establish out-of-sample accuracy or improved MATSim outcomes. Small regional samples and one matching seed "
        "limit precision; survey effective sample sizes and low-support flags are included in the CSVs.</p>",
        "<h2>Balanced regional fit</h2>",
        "<p>Absolute trip-rate errors are divided by national MZ means for each purpose, then averaged equally over the four purposes and canton/city levels. Each region is equally weighted within its level. Positive reduction means smaller error.</p>",
        balanced.to_html(
            index=False, border=0, float_format=lambda value: f"{value:.4f}"
        ),
        "<h2>Exclusion audit (flag counts may overlap)</h2>",
        exclusions.to_html(index=False, border=0),
        "<h2>Coverage and runtime</h2>",
        coverage.to_html(index=False, border=0, float_format=lambda x: f"{x:.1f}"),
        "<h2>Regional error summary</h2>",
        summary.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}"),
    ]
    for image in images:
        encoded = b64encode((output / (image + ".png")).read_bytes()).decode("ascii")
        parts.append(
            f'<h2>{escape(image.replace("_", " ").title())}</h2><img src="data:image/png;base64,{encoded}">'
        )
    parts.extend(
        [
            '<h2>Municipality counts (full-scale equivalents)</h2><div class="table">',
            city_table.to_html(
                index=False, border=0, float_format=lambda x: f"{x:,.2f}"
            ),
            "</div></body></html>",
        ]
    )
    path = output / "report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def balanced_errors(metrics):
    """Compare equal-purpose, equal-geographic-level errors within each matcher."""
    national = (
        metrics.loc[metrics.level.eq("national")].groupby("feature").survey_mean.first()
    )
    rows = metrics.loc[metrics.level.isin(["canton", "municipality"])].copy()
    rows["normalized_error"] = rows.absolute_rate_error / rows.feature.map(national)
    score = (
        rows.groupby(["variant", "level", "feature"])
        .normalized_error.mean()
        .groupby("variant")
        .mean()
    )
    output = []
    for algorithm in ["v1", "v2", "v3"]:
        density, zones = score[algorithm + "_density"], score[algorithm + "_zones"]
        output.append(
            {
                "algorithm": algorithm.upper(),
                "density_error": density,
                "zones_error": zones,
                "error_reduction_percent": 100 * (1 - zones / density),
            }
        )
    return pd.DataFrame(output)


def save(fig, output, name, images):
    fig.savefig(output / (name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(output / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)
    images.append(name)
