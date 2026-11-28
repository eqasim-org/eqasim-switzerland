import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .utils import FRENCH_COHORTS, MODE_LABELS, figures_directory

logger = logging.getLogger("synpp")

MODE_ORDER     = ["car", "pt", "walk", "bike", "car_passenger"]
MODE_COLORS    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # Distinct colors for modes
DATASET_STYLES = {
    "Target": {"color": "#1D4E89", "linestyle": "-", "marker": "x", "linewidth": 1.3, "markersize": 2.5},
    "Simulated": {"color": "#D1495B", "linestyle": "--", "marker": "o", "linewidth": 1.1, "markersize": 2.5},
}

plt.rcParams.update(
    {
        "axes.edgecolor": "#4A4A4A",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "font.size": 10,
        "figure.dpi": 150,
        "legend.fontsize": 10,
        "axes.grid": False,
    }
)


def configure(context):
    context.stage("analysis.mode_shares.trip_distributions")
    context.stage("analysis.mode_shares.target")
    context.config("analysis.mode_shares.from_eqasim", False)
    if not isinstance(context.config("analysis.mode_shares.from_eqasim"), bool):
        raise ValueError("analysis.mode_shares.from_eqasim must be true or false")
    context.stage("analysis.mode_shares.eqasim" if context.config("analysis.mode_shares.from_eqasim")
                  else "analysis.mode_shares.simulated", alias="mode_share_results")
    context.stage("data.spatial.cantons")
    context.stage("data.spatial.municipalities")

    context.config("include_external_population", default=True)
    context.config("group_income_classes_in_analysis", default=False)
    context.config("analysis.counts.simulation_path", default="")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = 'simulation_output')


def execute(context):

    # Prepare the output folder and load both analyzers' filtered results.

    figures_dir = str(figures_directory(context))
    dataset_label = "eqasim" if context.config("analysis.mode_shares.from_eqasim") else "MATSim"
    os.makedirs(figures_dir, exist_ok = True)

    # load data
    target_mode_shares    = context.stage("analysis.mode_shares.target")
    simulated_mode_shares = context.stage("mode_share_results")

    # plot mode shares by distance
    fig, ax       = plt.subplots(figsize = (11, 5))
    distance_bins = np.array(target_mode_shares["distance_bins"])

    distances = (distance_bins[1:] + distance_bins[:-1]) / 2
    distances[-1] = distance_bins[-2]

    for i, mode in enumerate(MODE_ORDER):
        color = MODE_COLORS[i]

        actual    = target_mode_shares["distance"][mode]
        simulated = simulated_mode_shares["distance"][mode]

        ax.plot(distances, actual, color = color,
                linestyle  = DATASET_STYLES["Target"]["linestyle"],
                marker     = DATASET_STYLES["Target"]["marker"],
                linewidth  = DATASET_STYLES["Target"]["linewidth"],
                markersize = DATASET_STYLES["Target"]["markersize"],
                label      = f"{mode} (Microcensus)")

        ax.plot(distances, simulated, color = color,
                linestyle  = DATASET_STYLES["Simulated"]["linestyle"],
                marker     = DATASET_STYLES["Simulated"]["marker"],
                linewidth  = DATASET_STYLES["Simulated"]["linewidth"],
                markersize = DATASET_STYLES["Simulated"]["markersize"],
                label      = f"{mode} ({dataset_label})")

    ax.set_ylabel("Mode Share", fontsize = 13)
    ax.set_xlabel("Euclidean distance [m]", fontsize = 13)
    plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.2), ncol = 5, fontsize = 10)
    plt.tight_layout()
    plt.grid(axis = "y", linestyle = "--", alpha = 0.7)
    _ = plt.xticks(rotation = 0, fontsize = 12)
    _ = plt.yticks(rotation = 0, fontsize = 12)
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_distance.png"), bbox_inches = "tight")
    plt.close()

    # Plot mode shares by canton
    df_cantons = context.stage("data.spatial.cantons")[["canton_id", "canton_name_en"]]
    df_cantons = df_cantons.rename(columns = {"canton_name_en": "canton_name"})

    target_canton = (target_mode_shares["canton"].reset_index()
                     .merge(df_cantons).set_index("canton_name")
                     .drop(columns = ["canton_id"])
                     .sort_values("car"))

    simulated_canton = (simulated_mode_shares["canton"].reset_index()
                       .merge(df_cantons).set_index("canton_name")
                       .drop(columns = ["canton_id"])
                       .reindex(target_canton.index))

    fig, ax = plt.subplots(1, len(MODE_ORDER), figsize = (10, 7), sharey = True)

    for j, mode in enumerate(MODE_ORDER):
        ax[j].plot(
            simulated_canton[mode], simulated_canton.index,
            linestyle  = DATASET_STYLES["Simulated"]["linestyle"],
            marker     = DATASET_STYLES["Simulated"]["marker"],
            label      = "Simulated",
            color      = DATASET_STYLES["Simulated"]["color"],
            linewidth  = DATASET_STYLES["Simulated"]["linewidth"],
            markersize = DATASET_STYLES["Simulated"]["markersize"]
        )

        ax[j].plot(
            target_canton[mode], target_canton.index,
            linestyle  = DATASET_STYLES["Target"]["linestyle"],
            marker     = DATASET_STYLES["Target"]["marker"],
            label      = "Actual",
            color      = DATASET_STYLES["Target"]["color"],
            alpha      = 0.6,
            linewidth  = DATASET_STYLES["Target"]["linewidth"],
            markersize = DATASET_STYLES["Target"]["markersize"]
        )

        ax[j].set_xlabel("Mode Share", fontsize = 12, labelpad = 16)
        ax[j].set_title(mode.replace("_", " "), fontsize = 14)
        ax[j].grid(axis = "y", linestyle = "--", alpha = 0.5)

        max_shares = max(max(target_canton[mode]), max(simulated_canton[mode]), 0.2)
        min_shares = min(min(target_canton[mode]), min(simulated_canton[mode]))
        ax[j].set_xlim([0.2 * min_shares, 1.2 * max_shares])

    handles, _ = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, [dataset_label, "Microcensus"],
        loc = "upper center", bbox_to_anchor = (0.6, 1.04),
        ncol = 2, fontsize = 14, frameon = False
    )
    plt.tight_layout(rect = [0, 0, 1, 0.98])
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_canton.png"), bbox_inches = "tight")
    plt.close()

    # plot global mode shares
    fig, ax = plt.subplots(figsize = (8, 3))
    shares = pd.DataFrame({
        "Microcensus": target_mode_shares["global"].reindex(MODE_ORDER)["mode_share"].fillna(0),
        dataset_label: simulated_mode_shares["global"].reindex(MODE_ORDER)["mode_share"].fillna(0),
    })
    shares.plot.bar(ax = ax, width = 0.8, color = [DATASET_STYLES["Target"]["color"], DATASET_STYLES["Simulated"]["color"]])
    plt.xticks(rotation = 0)
    plt.ylabel("Mode Share", fontsize = 12)
    ax.grid(axis = "y", linestyle = "--", alpha = 0.7)

    # write the values on top of the bars
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.01, f"{i.get_height():.1%}", ha = "center", fontsize = 9)

    plt.ylim([0, 0.48])
    ax.legend(loc = "upper right", fontsize = 12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "global_mode_shares.png"), bbox_inches = "tight")
    plt.close()

    # by age
    fig, ax    = plt.subplots(figsize = (11, 5))
    age_bins   = np.array(target_mode_shares["age_bins"])
    age_labels = target_mode_shares["age_labels"]
    ages       = (age_bins[1:] + age_bins[:-1]) / 2
    ages[-1]   = age_bins[-2]

    for i, mode in enumerate(MODE_ORDER):
        color     = MODE_COLORS[i]
        actual    = target_mode_shares["age"][mode]
        simulated = simulated_mode_shares["age"][mode]

        ax.plot(ages, actual, color = color,
                linestyle  = DATASET_STYLES["Target"]["linestyle"],
                marker     = DATASET_STYLES["Target"]["marker"],
                linewidth  = DATASET_STYLES["Target"]["linewidth"],
                markersize = DATASET_STYLES["Target"]["markersize"],
                label      = f"{mode} (Microcensus)")

        ax.plot(ages, simulated, color = color,
                linestyle  = DATASET_STYLES["Simulated"]["linestyle"],
                marker     = DATASET_STYLES["Simulated"]["marker"],
                linewidth  = DATASET_STYLES["Simulated"]["linewidth"],
                markersize = DATASET_STYLES["Simulated"]["markersize"],
                label      = f"{mode} ({dataset_label})")

    ax.set_ylabel("Mode Share",  fontsize = 13)
    ax.set_xlabel("Age [years]", fontsize = 13)
    ax.set_xticks(ages)
    ax.set_xticklabels(age_labels, rotation = 0, fontsize = 12)
    _ = plt.yticks(rotation = 0, fontsize = 12)
    plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.2), ncol = 5, fontsize = 10)
    plt.tight_layout()
    plt.grid(axis = "y", linestyle = "--", alpha = 0.7)
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_age.png"), bbox_inches = "tight")
    plt.close()

    # by income class: original nine classes or optional low/medium/high groups
    fig, ax        = plt.subplots(figsize = (11, 5))
    income_classes = sorted(target_mode_shares["income"].index.unique())
    income_labels = (
        {1: "Low", 2: "Medium", 3: "High"}
        if context.config("group_income_classes_in_analysis") else {}
    )

    for i, mode in enumerate(MODE_ORDER):
        color     = MODE_COLORS[i]
        actual    = target_mode_shares["income"].loc[income_classes, mode]
        simulated = simulated_mode_shares["income"].reindex(income_classes)[mode]

        ax.plot(income_classes, actual, color = color,
                linestyle  = DATASET_STYLES["Target"]["linestyle"],
                marker     = DATASET_STYLES["Target"]["marker"],
                linewidth  = DATASET_STYLES["Target"]["linewidth"],
                markersize = DATASET_STYLES["Target"]["markersize"],
                label      = f"{mode} (Microcensus)")

        ax.plot(income_classes, simulated, color = color,
                linestyle  = DATASET_STYLES["Simulated"]["linestyle"],
                marker     = DATASET_STYLES["Simulated"]["marker"],
                linewidth  = DATASET_STYLES["Simulated"]["linewidth"],
                markersize = DATASET_STYLES["Simulated"]["markersize"],
                label      = f"{mode} ({dataset_label})")

    ax.set_ylabel("Mode Share",   fontsize = 13)
    ax.set_xlabel("Income Class", fontsize = 13)
    ax.set_xticks(income_classes)
    ax.set_xticklabels([income_labels.get(c, c) for c in income_classes])
    plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.2), ncol = 5, fontsize = 10)
    plt.tight_layout()
    plt.grid(axis = "y", linestyle = "--", alpha = 0.7)
    _ = plt.xticks(rotation = 0, fontsize = 12)
    _ = plt.yticks(rotation = 0, fontsize = 12)
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_income_class.png"), bbox_inches = "tight")
    plt.close()

    # by sex
    fig, ax      = plt.subplots(figsize = (6, 5))
    sexes        = [0, 1]
    sexes_labels = ["Male", "Female"]

    for i, mode in enumerate(MODE_ORDER):
        color     = MODE_COLORS[i]
        actual    = target_mode_shares["sex"].loc[sexes, mode]
        simulated = simulated_mode_shares["sex"].reindex(sexes)[mode]

        ax.plot(sexes, actual, color = color,
                linestyle  = DATASET_STYLES["Target"]["linestyle"],
                marker     = DATASET_STYLES["Target"]["marker"],
                linewidth  = DATASET_STYLES["Target"]["linewidth"],
                markersize = DATASET_STYLES["Target"]["markersize"],
                label      = f"{mode} (Microcensus)")

        ax.plot(sexes, simulated, color = color,
                linestyle  = DATASET_STYLES["Simulated"]["linestyle"],
                marker     = DATASET_STYLES["Simulated"]["marker"],
                linewidth  = DATASET_STYLES["Simulated"]["linewidth"],
                markersize = DATASET_STYLES["Simulated"]["markersize"],
                label      = f"{mode} ({dataset_label})")

    ax.set_ylabel("Mode Share", fontsize = 13)
    ax.set_xlabel("Sex", fontsize = 13)
    ax.set_xticks(sexes)
    ax.set_xticklabels(sexes_labels)
    plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.2), ncol = 5, fontsize = 10)
    plt.grid(axis = "y", linestyle = "--", alpha = 0.7)
    _ = plt.xticks(rotation = 0, fontsize = 12)
    _ = plt.yticks(rotation = 0, fontsize = 12)
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_sex.png"), bbox_inches = "tight")
    plt.close()

    # plot by purpose
    fig, ax      = plt.subplots(figsize = (11, 5))
    purpose_ids  = sorted(target_mode_shares["purpose"].index.unique())

    for i, mode in enumerate(MODE_ORDER):
        color     = MODE_COLORS[i]
        actual    = target_mode_shares["purpose"].loc[purpose_ids, mode]
        simulated = simulated_mode_shares["purpose"].reindex(purpose_ids)[mode]

        ax.plot(purpose_ids, actual, color = color,
                linestyle  = DATASET_STYLES["Target"]["linestyle"],
                marker     = DATASET_STYLES["Target"]["marker"],
                linewidth  = DATASET_STYLES["Target"]["linewidth"],
                markersize = DATASET_STYLES["Target"]["markersize"],
                label      = f"{mode} (Microcensus)")

        ax.plot(purpose_ids, simulated, color = color,
                linestyle  = DATASET_STYLES["Simulated"]["linestyle"],
                marker     = DATASET_STYLES["Simulated"]["marker"],
                linewidth  = DATASET_STYLES["Simulated"]["linewidth"],
                markersize = DATASET_STYLES["Simulated"]["markersize"],
                label      = f"{mode} ({dataset_label})")

    ax.set_ylabel("Mode Share", fontsize = 13)
    ax.set_xlabel("Purpose", fontsize = 13)
    plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.2), ncol = 5, fontsize = 10)
    plt.tight_layout()
    plt.grid(axis = "y", linestyle = "--", alpha = 0.7)
    _ = plt.xticks(rotation = 0, fontsize = 12)
    _ = plt.yticks(rotation = 0, fontsize = 12)
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_purpose.png"), bbox_inches = "tight")
    plt.close()

    # plot the distribution of distances by mode
    fig, ax       = plt.subplots(1, 5, figsize = (20, 4), sharey = True)
    distance_bins = np.linspace(0, 50, 51)  # 20 bins from 0 to 50 km
    factors       = {"car": 1, "pt": 1, "walk": 0.5, "bike": 0.5, "car_passenger": 1}

    for mode in MODE_ORDER:
        f = factors[mode]
        simulated_distances = simulated_mode_shares["distance_by_mode"][mode]
        target_distances    = target_mode_shares["distance_by_mode"][mode]["distances"]
        target_weights      = target_mode_shares["distance_by_mode"][mode]["weights"]

        simulated_distances = simulated_distances[simulated_distances < 50]
        sel                 = target_distances < 50
        target_distances    = target_distances[sel]
        target_weights      = target_weights[sel]

        index = MODE_ORDER.index(mode)

        ax[index].hist(simulated_distances, 
                       bins    = distance_bins * f, 
                       alpha   = 0.6, 
                       label   = dataset_label,
                       color   = DATASET_STYLES["Simulated"]["color"], 
                       density = True)
        
        ax[index].hist(target_distances, 
                       bins    = distance_bins * f, 
                       alpha   = 0.6, 
                       label   = "Microcensus", 
                       color   = DATASET_STYLES["Target"]["color"], 
                       density = True)
        
        if len(simulated_distances):
            ax[index].axvline(np.mean(simulated_distances),
                             color     = DATASET_STYLES["Simulated"]["color"],
                             linestyle = "--",
                             linewidth = 2,
                             label     = f"Mean ({dataset_label}): {np.mean(simulated_distances):.1f}")

        if target_weights.sum() > 0:
            ax[index].axvline(np.average(target_distances, weights = target_weights),
                             color     = DATASET_STYLES["Target"]["color"],
                             linestyle = "--",
                             linewidth = 2,
                             label     = f"Mean (Microcensus): {np.average(target_distances, weights = target_weights):.1f}")

        ax[index].set_title(mode.replace("_", " ").title())
        ax[index].set_xlabel("Distance [km]")
        ax[index].set_ylabel("Frequency")
        ax[index].legend(fontsize = 8)
        ax[index].set_ylabel("")

    plt.ylim([0, 0.3])
    ax[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "distance_distribution_by_mode.png"), bbox_inches = "tight", dpi = 300)
    plt.close()

    # plot the distribution of distances by mode and purpose
    purposes      = target_mode_shares["distance_by_mode_and_purpose"].keys()
    fig, ax       = plt.subplots(len(purposes), len(MODE_ORDER), figsize = (20, 4 * len(purposes)), sharey = True, squeeze = False)
    distance_bins = np.linspace(0, 50, 51)  # 20 bins from 0 to 50 km

    for i, purpose in enumerate(purposes):
        for j, mode in enumerate(MODE_ORDER):
            f = factors[mode]
            simulated_distances = simulated_mode_shares["distance_by_mode_and_purpose"].get(purpose, {}).get(mode, np.array([]))
            target_distances    = target_mode_shares["distance_by_mode_and_purpose"][purpose][mode]["distances"]
            target_weights      = target_mode_shares["distance_by_mode_and_purpose"][purpose][mode]["weights"]

            simulated_distances = simulated_distances[simulated_distances < 50]
            sel                 = target_distances < 50
            target_distances    = target_distances[sel]
            target_weights      = target_weights[sel]

            ax[i, j].hist(simulated_distances, 
                          bins    = distance_bins * f, 
                          alpha   = 0.6, 
                          label   = dataset_label,
                          color   = DATASET_STYLES["Simulated"]["color"], 
                          density = True)
            
            ax[i, j].hist(target_distances, 
                          bins    = distance_bins * f, 
                          alpha   = 0.6, 
                          label   = "Microcensus", 
                          color   = DATASET_STYLES["Target"]["color"], 
                          density = True)

            if len(simulated_distances):
                ax[i, j].axvline(np.mean(simulated_distances),
                                color     = DATASET_STYLES["Simulated"]["color"],
                                linestyle = "--",
                                linewidth = 2,
                                label     = f"Mean ({dataset_label}): {np.mean(simulated_distances):.1f}")

            if target_weights.sum() > 0:
                ax[i, j].axvline(np.average(target_distances, weights = target_weights),
                                color     = DATASET_STYLES["Target"]["color"],
                                linestyle = "--",
                                linewidth = 2,
                                label     = f"Mean (Microcensus): {np.average(target_distances, weights = target_weights):.1f}")

            if i == 0:
                ax[i, j].set_title(mode.replace("_", " ").title())

            if j == 0:
                ax[i, j].set_ylabel(purpose.title())

            ax[i, j].set_xlabel("Distance [km]")
            ax[i, j].legend(fontsize = 8)
            ax[i, j].set_ylim([0, 0.6])

    plt.savefig(os.path.join(figures_dir, "distance_distribution_by_mode_and_purpose.png"), bbox_inches = "tight", dpi = 300)
    plt.close()

    # Plot the French survey and simulation when external residents are included.

    if context.config("include_external_population"):
        plot_french_comparison(target_mode_shares["french"], simulated_mode_shares["french"],
                               Path(figures_dir) / "french", dataset_label=dataset_label)

    # Build the interactive comparison from the same filtered trip cohorts.

    from .spatial import write_report
    municipalities, _ = context.stage("data.spatial.municipalities")
    report = write_report(figures_dir, target_mode_shares["spatial_trips"],
                          simulated_mode_shares["spatial_trips"], municipalities,
                          context.stage("data.spatial.cantons"), dataset_label=dataset_label)
    logger.info("Interactive spatial mode shares: %s", report)

    # Return the existing output contract with the HTML path added.

    return dict(done=True, path=figures_dir, report=report)


def validate(context):
    from .spatial import implementation_token
    return implementation_token()


def plot_distance_panel(axes, data, labels, title, dataset_label="MATSim"):
    """Show shares and their sample support; leave empty bins undefined."""
    positions = np.arange(len(labels))
    for source, style, marker, offset, colour in [("simulation", "--", "o", -.19, "#7b8c9c"),
                                                ("survey", "-", "s", .19, "#c78f4a")]:
        selected = data.loc[data.source.eq(source)]
        shares = selected.pivot(index="distance_bin_km", columns="mode", values="share").reindex(labels)
        totals = selected.drop_duplicates("distance_bin_km").set_index("distance_bin_km").bin_sample_trips.reindex(labels)
        for mode, mode_colour in zip(MODE_ORDER, MODE_COLORS):
            axes[0].plot(positions, shares[mode], color=mode_colour, marker=marker, markersize=4,
                         linestyle=style, linewidth=1.8)
        axes[1].bar(positions + offset, totals, color=colour, width=.36, label=source.title())
    axes[0].set_title(title, fontsize=13, pad=12)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of trips")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].grid(axis="y", alpha=.25)
    handles = [Line2D([], [], color=colour, label=MODE_LABELS[mode])
               for mode, colour in zip(MODE_ORDER, MODE_COLORS)]
    handles += [Line2D([], [], color="#333", linestyle=style, marker=marker, label=label)
                for style, marker, label in [("--", "o", dataset_label), ("-", "s", "Survey target")]]
    axes[0].legend(handles=handles, ncol=3, fontsize=9, loc="upper right")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_ylabel("Sample\ntrips", fontsize=9)
    axes[1].set_xticks(positions, labels, rotation=35, ha="right", fontsize=9)
    axes[1].set_xlabel("Euclidean trip distance (km; common distance bins)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)


def plot_overall_shares(data, path, dataset_label="MATSim"):
    figure, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True, constrained_layout=True)
    positions = np.arange(len(MODE_ORDER))
    for axis, cohort in zip(axes, FRENCH_COHORTS):
        for source, offset, colour in [("simulation", -.2, "#4c78a8"), ("survey", .2, "#e0a04b")]:
            selected = data.loc[data.cohort.eq(cohort) & data.source.eq(source)].set_index("mode").reindex(MODE_ORDER)
            bars = axis.bar(positions + offset, selected.share, width=.38, color=colour,
                            label=dataset_label if source == "simulation" else "Survey target")
            axis.bar_label(bars, labels=[f"{value:.1%}" for value in selected.share], fontsize=8, padding=3)
        axis.set_xticks(positions, [MODE_LABELS[mode] for mode in MODE_ORDER], rotation=20, ha="right")
        axis.set_title(FRENCH_COHORTS[cohort])
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylim(0, .85)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.legend()
    axes[0].set_ylabel("Share of trips")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_french_comparison(target, simulated, output, *, dataset_label="MATSim"):
    """Write optional French figures from the target and simulated stage summaries."""
    output.mkdir(parents=True, exist_ok=True)
    labels = target["distance_labels"]
    if labels != simulated["distance_labels"]:
        raise ValueError("French target and simulation must use the same distance bins.")
    overall = pd.concat([simulated["overall"], target["overall"]], ignore_index=True)
    binned = pd.concat([simulated["distance"], target["distance"]], ignore_index=True)
    overall.to_csv(output / "mode_shares_overall.csv", index=False)
    binned.to_csv(output / "mode_shares_by_distance.csv", index=False)
    for cohort, title in FRENCH_COHORTS.items():
        figure, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True,
                                   gridspec_kw={"height_ratios": [4, 1]}, constrained_layout=True)
        plot_distance_panel(axes, binned.loc[binned.cohort.eq(cohort)], labels, title, dataset_label)
        figure.savefig(output / f"{cohort}_by_distance.png", dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(19, 7), sharex=True,
                               gridspec_kw={"height_ratios": [4, 1]}, constrained_layout=True)
    for column, (cohort, title) in enumerate(FRENCH_COHORTS.items()):
        plot_distance_panel(axes[:, column], binned.loc[binned.cohort.eq(cohort)], labels, title, dataset_label)
    figure.savefig(output / "french_mode_shares_by_distance.png", dpi=180)
    plt.close(figure)
    plot_overall_shares(overall, output / "french_mode_shares_overall.png", dataset_label)
    manifest = {"simulation": simulated["metadata"], "survey_target": target["metadata"],
                "switzerland_involved": "Origin OR destination in Switzerland, including trips within Switzerland; each trip counted once.",
                "denominator": "Weighted trips within each cohort or distance bin; empty bins have undefined shares."}
    (output / "analysis_sources.json").write_text(json.dumps(manifest, indent=2))
    logger.info("French mode-share figures: %s", output)
