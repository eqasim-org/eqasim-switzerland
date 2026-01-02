import logging
import os
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd

logger = logging.getLogger("synpp")

MODE_ORDER = ["car", "pt", "walk", "bike", "car_passenger"]
MODE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # Distinct colors for modes
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
    context.stage("analysis.mode_shares.target")
    context.stage("analysis.mode_shares.simulated")
    context.stage("data.spatial.cantons")

    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default="simulation_output")


def execute(context):
    figures_dir = os.path.join(
        context.config("output_path"),
        context.config("output_id"),
        context.config("simulation_directory"),
        "mode_shares",
    )
    os.makedirs(figures_dir, exist_ok=True)

    # load data
    target_mode_shares = context.stage("analysis.mode_shares.target")
    simulated_mode_shares = context.stage("analysis.mode_shares.simulated")

    # plot mode shares by distance        
    fig, ax = plt.subplots(figsize=(11, 5))
    distance_bins = np.array(target_mode_shares["distance_bins"])
    distance_labels = target_mode_shares["distance_labels"]

    distances = (distance_bins[1:] + distance_bins[:-1]) / 2
    distances[-1] = distance_bins[-2]

    for i, mode in enumerate(MODE_ORDER):
        color = MODE_COLORS[i]
        actual = target_mode_shares["distance"][mode]
        simulated = simulated_mode_shares["distance"][mode]
                
        ax.plot(distances, actual, color=color, linestyle=DATASET_STYLES["Target"]["linestyle"], 
                marker=DATASET_STYLES["Target"]["marker"], linewidth=DATASET_STYLES["Target"]["linewidth"], 
                markersize=DATASET_STYLES["Target"]["markersize"], label=f"{mode} (Microcensus)")
        ax.plot(distances, simulated, color=color, linestyle=DATASET_STYLES["Simulated"]["linestyle"], 
                marker=DATASET_STYLES["Simulated"]["marker"], linewidth=DATASET_STYLES["Simulated"]["linewidth"], 
                markersize=DATASET_STYLES["Simulated"]["markersize"], label=f"{mode} (MATSim)")

    ax.set_ylabel("Mode Share", fontsize=13)
    ax.set_xlabel("Euclidean distance [km]", fontsize=13)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5, fontsize=10)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    _ = plt.xticks(rotation=0, fontsize=12)
    _ = plt.yticks(rotation=0, fontsize=12)
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_distance.png"), bbox_inches="tight")
    plt.close()

    # Plot mode shares by canton 
    df_cantons = context.stage("data.spatial.cantons")[["canton_id","canton_name_en"]]
    df_cantons = df_cantons.rename(columns={"canton_name_en":"canton_name"})

    target_canton = (target_mode_shares["canton"].reset_index()
                     .merge(df_cantons).set_index("canton_name")
                     .drop(columns=["canton_id"])
                     .sort_values("car"))
    simulated_canton = (simulated_mode_shares["canton"].reset_index()
                       .merge(df_cantons).set_index("canton_name")
                       .drop(columns=["canton_id"])
                       .reindex(target_canton.index))

    fig, ax = plt.subplots(1, len(MODE_ORDER), figsize=(10, 7), sharey=True)    

    for j, mode in enumerate(MODE_ORDER):    
        ax[j].plot(
            simulated_canton[mode], simulated_canton.index,
            linestyle=DATASET_STYLES["Simulated"]["linestyle"], marker=DATASET_STYLES["Simulated"]["marker"], 
            label="Simulated", color=DATASET_STYLES["Simulated"]["color"],
            linewidth=DATASET_STYLES["Simulated"]["linewidth"], markersize=DATASET_STYLES["Simulated"]["markersize"]
        )
        ax[j].plot(
            target_canton[mode], target_canton.index,
            linestyle=DATASET_STYLES["Target"]["linestyle"], marker=DATASET_STYLES["Target"]["marker"], 
            label="Actual", color=DATASET_STYLES["Target"]["color"], 
            alpha=0.6, linewidth=DATASET_STYLES["Target"]["linewidth"], 
            markersize=DATASET_STYLES["Target"]["markersize"]
        )
        ax[j].set_xlabel("Mode Share", fontsize=12, labelpad=16)
        ax[j].set_title(mode.replace('_',' '), fontsize=14)
        ax[j].grid(axis='y', linestyle='--', alpha=0.5)
        
        max_shares = max(max(target_canton[mode]),max(simulated_canton[mode]), 0.2)
        min_shares = min(min(target_canton[mode]),min(simulated_canton[mode]))
        ax[j].set_xlim([0.2*min_shares, 1.2*max_shares])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, ["MATSim", "Microcensus"],
        loc="upper center", bbox_to_anchor=(0.6, 1.04),
        ncol=2, fontsize=14, frameon=False
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_canton.png"), bbox_inches="tight")
    plt.close()

    # plot global mode shares
    fig, ax = plt.subplots(figsize=(8,3))
    shares = pd.DataFrame(dict(MATSim=simulated_mode_shares["global"].loc[MODE_ORDER, "mode_share"], 
                               Microcensus=target_mode_shares["global"].loc[MODE_ORDER, "mode_share"]))
    shares.plot.bar(ax=ax, width=0.8, color=[DATASET_STYLES["Target"]["color"], DATASET_STYLES["Simulated"]["color"]])
    plt.xticks(rotation=0)
    plt.ylabel("Mode Share", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # write the values on top of the bars
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.01, f"{i.get_height():.1%}", ha='center', fontsize=9)

    plt.ylim([0, 0.48])
    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "global_mode_shares.png"), bbox_inches="tight")
    plt.close()

    # by age
    fig, ax = plt.subplots(figsize=(11, 5))
    age_bins = np.array(target_mode_shares["age_bins"])
    age_labels = target_mode_shares["age_labels"]
    ages = (age_bins[1:] + age_bins[:-1]) / 2
    ages[-1] = age_bins[-2]
    for i, mode in enumerate(MODE_ORDER):
        color = MODE_COLORS[i]
        actual = target_mode_shares["age"][mode]
        simulated = simulated_mode_shares["age"][mode]
                
        ax.plot(ages, actual, color=color, linestyle=DATASET_STYLES["Target"]["linestyle"], 
                marker=DATASET_STYLES["Target"]["marker"], linewidth=DATASET_STYLES["Target"]["linewidth"], 
                markersize=DATASET_STYLES["Target"]["markersize"], label=f"{mode} (Microcensus)")
        ax.plot(ages, simulated, color=color, linestyle=DATASET_STYLES["Simulated"]["linestyle"], 
                marker=DATASET_STYLES["Simulated"]["marker"], linewidth=DATASET_STYLES["Simulated"]["linewidth"], 
                markersize=DATASET_STYLES["Simulated"]["markersize"], label=f"{mode} (MATSim)")
    ax.set_ylabel("Mode Share", fontsize=13)
    ax.set_xlabel("Age [years]", fontsize=13)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5, fontsize=10)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    _ = plt.xticks(rotation=0, fontsize=12)
    _ = plt.yticks(rotation=0, fontsize=12)
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_age.png"), bbox_inches="tight")
    plt.close()

    # by income class
    fig, ax = plt.subplots(figsize=(11, 5))
    income_classes = sorted(target_mode_shares["income"].index.unique())
    for i, mode in enumerate(MODE_ORDER):
        color = MODE_COLORS[i]
        actual = target_mode_shares["income"].loc[income_classes, mode]
        simulated = simulated_mode_shares["income"].loc[income_classes, mode]
                
        ax.plot(income_classes, actual, color=color, linestyle=DATASET_STYLES["Target"]["linestyle"], 
                marker=DATASET_STYLES["Target"]["marker"], linewidth=DATASET_STYLES["Target"]["linewidth"], 
                markersize=DATASET_STYLES["Target"]["markersize"], label=f"{mode} (Microcensus)")
        ax.plot(income_classes, simulated, color=color, linestyle=DATASET_STYLES["Simulated"]["linestyle"], 
                marker=DATASET_STYLES["Simulated"]["marker"], linewidth=DATASET_STYLES["Simulated"]["linewidth"], 
                markersize=DATASET_STYLES["Simulated"]["markersize"], label=f"{mode} (MATSim)")
    ax.set_ylabel("Mode Share", fontsize=13)
    ax.set_xlabel("Income Class", fontsize=13)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5, fontsize=10)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    _ = plt.xticks(rotation=0, fontsize=12)
    _ = plt.yticks(rotation=0, fontsize=12)
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_income_class.png"), bbox_inches="tight")
    plt.close()

    # by sex
    fig, ax = plt.subplots(figsize=(6, 5))
    sexes = [0, 1]
    sexes_labels = ["Male", "Female"]
    for i, mode in enumerate(MODE_ORDER):
        color = MODE_COLORS[i]
        actual = target_mode_shares["sex"].loc[sexes, mode]
        simulated = simulated_mode_shares["sex"].loc[sexes, mode]
        ax.plot(sexes, actual, color=color, linestyle=DATASET_STYLES["Target"]["linestyle"], 
                marker=DATASET_STYLES["Target"]["marker"], linewidth=DATASET_STYLES["Target"]["linewidth"], 
                markersize=DATASET_STYLES["Target"]["markersize"], label=f"{mode} (Microcensus)")
        ax.plot(sexes, simulated, color=color, linestyle=DATASET_STYLES["Simulated"]["linestyle"], 
                marker=DATASET_STYLES["Simulated"]["marker"], linewidth=DATASET_STYLES["Simulated"]["linewidth"], 
                markersize=DATASET_STYLES["Simulated"]["markersize"], label=f"{mode} (MATSim)")
    ax.set_ylabel("Mode Share", fontsize=13)
    ax.set_xlabel("Sex", fontsize=13)
    ax.set_xticks(sexes)
    ax.set_xticklabels(sexes_labels)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5, fontsize=10)    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    _ = plt.xticks(rotation=0, fontsize=12)
    _ = plt.yticks(rotation=0, fontsize=12)
    plt.savefig(os.path.join(figures_dir, "mode_shares_by_sex.png"), bbox_inches="tight")
    plt.close()

    return dict(done=True, path=figures_dir)