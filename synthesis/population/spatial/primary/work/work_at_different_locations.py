import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data.structural_survey.structural_survey import get_filtered_data

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("synthesis.population.sampled")
    context.config("random_seed")


def add_age_bin(df):
    # Coarse bins reduce sparsity and overfitting.
    age_bins = [0, 25, 41, 66, 200]
    age_labels = [0, 1, 2, 3]
    df = df.copy()
    df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False, include_lowest=True)
    df["age_bin"] = df["age_bin"].cat.add_categories([-1]).fillna(-1).astype(int)
    return df

def sanitize_features(df, columns):
    df = df.copy()
    for c in columns:
        if c in df.columns:
            df[c] = df[c].fillna(-1).astype(int)
    return df

def group_job_positions(df):
    # Group job positions into broader categories to reduce sparsity
    df = df.copy()
    mapping = {
        11: "self_employed_owner",
        12: "self_employed_owner",
        31: "self_employed_owner",
        32: "self_employed_owner",
        20: "self_employed_owner",
        41: "management",
        42: "management",
        43: "employee",
        50: "apprentice",
        60: "not_working",
        70: "not_working",
        -1: "unknown"
    }
    df["job_position"] = df["job_position"].map(mapping).fillna("unknown")
    return df

def aggregate_smoothed_rate(df, group_cols, alpha, beta):
    tmp = df[group_cols + ["weight", "work_diff_locations"]].copy()
    tmp["weighted_remote"] = tmp["weight"] * tmp["work_diff_locations"]
    agg = tmp.groupby(group_cols, dropna=False).agg(
        s=("weighted_remote", "sum"),
        n=("weight", "sum")
    ).reset_index()
    agg["p"] = (agg["s"] + alpha) / (agg["n"] + alpha + beta)
    return agg[group_cols + ["n", "p"]]


def execute(context):
    # Load survey and population    
    df_survey = get_filtered_data(context, "all")[[
        "home_municipality_id", "employed", "job_position", "age", "sex",
        "weight", "nationality", "start_work", "canton_id", "home_zone_id"
    ]].copy()

    df_population = context.stage("synthesis.population.sampled")[[
        "household_id", "person_id", "sex", "age", "home_municipality_id",
        "employed", "nationality", "canton_id", "job_position", "home_x", "home_y"
    ]].copy()

    num_agents = (df_population["employed"]==1).sum()
    # Prepare survey observations    
    df_survey = df_survey[~df_survey["home_municipality_id"].isna()]
    df_survey = df_survey[df_survey["start_work"].isin([1, 2, 3, 4, 5, 6])]

    # 1 means "at domicile" in structural survey coding -> remote work
    df_survey["work_diff_locations"] = (df_survey["start_work"] == 2).astype(float)
    df_survey["weight"] = df_survey["weight"].fillna(df_survey["weight"].mean()).clip(lower=0.0)

    survey_remote_share = np.average(df_survey["work_diff_locations"], weights=df_survey["weight"])
    logger.info(
        f"Remote share in employed structural survey: {100.0 * survey_remote_share:.2f}%%"
    )

    # Prepare modeling features
    base_features = ["home_municipality_id", "job_position", "sex", "age_bin", "nationality", "canton_id"]
    df_survey = add_age_bin(df_survey)
    df_population = add_age_bin(df_population)

    df_survey = sanitize_features(df_survey, base_features)
    df_population = sanitize_features(df_population, base_features)

    # grouping job positions
    df_survey = group_job_positions(df_survey)
    df_population = group_job_positions(df_population)
    
    # Global prior for shrinkage (prevents overfitting on sparse cells)
    prior_strength = 60.0
    alpha = survey_remote_share * prior_strength
    beta = (1.0 - survey_remote_share) * prior_strength

    # Hierarchical groupings from specific -> general
    group_levels = [
        ["canton_id"],
        ["home_municipality_id"],
        ["canton_id", "sex", "age_bin"],
        ["home_municipality_id", "sex", "age_bin"],    
        ["home_municipality_id", "sex", "age_bin", "job_position"],    
        ["home_municipality_id", "sex", "age_bin", "nationality", "job_position"]              
    ]

    # Predict for employed population only
    employed_mask = df_population["employed"] == 1
    df_employed = df_population.loc[employed_mask].copy()

    # Start from global rate and refine by hierarchical shrinkage
    p = np.full(len(df_employed), survey_remote_share, dtype=float)

    # Controls how fast we trust a group estimate as sample size grows
    blend_tau = 100
    pop_threshold = 50
    with context.progress(total=len(group_levels)+1, label="Prediction work at different locations ") as prog:
        for level in group_levels:
            agg = aggregate_smoothed_rate(df_survey, level, alpha, beta)
            merged = df_employed[level].merge(agg, on=level, how="left")
            n_level = merged["n"].to_numpy(dtype=float)
            p_level = merged["p"].to_numpy(dtype=float)

            found = ~np.isnan(p_level) & (n_level >= pop_threshold)
            # Weight toward group estimate when group has more weighted observations
            w = np.zeros_like(p)
            w[found] = n_level[found] / (n_level[found] + blend_tau)
            p[found] = w[found] * p_level[found] + (1.0 - w[found]) * p[found]
            prog.update()

        p = np.clip(p, 0.0, 1.0)
        # Probabilistic assignment
        rng = np.random.RandomState(context.config("random_seed"))
        remote_flag = rng.random(len(df_employed)) < p

        # Build output for all agents
        out = df_population[["person_id"]].copy()
        out["p_work_diff_locations"] = 0.0
        out["work_diff_locations"] = False

        out.loc[employed_mask, "p_work_diff_locations"] = p
        out.loc[employed_mask, "work_diff_locations"] = remote_flag
        prog.update()

    logger.info(
        "Survey work at different locations: %.2f%% | Assigned: %.2f%% | Avg prob: %.2f%%" % (
            100.0 * survey_remote_share,
            100.0 * out.loc[employed_mask, "work_diff_locations"].mean(),
            100.0 * out.loc[employed_mask, "p_work_diff_locations"].mean()
        )
    )
    # plot analysis
    plot_analysis(context, df_survey, df_population, out)
    
    # only keep those agents
    out = out[out["work_diff_locations"] == True].reset_index(drop=True)
    assert num_agents * 0.3 > len(out), f"We cannot have more than 30% of the employed population working from different locations"
    return out[["person_id"]]




def plot_analysis(context, df_survey, df_population, out):    
    # Define groupings for subplots
    groupings = [
        (["canton_id"], "By Canton"),
        (["home_municipality_id"], "By Municipality"),
        (["home_municipality_id","sex"], "By Municipality and Sex"),
        (["canton_id", "sex"], "By Canton and Sex"),
        (["canton_id", "age_bin", "sex"], "By Canton, Age Bin, and Sex"),  
        (["canton_id", "job_position", "nationality"], "By Canton, Job Position, and Nationality"),
    ]
    employed_mask = df_population["employed"] == 1
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (group_cols, title) in enumerate(groupings):
        # Survey group remote share
        survey_group = df_survey.groupby(group_cols).apply(
            lambda x: np.average(x["work_diff_locations"], weights=x["weight"])
        ).reset_index(name="survey_work_diff_locations_share")

        # Assigned group remote share
        assigned_group = out.copy()
        for col in group_cols:
            assigned_group[col] = df_population[col]
        assigned_group = assigned_group[employed_mask]
        num_obs = assigned_group.groupby(group_cols).person_id.transform(len)
        assigned_group = assigned_group[num_obs>100]
        assigned_group = assigned_group.groupby(group_cols)["work_diff_locations"].mean().reset_index(name="assigned_work_diff_locations_share")

        # Merge for comparison
        group_compare = pd.merge(survey_group, assigned_group, on=group_cols, how="inner")

        # Plot
        ax = axes[idx]
        
        if len(group_cols) > 1:
            # Color by the second column
            unique_colors = group_compare[group_cols[1]].unique()
            color_map = {val: plt.cm.tab10(i) for i, val in enumerate(unique_colors)}
            colors = group_compare[group_cols[1]].map(color_map)
            
            for color_val in unique_colors:
                mask = group_compare[group_cols[1]] == color_val
                ax.scatter(group_compare[mask]["survey_work_diff_locations_share"], 
                          group_compare[mask]["assigned_work_diff_locations_share"], 
                          alpha=0.7, label=f"{group_cols[1]}={color_val}", color=color_map[color_val])
        else:
            ax.scatter(group_compare["survey_work_diff_locations_share"], 
                      group_compare["assigned_work_diff_locations_share"], alpha=0.7)
        
        ax.plot([0, 0.2], [0, 0.2], 'r--', label="y=x")
        ax.set_xlabel("Survey Work Diff Locations Share")
        ax.set_ylabel("Assigned Work Diff Locations Share")
        ax.set_title(f"Work Diff Locations Share: {title}")
        ax.legend()

    plt.tight_layout()
    path_to_figure = os.path.join(context.path(), "work_diff_locations_share_comparison.png")
    plt.savefig(path_to_figure, dpi=200, bbox_inches="tight")
    plt.close()    