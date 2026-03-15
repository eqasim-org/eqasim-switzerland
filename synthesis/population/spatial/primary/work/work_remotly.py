import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("synthesis.population.models.employment")
    context.config("random_seed")


def _add_age_bin(df):
    # Coarse bins reduce sparsity and overfitting.
    age_bins = [0, 24, 41, 66, 200]
    age_labels = [0, 1, 2, 3]
    df = df.copy()
    df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False, include_lowest=True)
    df["age_bin"] = df["age_bin"].cat.add_categories([-1]).fillna(-1).astype(int)
    return df


def _sanitize_features(df, columns):
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
    df["job_position"] = df["job_position"].map(mapping).fillna("Other")
    return df

def _aggregate_smoothed_rate(df, group_cols, alpha, beta):
    tmp = df[group_cols + ["weight", "work_remotly"]].copy()
    tmp["weighted_remote"] = tmp["weight"] * tmp["work_remotly"]
    agg = tmp.groupby(group_cols, dropna=False).agg(
        s=("weighted_remote", "sum"),
        n=("weight", "sum")
    ).reset_index()
    agg["p"] = (agg["s"] + alpha) / (agg["n"] + alpha + beta)
    return agg[group_cols + ["n", "p"]]


def execute(context):
    # Load survey and population    
    df_survey = context.stage("data.structural_survey.structural_survey")[[
        "home_zone_id", "employed", "job_position", "age", "sex",
        "weight", "nationality", "start_work", "canton_id","home_municipality_id"
    ]].copy()

    df_population = context.stage("synthesis.population.models.employment")[[
        "person_id", "sex", "age", "home_zone_id","home_municipality_id",
        "employed", "nationality", "canton_id", "job_position"
    ]].copy()

    # Prepare survey observations
    df_survey = df_survey[df_survey["employed"] == 1].copy()
    df_survey = df_survey[~df_survey["home_zone_id"].isna()]
    df_survey = df_survey[~df_survey["start_work"].isna()]
    df_survey = df_survey[df_survey["start_work"].isin([1, 2, 3, 4, 5, 6])].copy()

    # 1 means "at domicile" in structural survey coding -> remote work
    df_survey["work_remotly"] = (df_survey["start_work"] == 1).astype(float)
    df_survey["weight"] = df_survey["weight"].fillna(df_survey["weight"].mean()).clip(lower=0.0)

    survey_remote_share = np.average(df_survey["work_remotly"], weights=df_survey["weight"])
    logger.info(
        f"Remote share in employed structural survey: {100.0 * survey_remote_share:.2f}%%"
    )

    # Prepare modeling features
    base_features = ["home_zone_id", "job_position", "sex", "age_bin", "nationality", "canton_id"]
    df_survey = _add_age_bin(df_survey)
    df_population = _add_age_bin(df_population)

    df_survey = _sanitize_features(df_survey, base_features)
    df_population = _sanitize_features(df_population, base_features)

    # grouping job positions
    df_survey = group_job_positions(df_survey)
    df_population = group_job_positions(df_population)
    
    # Global prior for shrinkage (prevents overfitting on sparse cells)
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
        ["home_municipality_id", "sex", "age_bin", "nationality"],    
        ["home_municipality_id", "sex", "age_bin", "nationality", "job_position"]        
    ]

    # Predict for employed population only
    employed_mask = df_population["employed"] == 1
    df_employed = df_population.loc[employed_mask].copy()

    # Start from global rate and refine by hierarchical shrinkage
    p = np.full(len(df_employed), survey_remote_share, dtype=float)

    # Controls how fast we trust a group estimate as sample size grows
    blend_tau = 150
    pop_threshold = 50
    with context.progress(total=len(group_levels)+1, label="Prediction remote working ") as prog:
        for level in group_levels:
            agg = _aggregate_smoothed_rate(df_survey, level, alpha, beta)
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
        out["p_work_remotly"] = 0.0
        out["work_remotly"] = False

        out.loc[employed_mask, "p_work_remotly"] = p
        out.loc[employed_mask, "work_remotly"] = remote_flag
        prog.update()

    logger.info(
        "Survey remote work: %.2f%% | Assigned: %.2f%% | Avg prob: %.2f%%" % (
            100.0 * survey_remote_share,
            100.0 * out.loc[employed_mask, "work_remotly"].mean(),
            100.0 * out.loc[employed_mask, "p_work_remotly"].mean()
        )
    )

    return out[["person_id", "work_remotly"]]