import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("synpp")


def configure(context):
    context.config("analysis_path")
    context.config("specific_day_scenario", default="workday")
    context.config("compare_matching_top_chains", 15)
    context.config("compare_matching_cantons", [22, 1, 2, 19,25])

    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.matching.matched_v1")
    context.stage("synthesis.population.matching.matched_v2")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.activity_chains")


def _extract_matching_df(stage_result):
    value = stage_result.get() if hasattr(stage_result, "get") else stage_result
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], pd.DataFrame):
        value = value[0]
    if not isinstance(value, pd.DataFrame):
        raise RuntimeError("Matching stage output is not a DataFrame.")
    if "mz_person_id" not in value.columns:
        raise RuntimeError("Matching DataFrame must contain 'mz_person_id'.")
    return value.copy()


def _sanitize_chain(chain):
    if pd.isna(chain):
        return "H"
    chain = str(chain)
    if chain.strip() == "":
        return "H"
    mapping = {
        "home": "H", "work": "W", "work_secondary": "W", "education": "E",
        "shop": "S", "leisure": "L", "other": "O", "interaction": "I",
        "border": "B"
    }
    return "-".join([mapping.get(token, token[:1].upper()) for token in chain.split("-")])


def _weighted_distribution(df, value_col, weight_col=None):
    if len(df) == 0:
        return pd.Series(dtype=float)
    if weight_col is None:
        s = df[value_col].value_counts(normalize=True)
    else:
        s = df.groupby(value_col)[weight_col].sum()
        s = s / s.sum() if s.sum() > 0 else s
    return s


def _prepare_groups(cantons):
    groups = [
        ("all", None, None),
        ("women", 1, None),
        ("men", 0, None),
    ]
    for cid in cantons:
        groups.extend([
            (f"all_canton_{cid}", None, cid),
            (f"women_canton_{cid}", 1, cid),
            (f"men_canton_{cid}", 0, cid),
        ])
    return groups


def _apply_group_filter(df, sex=None, canton_id=None):
    out = df
    if sex is not None and "sex" in out.columns:
        out = out[out["sex"] == sex]
    if canton_id is not None and "canton_id" in out.columns:
        out = out[out["canton_id"] == canton_id]
    return out


def _normalize_sex_column(df, col="sex"):
    if col not in df.columns:
        return df

    s = df[col]
    if s.dtype == object:
        s_str = s.astype(str).str.strip().str.lower()
        map_str = {
            "male": 0, "man": 0, "m": 0,
            "female": 1, "woman": 1, "f": 1,
            "0": 0, "1": 1,
        }
        s_num = s_str.map(map_str)
        s_fallback = pd.to_numeric(s, errors="coerce")
        s = s_num.fillna(s_fallback)
    else:
        s = pd.to_numeric(s, errors="coerce")

    vals = set(pd.Series(s).dropna().astype(int).unique().tolist())
    if vals.issubset({0, 1}):
        df[col] = pd.Series(s, index=df.index).fillna(-1).astype(int)
        return df

    # Common coding in some inputs: 1=male, 2=female.
    if vals.issubset({1, 2}):
        df[col] = (pd.Series(s, index=df.index) - 1).fillna(-1).astype(int)
        return df

    # Last fallback: keep only 0/1 valid values, others as -1.
    s = pd.Series(s, index=df.index)
    s = s.where(s.isin([0, 1]), -1)
    df[col] = s.astype(int)
    return df


def _plot_chain_comparison(out_path, group_name, dist_micro, dist_v1, dist_v2, top_n=15):
    if len(dist_micro) == 0 and len(dist_v1) == 0 and len(dist_v2) == 0:
        return

    top_chains = list(dist_micro.sort_values(ascending=False).head(top_n).index)
    for s in [dist_v1, dist_v2]:
        for c in s.sort_values(ascending=False).head(top_n).index:
            if c not in top_chains:
                top_chains.append(c)
            if len(top_chains) >= top_n:
                break

    if len(top_chains) == 0:
        return

    y_micro = np.array([100.0 * dist_micro.get(c, 0.0) for c in top_chains])
    y_v1 = np.array([100.0 * dist_v1.get(c, 0.0) for c in top_chains])
    y_v2 = np.array([100.0 * dist_v2.get(c, 0.0) for c in top_chains])

    x = np.arange(len(top_chains))
    w = 0.28

    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(top_chains)), 5), dpi=220)
    ax.bar(x - w, y_micro, width=w, label="Microcensus", color="#1f4e79")
    ax.bar(x, y_v1, width=w, label="Matching v1", color="#7f8c8d")
    ax.bar(x + w, y_v2, width=w, label="Matching v2", color="#c0392b")

    ax.set_title(f"Activity chain shares - {group_name}")
    ax.set_ylabel("Share [%]")
    ax.set_xlabel("Activity chain")
    ax.set_xticks(x)
    ax.set_xticklabels(top_chains, rotation=40, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, f"matching_chains_{group_name}.png"))
    plt.close(fig)


def _compute_metrics(df_group, trips_stats, unmatched_as_zero=True):
    keep_cols = ["person_id", "employed", "mz_person_id"]
    if "person_weight" in df_group.columns:
        keep_cols.append("person_weight")
    d = df_group[keep_cols].copy()
    d = d.merge(trips_stats, how="left", left_on="mz_person_id", right_on="person_id", suffixes=("", "_mz"))
    for col in ["trip_count", "work_trip_count"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
        if unmatched_as_zero:
            d[col] = d[col].fillna(0)

    employed_mask = d["employed"] == 1
    if "person_weight" in d.columns:
        w = pd.to_numeric(d["person_weight"], errors="coerce").fillna(0).values
    else:
        w = np.ones((len(d),), dtype=float)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        w = np.ones((len(d),), dtype=float)
        w_sum = float(len(d)) if len(d) else 1.0

    w_emp = float(np.sum(w[employed_mask]))

    return {
        "employment_rate": float(np.sum(w * (d["employed"].values == 1)) / w_sum) if len(d) else 0.0,
        "avg_trips_per_person": float(np.sum(w * d["trip_count"].fillna(0).values) / w_sum) if len(d) else 0.0,
        "avg_work_trips_per_person": float(np.sum(w * d["work_trip_count"].fillna(0).values) / w_sum) if len(d) else 0.0,
        "avg_work_trips_per_employed": float(np.sum(w[employed_mask] * d.loc[employed_mask, "work_trip_count"].fillna(0).values) / w_emp) if w_emp > 0 else 0.0,
        "share_with_work_trip": float(np.sum(w * (d["work_trip_count"].fillna(0).values > 0)) / w_sum) if len(d) else 0.0,
        "share_unmatched": float(np.sum(w * (pd.to_numeric(d["mz_person_id"], errors="coerce").fillna(-1).values < 0)) / w_sum) if len(d) else 0.0,
    }


def _compute_micro_metrics(df_mz_group, trips_stats):
    d = df_mz_group[["person_id", "person_weight", "employed"]].copy()
    d = d.merge(trips_stats, how="left", on="person_id")
    d["trip_count"] = pd.to_numeric(d["trip_count"], errors="coerce").fillna(0)
    d["work_trip_count"] = pd.to_numeric(d["work_trip_count"], errors="coerce").fillna(0)
    w = pd.to_numeric(d["person_weight"], errors="coerce").fillna(0).values
    w_sum = np.sum(w)
    if w_sum <= 0:
        w = np.ones((len(d),), dtype=float)
        w_sum = float(len(d))

    employed = (d["employed"] == 1).values
    w_emp = np.sum(w[employed])

    return {
        "employment_rate": float(np.sum(w * employed) / w_sum),
        "avg_trips_per_person": float(np.sum(w * d["trip_count"].values) / w_sum),
        "avg_work_trips_per_person": float(np.sum(w * d["work_trip_count"].values) / w_sum),
        "avg_work_trips_per_employed": float(np.sum(w[employed] * d.loc[employed, "work_trip_count"].values) / w_emp) if w_emp > 0 else 0.0,
        "share_with_work_trip": float(np.sum(w * (d["work_trip_count"].values > 0)) / w_sum),
        "share_unmatched": 0.0,
    }


def _plot_metric_errors(out_path, group_name, metrics_micro, metrics_v1, metrics_v2):
    metric_names = [
        "employment_rate",
        "avg_trips_per_person",
        "avg_work_trips_per_person",
        "avg_work_trips_per_employed",
        "share_with_work_trip",
        "share_unmatched",
    ]
    labels = [
        "Employment rate",
        "Avg trips/person",
        "Avg work trips/person",
        "Avg work trips/employed",
        "Share with >=1 work trip",
        "Share unmatched",
    ]

    e1 = np.array([abs(metrics_v1[m] - metrics_micro[m]) for m in metric_names])
    e2 = np.array([abs(metrics_v2[m] - metrics_micro[m]) for m in metric_names])

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5), dpi=220)
    ax.bar(x - w / 2, e1, width=w, label="|v1 - micro|", color="#7f8c8d")
    ax.bar(x + w / 2, e2, width=w, label="|v2 - micro|", color="#c0392b")
    ax.set_title(f"Metric error vs microcensus - {group_name} (lower is better)")
    ax.set_ylabel("Absolute error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, f"matching_metric_errors_{group_name}.png"))
    plt.close(fig)


def execute(context):
    analysis_path = context.config("analysis_path")
    out_path = os.path.join(analysis_path, "matching_compare")
    os.makedirs(out_path, exist_ok=True)

    scenario_day = context.config("specific_day_scenario")
    top_n = int(context.config("compare_matching_top_chains"))
    cantons = list(context.config("compare_matching_cantons"))

    df_sampled = context.stage("synthesis.population.sampled").copy()
    df_m1 = _extract_matching_df(context.stage("synthesis.population.matching.matched_v1"))
    df_m2 = _extract_matching_df(context.stage("synthesis.population.matching.matched_v2"))

    if "person_id" not in df_m1.columns or "person_id" not in df_m2.columns:
        raise RuntimeError("Matching outputs must include 'person_id'.")

    df_pop_m1 = df_sampled.merge(df_m1[["person_id", "mz_person_id"]], on="person_id", how="left")
    df_pop_m2 = df_sampled.merge(df_m2[["person_id", "mz_person_id"]], on="person_id", how="left")

    if "person_weight" not in df_pop_m1.columns:
        df_pop_m1["person_weight"] = 1.0
    if "person_weight" not in df_pop_m2.columns:
        df_pop_m2["person_weight"] = 1.0

    df_mz_persons = context.stage("data.microcensus.persons").copy()
    df_chains = context.stage("data.microcensus.activity_chains")["person_id person_weight weekend workday day activity_chain".split()].copy()
    df_trips = context.stage("data.microcensus.trips")[0][["person_id", "trip_id", "purpose", "origin_purpose"]].copy()

    if scenario_day == "workday":
        keep_ids = set(df_mz_persons.loc[df_mz_persons["workday"], "person_id"])
    elif scenario_day == "weekend":
        keep_ids = set(df_mz_persons.loc[df_mz_persons["weekend"], "person_id"])
    elif scenario_day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        keep_ids = set(df_mz_persons.loc[df_mz_persons["day"] == scenario_day, "person_id"])
    else:
        keep_ids = set(df_mz_persons["person_id"])

    df_mz_persons = df_mz_persons[df_mz_persons["person_id"].isin(keep_ids)].copy()
    df_chains = df_chains[df_chains["person_id"].isin(keep_ids)].copy()
    df_trips = df_trips[df_trips["person_id"].isin(keep_ids)].copy()

    # Ensure men/women filters are consistent across all datasets.
    df_sampled = _normalize_sex_column(df_sampled, "sex")
    df_pop_m1 = _normalize_sex_column(df_pop_m1, "sex")
    df_pop_m2 = _normalize_sex_column(df_pop_m2, "sex")
    df_mz_persons = _normalize_sex_column(df_mz_persons, "sex")

    df_chains["chain_short"] = df_chains["activity_chain"].apply(_sanitize_chain)
    chain_lookup = df_chains[["person_id", "chain_short"]].rename(columns={"person_id": "mz_person_id"})

    trips_stats = df_trips.groupby("person_id", as_index=False).agg(
        trip_count=("trip_id", "count"),
        work_trip_count=("purpose", lambda s: np.sum((s == "work") | (s == "work_secondary")))
    )

    groups = _prepare_groups(cantons)
    summary_rows = []

    for group_name, sex, canton_id in groups:
        mz_group = _apply_group_filter(df_mz_persons, sex=sex, canton_id=canton_id)
        if len(mz_group) == 0:
            logger.info("Skipping group '%s' (no microcensus records).", group_name)
            continue

        m1_group = _apply_group_filter(df_pop_m1, sex=sex, canton_id=canton_id).copy()
        m2_group = _apply_group_filter(df_pop_m2, sex=sex, canton_id=canton_id).copy()

        m1_group = m1_group.merge(chain_lookup, on="mz_person_id", how="left")
        m2_group = m2_group.merge(chain_lookup, on="mz_person_id", how="left")
        m1_group["chain_short"] = m1_group["chain_short"].fillna("H")
        m2_group["chain_short"] = m2_group["chain_short"].fillna("H")

        mz_chain = df_chains[df_chains["person_id"].isin(set(mz_group["person_id"]))]
        dist_micro = _weighted_distribution(mz_chain, "chain_short", "person_weight")
        dist_v1 = _weighted_distribution(m1_group, "chain_short", "person_weight")
        dist_v2 = _weighted_distribution(m2_group, "chain_short", "person_weight")

        _plot_chain_comparison(out_path, group_name, dist_micro, dist_v1, dist_v2, top_n=top_n)

        metrics_micro = _compute_micro_metrics(mz_group, trips_stats)
        metrics_v1 = _compute_metrics(m1_group, trips_stats)
        metrics_v2 = _compute_metrics(m2_group, trips_stats)

        _plot_metric_errors(out_path, group_name, metrics_micro, metrics_v1, metrics_v2)

        for metric_name in metrics_micro.keys():
            summary_rows.append({
                "group": group_name,
                "metric": metric_name,
                "microcensus": metrics_micro[metric_name],
                "matching_v1": metrics_v1[metric_name],
                "matching_v2": metrics_v2[metric_name],
                "abs_err_v1": abs(metrics_v1[metric_name] - metrics_micro[metric_name]),
                "abs_err_v2": abs(metrics_v2[metric_name] - metrics_micro[metric_name]),
                "winner": "v2" if abs(metrics_v2[metric_name] - metrics_micro[metric_name]) < abs(metrics_v1[metric_name] - metrics_micro[metric_name]) else "v1",
            })

    df_summary = pd.DataFrame(summary_rows)
    if len(df_summary) > 0:
        df_summary.to_csv(os.path.join(out_path, "matching_metrics_summary.csv"), index=False)

        winners = df_summary.groupby("winner").size().rename("count").reset_index()
        fig, ax = plt.subplots(figsize=(6, 4), dpi=220)
        ax.bar(winners["winner"], winners["count"], color=["#7f8c8d" if w == "v1" else "#c0392b" for w in winners["winner"]])
        ax.set_title("Metric wins across groups")
        ax.set_ylabel("Number of metrics won")
        fig.tight_layout()
        fig.savefig(os.path.join(out_path, "matching_global_winner_counts.png"))
        plt.close(fig)

    logger.info("Matching comparison done. Outputs written to: %s", out_path)
    return df_summary