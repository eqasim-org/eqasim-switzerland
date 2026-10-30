"""Vector figures and paired, clustered summaries of out-of-fold predictions."""

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import clustered_interval
from .explanations import FIGURES

SCORES = ("energy", "variogram", "chain_brier", "mask_violation_employed", "mask_violation_is_student",
          "brier_mobile", "brier_work", "brier_education",
          "brier_car", "brier_pt", "crps_trip_count", "crps_travel_minutes", "crps_distance_km",
          "crps_first_departure_hour", "crps_minutes_work", "crps_minutes_education")


def save_figure(fig, output, name):
    fig.tight_layout()
    for extension in ("pdf", "svg"):
        fig.savefig(Path(output) / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def paired_summary(scores, respondents, bootstrap, seed):
    # Collapse model repeats before uncertainty estimation to avoid pseudoreplication.
    respondents = respondents.assign(respondent_weight=1.)
    available = [col for col in SCORES if col in scores]
    average = scores.groupby(["mz_id", "algorithm"], as_index=False)[available + ["covered"]].mean()
    average = average.merge(respondents, on="mz_id", validate="many_to_one")
    rng, rows, differences = np.random.RandomState(seed), [], []
    for weighting in ("respondent_weight", "survey_weight", "target_weight"):
        for algorithm, group in average.groupby("algorithm"):
            for metric in available + ["covered"]:
                result = clustered_interval(group[metric], group[weighting], group["household_group"], rng, bootstrap)
                rows.append(dict(algorithm=algorithm, metric=metric, weighting=weighting,
                                 estimate=result[0], low=result[1], high=result[2], households=result[3], effective_n=result[4]))
        for metric in available:
            pivot = average.pivot(index="mz_id", columns="algorithm", values=metric)
            metadata = respondents.set_index("mz_id").reindex(pivot.index)
            if "V2" not in pivot:
                continue
            for reference in ("V1", "raw_knn", "stratified_random"):
                if reference not in pivot:
                    continue
                delta = pivot["V2"] - pivot[reference]
                result = clustered_interval(delta, metadata[weighting], metadata["household_group"], rng, bootstrap)
                relative = clustered_interval(delta, metadata[weighting], metadata["household_group"], rng, bootstrap,
                                               denominator=pivot[reference])
                differences.append(dict(reference=reference, metric=metric, weighting=weighting,
                                        delta=result[0], low=result[1], high=result[2],
                                        relative_percent=100*relative[0], relative_low=100*relative[1],
                                        relative_high=100*relative[2],
                                        households=result[3], effective_n=result[4]))
    return pd.DataFrame(rows), pd.DataFrame(differences), average


def conditional_summary(average, minimum=30):
    rows = []
    for feature in ("age", "income_class", "N_children_under_12", "household_size", "population_density",
                    "employed", "is_student", "sex", "sp_region", "canton_id"):
        if feature not in average:
            continue
        values = average[feature]
        categories = (pd.qcut(values, 5, duplicates="drop").astype("string")
                      if feature in ("age", "population_density") and values.nunique() > 5
                      else values.astype("string"))
        for (algorithm, category), group in average.assign(category=categories).groupby(["algorithm", "category"]):
            if len(group) < minimum:
                continue
            for metric in ("energy", "brier_mobile", "brier_education", "brier_work", "crps_trip_count"):
                if metric in group:
                    valid = group[metric].notna()
                    rows.append(dict(feature=feature, category=str(category), algorithm=algorithm,
                        metric=metric, n=len(group), coverage=float(group["covered"].mean()),
                        survey_risk=float(np.average(group.loc[valid, metric], weights=group.loc[valid, "survey_weight"])) if valid.any() else np.nan,
                        target_risk=float(np.average(group.loc[valid, metric], weights=group.loc[valid, "target_weight"])) if valid.any() else np.nan))
    return pd.DataFrame(rows)


def calibration_tables(scores, respondents, draws, seed, bootstrap=500):
    probability_cols = [col for col in scores if col.startswith(("prob_", "truth_"))]
    average = scores.groupby(["mz_id", "algorithm"], as_index=False)[probability_cols].mean()
    average = average.merge(respondents, on="mz_id", validate="many_to_one")
    reliability, rng = [], np.random.RandomState(seed)
    for event in ("mobile", "education", "work", "car", "pt", "leisure"):
        if "prob_" + event not in average:
            continue
        bins = pd.cut(average["prob_" + event], np.linspace(0, 1, 6), include_lowest=True)
        for (algorithm, interval), group in average.assign(bin=bins).groupby(["algorithm", "bin"], observed=True):
            valid = group["prob_" + event].notna()
            group = group.loc[valid]
            if not len(group):
                continue
            weights = group["survey_weight"].to_numpy()
            observed, low, high, _, ess = clustered_interval(
                group["truth_" + event], weights, group["household_group"], rng, bootstrap)
            reliability.append(dict(algorithm=algorithm, event=event, bin=str(interval), n=len(group),
                                    predicted=np.average(group["prob_" + event], weights=weights),
                                    observed=observed, low=low, high=high, effective_n=ess))
    rng = np.random.RandomState(seed)
    ranks = scores[["mz_id", "algorithm", "repeat", "fold"]].copy()
    for metric in ("trip_count", "travel_minutes", "distance_km", "first_departure_hour"):
        if "less_" + metric in scores:
            ranks[metric] = (scores["less_" + metric] + rng.rand(len(scores)) * (scores["equal_" + metric] + 1)) / (draws + 1)
    ranks = ranks.merge(respondents[["mz_id", "survey_weight"]], on="mz_id", validate="many_to_one")
    return pd.DataFrame(reliability), ranks


def comparison_figures(output, differences, reliability, ranks, average):
    panels = differences.loc[(differences["weighting"] == "survey_weight") &
                              differences["metric"].isin(["energy", "variogram", "chain_brier", "brier_mobile", "brier_education"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    labels = {"energy": "Joint behaviour (energy)", "variogram": "Dependence (variogram)",
              "chain_brier": "Ordered diary (Brier)", "brier_mobile": "Travel participation (Brier)",
              "brier_education": "Education participation (Brier)"}
    for axis, reference in zip(axes, ("V1", "raw_knn", "stratified_random")):
        group = panels.loc[panels["reference"] == reference]
        y = np.arange(len(group))
        axis.errorbar(group["relative_percent"], y,
                      xerr=np.maximum(np.vstack([group["relative_percent"]-group["relative_low"],
                                                  group["relative_high"]-group["relative_percent"]]), 0),
                      fmt="o", color="#0072B2", capsize=3)
        axis.axvline(0, color="black", linewidth=.8)
        axis.set_yticks(y, [labels.get(metric, metric) for metric in group["metric"]])
        axis.set_title("V2 minus " + reference.replace("_", " "))
        axis.set_xlabel("Difference (% of reference score); lower favours V2")
    save_figure(fig, output, "paired_predictive_skill")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for axis, event in zip(axes.ravel(), ("mobile", "education", "work", "car", "pt", "leisure")):
        for algorithm, group in reliability.loc[reliability["event"] == event].groupby("algorithm"):
            axis.errorbar(group["predicted"], group["observed"],
                          yerr=np.maximum(np.vstack([group["observed"]-group["low"], group["high"]-group["observed"]]), 0),
                          fmt="o-", label=algorithm, markersize=4, capsize=2)
        axis.plot([0, 1], [0, 1], ":", color="grey")
        axis.set(title=event, xlabel="Predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    axes.ravel()[0].legend(fontsize=7)
    save_figure(fig, output, "event_reliability")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for axis, metric in zip(axes, ("trip_count", "travel_minutes")):
        if metric in ranks:
            for algorithm, group in ranks.groupby("algorithm"):
                axis.hist(group[metric].dropna(), bins=np.linspace(0, 1, 11), density=True,
                          weights=group.loc[group[metric].notna(), "survey_weight"], histtype="step", label=algorithm)
        axis.axhline(1, linestyle=":", color="grey")
        axis.set(title=metric, xlabel="Randomized ensemble rank", ylabel="Density")
    axes[0].legend(fontsize=7)
    save_figure(fig, output, "rank_calibration")
    # Age is continuous here; the smoothing grid is a reporting device, not a matching mask.
    pivot = average.pivot(index="mz_id", columns="algorithm", values="energy")
    if {"V1", "V2"}.issubset(pivot.columns):
        meta = average.drop_duplicates("mz_id").set_index("mz_id").reindex(pivot.index)
        delta = (pivot["V2"] - pivot["V1"]).to_numpy()
        ages = meta["age"].to_numpy()
        grid = np.linspace(np.min(ages), np.max(ages), 50)
        estimates = []
        for center in grid:
            weight = np.exp(-.5 * ((ages - center) / 4.)**2) * meta["survey_weight"].to_numpy()
            valid = np.isfinite(delta)
            estimates.append(np.average(delta[valid], weights=weight[valid]))
        fig, axis = plt.subplots(figsize=(7, 3.5))
        axis.plot(grid, estimates, color="#0072B2")
        axis.axhline(0, color="black", linewidth=.8)
        axis.set(xlabel="Age (years; Gaussian smoothing bandwidth 4)", ylabel="V2 minus V1 energy score")
        save_figure(fig, output, "continuous_age_skill")


def write_html(output, title, text, tables, figures, finding=None):
    body = ["<!doctype html><meta charset='utf-8'><style>body{font:16px system-ui;margin:3rem;max-width:1200px;line-height:1.55}table{border-collapse:collapse;font-size:13px}td,th{padding:.4rem;border:1px solid #ddd}img{max-width:100%}section{margin:3rem 0}details{overflow:auto;margin:1.5rem 0}summary{cursor:pointer;font-weight:600}.finding{background:#eef5fa;padding:1rem}</style>",
            "<h1>" + html.escape(title) + "</h1>", "<p>" + html.escape(text) + "</p>"]
    if finding:
        body.append("<p class='finding'>" + html.escape(finding) + "</p>")
    for name in figures:
        title, explanation, reading = FIGURES.get(name, (name.replace("_", " "), "", ""))
        escaped = html.escape(name, quote=True)
        body.extend(["<section><h2>" + html.escape(title) + "</h2>",
                     "<p>" + html.escape(explanation) + "</p>",
                     "<p><strong>How to read this:</strong> " + html.escape(reading) + "</p>",
                     f"<figure><img src='{escaped}.svg' alt='{html.escape(title, quote=True)}'>"
                     f"<figcaption><a href='{escaped}.pdf'>Download PDF</a> · <a href='{escaped}.svg'>Download SVG</a>"
                     "</figcaption></figure></section>"])
    body.append("<h2>Detailed tables and audit information</h2><p>Expand a table to inspect values, sample sizes and uncertainty.</p>")
    for name, table in tables:
        body.extend(["<details><summary>" + html.escape(name) + "</summary>",
                     table.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4g}"), "</details>"])
    (Path(output) / "report.html").write_text("\n".join(body))
