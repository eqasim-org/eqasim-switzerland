"""Summarize and rebuild the shared-model experiment using saved evidence only."""

import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import predictive_report
from .predictive_figures import ALGORITHMS, COLORS, PURPOSES, LABELS

OUTCOMES = ("distance_km", "travel_minutes", "trip_count", *("count_" + p for p, _ in PURPOSES))
DEFINITIONS = {
    "Energy score": "For each person, take the mean Euclidean distance between a predicted daily-behaviour vector and their observed vector, then subtract half the mean distance between pairs of predictions. Components are scaled using donor-group variances. Lower is better. It measures the joint predictive distribution, not just the mean day.",
    "Variogram score": "For selected pairs of daily outcomes, compare their observed square-root absolute difference with its mean across predicted days, then square and average these errors. Lower is better. This complements the energy score by checking dependence between outcomes.",
    "CRPS": "Continuous ranked probability score: mean absolute error across predicted values minus half the mean absolute difference between pairs of predictions. It is measured in the outcome's units, such as km/day or shopping visits/day. Lower is better. All methods use equally sized ensembles.",
    "Chain Brier": "For each complete ordered purpose-and-mode chain, compare its predicted probability (the fraction of donor draws with that chain) with its observed indicator, square the error and sum over chains. Lower is better. A wrong, overly confident prediction is penalized strongly.",
    "Event Brier": "Square the difference between the predicted probability of an event and its observed zero/one indicator; then average over people. Events include any work, education, travel, car driving or public transport. Lower is better.",
    "Calibration": "Group respondents by predicted probability and compare the group-average prediction with its observed event frequency. Closer to the diagonal is better. A model may be calibrated while failing to distinguish different people, so predictive accuracy is also needed.",
    "Coverage": "Matching coverage is the proportion receiving a valid donor in every requested draw. Higher is better, but it does not measure accuracy. Prediction-interval coverage instead counts observed values inside an ensemble interval; the two concepts are different.",
    "CRPS / energy ensembles": "Scores evaluate the empirical distribution of all donor draws using the V-statistic convention, including identical pairs. Fifty draws are used per Microcensus person. Equal draw counts permit comparisons between algorithms; this does not make them independent models.",
    "Daily distance": "Sum of straight-line distances of all trips in a person's observed or donated diary, including return-home trips. This is not routed distance. For Microcensus the person's observed day is the benchmark. For synthetic residents the survey is only a descriptive reference, not known ground truth.",
    "Activity counts": "Count trip arrivals with a given non-home purpose, including repeated visits. A person with no shopping visits contributes zero. Means include all eligible people, rather than only those doing the activity. Closer to the observed benchmark is better for aggregate agreement, but not proof of individual accuracy.",
    "Activity chains": "A complete ordered sequence of activity purposes in a diary. For population charts, the survey's fifteen most frequent chains are shown as shares of all person-days, retaining other chains in the denominator. This descriptive purpose-chain plot differs from the Microcensus chain Brier score, whose labels also include travel modes.",
    "Uncertainty": "Microcensus error bars resample households, holding the encoder and group split fixed. Population error bars show the minimum and maximum across ten matching repetitions. They describe matching randomness, not demographic, survey, model-fitting or forecasting uncertainty.",
    "Computation": "Wall-clock time for the actual training or matching calls in the same job. The shared encoder is fitted once. Matching times include algorithm diagnostics. Lower is faster, but hardware, caching and prediction quality must also be considered.",
    "Donor-restricted reconstruction": "The encoder is learned once with 90% of eligible survey respondents; 10% monitor validation loss and choose the checkpoint. Three separate groups restrict donor availability only. A recipient can never select their own household's diary, but their behaviour may have contributed to training. This is reconstruction under donor exclusion, not fully held-out model validation.",
}


def _groups(frame, manifest, survey=False):
    density = frame.population_density.to_numpy() / manifest["density_area_km2"]
    yield "national", "Switzerland", np.ones(len(frame), dtype=bool)
    for code, label in ((25, "Geneva"), (1, "Zürich"), (22, "Vaud")):
        yield f"canton_{code}", label, frame.canton_id.eq(code).to_numpy()
    bounds = manifest["density_bounds"]
    edges = [-np.inf, *bounds, np.inf]
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        label = f"{max(low, 0):g}–{high:g} residents/km²" if np.isfinite(high) else f"≥{low:g} residents/km²"
        yield f"density_{i}", label, np.isfinite(density) & (density >= max(low, 0)) & (density < high)


def summarize_population(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    pop = root / "population"
    recipients = pd.read_csv(pop / "recipients.csv")
    source = pd.read_csv(root / "survey.csv")
    outcomes = pd.read_csv(root / "survey_outcomes.csv").set_index("mz_id").loc[source.mz_id]
    lookup = pd.Index(source.mz_id)
    common = np.ones(len(recipients), dtype=bool)
    coverage = []
    for algorithm in ALGORITHMS:
        for draw in range(manifest["population_repetitions"]):
            with np.load(pop / algorithm / f"draw_{draw}" / "prediction.npz") as data:
                rows = data["rows"]
                if len(rows) != len(recipients):
                    raise ValueError("Saved population mapping changed recipient alignment")
                valid = rows >= 0
                common &= valid
                coverage.append(dict(algorithm=algorithm, repeat=draw, matched=int(valid.sum()), eligible=len(rows)))
    pd.DataFrame(coverage).to_csv(pop / "coverage.csv", index=False)
    recipients["common"] = common
    recipients.to_csv(pop / "common_cohort.csv", index=False)
    recipients = recipients.loc[common].reset_index(drop=True)
    masks = list(_groups(recipients, manifest))
    values = outcomes[list(OUTCOMES)].to_numpy()
    chain_codes, chain_names = pd.factorize(source.activity_chain)
    survey_weights = source.person_weight.to_numpy()
    means, histograms, chains = [], [], []
    distance_edges = np.r_[0, .1, np.geomspace(.2, 500, 35), np.inf]

    def add(dataset, repeat, indices, weights, groups):
        for group, label, mask in groups:
            if not mask.any():
                continue
            selected, mass = indices[mask], weights[mask]
            for col, outcome in enumerate(OUTCOMES):
                v = values[selected, col]
                means.append(dict(group=group, label=label, algorithm=dataset, repeat=repeat,
                    outcome=outcome, mean=np.average(v, weights=mass), people=int(mask.sum())))
                if outcome.startswith("count_"):
                    histogram = np.bincount(np.minimum(v.astype(int), 5), weights=mass, minlength=6)
                    bins = ["0", "1", "2", "3", "4", "5+"]
                elif outcome == "distance_km":
                    histogram = np.histogram(v, bins=distance_edges, weights=mass)[0]
                    bins = [str(x) for x in distance_edges[1:]]
                else:
                    continue
                for bin_, total in zip(bins, histogram):
                    histograms.append(dict(group=group, label=label, algorithm=dataset, repeat=repeat,
                        outcome=outcome, bin=bin_, percent=100 * total / mass.sum()))
            chain_mass = np.bincount(chain_codes[selected], weights=mass, minlength=len(chain_names))
            for chain, weight in zip(chain_names, chain_mass):
                chains.append(dict(group=group, label=label, algorithm=dataset, repeat=repeat,
                                   chain=chain, percent=100 * weight / mass.sum()))
    add("Survey reference", -1, np.arange(len(source)), survey_weights, list(_groups(source, manifest)))
    for algorithm in ALGORITHMS:
        for draw in range(manifest["population_repetitions"]):
            with np.load(pop / algorithm / f"draw_{draw}" / "prediction.npz") as data:
                donor_ids = data["rows"][common]
            indices = lookup.get_indexer(donor_ids)
            if (indices < 0).any():
                raise ValueError("Population selected donors absent from saved survey outcomes")
            add(algorithm, draw, indices, np.ones(len(indices)), masks)
    pd.DataFrame(means).to_csv(pop / "daily_means.csv", index=False)
    pd.DataFrame(histograms).to_csv(pop / "distributions.csv", index=False)
    pd.DataFrame(chains).to_csv(pop / "activity_chains.csv", index=False)


def _metric_dialog():
    buttons = " ".join(f'<button type="button" data-metric="{html.escape(k, quote=True)}">{html.escape(k)}</button>' for k in DEFINITIONS)
    payload = json.dumps(DEFINITIONS).replace("<", "\\u003c")
    return ('<aside><strong>Click a metric for its definition and calculation:</strong><p>' + buttons + '</p></aside>'
        '<dialog id="metric-dialog"><h3 id="metric-title"></h3><p id="metric-text"></p><button id="metric-close">Close</button></dialog>'
        '<script>const metricDefinitions=' + payload + ';const metricDialog=document.getElementById("metric-dialog");'
        'document.addEventListener("click",e=>{const b=e.target.closest("[data-metric]");if(!b)return;'
        'document.getElementById("metric-title").textContent=b.dataset.metric;'
        'document.getElementById("metric-text").textContent=metricDefinitions[b.dataset.metric];metricDialog.showModal();});'
        'document.getElementById("metric-close").onclick=()=>metricDialog.close();'
        'metricDialog.addEventListener("click",e=>{if(e.target===metricDialog)metricDialog.close();});</script>')


def render(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    micro = root / "microcensus"
    predictive_report.render(micro)
    micro_html = (micro / "report.html").read_text().split("</style>", 1)[-1]
    micro_html = re.sub(r'''(src|href)=(["'])(?!\#|https?:)([^"']+)\2''', r'\1=\2microcensus/\3\2', micro_html)
    sections = []

    def figure(fig, name, title, description):
        fig.tight_layout()
        fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        metric = ("Activity chains" if "chains" in name else "Coverage" if "coverage" in name else
                  "Activity counts" if "activities" in name else "Daily distance" if "distance" in name or "totals" in name
                  else "Donor-restricted reconstruction")
        sections.append(f'<h3>{title} <button type="button" data-metric="{metric}">ⓘ Definition</button></h3>'
                        f'<p>{description}</p><figure><img src="{name}.png" alt="{title}"></figure>')

    history = pd.DataFrame(manifest["training_report"]["history"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.epoch, history.train_loss, label="Training objective")
    ax.plot(history.epoch, history.validation_loss, label="Validation behaviour loss")
    ax.axvline(manifest["training_report"]["best_epoch"], ls="--", color="black", label="Retained checkpoint")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend()
    figure(fig, "shared_training", "One fitted encoder for the whole experiment",
        "The encoder uses a 90% fitting / 10% early-stopping validation split. Monitoring starts at epoch 10; "
        "patience stops training after validation improvement stalls. Lower validation loss is better. "
        "Training and validation use different objective terms, so their gap is not a direct overfitting measure. "
        "The dashed line identifies the checkpoint used for every matching repetition.")
    training_html = "".join(sections)
    sections = []
    scores = pd.read_csv(micro / "out_of_fold_scores.csv").merge(pd.read_csv(micro / "respondents.csv"), on="mz_id", validate="many_to_one")
    common = scores.groupby("mz_id").covered.min()
    scores = scores.loc[scores.mz_id.isin(common.index[common.eq(1)])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ("energy", "chain_brier", "crps_distance_km")):
        for index, algorithm in enumerate(ALGORITHMS):
            values = []
            for role in ("fit", "early_stopping_validation"):
                rows = scores.loc[scores.algorithm.eq(algorithm) & scores.training_role.eq(role)]
                values.append(np.average(rows[metric], weights=rows.survey_weight))
            ax.bar(np.arange(2) + (index - 1) * .23, values, .23, color=COLORS[algorithm], label=algorithm)
        ax.set(xticks=[0, 1], xticklabels=["Used for fitting", "Monitored validation"], title=LABELS[metric])
        ax.legend()
    figure(fig, "training_role_scores", "Separate reconstruction results by training role",
        "Lower is better within each group. The fitting group contributed to weight updates; the 10% validation "
        "group did not, but chose the stopping checkpoint. It is not an untouched final test set. Differences "
        "between groups also reflect their composition, so this is a diagnostic rather than a causal overfitting test.")
    role_html = "".join(sections)
    sections = []
    pop = root / "population"
    means = pd.read_csv(pop / "daily_means.csv")
    distributions = pd.read_csv(pop / "distributions.csv")
    chains = pd.read_csv(pop / "activity_chains.csv")
    coverage = pd.read_csv(pop / "coverage.csv")
    common_n = int(pd.read_csv(pop / "common_cohort.csv").common.sum())
    pop_intro = (f'<h2 id="population">2. Matching the synthetic population</h2><p>Each method matches the entire configured '
        f'synthetic population {manifest["population_repetitions"]} times. All figures use the same {common_n:,} '
        'residents matched in every repetition by every method. Explicit home-only scenario bypasses are excluded; '
        'genuine non-travel donor diaries remain. Black is a survey reference, not a known correct population total. '
        'Closer to black means aggregate agreement only; local or demographic differences can be appropriate.</p>')
    fig, ax = plt.subplots(figsize=(9, 3))
    for i, algorithm in enumerate(ALGORITHMS):
        rows = coverage.loc[coverage.algorithm.eq(algorithm)]
        rates = rows.matched / rows.eligible * 100
        ax.barh(i, rates.mean(), color=COLORS[algorithm])
        ax.text(2, i, f"{algorithm}: {rates.min():.3f}–{rates.max():.3f}%", va="center", color="white")
    ax.set(yticks=[], xlim=(0, 100), xlabel="Matched eligible residents (%)")
    figure(fig, "population_coverage", "Population matching coverage",
        "Higher coverage is better. Labels show the minimum and maximum over repetitions. Excluding unmatched "
        "residents from behaviour plots keeps their comparison consistent but can hide unsupported groups; this "
        "coverage figure makes the exclusion visible.")
    for group in ("national", "canton_25", "canton_1", "canton_22"):
        data = means.loc[means.group.eq(group)]
        if data.empty:
            continue
        label = data.label.iloc[0]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, outcome, title in zip(axes, OUTCOMES[:3], ("Distance (km/day)", "Travel time (min/day)", "Trips/day")):
            for index, algorithm in enumerate(("Survey reference", *ALGORITHMS)):
                values = data.loc[data.algorithm.eq(algorithm) & data.outcome.eq(outcome), "mean"]
                if values.empty:
                    continue
                mean = values.mean()
                color = "black" if algorithm == "Survey reference" else COLORS[algorithm]
                ax.bar(index, mean, color=color)
                ax.errorbar(index, mean, yerr=np.maximum(0, [[mean - values.min()], [values.max() - mean]]), color="black", capsize=4)
                ax.text(index, values.max(), f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
            ax.set(xticks=range(4), xticklabels=["Survey", *ALGORITHMS], title=title)
            ax.margins(y=.2)
        figure(fig, f"population_totals_{group}", f"{label}: daily distances, travel times and trips",
            "Bars show the mean across matching repetitions; whiskers show their minimum-to-maximum range. "
            "Closer to the survey bar indicates descriptive agreement, not proven population accuracy. Straight-line "
            "daily distances include return-home legs. Ranges measure matching randomness only.")
        fig, ax = plt.subplots(figsize=(12, 4))
        for index, algorithm in enumerate(("Survey reference", *ALGORITHMS)):
            values, lows, highs = [], [], []
            for purpose, _ in PURPOSES:
                rows = data.loc[data.algorithm.eq(algorithm) & data.outcome.eq("count_" + purpose), "mean"]
                values.append(rows.mean()); lows.append(rows.min()); highs.append(rows.max())
            color = "black" if algorithm == "Survey reference" else COLORS[algorithm]
            x = np.arange(5) + (index - 1.5) * .19
            bars = ax.bar(x, values, .19, color=color, label=algorithm)
            ax.errorbar(x, values, yerr=np.maximum(0, [np.array(values) - lows, np.array(highs) - values]), fmt="none", color="black", capsize=3)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set(xticks=range(5), xticklabels=[name for _, name in PURPOSES], ylabel="Visits per eligible person-day")
        ax.margins(y=.25); ax.legend(ncols=4)
        figure(fig, f"population_activities_{group}", f"{label}: shopping, work and other visits",
            "The average includes residents with zero visits; repeated visits count separately. Coloured bars average "
            "ten matching repetitions; whiskers show their range. The survey is a descriptive reference, so there "
            "is no unconditional higher-is-better or lower-is-better direction.")
        fig, ax = plt.subplots(figsize=(10, 4))
        for algorithm in ("Survey reference", *ALGORITHMS):
            rows = distributions.loc[distributions.group.eq(group) & distributions.algorithm.eq(algorithm) & distributions.outcome.eq("distance_km")].copy()
            rows["upper"] = rows.bin.astype(float)
            curve = rows.groupby("upper").percent.mean().sort_index().cumsum() / 100
            curve = curve.loc[np.isfinite(curve.index)]
            ax.step(curve.index, curve, where="post", label=algorithm,
                    color="black" if algorithm == "Survey reference" else COLORS[algorithm])
        ax.set_xscale("symlog", linthresh=1)
        ax.set(xlabel="Daily straight-line distance (km)", ylabel="Fraction at or below this distance", ylim=(0, 1.02))
        ax.legend(); ax.grid(alpha=.2)
        figure(fig, f"population_distance_distribution_{group}", f"{label}: distribution of daily distances",
            "Curves average daily-distance histogram fractions over repetitions and retain zero-distance days. "
            "The horizontal scale is compressed above 1 km. Closer to black indicates survey-distribution agreement, "
            "which does not establish that the synthetic population should have that exact distribution.")
        survey_chains = chains.loc[chains.group.eq(group) & chains.algorithm.eq("Survey reference")].nlargest(15, "percent")
        top = survey_chains.chain.tolist()
        if top:
            fig, ax = plt.subplots(figsize=(13, 6))
            for index, algorithm in enumerate(("Survey reference", *ALGORITHMS)):
                values = chains.loc[chains.group.eq(group) & chains.algorithm.eq(algorithm)].groupby("chain").percent.mean().reindex(top)
                ax.barh(np.arange(len(top)) + (index - 1.5) * .19, values, .19, label=algorithm,
                    color="black" if algorithm == "Survey reference" else COLORS[algorithm])
            ax.set(yticks=range(len(top)), yticklabels=top, xlabel="Person-days (%)")
            ax.invert_yaxis(); ax.legend(ncols=4)
            figure(fig, f"population_chains_{group}", f"{label}: fifteen most frequent survey activity chains",
                "The survey's top fifteen chains define the displayed categories; other chains remain in the "
                "denominator. Coloured shares are averaged over ten repetitions. Similar aggregate shares can "
                "coexist with different individual matches. No aggregate agreement alone establishes a winner.")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (purpose, label) in zip(axes.flat, PURPOSES):
        for index, algorithm in enumerate(("Survey reference", *ALGORITHMS)):
            rows = distributions.loc[distributions.group.eq("national") & distributions.algorithm.eq(algorithm) &
                                     distributions.outcome.eq("count_" + purpose)]
            values = rows.groupby("bin").percent.mean().reindex(["0", "1", "2", "3", "4", "5+"])
            ax.bar(np.arange(6) + (index - 1.5) * .19, values, .19, label=algorithm,
                   color="black" if algorithm == "Survey reference" else COLORS[algorithm])
        ax.set(title=label, xticks=range(6), xticklabels=["0", "1", "2", "3", "4", "5+"],
               xlabel="Visits per resident-day", ylabel="Resident-days (%)")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    figure(fig, "population_activities_distribution", "Switzerland: zero, one or repeated visits",
        "These shares show whether similar mean activity counts conceal different numbers of people making "
        "zero, one or repeated visits. Predicted shares average matching repetitions; black is the survey "
        "reference. Agreement with that reference is descriptive, not a population accuracy score.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    density = means.loc[means.group.str.startswith("density_")]
    group_order = list(density.group.drop_duplicates())
    for ax, outcome, title in zip(axes.flat, ("distance_km", "trip_count", "count_work", "count_shop"),
                                  ("Daily distance (km)", "Trips/day", "Work visits/day", "Shopping visits/day")):
        for algorithm in ("Survey reference", *ALGORITHMS):
            rows = density.loc[density.algorithm.eq(algorithm) & density.outcome.eq(outcome)]
            stats = rows.groupby("group")["mean"].agg(["mean", "min", "max"]).reindex(group_order)
            ax.errorbar(range(len(group_order)), stats["mean"], yerr=np.maximum(0,
                [stats["mean"] - stats["min"], stats["max"] - stats["mean"]]), fmt="o-", capsize=3,
                label=algorithm, color="black" if algorithm == "Survey reference" else COLORS[algorithm])
        labels = [density.loc[density.group.eq(g), "label"].iloc[0].replace(" residents/km²", "") for g in group_order]
        ax.set(title=title, xticks=range(len(labels)), xticklabels=labels, xlabel="Residence density (residents/km²)")
        ax.tick_params(axis="x", rotation=20); ax.legend(fontsize=8); ax.grid(alpha=.2)
    figure(fig, "population_density_totals", "How do the matched days vary with residence density?",
        "Residents are grouped by their own home density, while the survey reference uses respondents' own "
        "density. Points average ten population matching repetitions and whiskers show their range. Trends are "
        "associations, not causal density effects. Neither higher nor lower values are intrinsically better.")
    population_html = "".join(sections)
    style = ('<style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:40px auto;padding:20px}img{max-width:100%}'
        'figure{margin:20px 0}figcaption{font-size:14px;color:#555}details{overflow:auto}section{margin:35px 0}'
        'table{border-collapse:collapse;font-size:13px}td,th{padding:5px;border:1px solid #ddd}'
        'button{margin:3px;padding:6px 12px;cursor:pointer}dialog{max-width:650px;border:1px solid #888;border-radius:8px}'
        'dialog::backdrop{background:#0007}aside,.note{background:#eef4f7;padding:15px}nav{position:sticky;top:0;background:white;padding:10px;z-index:5}</style>')
    document = ('<!doctype html><meta charset="utf-8"><title>Repeated matching experiments</title>' + style +
        '<h1>Three matching algorithms: one shared V2 model</h1><nav><a href="#microcensus">1. Microcensus experiment</a> · '
        '<a href="#population">2. Synthetic population</a></nav>' + _metric_dialog() +
        '<h2 id="microcensus">1. Microcensus donor-restricted reconstruction</h2>' + training_html + role_html + micro_html +
        pop_intro + population_html + '<p><a href="population/daily_means.csv">Population means</a> · '
        '<a href="population/distributions.csv">Population distributions</a> · <a href="population/activity_chains.csv">Chain shares</a> · '
        '<a href="manifest.json">Experiment settings</a></p>')
    (root / "report.html").write_text(document)
    return str(root / "report.html")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    render(args.directory)
