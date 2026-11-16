"""Figures from saved held-out scores and donor draws; no matching is executed."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALGORITHMS = ("V1", "V2", "V3")
COLORS = {"Observed": "#222222", "V1": "#df8a2e", "V2": "#269568", "V3": "#427fc0"}
LABELS = {"energy": "Whole-day energy score", "variogram": "Dependence (variogram) score",
    "chain_brier": "Complete-chain Brier score", "crps_distance_km": "Distance CRPS (km/day)",
    "crps_travel_minutes": "Travel-time CRPS (min/day)", "crps_trip_count": "Trip-count CRPS",
    **{f"crps_count_{p}": f"{name} count CRPS" for p, name in (
        ("work", "Work"), ("education", "Education"), ("shop", "Shopping"), ("leisure", "Leisure"), ("other", "Other"))},
    **{f"brier_{p}": f"{p.title()} event Brier score" for p in ("work", "education", "mobile", "car", "pt")}}
PURPOSES = (("work", "Work"), ("education", "Education"), ("shop", "Shopping"),
            ("leisure", "Leisure"), ("other", "Other"))


def build(output, scores, primary, summary, paired, manifest):
    """Return illustrated HTML sections, with the same shared cohort throughout."""
    sections = []

    def save(fig, name, title, description, direction):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=190, bbox_inches="tight")
        plt.close(fig)
        metric = ("CRPS" if name == "activity_count_scores" else "Event Brier" if name == "event_scores" else
                  "Computation" if name == "runtime_comparison" else
                  "Coverage" if "coverage" in name else "Calibration" if "calibration" in name else
                  "Activity counts" if "activity" in name else "Daily distance" if "daily" in name else
                  "Uncertainty" if "paired" in name or "runtime" in name else "Energy score")
        definition = f' <button type="button" data-metric="{metric}">ⓘ Definition</button>' if manifest.get("shared_model") else ""
        if manifest.get("shared_model"):
            title = title.replace("an unseen day", "another survey diary")
        sections.append(f'<section><h2>{title}{definition}</h2><p>{description} <strong>{direction}</strong></p>'
                        f'<figure><a href="{name}.png"><img src="{name}.png" alt="{title}" loading="lazy"></a>'
                        f'<figcaption>{description} {direction}</figcaption></figure></section>')

    def score_panels(metrics, name, title, description):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, metric in zip(axes.flat, metrics):
            rows = summary.loc[summary.weighting.eq("survey") & summary.metric.eq(metric)].set_index("algorithm").reindex(ALGORITHMS)
            best = rows["mean"].idxmin()
            for x, algorithm in enumerate(ALGORITHMS):
                row = rows.loc[algorithm]
                ax.errorbar(x, row["mean"], yerr=np.maximum(0, [[row["mean"] - row.lower],
                    [row.upper - row["mean"]]]), fmt="o", color=COLORS[algorithm], capsize=5, markersize=8)
                ax.annotate(f'{row["mean"]:.3g}', (x, row.upper), xytext=(0, 9), textcoords="offset points", ha="center")
            ax.scatter(ALGORITHMS.index(best), rows.loc[best, "mean"], s=230, marker="*",
                       facecolors="none", edgecolors=COLORS[best], linewidths=1.2)
            ax.set(xticks=range(3), xticklabels=ALGORITHMS, title=LABELS.get(metric, metric), xlim=(-.6, 2.6))
            ax.margins(y=.3)
            ax.grid(axis="y", alpha=.2)
        for ax in list(axes.flat)[len(metrics):]:
            ax.set_visible(False)
        save(fig, name, title, description,
             "Lower is better. A star marks the lowest estimate; whiskers are 95% household bootstrap intervals, not proof of superiority.")

    coverage = scores.groupby("algorithm").covered.mean().reindex(ALGORITHMS) * 100
    fig, ax = plt.subplots(figsize=(9, 3))
    bars = ax.barh(ALGORITHMS, coverage, color=[COLORS[a] for a in ALGORITHMS])
    ax.bar_label(bars, labels=[f"{x:.4f}%" for x in coverage], padding=-85, color="white")
    ax.set(xlim=(0, 100), xlabel="Respondents matched in every draw (%)")
    save(fig, "matching_coverage", "1. How many people can each method match?",
         f"All accuracy and behaviour figures below use the same {primary.mz_id.nunique():,} respondents with valid predictions from all methods.",
         "Higher coverage is better; it does not measure diary accuracy.")
    score_panels(["energy", "variogram", "chain_brier", "crps_distance_km", "crps_travel_minutes", "crps_trip_count"],
        "whole_day_scores", "2. How well does each method predict an unseen day?",
        "Energy checks the whole day jointly; the variogram checks selected relationships between daily totals. Chain Brier checks the full ordered purpose-and-mode chain. CRPS checks the predicted distribution of each named quantity.")
    score_panels([f"crps_count_{p}" for p, _ in PURPOSES], "activity_count_scores",
        "3. How well are activity counts predicted for each person?",
        "These scores compare each person's distribution of predicted activity counts with their observed count. Shopping, work, education, leisure and other visits are evaluated separately; zero-activity days are included.")
    score_panels(["brier_work", "brier_education", "brier_mobile", "brier_car", "brier_pt"], "event_scores",
        "4. Can the method predict which activities and modes occur?",
        "An event score checks a probability, for example the chance of doing any work or using public transport during the day. Mobile means making at least one trip.")

    metrics = ["energy", "chain_brier", "crps_distance_km", "crps_travel_minutes", "crps_trip_count", "crps_count_shop"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, comparison in zip(axes, ("V2 minus V1", "V3 minus V1", "V3 minus V2")):
        data = paired.loc[paired.weighting.eq("survey") & paired.comparison.eq(comparison)].set_index("metric").reindex(metrics)
        baseline = comparison.split()[-1]
        base = summary.loc[summary.weighting.eq("survey") & summary.algorithm.eq(baseline)].set_index("metric").reindex(metrics)["mean"].to_numpy()
        for y, (_, row) in enumerate(data.iterrows()):
            if not np.isfinite(base[y]) or base[y] <= 0:
                ax.text(.02, y, "N/A: zero comparator score", transform=ax.get_yaxis_transform(), fontsize=8)
                continue
            x, lo, hi = np.asarray([row.difference, row.lower, row.upper]) / base[y] * 100
            color = "#269568" if hi < 0 else ("#c64c4c" if lo > 0 else "#777777")
            ax.errorbar(x, y, xerr=np.maximum(0, [[x - lo], [hi - x]]), fmt="o", color=color, capsize=4)
        ax.axvline(0, color="black", lw=1)
        ax.set(yticks=range(len(metrics)), yticklabels=[LABELS[m] for m in metrics], title=comparison,
               xlabel="Score change relative to comparator (%)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    save(fig, "paired_score_changes", "5. Are the differences clear in this experiment?",
         "Differences compare the same people, resampling households. Green intervals lie entirely below zero, red above zero, and grey span zero. Percentages use the comparator's mean score as denominator; intervals are not adjusted for multiple comparisons.",
         "Further left favours the first named method; an interval spanning zero gives no clear separation.")

    # Saved donor rows recover actual predicted days, rather than distributions of ensemble means.
    truth_columns = [c for c in scores if c.startswith("truth_")]
    truth = scores.groupby("mz_id")[truth_columns].first()
    actual = primary.loc[primary.algorithm.eq("V1")].set_index("mz_id").sort_index()
    ids = actual.index
    weights = actual.survey_weight.to_numpy()
    outcomes = ["distance_km", "travel_minutes", "trip_count"] + [f"count_{p}" for p, _ in PURPOSES]
    draws = {a: {outcome: [] for outcome in outcomes} for a in ALGORITHMS}
    draw_weights = {a: [] for a in ALGORITHMS}
    for fold in range(manifest["folds"]):
        for algorithm in ALGORITHMS:
            path = output / f"fold_{fold}" / f"{algorithm}_donor_rows.npz"
            if not path.exists():
                raise FileNotFoundError(f"Saved donor draws needed for behaviour distributions: {path}")
            with np.load(path) as archive:
                keep = np.isin(archive["test_ids"], ids)
                rows = archive["rows"][keep]
                recipient_weights = actual.loc[archive["test_ids"][keep], "survey_weight"].to_numpy()
                if (rows < 0).any():
                    raise ValueError("Unmatched draw entered the common cohort")
                donor_ids = archive["source_ids"][rows.ravel()]
                for outcome in outcomes:
                    values = truth.loc[donor_ids, "truth_" + outcome].to_numpy()
                    if not np.isfinite(values).all():
                        raise ValueError(f"Saved donor truth unavailable for {outcome}; cannot reconstruct distribution")
                    draws[algorithm][outcome].append(values)
                draw_weights[algorithm].append(np.repeat(recipient_weights / rows.shape[1], rows.shape[1]))
    draw_weights = {a: np.concatenate(v) for a, v in draw_weights.items()}
    draws = {a: {k: np.concatenate(v) for k, v in columns.items()} for a, columns in draws.items()}
    totals = []
    for outcome in outcomes:
        totals.append(dict(dataset="Observed", outcome=outcome, mean=np.average(actual["truth_" + outcome], weights=weights)))
        for algorithm in ALGORITHMS:
            totals.append(dict(dataset=algorithm, outcome=outcome,
                               mean=np.average(draws[algorithm][outcome], weights=draw_weights[algorithm])))
    totals = pd.DataFrame(totals)
    totals.to_csv(output / "observed_predicted_daily_means.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, outcome, label in zip(axes, outcomes[:3], ("Daily distance (km)", "Daily travel time (minutes)", "Trips per day")):
        rows = totals.loc[totals.outcome.eq(outcome)].set_index("dataset").reindex(["Observed", *ALGORITHMS])
        bars = ax.bar(rows.index, rows["mean"], color=[COLORS[a] for a in rows.index])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.axhline(rows.loc["Observed", "mean"], color="black", ls="--", lw=1)
        ax.set(title=label, ylabel="Survey-weighted mean per person-day")
        ax.margins(y=.2)
    save(fig, "daily_totals", "6. How far and how long do the predicted people travel?",
         f"Black bars show the respondents' observed diaries. Coloured bars average the {manifest['draws']} predicted donor days for those same people. Distance is the sum of straight-line trip distances, including return-home trips; it is not routed network distance.",
         "Closer to black is better for this aggregate check; neither a larger nor a smaller total is inherently better.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    quantiles = []
    for ax, outcome, label in zip(axes, outcomes[:2], ("Daily distance (km)", "Daily travel time (minutes)")):
        for dataset in ("Observed", *ALGORITHMS):
            values = actual["truth_" + outcome].to_numpy() if dataset == "Observed" else draws[dataset][outcome]
            mass = weights if dataset == "Observed" else draw_weights[dataset]
            order = np.argsort(values)
            x, w = values[order], mass[order]
            cdf = np.cumsum(w) / w.sum()
            ax.step(np.r_[0, x], np.r_[0, cdf], where="post", label=dataset, color=COLORS[dataset], lw=2 if dataset == "Observed" else 1.5)
            for q in (.5, .75, .9, .95, .99):
                quantiles.append(dict(dataset=dataset, outcome=outcome, quantile=q,
                                      value=x[min(np.searchsorted(cdf, q), len(x) - 1)]))
        ax.set(xlabel=label, ylabel="Share of person-days at or below this value", ylim=(0, 1.02))
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(alpha=.2)
        ax.legend()
    pd.DataFrame(quantiles).to_csv(output / "daily_distribution_quantiles.csv", index=False)
    save(fig, "daily_distributions", "7. Does the full travel distribution resemble the observed days?",
         "Each curve is a cumulative distribution of complete days, including zero-travel days. For example, the height at 20 km is the share travelling no more than 20 km in the day. Predicted curves use all donor draws, not just each person's predicted mean. The horizontal scale is compressed above one unit to retain long-distance days.",
         "Closer to the black curve indicates better aggregate distribution agreement; individual predictive scores remain the stronger person-level test.")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(5)
    for index, dataset in enumerate(("Observed", *ALGORITHMS)):
        values = [totals.loc[totals.dataset.eq(dataset) & totals.outcome.eq("count_" + p), "mean"].iloc[0] for p, _ in PURPOSES]
        bars = ax.bar(x + (index - 1.5) * .19, values, .19, label=dataset, color=COLORS[dataset])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=[label for _, label in PURPOSES], ylabel="Visits per person-day (including people with zero visits)")
    ax.margins(y=.25)
    ax.legend(ncols=4)
    save(fig, "activity_means", "8. How many shopping, work and other activities are expected?",
         "Black shows the average observed number of visits per respondent-day; coloured bars show the model expectations for the same respondents. A visit is a trip arrival with the named purpose, so repeated visits count separately. These are activity counts, not the fraction of people doing that activity.",
         "Closer to the black bar is better for this check. These are held-out survey benchmarks, not known totals for Switzerland.")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    distribution_rows = []
    for ax, (purpose, label) in zip(axes.flat, PURPOSES):
        outcome = "count_" + purpose
        for index, dataset in enumerate(("Observed", *ALGORITHMS)):
            values = actual["truth_" + outcome].to_numpy() if dataset == "Observed" else draws[dataset][outcome]
            mass = weights if dataset == "Observed" else draw_weights[dataset]
            hist = np.bincount(np.minimum(values.astype(int), 5), weights=mass, minlength=6) / mass.sum() * 100
            ax.bar(np.arange(6) + (index - 1.5) * .19, hist, .19, label=dataset, color=COLORS[dataset])
            distribution_rows.extend(dict(dataset=dataset, purpose=purpose, visits=bin_, percent=value)
                                     for bin_, value in zip(("0", "1", "2", "3", "4", "5+"), hist))
        ax.set(title=label, xticks=range(6), xticklabels=("0", "1", "2", "3", "4", "5+"), xlabel="Visits per day", ylabel="Person-days (%)")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    pd.DataFrame(distribution_rows).to_csv(output / "activity_count_distributions.csv", index=False)
    save(fig, "activity_distributions", "9. Are visits spread across people in the expected way?",
         "Two methods can have the same average shopping count while assigning visits to different numbers of people. These panels compare no visits, one visit, repeated visits and five or more visits, using full predicted donor-day distributions.",
         "Closer to black in every bin is better for distribution agreement; no single high or low bin is inherently better.")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    calibration = pd.read_csv(output / "calibration.csv")
    for ax, event in zip(axes, ("work", "education", "mobile")):
        for algorithm in ALGORITHMS:
            data = calibration.loc[calibration.algorithm.eq(algorithm) & calibration.event.eq(event)]
            ax.plot(data.predicted, data.observed, "o-", label=algorithm, color=COLORS[algorithm])
        ax.plot([0, 1], [0, 1], "k--", alpha=.6)
        ax.set(title=event.title(), xlabel="Predicted probability", ylabel="Observed fraction", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
    save(fig, "event_calibration", "10. Do predicted probabilities mean what they say?",
         "People are grouped by their predicted probability. A predicted work probability of 0.3 should correspond to work being observed about 30% of the time. Points show these group averages.",
         "Closer to the dashed diagonal is better calibration; calibration alone does not establish accuracy.")

    subgroup = pd.read_csv(output / "subgroup_scores.csv")
    for number, (grouping, title) in enumerate((("canton_id", "canton"), ("density_bin", "residence density")), start=11):
        part = subgroup.loc[subgroup.grouping.eq(grouping)]
        if part.empty:
            continue
        group_names = list(part["group"].drop_duplicates())
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, ("energy", "chain_brier", "crps_distance_km")):
            for algorithm in ALGORITHMS:
                rows = part.loc[part.algorithm.eq(algorithm) & part.metric.eq(metric)].set_index("group").reindex(group_names)
                ax.plot(range(len(group_names)), rows["mean"], "o-", label=algorithm, color=COLORS[algorithm])
            labels = [{"1": "Zürich", "22": "Vaud", "25": "Geneva"}.get(str(g), str(g)) for g in group_names]
            ax.set(xticks=range(len(labels)), xticklabels=labels, title=LABELS[metric])
            ax.tick_params(axis="x", rotation=25)
            ax.grid(alpha=.2)
            ax.legend()
        save(fig, f"scores_by_{grouping}", f"{number}. Does performance vary by {title}?",
             "Each point is the survey-weighted average score within the labelled shared-cohort subgroup. Density intervals are residents per square kilometre. These descriptive subgroup means do not include uncertainty intervals, and small subgroups need caution.",
             "Lower is better within the same subgroup; different groups can have different prediction difficulty.")

    runtime = pd.read_csv(output / "runtime.csv").groupby(["algorithm", "phase"]).seconds.sum().unstack(fill_value=0).reindex(ALGORITHMS)
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(3)
    for phase in runtime:
        values = runtime[phase].to_numpy() / 60
        ax.barh(ALGORITHMS, values, left=bottom, label=phase)
        bottom += values
    for y, value in enumerate(bottom):
        ax.text(value, y, f" {value:.1f} min", va="center")
    ax.set(xlabel="Total wall time across validation folds (minutes)")
    ax.margins(x=.2)
    ax.legend()
    save(fig, "runtime_comparison", "13. What is the computational cost?",
         "Training and matching times are summed across the same validation experiment. V2 learns one model per fold and reuses it across draws. Times include saved diagnostics and reflect the job's hardware and implementation.",
         "Lower time is faster; assess it alongside accuracy and coverage.")
    return sections
