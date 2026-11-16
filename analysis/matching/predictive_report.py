"""Rebuild predictive-validation figures and HTML using saved scores only."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import clustered_interval
from . import predictive_figures

ALGORITHMS = ("V1", "V2", "V3")
SCORES = ("energy", "variogram", "chain_brier", "crps_distance_km", "crps_travel_minutes",
          "crps_trip_count", "crps_minutes_work", "crps_minutes_education", "brier_work",
          "brier_education", "brier_mobile", "brier_car", "brier_pt",
          "crps_count_work", "crps_count_education", "crps_count_shop", "crps_count_leisure", "crps_count_other")


def render(directory):
    output = Path(directory)
    manifest = json.loads((output / "manifest.json").read_text())
    scores = pd.read_csv(output / "out_of_fold_scores.csv")
    people = pd.read_csv(output / "respondents.csv")
    scores = scores.merge(people, on="mz_id", validate="many_to_one")
    common = scores.groupby("mz_id").covered.agg(["min", "count"])
    common_ids = common.index[common["min"].eq(1) & common["count"].eq(3)]
    scores = scores.copy()
    scores["common"] = scores.mz_id.isin(common_ids)
    density = people.population_density / manifest["density_area_km2"]
    people["density_bin"] = pd.cut(density, [-np.inf] + manifest["density_bounds"] + [np.inf],
                                   right=False).astype(str)
    scores = scores.merge(people[["mz_id", "density_bin"]], on="mz_id", validate="many_to_one")
    sections = ["<h1>Predicting unseen travel diaries: V1, V2 and V3</h1>",
        "<p>Each respondent is temporarily treated as a synthetic person. Their observed diary is hidden, "
        "and each algorithm selects complete diaries from respondents in other households. "
        "We compare the predicted distribution with that person's observed day. This tests individual "
        "prediction without requiring the synthetic population to reproduce survey-wide totals.</p>",
        f"<p>{manifest['respondents']:,} respondents; {manifest['folds']} household-disjoint folds; "
        f"{manifest['draws']} draws per person; {manifest['matching_executions_per_algorithm']} "
        "matching executions per algorithm in total. V2 is trained once in each donor fold; V1 and V3 "
        "use the production matching stages. Fitted neural scalers and early stopping use donor data. "
        "Genuine non-travel days remain eligible. Scenario-imposed home-only bypasses are not predictions.</p>",
        "<p>V1 uses ordered categorical relaxation. V2 learns a behaviour-informed latent space. "
        "V3 searches categorical mismatch counts without a fixed relaxation order. Current configured "
        "constraints are retained for each method; these results compare the complete algorithms.</p>"]
    shared = manifest.get("shared_model", False)
    if shared:
        sections[:3] = ["<h1>Microcensus diary reconstruction with donor exclusion</h1>",
            "<p>One V2 encoder is fitted once using a 90% fitting / 10% early-stopping validation split. "
            "Three household groups restrict donor availability only. Each recipient receives diaries "
            "from the other two groups, with their own household excluded. Their behaviour may already "
            "have contributed to learning the encoder. This is donor-restricted reconstruction, not "
            "prediction of wholly unseen behaviour.</p>",
            f"<p>{manifest['respondents']:,} respondents; three donor-exclusion groups; "
            f"{manifest['draws']} matching repetitions per person and method. V2 is never retrained "
            "between groups or repetitions. Genuine non-travel days remain eligible.</p>"]
    coverage = scores.groupby("algorithm").agg(respondents=("mz_id", "size"),
                                               matching_coverage=("covered", "mean"))
    coverage.to_csv(output / "coverage.csv")
    sections.extend(["<h2>Matching coverage</h2><p>Coverage is the fraction receiving a valid donor in every "
                     "draw. The main scores use exactly the same respondents for all methods. "
                     "Coverage is reported separately because excluding difficult cases can improve scores.</p>",
                     coverage.to_html(float_format=lambda x: f"{x:.4f}"),
                     f"<p>The shared evaluation cohort contains {len(common_ids):,} respondents.</p>",
                     "<h2>Predictive accuracy</h2><p>Lower scores are better. Energy measures error in a "
                     "whole-day vector of counts, distances, times and mode use, standardized within each donor fold. "
                     "The variogram score also checks relationships between selected daily totals. Chain Brier "
                     "measures probabilities of complete ordered purpose-and-mode chains. CRPS measures the "
                     "distance between a predicted distribution and the observed value, in the outcome's units. "
                     "Event Brier measures probability errors, for example whether any work activity occurs.</p>",
                     "<p>Intervals resample households 500 times by default. They describe respondent sampling "
                     "uncertainty conditional on these fitted models and splits; they do not include retraining "
                     "or survey selection uncertainty.</p>"])
    summaries, differences = [], []
    rng = np.random.RandomState(manifest["seed"])
    primary = scores.loc[scores.common]
    weightings = (("survey", "survey_weight"),) if shared else (("survey", "survey_weight"), ("target_sensitivity", "target_weight"))
    for weighting, weight in weightings:
        for metric in SCORES:
            if metric not in primary:
                continue
            for algorithm in ALGORITHMS:
                rows = primary.loc[primary.algorithm.eq(algorithm)]
                mean, low, high, households, ess = clustered_interval(rows[metric], rows[weight],
                    rows.household_group, rng, manifest["bootstrap"])
                summaries.append(dict(weighting=weighting, metric=metric, algorithm=algorithm,
                                      mean=mean, lower=low, upper=high, households=households, effective_n=ess))
            wide = primary.pivot(index="mz_id", columns="algorithm", values=metric)
            ref = people.set_index("mz_id").loc[wide.index]
            for a, b in (("V2", "V1"), ("V3", "V1"), ("V3", "V2")):
                mean, low, high, _, _ = clustered_interval(wide[a] - wide[b], ref[weight],
                    ref.household_group, rng, manifest["bootstrap"])
                differences.append(dict(weighting=weighting, metric=metric, comparison=f"{a} minus {b}",
                                        difference=mean, lower=low, upper=high))
    summary = pd.DataFrame(summaries)
    paired = pd.DataFrame(differences)
    summary.to_csv(output / "score_summary.csv", index=False)
    paired.to_csv(output / "paired_differences.csv", index=False)
    sections.append(summary.loc[summary.weighting.eq("survey"), ["metric", "algorithm", "mean", "lower", "upper"]]
                    .to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    sections.append("<h3>Paired comparisons</h3><p>A negative difference favours the first method. "
                    "An interval spanning zero gives no clear separation with this experiment. "
                    "These intervals are exploratory and are not adjusted for multiple comparisons.</p>" +
                    paired.loc[paired.weighting.eq("survey")].drop(columns="weighting").to_html(
                        index=False, float_format=lambda x: f"{x:.4f}"))
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.flat, ("energy", "chain_brier", "crps_distance_km",
                                      "crps_travel_minutes", "crps_trip_count", "brier_work")):
        rows = summary.loc[summary.weighting.eq("survey") & summary.metric.eq(metric)].set_index("algorithm").reindex(ALGORITHMS)
        ax.errorbar(np.arange(3), rows["mean"], yerr=np.maximum(0, np.vstack([
            rows["mean"] - rows.lower, rows.upper - rows["mean"]])), fmt="o", capsize=5)
        ax.set_xticks(np.arange(3), ALGORITHMS)
        ax.set_title(metric.replace("_", " "))
        ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "predictive_scores.png", dpi=180)
    plt.close(figure)
    sections.append('<img src="predictive_scores.png" alt="Predictive scores and household intervals">')
    sections.append("<h2>Probability calibration</h2><p>A calibrated 30% work prediction should correspond "
                    "to work being observed about 30% of the time. Points group people by predicted event "
                    "probability; the diagonal indicates agreement. Calibration alone does not measure accuracy.</p>")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    calibration = []
    for ax, event in zip(axes, ("work", "education", "mobile")):
        for algorithm in ALGORITHMS:
            rows = primary.loc[primary.algorithm.eq(algorithm)].copy()
            rows["bin"] = np.minimum((rows["prob_" + event] * 5).astype(int), 4)
            points = []
            for _, group in rows.groupby("bin"):
                p = np.average(group["prob_" + event], weights=group.survey_weight)
                y = np.average(group["truth_" + event], weights=group.survey_weight)
                calibration.append(dict(event=event, algorithm=algorithm, predicted=p, observed=y, people=len(group)))
                points.append((p, y))
            if points:
                ax.plot(*np.asarray(points).T, "o-", label=algorithm)
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.set(title=event, xlabel="Predicted probability", ylabel="Observed fraction", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
    figure.tight_layout()
    figure.savefig(output / "calibration.png", dpi=180)
    plt.close(figure)
    pd.DataFrame(calibration).to_csv(output / "calibration.csv", index=False)
    sections.append('<img src="calibration.png" alt="Event probability calibration">')
    interval = primary.groupby("algorithm")[[c for c in primary if c.startswith("coverage_") or c.startswith("width_")]].apply(
        lambda g: pd.Series({c: np.average(g[c], weights=primary.loc[g.index, "survey_weight"]) for c in g}))
    interval.to_csv(output / "interval_calibration.csv")
    draws = manifest["draws"]
    sections.append("<p>Interval coverage and widths are available in <a href='interval_calibration.csv'>CSV</a>. "
                    f"There are {draws} draws per person. The expected coverage of the sample minimum-to-maximum "
                    f"range for independent continuous exchangeable observations is {(draws - 1)/(draws + 1):.1%}, even for a correct "
                    "predictive law; interpolated 95% intervals are narrower still. Discrete outcomes and ties change "
                    "this benchmark. Do not interpret finite-ensemble undercoverage as model error alone.</p>")
    strata = []
    for grouping in ("canton_id", "density_bin", "employment_status", "education_distance_source"):
        for (value, algorithm), rows in primary.groupby([grouping, "algorithm"], dropna=False):
            if grouping == "canton_id" and int(value) not in (1, 22, 25):
                continue
            for metric in ("energy", "chain_brier", "crps_distance_km", "brier_work"):
                strata.append(dict(grouping=grouping, group=value, algorithm=algorithm, metric=metric,
                    people=len(rows), mean=np.average(rows[metric], weights=rows.survey_weight)))
    pd.DataFrame(strata).to_csv(output / "subgroup_scores.csv", index=False)
    sections.append("<h2>Subgroups and survey limitations</h2><p><a href='subgroup_scores.csv'>Subgroup scores</a> "
        "cover Zürich (1), Vaud (22), Geneva (25), residence density, employment status and education-commute "
        "provenance. These comparisons use each subgroup's shared cohort; small groups need caution.</p>"
        "<p>The survey still provides the observed outcomes in this test. Household separation avoids copying "
        "the test diary, but cannot remove survey selection bias or prove accuracy for the full population. "
        "Work-location type and some education-commute predictors can be derived from diary information; "
        "the test is conditional on these known predictors and is not wholly independent of the diary. "
        "Commute imputation and shared source cleaning precede splitting. Education provenance is exported "
        "to help assess this limitation. No population-wide superiority claim follows from these scores alone.</p>"
        + ("" if shared else "<p>Population reweighting uses a classifier to distinguish survey and synthetic-person attributes, "
        "then weights survey errors towards the population's attributes. This assumes that behaviour conditional "
        "on attributes transfers between them. Check <a href='covariate_balance.csv'>balance</a> and "
        "<a href='population_support.csv'>support</a> first; reweighted scores are sensitivity results.</p>"))
    sections.append(summary.loc[summary.weighting.eq("target_sensitivity")].to_html(index=False,
                                                                                   float_format=lambda x: f"{x:.4f}"))
    runtime = pd.read_csv(output / "runtime.csv").groupby(["algorithm", "phase"]).seconds.agg(["count", "sum", "mean"])
    sections.extend(["<h2>Computation</h2><p>Wall times include donor selection and diagnostics; V2 training "
                     "is reported separately. Algorithms run sequentially in the same job.</p>", runtime.to_html(),
                     "<p><a href='manifest.json'>Experiment settings</a> · <a href='out_of_fold_scores.csv'>Per-person scores</a> · "
                     "<a href='score_summary.csv'>All scores</a> · <a href='paired_differences.csv'>Paired intervals</a></p>"])
    illustrated = predictive_figures.build(output, scores, primary, summary, paired, manifest)
    caveat = ("<p class='note'><strong>How to read this report.</strong> Stars identify the lowest point estimate, "
        "not a proven winner. All behaviour comparisons use the same held-out respondents and survey weights. "
        f"{draws} diary draws per person approximate each predictive distribution. Matching an aggregate total does "
        "not guarantee correct person-level behaviour. Household separation cannot remove survey selection bias, "
        "and some work-location and education-commute inputs reflect diary information.</p>")
    downloads = ("<h2>Saved evidence</h2><p><a href='observed_predicted_daily_means.csv'>Daily means</a> · "
        "<a href='activity_count_distributions.csv'>Activity-count distributions</a> · "
        "<a href='daily_distribution_quantiles.csv'>Distance and time quantiles</a> · "
        "<a href='score_summary.csv'>All scores</a> · <a href='paired_differences.csv'>Paired comparisons</a></p>")
    details = "\n".join(sections[3:]).replace('<img src="predictive_scores.png" alt="Predictive scores and household intervals">', '').replace(
        '<img src="calibration.png" alt="Event probability calibration">', '')
    document = "<!doctype html><meta charset='utf-8'><title>Matching predictive validation</title>" + \
        "<style>body{font:16px/1.6 sans-serif;max-width:1250px;margin:40px auto;padding:20px}img{max-width:100%}" \
        "figure{margin:24px 0}figcaption{font-size:14px;color:#555}section{margin:48px 0}" \
        "details{margin:24px 0;overflow:auto}summary{cursor:pointer;font-weight:bold}.note{padding:18px;background:#eef4f7}" \
        "table{border-collapse:collapse;font-size:13px}td,th{padding:5px;border:1px solid #ddd}</style>" + \
        "\n".join(sections[:3]) + caveat + "\n".join(illustrated) + downloads + \
        "<details><summary>Technical definitions, full numerical tables and sensitivity analysis</summary>" + details + "</details>"
    (output / "report.html").write_text(document)
    return str(output / "report.html")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    render(parser.parse_args().directory)
