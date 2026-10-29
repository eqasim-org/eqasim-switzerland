"""Reader-facing diagnostics computed only from saved out-of-fold predictions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import clustered_interval
from .reporting import save_figure

COLORS = {"V1": "#D55E00", "V2": "#0072B2"}


def comparison_diagnostics(output, scores, respondents, average, bootstrap, seed):
    """Average repeated fits per person before comparing algorithms or bootstrapping."""
    numeric = ("distance_km", "travel_minutes", "trip_count")
    columns = [f"{prefix}_{metric}" for metric in numeric for prefix in ("truth", "mean")]
    columns += [f"{prefix}_{level}_{metric}" for metric in numeric for level in (.5, .8, .95)
                for prefix in ("coverage", "width")]
    columns += ["truth_mobile", "truth_work", "truth_education"]
    predictions = scores.groupby(["mz_id", "algorithm"], as_index=False)[columns].mean()
    predictions = predictions.merge(respondents, on="mz_id", validate="many_to_one")
    # The same respondents support both curves; unmatched people are reported separately.
    paired_ids = _paired_ids(average)
    predictions = predictions.loc[predictions.mz_id.isin(paired_ids) & predictions.algorithm.isin(COLORS)]
    paired = average.loc[average.mz_id.isin(paired_ids) & average.algorithm.isin(COLORS)]
    _error_distributions(paired, output)
    day_types = _day_type_table(paired, predictions, bootstrap, seed)
    day_types.to_csv(output / "chain_skill_by_day_type.csv", index=False)
    _day_type_figure(day_types, output)
    calibration = _mean_calibration(predictions, output)
    calibration.to_csv(output / "mean_prediction_calibration.csv", index=False)
    coverage = _interval_coverage(predictions, output)
    coverage.to_csv(output / "predictive_interval_coverage.csv", index=False)
    cohorts = _cohort_differences(paired, bootstrap, seed)
    cohorts.to_csv(output / "cohort_paired_differences.csv", index=False)
    _cohort_figure(cohorts, output)
    predictions.to_csv(output / "per_person_mean_predictions.csv", index=False)
    figures = ["behaviour_error_distributions", "chain_skill_by_day_type", "mean_prediction_calibration",
               "predictive_interval_coverage", "cohort_predictive_skill"]
    return figures, [("Ordered-chain prediction by observed day type", day_types),
                     ("Prediction interval coverage and average width", coverage),
                     ("Paired behaviour skill across groups", cohorts)]


def _paired_ids(average):
    table = average.loc[average.algorithm.isin(COLORS)].pivot(index="mz_id", columns="algorithm", values="energy")
    if not set(COLORS).issubset(table.columns):
        raise ValueError("Saved scores must contain both V1 and V2")
    return table.dropna(subset=list(COLORS)).index


def _error_distributions(average, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, label in zip(axes, ("crps_distance_km", "crps_travel_minutes", "chain_brier"),
                                  ("Distance error (CRPS; km)", "Travel-time error (CRPS; minutes)",
                                   "Ordered-chain probability error (Brier)")):
        for algorithm, group in average.groupby("algorithm"):
            group = group.loc[group[metric].notna()].sort_values(metric)
            cumulative = group.survey_weight.cumsum() / group.survey_weight.sum()
            axis.step(group[metric], cumulative, where="post", label=algorithm, color=COLORS[algorithm])
        # Keep outliers in the CSV; limit the horizontal view to the pooled 98th percentile.
        finite = average[metric].dropna()
        if len(finite) and finite.quantile(.98) > 0:
            axis.set_xlim(0, finite.quantile(.98) * 1.03)
        axis.set(xlabel=label, ylabel="Share of respondents at or below this error", ylim=(0, 1))
        axis.grid(alpha=.2)
    axes[0].legend()
    save_figure(fig, output, "behaviour_error_distributions")


def _day_type_table(average, predictions, bootstrap, seed):
    truth = predictions.drop_duplicates("mz_id").set_index("mz_id")
    categories = pd.Series(np.select(
        [truth.truth_mobile.eq(0), truth.truth_education.gt(0), truth.truth_work.gt(0)],
        ["Stayed home", "Education day (may include work)", "Work day (no education)"],
        default="Other travelling day"), index=truth.index)
    frame = average.merge(categories.rename("day_type"), left_on="mz_id", right_index=True, validate="many_to_one")
    rows, rng = [], np.random.RandomState(seed)
    for (category, algorithm), group in frame.groupby(["day_type", "algorithm"]):
        value, low, high, n, ess = clustered_interval(group.chain_brier, group.survey_weight,
                                                      group.household_group, rng, bootstrap)
        rows.append(dict(day_type=category, algorithm=algorithm, chain_brier=value, low=low, high=high,
                         respondents=len(group), households=n, effective_n=ess))
    return pd.DataFrame(rows)


def _day_type_figure(table, output):
    categories = table.day_type.unique()
    fig, axis = plt.subplots(figsize=(10, 4))
    for offset, algorithm in zip((-.12, .12), COLORS):
        group = table.loc[table.algorithm.eq(algorithm)].set_index("day_type").reindex(categories)
        axis.errorbar(group.chain_brier, np.arange(len(categories)) + offset,
                      xerr=np.maximum([group.chain_brier-group.low, group.high-group.chain_brier], 0),
                      fmt="o", capsize=3, label=algorithm, color=COLORS[algorithm])
    axis.set_yticks(np.arange(len(categories)), categories)
    axis.set_xlabel("Ordered-chain probability error (Brier); lower is better")
    axis.legend()
    save_figure(fig, output, "chain_skill_by_day_type")


def _mean_calibration(predictions, output):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, label in zip(axes, ("distance_km", "travel_minutes", "trip_count"),
                                  ("Daily distance (km)", "Daily travel time (minutes)", "Daily trip count")):
        points = []
        for algorithm, group in predictions.groupby("algorithm"):
            bins = pd.qcut(group["mean_" + metric], min(10, len(group)), duplicates="drop")
            # Constant predictions still have one meaningful calibration point.
            if bins.isna().all():
                bins = pd.Series("all", index=group.index)
            xs, ys = [], []
            for interval, part in group.groupby(bins, observed=True):
                x = np.average(part["mean_" + metric], weights=part.survey_weight)
                y = np.average(part["truth_" + metric], weights=part.survey_weight)
                xs.append(x); ys.append(y)
                rows.append(dict(metric=metric, algorithm=algorithm, predicted_bin=str(interval),
                                 predicted=x, observed=y, respondents=len(part)))
            axis.plot(xs, ys, "o-", color=COLORS[algorithm], label=algorithm, markersize=4)
            points.extend(xs + ys)
        limit = max(points + [1]) * 1.05
        axis.plot([0, limit], [0, limit], ":", color="grey")
        axis.set(xlabel="Predicted mean: " + label, ylabel="Observed mean", xlim=(0, limit), ylim=(0, limit))
    axes[0].legend()
    save_figure(fig, output, "mean_prediction_calibration")
    return pd.DataFrame(rows)


def _interval_coverage(predictions, output):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, label in zip(axes, ("distance_km", "travel_minutes", "trip_count"),
                                  ("Daily distance", "Daily travel time", "Daily trip count")):
        for algorithm, group in predictions.groupby("algorithm"):
            observed = []
            for nominal in (.5, .8, .95):
                coverage = np.average(group[f"coverage_{nominal}_{metric}"], weights=group.survey_weight)
                width = np.average(group[f"width_{nominal}_{metric}"], weights=group.survey_weight)
                observed.append(coverage)
                rows.append(dict(metric=metric, algorithm=algorithm, nominal=nominal, coverage=coverage,
                                 mean_width=width, respondents=len(group)))
            axis.plot([50, 80, 95], np.array(observed)*100, "o-", color=COLORS[algorithm], label=algorithm)
        axis.plot([0, 100], [0, 100], ":", color="grey")
        axis.set(title=label, xlabel="Nominal interval coverage (%)", ylabel="Actual coverage (%)", xlim=(40, 100), ylim=(0, 100))
    axes[0].legend()
    save_figure(fig, output, "predictive_interval_coverage")
    return pd.DataFrame(rows)


def _cohort_differences(average, bootstrap, seed):
    table = average.pivot(index="mz_id", columns="algorithm", values="energy")
    metadata = average.drop_duplicates("mz_id").set_index("mz_id").reindex(table.index)
    metadata["delta"] = table.V2 - table.V1
    rows, rng = [], np.random.RandomState(seed)
    for feature in ("age", "income_class", "N_children_under_12", "population_density"):
        value = metadata[feature]
        if feature in ("age", "population_density") and value.nunique() > 5:
            category = pd.qcut(value, 5, duplicates="drop")
        elif feature == "N_children_under_12":
            category = value.clip(upper=3)
        else:
            category = value
        for label, group in metadata.groupby(category, observed=True):
            if len(group) < 30:
                continue
            estimate, low, high, n, ess = clustered_interval(group.delta, group.survey_weight,
                                                            group.household_group, rng, bootstrap)
            if feature == "age":
                label = f"{group.age.min():g}–{group.age.max():g}"
            elif feature == "population_density":
                label = f"{group.population_density.min():,.0f}–{group.population_density.max():,.0f}"
            elif feature == "N_children_under_12" and label == 3:
                label = "3+"
            else:
                label = f"{label:g}"
            rows.append(dict(feature=feature, category=label, delta=estimate, low=low, high=high,
                             respondents=len(group), households=n, effective_n=ess))
    return pd.DataFrame(rows, columns=["feature", "category", "delta", "low", "high", "respondents", "households", "effective_n"])


def _cohort_figure(table, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, feature, label in zip(axes.ravel(), ("age", "income_class", "N_children_under_12", "population_density"),
                                   ("Age ranges (years)", "Original income class", "Children under 12", "Residential density ranges")):
        group = table.loc[table.feature.eq(feature)]
        if group.empty:
            axis.text(.5, .5, "Too few respondents (minimum 30 per group)", ha="center", transform=axis.transAxes)
        else:
            axis.errorbar(group.delta, np.arange(len(group)),
                          xerr=np.maximum([group.delta-group.low, group.high-group.delta], 0),
                          fmt="o", color=COLORS["V2"], capsize=3)
            axis.set_yticks(np.arange(len(group)), group.category)
        axis.axvline(0, color="grey", linewidth=.8)
        axis.set(title=label, xlabel="V2 minus V1 joint behaviour error; negative favours V2")
    save_figure(fig, output, "cohort_predictive_skill")


def saved_findings(differences):
    selected = differences.loc[differences.reference.eq("V1") & differences.weighting.eq("survey_weight")].set_index("metric")
    labels = {"energy": "joint day behaviour", "chain_brier": "ordered activity/mode chains",
              "crps_distance_km": "daily distance", "crps_travel_minutes": "daily travel time"}
    findings = []
    for metric, label in labels.items():
        if metric not in selected.index:
            continue
        row = selected.loc[metric]
        value = row.relative_percent
        if not np.isfinite(value):
            continue
        direction = "lower" if value < 0 else "higher"
        conclusion = ("The paired interval excludes zero." if row.high < 0 or row.low > 0
                      else "The paired interval includes zero; this test does not establish a clear difference.")
        findings.append(f"For {label}, V2's error is {abs(value):.1f}% {direction} than V1's "
                        f"(95% interval for V2 minus V1: {row.relative_low:.1f}% to {row.relative_high:.1f}%). {conclusion}")
    return " ".join(findings) + " These are survey-weighted held-out results, conditional on the saved fitted models."
