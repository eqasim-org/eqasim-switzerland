"""Validate a small shared distance penalty using held-out survey years."""

from pathlib import Path
import numpy as np
import pandas as pd
from data.od.education_matrix import prepare_observations
from data.od.tertiary_education_smoothing import (
    estimate_profiles, regularise_distances, DEFAULT_PRIOR_STRENGTH,
    DEFAULT_NEIGHBORS, DEFAULT_BANDWIDTH)


def configure(context):
    for stage in ["data.structural_survey.structural_survey", "data.spatial.zones",
                  "data.spatial.municipalities", "data.spatial.quarters"]:
        context.stage(stage)
    context.config("analysis_path")


def audit(survey, zones, municipalities, quarters, mapping, output_path):
    """Keep the grid small and fixed; do not tune to synthetic age-bin means."""
    observations = prepare_observations(survey, tertiary_only=True)
    scores = []
    for year in sorted(observations["survey_year"].unique()):
        train = observations[observations["survey_year"].ne(year)]
        test = observations[observations["survey_year"].eq(year)]
        profiles, diagnostics = estimate_profiles(train, zones, municipalities, quarters, mapping,
            DEFAULT_PRIOR_STRENGTH, DEFAULT_NEIGHBORS, DEFAULT_BANDWIDTH)
        rows = profiles.index.get_indexer(test["home_zone_id"])
        columns = profiles.columns.get_indexer(test["education_zone_id"])
        known = (rows >= 0) & (columns >= 0)
        for decay in [0., .005, .01, .02]:
            adjusted = regularise_distances(profiles, zones, municipalities, quarters, decay,
                                             diagnostics["observed_share"].to_numpy()).to_numpy()
            probabilities = np.zeros(len(test))
            probabilities[known] = adjusted[rows[known], columns[known]]
            loss = -np.log(np.maximum(probabilities, 1e-12))
            brier = np.ones(len(test))
            valid_rows = rows >= 0
            brier[valid_rows] += (adjusted[rows[valid_rows]] ** 2).sum(axis=1) - 2 * probabilities[valid_rows]
            scores.append({"held_out_year": year, "distance_decay_per_km": decay,
                "weighted_log_loss": np.average(loss, weights=test["weight"]),
                "weighted_brier": np.average(brier, weights=test["weight"]),
                "weighted_zero_share": np.average(probabilities == 0, weights=test["weight"])})
    scores = pd.DataFrame(scores)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path / "held_out_year_scores.csv", index=False)
    summary = scores.groupby("distance_decay_per_km").agg(
        mean_log_loss=("weighted_log_loss", "mean"), mean_brier=("weighted_brier", "mean"),
        mean_zero_share=("weighted_zero_share", "mean")).reset_index()
    summary.to_csv(output_path / "setting_comparison.csv", index=False)
    return summary


def execute(context):
    municipalities, mapping = context.stage("data.spatial.municipalities")
    return audit(context.stage("data.structural_survey.structural_survey"), context.stage("data.spatial.zones"),
                 municipalities, context.stage("data.spatial.quarters"), mapping,
                 Path(context.config("analysis_path")) / "tertiary_education_distance_regularisation")
