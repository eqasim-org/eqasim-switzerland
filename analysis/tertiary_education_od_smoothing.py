"""Validate tertiary OD smoothing by holding out each survey year."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.tertiary_education_od import InputContext
from data.od.education_matrix import prepare_observations
from data.od.tertiary_education_smoothing import estimate_profiles, build_smoothed_matrix


def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.config("analysis_path")


def validate(context, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    observations = prepare_observations(
        context.stage("data.structural_survey.structural_survey"), tertiary_only=True)
    zones = context.stage("data.spatial.zones")
    municipalities, mapping = context.stage("data.spatial.municipalities")
    quarters = context.stage("data.spatial.quarters")
    scores = []
    for year in sorted(observations["survey_year"].unique()):
        train = observations[observations["survey_year"].ne(year)]
        test = observations[observations["survey_year"].eq(year)]
        for strength in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            for neighbors in [3, 5, 10, 20, 50]:
                profiles, diagnostics = estimate_profiles(
                    train, zones, municipalities, quarters, mapping, strength, neighbors)
                rows = profiles.index.get_indexer(test["home_zone_id"])
                columns = profiles.columns.get_indexer(test["education_zone_id"])
                probabilities = np.zeros(len(test))
                valid = (rows >= 0) & (columns >= 0)
                probabilities[valid] = profiles.to_numpy()[rows[valid], columns[valid]]
                loss = -np.log(np.maximum(probabilities, 1e-12))
                row_squared_probabilities = (profiles.to_numpy() ** 2).sum(axis=1)
                brier = np.ones(len(test))
                known_origins = rows >= 0
                brier[known_origins] += row_squared_probabilities[rows[known_origins]] - 2 * probabilities[known_origins]
                scores.append({"held_out_year": year, "prior_strength": strength,
                               "neighbors": neighbors, "bandwidth_m": 20000.0,
                               "respondents": len(test), "weighted_persons": test["weight"].sum(),
                               "weighted_log_loss": np.average(loss, weights=test["weight"]),
                               "log_loss": loss.mean(), "weighted_brier": np.average(brier, weights=test["weight"]),
                               "zero_prediction_share": (probabilities == 0).mean()})
        print("Validated held-out year", year, flush=True)
    scores = pd.DataFrame(scores)
    scores.to_csv(output_path / "held_out_year_scores.csv", index=False)
    scores["weighted_loss_total"] = scores["weighted_log_loss"] * scores["weighted_persons"]
    scores["loss_total"] = scores["log_loss"] * scores["respondents"]
    scores["weighted_brier_total"] = scores["weighted_brier"] * scores["weighted_persons"]
    settings = scores.groupby(["prior_strength", "neighbors", "bandwidth_m"]).agg(
        weighted_loss_total=("weighted_loss_total", "sum"), weighted_persons=("weighted_persons", "sum"),
        loss_total=("loss_total", "sum"), respondents=("respondents", "sum"),
        weighted_brier_total=("weighted_brier_total", "sum"))
    settings["weighted_log_loss"] = settings["weighted_loss_total"] / settings["weighted_persons"]
    settings["log_loss"] = settings["loss_total"] / settings["respondents"]
    settings["weighted_brier"] = settings["weighted_brier_total"] / settings["weighted_persons"]
    settings = settings.sort_values("weighted_log_loss").reset_index()
    settings.to_csv(output_path / "setting_comparison.csv", index=False)
    best = settings.iloc[0]
    pdf, cdf, diagnostics = build_smoothed_matrix(
        observations, zones, municipalities, quarters, mapping,
        best["prior_strength"], int(best["neighbors"]), best["bandwidth_m"])
    diagnostics.to_csv(output_path / "origin_smoothing.csv", index=False)
    np.savez_compressed(output_path / "validated_matrix.npz", pdf=pdf, cdf=cdf,
                        zone_ids=zones["zone_id"].to_numpy())
    return settings


def execute(context):
    return validate(context, Path(context.config("analysis_path")) / "tertiary_education_od_smoothing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    print(validate(InputContext(args.data_path), args.output_path).to_string(index=False))


if __name__ == "__main__":
    main()
