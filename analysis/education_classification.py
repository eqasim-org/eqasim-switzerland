"""Validate joint education imputation against held-out structural-survey years."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from synthesis.population.models.education import (
    DEFAULT_PRIOR_STRENGTH, OUTCOMES, EDUCATION_GROUPS, prepare_training,
    fit_model, predict_probabilities)


def configure(context):
    context.stage("synthesis.population.models.education")
    context.stage("data.structural_survey.student_training")
    context.config("analysis_path")
    context.config("education_level_prior_strength", DEFAULT_PRIOR_STRENGTH)


def distributions(frame, weight_column=None):
    frame = frame.copy()
    frame["_audit_weight"] = frame[weight_column] if weight_column else 1.0
    return frame.groupby(["education_group", "education_destination_status"], dropna=False).agg(
        persons=("_audit_weight", "size"), weighted_persons=("_audit_weight", "sum")).reset_index()


def audit(survey, output_path, prior_strength=DEFAULT_PRIOR_STRENGTH):
    """Hold out each survey year to assess joint education predictions and backoff."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    scores, comparisons, methods = [], [], []
    for year in sorted(survey["survey_year"].unique()):
        train = survey[survey["survey_year"].ne(year)]
        held_out = prepare_training(survey[survey["survey_year"].eq(year)]).reset_index(drop=True)
        actual = held_out["_outcome"].to_numpy()
        weights = held_out["_weight"].to_numpy()
        training = prepare_training(train)
        global_probability = np.bincount(training["_outcome"], weights=training["_weight"], minlength=len(OUTCOMES))
        global_probability = global_probability / global_probability.sum()
        for strength in sorted(set([0.0, prior_strength])):
            model = fit_model(train, strength)
            probability, method, support = predict_probabilities(model, held_out)
            truth_probability = probability[np.arange(len(actual)), actual]
            scores.append({"held_out_year": year, "prior_strength": strength, "observations": len(actual),
                           "weighted_joint_log_loss": np.average(-np.log(np.maximum(truth_probability, 1e-12)), weights=weights),
                           "weighted_joint_brier": np.average((probability ** 2).sum(axis=1) - 2 * truth_probability + 1, weights=weights),
                           "weighted_zero_probability_share": np.average(truth_probability == 0, weights=weights),
                           "global_joint_log_loss": np.average(-np.log(np.maximum(global_probability[actual], 1e-12)), weights=weights)})
            if strength != prior_strength:
                continue
            for outcome, (code, status) in enumerate(OUTCOMES):
                comparisons.append({"held_out_year": year, "education_group": EDUCATION_GROUPS.get(code, "unresolved"),
                                    "current_education": code, "education_destination_status": status,
                                    "observed_weighted_persons": weights[actual == outcome].sum(),
                                    "expected_weighted_persons": np.dot(weights, probability[:, outcome])})
            method_table = pd.DataFrame({"method": method, "effective_n": support})
            method_table["held_out_year"] = year
            methods.append(method_table.groupby(["held_out_year", "method"]).agg(
                persons=("effective_n", "size"), median_effective_n=("effective_n", "median")).reset_index())
        print("Validated held-out survey year %s (%s student observations)" % (year, len(held_out)), flush=True)
    scores = pd.DataFrame(scores)
    scores.to_csv(output_path / "held_out_scores.csv", index=False)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output_path / "held_out_joint_distribution.csv", index=False)
    comparison.groupby(["held_out_year", "education_group"]).agg(
        observed_weighted_persons=("observed_weighted_persons", "sum"),
        expected_weighted_persons=("expected_weighted_persons", "sum"),
    ).reset_index().to_csv(output_path / "held_out_group_distribution.csv", index=False)
    pd.concat(methods).to_csv(output_path / "held_out_imputation_methods.csv", index=False)
    (output_path / "README.md").write_text(
        "# Joint education classification validation\n\n"
        "Each structural-survey year is held out and predicted from the other two. "
        "Weights are used within a held-out year, not summed as an annual population. "
        "Scores describe current study level and destination-country status among "
        "age-15+ people already flagged as students; the existing binary student "
        "model is not retrained by this audit.\n\n"
        "Predictions use weighted age-band/employment/apprenticeship/sex/canton distributions with "
        "hierarchical smoothing. Unknown education levels and country status remain "
        "outcomes. Abroad is based on current school-address country, not nationality "
        "or educational origin. Domestic status uses a Swiss destination zone when "
        "the school-country field is not applicable. The standalone audit uses positive "
        "raw Swiss school-municipality codes instead of mapped zones; foreign country "
        "takes priority in both cases. No donor education destinations are copied.\n\n"
        "held_out_scores.csv compares smoothing with unsmoothed profiles and a global "
        "joint distribution. Other tables report expected versus observed shares and "
        "fallback support. This validates against real survey observations; it is not "
        "a run of the full synthetic-population pipeline or the existing MATSim export.\n",
        encoding="utf-8")
    print(scores.to_string(index=False), flush=True)
    return scores


def execute(context):
    path = Path(context.config("analysis_path")) / "education_classification"
    path.mkdir(parents=True, exist_ok=True)
    population = context.stage("synthesis.population.models.education")
    distributions(population).to_csv(path / "synthetic_distribution.csv", index=False)
    population.groupby(["education_imputation_method", "education_group"]).size().rename("persons").reset_index().to_csv(
        path / "synthetic_imputation_methods.csv", index=False)
    return audit(context.stage("data.structural_survey.student_training"), path,
                 context.config("education_level_prior_strength"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH)
    args = parser.parse_args()
    # Reuse raw survey inputs, applying the same sex/apprenticeship recodes as
    # the processed stage; validation needs no spatial imputation.
    import data.structural_survey.raw
    from analysis.tertiary_education_od import InputContext
    survey = data.structural_survey.raw.execute(InputContext(args.data_path)).copy()
    survey["sex"] -= 1
    survey.loc[survey["current_education"].eq(6), "employed"] = 1
    audit(survey, args.output_path, args.prior_strength)
