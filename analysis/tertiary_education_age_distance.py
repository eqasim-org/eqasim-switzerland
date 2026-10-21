"""Held-out-year validation of a national young-tertiary distance preference."""

from pathlib import Path

import numpy as np
import pandas as pd

from data.od.education_matrix import prepare_observations
from data.od.tertiary_education_age import (
    AGE_LIMIT, adjust_probabilities, distance_matrix, fit_decay, zone_centres)
from data.od.tertiary_education_smoothing import (
    estimate_profiles, regularise_distances, DEFAULT_PRIOR_STRENGTH,
    DEFAULT_NEIGHBORS, DEFAULT_BANDWIDTH, DEFAULT_DISTANCE_DECAY)


def configure(context):
    for stage in ["data.structural_survey.structural_survey", "data.spatial.zones",
                  "data.spatial.municipalities", "data.spatial.quarters"]:
        context.stage(stage)
    context.config("analysis_path")
    for name, default in [("tertiary_od_prior_strength", DEFAULT_PRIOR_STRENGTH),
                          ("tertiary_od_neighbors", DEFAULT_NEIGHBORS),
                          ("tertiary_od_bandwidth", DEFAULT_BANDWIDTH),
                          ("tertiary_od_distance_decay", DEFAULT_DISTANCE_DECAY)]:
        context.config(name, default)


def validate(context):
    import hashlib
    from data.od import tertiary_education_age, tertiary_education_smoothing, education_matrix
    from data.structural_survey import education_origins
    return {module.__file__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in [tertiary_education_age, tertiary_education_smoothing,
                           education_matrix, education_origins]}


def audit(survey, zones, municipalities, quarters, mapping, output_path,
          prior_strength=DEFAULT_PRIOR_STRENGTH, neighbors=DEFAULT_NEIGHBORS,
          bandwidth=DEFAULT_BANDWIDTH, distance_decay=DEFAULT_DISTANCE_DECAY):
    observations = prepare_observations(survey, tertiary_only=True)
    observations["age"] = survey.loc[observations.index, "age"]
    centres = zone_centres(zones, municipalities, quarters)
    scores, fits = [], []
    for year in sorted(observations.survey_year.unique()):
        # Both the base OD and age coefficient are fitted without the test year.
        train = observations.loc[observations.survey_year.ne(year)]
        test = observations.loc[observations.survey_year.eq(year)]
        profiles, diagnostics = estimate_profiles(train, zones, municipalities, quarters, mapping,
                                                   prior_strength, neighbors, bandwidth)
        profiles = regularise_distances(profiles, zones, municipalities, quarters,
                                         distance_decay, diagnostics.observed_share.to_numpy())
        distances = distance_matrix(profiles.index, profiles.columns, centres)
        fitted = fit_decay(profiles, distances, train)
        fits.append(dict(held_out_year=year, **fitted))
        rows = profiles.index.get_indexer(test.home_zone_id)
        columns = profiles.columns.get_indexer(test.education_zone_id)
        known = (rows >= 0) & (columns >= 0)
        young = test.age.lt(AGE_LIMIT).to_numpy()
        for label, strength in [("baseline", 0.), ("young_half", .5), ("young_full", 1.)]:
            base = profiles.to_numpy()
            adjusted = adjust_probabilities(base, distances, strength * fitted["decay_per_km"])
            probability = np.zeros(len(test))
            brier = np.ones(len(test))
            expected_distance = np.full(len(test), np.nan)
            for selected, values in [(young, adjusted), (~young, base)]:
                indices = np.flatnonzero(selected & known)
                probability[indices] = values[rows[indices], columns[indices]]
                indices = np.flatnonzero(selected & (rows >= 0))
                brier[indices] += (values ** 2).sum(axis=1)[rows[indices]] - 2 * probability[indices]
                expected_distance[indices] = (values * distances).sum(axis=1)[rows[indices]]
            for age, age_mask in [("all", np.ones(len(test), bool)), ("under24", young),
                                  ("18–23", test.age.between(18, 23).to_numpy()), ("24plus", ~young)]:
                for source in ["all", "school_departure", "residence_fallback"]:
                    selected = age_mask & (True if source == "all" else test.education_origin_source.eq(source).to_numpy())
                    weight = test.loc[selected, "weight"].to_numpy()
                    if not len(weight):
                        continue
                    scores.append(dict(held_out_year=year, model=label, age=age, source=source,
                        observations=len(weight), effective_n=weight.sum() ** 2 / (weight ** 2).sum(),
                        weighted_log_loss=np.average(-np.log(np.maximum(probability[selected], 1e-12)), weights=weight),
                        weighted_brier=np.average(brier[selected], weights=weight),
                        weighted_zero_share=np.average(probability[selected] == 0, weights=weight),
                        expected_centroid_mean_km=np.average(expected_distance[selected], weights=weight)))
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(scores)
    scores.to_csv(output_path / "held_out_year_scores.csv", index=False)
    pd.DataFrame(fits).to_csv(output_path / "fitted_coefficients.csv", index=False)
    summary = scores.groupby(["model", "age", "source"], as_index=False).agg(
        mean_log_loss=("weighted_log_loss", "mean"), mean_brier=("weighted_brier", "mean"))
    summary.to_csv(output_path / "setting_comparison.csv", index=False)
    return summary


def execute(context):
    municipalities, mapping = context.stage("data.spatial.municipalities")
    return audit(context.stage("data.structural_survey.structural_survey"),
        context.stage("data.spatial.zones"), municipalities, context.stage("data.spatial.quarters"), mapping,
        Path(context.config("analysis_path")) / "tertiary_education_age_distance",
        context.config("tertiary_od_prior_strength"), context.config("tertiary_od_neighbors"),
        context.config("tertiary_od_bandwidth"), context.config("tertiary_od_distance_decay"))
