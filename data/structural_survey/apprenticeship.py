"""Partial repair of ambiguous SS vocational records and national stock checks."""

import numpy as np
import pandas as pd


# First national sensitivity experiment: in SS2023 this repairs about 75,000
# weighted people, bringing explicit + repaired apprentices close to the BFS
# dual-training stock. This is a scenario assumption, not an observed label or
# an exact BFS calibration: the survey and administrative populations differ.
DEFAULT_REPAIR_FRACTION = 0.65


def apprentice_summary(persons, weight_column=None, include_students=True):
    """Count ages15+ nationally; keep survey expansion and sample counts distinct.

    Call separately for each survey year. Summing pooled SS weights would count
    several annual populations. Population counts here are actual sample counts;
    they are not silently expanded to the full population or calibrated to BFS.
    """
    adults = persons.loc[persons.age.ge(15)]
    weights = (adults[weight_column].astype(float) if weight_column else
               pd.Series(1., index=adults.index))
    employed = adults.employed.eq(1)
    apprentice = adults.job_position.eq(50)  # SS coding retained by both models.
    result = {
        "observations": len(adults), "persons": float(weights.sum()),
        "employed": float(weights.loc[employed].sum()),
        "apprentices": float(weights.loc[apprentice].sum()),
        "apprentices_not_employed": float(weights.loc[apprentice & ~employed].sum()),
    }
    for name, mask in {
        "students": adults.is_student.eq(1) if include_students else None,
        "employed_students": employed & adults.is_student.eq(1) if include_students else None,
        "apprentice_students": apprentice & adults.is_student.eq(1) if include_students else None,
        "apprentice_nonstudents": apprentice & adults.is_student.ne(1) if include_students else None,
    }.items():
        result[name] = float(weights.loc[mask].sum()) if mask is not None else np.nan
    result["apprentice_share"] = result["apprentices"] / result["persons"] if result["persons"] else np.nan
    return result


def model_apprentice_diagnostics(survey, population, include_students=True):
    """Small aggregate table for before/after student assignment, without quotas."""
    rows = []
    for year, group in survey.groupby("survey_year", sort=True):
        rows.append(dict(source="structural_survey", survey_year=year,
                         count_unit="survey_weighted_persons", population_group="all",
                         **apprentice_summary(group, "weight")))
    rows.append(dict(source="synthetic_population", survey_year=np.nan,
                     count_unit="sample_persons", population_group="all_modelled",
                     **apprentice_summary(population, include_students=include_students)))
    # Recovered residents use ordinary models, but are outside the survey's
    # private-household frame. Report their contribution rather than concealing
    # it in the national comparison. Home-only people are excluded by the caller.
    if "is_synthetic_household" in population:
        from data.utils import coerce_boolean_series
        recovered = coerce_boolean_series(population.is_synthetic_household,
                                          name="is_synthetic_household")
        for name, mask in [("private_households", ~recovered), ("recovered", recovered)]:
            rows.append(dict(source="synthetic_population", survey_year=np.nan,
                             count_unit="sample_persons", population_group=name,
                             **apprentice_summary(population.loc[mask], include_students=include_students)))
    return pd.DataFrame(rows)


def repair_vocational_apprentices(survey, fraction=DEFAULT_REPAIR_FRACTION, seed=0):
    """Repair a weighted fraction of ambiguous vocational students per survey year.

    Programme6 includes school-based training and does not prove employment.
    Preserve explicit apprentices, valid other jobs, student labels and weights.
    Only unemployed/inactive vocational students with corresponding job codes
    enter the repair pool. Unselected people retain their original responses.

    Randomise the pool, then take the prefix nearest the requested weighted mass.
    This avoids independent Bernoulli noise dominating the national stock test.
    The residual is at most half the largest eligible survey weight. There are
    no age/canton targets, and each year gets its own reproducible random stream.
    """
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("vocational_apprentice_repair_fraction must be between 0 and 1")
    result = survey.copy()
    eligible = (survey.age.ge(15) & survey.current_education.eq(6)
                & survey.is_student.eq(1) & survey.employed.isin([2, 3])
                & survey.job_position.isin([60, 70]))
    weights = survey.weight.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Apprentice repair requires finite, positive survey weights")
    if survey.survey_year.isna().any():
        raise ValueError("Apprentice repair requires a survey year for every record")
    selected = np.zeros(len(survey), dtype=bool)
    rows = []
    # Work positionally: pooled SS inputs can have repeated DataFrame indices.
    for year, positions in survey.groupby("survey_year", sort=True).indices.items():
        pool = positions[eligible.to_numpy()[positions]]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(year)]))
        pool = rng.permutation(pool)
        prefix_weight = np.r_[0., np.cumsum(weights[pool])]
        target = fraction * prefix_weight[-1]
        count = int(np.abs(prefix_weight - target).argmin())
        selected[pool[:count]] = True
        group = survey.iloc[positions]
        explicit = group.job_position.eq(50).to_numpy()
        vocational = group.current_education.eq(6).to_numpy()
        rows.append(dict(
            survey_year=year, repair_fraction=fraction,
            weighted_persons=float(weights[positions].sum()),
            students=float(weights[positions][group.is_student.eq(1).to_numpy()].sum()),
            raw_employed=float(weights[positions][group.employed.eq(1).to_numpy()].sum()),
            raw_apprentices=float(weights[positions][explicit].sum()),
            legacy_blanket_apprentices=float(weights[positions][explicit | vocational].sum()),
            eligible_observations=len(pool), eligible_weight=float(prefix_weight[-1]),
            requested_repair_weight=float(target),
            repaired_observations=count, repaired_weight=float(weights[pool[:count]].sum()),
        ))
    result.iloc[np.flatnonzero(selected), result.columns.get_loc("employed")] = 1
    result.iloc[np.flatnonzero(selected), result.columns.get_loc("job_position")] = 50
    diagnostics = pd.DataFrame(rows)
    diagnostics["repaired_apprentices"] = diagnostics.raw_apprentices + diagnostics.repaired_weight
    diagnostics["repaired_employed"] = diagnostics.raw_employed + diagnostics.repaired_weight
    return result, diagnostics
