"""Separate student participation, study-level imputation and BFS coverage gaps."""
from pathlib import Path

import numpy as np
import pandas as pd

from data.structural_survey.structural_survey import EDUCATION_LEVELS
from synthesis.population.models.education import (
    OUTCOMES, TERTIARY_CODE, MODEL_LEVELS, fit_model, predict_probabilities, prepare_training)
from synthesis.population.models.non_private import dorm_candidate_mask, home_only_mask


DEGREE_LEVELS = ["bachelor", "master", "doctorate_habilitation"]
AGE_EDGES = [15, 18, 24, 30, 40, 50, 60, np.inf]
AGE_LABELS = ["15–17", "18–23", "24–29", "30–39", "40–49", "50–59", "60+"]


def configure(context):
    for stage in ["synthesis.population.enriched", "synthesis.population.activities",
                  "data.structural_survey.student_training", "data.structural_survey.raw",
                  "data.statent.education_enrolment"]:
        context.stage(stage)
    context.config("analysis_path")
    context.config("input_downsampling", 1.0)
    context.config("education_stock_survey_year", 2023)
    context.config("education_enrolment_year", "2023/24")
    context.config("education_level_prior_strength", 20.0)


def population_cohorts(persons):
    """Separate accommodation evidence and home-only policy from modelled behaviour."""
    recovered = persons.get("is_synthetic_household", pd.Series(False, index=persons.index)).fillna(False)
    result = pd.Series("private", index=persons.index)
    result.loc[recovered] = "recovered_ordinary"
    result.loc[dorm_candidate_mask(persons)] = "recovered_dorm_candidate"
    result.loc[home_only_mask(persons)] = "recovered_home_only"
    return result


def population_stock(persons, activities, fraction):
    """Annual student stock and daily education visits are distinct denominators."""
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Education stock audit requires a sampling fraction in (0, 1]")
    if persons.person_id.duplicated().any():
        raise ValueError("Education stock audit requires unique person IDs")
    selected = persons.loc[persons.age.ge(15)].copy()
    selected["population_cohort"] = population_cohorts(selected)
    active = activities.loc[activities.purpose.eq("education"), "person_id"].unique()
    selected["daily_visitor"] = selected.person_id.isin(active)
    selected["age_group"] = pd.cut(selected.age, AGE_EDGES, labels=AGE_LABELS, right=False)
    stock = selected.groupby(["population_cohort", "study_level", "education_destination_status", "employed"],
                             observed=True, dropna=False).agg(
        agents=("person_id", "size"), daily_agents=("daily_visitor", "sum")).reset_index()
    stock["estimated_persons"] = stock.agents / fraction
    stock["estimated_daily_visitors"] = stock.daily_agents / fraction
    stock["attendance_share"] = stock.daily_agents / stock.agents
    return selected, stock


def survey_stock(survey):
    """Do not sum pooled survey-year weights as though they describe one year.

    Country status uses exactly the training stage's interpretation. Keep its
    very small group of reported study levels with a nonstudent flag separately.
    """
    # Raw pooled years repeat their original row indices; reset before reusing
    # the country parser so label selection cannot multiply survey records.
    eligible = survey.loc[survey.age.ge(15) & survey.weight.gt(0)].copy().reset_index(drop=True)
    # This temporary flag only allows the shared country parser to inspect all
    # reports, including participation conflicts; it never changes model inputs.
    parser_input = eligible.copy()
    parser_input["is_student"] = 1
    parsed = prepare_training(parser_input)
    eligible = eligible.loc[parsed.index].copy()
    eligible["education_destination_status"] = parsed["_destination_status"]
    eligible["study_level"] = eligible.current_education.map(EDUCATION_LEVELS).fillna("unresolved")
    eligible["weight_squared"] = eligible.weight ** 2
    table = eligible.groupby(["survey_year", "study_level", "education_destination_status",
                              "is_student", "in_education", "employed"], observed=True, dropna=False).agg(
        observations=("weight", "size"), weighted_persons=("weight", "sum"),
        squared_weights=("weight_squared", "sum")).reset_index()
    table["effective_observations"] = table.weighted_persons ** 2 / table.squared_weights
    return eligible, table


def compare_degree_totals(stock, survey_table, enrolment, academic_year, survey_year):
    """Compare national sources while retaining domestic/unknown/abroad scope."""
    reference = enrolment.loc[enrolment.academic_year.eq(academic_year)
                             & enrolment.study_level.isin(DEGREE_LEVELS)]
    if reference.empty or reference.students.isna().any():
        raise ValueError("Degree stock comparison requires complete BFS reference totals")
    if reference.duplicated(["institution_type", "bfs_institution_code", "study_level"]).any():
        raise ValueError("BFS institution-level enrolment is duplicated")
    bfs = reference.groupby("study_level").students.sum()
    # Codebook 25 includes FH/PH diplomas in code 11 and university/ETH
    # licentiates/diplomas in code 12. Keep strict totals visible alongside the
    # aligned source reference. Production campus/type shares now pool programmes.
    legacy = enrolment.loc[enrolment.academic_year.eq(academic_year)
                           & enrolment.study_level.eq("legacy_degree")]
    if legacy.students.isna().any():
        raise ValueError("Degree stock comparison requires complete diploma totals")
    legacy_by_level = {"bachelor": legacy.loc[legacy.institution_type.eq("FH_PH"), "students"].sum(),
                       "master": legacy.loc[legacy.institution_type.eq("UH"), "students"].sum(),
                       "doctorate_habilitation": 0.}
    surveyed = survey_table.loc[survey_table.is_student.eq(1)]
    years = sorted(surveyed.survey_year.unique())
    if survey_year not in years:
        raise ValueError("Selected structural-survey year is unavailable")
    rows = []
    for level in DEGREE_LEVELS:
        model = stock.loc[stock.study_level.eq(level)]
        ss = surveyed.loc[surveyed.study_level.eq(level)]
        domestic = model.loc[model.education_destination_status.eq("domestic")]
        ss_domestic = ss.loc[ss.education_destination_status.eq("domestic")]
        # Reindex absent years to zero before averaging; summing all years would
        # inflate an annual comparison by the number of pooled survey years.
        annual = ss_domestic.groupby("survey_year").weighted_persons.sum().reindex(years, fill_value=0)
        row = dict(study_level=level, bfs_enrolment=float(bfs[level]),
                   bfs_codebook_legacy_enrolment=float(legacy_by_level[level]),
                   bfs_codebook_aligned_enrolment=float(bfs[level]+legacy_by_level[level]),
                   survey_reference_year=survey_year, academic_year=academic_year,
                   survey_domestic_reference_year=float(annual.loc[survey_year]),
                   survey_domestic_average_year=float(annual.mean()),
                   survey_all_countries_reference_year=float(ss.loc[ss.survey_year.eq(survey_year), "weighted_persons"].sum()),
                   model_domestic=float(domestic.estimated_persons.sum()),
                   model_all_countries=float(model.estimated_persons.sum()),
                   model_domestic_daily=float(domestic.estimated_daily_visitors.sum()))
        for cohort in ["private", "recovered_ordinary", "recovered_dorm_candidate", "recovered_home_only"]:
            row["model_domestic_" + cohort] = float(domestic.loc[domestic.population_cohort.eq(cohort), "estimated_persons"].sum())
        row["model_domestic_ordinary"] = row["model_domestic_private"] + row["model_domestic_recovered_ordinary"] + row["model_domestic_recovered_dorm_candidate"]
        row["model_ordinary_minus_survey"] = row["model_domestic_ordinary"] - row["survey_domestic_reference_year"]
        row["model_total_minus_bfs"] = row["model_domestic"] - row["bfs_enrolment"]
        row["model_total_minus_codebook_aligned_bfs"] = row["model_domestic"] - row["bfs_codebook_aligned_enrolment"]
        if stock.study_level.eq("tertiary").any():
            # Detailed synthetic programmes are deliberately no longer imputed.
            # Missing comparisons must not look like zero university enrolment.
            for column in row:
                if column.startswith("model_"):
                    row[column] = np.nan
            row["comparison_status"] = "synthetic_tertiary_pooled_no_detailed_programme_prediction"
        rows.append(row)
    return pd.DataFrame(rows)


def standardised_student_rates(persons, survey, survey_year=None, condition_employment=True):
    """Descriptive standardisation, not a refit of the student classifier.

    Apply observed survey student rates to the private population's demographic
    cells. This distinguishes a student-rate mismatch from a different age or
    employment mix. Retain explicit backoff when a geographic cell is absent.
    """
    population = persons.loc[persons.population_cohort.eq("private")].copy()
    source = survey.loc[survey.age.ge(15) & survey.weight.gt(0)].copy()
    if survey_year is not None:
        source = source.loc[source.survey_year.eq(survey_year)]
    for frame in [population, source]:
        frame["age_group"] = pd.cut(frame.age, AGE_EDGES, labels=AGE_LABELS, right=False).astype(str)
        for col in ["sex", "canton_id", "employed"]:
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype(int)
    source["student_weight"] = source.weight * source.is_student
    keys = ["age_group"] + (["employed"] if condition_employment else []) + ["sex", "canton_id"]
    profiles = []
    for depth in range(1, len(keys) + 1):
        table = source.groupby(keys[:depth], observed=True).agg(
            population_weight=("weight", "sum"), student_weight=("student_weight", "sum"),
            observations=("weight", "size"))
        profiles.append({(key if isinstance(key, tuple) else (key,)):
                         (row.student_weight / row.population_weight, row.observations)
                         for key, row in table.iterrows()})
    rows = []
    for key, group in population.groupby(keys, observed=True):
        chosen = None
        for depth in range(len(keys), 0, -1):
            chosen = profiles[depth-1].get(key[:depth])
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError("No survey student-rate support for age group %s" % key[0])
        rate, observations = chosen
        rows.append(dict(zip(keys, key), population_agents=len(group),
            observed_student_agents=int(group.is_student.sum()), expected_student_agents=len(group)*rate,
            survey_observations=observations, support_depth=depth,
            year="pooled" if survey_year is None else str(survey_year),
            employment_conditioned=condition_employment))
    return pd.DataFrame(rows)


def expected_level_counts(model, people):
    """Use probabilities only; do not redraw or overwrite anyone's study level."""
    if people.empty:
        return np.zeros(len(OUTCOMES)), np.zeros(len(OUTCOMES))
    probabilities, _, _ = predict_probabilities(model, people)
    # This SD describes independent study-level draws conditional on the saved
    # students; it is not survey or household-sampling uncertainty.
    return probabilities.sum(axis=0), np.sqrt((probabilities * (1-probabilities)).sum(axis=0))


def imputation_audit(persons, survey, fraction, reference_year, strength):
    persons = persons.copy()
    if "study_level" not in persons:
        persons["study_level"] = persons.current_education.map(EDUCATION_LEVELS).fillna("unresolved")
    modelled = persons.loc[persons.is_student.eq(1)
                           & ~persons.population_cohort.eq("recovered_home_only")]
    rows = []
    scenarios = [("pooled_current", survey, strength),
                 ("pooled_without_smoothing", survey, 0.),
                 ("reference_year_only", survey.loc[survey.survey_year.eq(reference_year)], strength)]
    for name, source, prior in scenarios:
        people = modelled
        model = fit_model(source, prior)
        expected, sd = expected_level_counts(model, people)
        for i,(code,status) in enumerate(OUTCOMES):
            # The production imputer stores unresolved outcome code 0 as NA;
            # audit it under the same outcome rather than losing those students.
            level = (people.study_level.eq("tertiary") if code == TERTIARY_CODE else
                     people.current_education.eq(code).fillna(False) if code else
                     people.current_education.isna() & ~people.study_level.eq("tertiary"))
            actual = int((level & people.education_destination_status.eq(status)).sum())
            rows.append(dict(scenario=name,current_education=code,
                study_level=MODEL_LEVELS.get(code,"unresolved"),education_destination_status=status,
                actual_agents=actual,expected_agents=float(expected[i]),draw_sd_agents=float(sd[i]),
                expected_persons=float(expected[i]/fraction),actual_persons=actual/fraction,
                draw_z_score=(actual-expected[i])/sd[i] if sd[i]>0 else np.nan,
                observations=model["training_observations"],prior_strength=prior))
    return pd.DataFrame(rows)


def write_audit(persons, activities, survey, raw, enrolment, output_path, fraction=.05,
                survey_year=2023, academic_year="2023/24", strength=20., pooled_reference=None):
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    population, stock = population_stock(persons, activities, fraction)
    observed, ss = survey_stock(survey)
    _, raw_table = survey_stock(raw)
    stock.to_csv(path/"population_stock_and_attendance.csv",index=False)
    ss.to_csv(path/"survey_stock_by_year.csv",index=False)
    raw_table.to_csv(path/"raw_survey_stock_by_year.csv",index=False)
    comparison = compare_degree_totals(stock,ss,enrolment,academic_year,survey_year)
    comparison.to_csv(path/"degree_stock_comparison.csv",index=False)
    # Detailed BFS/SS programmes remain source evidence. Synthetic tertiary is
    # deliberately pooled, so its meaningful stock comparison is a pooled total.
    tertiary = population.loc[population.is_student.eq(1) & population.study_level.eq("tertiary")]
    ss_tertiary = ss.loc[ss.is_student.eq(1) & ss.survey_year.eq(survey_year)
                         & ss.study_level.isin(["bachelor", "master", "doctorate_habilitation",
                                               "higher_professional_examination", "higher_vocational_college"])]
    bfs_total = (float(pooled_reference.loc[pooled_reference.academic_year.eq(academic_year), "students"].sum())
                 if pooled_reference is not None else np.nan)
    pd.DataFrame([dict(study_group="tertiary", survey_year=survey_year, academic_year=academic_year,
        model_all_countries=len(tertiary)/fraction,
        model_domestic=int(tertiary.education_destination_status.eq("domestic").sum())/fraction,
        model_domestic_daily=int((tertiary.education_destination_status.eq("domestic") & tertiary.daily_visitor).sum())/fraction,
        ss_all_countries=ss_tertiary.weighted_persons.sum(),
        ss_domestic=ss_tertiary.loc[ss_tertiary.education_destination_status.eq("domestic"), "weighted_persons"].sum(),
        bfs_all_published_programmes=bfs_total,
        coverage_note="BFS includes university continuing/other and higher vocational programmes; institution and resident frames differ")
        ]).to_csv(path/"pooled_tertiary_stock_comparison.csv", index=False)
    # Keep all BFS level categories visible, including the continuing-education
    # categories that cannot be allocated to survey codes from these data alone.
    enrolment.loc[enrolment.academic_year.eq(academic_year)].groupby(
        ["institution_type","bfs_study_level_label","study_level"],dropna=False).students.sum().rename(
            "enrolled_students").reset_index().to_csv(path/"bfs_degree_definition_bridge.csv",index=False)
    diagnostics = population.groupby(["population_cohort","age_group","employed"],observed=True).agg(
        population_agents=("person_id","size"),student_agents=("is_student","sum"),
        education_visitors=("daily_visitor","sum")).reset_index()
    diagnostics["student_share"] = diagnostics.student_agents/diagnostics.population_agents
    diagnostics.to_csv(path/"population_student_rates.csv",index=False)
    rates = pd.concat([standardised_student_rates(population,survey,year,employment)
                       for year in [None,survey_year] for employment in [False,True]],ignore_index=True)
    rates.to_csv(path/"standardised_student_cells.csv",index=False)
    rates.groupby(["year","employment_conditioned","age_group"],observed=True).agg(
        population_agents=("population_agents","sum"),observed_student_agents=("observed_student_agents","sum"),
        expected_student_agents=("expected_student_agents","sum")).reset_index().to_csv(
        path/"standardised_student_rates.csv",index=False)
    imputation = imputation_audit(population,survey,fraction,survey_year,strength)
    imputation.to_csv(path/"level_imputation_counterfactuals.csv",index=False)
    # Directly inspect work/study overlap and the training gate, including rare
    # explicit ongoing degrees accompanied by a zero student flag.
    ss.loc[ss.study_level.isin(DEGREE_LEVELS)].to_csv(path/"degree_employment_participation.csv",index=False)
    # Already holding the reported degree can indicate a second degree or a
    # broader programme category. It is evidence to inspect, not an exclusion.
    observed.loc[observed.current_education.isin([11,12,13])].groupby(
        ["survey_year","study_level","education_destination_status","highest_completed_education"],
        dropna=False).agg(observations=("weight","size"),weighted_persons=("weight","sum")).reset_index().to_csv(
            path/"survey_current_and_completed_education.csv",index=False)
    # Generic communal-building proximity does not verify student occupancy.
    # Report candidate participation without imposing student status.
    candidates = population.loc[population.population_cohort.eq("recovered_dorm_candidate")]
    if "non_private_residence_evidence" in candidates:
        candidates.groupby(["non_private_residence_evidence","age_group","study_level"],observed=True,dropna=False).agg(
            agents=("person_id","size"),daily_visitors=("daily_visitor","sum")).reset_index().to_csv(
                path/"recovered_residence_evidence.csv",index=False)
    observed.loc[observed.current_education.isin([11,12,13])].groupby(
        ["survey_year","study_level","education_destination_status","institution_type"],dropna=False).agg(
            observations=("weight","size"),weighted_persons=("weight","sum")).reset_index().to_csv(
                path/"degree_institution_evidence.csv",index=False)
    population.loc[population.study_level.isin(DEGREE_LEVELS + ["tertiary"])].groupby(
        ["study_level","education_destination_status","population_cohort","education_imputation_method"],dropna=False
        ).agg(agents=("person_id","size"),median_effective_n=("education_imputation_effective_n","median")).reset_index().to_csv(
            path/"population_tertiary_imputation_support.csv",index=False)
    (path/"README.md").write_text(
        "# Student stock and education classification audit\n\n"
        "This audit reads saved population and survey outputs. It does not change student flags, "
        "employment, diaries, locations or quotas. Population counts are estimates obtained by dividing "
        "the actual sampled agents by the configured sampling fraction. Stock counts cover ages 15+ "
        "whether or not an education activity occurs that day; attendance is reported separately.\n\n"
        "BFS UH+FH/PH institution enrolment is compared by degree and academic year. SS current school "
        "country/zone defines domestic, abroad and unknown status. Unknown is not silently domestic. "
        "BFS educational origin does not establish current residence. These sources have different "
        "frames, reference dates and definitions; their differences are not automatically model errors. "
        "Survey years are exported separately. The pooled domestic count is the mean of annual totals, "
        "not their sum. Raw and processed survey counts expose preprocessing effects.\n\n"
        "Private population student rates are descriptively standardised using observed SS age-group, "
        "sex and canton rates, with and without conditioning on employment. Pooled rates use source "
        "weights; reference-year-only rates are also reported. Unsupported cells back off explicitly. "
        "This comparison is not a validation of the CatBoost classifier's held-out accuracy. Recovered "
        "and home-only residents are kept separate from private-household predictions.\n\n"
        "Level-imputation counterfactuals compute expected probabilities on the saved student cohort "
        "using the existing conditional imputer: current pooled smoothing, no smoothing, reference "
        "year only, for all modelled residents. No outcomes are redrawn. draw_sd_agents measures "
        "independent conditional imputation draws, not sampling or survey uncertainty. Codes 11/12 "
        "include FH/PH/university legacy diplomas; code 13 includes habilitation as documented in the "
        "supplied BFS codebook. Strict BFS totals and a separate codebook-aligned legacy-diploma bridge "
        "are both reported as source evidence. Synthetic tertiary study is pooled: "
        "detailed model programme counts are unavailable, not zero. The pooled reference "
        "includes university continuing/other and national higher vocational programmes, "
        "whose student and preparation-course coverage differs from SS. "
        "The work/study table includes explicit degree reports with a zero student flag, so they are "
        "visible even though the existing education imputer excludes them.\n",
        encoding="utf-8")
    return comparison


def execute(context):
    return write_audit(context.stage("synthesis.population.enriched"),
        context.stage("synthesis.population.activities"),context.stage("data.structural_survey.student_training"),
        context.stage("data.structural_survey.raw"),context.stage("data.statent.education_enrolment")["enrolment"],
        Path(context.config("analysis_path"))/"education_student_stock",context.config("input_downsampling"),
        context.config("education_stock_survey_year"),context.config("education_enrolment_year"),
        context.config("education_level_prior_strength"),
        context.stage("data.statent.education_enrolment")["pooled_enrolment"])
