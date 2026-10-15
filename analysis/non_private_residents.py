"""Inspect recovered residence evidence, ordinary behaviour and home-only invariants."""
from pathlib import Path

import pandas as pd

from synthesis.population.models.non_private import dorm_candidate_mask, home_only_mask


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.spatial.primary.education.locations")
    context.config("analysis_path")


def execute(context):
    path = Path(context.config("analysis_path")) / "non_private_residents"
    path.mkdir(parents=True, exist_ok=True)
    persons = context.stage("synthesis.population.enriched").copy()
    activities = context.stage("synthesis.population.activities")
    education = context.stage("synthesis.population.spatial.primary.education.locations")
    candidates, home = dorm_candidate_mask(persons), home_only_mask(persons)
    persons["residence_scenario"] = "ordinary"
    persons.loc[persons.is_synthetic_household, "residence_scenario"] = "recovered_ordinary"
    persons.loc[candidates, "residence_scenario"] = "recovered_dorm_candidate"
    persons.loc[home, "residence_scenario"] = "home_only_not_modelled"
    persons["age_group"] = pd.cut(persons.age, [0, 15, 18, 24, 30, 39, 50, 65, 75, 85, 200],
                                   right=False, labels=["0–14", "15–17", "18–23", "24–29", "30–38",
                                                        "39–49", "50–64", "65–74", "75–84", "85+"])
    active = activities.loc[activities.purpose.eq("education"), "person_id"].unique()
    persons["education_activity_today"] = persons.person_id.isin(active)
    persons.groupby(["residence_scenario", "age_group"], observed=True).agg(
        persons=("person_id", "size"), students=("is_student", "sum"),
        daily_education_visitors=("education_activity_today", "sum")).reset_index().to_csv(
        path / "cohorts_and_attendance.csv", index=False)
    home_ids = persons.loc[home, "person_id"]
    home_activities = activities.loc[activities.person_id.isin(home_ids)]
    if (len(home_activities) != len(home_ids) or home_activities.person_id.duplicated().any()
            or not home_activities.purpose.eq("home").all()
            or not persons.loc[home, "mz_person_id"].eq(-1).all()):
        raise ValueError("Home-only residents must have no donor and exactly one home activity")
    # Candidate accommodation does not force student status, employment, study
    # level or campus. Report those outcomes so a later dorm-swap decision can
    # use actual modelled participation rather than assumed occupancy.
    persons.loc[candidates].groupby(["age_group", "is_student", "employed", "study_level"],
        observed=True, dropna=False).size().rename("persons").reset_index().to_csv(
            path / "dorm_candidate_behaviour.csv", index=False)
    selected = education.loc[education.person_id.isin(persons.loc[candidates & persons.education_activity_today, "person_id"])]
    selected.groupby(["education_assignment_group", "education_assignment_status"], dropna=False).size().rename(
        "daily_visitors").reset_index().to_csv(path / "dorm_candidate_assignment_status.csv", index=False)
    return {"dorm_candidates": int(candidates.sum()), "home_only": int(home.sum()),
            "dorm_candidate_students": int((candidates & persons.is_student.eq(1)).sum()),
            "dorm_candidate_education_visitors": int((candidates & persons.education_activity_today).sum())}
