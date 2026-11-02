"""Annual pooled tertiary campus shares, independent of diary attendance."""

import numpy as np
import pandas as pd

from synthesis.population.spatial.primary.education.tertiary import domestic_cohort


def configure(context):
    context.stage("synthesis.population.sampled")
    context.stage("data.statent.education_enrolment")
    context.stage("data.statent.education_campus_weights")
    context.config("education_enrolment_year", "2023/24")


def validate(context):
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(__file__).with_name("tertiary.py").read_bytes()).hexdigest()


def build_target_preview(persons, enrolment, campus_weights, academic_year):
    """Normalise annual BFS shares to all domestic tertiary students in the sample.

    Includes university continuing/other programmes and national higher vocational
    enrolment. This is an institution-frame reference, not a verified Swiss-resident
    stock cap. Daily attendance is a later diary outcome and validation measure.
    """
    cohort = domestic_cohort(persons)
    source = enrolment.loc[enrolment.academic_year.eq(academic_year)].copy()
    keys = ["institution_type", "bfs_institution_code"]
    if source.empty or source.duplicated(keys).any():
        raise ValueError("Pooled target construction requires unique annual institution totals")
    if set(source.institution_type) != {"UH", "FH_PH", "higher_vocational"}:
        raise ValueError("Pooled reference requires UH, FH/PH and higher-vocational totals")
    if not np.isfinite(source.students).all() or source.students.lt(0).any() or source.students.sum() <= 0:
        raise ValueError("Pooled target reference has invalid enrolment counts")
    total = source.students.sum()
    types = source.groupby("institution_type", as_index=False).students.sum().rename(
        columns={"students":"bfs_type_enrolment"})
    types["study_level"] = "tertiary"
    types["bfs_level_enrolment"] = total
    types["cohort_persons"] = len(cohort)
    types["enrolment_share"] = types.bfs_type_enrolment / total
    types["preview_cohort_students"] = len(cohort) * types.enrolment_share
    preview = campus_weights.loc[campus_weights.academic_year.eq(academic_year)].copy()
    if preview.duplicated(["institution_id", "campus_id"]).any():
        raise ValueError("Pooled campus enrolment must be counted once per campus")
    preview["bfs_level_enrolment"] = total
    preview["cohort_persons"] = len(cohort)
    preview["preview_cohort_students"] = len(cohort) * preview.estimated_enrolled_students / total
    for frame in [types, preview]:
        frame["ready_for_assignment"] = False
        frame["coverage_assumption"] = "Equal Swiss-residence shares across pooled tertiary institutions; higher-vocational coverage incomplete"
        frame["target_method"] = "annual_pooled_bfs_enrolment_share"
        frame["target_scope"] = "all_classified_domestic_students_not_daily_visitors"
    counts = pd.DataFrame({"study_level":["tertiary"], "cohort_persons":[len(cohort)],
                           "target_status":["annual_pooled_reference"]})
    return {"cohort_counts":counts, "institution_type_targets":types, "campus_targets":preview}


def execute(context):
    return build_target_preview(context.stage("synthesis.population.sampled"),
        context.stage("data.statent.education_enrolment")["pooled_enrolment"],
        context.stage("data.statent.education_campus_weights")["campus_weights"],
        context.config("education_enrolment_year"))
