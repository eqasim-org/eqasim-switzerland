"""Comparable usual work/education commute distances for MZ donor matching."""

import numpy as np
import pandas as pd


AGE_EDGES = [-np.inf, 15, 18, 24, 30, 40, 50, 60, np.inf]


def _valid_distance(values):
    values = pd.to_numeric(values, errors="coerce")
    return values.where(np.isfinite(values) & values.ge(0))


def combine_distances(frame):
    """Use the longer available commute; -1 means neither is known, never zero."""
    result = frame.copy()
    work = _valid_distance(result.commute_distance_work)
    education = _valid_distance(result.commute_distance_education)
    result["commute_distance"] = pd.concat([work, education], axis=1).max(axis=1).fillna(-1.)
    return result


def population_commutes(persons, work_locations, education_locations=None):
    """Attach pre-matching anchors without changing population identity or order."""
    work = work_locations[["person_id", "work_location_type", "commute_distance"]].rename(
        columns={"commute_distance":"commute_distance_work"})
    result = persons.merge(work, on="person_id", how="left", validate="one_to_one")
    result["commute_distance_education"] = np.nan
    result["education_distance_source"] = "not_student"
    if education_locations is not None:
        education = education_locations.set_index("person_id")
        if not education.index.is_unique:
            raise ValueError("Matching requires one education anchor per student")
        result["commute_distance_education"] = result.person_id.map(education.commute_distance)
        student = result.is_student.eq(1)
        if result.loc[student, "commute_distance_education"].isna().any():
            raise ValueError("A student has no education distance before diary matching")
        result.loc[student, "education_distance_source"] = "assigned_usual_education_anchor"
        result.loc[~student, "commute_distance_education"] = np.nan
    return combine_distances(result)


def donor_commutes(persons, education_diaries, seed=0, impute_missing=True):
    """Reported education address first, diary fallback, then flagged peer draws.

    The reported AU_* location is independent of today's education attendance.
    Remaining unknown student distances can draw weighted age/employment peers;
    this changes a matching predictor, never the donor's observed activity chain.
    Every fallback keeps its provenance for attendance-bias validation.
    """
    result = persons.copy()
    if result.person_id.duplicated().any() or education_diaries.person_id.duplicated().any():
        raise ValueError("MZ usual commutes require unique respondent identifiers")
    student = result.employment_status.isin([2, 3])
    result["commute_distance_work"] = _valid_distance(result.commute_distance).where(result.employed.eq(1))
    reported = result.get("education_commute_distance", pd.Series(np.nan, index=result.index))
    result["commute_distance_education"] = _valid_distance(reported).where(student)
    result["education_distance_source"] = np.where(student, "unknown", "not_student")
    result.loc[student & result.commute_distance_education.notna(), "education_distance_source"] = "reported_education_place"
    diary = result.person_id.map(education_diaries.set_index("person_id").commute_home_distance)
    # Attendance is observed in the diary, even when its coordinates are missing.
    # Imputing a usual distance must never manufacture an education visit.
    result["education_diary_attendance"] = result.person_id.isin(education_diaries.person_id)
    missing = student & result.commute_distance_education.isna()
    valid = missing & _valid_distance(diary).notna()
    result.loc[valid, "commute_distance_education"] = diary.loc[valid]
    result.loc[valid, "education_distance_source"] = "diary_education_place_fallback"
    if impute_missing:
        age = pd.cut(result.age, AGE_EDGES, right=False, labels=False)
        # A reported overseas school can be a genuine multi-thousand-km usual
        # location. Keep it for its respondent, but do not draw it as a domestic
        # missing commute. Diary fallback peers must have a domestic-only diary.
        country = pd.to_numeric(result.get("education_country_code",
            pd.Series(8100, index=result.index)), errors="coerce")
        crosses_border = result.get("is_crossing_the_border", pd.Series(False, index=result.index)).fillna(True)
        domestic_reference = country.eq(8100) | (
            result.education_distance_source.eq("diary_education_place_fallback") & ~crosses_border)
        reference = result.loc[student & result.commute_distance_education.notna()
                               & domestic_reference & result.person_weight.gt(0)
                               & np.isfinite(result.person_weight)].copy()
        reference["_age_band"] = age.loc[reference.index]
        rng = np.random.RandomState(seed)
        missing = result.loc[student & result.commute_distance_education.isna()].copy()
        missing["_age_band"] = age.loc[missing.index]
        for (band, employed), members in missing.groupby(["_age_band", "employed"], sort=True):
            peers = reference.loc[reference._age_band.eq(band) & reference.employed.eq(employed)]
            method = "imputed_mz_age_employment"
            if peers.empty:
                peers = reference.loc[reference._age_band.eq(band)]
                method = "imputed_mz_age"
            if peers.empty:
                peers = reference
                method = "imputed_mz_all_students"
            if peers.empty:
                continue
            # Stable ID ordering keeps draws independent of DataFrame row order.
            selected = members.sort_values("person_id").index
            peers = peers.sort_values("person_id")
            weights = peers.person_weight.to_numpy(dtype=float)
            result.loc[selected, "commute_distance_education"] = rng.choice(
                peers.commute_distance_education.to_numpy(), len(selected), p=weights/weights.sum())
            result.loc[selected, "education_distance_source"] = method
    return combine_distances(result)


def write_diagnostics(donors, population, output_path):
    """Keep provenance and distance distributions inspectable without a new model."""
    from pathlib import Path
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    for name, people in [("mz", donors), ("population", population)]:
        frame = people.copy()
        frame["age_band"] = pd.cut(frame.age, AGE_EDGES, right=False)
        groups = ["age_band", "employed", "education_distance_source"]
        summary = frame.groupby(groups, observed=True).agg(
            persons=("commute_distance", "size"),
            mean_work_m=("commute_distance_work", "mean"),
            median_work_m=("commute_distance_work", "median"),
            mean_education_m=("commute_distance_education", "mean"),
            median_education_m=("commute_distance_education", "median"),
            mean_combined_m=("commute_distance", "mean")).reset_index()
        if name == "mz":
            frame["education_attendance_weight"] = frame.person_weight * frame.education_diary_attendance
            attendance = frame.groupby(groups, observed=True).agg(
                education_visitors=("education_diary_attendance", "sum"),
                attendance_share=("education_diary_attendance", "mean"),
                person_weight=("person_weight", "sum"),
                education_attendance_weight=("education_attendance_weight", "sum")).reset_index()
            attendance["weighted_attendance_share"] = attendance.education_attendance_weight / attendance.person_weight
            summary = summary.merge(attendance, on=groups, validate="one_to_one")
        summary.to_csv(
                path / (name + "_matching_commute_diagnostics.csv"), index=False)
