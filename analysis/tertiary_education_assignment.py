"""Report active OD assignments and provisional cohort-normalised campus targets."""

from pathlib import Path

import pandas as pd


def configure(context):
    context.stage("synthesis.population.spatial.primary.education.locations")
    context.stage("synthesis.population.spatial.primary.education.targets")
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.sampled")
    context.stage("data.microcensus.persons")
    context.stage("data.statent.education_institutions")
    context.stage("synthesis.population.spatial.primary.education.balancing")
    # Keep the read-only stock comparison, without recovered-first cap scenarios.
    context.stage("analysis.education_student_stock")
    context.config("analysis_path")


def execute(context):
    path = Path(context.config("analysis_path")) / "tertiary_education_assignment"
    path.mkdir(parents=True, exist_ok=True)
    locations = context.stage("synthesis.population.spatial.primary.education.locations")
    activities = context.stage("synthesis.population.activities")
    active_ids = activities.loc[activities["purpose"].eq("education"), "person_id"].unique()
    # Calibration concerns annual pre-matching anchors, not today's visitors.
    # Keep both populations visible when donor matching drops an unmatched person.
    anchors = locations.copy()
    locations = anchors.loc[anchors.person_id.isin(active_ids)].copy()
    # Audit all matched people, not just classified students. Otherwise education
    # activities inherited by a nonstudent would disappear from the diagnostics.
    persons = context.stage("synthesis.population.enriched")
    if {"original_purpose", "nonstudent_education_recoded"}.issubset(activities.columns):
        # A falling campus education count must be distinguishable from dropping
        # trips. Keep the visits routed to secondary other visible in the audit.
        reported = activities.loc[activities.original_purpose.isin(
            ["education", "education_main", "education_secondary"])].merge(
                persons[["person_id", "is_student"]], on="person_id", validate="many_to_one")
        reported.groupby(["is_student", "original_purpose", "purpose", "nonstudent_education_recoded"],
                         dropna=False).agg(persons=("person_id", "nunique"),
                                           activities=("person_id", "size")).reset_index().to_csv(
            path / "education_activity_reclassification.csv", index=False)
    donors = context.stage("data.microcensus.persons")[["person_id", "employment_status"]].rename(
        columns={"person_id": "mz_person_id", "employment_status": "donor_employment_status"})
    audit = persons.loc[persons["age"].ge(15), ["person_id", "mz_person_id", "is_student"]].merge(
        donors, on="mz_person_id", how="left", validate="many_to_one")
    audit["donor_student"] = pd.to_numeric(audit["donor_employment_status"], errors="raise").isin([2, 3]).astype("Int64")
    audit.loc[audit["donor_employment_status"].isna(), "donor_student"] = pd.NA
    audit["has_education_activity"] = audit["person_id"].isin(active_ids)
    audit.groupby(["is_student", "donor_student", "has_education_activity"], dropna=False).size().rename(
        "persons").reset_index().to_csv(path / "student_donor_compatibility.csv", index=False)
    locations.groupby(["education_assignment_group", "education_assignment_status", "education_od_status"], dropna=False).size().rename(
        "persons").reset_index().to_csv(path / "assignment_status.csv", index=False)
    eligible = locations[locations["education_od_status"].ne("outside_domestic_tertiary_cohort")]
    eligible.groupby(["education_od_origin_zone_id", "education_assignment_group", "education_od_status"], dropna=False).agg(
        persons=("person_id", "size"), retained_probability_mass=("education_od_compatible_mass", "first")).reset_index().to_csv(
        path / "origin_compatibility.csv", index=False)
    assigned = anchors[anchors["education_od_status"].isin([
        "pooled_tertiary_od", "balanced_tertiary_od"])]
    assigned.groupby(["education_od_origin_zone_id", "education_od_destination_zone_id", "education_assignment_group"]).size().rename(
        "annual_students").reset_index().to_csv(path / "assigned_flows.csv", index=False)
    targets = context.stage("synthesis.population.spatial.primary.education.targets")
    diagnostics = context.stage("synthesis.population.spatial.primary.education.balancing")["diagnostics"]
    diagnostics.to_csv(path / "balancing_diagnostics.csv", index=False)
    if not diagnostics.empty:
        # These actual counts use the same supported cohort as fitted margins.
        # Full-cohort previews below also include any compatible proxy fallbacks.
        counts = assigned[assigned["education_od_status"].isin([
            "balanced_tertiary_od"])].groupby(
            ["education_assignment_group", "education_od_calibration_group"]).size().rename("assigned_persons").reset_index()
        counts = counts.rename(columns={"education_assignment_group": "study_level",
                                       "education_od_calibration_group": "calibration_group"})
        fitted = diagnostics.merge(counts, on=["study_level", "calibration_group"], how="left", validate="one_to_one")
        fitted["assigned_persons"] = fitted["assigned_persons"].fillna(0)
        fitted["assigned_minus_expected"] = fitted["assigned_persons"] - fitted["expected_students"]
        # Float margins describe the reference; integer margins describe the
        # actual assignment contract. Report both so rounding is not a mismatch.
        fitted["assigned_minus_integer_target"] = fitted["assigned_persons"] - fitted["integer_target_students"]
        fitted.to_csv(path / "calibrated_group_comparison.csv", index=False)
    for name, table in targets.items():
        table.to_csv(path / (name + ".csv"), index=False)
    registry = context.stage("data.statent.education_institutions")
    # Campus/type totals include compatible proxies too; OD flows above include
    # only successful matrix assignments. This keeps comparisons on cohort scope.
    annual = anchors.loc[anchors.education_od_status.ne("outside_domestic_tertiary_cohort")].copy()
    annual["daily_visitor"] = annual.person_id.isin(active_ids)
    measured = annual.merge(registry["establishments"][["enterprise_id", "institution_id", "campus_id", "institution_type"]],
                              left_on="destination_id", right_on="enterprise_id", validate="many_to_one")
    measured["study_level"] = measured["education_assignment_group"]
    measured.groupby(["study_level", "institution_type"]).size().rename("assigned_persons").reset_index().to_csv(
        path / "assigned_institution_types.csv", index=False)
    campus_counts = measured.groupby(["institution_id", "campus_id", "study_level"], as_index=False).agg(
        assigned_persons=("person_id", "size"), daily_assigned_persons=("daily_visitor", "sum"))
    campus_counts["daily_attendance_share"] = campus_counts.daily_assigned_persons / campus_counts.assigned_persons
    campus_counts.to_csv(path / "assigned_campuses.csv", index=False)
    comparison = targets["campus_targets"].merge(campus_counts, on=["institution_id", "campus_id", "study_level"], how="left")
    comparison["assigned_persons"] = comparison["assigned_persons"].fillna(0)
    comparison["assigned_minus_preview"] = comparison["assigned_persons"] - comparison["preview_cohort_students"]
    comparison.to_csv(path / "campus_target_comparison.csv", index=False)
    sampled = context.stage("synthesis.population.sampled")
    students = sampled.loc[sampled.is_student.eq(1)].copy()
    students["anchor_assigned"] = students.person_id.isin(anchors.person_id)
    students["matched_person"] = students.person_id.isin(persons.person_id)
    students["education_today"] = students.person_id.isin(active_ids)
    students.groupby(["study_level", "education_destination_status"], dropna=False).agg(
        students=("person_id", "size"), anchors=("anchor_assigned", "sum"),
        matched_students=("matched_person", "sum"), daily_visitors=("education_today", "sum")
        ).reset_index().to_csv(path / "student_anchor_and_attendance.csv", index=False)
    secondary = anchors.loc[anchors.education_secondary_od_status.eq("pooled_secondary_od")]
    secondary.groupby(["education_secondary_origin_zone_id", "education_secondary_destination_zone_id"]).size().rename(
        "annual_students").reset_index().to_csv(path / "secondary_assigned_flows.csv", index=False)
    (path / "README.md").write_text(
        "# Annual education anchors and actual daily attendance\n\n"
        "Every sampled student gets a usual education anchor before MZ diary matching. "
        "Campus targets, calibrated_group_comparison and assigned_flows use the annual "
        "domestic tertiary cohort. All tertiary programmes share one classification, "
        "OD matrix and teaching-site pool. Secondary has its own pooled SS OD matrix, "
        "with programme-compatible schools; children use independent MZ age/school "
        "distance draws. Foreign/unknown study and unsupported OD retain labelled proxies.\n\n"
        "BFS targets use annual institution shares including continuing/other programmes "
        "and the national higher-vocational reference. FTE estimates campus splits. "
        "Reference coverage differs from resident student stock, particularly preparation "
        "courses, so shares are modelling assumptions rather than absolute stock caps. "
        "IPF preserves origin totals and OD zeros; integer targets apply to annual anchors. "
        "daily_assigned_persons counts only actual primary education visitors; it is not "
        "compared with an annual integer target. student_anchor_and_attendance also reports "
        "student losses during matching. Nonstudent course visits remain other activities.\n",
        encoding="utf-8")
    return targets
