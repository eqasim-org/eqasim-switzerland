"""Audit donor distances and nearest-school restrictions for ages 15–17."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from data.structural_survey.structural_survey import EDUCATION_LEVELS
from synthesis.population.spatial.primary.education.locations import get_distance_weight, MIN_PROB
from synthesis.population.spatial.primary.education.secondary import SCHOOL_CODES


def configure(context):
    for stage in ["synthesis.population.enriched", "synthesis.population.activities",
                  "synthesis.population.spatial.primary.education.locations",
                  "data.microcensus.commute", "data.statent.statent"]:
        context.stage(stage)
    context.config("analysis_path")


def audit(persons, activities, locations, commute, statent, output_path):
    """Inspect saved assignments without drawing new locations or changing caches.

    Each education-active person appears once. Donor distances are a post-match
    comparison: usual anchors are now assigned before an MZ donor is selected.
    The 100-school comparison is diagnostic; it does not propose that every
    secondary school is compatible with every programme.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    active = activities.loc[activities["purpose"].eq("education"), "person_id"].unique()
    cohort = persons.loc[persons["age"].between(15, 17) & persons["person_id"].isin(active)].copy()
    cohort = cohort.merge(locations, on="person_id", validate="one_to_one")
    cohort = cohort.merge(commute[["person_id", "commute_home_distance"]].rename(
        columns={"person_id": "mz_person_id"}), on="mz_person_id", how="left", validate="many_to_one")
    # Restore GeoSeries after the pandas merge; all distances use metric LV95.
    geometry = gpd.GeoSeries(cohort["geometry"], crs=locations.crs).to_crs(2056)
    cohort["assigned_km"] = np.hypot(geometry.x - cohort["home_x"], geometry.y - cohort["home_y"]) / 1000
    cohort["donor_km"] = cohort["commute_home_distance"] / 1000
    cohort["assigned_minus_donor_km"] = cohort["assigned_km"] - cohort["donor_km"]
    # Keep the historical donor reference, but evaluate the actual survey target
    # separately once secondary distance calibration is enabled.
    cohort["target_km"] = cohort["donor_km"]
    if "education_secondary_distance_source" in cohort:
        calibrated = cohort["education_secondary_distance_source"].ne("not_applicable")
        cohort.loc[calibrated, "target_km"] = cohort.loc[calibrated, "education_secondary_target_distance_m"] / 1000.
    cohort["assigned_minus_target_km"] = cohort["assigned_km"] - cohort["target_km"]
    cohort["study_label"] = cohort["current_education"].map(EDUCATION_LEVELS).fillna("unresolved")
    for column in ["seventh_school_km", "nearest_seven_best_error_km",
                   "nearest_100_best_error_km"]:
        cohort[column] = np.nan
    cohort["donor_beyond_seventh_school"] = False
    cohort["all_seven_more_than_1km_from_donor"] = False
    cohort["all_seven_distance_weights_at_floor"] = pd.Series(pd.NA, index=cohort.index, dtype="boolean")

    # Counterfactual diagnostics retain the legacy broad secondary pool and its
    # seven-nearest-school rule, even after compatible assignment is enabled.
    # Rare tertiary classifications and age fallbacks to tertiary stay separate.
    secondary = cohort["education_assignment_group"].isin(["classified_secondary", "age_secondary"])
    schools = statent.loc[statent["education_type"].eq("secondary")].copy()
    if secondary.any() and len(schools):
        distances, _ = KDTree(schools[["x", "y"]].to_numpy()).query(
            cohort.loc[secondary, ["home_x", "home_y"]].to_numpy(), k=min(100, len(schools)))
        distances = distances / 1000
        target = cohort.loc[secondary, "donor_km"].to_numpy()[:, None]
        errors = np.abs(distances - target)
        seven = min(7, distances.shape[1])
        cohort.loc[secondary, "seventh_school_km"] = distances[:, seven - 1]
        cohort.loc[secondary, "nearest_seven_best_error_km"] = errors[:, :seven].min(axis=1)
        cohort.loc[secondary, "nearest_100_best_error_km"] = errors.min(axis=1)
        cohort.loc[secondary, "donor_beyond_seventh_school"] = target[:, 0] > distances[:, seven - 1]
        cohort.loc[secondary, "all_seven_more_than_1km_from_donor"] = errors[:, :seven].min(axis=1) > 1

        # Historical counterfactual only: donor-distance assignment is no longer
        # the production rule. Reconstruct its full group for comparison.
        full = persons.merge(locations[["person_id", "education_assignment_group"]],
                             on="person_id", validate="one_to_one")
        full = full.merge(commute[["person_id", "commute_home_distance"]].rename(
            columns={"person_id": "mz_person_id"}), on="mz_person_id", validate="many_to_one")
        tree = KDTree(schools[["x", "y"]].to_numpy())
        for group in ["classified_secondary", "age_secondary"]:
            members = full.loc[full["education_assignment_group"].eq(group)]
            if members.empty:
                continue
            distance, _ = tree.query(members[["home_x", "home_y"]].to_numpy(), k=min(7, len(schools)))
            weights = get_distance_weight(distance - members["commute_home_distance"].to_numpy()[:, None])
            at_floor = pd.Series(np.all(weights <= MIN_PROB, axis=1), index=members["person_id"])
            selected = cohort["education_assignment_group"].eq(group)
            cohort.loc[selected, "all_seven_distance_weights_at_floor"] = cohort.loc[selected, "person_id"].map(at_floor).to_numpy()

    cohort["school_code_allowed"] = pd.Series(pd.NA, index=cohort.index, dtype="boolean")
    cohort["compatible_nearest_school_km"] = np.nan
    cohort["compatible_best_distance_error_km"] = np.nan
    cohort["compatible_best_target_distance_error_km"] = np.nan
    assigned_noga = cohort["destination_id"].map(statent.set_index("enterprise_id")["noga"])
    for code, codes in SCHOOL_CODES.items():
        selected = (cohort["current_education"].eq(code) & cohort["is_student"].eq(1)
                    & cohort["education_destination_status"].eq("domestic")).fillna(False)
        if not selected.any():
            continue
        cohort.loc[selected, "school_code_allowed"] = assigned_noga.loc[selected].astype(str).isin(codes).to_numpy()
        sites = statent.loc[statent["noga"].astype(str).isin(codes), ["x", "y"]].to_numpy()
        if not len(sites):
            continue
        # Quantify the best achievable distance within the compatible pool. A
        # residual can reflect missing mixed-school sites, not just sampling.
        indices = cohort.index[selected].to_numpy()
        for start in range(0, len(indices), 256):
            rows = indices[start:start + 256]
            homes = cohort.loc[rows, ["home_x", "home_y"]].to_numpy()
            distances = np.hypot(homes[:, 0, None] - sites[None, :, 0],
                                 homes[:, 1, None] - sites[None, :, 1]) / 1000
            cohort.loc[rows, "compatible_nearest_school_km"] = distances.min(axis=1)
            cohort.loc[rows, "compatible_best_distance_error_km"] = np.abs(
                distances - cohort.loc[rows, "donor_km"].to_numpy()[:, None]).min(axis=1)
            cohort.loc[rows, "compatible_best_target_distance_error_km"] = np.abs(
                distances - cohort.loc[rows, "target_km"].to_numpy()[:, None]).min(axis=1)

    def summarise(frame, label):
        paired = frame.loc[np.isfinite(frame["donor_km"]) & frame["donor_km"].ge(0)]
        tested = paired.loc[paired["seventh_school_km"].notna()]
        return {"group": label, "persons": len(frame), "unique_donors": frame["mz_person_id"].nunique(),
                "valid_donor_distances": len(paired), "assigned_mean_km": frame["assigned_km"].mean(),
                "donor_mean_km": paired["donor_km"].mean(),
                "paired_assigned_mean_km": paired["assigned_km"].mean(),
                "mean_assigned_minus_donor_km": paired["assigned_minus_donor_km"].mean(),
                "assigned_median_km": frame["assigned_km"].median(),
                "donor_median_km": paired["donor_km"].median(),
                "assigned_p90_km": frame["assigned_km"].quantile(.9),
                "donor_p90_km": paired["donor_km"].quantile(.9),
                "assigned_p95_km": frame["assigned_km"].quantile(.95),
                "donor_p95_km": paired["donor_km"].quantile(.95),
                "mean_absolute_distance_error_km": paired["assigned_minus_donor_km"].abs().mean(),
                "target_mean_km": frame.loc[frame["target_km"].ge(0), "target_km"].mean(),
                "target_median_km": frame.loc[frame["target_km"].ge(0), "target_km"].median(),
                "mean_absolute_target_error_km": frame.loc[frame["target_km"].ge(0), "assigned_minus_target_km"].abs().mean(),
                "compatible_pool_tested": frame["school_code_allowed"].notna().sum(),
                "school_code_not_allowed": frame["school_code_allowed"].eq(False).sum(),
                "compatible_mean_best_error_km": frame["compatible_best_distance_error_km"].mean(),
                "secondary_candidates_tested": len(tested),
                "donor_beyond_seventh_school": tested["donor_beyond_seventh_school"].sum(),
                "all_seven_more_than_1km_from_donor": tested["all_seven_more_than_1km_from_donor"].sum(),
                "all_seven_distance_weights_at_floor": tested["all_seven_distance_weights_at_floor"].sum(),
                "nearest_seven_mean_best_error_km": tested["nearest_seven_best_error_km"].mean(),
                "nearest_100_mean_best_error_km": tested["nearest_100_best_error_km"].mean()}

    rows = [summarise(cohort, "all_education"), summarise(cohort.loc[cohort["is_student"].eq(1)], "students_only"),
            summarise(cohort.loc[secondary & cohort["is_student"].eq(1)], "secondary_students")]
    group_columns = ["age", "education_assignment_group", "study_label", "is_student", "education_destination_status"]
    group_columns += [name for name in ["municipality_type", "education_secondary_method", "education_secondary_distance_source"] if name in cohort]
    for column in group_columns:
        columns = [column]
        for key, frame in cohort.groupby(columns, dropna=False):
            rows.append(summarise(frame, "%s=%s" % (columns[0], key[0])))
    summary = pd.DataFrame(rows)
    columns = ["person_id", "age", "is_student", "mz_person_id", "current_education", "study_label",
               "education_destination_status", "education_assignment_group", "education_assignment_status",
               "destination_id", "donor_km", "assigned_km", "assigned_minus_donor_km",
               "seventh_school_km", "nearest_seven_best_error_km", "nearest_100_best_error_km",
               "donor_beyond_seventh_school", "all_seven_more_than_1km_from_donor",
               "all_seven_distance_weights_at_floor", "school_code_allowed",
               "compatible_nearest_school_km", "compatible_best_distance_error_km"]
    columns += ["target_km", "assigned_minus_target_km", "compatible_best_target_distance_error_km"]
    columns += [name for name in ["municipality_type", "education_secondary_method",
                "education_secondary_distance_source", "education_secondary_distance_effective_n"] if name in cohort]
    cohort[columns].to_csv(output_path / "person_diagnostics.csv", index=False)
    summary.to_csv(output_path / "summary.csv", index=False)
    schools.groupby("noga", observed=True).agg(establishments=("enterprise_id", "size"),
        employees=("number_employees", "sum")).reset_index().to_csv(output_path / "secondary_school_types.csv", index=False)
    # Retain detailed establishment codes so programme compatibility can be
    # reviewed before replacing the current broad secondary candidate pool.
    assigned_types = cohort.merge(statent[["enterprise_id", "noga"]], left_on="destination_id",
                                  right_on="enterprise_id", validate="many_to_one")
    assigned_types.groupby(["is_student", "study_label", "noga"], observed=True).size().rename(
        "persons").reset_index().to_csv(output_path / "assigned_school_types_by_study.csv", index=False)
    (output_path / "README.md").write_text(
        "# Secondary education assignment audit\n\n"
        "Ages 15–17 with an actual education activity; one record per person. Distances are straight-line home-to-education kilometres. "
        "Population means give each person equal weight; repeated donor use is reflected in donor means. "
        "These donor means differ from weighted MZ survey means. Study labels are the population's imputed classifications, "
        "not necessarily the matched donor's reported programme.\n\n"
        "Legacy candidate diagnostics cover classified_secondary and age_secondary only. They compare the former seven nearest secondary establishments "
        "with the nearest 100, using the same broad secondary pool. This is an accessibility diagnostic, not a new assignment or school compatibility model. "
        "Missing donor distances are excluded from paired statistics. Abroad and unknown classifications are reported separately. "
        "Compatibility diagnostics use the documented SCHOOL_CODES mapping, not verified programme lists.\n")
    return summary


def execute(context):
    return audit(context.stage("synthesis.population.enriched"), context.stage("synthesis.population.activities"),
                 context.stage("synthesis.population.spatial.primary.education.locations"),
                 context.stage("data.microcensus.commute")["education"], context.stage("data.statent.statent"),
                 Path(context.config("analysis_path")) / "secondary_education_assignment")
