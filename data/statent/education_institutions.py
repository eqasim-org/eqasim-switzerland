"""Anonymous tertiary institution groups, provisional campuses and compatibility."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from data.structural_survey.structural_survey import TERTIARY_INSTITUTIONS


DEFAULT_CAMPUS_DISTANCE = 500.0
COMPATIBLE_LEVELS = {
    "UH": ["bachelor", "master", "doctorate_habilitation"],
    "FH_PH": ["bachelor", "master"],
    "higher_vocational": ["higher_professional_examination", "higher_vocational_college"],
}


def configure(context):
    context.stage("data.statent.statent")
    context.config("education_campus_cluster_distance", DEFAULT_CAMPUS_DISTANCE)


def build_registry(df_statent, campus_distance=DEFAULT_CAMPUS_DISTANCE):
    """Group anonymous enterprise sites into provisional campuses and study pools.

    Spatial clusters identify candidate campuses, not verified institution names.
    Employees remain employment data; student capacity is deliberately unknown.
    """
    if not np.isfinite(campus_distance) or campus_distance <= 0:
        raise ValueError("Education campus clustering distance must be finite and positive")
    selected = df_statent["noga"].isin(TERTIARY_INSTITUTIONS) | df_statent["enterprise_noga"].isin(TERTIARY_INSTITUTIONS)
    fields = ["enterprise_id", "statent_enterprise_id", "statent_local_id", "noga", "enterprise_noga",
              "x", "y", "zone_id", "zone_municipality_id", "number_employees", "number_employees_fte", "statent_year"]
    sites = pd.DataFrame(df_statent.loc[selected, fields]).copy().reset_index(drop=True)
    if sites[["statent_enterprise_id", "statent_local_id"]].isna().any().any():
        raise ValueError("Education registry requires source enterprise and establishment identifiers")
    if sites["statent_local_id"].duplicated().any():
        raise ValueError("Education establishment identifiers must be unique")
    if not np.isfinite(sites[["x", "y"]].to_numpy()).all():
        raise ValueError("Education registry requires finite establishment coordinates")

    local_type = sites["noga"].map(TERTIARY_INSTITUTIONS)
    enterprise_type = sites["enterprise_noga"].map(TERTIARY_INSTITUTIONS)
    sites["institution_type"] = local_type.fillna(enterprise_type)
    sites["institution_type_source"] = np.where(local_type.notna(), "establishment", "enterprise")
    sites["institution_type_conflict"] = local_type.notna() & enterprise_type.notna() & local_type.ne(enterprise_type)
    sites["primary_tertiary_site"] = local_type.notna()
    sites["eligible_teaching_site"] = sites["primary_tertiary_site"] & ~sites["institution_type_conflict"]
    # Research establishments within a UH enterprise can host doctoral research.
    # This is nominal affiliation evidence, not verified programme/site coverage.
    affiliated_research = sites["noga"].str.startswith("721") & sites["enterprise_noga"].eq("854201")
    sites["eligible_doctoral_site"] = (sites["eligible_teaching_site"] & sites["institution_type"].eq("UH")) | affiliated_research
    sites["eligible_doctoral_site"] &= ~sites["institution_type_conflict"]
    # Keep downstream enterprise_id unchanged; registry IDs use the stable source IDs.
    sites["institution_id"] = "STATENT_ENTERPRISE_" + sites["statent_enterprise_id"].astype(str)
    sites["campus_id"] = ""
    for institution_id, group in sites.groupby("institution_id", sort=True):
        group = group.sort_values("statent_local_id")
        labels = DBSCAN(eps=campus_distance, min_samples=1).fit_predict(group[["x", "y"]])
        for label in np.unique(labels):
            members = group.index[labels == label]
            anchor = sites.loc[members, "statent_local_id"].min()
            sites.loc[members, "campus_id"] = institution_id + "_CAMPUS_" + str(anchor)

    sites["student_capacity"] = pd.Series(pd.NA, index=sites.index, dtype="Float64")
    institutions = sites.groupby(["institution_id", "statent_enterprise_id"]).agg(
        establishments=("statent_local_id", "size"), campuses=("campus_id", "nunique"),
        primary_sites=("primary_tertiary_site", "sum"), eligible_sites=("eligible_teaching_site", "sum"),
        employees=("number_employees", "sum"), employees_fte=("number_employees_fte", "sum"),
        institution_types=("institution_type", lambda x: ";".join(sorted(x.unique()))),
        type_conflicts=("institution_type_conflict", "sum"),
    ).reset_index()
    institutions["institution_name"] = pd.NA
    institutions["name_verified"] = False
    institutions["student_capacity"] = pd.Series(pd.NA, index=institutions.index, dtype="Float64")

    campuses = sites.groupby(["institution_id", "campus_id"]).agg(
        establishments=("statent_local_id", "size"), primary_sites=("primary_tertiary_site", "sum"),
        eligible_sites=("eligible_teaching_site", "sum"), x=("x", "mean"), y=("y", "mean"),
        min_x=("x", "min"), max_x=("x", "max"), min_y=("y", "min"), max_y=("y", "max"),
        destination_zones=("zone_id", "nunique"), municipality_zones=("zone_municipality_id", "nunique"),
        employees=("number_employees", "sum"), employees_fte=("number_employees_fte", "sum"),
        institution_types=("institution_type", lambda x: ";".join(sorted(x.unique()))),
        type_conflicts=("institution_type_conflict", "sum"),
    ).reset_index()
    campuses["extent_m"] = np.hypot(campuses["max_x"] - campuses["min_x"], campuses["max_y"] - campuses["min_y"])
    campuses["grouping_review"] = campuses["extent_m"].gt(2 * campus_distance) | campuses["type_conflicts"].gt(0)
    campuses["student_capacity"] = pd.Series(pd.NA, index=campuses.index, dtype="Float64")

    # A campus can cross zone boundaries; each establishment keeps its own zone.
    campus_zones = sites.groupby(["institution_id", "campus_id", "zone_id", "zone_municipality_id"]).agg(
        establishments=("statent_local_id", "size"), eligible_sites=("eligible_teaching_site", "sum"),
    ).reset_index()
    compatibility = []
    for row in sites[sites["eligible_teaching_site"]].itertuples():
        for level in COMPATIBLE_LEVELS[row.institution_type]:
            compatibility.append({"enterprise_id": row.enterprise_id, "statent_local_id": row.statent_local_id,
                                  "institution_id": row.institution_id, "campus_id": row.campus_id,
                                  "zone_id": row.zone_id, "zone_municipality_id": row.zone_municipality_id,
                                  "study_level": level, "institution_type": row.institution_type,
                                  "evidence": "local_noga_nominal_compatibility"})
    for row in sites[sites["eligible_doctoral_site"] & ~sites["eligible_teaching_site"]].itertuples():
        compatibility.append({"enterprise_id": row.enterprise_id, "statent_local_id": row.statent_local_id,
                              "institution_id": row.institution_id, "campus_id": row.campus_id,
                              "zone_id": row.zone_id, "zone_municipality_id": row.zone_municipality_id,
                              "study_level": "doctorate_habilitation", "institution_type": row.institution_type,
                              "evidence": "affiliated_uh_research_nominal_compatibility"})
    compatibility = pd.DataFrame(compatibility, columns=[
        "enterprise_id", "statent_local_id", "institution_id", "campus_id", "zone_id",
        "zone_municipality_id", "study_level", "institution_type", "evidence"])
    return {"establishments": sites, "institutions": institutions, "campuses": campuses,
            "campus_zones": campus_zones, "compatibility": compatibility}


def execute(context):
    return build_registry(context.stage("data.statent.statent"),
                          context.config("education_campus_cluster_distance"))


def regular_compatibility(registry, rules):
    """Apply reviewed regular-destination exclusions, retaining placement evidence."""
    keys = ["statent_local_id", "study_level"]
    if rules.duplicated(keys).any():
        raise ValueError("Duplicate establishment/study-level eligibility rule")
    if not rules["statent_local_id"].isin(registry["establishments"]["statent_local_id"]).all():
        raise ValueError("Regular eligibility rule references an unknown establishment")
    if not rules["regular_eligible"].isin([True, False]).all():
        raise ValueError("Regular eligibility must be boolean")
    compatibility = registry["compatibility"].merge(rules[keys + ["regular_eligible"]],
                                                    on=keys, how="left", validate="one_to_one")
    requested = rules.loc[rules["regular_eligible"].eq(True)]
    if len(requested.merge(registry["compatibility"][keys], on=keys)) != len(requested):
        raise ValueError("An inclusion rule requires supported nominal compatibility")
    return compatibility[compatibility["regular_eligible"].ne(False)].drop(columns="regular_eligible")


def pooled_compatibility(registry, rules):
    """One tertiary teaching pool, without degree-specific assignment margins.

    Research-only sites are not regular teaching campuses for an undifferentiated
    student. Reviewed sites eligible for at least one teaching programme remain;
    exclusions covering every programme (e.g. a conference centre) stay excluded.
    """
    compatible = regular_compatibility(registry, rules)
    teaching = registry["establishments"].loc[
        registry["establishments"].eligible_teaching_site, "enterprise_id"]
    result = compatible.loc[compatible.enterprise_id.isin(teaching)].copy()
    result["study_level"] = "tertiary"
    return result.drop_duplicates(["enterprise_id", "study_level"]).reset_index(drop=True)
