"""Provisional study-level campus enrolment shares using eligible-site FTE."""

import numpy as np
import pandas as pd
import hashlib
from pathlib import Path


DEFAULT_ELIGIBILITY_RULES = Path("education_enrolment/education_site_eligibility.csv")


def configure(context):
    context.config("data_path")
    context.stage("data.statent.education_institutions")
    context.stage("data.statent.education_enrolment")
    context.config("education_site_eligibility", str(Path(context.config("data_path")) / DEFAULT_ELIGIBILITY_RULES))


def validate(context):
    """Track eligibility-rule contents, not only their unchanged file path."""
    return hashlib.sha256(Path(context.config("education_site_eligibility")).read_bytes()).hexdigest()


def estimate_campus_weights(registry, linked_enrolment, include_doctoral_research=True, eligibility_rules=None):
    """Split each linked BFS level total by eligible campus FTE within its institution.

    Programme rules can exclude a site for one level while retaining another.
    Input totals use the codebook-aligned reference, with diplomas already
    combined into Bachelor/Master once per institution before FTE splitting.
    Doctoral research sites are a separate assumption. Missing eligible FTE stays
    unresolved rather than redistributing its students over the remaining sites.
    """
    totals = linked_enrolment[linked_enrolment["model_study_level"]
                             & linked_enrolment["match_status"].isin(["proposed", "verified"])].copy()
    keys = ["institution_type", "bfs_institution_code", "academic_year", "study_level"]
    if totals.duplicated(keys).any():
        raise ValueError("Campus weighting requires one enterprise per BFS institution; avoid duplicating totals")
    if totals.groupby("statent_enterprise_id")["bfs_institution_code"].nunique().gt(1).any():
        raise ValueError("A STATENT enterprise cannot receive multiple institution totals")
    sites = registry["establishments"]
    if eligibility_rules is not None:
        if eligibility_rules.duplicated(["statent_local_id", "study_level"]).any():
            raise ValueError("Duplicate establishment/study-level eligibility rule")
        if not eligibility_rules["statent_local_id"].isin(sites["statent_local_id"]).all():
            raise ValueError("Eligibility rule references an unknown establishment")
        if not eligibility_rules["study_level"].isin(["bachelor", "master", "doctorate_habilitation"]).all():
            raise ValueError("Eligibility rule has an unsupported study level")
        if eligibility_rules[["decision_kind", "evidence_url", "note"]].isna().any().any():
            raise ValueError("Eligibility decisions require evidence and an explicit decision kind")
        for column in ["regular_eligible", "placement_eligible"]:
            if not eligibility_rules[column].isin([True, False]).all():
                raise ValueError("Eligibility decisions must be boolean")
    compatibility = registry["compatibility"]
    if not include_doctoral_research:
        compatibility = compatibility[compatibility["evidence"].eq("local_noga_nominal_compatibility")]
    if compatibility.duplicated(["statent_local_id", "study_level"]).any():
        raise ValueError("Duplicate site/study-level compatibility would duplicate FTE")
    decisions, campuses, summaries = [], [], []
    for total in totals.itertuples():
        group = sites[sites["statent_enterprise_id"].eq(total.statent_enterprise_id)].copy()
        if group.empty:
            raise ValueError("Enrolment link has no retained STATENT establishments")
        permitted = compatibility[compatibility["study_level"].eq(total.study_level)]
        group = group.merge(permitted[["statent_local_id", "evidence"]], on="statent_local_id", how="left",
                            validate="one_to_one")
        group["eligible_for_level"] = group["evidence"].notna()
        group["eligibility_reason"] = group["evidence"].fillna("no_nominal_compatibility_for_regular_destination")
        group["placement_eligible"] = False
        group["decision_kind"] = "nominal_noga_assumption"
        group["decision_source"] = ""
        group["decision_note"] = "Programme coverage and attendance frequency remain unverified"
        if eligibility_rules is not None:
            overrides = eligibility_rules[eligibility_rules["study_level"].eq(total.study_level)].set_index("statent_local_id")
            matched = group["statent_local_id"].isin(overrides.index)
            for target, source in [("eligible_for_level", "regular_eligible"), ("placement_eligible", "placement_eligible"),
                                   ("decision_kind", "decision_kind"), ("decision_source", "evidence_url"), ("decision_note", "note")]:
                if matched.any():
                    values = group.loc[matched, "statent_local_id"].map(overrides[source])
                    group.loc[matched, target] = values.to_numpy(dtype=bool) if target in ["eligible_for_level", "placement_eligible"] else values.to_numpy()
            group.loc[matched, "eligibility_reason"] = "reviewed_regular_destination_assumption"
        group.loc[group["institution_type_conflict"], "eligibility_reason"] = "institution_type_conflict_review"
        group.loc[group["institution_type_conflict"], "eligible_for_level"] = False
        fte = pd.to_numeric(group["number_employees_fte"], errors="raise")
        eligible = group["eligible_for_level"]
        invalid = eligible & (fte.isna() | ~np.isfinite(fte.fillna(0)) | fte.lt(0).fillna(False))
        # Never silently redistribute the missing FTE of an eligible site.
        group["eligible_fte"] = fte.where(eligible, 0.0)
        denominator = group.loc[eligible, "eligible_fte"].sum() if not invalid.any() else np.nan
        status = "fte_proxy_estimate"
        if invalid.any():
            status = "unresolved_invalid_eligible_fte"
        elif denominator <= 0:
            status = "unresolved_no_positive_eligible_fte"
        elif pd.isna(total.students):
            status = "unresolved_missing_enrolment"
        supported = status == "fte_proxy_estimate"
        for column, value in [("study_level", total.study_level), ("institution_name", total.institution_name),
                              ("match_status", total.match_status), ("academic_year", total.academic_year),
                              ("bfs_institution_code", total.bfs_institution_code), ("estimate_status", status)]:
            group[column] = value
        group["eligibility_assumed"] = True
        decisions.append(group[["institution_name", "institution_id", "statent_enterprise_id", "statent_local_id",
                                "campus_id", "study_level", "academic_year", "bfs_institution_code", "noga",
                                "number_employees_fte", "eligible_for_level", "eligible_fte", "eligibility_reason",
                                "eligibility_assumed", "placement_eligible", "decision_kind", "decision_source", "decision_note",
                                "match_status", "estimate_status"]])
        weighted = group.groupby(["institution_id", "campus_id"], as_index=False).agg(
            establishments=("statent_local_id", "size"), eligible_sites=("eligible_for_level", "sum"),
            eligible_fte=("eligible_fte", lambda x: x.sum() if x.notna().all() else np.nan),
        )
        weighted["campus_share"] = weighted["eligible_fte"] / denominator if supported else np.nan
        weighted["estimated_enrolled_students"] = weighted["campus_share"] * total.students
        weighted["institution_enrolled_students"] = total.students
        weighted["institution_eligible_fte"] = denominator
        for column, value in [("study_level", total.study_level), ("institution_name", total.institution_name),
                              ("match_status", total.match_status), ("academic_year", total.academic_year),
                              ("bfs_institution_code", total.bfs_institution_code), ("estimate_status", status),
                              ("institution_type", total.institution_type), ("eligibility_assumed", True),
                              ("weight_method", "eligible_site_fte"), ("ready_for_assignment", False)]:
            weighted[column] = value
        campuses.append(weighted)
        summaries.append({"institution_name": total.institution_name, "institution_id": total.institution_id,
                          "study_level": total.study_level, "academic_year": total.academic_year,
                          "institution_enrolled_students": total.students,
                          "eligible_sites": int(eligible.sum()), "eligible_fte": denominator,
                          "estimate_status": status, "match_status": total.match_status,
                          "estimated_students_sum": weighted["estimated_enrolled_students"].sum(min_count=1)})
    # No proposed or verified mappings means there are no campus estimates yet.
    return {"site_eligibility": pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame(),
            "campus_weights": pd.concat(campuses, ignore_index=True) if campuses else pd.DataFrame(),
            "allocation_summary": pd.DataFrame(summaries)}


def execute(context):
    enrolment = context.stage("data.statent.education_enrolment")
    return estimate_pooled_campus_weights(context.stage("data.statent.education_institutions"),
        enrolment["pooled_enrolment"], enrolment["statent_links"],
        pd.read_csv(context.config("education_site_eligibility"), dtype={"statent_local_id": str}))


def estimate_pooled_campus_weights(registry, enrolment, links, rules):
    """Distribute annual all-programme institution totals over teaching campuses.

    Only institutions with reviewed links get named margins. The national
    residual remains with other campuses of the same type; missing links never
    inflate the five named institutions. FTE is a campus proxy, not capacity.
    """
    from data.statent.education_institutions import pooled_compatibility
    compatible = pooled_compatibility(registry, rules)
    keys = ["institution_type", "bfs_institution_code"]
    mapped = links.loc[links.match_status.isin(["proposed", "verified"])].copy()
    if "institution_name_link" in mapped:
        mapped["institution_name"] = mapped.institution_name_link.fillna(mapped.institution_name)
    if mapped.duplicated(keys).any() or mapped.institution_id.duplicated().any():
        raise ValueError("Pooled campus weighting requires an unambiguous institution link")
    sites = registry["establishments"].loc[
        registry["establishments"].enterprise_id.isin(compatible.enterprise_id)].copy()
    rows, site_rows = [], []
    for link in mapped.itertuples():
        group = sites.loc[sites.institution_id.eq(link.institution_id)].copy()
        fte = pd.to_numeric(group.number_employees_fte, errors="raise")
        if group.empty or not np.isfinite(fte).all() or fte.lt(0).any() or fte.sum() <= 0:
            raise ValueError("Linked pooled institution has no valid positive teaching FTE")
        reference = enrolment.loc[enrolment.institution_type.eq(link.institution_type)
                                  & enrolment.bfs_institution_code.eq(link.bfs_institution_code)]
        for total in reference.itertuples():
            campuses = group.groupby(["institution_id", "campus_id"], as_index=False).agg(
                establishments=("enterprise_id", "size"), eligible_fte=("number_employees_fte", "sum"))
            campuses["eligible_sites"] = campuses.establishments
            campuses["campus_share"] = campuses.eligible_fte / fte.sum()
            campuses["institution_enrolled_students"] = total.students
            campuses["estimated_enrolled_students"] = campuses.campus_share * total.students
            for column, value in [("study_level", "tertiary"), ("institution_type", link.institution_type),
                ("institution_name", link.institution_name), ("bfs_institution_code", link.bfs_institution_code),
                ("academic_year", total.academic_year), ("match_status", link.match_status),
                ("weight_method", "pooled_teaching_fte"), ("estimate_status", "fte_proxy_estimate"),
                ("ready_for_assignment", False)]:
                campuses[column] = value
            rows.append(campuses)
        group["study_level"] = "tertiary"
        group["eligible_for_level"] = True
        group["eligible_fte"] = fte
        site_rows.append(group)
    return {"campus_weights": pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
            "site_eligibility": pd.concat(site_rows, ignore_index=True) if site_rows else pd.DataFrame()}
