"""Audit eligibility assumptions and provisional FTE-weighted campus shares."""

import argparse
from pathlib import Path

import pandas as pd

from analysis.education_institution_matching import DEFAULT_REFERENCES, review_matches
from data.statent.education_institutions import build_registry
from data.statent.education_enrolment import DEFAULT_MAPPING, read_enrolment, link_institutions
from data.statent.education_campus_weights import estimate_campus_weights, DEFAULT_ELIGIBILITY_RULES


def configure(context):
    context.config("data_path")
    context.stage("data.statent.education_institutions")
    context.stage("data.statent.education_enrolment")
    context.stage("data.statent.education_campus_weights")
    context.config("education_institution_links", str(Path(context.config("data_path")) / DEFAULT_MAPPING))
    context.config("education_site_eligibility", str(Path(context.config("data_path")) / DEFAULT_ELIGIBILITY_RULES))
    context.config("education_reference_sites", str(Path(context.config("data_path")) / DEFAULT_REFERENCES))
    context.config("analysis_path")


def audit(registry, enrolment, result, links, references, output_path, eligibility_rules=None):
    """Expose FTE allocation assumptions, excluded sites and doctoral-site sensitivity."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    result = {name: table.copy() for name, table in result.items()}
    reviewed = review_matches(registry, links, references)["site_reviews"]
    review_fields = ["statent_local_id", "nearest_reference_site", "distance_m", "campus_location_supported", "location_review"]
    result["site_eligibility"] = result["site_eligibility"].merge(
        reviewed[review_fields], on="statent_local_id", how="left", validate="many_to_one")
    eligibility = result["site_eligibility"]
    eligibility["programme_review_note"] = "Nominal classification; programme coverage and regular attendance remain unverified"
    eligibility["programme_review_source"] = ""
    sion = eligibility["statent_local_id"].eq("712752222409") & eligibility["study_level"].isin(["bachelor", "master"])
    eligibility.loc[sion, "programme_review_note"] = (
        "Sion sources document master projects and doctoral research; regular bachelor/master attendance shares require review")
    eligibility.loc[sion, "programme_review_source"] = "https://www.epfl.ch/schools/sb/research/isic/valais/"
    remote = eligibility["statent_local_id"].eq("764054394536")
    eligibility.loc[remote, "programme_review_note"] = (
        "Remote research branch candidate; teaching and placements do not establish a regular campus student total")
    eligibility.loc[remote, "programme_review_source"] = "https://www.zhaw.ch/en/lsfm/institutes-centres/iunr/tourism-and-sustainable-development"
    eligibility["programme_review_required"] = True
    eligibility["regular_destination_rule_recorded"] = eligibility["decision_kind"].eq("modelling_assumption")
    campus_reviews = reviewed.groupby("campus_id")["location_review"].max()
    result["campus_weights"]["location_review"] = result["campus_weights"]["campus_id"].map(campus_reviews)
    names = reviewed[reviewed["campus_location_supported"]].sort_values("distance_m").drop_duplicates("campus_id")
    result["campus_weights"]["reference_label"] = result["campus_weights"]["campus_id"].map(
        names.set_index("campus_id")["nearest_reference_site"]).fillna("Location requires review")
    for name, table in result.items():
        table.to_csv(output_path / (name + ".csv"), index=False)
    registry["compatibility"].to_csv(output_path / "study_compatibility.csv", index=False)
    baseline = estimate_campus_weights(registry, enrolment["linked_model_enrolment"], include_doctoral_research=False,
                                      eligibility_rules=eligibility_rules)
    baseline["campus_weights"].to_csv(output_path / "teaching_only_baseline.csv", index=False)
    nominal = estimate_campus_weights(registry, enrolment["linked_model_enrolment"])
    nominal["campus_weights"].to_csv(output_path / "nominal_noga_baseline.csv", index=False)
    keys = ["institution_id", "campus_id", "study_level", "academic_year"]
    comparison = result["campus_weights"].merge(
        baseline["campus_weights"][keys + ["campus_share", "estimated_enrolled_students"]],
        on=keys, suffixes=("", "_teaching_only"), validate="one_to_one")
    comparison["students_change"] = comparison["estimated_enrolled_students"] - comparison["estimated_enrolled_students_teaching_only"]
    comparison[comparison["study_level"].eq("doctorate_habilitation")].to_csv(
        output_path / "doctoral_research_sensitivity.csv", index=False)
    (output_path / "README.md").write_text(
        "# Provisional campus enrolment estimates\n\n"
        "Campus share = eligible establishment FTE in that campus / eligible FTE "
        "across the linked institution, calculated separately for each study level. "
        "Estimated enrolled students = BFS institution enrolment at that level * share. "
        "Counts remain fractional estimates, not measured campus capacity or domestic "
        "commuter targets. No downsampling or residence adjustment is applied.\n\n"
        "Bachelor/master eligibility follows conflict-free local tertiary NOGA. "
        "Doctoral eligibility additionally includes research NOGA 721* establishments "
        "within a university enterprise (enterprise NOGA 854201). Other ancillary "
        "sites and type conflicts remain excluded from automatic eligibility. "
        "FH/PH research sites do not receive university doctoral totals merely because "
        "they host collaborative research. These are nominal assumptions, not verified "
        "programme offerings. Bachelor internships and master projects can occur in "
        "research sites; this default does not model their placement frequency.\n\n"
        "All three named links remain proposed. Both institution identity and site "
        "eligibility require review. FTE includes researchers and support staff and "
        "does not measure students. Missing/non-finite/negative eligible FTE or zero "
        "eligible FTE leave the institution-level allocation unresolved, with no "
        "silent uniform or headcount fallback.\n\n"
        "site_eligibility.csv records each site/level decision, reason, FTE and location "
        "review. campus_weights.csv reports shares, estimates and reference labels. "
        "allocation_summary.csv checks conservation against BFS level totals. "
        "doctoral_research_sensitivity.csv compares the new doctoral rule with the "
        "teaching-only baseline. Location flags retain unsupported campuses; "
        "their absence from our incomplete reference list does not justify exclusion. "
        "FH/PH diplomas are included in Bachelor and university/ETH licentiates/diplomas "
        "in Master, matching the survey codebook. Continuing education and unmatched institutions receive "
        "no campus estimates in this pilot.\n\n"
        "Reviewed modelling rules now exclude Sion from the regular bachelor split "
        "and the remote ZHAW branch from regular bachelor/master splits, preserving "
        "placement eligibility. Sion remains master-eligible for possible thesis "
        "destinations. These are explicit assumptions, not measured attendance "
        "frequencies or proof of absent teaching. nominal_noga_baseline.csv retains "
        "the prior mechanical weights for comparison.\n\n"
        "The base registry's student_capacity remains unset and ready_for_assignment "
        "is false for these detailed source estimates. These programme-specific tables "
        "are sensitivity evidence, not production campus targets. The pipeline uses "
        "separate pooled teaching-site weights for education assignment.\n",
        encoding="utf-8")
    print(result["allocation_summary"].to_string(index=False))
    return result


def execute(context):
    registry = context.stage("data.statent.education_institutions")
    enrolment = context.stage("data.statent.education_enrolment")
    rules = pd.read_csv(context.config("education_site_eligibility"), dtype={"statent_local_id": str})
    path = Path(context.config("analysis_path")) / "education_campus_weights"
    # Retain the detailed source sensitivity report; production weights are now
    # pooled and have no per-degree doctoral sensitivity or decision_kind column.
    detailed = estimate_campus_weights(registry, enrolment["linked_model_enrolment"], eligibility_rules=rules)
    result = audit(registry, enrolment, detailed,
                 pd.read_csv(context.config("education_institution_links"), dtype=str),
                 pd.read_csv(context.config("education_reference_sites"), dtype={"bfs_institution_code": str}),
                 path, rules)
    for name, table in context.stage("data.statent.education_campus_weights").items():
        result["pooled_" + name] = table
        table.to_csv(path / ("pooled_" + name + ".csv"), index=False)
    with (path / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("\nThe detailed tables above are source sensitivity evidence. Production uses "
                     "pooled_campus_weights.csv and pooled_site_eligibility.csv: one annual tertiary "
                     "group including continuing programmes, restricted to teaching sites.\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--enrolment-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--mapping-path", required=True)
    parser.add_argument("--eligibility-path", required=True)
    parser.add_argument("--reference-path", required=True)
    parser.add_argument("--academic-year", default="2023/24")
    parser.add_argument("--campus-distance", type=float, default=500.0)
    args = parser.parse_args()
    # Rebuild with current compatibility rules using already-imputed site coordinates.
    sites = pd.read_csv(Path(args.registry_path) / "establishments.csv",
                        dtype={"statent_enterprise_id": str, "statent_local_id": str,
                               "noga": str, "enterprise_noga": str})
    registry = build_registry(sites, args.campus_distance)
    links = pd.read_csv(args.mapping_path, dtype=str)
    enrolment = link_institutions(read_enrolment(args.enrolment_path), registry["institutions"],
                                 links, args.academic_year)
    rules = pd.read_csv(args.eligibility_path, dtype={"statent_local_id": str})
    result = estimate_campus_weights(registry, enrolment["linked_model_enrolment"], eligibility_rules=rules)
    audit(registry, enrolment, result, links,
          pd.read_csv(args.reference_path, dtype={"bfs_institution_code": str}), args.output_path, rules)
