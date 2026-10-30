"""Review named institution candidates against documented campus locations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.education_institutions import create_map
from data.statent.education_enrolment import DEFAULT_MAPPING, link_institutions, read_enrolment


DEFAULT_REFERENCES = DEFAULT_MAPPING.with_name("education_reference_sites.csv")


def configure(context):
    context.config("data_path")
    context.stage("data.statent.education_institutions")
    context.stage("data.statent.education_enrolment")
    context.config("education_institution_links", str(Path(context.config("data_path")) / DEFAULT_MAPPING))
    context.config("education_reference_sites", str(Path(context.config("data_path")) / DEFAULT_REFERENCES))
    context.config("analysis_path")


def review_matches(registry, links, references, distance=500.0):
    """Rank geographic evidence for proposed links; proximity does not verify identity."""
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Institution review distance must be finite and positive")
    if references.duplicated(["institution_type", "bfs_institution_code", "reference_site"]).any():
        raise ValueError("Duplicate institution reference site")
    if not np.isfinite(references[["x", "y"]].to_numpy()).all():
        raise ValueError("Institution reference sites require finite LV95 coordinates")
    sites = registry["establishments"]
    records = []
    for anchor in references.itertuples():
        candidates = sites[sites["institution_type"].eq(anchor.institution_type)].copy()
        candidates["distance_m"] = np.hypot(candidates["x"] - anchor.x, candidates["y"] - anchor.y)
        nearest = candidates.sort_values(["distance_m", "statent_local_id"]).drop_duplicates("statent_enterprise_id")
        for row in nearest.itertuples():
            records.append({"institution_type": anchor.institution_type,
                            "bfs_institution_code": anchor.bfs_institution_code,
                            "institution_name": anchor.institution_name,
                            "reference_site": anchor.reference_site,
                            "statent_enterprise_id": row.statent_enterprise_id,
                            "nearest_local_id": row.statent_local_id,
                            "campus_id": row.campus_id, "distance_m": row.distance_m,
                            "within_review_distance": row.distance_m <= distance,
                            "source_url": anchor.source_url})
    evidence = pd.DataFrame(records)
    keys = ["institution_type", "bfs_institution_code", "statent_enterprise_id"]
    score = evidence.groupby(keys, as_index=False).agg(
        reference_sites=("reference_site", "size"),
        supported_reference_sites=("within_review_distance", "sum"),
        nearest_distance_m=("distance_m", "min"),
    )
    score = score[score["supported_reference_sites"].gt(0)].sort_values(
        ["institution_type", "bfs_institution_code", "supported_reference_sites", "nearest_distance_m", "statent_enterprise_id"],
        ascending=[True, True, False, True, True])
    score["spatial_rank"] = score.groupby(["institution_type", "bfs_institution_code"]).cumcount() + 1
    # Rank is evidence for review, never an automatic identity verification.
    reviewed = links.merge(score, on=keys, how="left", validate="one_to_one")
    hit_counts = evidence[evidence["within_review_distance"]].groupby(
        ["institution_type", "bfs_institution_code", "reference_site"])["statent_enterprise_id"].nunique()
    evidence = evidence.merge(hit_counts.rename("nearby_enterprise_groups"),
                              on=["institution_type", "bfs_institution_code", "reference_site"], how="left")
    evidence["nearby_enterprise_groups"] = evidence["nearby_enterprise_groups"].fillna(0).astype(int)
    evidence["shared_location_review"] = evidence["nearby_enterprise_groups"].gt(1)
    evidence = evidence.merge(links[keys + ["match_status"]], on=keys, how="left")

    site_reviews = []
    for link in links.itertuples():
        anchors = references[references["institution_type"].eq(link.institution_type)
                             & references["bfs_institution_code"].eq(link.bfs_institution_code)]
        for site in sites[sites["statent_enterprise_id"].eq(link.statent_enterprise_id)].itertuples():
            distances = np.hypot(anchors["x"] - site.x, anchors["y"] - site.y)
            if distances.empty:
                nearest_name, nearest_distance = None, np.nan
            else:
                closest = distances.idxmin()
                nearest_name, nearest_distance = anchors.loc[closest, "reference_site"], distances.loc[closest]
            site_reviews.append({"institution_name": link.institution_name,
                                 "statent_enterprise_id": site.statent_enterprise_id,
                                 "statent_local_id": site.statent_local_id, "campus_id": site.campus_id,
                                 "nearest_reference_site": nearest_name, "distance_m": nearest_distance,
                                 "location_review": not np.isfinite(nearest_distance) or nearest_distance > distance,
                                 "noga": site.noga, "eligible_teaching_site": site.eligible_teaching_site,
                                 "match_status": link.match_status})
    site_reviews = pd.DataFrame(site_reviews)
    site_reviews["site_distance_review"] = site_reviews["location_review"]
    # A reference point can support a large connected campus without every site
    # being within the review radius of that one point.
    supported_campuses = site_reviews.groupby(["institution_name", "campus_id"])["site_distance_review"].transform(
        lambda values: (~values).any())
    site_reviews["campus_location_supported"] = supported_campuses
    site_reviews["location_review"] = ~supported_campuses
    return {"reviewed_links": reviewed, "candidate_groups": score,
            "reference_matches": evidence, "site_reviews": site_reviews,
            "reference_sites": references}


def audit(registry, links, references, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    result = review_matches(registry, links, references)
    for name, table in result.items():
        table.to_csv(output_path / (name + ".csv"), index=False)
    sensitivity = []
    for distance in [250.0, 500.0, 1000.0]:
        tested = review_matches(registry, links, references, distance)
        sensitivity.append(tested["reviewed_links"].assign(review_distance_m=distance))
    pd.concat(sensitivity).to_csv(output_path / "distance_sensitivity.csv", index=False)
    create_map(registry, output_path, institution_links=links, reference_sites=references)
    (output_path / "README.md").write_text(
        "# Institution location review\n\n"
        "Reviewed candidates: EPFL, UNIL and ZHAW. Official institution pages supply "
        "campus addresses; swisstopo address geocoding supplies LV95 reference points. "
        "EPFL Lausanne uses the approximate centre of an official campus-map view. "
        "These references are not complete campus boundaries or a historical 2022 "
        "building inventory. Distances are measured to the nearest retained STATENT "
        "site of the same nominal institution type, not to cluster centroids.\n\n"
        "candidate_groups.csv ranks enterprises by reference points within 500 m. "
        "reference_matches.csv retains distances to every type-compatible enterprise "
        "and flags shared locations. distance_sensitivity.csv tests 250/500/1000 m. "
        "site_reviews.csv keeps individual distances and flags campuses with no site "
        "near a reference point. A supported campus can extend beyond a single anchor. "
        "Such flags do not prove a site is unrelated: the reference list can be incomplete "
        "or locations may have changed since STATENT 2022.\n\n"
        "The map labels candidate groups as proposed, with official reference points "
        "in a separate layer. Spatial support does not prove the identity of an anonymous "
        "enterprise. No links are automatically verified, no additional nearby "
        "enterprises inherit student totals, and no student capacities or assignments "
        "are changed.\n", encoding="utf-8")
    print(result["reviewed_links"][["institution_name", "statent_enterprise_id",
                                   "supported_reference_sites", "reference_sites", "spatial_rank"]].to_string(index=False))
    return result


def execute(context):
    # Validate the mapping against both the BFS source and STATENT before review.
    context.stage("data.statent.education_enrolment")
    registry = context.stage("data.statent.education_institutions")
    links = pd.read_csv(context.config("education_institution_links"), dtype=str)
    references = pd.read_csv(context.config("education_reference_sites"), dtype={"bfs_institution_code": str})
    return audit(registry, links, references,
                 Path(context.config("analysis_path")) / "education_institution_matching")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--enrolment-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--mapping-path", required=True)
    parser.add_argument("--reference-path", required=True)
    parser.add_argument("--academic-year", default="2023/24")
    args = parser.parse_args()
    registry = {name: pd.read_csv(Path(args.registry_path) / (name + ".csv"),
                                 dtype={"statent_enterprise_id": str, "statent_local_id": str})
                for name in ["institutions", "campuses", "establishments"]}
    links = pd.read_csv(args.mapping_path, dtype=str)
    references = pd.read_csv(args.reference_path, dtype={"bfs_institution_code": str})
    link_institutions(read_enrolment(args.enrolment_path), registry["institutions"], links, args.academic_year)
    audit(registry, links, references, args.output_path)
