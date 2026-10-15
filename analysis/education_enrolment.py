"""Download reproducible BFS snapshots and audit institution enrolment links."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from data.statent.education_enrolment import (
    TABLES, DEFAULT_MAPPING, build_query, table_url, read_enrolment, link_institutions)


def configure(context):
    context.stage("data.statent.education_enrolment")
    context.config("analysis_path")
    context.config("education_enrolment_year", "2023/24")


def download(input_path, academic_years):
    """Freeze the BFS response, query and metadata together for reproducible totals."""
    input_path = Path(input_path)
    input_path.mkdir(parents=True, exist_ok=True)
    manifest = []
    for institution_type, table in TABLES.items():
        url = table_url(institution_type, api=True)
        with urlopen(url, timeout=60) as response:
            metadata_bytes = response.read()
        metadata = json.loads(metadata_bytes)
        query = build_query(metadata, academic_years)
        query_bytes = json.dumps(query, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=query_bytes, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:
            result_bytes = response.read()
        # Validate the response before saving it as an input snapshot.
        result = json.loads(result_bytes)
        if result.get("class") != "dataset":
            raise ValueError("BFS did not return a JSON-stat2 dataset")
        for suffix, content in [(".metadata.json", metadata_bytes),
                                (".query.json", query_bytes), (".json", result_bytes)]:
            (input_path / (table + suffix)).write_bytes(content)
        manifest.append({"institution_type": institution_type, "table_id": table,
                         "api_url": url, "academic_years": academic_years,
                         "retrieved_at": datetime.now(timezone.utc).isoformat(),
                         "response_sha256": hashlib.sha256(result_bytes).hexdigest()})
    (input_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def audit(result, output_path, academic_year):
    """Report linked and unmatched enrolment without assigning missing institutions zero."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(output_path / (name + ".csv"), index=False)
    totals = result["institution_totals"]
    totals[totals["verified_groups"].eq(0)].to_csv(output_path / "unmatched_bfs_institutions.csv", index=False)
    groups = result["statent_links"]
    groups[~groups["match_status"].eq("verified")].to_csv(
        output_path / "unmatched_statent_groups.csv", index=False)
    counts = totals.groupby("institution_type").agg(
        bfs_categories=("bfs_institution_code", "size"),
        positive_categories=("enrolled_students", lambda x: x.gt(0).sum()),
        enrolled_students=("enrolled_students", lambda x: x.sum() if x.notna().all() else float("nan")),
        model_level_enrolled_students=("model_level_enrolled_students", lambda x: x.sum() if x.notna().all() else float("nan")),
        strict_degree_enrolled_students=("strict_degree_enrolled_students", lambda x: x.sum() if x.notna().all() else float("nan")),
        candidate_groups=("candidate_groups", "sum"), verified_groups=("verified_groups", "sum"),
        missing_source_cells=("missing_source_cells", "sum"),
    ).reset_index()
    counts.to_csv(output_path / "coverage.csv", index=False)
    (output_path / "README.md").write_text(
        "# BFS enrolment and STATENT institution links\n\n"
        "Reference academic year: %s. Snapshots also retain the other requested years.\n\n"
        "enrolment.csv contains institution and study-level totals, summing all subjects "
        "and educational-origin/sex categories. Missing source cells remain missing. "
        "Legacy degrees and continuing education are retained separately from the current "
        "bachelor/master/doctoral model levels. All counts are institution enrolment, "
        "not campus capacity, commuting counts or domestic-resident targets. Educational "
        "origin is not current residence.\n\n"
        "model_enrolment.csv and linked_model_enrolment.csv align with survey codes 11–13: "
        "FH/PH diplomas join Bachelor; UH licentiates/diplomas join Master. Strict and legacy "
        "contributions remain visible, combined once before campus splitting. Continuing education "
        "stays excluded and doctoral totals are unchanged. model_level_enrolled_students uses this "
        "aligned reference; strict_degree_enrolled_students preserves the pure degree total.\n\n"
        "linked_enrolment.csv exposes proposed links but only verified links populate "
        "verified_enrolled_students. EPFL, UNIL and ZHAW are spatially reviewed candidates, "
        "not confirmed identities of the anonymous STATENT enterprises. The mapping must "
        "be reviewed before allocation. A verified BFS institution can currently link to "
        "one STATENT group; multiple groups require an explicit allocation rule to avoid "
        "duplicating its student total.\n\n"
        "institution_totals.csv counts each BFS category once, regardless of proposed links. "
        "Zero categories can represent historical institutions. Other-provider aggregates "
        "cannot be linked as individual institutions. Coverage is therefore incomplete, "
        "particularly for PH and umbrella FH organisations. Higher vocational education "
        "is outside these tables. unmatched_bfs_institutions.csv and "
        "unmatched_statent_groups.csv include proposed links until verified.\n\n"
        "Student assignment, research-site eligibility, campus weights and capacities "
        "are unchanged in this step.\n" % academic_year, encoding="utf-8")
    print(counts.to_string(index=False))


def execute(context):
    result = context.stage("data.statent.education_enrolment")
    audit(result, Path(context.config("analysis_path")) / "education_enrolment",
          context.config("education_enrolment_year"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="BFS snapshot directory")
    parser.add_argument("--registry-path", required=True, help="Existing campus audit directory")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--mapping-path", default=str(DEFAULT_MAPPING))
    parser.add_argument("--academic-year", default="2023/24")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--years", nargs="+", default=["2021/22", "2022/23", "2023/24"])
    args = parser.parse_args()
    if args.download:
        download(args.input_path, args.years)
    institutions = pd.read_csv(Path(args.registry_path) / "institutions.csv",
                               dtype={"statent_enterprise_id": str})
    links = pd.read_csv(args.mapping_path, dtype=str)
    result = link_institutions(read_enrolment(args.input_path), institutions, links, args.academic_year)
    audit(result, args.output_path, args.academic_year)
