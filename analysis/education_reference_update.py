"""Prepare versioned education references and review them without changing models.

Run ``python -m analysis.education_reference_update --help``. Preparation copies
reviewed A/B/D unchanged and generates enrolment inputs; review never repairs or
verifies links automatically. Every command writes to a new output directory.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from analysis.education_enrolment import download
from data.statent.education_enrolment import TABLES, LINK_FIELDS, read_enrolment, pooled_enrolment_reference
from data.statent import education_higher_vocational as higher


REFERENCES = {
    "A": "education_institution_links.csv",
    "B": "education_site_eligibility.csv",
    "C": "education_higher_vocational_enrolment.csv",
    "D": "education_reference_sites.csv",
}
REGISTRY_FILES = ["institutions", "establishments", "compatibility"]
B_FIELDS = ["statent_local_id", "study_level", "regular_eligible", "placement_eligible",
            "decision_kind", "evidence_url", "note"]
D_FIELDS = ["institution_type", "bfs_institution_code", "institution_name", "reference_site",
            "address", "source_url", "x", "y", "coordinate_source", "coordinate_note", "reviewed_on"]


def file_info(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def new_directory(path):
    path = Path(path)
    # Refuse even an existing empty directory: a version is never overwritten.
    path.mkdir(parents=True, exist_ok=False)
    return path


def prepare(reference_path, output_path, academic_years, snapshot_path=None, fetch=False):
    """Create an isolated candidate bundle, preserving the current references.

    Offline replay requires the three official cubes and metadata, including the
    new higher-vocational cube. It cannot reconstruct a historic official response
    from a hand-aggregated CSV. Retain its source manifest when one is available.
    """
    if fetch == (snapshot_path is not None):
        raise ValueError("Choose exactly one of --download or --snapshot-path")
    if not academic_years or len(set(academic_years)) != len(academic_years):
        raise ValueError("Choose at least one unique academic year")
    source = Path(reference_path)
    required = [source / REFERENCES[k] for k in ["A", "B", "D"]]
    if snapshot_path is not None:
        snapshots = Path(snapshot_path)
        required += [snapshots / (table + suffix) for table in [*TABLES.values(), higher.TABLE]
                     for suffix in [".json", ".metadata.json", ".query.json"]]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    out = new_directory(output_path)
    manifest = {"schema_version": 1, "operation": "education_reference_prepare",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "academic_years": list(academic_years), "status": "in_progress",
                "source_mode": "download" if fetch else "offline_replay",
                "automatic_promotion": False, "inputs": [file_info(p) for p in required],
                "tool_sources": [file_info(p) for p in [Path(__file__), Path(higher.__file__),
                                  Path(__file__).with_name("education_enrolment.py")]],
                "review_policy": "A/B/D copied unchanged, not newly reviewed or verified; run review against target STATENT"}
    manifest_path = out / "reference_manifest.json"
    write_json(manifest_path, manifest)
    try:
        for key in ["A", "B", "D"]:
            shutil.copy2(source / REFERENCES[key], out / REFERENCES[key])
        if fetch:
            download(out, academic_years, include_higher_vocational=True)
        else:
            for path in required[3:]:
                shutil.copy2(path, out / path.name)
            for name in ["manifest.json", "reference_manifest.json"]:
                path = snapshots / name
                if path.exists():
                    shutil.copy2(path, out / ("source_" + name))
            cube = json.loads((out / (higher.TABLE + ".json")).read_text())
            metadata = json.loads((out / (higher.TABLE + ".metadata.json")).read_text())
            ref, checks = higher.parse_reference(cube, metadata, academic_years)
            ref.to_csv(out / REFERENCES["C"], index=False)
            checks.to_csv(out / "higher_vocational_conservation.csv", index=False)
        enrolment = read_enrolment(out)
        if not set(academic_years) <= set(enrolment.academic_year):
            raise ValueError("University snapshots do not cover every requested year")
        enrolment = enrolment.loc[enrolment.academic_year.isin(academic_years)]
        # The production helper rejects missing cells and invalid pooled totals.
        pooled = pooled_enrolment_reference(enrolment, pd.read_csv(out / REFERENCES["C"]))
        totals = pooled.groupby(["academic_year", "institution_type"], as_index=False).students.sum()
        totals.to_csv(out / "enrolment_totals.csv", index=False)
        previous_file = source / REFERENCES["C"]
        if previous_file.exists():
            previous = pd.read_csv(previous_file)
            current = pd.read_csv(out / REFERENCES["C"])
            comparison = current.merge(previous[["academic_year", "study_level", "students"]],
                on=["academic_year", "study_level"], how="outer", suffixes=("_new", "_previous"),
                validate="one_to_one")
            comparison["change_same_year"] = comparison.students_new - comparison.students_previous
            comparison.to_csv(out / "higher_vocational_previous_comparison.csv", index=False)
            manifest["previous_higher_vocational"] = file_info(previous_file)
        manifest["status"] = "prepared_requires_review"
        manifest["files"] = {p.name: file_info(p) for p in sorted(out.iterdir()) if p != manifest_path}
    except Exception as error:
        # A partial network run must never look like a complete usable release.
        manifest.update(status="failed", error=str(error))
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    return totals


def review_tables(registry, links, rules, references, enrolment, previous=None, distance=500.):
    """Return all actionable issues, including stale IDs that production rejects.

    Do not call the production linking helper first: it intentionally stops at
    the first missing identifier, exactly when this maintenance report is useful.
    """
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Review distance must be positive and finite")
    issues, spatial, changes = [], [], []

    def issue(severity, file, key, code, detail):
        issues.append(dict(severity=severity, file=file, key=str(key), code=code, detail=str(detail)))

    tables = {"A": (links, LINK_FIELDS), "B": (rules, B_FIELDS), "D": (references, D_FIELDS),
              "institutions": (registry["institutions"], ["statent_enterprise_id", "institution_types"]),
              "establishments": (registry["establishments"], ["statent_enterprise_id", "statent_local_id",
                  "institution_type", "noga", "enterprise_noga", "x", "y", "statent_year"]),
              "compatibility": (registry["compatibility"], ["statent_local_id", "study_level"])}
    for name, (frame, columns) in tables.items():
        missing = sorted(set(columns) - set(frame))
        if missing:
            issue("error", name, "schema", "missing_columns", missing)
    if issues:
        return pd.DataFrame(issues), pd.DataFrame(), pd.DataFrame()
    for name, frame, keys in [("A", links, ["institution_type", "bfs_institution_code"]),
                              ("A", links, ["statent_enterprise_id"]),
                              ("B", rules, ["statent_local_id", "study_level"]),
                              ("D", references, ["institution_type", "bfs_institution_code", "reference_site"]),
                              ("institutions", registry["institutions"], ["statent_enterprise_id"]),
                              ("establishments", registry["establishments"], ["statent_local_id"]),
                              ("compatibility", registry["compatibility"], ["statent_local_id", "study_level"])]:
        for _, row in frame.loc[frame.duplicated(keys, keep=False)].iterrows():
            issue("error", name, " / ".join(str(row[k]) for k in keys), "duplicate_key", ", ".join(keys))
    sites = registry["establishments"].copy()
    for col in ["x", "y"]:
        sites[col] = pd.to_numeric(sites[col], errors="coerce")
    valid_xy = np.isfinite(sites[["x", "y"]]).all(axis=1) & sites[["x", "y"]].gt(0).all(axis=1)
    for key in sites.loc[~valid_xy, "statent_local_id"]:
        issue("error", "establishments", key, "invalid_coordinates", "Expected finite positive LV95 coordinates")
    institutions = registry["institutions"].drop_duplicates("statent_enterprise_id").set_index("statent_enterprise_id")
    bfs_keys = set(zip(enrolment.institution_type, enrolment.bfs_institution_code))
    for name, frame, columns in [("A", links, ["institution_name", "evidence_url", "evidence_note"]),
                                  ("B", rules, ["decision_kind", "evidence_url", "note"]),
                                  ("D", references, ["source_url", "coordinate_source", "coordinate_note", "reviewed_on"])]:
        for idx, row in frame.iterrows():
            for column in columns:
                if pd.isna(row[column]) or not str(row[column]).strip():
                    issue("error", name, idx, "missing_evidence", column)
    for row in links.itertuples():
        key = f"{row.institution_type}:{row.bfs_institution_code}:{row.statent_enterprise_id}"
        if row.match_status not in ["proposed", "verified"]:
            issue("error", "A", key, "invalid_match_status", row.match_status)
        elif row.match_status == "proposed":
            issue("review", "A", key, "provisional_identity", "Spatial evidence is not independent identity verification")
        if (row.institution_type, row.bfs_institution_code) not in bfs_keys:
            issue("error", "A", key, "unknown_bfs_institution", "Missing from selected academic-year snapshot")
        else:
            source = enrolment.loc[enrolment.institution_type.eq(row.institution_type)
                                   & enrolment.bfs_institution_code.eq(row.bfs_institution_code)]
            if source.aggregate_provider.any():
                issue("error", "A", key, "aggregate_provider", "An umbrella/other-provider aggregate cannot identify one enterprise without a reviewed allocation")
            if source.students.sum() == 0:
                issue("review", "A", key, "zero_enrolment", "May be a historic or changed BFS institution category")
        if row.statent_enterprise_id not in institutions.index:
            issue("error", "A", key, "missing_enterprise", "Do not transfer this link to a nearby enterprise automatically")
        elif row.institution_type not in str(institutions.loc[row.statent_enterprise_id, "institution_types"]).split(";"):
            issue("error", "A", key, "incompatible_institution_type", institutions.loc[row.statent_enterprise_id, "institution_types"])
        if not ((references.institution_type == row.institution_type) & (references.bfs_institution_code == row.bfs_institution_code)).any():
            issue("review", "A", key, "no_reference_points", "D has no spatial review evidence for this institution")
    nominal = set(zip(registry["compatibility"].statent_local_id, registry["compatibility"].study_level))
    ids = set(sites.statent_local_id)
    for row in rules.itertuples():
        key = f"{row.statent_local_id}:{row.study_level}"
        if row.statent_local_id not in ids:
            issue("error", "B", key, "missing_establishment", "Review against the new extract; do not substitute a nearby ID")
        if row.study_level not in ["bachelor", "master", "doctorate_habilitation"]:
            issue("error", "B", key, "invalid_study_level", row.study_level)
        for column in ["regular_eligible", "placement_eligible"]:
            if str(getattr(row, column)) not in ["True", "False"]:
                issue("error", "B", key, "invalid_boolean", column + " must be True or False")
        if (row.statent_local_id, row.study_level) not in nominal:
            issue("error" if str(row.regular_eligible) == "True" else "review", "B", key,
                  "unsupported_eligibility", "No nominal site/programme compatibility in target registry")
    for row in references.itertuples():
        key = f"{row.institution_type}:{row.bfs_institution_code}:{row.reference_site}"
        x, y = pd.to_numeric(pd.Series([row.x, row.y]), errors="coerce")
        if not np.isfinite([x, y]).all() or min(x, y) <= 0:
            issue("error", "D", key, "invalid_coordinates", "Expected finite positive LV95 coordinates")
            continue
        if pd.isna(pd.to_datetime(row.reviewed_on, errors="coerce")):
            issue("error", "D", key, "invalid_review_date", row.reviewed_on)
        candidates = sites.loc[valid_xy & sites.institution_type.eq(row.institution_type)].copy()
        candidates["distance_m"] = np.hypot(candidates.x-x, candidates.y-y)
        candidates = candidates.sort_values(["distance_m", "statent_local_id"])
        linked = links.loc[links.institution_type.eq(row.institution_type)
                           & links.bfs_institution_code.eq(row.bfs_institution_code), "statent_enterprise_id"]
        selected = candidates.loc[candidates.statent_enterprise_id.isin(linked)]
        linked_distance = selected.distance_m.min() if len(selected) else np.nan
        if not np.isfinite(linked_distance) or linked_distance > distance:
            issue("review", "D", key, "reference_not_supported", "No linked type-compatible site within %.0f m; anchor is not a campus boundary" % distance)
        spatial.append(dict(reference_site=row.reference_site, institution_name=row.institution_name,
                            linked_distance_m=linked_distance,
                            closest_enterprise=candidates.iloc[0].statent_enterprise_id if len(candidates) else None,
                            closest_distance_m=candidates.distance_m.min() if len(candidates) else np.nan,
                            nearby_enterprises=candidates.loc[candidates.distance_m.le(distance), "statent_enterprise_id"].nunique(),
                            reviewed_on=row.reviewed_on))
    if previous is None:
        issue("review", "registry", "baseline", "no_previous_registry", "Identity/NOGA/location changes cannot be assessed without a previous registry snapshot")
    else:
        required = ["statent_enterprise_id", "institution_type", "noga", "enterprise_noga", "x", "y"]
        old = previous["establishments"]
        missing = set(["statent_local_id", *required]) - set(old)
        if missing:
            issue("error", "previous_registry", "schema", "missing_columns", sorted(missing))
        else:
            # Report a malformed baseline instead of silently trusting one of
            # several conflicting records for the same anonymous identifier.
            for key in old.loc[old.statent_local_id.duplicated(keep=False), "statent_local_id"]:
                issue("error", "previous_registry", key, "duplicate_key", "statent_local_id")
            old = old.drop_duplicates("statent_local_id").set_index("statent_local_id")
            new = sites.drop_duplicates("statent_local_id").set_index("statent_local_id")
            relevant = set(rules.statent_local_id) | set(sites.loc[sites.statent_enterprise_id.isin(links.statent_enterprise_id), "statent_local_id"])
            relevant |= set(old.index[old.statent_enterprise_id.isin(links.statent_enterprise_id)])
            for key in sorted(set(old.index) | set(new.index)):
                changes_for_site = []
                if key not in old.index:
                    changes_for_site.append(("new_establishment", "", "present"))
                elif key not in new.index:
                    changes_for_site.append(("removed_establishment", "present", ""))
                else:
                    for col in required[:4]:
                        if str(old.loc[key, col]) != str(new.loc[key, col]):
                            changes_for_site.append(("changed_" + col, str(old.loc[key, col]), str(new.loc[key, col])))
                    xy = pd.to_numeric(old.loc[key, ["x", "y"]], errors="coerce").to_numpy(dtype=float)
                    movement = np.linalg.norm(new.loc[key, ["x", "y"]].to_numpy(dtype=float)-xy)
                    if not np.isfinite(movement) or movement > 0:
                        changes_for_site.append(("coordinate_change_m", "0", str(movement)))
                for code, before, after in changes_for_site:
                    changes.append(dict(statent_local_id=key, change=code, before=before, after=after,
                                        affects_reviewed_inputs=key in relevant))
                    # Every relevant movement is exposed for review; the radius
                    # only controls the D proximity check, not silent ID reuse.
                    if key in relevant:
                        issue("review", "registry", key, code, f"{before} -> {after}")
    return (pd.DataFrame(issues, columns=["severity", "file", "key", "code", "detail"]),
            pd.DataFrame(spatial, columns=["reference_site", "institution_name", "linked_distance_m",
                         "closest_enterprise", "closest_distance_m", "nearby_enterprises", "reviewed_on"]),
            pd.DataFrame(changes, columns=["statent_local_id", "change", "before", "after", "affects_reviewed_inputs"]))


def review(reference_path, registry_path, output_path, academic_year, previous_registry_path=None, distance=500.):
    source, reg = Path(reference_path), Path(registry_path)
    inputs = [source / REFERENCES[k] for k in REFERENCES]
    inputs += [reg / (name + ".csv") for name in REGISTRY_FILES]
    inputs += [source / (table + suffix) for table in TABLES.values() for suffix in [".json", ".metadata.json"]]
    previous = None
    if previous_registry_path is not None:
        p = Path(previous_registry_path) / "establishments.csv"
        inputs.append(p)
        previous = {"establishments": pd.read_csv(p, dtype=str)}
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    registry = {name: pd.read_csv(reg / (name + ".csv"), dtype=str) for name in REGISTRY_FILES}
    enrolment = read_enrolment(source)
    enrolment = enrolment.loc[enrolment.academic_year.eq(academic_year)]
    if enrolment.empty:
        raise ValueError("Selected academic year absent from university snapshots")
    issues, spatial, changes = review_tables(registry,
        pd.read_csv(source / REFERENCES["A"], dtype=str),
        pd.read_csv(source / REFERENCES["B"], dtype=str),
        pd.read_csv(source / REFERENCES["D"], dtype=str), enrolment, previous, distance)
    extra = []
    try:
        ref = pd.read_csv(source / REFERENCES["C"])
        required = ["academic_year", "study_level", "students", "source_url", "coverage_note"]
        if not set(required) <= set(ref):
            raise ValueError("Missing C columns: " + str(sorted(set(required)-set(ref))))
        selected = ref.loc[ref.academic_year.eq(academic_year)]
        if set(selected.study_level) != set(higher.GROUPS) or selected.study_level.duplicated().any():
            raise ValueError("C requires exactly the three mapped groups for the requested year")
        values = pd.to_numeric(selected.students, errors="raise")
        if not np.isfinite(values).all() or values.lt(0).any() or values.ne(values.round()).any():
            raise ValueError("C requires finite nonnegative integer counts")
        cube_path, meta_path = source / (higher.TABLE + ".json"), source / (higher.TABLE + ".metadata.json")
        if cube_path.exists() and meta_path.exists():
            inputs += [cube_path, meta_path]
            # Saved cubes may hold several years; verify the complete selection
            # before comparing the requested year's derived CSV values.
            cube, meta = json.loads(cube_path.read_text()), json.loads(meta_path.read_text())
            years = list(cube["dimension"]["Jahr"]["category"]["label"].values())
            parsed, _ = higher.parse_reference(cube, meta, years)
            expected = parsed.loc[parsed.academic_year.eq(academic_year)].set_index("study_level").students.sort_index()
            actual = selected.assign(students=values).set_index("study_level").students.sort_index()
            if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
                raise ValueError("C counts disagree with its frozen official source cube")
        else:
            extra.append(dict(severity="review", file="C", key=academic_year, code="no_frozen_source_cube",
                              detail="Legacy aggregate has no reproducible official response; preserve and review its source notes"))
    except (ValueError, KeyError, TypeError) as error:
        extra.append(dict(severity="error", file="C", key=academic_year, code="invalid_higher_vocational_reference", detail=str(error)))
    issues = pd.concat([issues, pd.DataFrame(extra)], ignore_index=True)
    out = new_directory(output_path)
    issues.to_csv(out / "review_issues.csv", index=False)
    spatial.to_csv(out / "spatial_review.csv", index=False)
    changes.to_csv(out / "registry_changes.csv", index=False)
    # Archive the registry used for this review as next year's comparison baseline.
    (out / "registry").mkdir()
    for name in REGISTRY_FILES:
        shutil.copy2(reg / (name + ".csv"), out / "registry" / (name + ".csv"))
    errors = int(issues.severity.eq("error").sum())
    summary = {"academic_year": academic_year, "errors": errors,
               "review_items": int(issues.severity.eq("review").sum()), "automatic_promotion": False,
               "status": "blocked" if errors else "requires_human_review" if len(issues) else "checks_passed",
               "statent_years": sorted(registry["establishments"].get("statent_year", pd.Series(dtype=str)).dropna().unique().tolist()),
               "review_distance_m": distance,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "inputs": [file_info(p) for p in inputs], "tool_source": file_info(__file__)}
    write_json(out / "review_manifest.json", summary)
    (out / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Education reference review</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:40px auto;padding:20px}'
        'table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccc;text-align:left}'
        'td{overflow-wrap:anywhere}h2{margin-top:36px}</style>'
        '<h1>Education reference review</h1><p>' + escape(summary["status"]) +
        f': {errors} errors; {summary["review_items"]} review items.</p>'
        '<p>Input files were not modified. Proximity is evidence, not identity verification. '
        'No links or eligibility decisions are automatically transferred or approved.</p>'
        '<h2>Issues requiring attention</h2>' + issues.to_html(index=False, escape=True) +
        '<h2>Campus reference evidence</h2>' + spatial.to_html(index=False, escape=True) +
        '<h2>Changes from previous STATENT</h2>' + changes.to_html(index=False, escape=True), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Create a new candidate input bundle; never overwrite an existing release")
    prep.add_argument("--reference-path", required=True, help="Existing directory containing reviewed A/B/D")
    prep.add_argument("--output-path", required=True, help="New directory for the candidate release")
    prep.add_argument("--years", nargs="+", required=True, help="Academic-year labels, e.g. 2024/25")
    mode = prep.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true", help="Fetch official UH/FH/PH and higher-vocational cubes")
    mode.add_argument("--snapshot-path", help="Offline replay directory with all three frozen cubes, queries and metadata")
    check = commands.add_parser("review", help="Check references against the target STATENT registry without changing them")
    check.add_argument("--reference-path", required=True)
    check.add_argument("--registry-path", required=True, help="Export from analysis.education_institutions")
    check.add_argument("--previous-registry-path", help="Previous review's registry subdirectory")
    check.add_argument("--output-path", required=True, help="New directory for report and registry baseline")
    check.add_argument("--academic-year", required=True)
    check.add_argument("--review-distance", type=float, default=500., help="Reference-point support radius in metres; not automatic matching")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.reference_path, args.output_path, args.years, args.snapshot_path, args.download).to_string(index=False))
    else:
        summary = review(args.reference_path, args.registry_path, args.output_path,
                         args.academic_year, args.previous_registry_path, args.review_distance)
        print(json.dumps({k: summary[k] for k in ["status", "errors", "review_items"]}))
        if summary["errors"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
