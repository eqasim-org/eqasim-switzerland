"""Locate reported road exits without changing sampled interview crossings."""

import logging
import json
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd

from data.cross_border.generate_od import read_road_exit_names
from data.cross_border.reference_files import reference_directory, file_versions

logger = logging.getLogger("synpp")

# This is a conservative reference check, not a parameter fitted to traffic.
# A smaller-crossing label must identify exactly one existing road point within
# the first 500 m search radius also used by the directional network matcher.
REFERENCE_RADIUS = 500.0

# Explicit 2021 codebook-name / collected-label pairs. These are candidate
# aliases, not unconditional mappings: the geographic checks below can reject
# them. Checking the official name too prevents silently reusing a changed code.
EXIT_LABELS = {
    1013: ("Riehen Lörracherstrasse (150)", "Riehen-Lörrach Stetten"),
    1018: ("Riehen Inzlingerstrasse (früher 153)", "Inzlingen-Riehen"),
    1022: ("Stein / Bad Säckingen (159)", "Bad Säckingen-Stein Holzbrücke"),
    1023: ("Waldshut-Koblenz (204)", "Koblenz-Waldshut Rheinbrücke"),
    1026: ("Laufenburg (160)", "Laufenburg-Laufenburg Brücke"),
    1029: ("Kaiserstuhl (207)", "Kaiserstuhl-Hohentengen"),
    1046: ("Tägerwilen (früher 266)", "Tägerwilen-Konstanz Tägermoos"),
    1047: ("Stein am Rhein (früher 244)", "Stein am Rhein-Öhningen"),
    1067: ("Diepoldsau (345)", "Diepoldsau-Hohenems"),
    1070: ("Schaanwald (320)", "Schaanwald-Tisis (FL-Vorarlberg)"),
    1080: ("Martina (326)", "Martina-Nauders (Reschen)"),
    1083: ("Castasegna (340)", "Castasegna (Bregaglia)"),
    1084: ("Müstair (330)", "Müstair-Taufers im Münstertal"),
    1086: ("Splügenberg (342)", "Splügen Pass"),
    1093: ("Stabio Confine (428)", "Stabio-Gaggiolo"),
    1094: ("Ponte Tresa (410)", "Ponte Tresa"),
    1095: ("Brusata (409)", "Brusata-Novazzano"),
    1107: ("Indemini (435)", "Indemini-Veddasca"),
    1109: ("Camedo (448)", "Camedo-Ribellasca (Centovalli)"),
    1110: ("Grand-St-Bernard Tunnel (570)", "Gd-St-Bernard Tunnel"),
    1113: ("Le Châtelard (518)", "Le Châtelard-Vallorcine"),
    1114: ("Morgins (520)", "Morgins-Châtel"),
    1115: ("Perly (626)", "Perly"),
    1119: ("La Croix-de-Rozon (630)", "Croix-de-Rozon"),
    1127: ("Hermance", "Hermance"),
    1139: ("Ferney Voltaire (645)", "Ferney-Voltaire"),
    1150: ("La Cure (537)", "Saint-Cergue-La Cure"),
    1153: ("Les Verrières (549)", "Les Verrières-Pontarlier"),
    1158: ("Crassier", "Crassier-Divonne"),
    1164: ("Goumois (103)", "Goumois (JU-Doubs)"),
    1165: ("Damvant (früher 109)", "Damvant-Réchésy"),
    1183: ("Allschwil (140)", "Allschwil-Hégenheim"),
    1190: ("Hofstetten-Flüh", "Leymen-Flühen"),
}

# Official 2021 office locations exposed these mislabeled collected points.
# Block the known wrong aliases even if the optional official file is absent;
# it must not silently restore the earlier assignments on another machine.
REVIEW_REQUIRED = {code: "collected_label_identity_conflict" for code in [1013, 1067, 1095, 1109, 1119, 1158, 1183]}
# The supplied Schaanwald label says FL–Austria, whereas its point is on the
# Swiss border. Proximity to another road point does not resolve that identity.
REVIEW_REQUIRED[1070] = "crossing_identity_requires_review"


def configure(context):
    context.config("data_path")
    context.config("cross_border_reference_path", None)
    context.stage("data.cross_border.interview_places")
    context.stage("data.spatial.swiss_border")


def execute(context):

    # Read the official names and optional individual labels collected locally.

    survey_root = Path(context.config("data_path")) / "crossborder"
    root = reference_directory(context.config("data_path"), context.config("cross_border_reference_path"))
    names = read_road_exit_names(survey_root / "AuGQPV_2021" / "AGQPV21_finale_Auswertungsdatenbank_Codebuch.xlsx")
    candidate_path = root / "crossings_1.gpkg"
    candidates = gpd.read_file(candidate_path) if candidate_path.exists() else None
    official_path = root / "road_exit_points_official.gpkg"
    official = gpd.read_file(official_path, layer = "road_exits") if official_path.exists() else None
    reviewed_path = root / "road_exit_points_reviewed.json"
    reviewed = json.loads(reviewed_path.read_text()) if reviewed_path.exists() else None
    if candidates is None:
        logger.warning("Optional collected crossing labels are absent at %s; those local aliases are unavailable.", candidate_path)
    points = context.stage("data.cross_border.interview_places")
    border = context.stage("data.spatial.swiss_border").iloc[0]

    # Reuse canonical IDs and geometry; do not edit the entry-sampling reference.

    mapped, audit = build_exit_reference(points, names, candidates, border)
    if official is not None:
        mapped, audit = apply_official_reference(mapped, audit, points, names, official, border, reviewed = reviewed)
    audit.to_csv(Path(context.path()) / "road_exit_reference.csv", index = False)
    logger.info("Reported road-exit reference: %s", audit["status"].value_counts().to_dict())

    # Return only located exits; rejected codes remain visible in the audit.

    return mapped


def crossing_name(value):
    """Normalize spelling only; a group name must never become a single site."""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", value)


def build_exit_reference(points, names, candidates, border):
    """Check explicit aliases against existing canonical road locations.

    Major crossings keep the previous exact-name mapping. An alias must have
    one road-only collected label, no conflicting use of its coordinate, and
    exactly one nearby canonical road point in the same interview group.
    """
    road = points.loc[points["label"].eq("road")].to_crs(2056).copy()
    exact = {name: group for name, group in road.groupby(road["interview_place"].map(crossing_name))}
    if candidates is not None:
        required = {"candidate_label", "interview_place", "modes", "geometry"}
        if not required.issubset(candidates.columns):
            raise ValueError("crossings_1.gpkg requires columns %s" % sorted(required))
        candidates = candidates.to_crs(2056).copy()
        car = candidates["modes"].fillna("").map(lambda value: "car" in {mode.strip() for mode in value.split(",")})
        candidates = candidates.loc[car].copy()
        if candidates.geometry.isna().any() or not candidates.geom_type.eq("Point").all():
            raise ValueError("Road candidates in crossings_1.gpkg must have point geometry.")
        candidates["name_key"] = candidates["candidate_label"].map(crossing_name)
        candidates["coordinate_key"] = candidates.geometry.to_wkb(hex = True)
        conflicts = candidates.groupby("coordinate_key")["name_key"].nunique().gt(1)
        candidates["coordinate_conflict"] = candidates["coordinate_key"].map(conflicts)
    rows, audit = [], []
    for code, name in sorted(names.items()):
        record = dict(survey_exit_code = code, survey_exit_name = name, candidate_label = None,
                      status = "no_reviewed_label", nearest_distance_m = None, boundary_distance_m = None,
                      nearby_points = 0, border_crossing_point_id = None)
        selected = exact.get(crossing_name(name))
        if selected is not None:
            record["status"] = "exact_name"
        elif code in EXIT_LABELS:
            expected_name, label = EXIT_LABELS[code]
            record["candidate_label"] = label
            if crossing_name(name) != crossing_name(expected_name):
                record["status"] = "codebook_name_changed"
            elif candidates is None:
                record["status"] = "candidate_file_missing"
            else:
                matches = candidates[candidates["name_key"].eq(crossing_name(label))]
                if len(matches) != 1:
                    record["status"] = "candidate_missing" if matches.empty else "candidate_label_ambiguous"
                else:
                    candidate = matches.iloc[0]
                    group = road[road["interview_place"].map(crossing_name).eq(crossing_name(candidate.interview_place))]
                    distances = group.geometry.distance(candidate.geometry)
                    nearby = group[distances.le(REFERENCE_RADIUS)]
                    record.update(nearest_distance_m = distances.min(), nearby_points = len(nearby),
                                  boundary_distance_m = candidate.geometry.distance(border.boundary))
                    if code in REVIEW_REQUIRED:
                        record["status"] = REVIEW_REQUIRED[code]
                    elif candidate.coordinate_conflict:
                        record["status"] = "conflicting_coordinate_labels"
                    elif record["boundary_distance_m"] > REFERENCE_RADIUS:
                        record["status"] = "candidate_far_from_border"
                    elif nearby.empty:
                        record["status"] = "no_nearby_canonical_point"
                    elif len(nearby) > 1:
                        record["status"] = "canonical_point_ambiguous"
                    else:
                        selected = nearby
                        record["status"] = "collected_label"
        if selected is not None:
            selected = selected.copy()
            selected["survey_exit_code"] = code
            selected["survey_exit_name"] = name
            selected["exit_mapping_source"] = record["status"]
            selected["candidate_label"] = record["candidate_label"]
            rows.append(selected)
            record["border_crossing_point_id"] = ",".join(selected["border_crossing_point_id"])
        audit.append(record)
    empty = road.iloc[:0].assign(survey_exit_code = pd.Series(dtype = int), survey_exit_name = None,
                                exit_mapping_source = None, candidate_label = None)
    mapped = gpd.GeoDataFrame(pd.concat(rows, ignore_index = True) if rows else empty, geometry = "geometry", crs = road.crs)
    return mapped, pd.DataFrame(audit, columns = ["survey_exit_code", "survey_exit_name", "candidate_label", "status",
        "nearest_distance_m", "boundary_distance_m", "nearby_points", "border_crossing_point_id"])


def apply_official_reference(mapped, audit, points, names, official, border, reviewed = None):
    """Use dated official office locations to identify existing crossing points.

    Unlike the unverified collected labels, these records identify a named site
    independently of the canonical layer's survey-group labels. The nearest
    canonical road point within 500 m represents that site; multiple nearby
    lane points need not mean different crossings. Visually reviewed offices use
    their own exit-only anchors. The audit records distances, alternatives,
    provenance and any correction to an earlier local alias.
    """
    required = {"survey_exit_code", "survey_exit_name", "source_name", "source_id",
                "reference_year", "source_url", "geometry"}
    if not required.issubset(official.columns):
        raise ValueError("Official road exits require columns %s" % sorted(required))
    if official["survey_exit_code"].isna().any() or official["survey_exit_code"].duplicated().any():
        raise ValueError("Official road exits must identify each survey code at most once.")
    official = official.to_crs(2056)
    if official.geometry.isna().any() or not official.geom_type.eq("Point").all():
        raise ValueError("Official road exits must have point geometry.")
    if official[["source_name", "source_id", "source_url", "reference_year"]].isna().any().any():
        raise ValueError("Official road exits must retain their source identity, year and URL.")
    road = points.loc[points["label"].eq("road")].to_crs(2056).copy()
    approved = validate_reviewed_sites(reviewed, official) if reviewed is not None else {}
    audit = audit.set_index("survey_exit_code").copy()
    records = []
    replaced = []
    for office in official.itertuples(index = False):
        code = office.survey_exit_code
        if code not in names:
            continue
        # Preserve the previous major-crossing contract. Official locations
        # supplement smaller exits and supersede their less reliable aliases.
        if audit.loc[code, "status"] == "exact_name":
            continue
        replaced.append(code)
        record = audit.loc[code].to_dict()
        record.update(previous_status = record["status"], previous_point_ids = record["border_crossing_point_id"],
                      reference_name = office.source_name, reference_id = office.source_id,
                      reference_year = office.reference_year, reference_url = office.source_url,
                      reference_x = office.geometry.x, reference_y = office.geometry.y,
                      candidate_label = office.source_name, border_crossing_point_id = None,
                      boundary_distance_m = office.geometry.distance(border.boundary))
        selected = None
        distances = road.geometry.distance(office.geometry).sort_values(kind = "stable")
        record["nearby_points"] = int(distances.le(REFERENCE_RADIUS).sum())
        record["nearest_distance_m"] = distances.min()
        record["second_distance_m"] = distances.iloc[1] if len(distances) > 1 else None
        if crossing_name(office.survey_exit_name) != crossing_name(names[code]):
            record["status"] = "official_codebook_name_changed"
        elif code in approved:
            # A reviewed office need not lie on the surface national boundary.
            # Give it a distinct exit-only ID and retain its actual coordinate;
            # through_crossings must still determine the appropriate road link.
            review = approved[code]
            point_id = "BCP_OFFICIAL_" + re.sub(r"[^A-Za-z0-9]", "", str(office.source_id))
            selected = gpd.GeoDataFrame([dict(border_crossing_point_id = point_id,
                interview_place = office.survey_exit_name, label = "road", importance = 0.0,
                geometry = office.geometry)], geometry = "geometry", crs = road.crs)
            record.update(status = "reviewed_official_office", border_crossing_point_id = point_id,
                          border_scope = review["border_scope"], reviewed_by = reviewed["reviewed_by"],
                          reviewed_at = reviewed["reviewed_at"])
        elif record["boundary_distance_m"] > REFERENCE_RADIUS:
            record["status"] = "official_office_far_from_border"
        elif distances.empty or distances.iloc[0] > REFERENCE_RADIUS:
            record["status"] = "official_no_nearby_canonical_point"
        elif len(distances) > 1 and abs(distances.iloc[1] - distances.iloc[0]) < 1e-6:
            record["status"] = "official_canonical_point_tied"
        else:
            selected = road.loc[[distances.index[0]]].copy()
            record["status"] = "official_office"
            record["border_crossing_point_id"] = selected["border_crossing_point_id"].iloc[0]
        if selected is not None:
            selected["survey_exit_code"] = code
            selected["survey_exit_name"] = names[code]
            selected["exit_mapping_source"] = record["status"]
            selected["border_scope"] = record.get("border_scope", "swiss")
            selected["candidate_label"] = office.source_name
            for field in ["reference_name", "reference_id", "reference_year", "reference_url", "reference_x", "reference_y"]:
                selected[field] = record[field]
            records.append(selected)
        for field, value in record.items():
            if field not in audit.columns:
                audit[field] = None
            audit.loc[code, field] = value
    # Do not fall back to a contradictory local alias when an official office
    # cannot be attached confidently (e.g. an inland customs office or FL–AT).
    retained = mapped.loc[~mapped["survey_exit_code"].isin(replaced)]
    mapped = gpd.GeoDataFrame(pd.concat([retained] + records, ignore_index = True), geometry = "geometry", crs = road.crs)
    return mapped, audit.reset_index()


def validate_reviewed_sites(reviewed, official):
    """Bind human review to a dated source identity and its exact coordinates."""
    if reviewed.get("schema_version") != 1 or not reviewed.get("reviewed_by") or not reviewed.get("reviewed_at"):
        raise ValueError("Reviewed road exits require schema_version 1, reviewer and review date.")
    approved = {}
    for site in reviewed.get("sites", []):
        if site.get("border_scope") not in {"swiss", "swiss_liechtenstein"}:
            raise ValueError("Reviewed road exit has an unknown border_scope.")
        for code in site["survey_exit_codes"]:
            rows = official[official["survey_exit_code"].eq(code)]
            if code in approved or len(rows) != 1:
                raise ValueError("Reviewed road exit code %s is duplicated or absent from the official reference." % code)
            row = rows.iloc[0]
            if (str(row.source_id) != str(site["source_id"]) or int(row.reference_year) != int(site["reference_year"])
                    or crossing_name(row.source_name) != crossing_name(site["source_name"])
                    or abs(row.geometry.x - site["x"]) > 0.01 or abs(row.geometry.y - site["y"]) > 0.01):
                raise ValueError("Official location for reviewed code %s changed; review the updated source before reusing approval." % code)
            approved[code] = site
    return approved


def read_liechtenstein_border(data_path):
    """Use the same national-boundary source as data.spatial.swiss_border."""
    countries = gpd.read_file(Path(data_path) / "spatial/country/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp").to_crs(2056)
    geometry = countries.loc[countries["NAME"].eq("Liechtenstein"), "geometry"]
    if len(geometry) != 1:
        raise ValueError("Reviewed Liechtenstein exits require one Liechtenstein boundary in the country layer.")
    return geometry.iloc[0]


def validate(context):
    """Invalidate this reference if optional labels or the codebook are updated."""
    survey_root = Path(context.config("data_path")) / "crossborder"
    root = reference_directory(context.config("data_path"), context.config("cross_border_reference_path"))
    paths = [root / "crossings_1.gpkg", root / "road_exit_points_official.gpkg", root / "road_exit_points_reviewed.json",
             survey_root / "AuGQPV_2021" / "AGQPV21_finale_Auswertungsdatenbank_Codebuch.xlsx"]
    return file_versions(paths)
