"""Inspect assigned border demand before whole-population routing/preparation."""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from data.cross_border.reference_files import reference_directory, file_versions
from data.cross_border.count_references import counts_path, check_reference, build_sites
from data.cross_border.count_assignment_inputs import configure_foreign, configure_swiss, assigned_foreign, assigned_swiss, active, STAGE
from data.utils import coerce_boolean_series

logger = logging.getLogger("synpp")
CATEGORIES = ("foreign_visitor", "swiss", "transit")
UNKNOWN = "unresolved_exit"


def configure(context):
    context.config("analysis_path")
    context.config("data_path")
    context.config("cross_border_reference_path", None)
    context.config("cross_border_assignment_counts_file", None)
    include_sample = context.config("cross_border_assignment_include_sample", False)
    context.config("input_downsampling")
    context.config("specific_day_scenario", default="workday")
    for stage in ("data.cross_border.interview_places", "data.cross_border.road_candidates",
                  "data.cross_border.road_exit_points", "data.cross_border.generate_od",
                  "data.cross_border.swiss_residents_od"):
        context.stage(stage)
    configure_foreign(context)
    # Exact sampled counts require the later licence/car models and scaling.
    # The default stops where Swiss crossing events have already been assigned.
    if include_sample:
        for stage in ("data.cross_border.sample", "data.cross_border.through_crossings",
                      "data.census.selected", "synthesis.population.sampled"):
            context.stage(stage)
    else:
        if context.config("census", default="statpop") != "statpop":
            raise ValueError("Early crossing analysis requires census=statpop; "
                             "use cross_border_assignment_include_sample=true for another census")
        configure_swiss(context)


def execute(context):

    # Identify physical choices and preserve the source GeoPackage FIDs.

    root = reference_directory(context.config("data_path"), context.config("cross_border_reference_path"))
    source = gpd.read_file(root / "border_interview_places_road_and_rail.gpkg", fid_as_index=True)
    points = context.stage("data.cross_border.interview_places")
    check_reference(source, points)
    roads = context.stage("data.cross_border.road_candidates")
    exits = context.stage("data.cross_border.road_exit_points")
    sites, canonical = build_sites(source, points, roads, exits)
    probability = float(context.config("input_downsampling"))
    if not np.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("Early crossing analysis requires 0 < input_downsampling <= 1")

    # Count explicit assignments, not routed link passages or donor-pool rows.

    include_sample = context.config("cross_border_assignment_include_sample")
    swiss_stage = "data.census.selected" if include_sample else "synthesis.population.models.cross_border"
    parts = [assignment_events(assigned_foreign(context),
                               context.stage(swiss_stage) if include_sample else assigned_swiss(context), roads, exits, canonical, "full")]
    if include_sample:
        parts.append(assignment_events(context.stage("data.cross_border.sample"),
                                       context.stage("synthesis.population.sampled"), roads, exits,
                                       canonical, "sample", context.stage("data.cross_border.through_crossings")))
    events = pd.concat(parts, ignore_index=True)
    missing = sorted(set(events.point_id) - set(sites.point_id))
    if missing:
        # Never lose unlocated exits or guess their interview group from entry.
        sites = pd.concat([sites, pd.DataFrame([
            dict(point_id=p, fid=None, name="Unresolved crossing", interview_place="Unresolved",
                 kind="unresolved", x=np.nan, y=np.nan) for p in missing
        ])], ignore_index=True)
    counts = summarise(events, sites, probability, include_sample=include_sample)

    # Compare absolute observed counts; production uses their within-group shares.

    reference_path = counts_path(context)
    if context.config("cross_border_assignment_counts_file") and not reference_path.is_file():
        raise FileNotFoundError("Configured crossing-count reference is missing: %s" % reference_path)
    references = json.loads(reference_path.read_text()) if reference_path.exists() else []
    reference_cohort = "expanded" if include_sample else "full"
    counts, reference_audit = attach_references(counts, references, context.config("specific_day_scenario"),
                                               reference_cohort=reference_cohort)
    controls = survey_controls(context)
    places = place_summary(events, sites, probability, include_sample=include_sample)
    places = places.merge(controls, on="interview_place", how="outer")

    # Export aggregate results only; person identifiers stay out of the report.

    output = Path(context.config("analysis_path")) / "cross_border_assignment"
    output.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output / "crossings.csv", index=False)
    places.to_csv(output / "interview_places.csv", index=False)
    events.groupby(["cohort", "point_id", "category", "direction", "source"], dropna=False).size().rename(
        "assigned_events").reset_index().to_csv(output / "assignment_counts.csv", index=False)
    reference_audit.to_csv(output / "reference_checks.csv", index=False)
    diagnostics = {
        "sampling_fraction": probability,
        "include_sample": include_sample,
        "swiss_population_stage": swiss_stage,
        "full_population_basis": "Population used for sampling (after configured scaling)" if include_sample
                                 else "Cross-border model population (before scaling and sampling)",
        "reference_cohort": reference_cohort,
        "day": context.config("specific_day_scenario"),
        "reference_file": str(reference_path),
        "reference_file_present": reference_path.exists(),
        "unresolved_events": events[events.point_id.isin(sites.loc[sites.kind.eq("unresolved"), "point_id"])].groupby(["cohort", "source"]).size().to_dict(),
        "scope": "Assigned survey car category (including motorcycles); passengers and PT excluded. "
                 "Ordinary Swiss/French traffic, freight, later exclusions and route repair are not included.",
    }
    # Make any adaptive geographic allowance visible in the report, rather
    # than requiring users to discover it in the allocation-stage cache.
    diagnostics["count_assignment_groups"] = context.stage(STAGE).get("groups", []) if active(context) else []
    diagnostics["unresolved_events"] = {"/".join(k): int(v) for k, v in diagnostics["unresolved_events"].items()}
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    write_report(counts, places, diagnostics, output / "index.html")
    logger.info("Early crossing assignment: %s; no whole-population routing was run by this stage", output)

    # Return the report directory for downstream analyses.

    return str(output)


def validate(context):
    root = reference_directory(context.config("data_path"), context.config("cross_border_reference_path"))
    path = root / "border_interview_places_road_and_rail.gpkg"
    return file_versions([path, Path(str(path) + "-wal"), counts_path(context),
                          Path(__file__).with_name("assignment.html.template")])


def assignment_events(foreign, swiss, roads, exits, canonical, cohort, through=None):
    rows = []

    def add(point, category, direction, source, entry_place):
        point = canonical.get(str(point), str(point)) if pd.notna(point) else UNKNOWN
        rows.append((cohort, point, category, direction, source, entry_place))

    # Unique exit-code/link matches are safe; ambiguous aliases stay unresolved.
    code_points = exits.assign(point_id=exits.border_crossing_point_id.astype(str).map(
        lambda p: canonical.get(p, p))).groupby("survey_exit_code").point_id.agg(lambda x: sorted(set(x)))
    code_points = {int(k): v[0] for k, v in code_points.items() if len(v) == 1}
    link_points = roads.loc[roads.label.eq("road")].groupby("exit_link_id").canonical_point_id.agg(lambda x: sorted(set(x)))
    link_points = {str(k): v[0] for k, v in link_points.items() if len(v) == 1}
    lookup = through.set_index("cross_border_person_id").to_dict("index") if through is not None else {}
    car = foreign.loc[foreign.trip_mode.eq("car")]
    if car.cross_border_person_id.duplicated().any():
        raise ValueError("Foreign demand must have one row per generated person")
    for r in car.itertuples():
        category = {"From-To": "foreign_visitor", "Through": "transit"}.get(r.label)
        if category is None:
            raise ValueError("Unexpected foreign trip category: %s" % r.label)
        add(r.interview_point_id, category, "entry", "assigned", r.interview_place)
        if category == "foreign_visitor":
            add(r.interview_point_id, category, "exit", "return_assumed", r.interview_place)
            continue
        record = lookup.get(r.cross_border_person_id)
        point, source = None, "unresolved_full_exit" if through is None else "unresolved_sample_exit"
        if record is not None:
            facility = record.get("exit_facility_id")
            if pd.notna(facility):
                point = str(facility).removesuffix("_exit")
                source = "reported_exit"
            else:
                link = record.get("route_exit_link_id")
                point = link_points.get(str(link))
                source = "inferred_exit" if point is not None else "inferred_exit_outside_reference"
        elif through is None:
            code = getattr(r, "survey_exit_code", None)
            if pd.notna(code):
                point = code_points.get(int(code))
                if point is not None:
                    source = "reported_unique_exit"
        add(point, category, "exit", source, r.interview_place)

    # Swiss rows represent individual directional events, not return tours.
    selected = coerce_boolean_series(swiss.is_crossing_the_border, name="is_crossing_the_border")
    selected &= swiss.border_crossing_trip_mode.eq("car") & swiss.cross_border_person_id.notna()
    fields = ["interview_point_id", "border_crossing_direction", "border_interview_place"]
    for point, direction, place in swiss.loc[selected, fields].itertuples(index=False, name=None):
        if direction not in ("entry", "exit"):
            raise ValueError("Selected Swiss crosser has no valid direction")
        add(point, "swiss", direction, "selected_resident", place)
    return pd.DataFrame(rows, columns=["cohort", "point_id", "category", "direction", "source", "survey_entry_place"])


def summarise(events, sites, probability, include_sample=True):
    result = sites.copy().set_index("point_id")
    for cohort in ("full", "sample"):
        selected = events[events.cohort.eq(cohort)]
        for category in CATEGORIES:
            for direction in ("entry", "exit"):
                n = selected[selected.category.eq(category) & selected.direction.eq(direction)].groupby("point_id").size()
                col = f"{cohort}_{category}_{direction}"
                result[col] = n.reindex(result.index, fill_value=0).astype(int)
                if cohort == "sample":
                    result[f"expanded_{category}_{direction}"] = result[col] / probability
        for direction in ("entry", "exit"):
            result[f"{cohort}_{direction}"] = sum(result[f"{cohort}_{c}_{direction}"] for c in CATEGORIES)
        result[f"{cohort}_total"] = result[f"{cohort}_entry"] + result[f"{cohort}_exit"]
    for direction in ("entry", "exit", "total"):
        result[f"expanded_{direction}"] = result[f"sample_{direction}"] / probability
    if not include_sample:
        # Uncomputed sample counts must be blank, not falsely reported as zero.
        result[result.columns[result.columns.str.startswith(("sample_", "expanded_"))]] = np.nan
    return result.reset_index()


def place_summary(events, sites, probability, include_sample=True):
    # Aggregate physical entry/exit destinations; never attribute a transit exit
    # to the interview group where its entry was observed.
    located = events.merge(sites[["point_id", "interview_place"]], on="point_id", validate="many_to_one")
    rows = []
    for place, group in located.groupby("interview_place", dropna=False):
        row = dict(interview_place=place)
        for cohort in ("full", "sample"):
            for direction in ("entry", "exit"):
                mask = group.cohort.eq(cohort) & group.direction.eq(direction)
                row[f"{cohort}_{direction}"] = int(mask.sum())
                for category in CATEGORIES:
                    row[f"{cohort}_{category}_{direction}"] = int((mask & group.category.eq(category)).sum())
        for key, value in list(row.items()):
            if key.startswith("sample_"):
                row[key.replace("sample_", "expanded_", 1)] = value / probability
        if not include_sample:
            for key in row:
                if key.startswith(("sample_", "expanded_")):
                    row[key] = np.nan
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["interview_place"])


def survey_controls(context):
    frames = []
    for name, category in (("generate_od", "foreign"), ("swiss_residents_od", "swiss")):
        path = Path(context.path("data.cross_border." + name))
        ledger = pd.read_csv(path / "vehicle_calibration.csv")
        names = pd.read_csv(path / "survey_demand_audit.csv", usecols=["interview_place_id", "interview_place"]).drop_duplicates()
        if names.interview_place_id.duplicated().any():
            raise ValueError("Survey interview-place IDs have inconsistent names")
        ledger = ledger.merge(names, on="interview_place_id", how="left", validate="one_to_one")
        ledger.interview_place = ledger.interview_place.fillna("Unmatched survey ID " + ledger.interview_place_id.astype(str))
        ledger = ledger[["interview_place", "entries_2015", "entries_2021", "target_entries"]].rename(
            columns={c: category + "_" + c for c in ["entries_2015", "entries_2021", "target_entries"]})
        frames.append(ledger)
    return frames[0].merge(frames[1], on="interview_place", how="outer", validate="one_to_one")


def attach_references(counts, references, day, reference_cohort="expanded"):
    result = counts.copy()
    for col in ("official_low", "official_high"):
        result[col] = np.nan
    for col in ("official_period", "official_day_type", "official_scope", "official_source", "reference_note"):
        result[col] = ""
    audit, used = [], set()
    for ref in references:
        matches = result.fid.eq(ref["fid"])
        if int(ref["fid"]) in used:
            raise ValueError("Duplicate traffic-count reference FID")
        used.add(int(ref["fid"]))
        if ref.get("keep_assignment") is True:
            if ref.get("low") is not None or ref.get("high") is not None:
                raise ValueError("Unmeasured retained sites must not supply count targets")
            if matches.sum() == 1:
                i = result.index[matches][0]
                result.loc[i, ["name", "reference_note"]] = [ref["name"], ref.get("note", "")]
            audit.append(dict(fid=ref["fid"], status="explicitly_unmeasured"))
            continue
        low, high = float(ref["low"]), float(ref.get("high", ref["low"]))
        if not np.isfinite([low, high]).all() or low <= 0 or high < low:
            raise ValueError("Traffic-count references require 0 < low <= high")
        if matches.sum() != 1:
            audit.append(dict(fid=ref["fid"], status="missing_or_noncanonical_fid")); continue
        i = result.index[matches][0]
        distance = float(np.hypot(result.loc[i, "x"] - ref["x"], result.loc[i, "y"] - ref["y"]))
        if not np.isfinite(distance) or distance > 50:
            audit.append(dict(fid=ref["fid"], status="coordinate_mismatch", distance_m=distance)); continue
        result.loc[i, ["name", "official_low", "official_high", "official_period", "official_day_type", "official_scope", "official_source", "reference_note"]] = [
            ref["name"], low, high, ref["period"], ref["day_type"], ref["vehicle_scope"], ref["source_url"], ref.get("note", "")]
        audit.append(dict(fid=ref["fid"], status="matched", distance_m=distance))
    # Ratios remain indicative: observed all-vehicle counts include traffic
    # absent from this early assigned-car report, even when day types agree.
    weekday = day in ("workday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    result["day_type_matches"] = result.official_day_type.eq("weekday" if weekday else "weekend") & result.official_low.notna()
    result["reference_cohort"] = reference_cohort
    result["indicative_ratio_low"] = result[reference_cohort + "_total"] / result.official_high
    result["indicative_ratio_high"] = result[reference_cohort + "_total"] / result.official_low
    return result, pd.DataFrame(audit, columns=["fid", "status", "distance_m"])


def write_report(counts, places, diagnostics, path):
    transform = Transformer.from_crs(2056, 4326, always_xy=True)
    records = []
    for row in counts.to_dict("records"):
        if pd.notna(row.get("x")) and pd.notna(row.get("y")):
            row["lon"], row["lat"] = transform.transform(row["x"], row["y"])
        records.append(row)
    # pandas serialises NaN as JSON null; escape closing script tags in inputs.
    payload = json.dumps(dict(points=json.loads(pd.DataFrame(records).to_json(orient="records")),
                              places=json.loads(places.to_json(orient="records")), metadata=diagnostics)).replace("</", "<\\/")
    template = Path(__file__).with_name("assignment.html.template").read_text()
    path.write_text(template.replace("__ASSIGNMENT_DATA__", payload), encoding="utf-8")
