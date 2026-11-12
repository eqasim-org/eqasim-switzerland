"""Read-only comparison of prepared Swiss-resident plans and survey benchmarks.

Run as a module with --config and --output. No Synpp stages are executed.
"""

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import yaml
from lxml import etree


SECONDARY = {"shop", "leisure", "other", "work_secondary", "education_secondary", "home_secondary"}


def dependencies(index, key):
    seen = set()

    def visit(item):
        if item in seen:
            return
        seen.add(item)
        for child in index[item]["dependencies"]:
            visit(child)

    visit(key)
    return seen


class Snapshot:
    def __init__(self, root, index, key):
        self.root, self.index, self.key = root, index, key
        self.keys = dependencies(index, key)
        self.config = index[key]["config"]
        self.folder = root / (key + ".cache")

    def stage_key(self, name):
        keys = [key for key in self.keys if key.split("__")[0] == name]
        if len(keys) != 1:
            raise ValueError(f"Expected one dependency for {name}: {keys}")
        return keys[0]

    def load(self, name):
        with (self.root / (self.stage_key(name) + ".p")).open("rb") as stream:
            return pickle.load(stream)

    def directory(self, name):
        return self.root / (self.stage_key(name) + ".cache")


def assigned_survey_reference(snapshot):
    reference = snapshot.load("data.cross_border.swiss_residents_od")
    stage = "data.cross_border.count_assignment"
    if any(key.split("__")[0] == stage for key in snapshot.keys):
        aliases = snapshot.load(stage)["swiss_facilities"]
        reference = pd.concat([reference, aliases], ignore_index=True)
    if reference.cross_border_person_id.duplicated().any():
        raise ValueError("Assigned survey reference IDs must be unique")
    return reference


def attach_survey_purposes(crossings, reference):
    result = crossings.merge(reference[["cross_border_person_id", "trip_purpose"]],
                             on="cross_border_person_id", how="left", validate="many_to_one",
                             indicator="survey_match")
    unmatched = result.explicit & result.survey_match.ne("both")
    if unmatched.any():
        raise ValueError("Unresolved survey references for %d explicit crossing events" % unmatched.sum())
    return result.drop(columns="survey_match")


def read_plans(snapshot, output, label):
    population = snapshot.load("synthesis.population.sampled").copy()
    population["person_id"] = population.person_id.astype(str)
    ids = set(population.person_id)
    filename = snapshot.folder / (snapshot.config["output_prefix"] + "population.xml.gz")
    stat = filename.stat()
    signature = [str(filename), stat.st_size, stat.st_mtime_ns]
    cache = output / (label + "_extraction.pkl")
    if cache.exists():
        with cache.open("rb") as stream:
            saved = pickle.load(stream)
        if saved[0] == signature:
            return population, saved[1], saved[2], signature
    rows, present = [], []
    with gzip.open(filename, "rb") as stream:
        for _, person in etree.iterparse(stream, events=["end"], tag="person"):
            pid = person.get("id")
            if pid in ids:
                present.append(pid)
                plans = person.findall("plan")
                plan = next((p for p in plans if p.get("selected") == "yes"), plans[0])
                modes, idx = [], 0
                for child in plan:
                    if child.tag == "leg":
                        modes.append(child.get("mode"))
                    elif child.tag == "activity" and not child.get("type", "").endswith("interaction"):
                        mode = next((m for m in ["car", "car_passenger", "pt", "bike", "walk"] if m in modes), modes[0] if modes else None)
                        rows.append((pid, idx, child.get("type"), float(child.get("x", "nan")),
                                     float(child.get("y", "nan")), mode))
                        modes, idx = [], idx + 1
            person.clear()
            while person.getprevious() is not None:
                del person.getparent()[0]
    activities = pd.DataFrame(rows, columns=["person_id", "activity_index", "purpose", "x", "y", "mode"])
    with cache.open("wb") as stream:
        pickle.dump((signature, activities, present), stream)
    return population, activities, present, signature


def locate(frame, border):
    frame = frame.copy()
    frame["valid"] = np.isfinite(frame[["x", "y"]]).all(axis=1)
    frame["outside"] = frame.valid & ~shapely.contains_xy(border, frame.x, frame.y)
    frame["outside_100m"] = frame.valid & ~shapely.contains_xy(border.buffer(100), frame.x, frame.y)
    frame["secondary"] = frame.purpose.isin(SECONDARY)
    frame["secondary_abroad"] = frame.secondary & frame.outside
    frame["secondary_abroad_100m"] = frame.secondary & frame.outside_100m
    return frame


def transitions(activities):
    a = activities.sort_values(["person_id", "activity_index"]).copy()
    grouped = a.groupby("person_id", sort=False)
    a["origin_purpose"] = grouped.purpose.shift()
    a["origin_outside"] = grouped.outside.shift()
    a["origin_outside_100m"] = grouped.outside_100m.shift()
    a["origin_valid"] = grouped.valid.shift().fillna(False).astype(bool)
    if "in_overlap_region" in a:
        a["both_inside_overlap_region"] = a.in_overlap_region & grouped.in_overlap_region.shift().fillna(False).astype(bool)
    a = a.loc[a.activity_index.gt(0) & a.valid & a.origin_valid].copy()
    a["explicit"] = a.purpose.eq("border") | a.origin_purpose.eq("border")
    # Border endpoints represent the crossing even if the facility lies inside CH.
    a["crossing"] = a.explicit | a.outside.ne(a.origin_outside)
    a["crossing_100m"] = a.explicit | a.outside_100m.ne(a.origin_outside_100m)
    a["direction"] = np.where(a.purpose.eq("border") | (~a.explicit & a.outside), "exit", "entry")
    a["crossing_purpose"] = np.where(a.outside, a.purpose, a.origin_purpose)
    a.loc[a.explicit, "crossing_purpose"] = "border"
    return a


def weighted_person_total(activities, population, flag):
    return population.loc[population.person_id.isin(activities.loc[activities[flag], "person_id"]), "weight"].sum()


def summarize(label, population, activities, trips):
    eligible = population.loc[population.age.ge(6)]
    a = activities.loc[activities.person_id.isin(eligible.person_id)]
    t = trips.loc[trips.person_id.isin(eligible.person_id)]
    metrics = {
        "residents_age6plus": eligible.weight.sum(),
        "residents_with_secondary_abroad": weighted_person_total(a, eligible, "secondary_abroad"),
        "residents_with_secondary_abroad_100m": weighted_person_total(a, eligible, "secondary_abroad_100m"),
        "residents_with_crossing": weighted_person_total(t, eligible, "crossing"),
        "secondary_activities_abroad": a.loc[a.secondary_abroad, "weight"].sum(),
        "secondary_activities_abroad_100m": a.loc[a.secondary_abroad_100m, "weight"].sum(),
        "crossing_events": t.loc[t.crossing, "weight"].sum(),
        "crossing_events_100m": t.loc[t.crossing_100m, "weight"].sum(),
        "explicit_border_events": t.loc[t.explicit, "weight"].sum(),
        "ordinary_crossing_events": t.loc[t.crossing & ~t.explicit, "weight"].sum(),
        "ordinary_crossing_events_100m": t.loc[t.crossing_100m & ~t.explicit, "weight"].sum(),
        "ordinary_secondary_crossing_events": t.loc[t.crossing & ~t.explicit & t.crossing_purpose.isin(SECONDARY), "weight"].sum(),
    }
    return pd.DataFrame([dict(source=label, metric=k, expanded=v, per_1000_residents=v / metrics["residents_age6plus"] * 1000) for k, v in metrics.items()])


def grouped_counts(frame, keys, label):
    result = frame.groupby(keys, observed=True, dropna=False).agg(raw_rows=("weight", "size"), expanded=("weight", "sum")).reset_index()
    result.insert(0, "source", label)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--previous", help="Optional exact previous prepare cache key")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["working_directory"])
    index = json.loads((root / "pipeline.json").read_text())
    candidates = [k for k in index if k.startswith("matsim.simulation.prepare__") and (root / (k + ".p")).exists()]
    current = max(candidates, key=lambda k: index[k]["updated"])
    snapshots = [("Current", Snapshot(root, index, current))]
    if args.previous:
        snapshots.append(("Previous", Snapshot(root, index, args.previous)))
    border = snapshots[0][1].load("data.spatial.swiss_border").iloc[0]
    shapely.prepare(border)
    summary, purposes, modes, cantons, bands, provenance, validations = [], [], [], [], [], {}, {}
    overlap_tables, distance_tables = [], []
    survey_tables, donor_tables = [], []
    reference_population = None
    for label, snapshot in snapshots:
        print(f"Reading {label} prepared plans", flush=True)
        p, a, present, signature = read_plans(snapshot, output, label)
        p["weight"] = 1 / snapshot.config["input_downsampling"]
        p["explicit_resident"] = p.is_crossing_the_border.eq(1)
        p = p.loc[p.person_id.isin(present)].copy()
        home_points = shapely.points(p.home_x, p.home_y)
        home_distance = shapely.distance(home_points, border.simplify(100).boundary)
        p["home_border_band"] = pd.cut(home_distance / 1000, [0, 5, 10, 25, 50, np.inf], include_lowest=True,
                                         labels=["0–5 km", "5–10 km", "10–25 km", "25–50 km", "50+ km"])
        a = locate(a, border).merge(p[["person_id", "weight", "age", "canton_id", "home_border_band", "explicit_resident", "home_x", "home_y", "border_crossing_trip_mode", "cross_border_person_id"]], on="person_id", validate="many_to_one")
        region_files = snapshot.config.get("cross_border_exclude_shapefiles")
        region = None
        if snapshot.config.get("include_external_population") and region_files:
            if isinstance(region_files, str):
                region_files = [region_files]
            region = shapely.union_all([gpd.read_file(f).to_crs(2056).geometry.union_all() for f in region_files])
        a["in_overlap_region"] = shapely.contains_xy(region, a.x, a.y) if region is not None else False
        t = transitions(a)
        summary.append(summarize(label, p, a, t))
        aa = a.loc[a.age.ge(6) & a.secondary_abroad]
        distance_tables.append(distance_summary(aa, label))
        overlap_tables.append(grouped_counts(t.loc[t.age.ge(6) & t.crossing & ~t.explicit], ["both_inside_overlap_region", "mode", "crossing_purpose"], label))
        purposes.append(grouped_counts(aa, ["purpose"], label))
        modes.append(grouped_counts(t.loc[t.age.ge(6) & t.crossing], ["mode", "explicit", "direction"], label))
        cantons.append(grouped_counts(aa, ["canton_id", "purpose"], label))
        b = p.loc[p.age.ge(6)].copy()
        b["secondary_abroad"] = b.person_id.isin(aa.person_id)
        br = b.groupby("home_border_band", observed=True).apply(lambda g: pd.Series({"residents":g.weight.sum(), "residents_with_secondary_abroad":g.loc[g.secondary_abroad,"weight"].sum()})).reset_index()
        br["source"] = label
        bands.append(br)
        validations[label] = {"sampled_residents":len(present), "sampled_unique_residents":len(set(present)),
                              "sampled_stage_residents":len(snapshot.load("synthesis.population.sampled")),
                              "activities":len(a), "invalid_coordinates":int((~a.valid).sum()),
                              "secondary_abroad_all_ages":int(a.secondary_abroad.sum()),
                              "ordinary_secondary_abroad_in_explicit_residents":int((a.secondary_abroad & a.explicit_resident).sum()),
                              "observed_purposes":a.purpose.value_counts().to_dict()}
        assert len(present) == len(set(present))
        assert not a.duplicated(["person_id", "activity_index"]).any()
        provenance[label] = {"prepare_key":snapshot.key, "population_file_signature":signature,
                             "sample_fraction":snapshot.config["input_downsampling"],
                             "day":snapshot.config["specific_day_scenario"],
                             "dependencies":sorted(snapshot.keys)}
        # Donor geography distinguishes inherited international diaries from spillover.
        source_trips = snapshot.load("synthesis.population.trips")
        donor = source_trips[["person_id", "mz_person_id"]].drop_duplicates()
        donor.person_id = donor.person_id.astype(str)
        assert not donor.person_id.duplicated().any()
        mz_p = snapshot.load("data.microcensus.persons")
        mz_ids = set(mz_p.loc[mz_p.is_crossing_the_border | mz_p.is_outside_of_switzerland, "person_id"].astype(str))
        donor["international_mz_donor"] = donor.mz_person_id.astype(str).isin(mz_ids)
        d = aa.merge(donor[["person_id", "international_mz_donor"]], on="person_id", how="left", validate="many_to_one")
        donor_tables.append(grouped_counts(d, ["international_mz_donor", "purpose"], label))
        # Compare explicit prepared events with the actual retained survey controls.
        audit = pd.read_csv(snapshot.directory("synthesis.population.models.cross_border") / "crossing_event_assignment.csv")
        audit = audit.groupby(["trip_mode", "crossing_direction"])[["target_events", "assigned_events", "shortfall_events"]].sum().reset_index()
        actual = t.loc[t.explicit].groupby(["border_crossing_trip_mode", "direction"]).weight.sum().rename("prepared_explicit_events").reset_index().rename(columns={"border_crossing_trip_mode":"mode"})
        audit = audit.merge(actual, left_on=["trip_mode", "crossing_direction"], right_on=["mode", "direction"], how="outer").fillna(0)
        audit["source"] = label
        survey_tables.append(audit)
        if label == "Current":
            reference_population = p
            current_region = region
            from synthesis.population.models.cross_border import _journey_overlap
            od = snapshot.load("data.cross_border.swiss_residents_od").copy()
            overlap = _journey_overlap(od, gpd.GeoDataFrame(geometry=[region], crs=2056)).exclude if region is not None else pd.Series(False, index=od.index)
            od["scope"] = np.where(overlap, "overlap_region", "retained_region")
            od["weight"] = np.where(od.trip_mode.isin(["car", "car_passenger"]), 2., 1.)
            od.groupby(["scope", "trip_mode", "trip_purpose"]).weight.sum().rename("target_events").reset_index().to_csv(output / "survey_targets_by_scope_purpose.csv", index=False)
            attributed = attach_survey_purposes(t.loc[t.crossing], assigned_survey_reference(snapshot))
            attributed["attributed_purpose"] = np.where(attributed.explicit, attributed.trip_purpose, attributed.crossing_purpose)
            attributed["scope"] = np.where(~attributed.explicit & attributed.both_inside_overlap_region, "overlap_region", "retained_region")
            attributed["benchmark_mode"] = np.where(attributed.explicit, attributed.border_crossing_trip_mode, attributed["mode"])
            grouped_counts(attributed, ["scope", "benchmark_mode", "attributed_purpose", "explicit"], label).to_csv(output / "current_crossings_by_scope_purpose.csv", index=False)
            grouped_counts(attributed.loc[attributed.crossing_100m], ["scope", "benchmark_mode", "attributed_purpose", "explicit"], label).to_csv(output / "current_crossings_by_scope_purpose_100m.csv", index=False)
            source_locations = snapshot.load("synthesis.population.spatial.locations")
            source_locations.person_id = source_locations.person_id.astype(str)
            check = a.loc[a.secondary, ["person_id", "activity_index", "x", "y"]].merge(source_locations[["person_id", "activity_index", "geometry"]], on=["person_id", "activity_index"], how="left", validate="one_to_one")
            delta = np.hypot(check.x - check.geometry.map(lambda g:g.x), check.y - check.geometry.map(lambda g:g.y))
            validations[label]["secondary_coordinates_changed_over_2m_in_prepare"] = int(delta.gt(2).sum())
            a.loc[a.secondary_abroad].groupby(["purpose", "x", "y"]).agg(raw_activities=("person_id", "size"), expanded=("weight", "sum")).reset_index().sort_values("expanded", ascending=False).to_csv(output / "foreign_destination_clusters.csv", index=False)
            scope = pd.read_csv(snapshot.directory("synthesis.population.models.cross_border") / "crossing_event_scope.csv")
            scope.to_csv(output / "survey_scope.csv", index=False)
            # Aggregate source counts only; never export individual survey diaries.
            survey = pd.read_csv(snapshot.directory("data.cross_border.swiss_residents_od") / "survey_demand_audit.csv")
            survey.groupby(["trip_mode", "assignment_scope"]).agg(observations=("interview_id", "size"), raw_weight=("uncalibrated_weight", "sum"), calibrated_weight=("weight", "sum")).to_csv(output / "survey_weight_audit.csv")

    # MZ's person weights are normalized within the matching day to the current
    # age-eligible population, rather than summing weekday weights as a population.
    snapshot = snapshots[0][1]
    mp = snapshot.load("data.microcensus.persons").copy()
    day = snapshot.config["specific_day_scenario"]
    if day in {"workday", "weekend"}:
        mp = mp.loc[mp[day]].copy()
    else:
        raise ValueError("This report currently supports workday or weekend scenarios")
    mp.person_id = mp.person_id.astype(str)
    target = reference_population.loc[reference_population.age.ge(6), "weight"].sum()
    mp["weight"] = mp.person_weight / mp.person_weight.sum() * target
    mt = snapshot.load("data.microcensus.trips")[0].copy()
    mt.person_id = mt.person_id.astype(str)
    mt = mt.loc[mt.person_id.isin(mp.person_id)].sort_values(["person_id", "trip_id"]).reset_index(drop=True)
    first = mt.groupby("person_id", sort=False).head(1)
    origin = first[["person_id", "origin_purpose", "origin_x", "origin_y"]].rename(columns={"origin_purpose":"purpose", "origin_x":"x", "origin_y":"y"})
    origin["activity_index"], origin["mode"] = 0, None
    dest = mt[["person_id", "purpose", "destination_x", "destination_y", "mode"]].rename(columns={"destination_x":"x", "destination_y":"y"})
    dest["activity_index"] = mt.groupby("person_id").cumcount() + 1
    ma = locate(pd.concat([origin, dest], ignore_index=True), border)
    ma = ma.merge(mp[["person_id", "weight", "age", "canton_id", "home_x", "home_y"]], on="person_id", validate="many_to_one")
    # Use the survey's actual trip origins, which may differ from the preceding
    # trip destination after diary cleaning. Keep the observed MZ purposes.
    mx = locate(dest, border).merge(mp[["person_id", "weight", "age", "canton_id"]], on="person_id", validate="many_to_one")
    mx["origin_purpose"] = mt.origin_purpose.to_numpy()
    assert mx.person_id.tolist() == mt.person_id.tolist()
    mx["origin_outside"] = ~shapely.contains_xy(border, mt.origin_x.to_numpy(), mt.origin_y.to_numpy())
    mx["origin_outside_100m"] = ~shapely.contains_xy(border.buffer(100), mt.origin_x.to_numpy(), mt.origin_y.to_numpy())
    mx["explicit"] = False
    mx["crossing"] = mx.outside.ne(mx.origin_outside)
    mx["crossing_100m"] = mx.outside_100m.ne(mx.origin_outside_100m)
    mx["direction"] = np.where(mx.outside, "exit", "entry")
    mx["crossing_purpose"] = np.where(mx.outside, mx.purpose, mx.origin_purpose)
    mx["both_inside_overlap_region"] = (shapely.contains_xy(current_region, mt.origin_x.to_numpy(), mt.origin_y.to_numpy())
                                          & shapely.contains_xy(current_region, mt.destination_x.to_numpy(), mt.destination_y.to_numpy())) if current_region is not None else False
    assert not mx[["origin_outside", "crossing", "both_inside_overlap_region"]].isna().any().any()
    overlap_tables.append(grouped_counts(mx.loc[mx.crossing], ["both_inside_overlap_region", "mode", "crossing_purpose"], "MZ workday"))
    summary.append(summarize("MZ workday", mp, ma, mx))
    in_scope = ~ma.person_id.isin(mp.loc[mp.is_outside_of_switzerland, "person_id"])
    summary.append(summarize("MZ excluding all-abroad days", mp, ma.loc[in_scope], mx.loc[~mx.person_id.isin(mp.loc[mp.is_outside_of_switzerland, "person_id"])]))
    distance_tables.append(distance_summary(ma.loc[ma.secondary_abroad & in_scope], "MZ excluding all-abroad days"))
    purposes.append(grouped_counts(ma.loc[ma.secondary_abroad & in_scope], ["purpose"], "MZ excluding all-abroad days"))
    purposes.append(grouped_counts(ma.loc[ma.secondary_abroad], ["purpose"], "MZ workday"))
    modes.append(grouped_counts(mx.loc[mx.crossing], ["mode", "explicit", "direction"], "MZ workday"))
    cantons.append(grouped_counts(ma.loc[ma.secondary_abroad], ["canton_id", "purpose"], "MZ workday"))
    cantons.append(grouped_counts(ma.loc[ma.secondary_abroad & in_scope], ["canton_id", "purpose"], "MZ excluding all-abroad days"))
    validations["MZ"] = {"persons":len(mp), "persons_with_secondary_abroad":int(mp.person_id.isin(ma.loc[ma.secondary_abroad,"person_id"]).sum()),
                          "secondary_abroad_activities":int(ma.secondary_abroad.sum()), "person_weight_sum":float(mp.person_weight.sum()),
                          "population_normalization":float(target), "invalid_coordinates":int((~ma.valid).sum()),
                          "all_abroad_day_persons":int(mp.is_outside_of_switzerland.sum())}
    provenance["MZ"] = {"persons":snapshot.stage_key("data.microcensus.persons"), "trips":snapshot.stage_key("data.microcensus.trips"), "day":day}
    for name, tables in [("summary",summary),("secondary_purpose",purposes),("crossing_modes",modes),("secondary_cantons",cantons),
                         ("home_distance_bands",bands),("donor_geography",donor_tables),("survey_comparison",survey_tables),
                         ("ordinary_crossing_overlap",overlap_tables),("home_to_secondary_distance",distance_tables)]:
        pd.concat(tables, ignore_index=True).to_csv(output / (name + ".csv"), index=False)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "validation.json").write_text(json.dumps(validations, indent=2))
    print(pd.concat(summary).pivot(index="metric", columns="source", values="expanded").round(1).to_string(), flush=True)


def distance_summary(frame, label):
    frame = frame.copy()
    frame["distance_km"] = np.hypot(frame.x - frame.home_x, frame.y - frame.home_y) / 1000
    rows = []
    for purpose, group in frame.groupby("purpose"):
        group = group.loc[np.isfinite(group.distance_km)].sort_values("distance_km")
        cdf = (group.weight.cumsum() - group.weight / 2) / group.weight.sum()
        q = np.interp([.5, .9], cdf, group.distance_km)
        rows.append(dict(source=label, purpose=purpose, mean_km=np.average(group.distance_km, weights=group.weight), median_km=q[0], p90_km=q[1]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
