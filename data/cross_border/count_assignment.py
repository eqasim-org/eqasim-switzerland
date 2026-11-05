"""Joint full-population crossing shares, constrained by Swiss road geography."""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.special import softmax

from data.cross_border.count_assignment_inputs import FOREIGN, SWISS, REFERENCE
from data.cross_border.count_balancing import allocate, exact_support, CountAssignmentConflict
from data.cross_border.count_references import counts_path, check_reference, build_sites
from data.cross_border.count_routing import SwissRoadCosts
from data.cross_border.reference_files import reference_directory, file_versions

logger = logging.getLogger("synpp")
ROW_KEYS = ["category", "direction", "interview_id", "origin_x", "origin_y",
            "destination_x", "destination_y", "reported_exit_point"]


def configure(context):
    context.config("data_path")
    context.config("cross_border_reference_path", None)
    context.config("cross_border_assignment_counts_file", None)
    context.config("random_seed")
    context.config("cross_border_count_max_extra_minutes", 30.)
    for stage in (FOREIGN, SWISS, REFERENCE, "data.cross_border.interview_places",
                  "data.cross_border.road_candidates", "data.cross_border.road_exit_points",
                  "data.spatial.swiss_border", "matsim.scenario.network.pt"):
        context.stage(stage)


def execute(context):

    # Validate references against the active crossing release before changing demand.

    empty = {key: pd.DataFrame() for key in ("foreign", "swiss", "swiss_facilities")}
    empty["groups"] = []
    path = counts_path(context)
    if not path.exists():
        if context.config("cross_border_assignment_counts_file"):
            raise FileNotFoundError(path)
        logger.info("No crossing counts supplied; retaining existing crossing assignment")
        return empty
    root = reference_directory(context.config("data_path"), context.config("cross_border_reference_path"))
    source = gpd.read_file(root / "border_interview_places_road_and_rail.gpkg", fid_as_index=True)
    points = context.stage("data.cross_border.interview_places")
    check_reference(source, points)
    exits = context.stage("data.cross_border.road_exit_points")
    sites, canonical = build_sites(source, points, context.stage("data.cross_border.road_candidates"), exits)
    groups, audit = reference_groups(sites, json.loads(path.read_text()))
    output = Path(context.path())
    empty["groups"] = audit
    if not groups:
        pd.DataFrame(audit).to_csv(output / "groups.csv", index=False)
        return empty

    # Jointly account for foreign tours, actual Swiss events and known transit exits.

    foreign = context.stage(FOREIGN)
    reference = context.stage(REFERENCE)
    code_points = exits.assign(point_id=exits.border_crossing_point_id.map(lambda p: canonical.get(p, p)))
    code_points = code_points.groupby("survey_exit_code").point_id.agg(lambda v: sorted(set(v)))
    code_points = {int(k): v[0] for k, v in code_points.items() if len(v) == 1}
    exit_geometry = exits.drop_duplicates("border_crossing_point_id").set_index("border_crossing_point_id").geometry
    transit = foreign[foreign.label.eq("Through") & foreign.trip_mode.eq("car")]
    fixed = transit.survey_exit_code.map(code_points).value_counts()
    demand = collect_demand(context, set(groups), canonical, code_points)
    router = SwissRoadCosts(context)
    rng = np.random.default_rng(context.config("random_seed"))
    replacements, totals, anchors = [], [], []

    # Fit shares within each group; preserve row totals and freeze unsupported trips.

    for place, choices in groups.items():
        persons = demand[demand.interview_place.eq(place)].copy().reset_index(drop=True)
        if persons.empty:
            audit.append(dict(interview_place=place, status="no_assigned_road_people"))
            continue
        try:
            changed, summary, details, offsets = assign_group(persons, choices, fixed, exit_geometry, router, rng,
                maximum=context.config("cross_border_count_max_extra_minutes"))
        except (ValueError, RuntimeError) as error:
            # A complete count-controlled group must not silently revert to the
            # old draw or soft shares. Record the conflict before stopping, so
            # downstream analyses cannot present a successful but unfitted run.
            audit.append(dict(interview_place=place, status="count_assignment_conflict", reason=str(error),
                              max_extra_minutes=context.config("cross_border_count_max_extra_minutes")))
            pd.DataFrame(audit).to_csv(output / "groups.csv", index=False)
            raise CountAssignmentConflict("%s: %s. See %s" % (place, error, output / "groups.csv")) from error
        if summary["extra_minutes_limit"] > 20:
            logger.warning("Count shares preserved at %s: Swiss extra-time allowance expanded from 20 to %.1f minutes; "
                           "%.1f%% of movable expected crossings use the expanded range.",
                           place, summary["extra_minutes_limit"], summary["expected_share_over_20_pct"])
        replacements.append(changed)
        audit.append(dict(interview_place=place, **summary))
        totals.append(details.assign(interview_place=place))
        anchors.extend(offsets)

    # Export aggregate checks and compact overrides, preserving population identity.

    pd.DataFrame(audit).to_csv(output / "groups.csv", index=False)
    pd.DataFrame(anchors).to_csv(output / "road_anchors.csv", index=False)
    if totals:
        pd.concat(totals, ignore_index=True).to_csv(output / "crossing_counts.csv", index=False)
    if not replacements:
        return empty
    changed = pd.concat(replacements, ignore_index=True)
    changed = changed.merge(sites[["point_id", "x", "y"]], left_on="selected_point", right_on="point_id", validate="many_to_one")
    changed["interview_point_id"] = changed.selected_point
    changed["interview_geometry_point"] = gpd.points_from_xy(changed.x, changed.y)
    foreign_changes = changed[changed.category.ne("swiss")][["cross_border_person_id", "interview_point_id", "interview_geometry_point"]].copy()
    swiss = changed[changed.category.eq("swiss")].copy()
    swiss["source_cross_border_person_id"] = swiss.cross_border_person_id
    swiss["cross_border_person_id"] = swiss.source_cross_border_person_id + "@" + swiss.interview_point_id
    swiss["border_crossing_point"] = swiss.interview_geometry_point
    keys = ["source_cross_border_person_id", "interview_point_id", "interview_geometry_point", "border_crossing_point", "cross_border_person_id"]
    aliases = swiss[keys].drop_duplicates("cross_border_person_id").merge(
        reference.drop(columns=["interview_point_id", "interview_geometry_point", "border_crossing_point"]).rename(
            columns={"cross_border_person_id": "source_cross_border_person_id"}),
        on="source_cross_border_person_id", validate="many_to_one")
    changed.drop(columns="interview_geometry_point").to_csv(output / "person_assignments.csv", index=False)
    logger.info("Count-guided crossings: %d foreign and %d Swiss overrides; diagnostics in %s",
                len(foreign_changes), len(swiss), output)

    # Facility aliases retain their original donor ID explicitly for auditing.

    return dict(foreign=foreign_changes, swiss=swiss[["person_id"] + keys], swiss_facilities=aliases, groups=audit)


def validate(context):
    root = reference_directory(context.config("data_path"), context.config("cross_border_reference_path"))
    gpkg = root / "border_interview_places_road_and_rail.gpkg"
    return file_versions([counts_path(context), gpkg, Path(str(gpkg) + "-wal"),
                          Path(__file__).with_name("count_routing.py"), Path(__file__).with_name("count_balancing.py"),
                          Path(__file__).with_name("count_references.py"),
                          Path(__file__).with_name("count_assignment_inputs.py")])


def reference_groups(sites, references):
    """Only complete, geographically checked physical-site sets can define shares."""
    refs = pd.DataFrame(references)
    if refs.empty:
        return {}, []
    required = {"fid", "x", "y", "low"}
    if not required.issubset(refs) or refs.fid.duplicated().any():
        raise ValueError("Count references require unique fid and x/y/low/high")
    if "high" not in refs:
        refs["high"] = refs.low
    else:
        refs["high"] = refs.high.fillna(refs.low)
    refs = refs.set_index("fid")
    groups, audit = {}, []
    used = set()
    for place, choices in sites[sites.kind.eq("entry_candidate")].groupby("interview_place", sort=True):
        choices = choices.sort_values("fid").reset_index(drop=True)
        intersection = set(choices.fid) & set(refs.index)
        if not intersection:
            continue
        used.update(intersection)
        if len(intersection) != len(choices):
            audit.append(dict(interview_place=place, status="unchanged_incomplete_counts"))
            continue
        values = refs.loc[choices.fid]
        xy = values[["x", "y"]].to_numpy(float)
        ranges = values[["low", "high"]].to_numpy(float)
        if (not np.isfinite(xy).all() or not np.isfinite(ranges).all()
                or (ranges <= 0).any() or (ranges[:, 0] > ranges[:, 1]).any()
                or (np.linalg.norm(xy-choices[["x", "y"]].to_numpy(), axis=1) > 50).any()):
            raise ValueError("Invalid/stale count reference for %s; check FIDs, coordinates and ranges" % place)
        if len(choices) == 1:
            audit.append(dict(interview_place=place, status="unchanged_single_crossing"))
            continue
        choices["reference"] = ranges.mean(axis=1)
        groups[place] = choices
    for fid in sorted(set(refs.index)-used):
        audit.append(dict(fid=int(fid), status="unchanged_reference_not_canonical_entry"))
    return groups, audit


def collect_demand(context, groups, canonical, code_points):
    foreign = context.stage(FOREIGN)
    f = foreign[foreign.interview_place.isin(groups) & foreign.trip_mode.isin(["car", "car_passenger"])].copy()
    f["category"] = f.label.map({"From-To": "foreign_visitor", "Through": "transit"})
    f["origin_x"], f["origin_y"] = f.residence_x, f.residence_y
    through = f.category.eq("transit")
    for coordinate in ("x", "y"):
        f.loc[through, "destination_"+coordinate] = f.loc[through, "destination_residence_"+coordinate]
    f["direction"] = np.where(through, "entry", "both")
    f["mass"] = np.where(f.trip_mode.eq("car"), np.where(through, 1., 2.), 0.)
    f["reported_exit_point"] = f.survey_exit_code.map(code_points)
    f["person_id"] = None

    # Keep only crossing residents in memory; do not copy the entire population.
    population = context.stage(SWISS)
    selected = population.border_interview_place.isin(groups) & population.border_crossing_trip_mode.isin(["car", "car_passenger"])
    s = population.loc[selected, ["person_id", "cross_border_person_id", "home_x", "home_y",
                                "interview_point_id", "border_crossing_direction", "border_crossing_trip_mode"]].copy()
    reference = context.stage(REFERENCE)
    s = s.merge(reference[["cross_border_person_id", "interview_id", "interview_place", "foreign_x", "foreign_y"]],
                on="cross_border_person_id", validate="many_to_one")
    # Both endpoint coordinates now belong to the survey-stage contract.
    s["destination_x"], s["destination_y"] = s.foreign_x, s.foreign_y
    s["origin_x"], s["origin_y"] = s.home_x, s.home_y
    s["category"], s["reported_exit_point"] = "swiss", None
    s["direction"], s["trip_mode"] = s.border_crossing_direction, s.border_crossing_trip_mode
    s["mass"] = s.trip_mode.eq("car").astype(float)
    columns = ROW_KEYS + ["person_id", "cross_border_person_id", "interview_place", "interview_point_id", "trip_mode", "mass"]
    combined = pd.concat([f[columns], s[columns]], ignore_index=True)
    combined["old_point"] = combined.interview_point_id.map(lambda p: canonical.get(p, p))
    if combined.category.isna().any() or combined.direction.isna().any():
        raise ValueError("Unrecognised crossing category/direction")
    return combined


def assign_group(persons, sites, fixed, exits, router, rng, maximum=30.):
    """Solve on unique ODs, then draw whole people/tours with a seeded RNG."""
    persons = persons.copy()
    # ngroup avoids joining on all-null exit-code columns (pandas can infer
    # different dtypes on either side), while keeping the original row order.
    persons["od_id"] = persons.groupby(ROW_KEYS, dropna=False, sort=True).ngroup()
    rows = persons.drop_duplicates("od_id").sort_values("od_id")[ROW_KEYS + ["od_id"]].reset_index(drop=True)
    rows["mass"] = persons.groupby("od_id").mass.sum().reindex(rows.od_id).to_numpy()
    site_index = pd.Index(sites.point_id)
    old = site_index.get_indexer(persons.old_point)
    if (old < 0).any():
        raise ValueError("Assigned entry is not an active physical candidate")
    cost, extra_minutes, supported, road, anchors = router.costs(rows, sites, exits)
    # Prohibited choices cannot become eligible by increasing the allowance.
    same_exit = (rows.category.eq("transit").to_numpy()[:, None]
                 & (rows.reported_exit_point.to_numpy()[:, None] == site_index.to_numpy()))
    extra_minutes = np.where(same_exit, np.inf, extra_minutes)
    supported = supported & np.isfinite(extra_minutes).any(axis=1)
    variable = supported & rows.mass.gt(0).to_numpy()
    fixed_exits = fixed.reindex(site_index, fill_value=0).to_numpy(float)
    total = persons.mass.sum() + fixed_exits.sum()
    target = total * sites.reference.to_numpy() / sites.reference.sum()
    frozen = ~variable[persons.od_id.to_numpy()]
    fixed_entries = np.bincount(old[frozen], weights=persons.loc[frozen, "mass"], minlength=len(sites))
    residual = target - fixed_exits - fixed_entries
    if not variable.any():
        raise ValueError("No geographically supported driver demand")
    # Preserve exact expected margins, expanding geographic eligibility only
    # when necessary. The ceiling is an explicit stop, never a soft-count switch.
    _, cap, checks = exact_support(rows.mass.to_numpy()[variable], residual, extra_minutes[variable], maximum)
    allowed = np.isfinite(extra_minutes) & (extra_minutes <= cap)
    flow, fit = allocate(rows.mass.to_numpy()[variable], cost[variable], residual, allowed[variable], penalty=None)
    expected = flow.sum(axis=0) + fixed_exits + fixed_entries
    if not np.allclose(expected, target, atol=.1, rtol=0):
        raise CountAssignmentConflict("Exact-share optimiser did not meet margins within 0.1 crossings")
    probability = np.zeros_like(cost)
    probability[variable] = flow / rows.mass.to_numpy()[variable, None]
    # Passengers do not contribute vehicles to count targets. Where no driver
    # shares their OD, use the fitted crossing attractiveness on their own
    # geographic support. The source does not retain household/vehicle pairs.
    # Passenger-only ODs do not force a larger driver allowance. Keep an
    # unsupported passenger at its existing crossing and expose it in the audit.
    supported = supported & allowed.any(axis=1)
    passengers = supported & ~variable
    if passengers.any():
        probability[passengers] = softmax(np.where(allowed[passengers],
            (fit["bias_minutes"]-cost[passengers])/5., -np.inf), axis=1)
    movable = supported[persons.od_id.to_numpy()]
    chosen = old.copy()
    cdf = probability[persons.loc[movable, "od_id"]].cumsum(axis=1)
    cdf[:, -1] = 1.
    chosen[movable] = (rng.random(movable.sum())[:, None] >= cdf).sum(axis=1)
    if not allowed[persons.loc[movable, "od_id"].to_numpy(), chosen[movable]].all():
        raise ValueError("A chosen crossing violates geographic support")
    realised = np.bincount(chosen, weights=persons.mass, minlength=len(sites)) + fixed_exits
    if not np.isclose(realised.sum(), total):
        raise ValueError("Count assignment changed total crossing demand")
    persons["selected_point"] = site_index.to_numpy()[chosen]
    persons["selected_extra_minutes"] = np.nan
    persons.loc[movable, "selected_extra_minutes"] = extra_minutes[persons.loc[movable, "od_id"].to_numpy(), chosen[movable]]
    # Keep even unchanged draws in the audit; only supported people receive
    # overrides, so unresolved transit remains byte-for-byte at its old entry.
    changed = persons.loc[movable].copy()
    details = sites[["fid", "point_id", "reference"]].copy()
    details["target"], details["expected"], details["assigned"] = target, expected, realised
    details["fixed_exits"], details["fixed_entries"] = fixed_exits, fixed_entries
    before = np.bincount(old, weights=persons.mass, minlength=len(sites)) + fixed_exits
    details["before"] = before
    extended = persons.selected_extra_minutes.gt(20).to_numpy()
    details["expected_crossings_over_20"] = np.where(extra_minutes[variable] > 20, flow, 0.).sum(axis=0)
    details["assigned_crossings_over_20"] = np.bincount(chosen[extended], weights=persons.loc[extended, "mass"], minlength=len(sites))
    summary = dict(status="exact_shares" if cap == 20 else "exact_shares_expanded_geography", total_crossings=float(total),
                   extra_minutes_limit=cap, max_extra_minutes=float(maximum), feasibility_checks=checks,
                   expected_share_over_20_pct=float(100*details.expected_crossings_over_20.sum()/flow.sum()),
                   assigned_crossings_over_20=float(details.assigned_crossings_over_20.sum()),
                   frozen_crossings=float(fixed_entries.sum()), movable_people=int(movable.sum()),
                   max_expected_gap=float(np.max(np.abs(expected-target))),
                   max_assigned_gap=float(np.max(np.abs(realised-target))), solve_seconds=fit["seconds"])
    return changed, summary, details, anchors
