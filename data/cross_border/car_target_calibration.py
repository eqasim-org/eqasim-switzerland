"""Joint vehicle controls accounting for entry and reported transit-exit events."""

from pathlib import Path

import numpy as np
import pandas as pd

from data.cross_border.car_targets import survey_paths
from data.cross_border.reference_files import file_versions
from data.cross_border.survey_calibration import calibrate_demand


def configure(context):
    context.config("data_path")
    context.config("specific_day_scenario", "workday")
    requested = context.config("cross_border_car_crossing_targets", None)
    context.stage("data.cross_border.car_targets")
    if requested:
        context.stage("data.cross_border.road_exit_points")
        context.stage("data.cross_border.road_candidates")


def execute(context):

    # Retain the existing name/day validation and original survey benchmark.

    targets = context.stage("data.cross_border.car_targets")
    if not targets:
        return {}
    old_path, new_path = survey_paths(context.config("data_path"))
    old = pd.read_csv(old_path, encoding="latin1", low_memory=False)
    new = pd.read_csv(new_path, sep=";", encoding="latin1", low_memory=False)
    reference = road_observations(old, 2015, context.config("specific_day_scenario"))
    observations = road_observations(new, 2021, context.config("specific_day_scenario"))

    # Use exactly the unique physical exit-code matches used by early assignment.

    exits = context.stage("data.cross_border.road_exit_points")
    roads = context.stage("data.cross_border.road_candidates")
    exit_places = unique_exit_places(exits, roads)
    controls, audit, contributions = joint_controls(reference, observations, targets, exit_places)

    # Record the coupled solution, including contributions from other entry groups.

    output = Path(context.path())
    pd.DataFrame(controls.values()).to_csv(output / "car_crossing_targets.csv", index=False)
    audit.to_csv(output / "target_event_balance.csv", index=False)
    contributions.to_csv(output / "target_contributions.csv", index=False)
    return controls


def road_observations(raw, year, day):
    """Mirror the OD readers' inbound driver weights, without occupancy expansion."""
    weekdays = {"workday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    weekends = {"weekend", "Saturday", "Sunday"}
    if day not in weekdays | weekends:
        raise ValueError("Unsupported specific_day_scenario: %s" % day)
    days = 260 if day in weekdays else 104
    types = ["Werktag"] if day in weekdays else ["Samstag", "Sonntag", "Wochenende"]
    selected = (pd.to_numeric(raw.VERKEHRSTRAEGER).eq(1)
                & pd.to_numeric(raw.UEBERGANGSART).eq(1)
                & pd.to_numeric(raw.FAHRZEUGTYP, errors="coerce").isin([1, 3] if year == 2015 else [1, 2])
                & raw.TAGESTYP.isin(types))
    rows = raw.loc[selected].copy()
    result = pd.DataFrame(dict(
        interview_place_id=pd.to_numeric(rows.BEFRAGUNGSORTID),
        interview_place=rows.BEFRAGUNGSORT.str.strip(),
        residence_country=rows.WOHNORTLANDISO.str.strip(),
        travel_cat=pd.to_numeric(rows.VERKEHRSART, errors="coerce"),
        origin_country=rows.STARTORTLANDISO.str.strip(),
        destination_country=rows.ZIELORTLANDISO.str.strip(),
        weight=pd.to_numeric(rows["GEWICHT" if year == 2015 else "GEWICHT_Fahrzeuge"]) / days,
        survey_exit_code=pd.to_numeric(rows.get("GUAUSREISESTRASSE", pd.Series(index=rows.index, dtype=float)), errors="coerce"),
    ))
    result["trip_mode"], result["trip_purpose"] = "car", "unused"
    result["group_weight"] = result.weight
    return result.reset_index(drop=True)


def unique_exit_places(exits, roads):
    canonical = dict(zip(roads.border_crossing_point_id, roads.canonical_point_id))
    mapped = exits.assign(point_id=exits.border_crossing_point_id.map(lambda p: canonical.get(p, p)))
    result = {}
    for code, group in mapped.groupby("survey_exit_code"):
        if group.point_id.nunique() == 1 and group.interview_place.nunique() == 1:
            result[int(code)] = group.interview_place.iloc[0]
    return result


def joint_controls(reference, observations, targets, exit_places):
    """Solve all manual controls together; leave other entry groups unchanged.

    Columns are entry-group multipliers. A transit contributes to its entry
    row and its uniquely reported exit row, which couples manual controls.
    Unknown exits are never attributed to the entry crossing as a substitute.
    """
    keys = sorted(targets)
    names = {value["interview_place"]: key for key, value in targets.items()}
    unit = {key: dict(value, override_factor=1.) for key, value in targets.items()}
    frames = []
    for swiss in (False, True):
        old = reference.loc[reference.residence_country.eq("CH").eq(swiss)]
        new = observations.loc[observations.residence_country.eq("CH").eq(swiss)]
        calibrated, _ = calibrate_demand(old, new, unit)
        frames.append(calibrated)
    weighted = pd.concat(frames, ignore_index=True)
    swiss = weighted.residence_country.eq("CH")
    visit = weighted.travel_cat.isin([3, 4])
    # The Swiss generator supports only journeys with a Swiss endpoint.
    visit &= ~swiss | weighted.origin_country.eq("CH") | weighted.destination_country.eq("CH")
    transit = ~swiss & weighted.travel_cat.eq(2)
    records = []
    local = weighted.loc[visit | transit].copy()
    local["target_id"] = local.interview_place_id
    local["event"] = np.where(local.travel_cat.eq(2), "transit_entry", "visit_entry_and_return")
    local["crossings"] = local.weight * np.where(local.travel_cat.eq(2), 1., 2.)
    records.append(local)
    through = weighted.loc[transit].copy()
    through["exit_place"] = through.survey_exit_code.map(exit_places)
    through["target_id"] = through.exit_place.map(names)
    through["event"], through["crossings"] = "transit_exit", through.weight
    records.append(through)
    events = pd.concat(records, ignore_index=True)
    events = events.loc[events.target_id.isin(keys)]
    contributions = events.groupby(["target_id", "interview_place_id", "event"], as_index=False).crossings.sum()
    matrix = contributions.pivot_table(index="target_id", columns="interview_place_id", values="crossings", aggfunc="sum", fill_value=0.)
    matrix = matrix.reindex(index=keys, fill_value=0.)
    fixed = matrix.loc[:, ~matrix.columns.isin(keys)].sum(axis=1).to_numpy()
    coupled = matrix.reindex(columns=keys, fill_value=0.).to_numpy()
    wanted = np.array([targets[key]["target_both_directions"] for key in keys])
    try:
        factors = np.linalg.solve(coupled, wanted - fixed)
    except np.linalg.LinAlgError as error:
        raise ValueError("Manual crossing targets have no unique supported joint solution") from error
    if not np.isfinite(factors).all() or (factors < 0).any():
        raise ValueError("Manual crossing targets are infeasible: fixed transit exits or coupled demand exceed a target")
    expected = coupled @ factors + fixed
    if not np.allclose(expected, wanted, rtol=0, atol=1e-6):
        raise ValueError("Joint crossing calibration did not meet its expected event targets")
    controls, audit = {}, []
    for i, key in enumerate(keys):
        controls[key] = dict(targets[key], override_factor=float(factors[i]),
                             balanced_override_factor=targets[key]["override_factor"],
                             calibration_basis="entry_and_reported_exit_events",
                             expected_both_directions=float(expected[i]),
                             fixed_other_group_crossings=float(fixed[i]))
        unresolved = through.loc[through.interview_place_id.eq(key) & through.exit_place.isna(), "weight"].sum()
        audit.append(dict(interview_place=targets[key]["interview_place"], target=wanted[i],
                          expected=expected[i], fixed_other_group_crossings=fixed[i],
                          multiplier=factors[i], unresolved_transit_exit_weight=unresolved * factors[i]))
    contributions["multiplier"] = contributions.interview_place_id.map(dict(zip(keys, factors))).fillna(1.)
    contributions["expected_crossings"] = contributions.crossings * contributions.multiplier
    return controls, pd.DataFrame(audit), contributions


def validate(context):
    return file_versions([*survey_paths(context.config("data_path")),
                          Path(__file__).with_name("survey_calibration.py")])
