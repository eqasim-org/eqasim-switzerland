"""Joint vehicle controls accounting for entry and reported transit-exit events."""

from pathlib import Path

import numpy as np
import pandas as pd

from data.cross_border.car_targets import survey_paths
from data.cross_border.journey_weights import road_journey_class
from data.cross_border.reference_files import file_versions
from data.cross_border.survey_calibration import calibrate_demand


def configure(context):
    context.config("data_path")
    context.config("specific_day_scenario", "workday")
    context.config("cross_border_car_2015_mix_groups", [])
    requested = context.config("cross_border_car_crossing_targets", None)
    context.config("cross_border_car_target_method", "entry_group")
    context.stage("data.cross_border.car_targets")
    if requested:
        context.stage("data.cross_border.road_exit_points")
        context.stage("data.cross_border.road_candidates")


def execute(context):

    # Retain the existing name/day validation and original survey benchmark.

    targets = context.stage("data.cross_border.car_targets")
    groups = context.config("cross_border_car_2015_mix_groups")
    if not isinstance(groups, list) or any(type(group) is not int for group in groups):
        raise ValueError("cross_border_car_2015_mix_groups must be a list of integer interview-place IDs")
    if not targets and not groups:
        return {}
    old_path, new_path = survey_paths(context.config("data_path"))
    old = pd.read_csv(old_path, encoding="latin1", low_memory=False)
    new = pd.read_csv(new_path, sep=";", encoding="latin1", low_memory=False)
    reference = road_observations(old, 2015, context.config("specific_day_scenario"))
    observations = road_observations(new, 2021, context.config("specific_day_scenario"))

    # Fit the optional mix using both residence populations before coupling crossing targets.

    weighted = residence_calibrated_weights(reference, observations, targets)
    mix, mix_audit = fit_road_mix(reference, weighted, groups)
    output = Path(context.path())
    controls = {}
    if targets:
        exits = context.stage("data.cross_border.road_exit_points")
        roads = context.stage("data.cross_border.road_candidates")
        exit_places = unique_exit_places(exits, roads)
        method = context.config("cross_border_car_target_method")
        if method == "entry_group":
            controls, audit, contributions = fit_entry_targets(weighted, targets, exit_places)
        elif method == "journey":
            controls, audit, contributions = fit_journey_targets(weighted, targets, exit_places, mix)
        else:
            raise ValueError("cross_border_car_target_method must be 'entry_group' or 'journey'")
        pd.DataFrame([controls[k] for k in targets]).to_csv(output / "car_crossing_targets.csv", index=False)
        audit.to_csv(output / "target_event_balance.csv", index=False)
        contributions.to_csv(output / "target_contributions.csv", index=False)

    # Publish factors to both OD stages and the full-population class totals for review.

    for key, factors in mix.items():
        controls.setdefault(key, {})["mix_factors"] = factors
    if mix:
        group_scale = {k: v.get("mix_journey_factor", v.get("override_factor", 1.))
                       for k, v in controls.items()}
        mix_audit["fitted_vehicles"] = mix_audit.mix_vehicles * mix_audit.interview_place_id.map(group_scale)
        mix_audit.to_csv(output / "car_2015_mix.csv", index=False)
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


def residence_calibrated_weights(reference, observations, targets):
    """Compute the two OD stages' baseline, before any manual total multiplier."""
    unit = {key: dict(value, override_factor=1.) for key, value in targets.items()}
    frames = []
    for swiss in (False, True):
        old = reference.loc[reference.residence_country.eq("CH").eq(swiss)]
        new = observations.loc[observations.residence_country.eq("CH").eq(swiss)]
        calibrated, _ = calibrate_demand(old, new, unit)
        frames.append(calibrated)
    return pd.concat(frames, ignore_index=True)


def fit_road_mix(reference, weighted, groups):
    """Reweight eligible entries in place and return factors for the two residence stages."""
    mix, audit = {}, []
    if not groups:
        return mix, pd.DataFrame()
    classes = ["swiss", "foreign_visit", "foreign_transit"]
    reference_classes = road_journey_class(reference)
    observed_classes = road_journey_class(weighted)
    for group in sorted(set(groups)):
        old = reference.loc[reference.interview_place_id.eq(group)]
        new = weighted.loc[weighted.interview_place_id.eq(group)]
        old_mass = old.weight.groupby(reference_classes.loc[old.index]).sum().reindex(classes, fill_value=0.)
        new_mass = new.weight.groupby(observed_classes.loc[new.index]).sum().reindex(classes, fill_value=0.)
        if old_mass.sum() <= 0 or new_mass.sum() <= 0:
            raise ValueError(f"Mix group {group} needs eligible road vehicles in both survey years")
        shares = old_mass / old_mass.sum()
        mix[group] = {}
        for name in classes:
            desired = shares[name] * new_mass.sum()
            if desired > 0 and new_mass[name] <= 0:
                raise ValueError(f"Mix group {group} has 2015 {name} demand without calibrated 2021 support")
            factor = float(desired / new_mass[name]) if new_mass[name] > 0 else 1.
            mix[group][name] = factor
            mask = weighted.interview_place_id.eq(group) & observed_classes.eq(name)
            weighted.loc[mask, ["weight", "group_weight"]] *= factor
            audit.append(dict(interview_place_id=group, journey_class=name, vehicles_2015=old_mass[name],
                              share_2015=shares[name], baseline_vehicles=new_mass[name], mix_factor=factor,
                              mix_vehicles=desired))
    return mix, pd.DataFrame(audit)


def joint_controls(reference, observations, targets, exit_places):
    return fit_entry_targets(residence_calibrated_weights(reference, observations, targets), targets, exit_places)


def fit_entry_targets(weighted, targets, exit_places):
    """Solve all manual controls together; leave other entry groups unchanged.

    Columns are entry-group multipliers. A transit contributes to its entry
    row and its uniquely reported exit row, which couples manual controls.
    Unknown exits are never attributed to the entry crossing as a substitute.
    """
    keys = sorted(targets)
    names = {value["interview_place"]: key for key, value in targets.items()}
    classes = road_journey_class(weighted)
    transit = classes.eq("foreign_transit")
    records = []
    local = weighted.loc[classes.ne("")].copy()
    local["target_id"] = local.interview_place_id
    local["event"] = np.where(transit.loc[local.index], "transit_entry", "visit_entry_and_return")
    local["crossings"] = local.weight * np.where(transit.loc[local.index], 1., 2.)
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
                          Path(__file__).with_name("survey_calibration.py"),
                          Path(__file__).with_name("journey_weights.py")])


def journey_controls(reference, observations, targets, exit_places):
    return fit_journey_targets(residence_calibrated_weights(reference, observations, targets), targets, exit_places, {})


def fit_journey_targets(weighted, targets, exit_places, mix):
    """Fit positive journey weights to all controlled entry-plus-exit totals.

    Minimize relative entropy from the usual 2015-calibrated weights. One
    journey gets one factor; both of its crossing contributions use that factor.
    Uncontrolled crossings may change when they share a controlled journey.
    """
    from scipy.optimize import least_squares
    from data.cross_border.journey_weights import crossing_events

    keys = sorted(targets)
    entry, exits, transit, exit_place = crossing_events(weighted, targets, exit_places)
    events = entry + exits
    wanted = np.array([targets[key]["target_both_directions"] for key in keys])
    weights = weighted.weight.to_numpy()

    # Aggregate identical event patterns before solving the small smooth system.

    # Fit one multiplier per selected group so its class shares and conditional ODs stay fixed.
    design = events.copy()
    group_patterns = {}
    if mix:
        eligible = road_journey_class(weighted).ne("")
        for group in mix:
            mask = (eligible & weighted.interview_place_id.eq(group)).to_numpy()
            group_patterns[group] = np.average(events[mask], axis=0, weights=weights[mask])
            design[mask] = group_patterns[group]
    patterns, inverse = np.unique(design, axis=0, return_inverse=True)
    masses = np.bincount(inverse, weights=weights, minlength=len(patterns))
    supported = (masses > 0) & patterns.any(axis=1)
    patterns, masses = patterns[supported], masses[supported]
    if not len(patterns) or ((patterns.T @ masses) == 0).any():
        raise ValueError("Journey crossing targets have no supported survey events")

    def residual(log_factors):
        adjusted = masses * np.exp(patterns @ log_factors)
        return (patterns.T @ adjusted - wanted) / wanted

    def jacobian(log_factors):
        adjusted = masses * np.exp(patterns @ log_factors)
        return (patterns.T @ (adjusted[:, None] * patterns)) / wanted[:, None]

    fit = least_squares(residual, np.zeros(len(keys)), jac=jacobian, bounds=(-30., 30.),
                        ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=1000)
    multipliers = np.exp(design @ fit.x)
    adjusted = weights * multipliers
    expected = events.T @ adjusted
    if not fit.success or not np.allclose(expected, wanted, rtol=0, atol=1e-5):
        raise ValueError("Journey crossing targets are infeasible or did not converge: "
                         + str(dict(zip(keys, expected - wanted))))

    # Export event factors and complete contributions, including other entries.

    controls, audit, records = {}, [], []
    for column, key in enumerate(keys):
        controls[key] = dict(targets[key], override_factor=1.,
                             balanced_override_factor=targets[key]["override_factor"],
                             calibration_basis="journey_entry_and_exit",
                             log_event_factor=float(fit.x[column]),
                             event_factor=float(np.exp(fit.x[column])),
                             reported_exit_codes=sorted(int(code) for code, name in exit_places.items()
                                                        if name == targets[key]["interview_place"]),
                             expected_both_directions=float(expected[column]))
        unresolved = transit & weighted.interview_place_id.eq(key).to_numpy() & exit_place.isna().to_numpy()
        audit.append(dict(interview_place=targets[key]["interview_place"], target=wanted[column],
                          expected=expected[column], event_factor=float(np.exp(fit.x[column])),
                          unresolved_transit_exit_weight=float(adjusted[unresolved].sum())))
        for name, coefficients in [("transit_entry", entry[:, column] * transit),
                                   ("visit_entry_and_return", entry[:, column] * ~transit),
                                   ("transit_exit", exits[:, column])]:
            mask = coefficients > 0
            part = pd.DataFrame(dict(interview_place_id=weighted.loc[mask, "interview_place_id"],
                                     crossings=(weights * coefficients)[mask],
                                     expected_crossings=(adjusted * coefficients)[mask]))
            part = part.groupby("interview_place_id", as_index=False).sum()
            part["target_id"], part["event"] = key, name
            part["multiplier"] = part.expected_crossings / part.crossings
            records.append(part)
    for group, pattern in group_patterns.items():
        controls.setdefault(group, {})["mix_journey_factor"] = float(np.exp(pattern @ fit.x))
    return controls, pd.DataFrame(audit), pd.concat(records, ignore_index=True)
