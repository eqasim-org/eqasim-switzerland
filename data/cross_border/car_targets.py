"""Optional daily car controls shared by Swiss and foreign survey demand."""

from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from data.cross_border.reference_files import file_versions


def configure(context):
    context.config("data_path")
    context.config("specific_day_scenario", default="workday")
    # Synpp drops empty dictionaries while flattening stage configuration.
    # A scalar None keeps the disabled option available during execution.
    context.config("cross_border_car_crossing_targets", default=None)


def execute(context):

    # An empty override leaves the existing survey calibration unchanged.

    requested = context.config("cross_border_car_crossing_targets")
    if requested is None:
        return {}
    if not isinstance(requested, dict):
        raise ValueError("cross_border_car_crossing_targets must map interview-place names to daily vehicle counts")
    if not requested:
        return {}
    requested = restore_place_names(requested)

    # Use the complete reference before splitting Swiss and foreign residents.
    # This gives both OD stages the same multiplier instead of applying the
    # entire requested total independently to each residence population.

    old_path, new_path = survey_paths(context.config("data_path"))
    columns = ["BEFRAGUNGSORTID", "BEFRAGUNGSORT", "VERKEHRSTRAEGER", "UEBERGANGSART"]
    reference = pd.read_csv(old_path, encoding="latin1", usecols=columns +
                            ["TAGESTYP", "FAHRZEUGTYP", "GEWICHT"])
    names = pd.read_csv(new_path, sep=";", encoding="latin1", usecols=columns)
    targets = build_targets(reference, names, requested, context.config("specific_day_scenario"))

    # Keep the original benchmark and the requested target reviewable together.

    pd.DataFrame(targets.values()).to_csv(Path(context.path()) / "car_crossing_targets.csv", index=False)
    return targets


def survey_paths(data_path):
    root = Path(data_path) / "crossborder"
    return (root / "AuGQPV_2015" / "Finale_Auswertungsdatenbank_AGQPV2015_V2.csv",
            root / "AuGQPV_2021" / "AGQPV21_finale_Auswertungsdatenbank.csv")


def restore_place_names(requested, prefix=""):
    """Rejoin literal dots in survey names that Synpp treats as nested keys."""
    result = {}
    for name, value in requested.items():
        name = prefix + str(name)
        if isinstance(value, dict):
            result.update(restore_place_names(value, name + "."))
        else:
            result[name] = value
    return result


def build_targets(reference, names, requested, day):
    """Resolve survey names to IDs and calculate full-population multipliers."""
    weekdays = {"workday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    weekends = {"weekend", "Saturday", "Sunday"}
    if day in weekdays:
        day_types, days = ["Werktag"], 260
    elif day in weekends:
        day_types, days = ["Samstag", "Sonntag", "Wochenende"], 104
    else:
        raise ValueError("Unsupported specific_day_scenario: %s" % day)

    # Match the existing car convention (2015 types 1 and 3 include motorcycles).
    # Road survey weights represent inbound vehicles; twice their daily sum is
    # the balanced two-direction benchmark, not a guaranteed routed-link count.
    road = (pd.to_numeric(reference.VERKEHRSTRAEGER).eq(1)
            & pd.to_numeric(reference.UEBERGANGSART).eq(1))
    vehicle = pd.to_numeric(reference.FAHRZEUGTYP, errors="coerce").isin([1, 3])
    selected = reference.loc[road & vehicle & reference.TAGESTYP.isin(day_types)].copy()
    selected["daily_crossings"] = pd.to_numeric(selected.GEWICHT) * 2 / days
    totals = selected.groupby("BEFRAGUNGSORTID").daily_crossings.sum()

    # Only road-border names are eligible: rail and Alpine interview places have
    # different survey coverage even where a place name happens to be identical.
    names = names.loc[pd.to_numeric(names.VERKEHRSTRAEGER).eq(1)
                      & pd.to_numeric(names.UEBERGANGSART).eq(1)].copy()
    names["BEFRAGUNGSORT"] = names.BEFRAGUNGSORT.str.strip()
    result = {}
    for name, target in requested.items():
        if (not isinstance(target, Real) or isinstance(target, bool)
                or not np.isfinite(target) or target <= 0):
            raise ValueError("Car crossing target for %r must be a finite positive daily vehicle count" % name)
        matches = names.loc[names.BEFRAGUNGSORT.eq(name), "BEFRAGUNGSORTID"].unique()
        if len(matches) != 1:
            raise ValueError("Unknown or ambiguous road interview place %r in cross_border_car_crossing_targets; "
                             "use its exact 2021 BEFRAGUNGSORT name" % name)
        place_id = int(matches[0])
        baseline = float(totals.get(place_id, 0))
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("No positive 2015 car reference for %r on %s" % (name, day))
        result[place_id] = dict(interview_place_id=place_id, interview_place=name,
                                reference_both_directions=baseline,
                                target_both_directions=float(target),
                                override_factor=float(target) / baseline)
    return result


def validate(context):
    # ValidateContext exposes flattened keys, unlike ExecuteContext: reading
    # the target dictionary here fails even though configure requested it.
    # Its values already enter Synpp's cache key. Only fingerprint input files;
    # file_versions also accepts absent files when this optional stage is off.
    return file_versions(survey_paths(context.config("data_path")))
