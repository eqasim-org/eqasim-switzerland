"""Select a force-model assignment against weighted weekday Microcensus destinations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import yaml
from scipy.spatial import cKDTree

from . import errand_centres, facility_preferences, school_proximity, subtypes

PREFIX = errand_centres.PREFIX
MODE = PREFIX + "calibration_mode"
STAGE = "synthesis.population.spatial.secondary.force_model.calibration"
ASSIGN = "synthesis.population.spatial.secondary.locations_rda"
SCALES = PREFIX + "calibration_scales"
MIN_PEOPLE = PREFIX + "calibration_min_respondents"
AXES = ("shop", "leisure", "other_services", "school", "density")
BANDS = np.array([1000., 2000., 4000., 8000.])


def configure(context):
    context.config("data_path")
    context.config(MIN_PEOPLE, default=20)
    if context.config("secondary_activities.chain_solver", default="default") != "force_model":
        raise ValueError("Automatic secondary calibration requires the force_model chain solver")
    for name in ("data.microcensus.trips", "data.spatial.swiss_border", "synthesis.population.trips",
                 "synthesis.population.sampled"):
        context.stage(name)
    if "other_escort_child" in subtypes.selected_purposes(context.config(subtypes.OPTION, default=False)):
        context.stage("data.statent.statent")
    for index, profile in enumerate(profiles(context, configure=True)):
        context.stage(ASSIGN, config=profile, alias=f"candidate_{index}")


def execute(context):

    # Build survey targets and identify the same arriving activities in each candidate.

    survey_trips = context.stage("data.microcensus.trips")[0]
    folder = Path(context.config("data_path")) / "microcensus"
    people = pd.read_csv(folder / "zielpersonen.csv", encoding="latin1", usecols=["HHNR", "tag", "alter", "WP"])
    people = people.loc[people.tag.between(1, 5) & people.alter.ge(6), ["HHNR", "WP"]]
    survey = survey_trips.merge(people, left_on="person_id", right_on="HHNR", validate="many_to_one")
    survey = survey.sort_values(["person_id", "trip_id"]).rename(columns={
        "destination_x": "x", "destination_y": "y", "WP": "weight"})
    if not np.isfinite(survey.weight).all() or not survey.weight.gt(0).all():
        raise ValueError("Invalid Microcensus respondent weights")
    selected = subtypes.selected_purposes(context.config(subtypes.OPTION))
    survey["size"] = chain_sizes(survey, "origin_purpose", "purpose")
    subtype = survey.other_subtype_code.map(subtypes.PURPOSES) if selected else survey.purpose
    use = survey.purpose.eq("other") & subtype.isin(selected)
    survey.loc[use, "purpose"] = subtype[use]
    border = context.stage("data.spatial.swiss_border").to_crs(2056).geometry.union_all()
    shapely.prepare(border)
    survey = survey.loc[shapely.intersects_xy(border, survey.x, survey.y)]
    trips = context.stage("synthesis.population.trips")
    persons = context.stage("synthesis.population.sampled")
    eligible = persons.age.ge(6) if "age" in persons else persons.age_class.ge(1)
    trips = trips[trips.person_id.isin(persons.loc[eligible, "person_id"])]
    border_trip = trips.preceding_purpose.eq("border") | trips.following_purpose.eq("border")
    trips = trips[~trips.person_id.isin(trips.loc[border_trip, "person_id"])].sort_values(["person_id", "trip_index"])
    if selected:
        trips = subtypes.assignment_purposes(trips, survey_trips, selected)
    trips = trips.assign(size=chain_sizes(trips, "preceding_purpose", "following_purpose"),
                         trip_index=trips.trip_index + 1).rename(columns={"following_purpose": "purpose"})
    keys = ["person_id", "trip_index"]
    arrivals = trips[keys + ["size", "purpose"]]

    # Measure each calibration area and child-escort target on separate respondent partitions.

    centres = get_centres(context, selected)
    if "other_services" not in selected and context.config(PREFIX + "density_beta.other"):
        centres["other"] = context.config(errand_centres.OPTION)
    schools = None
    if "other_escort_child" in selected:
        jobs = context.stage("data.statent.statent").dropna(subset=["x", "y"])
        sites = jobs.loc[jobs.noga.astype(str).str.startswith(subtypes.SECTORS["school_daycare"]), ["x", "y"]]
        if sites.empty:
            raise ValueError("No school or daycare establishments in STATENT")
        schools = cKDTree(sites)
    reserved = facility_preferences.holdout(survey.person_id)
    targets = {name: metrics(part, centres, schools) for name, part in
               (("fit", survey[~reserved]), ("holdout", survey[reserved]))}
    minimum = context.config(MIN_PEOPLE)
    if type(minimum) is not int or minimum < 1:
        raise ValueError(f"{MIN_PEOPLE} must be a positive integer")
    if not any(n >= minimum for _, n in targets["fit"].values()):
        raise ValueError("No calibration target has enough MZ respondents; check areas, subtypes and survey support")

    # Score complete assignments on the fitting partition; reserve holdout results for reporting.

    candidates = profiles(context)
    rows, scores = [], []
    reference_keys = None
    for index, profile in enumerate(candidates):
        locations, checks = context.stage(f"candidate_{index}")
        loc_index = pd.MultiIndex.from_frame(locations[keys]).sort_values()
        if not loc_index.is_unique or (reference_keys is not None and not reference_keys.equals(loc_index)):
            raise ValueError("Calibration candidates must assign the same unique person/activity keys")
        reference_keys = loc_index
        points = locations[keys].assign(x=locations.geometry.x, y=locations.geometry.y, weight=1.)
        points = points.merge(arrivals, on=keys, validate="one_to_one")
        points = points.loc[shapely.intersects_xy(border, points.x, points.y)]
        model = metrics(points, centres, schools)
        summary = {"candidate": index, "valid_chain_share": checks.valid.mean()}
        for partition, target in targets.items():
            errors = []
            for key, (observed, n) in target.items():
                predicted, model_n = model[key]
                # An empty model catchment gets the maximum distribution error.
                error = .5 * abs(predicted - observed).sum() if model_n and n else 1.
                included = n >= minimum
                if included:
                    errors.append(error)
                for band, (obs, pred) in enumerate(zip(observed, predicted, strict=True)):
                    rows.append(dict(candidate=index, partition=partition, metric=key, band=band,
                                     survey_respondents=n, model_people=model_n, included=included,
                                     survey_share=obs, model_share=pred))
            summary[partition + "_error_pp"] = 100 * np.mean(errors) if errors else np.nan
        scores.append(summary)
    winner = min(range(len(scores)), key=lambda i: scores[i]["fit_error_pp"])

    # Publish the fitting evidence and reuse the exact selected plans with their convergence table.

    output = Path(context.path())
    pd.DataFrame(rows).to_csv(output / "target_shares.csv", index=False)
    pd.DataFrame(scores).assign(selected=lambda d: d.candidate.eq(winner)).to_csv(output / "scores.csv", index=False)
    (output / "candidates.json").write_text(json.dumps(candidates, indent=2))
    settings = dict(candidates[winner])
    settings.update({"locations_algorithm": "rda", "secondary_activities.chain_solver": "force_model",
                     subtypes.OPTION: context.config(subtypes.OPTION),
                     PREFIX + "canton_ids": context.config(PREFIX + "canton_ids")})
    (output / "selected_settings.yml").write_text(yaml.safe_dump({"config": settings}, sort_keys=True))
    return context.stage(f"candidate_{winner}")


def mode_name(value):
    # YAML 1.1 also parses an unquoted off as the boolean False.
    if value is False:
        return "off"
    if value not in ("off", "manual", "auto"):
        raise ValueError(f"{MODE} expects off, manual or auto")
    return value


def get_centres(context, selected):
    return {purpose: context.config(PREFIX + key) for purpose, key in errand_centres.CENTRE_KEYS.items()
            if purpose != "other_services" or purpose in selected}


def profiles(context, configure=False):
    def setting(key, default):
        return context.config(key, default=default) if configure else context.config(key)

    selected = subtypes.selected_purposes(setting(subtypes.OPTION, False))
    scales = setting(SCALES, [0., .5, 1., 1.5])
    attraction = setting(PREFIX + "attractors", "uniform")
    setting(PREFIX + "canton_ids", list(range(1, 27)))
    betas = {purpose: setting(PREFIX + "density_beta." + purpose, 0.) for purpose in ("shop", "leisure", "other")}
    centres = {purpose: errand_centres.validate_centres(setting(PREFIX + key, []), PREFIX + key)
               for purpose, key in errand_centres.CENTRE_KEYS.items()}
    school = school_proximity.multipliers(setting(school_proximity.OPTION, 1.))
    if not isinstance(scales, list) or not scales:
        raise ValueError(f"{SCALES} needs a non-empty list of candidate scales")
    results = []
    for scale in scales:
        factors = dict.fromkeys(AXES, 1.)
        if isinstance(scale, dict):
            if set(scale) - set(AXES):
                raise ValueError(f"Unknown calibration axes: {set(scale) - set(AXES)}")
            factors.update(scale)
        else:
            factors = dict.fromkeys(AXES, scale)
        profile = {MODE: "manual", subtypes.REFIT: bool(selected), PREFIX + "attractors": attraction}
        for purpose, key in errand_centres.CENTRE_KEYS.items():
            active = purpose in selected or purpose != "other_services"
            values = axis_scales(factors[purpose], len(centres[purpose]), purpose)
            profile[PREFIX + key] = [dict(c, strength_m=c["strength_m"] * f)
                                     for c, f in zip(centres[purpose], values, strict=True)
                                     if active and c["strength_m"] * f > 0]
        powers = axis_scales(factors["school"], 2, "school")
        profile[school_proximity.OPTION] = ([v ** f for v, f in zip(school, powers, strict=True)]
                                            if "other_escort_child" in selected else [1., 1.])
        for (purpose, beta), f in zip(betas.items(), axis_scales(factors["density"], 3, "density"), strict=True):
            profile[PREFIX + "density_beta." + purpose] = beta * f
        if not any(profile[PREFIX + "density_beta." + p] for p in betas):
            profile[PREFIX + "attractors"] = "uniform"
        if profile not in results:
            results.append(profile)
    return results


def axis_scales(value, count, axis):
    values = value if isinstance(value, list) else [value]
    if any(type(v) not in (float, int) or not np.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"Invalid calibration scale for {axis}; expected finite nonnegative numbers")
    if isinstance(value, list) and len(value) != count:
        raise ValueError(f"Expected {count} calibration scales for {axis}")
    return values if isinstance(value, list) else values * count


def chain_sizes(trips, origin, destination):
    fixed = ("home", "work", "education")
    starts = trips[origin].isin(fixed).groupby(trips.person_id).cumsum()
    groups = [trips.person_id, starts]
    return ((~trips[destination].isin(fixed)).groupby(groups).transform("sum")
            + (~trips[origin].groupby(groups).transform("first").isin(fixed)).astype(int))


def metrics(visits, centres, schools):
    result = {}

    def record(key, frame, bins, count):
        weights = np.bincount(bins, weights=frame.weight, minlength=count)
        result[key] = (weights / weights.sum() if len(frame) else np.full(count, np.nan),
                       frame.person_id.nunique())

    for purpose, areas in centres.items():
        group = visits[visits.purpose.eq(purpose)]
        for area in areas:
            dx = (group.x - area["x"]).abs()
            dy = (group.y - area["y"]).abs()
            radius = np.maximum(dx, (dx + np.sqrt(3) * dy) / 2)
            inside = radius <= BANDS[-1]
            record(f'{purpose}@{area["x"]},{area["y"]}', group[inside], np.searchsorted(BANDS, radius[inside]), 4)
    if schools is not None:
        children = visits[visits.purpose.eq("other_escort_child")]
        for name, mask in (("single", children["size"].eq(1)), ("multiple", children["size"].gt(1))):
            group = children[mask]
            near = schools.query(group[["x", "y"]])[0] <= 25.
            record("child_escort_school_" + name, group, near.astype(int), 2)
    return result
