"""Select compatible secondary schools around the education commute target."""

import numpy as np


# BFS structural-survey 2023 codebook, sheet 25, and NOGA 2008 titles:
# https://www.bfs.admin.ch/bfsstatic/dam/assets/344611/master
# These are programme/site compatibility assumptions, not measured capacities.
# Mixed compulsory and special schools can accommodate older compulsory pupils.
# Code 8 combines Berufs- AND Fachmaturität; it must retain both site types.
# For code 5, the 2021–2023 survey reports 853102 more often than 853103.
# Retain both academic site codes: establishment NOGA is not a programme list.
# The same combined academic pool remains available for Fachmaturität (code 8).
# Bridging (code 4) has no dedicated NOGA code and retains a broad secondary pool.
SCHOOL_CODES = {
    2: ("853101", "852002", "852003"),
    3: ("853101", "852002", "852003"),
    4: ("853101", "853102", "853103", "853200"),
    5: ("853102", "853103"),
    6: ("853200",),
    7: ("853102",),
    8: ("853200", "853102", "853103"),
}


def choose_schools(persons, schools, rng, candidate_count=20):
    """Search the full compatible pool by distance mismatch, in bounded batches.

    Searching by radial distance error keeps schools near a long target commute
    accessible even in dense areas. It does not prescribe travel direction or
    impose an OD pattern. Employee counts provide only a clipped preference.
    """
    if schools.empty:
        raise ValueError("No compatible secondary schools")
    coordinates = schools[["x", "y"]].to_numpy(dtype=float)
    homes = persons[["home_x", "home_y"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or not np.isfinite(homes).all():
        raise ValueError("Secondary assignment requires finite LV95 coordinates")
    targets = persons["commute_home_distance"].to_numpy(dtype=float)
    valid = np.isfinite(targets) & (targets >= 0)
    # Missing/negative survey distances use the nearest compatible school; they
    # are not interpreted as zero-distance observations in validation statistics.
    targets = np.where(valid, targets, 0.)
    headcounts = schools["number_employees"].to_numpy(dtype=float)
    headcounts = np.where(np.isfinite(headcounts) & (headcounts > 0), headcounts, 0.)
    lower, upper = np.quantile(headcounts, [.25, .75])
    headcounts = np.maximum(np.clip(headcounts, lower, upper), 1.)
    selected = np.empty(len(persons), dtype=int)
    count = min(candidate_count, len(schools))
    if count < 1:
        raise ValueError("Secondary candidate count must be positive")
    for start in range(0, len(persons), 256):
        stop = min(start + 256, len(persons))
        distance = np.hypot(homes[start:stop, 0, None] - coordinates[None, :, 0],
                            homes[start:stop, 1, None] - coordinates[None, :, 1])
        errors = np.abs(distance - targets[start:stop, None])
        # Choose the 20 closest matches to the target distance, not the 20 closest
        # buildings to home. Full-pool search also handles scarce rural schools.
        indices = np.argpartition(errors, count - 1, axis=1)[:, :count]
        candidate_errors = np.take_along_axis(errors, indices, axis=1)
        # A 200 m minimum tolerance grows to 10% for longer target journeys. This is an
        # explicit modelling assumption, shared across programmes and origins.
        sigma = np.maximum(200., .1 * targets[start:stop])[:, None]
        log_weight = -.5 * (candidate_errors / sigma) ** 2 + np.log(headcounts[indices])
        # Subtract each person's maximum before exponentiating. No probability
        # floor flattens the distance preference when every school is far away.
        weights = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
        probability = weights / weights.sum(axis=1, keepdims=True)
        draws = rng.random_sample(stop - start)
        picks = (probability.cumsum(axis=1) < draws[:, None]).sum(axis=1)
        picks = np.minimum(picks, count - 1)
        selected[start:stop] = indices[np.arange(stop - start), picks]
        missing = ~valid[start:stop]
        selected[start:stop][missing] = distance[missing].argmin(axis=1)
    return schools.iloc[selected]["enterprise_id"].to_numpy(), valid


def assign_secondary_locations(persons, locations, statent, seed, distance_model=None):
    """Replace classified domestic secondary assignments; preserve proxy handling.

    An independent RNG avoids changing the random stream for children, tertiary
    students and unresolved or foreign classifications in the existing stage.
    """
    required = {"current_education", "is_student", "education_destination_status", "age"}
    if not required.issubset(persons.columns):
        return locations.copy()
    selected = (persons["age"].ge(15) & persons["is_student"].eq(1)
                & persons["education_destination_status"].eq("domestic")
                & persons["current_education"].isin(SCHOOL_CODES)).fillna(False)
    cohort = persons.loc[selected & persons["person_id"].isin(locations["person_id"])]
    if distance_model is not None:
        from synthesis.population.spatial.primary.education.secondary_distances import assign_distance_targets
        # Replace only the school-selection target; donor diaries and original
        # commute distances remain intact for timing, matching and diagnostics.
        targets = assign_distance_targets(cohort, distance_model, seed + 1)
        cohort = cohort.merge(targets, on="person_id", validate="one_to_one")
        cohort["commute_home_distance"] = cohort["education_secondary_target_distance_m"]
    result = locations.copy().set_index("person_id", drop=False)
    result["education_secondary_method"] = "not_applicable"
    if distance_model is not None:
        for column in ["education_secondary_target_distance_m", "education_secondary_distance_source",
                       "education_secondary_distance_effective_n"]:
            result[column] = "not_applicable" if column.endswith("source") else np.nan
            result.loc[cohort.person_id, column] = cohort[column].to_numpy()
    rng = np.random.RandomState(seed)
    sites = statent.set_index("enterprise_id", drop=False)
    for code, codes in SCHOOL_CODES.items():
        people = cohort.loc[cohort["current_education"].eq(code)]
        if people.empty:
            continue
        schools = statent.loc[statent["noga"].astype(str).isin(codes)]
        if schools.empty:
            raise ValueError("No schools for secondary study code %s; expected NOGA %s" % (code, codes))
        destinations, valid = choose_schools(people, schools, rng)
        ids = people["person_id"].to_numpy()
        result.loc[ids, "destination_id"] = destinations
        result.loc[ids, ["x", "y"]] = sites.loc[destinations, ["x", "y"]].to_numpy()
        result.loc[ids, "education_secondary_method"] = np.where(
            valid, "compatible_structural_survey_distance" if distance_model is not None else "compatible_donor_distance",
            "compatible_nearest_missing_distance")
    return result.reset_index(drop=True)


def sample_secondary_zones(persons, coordinates, zone_ids, zone_sites, probabilities, targets, rng):
    """Condition a pooled OD row on each student's existing SS distance target.

    A zone is evaluated at its best compatible school-distance match, rather
    than its centroid: large municipalities can contain both near and far sites.
    The broad, shared tolerance (1 km or 10% of the target) allows geographic
    variation without prescribing an exact school distance. Direction still
    comes from the pooled OD row. No new routes or programme-specific OD tables
    are introduced; missing SS evidence leaves the original row unchanged.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    homes = persons[["home_x", "home_y"]].to_numpy(dtype=float)
    targets = np.asarray(targets, dtype=float)
    prior = np.asarray(probabilities, dtype=float)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2
            or not np.isfinite(coordinates).all() or not np.isfinite(homes).all()
            or len(targets) != len(persons) or prior.shape != (len(zone_ids),)
            or len(zone_sites) != len(zone_ids) or not np.isfinite(prior).all()
            or (prior < 0).any() or prior.sum() <= 0
            or any(len(indices) == 0 for indices in zone_sites)):
        raise ValueError("Secondary zone conditioning needs finite locations and supported compatible OD zones")
    valid = np.isfinite(targets) & (targets >= 0)
    sigma = np.maximum(1000., .1 * np.where(valid, targets, 0.))
    # Flatten membership once, including sites belonging to both a quarter and
    # its municipality. Reducing each segment gives one error per candidate zone
    # without rewarding a zone merely for having more establishment records.
    indices = np.concatenate(zone_sites)
    starts = np.cumsum([0] + [len(i) for i in zone_sites[:-1]])
    log_prior = np.full(len(prior), -np.inf)
    positive = prior > 0
    log_prior[positive] = np.log(prior[positive])
    draws = rng.random_sample(len(persons))
    selected = np.empty(len(persons), dtype=int)
    for start in range(0, len(persons), 256):
        stop = min(start + 256, len(persons))
        distance = np.hypot(homes[start:stop, 0, None] - coordinates[None, :, 0],
                            homes[start:stop, 1, None] - coordinates[None, :, 1])
        target = np.where(valid[start:stop], targets[start:stop], 0.)
        errors = np.minimum.reduceat(np.abs(distance[:, indices] - target[:, None]), starts, axis=1)
        log_weight = log_prior[None, :] - .5 * (errors / sigma[start:stop, None]) ** 2
        log_weight[~valid[start:stop]] = log_prior
        # Log-space normalisation still selects a supported zone when all
        # compatible schools lie far from the target; never flatten to uniform.
        weights = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
        cdf = (weights / weights.sum(axis=1, keepdims=True)).cumsum(axis=1)
        cdf[:, -1] = 1.
        selected[start:stop] = (cdf <= draws[start:stop, None]).sum(axis=1)
    return np.asarray(zone_ids)[selected], valid, np.where(valid, sigma, np.nan)


def assign_secondary_od_locations(persons, locations, statent, zones, pdf, seed):
    """Use one pooled secondary OD row, retaining programme-compatible schools.

    The same OD row supplies directions for all secondary programmes. Existing
    SS age/programme distance targets condition zone choice before within-zone
    school selection, preventing long vocational flows from dominating shorter
    compulsory/gymnasium commutes. Unsupported origins retain labelled proxies.
    """
    import pandas as pd
    if np.shape(pdf) != (len(zones), len(zones)) or zones.zone_id.duplicated().any():
        raise ValueError("Secondary OD probabilities require unique spatial-zone order")
    result = locations.copy().set_index("person_id", drop=False)
    result["education_secondary_od_status"] = "outside_domestic_secondary_cohort"
    result["education_secondary_origin_zone_id"] = pd.Series(np.nan, index=result.index, dtype=object)
    result["education_secondary_destination_zone_id"] = pd.Series(np.nan, index=result.index, dtype=object)
    result["education_secondary_zone_distance_status"] = "not_applicable"
    result["education_secondary_zone_distance_sigma_m"] = np.nan
    positions = dict(zip(zones.zone_id, range(len(zones))))
    valid_origins = set(zones.loc[zones.zone_level.isin(["municipality", "quarter"]), "zone_id"])
    cohort = persons.loc[persons.age.ge(15) & persons.is_student.eq(1)
        & persons.education_destination_status.eq("domestic") & persons.current_education.isin(SCHOOL_CODES)]
    rng = np.random.RandomState(seed)
    for code, students in cohort.groupby("current_education", sort=True):
        sites = statent.loc[statent.noga.astype(str).isin(SCHOOL_CODES[code])].copy()
        pools = {}
        for site in sites.itertuples():
            for zone in {site.zone_id, site.zone_municipality_id}:
                if pd.notna(zone) and zone in positions:
                    pools.setdefault(zone, []).append(site.enterprise_id)
        eligible_zones = sorted(pools)
        sites = sites.set_index("enterprise_id", drop=False)
        zone_sites = [sites.index.get_indexer(pools[z]) for z in eligible_zones]
        coordinates = sites[["x", "y"]].to_numpy(dtype=float)
        for origin, group in students.groupby("home_zone_id", sort=True, dropna=False):
            ids = group.person_id.to_numpy()
            result.loc[ids, "education_secondary_origin_zone_id"] = origin
            result.loc[ids, "education_secondary_od_status"] = "distance_proxy_missing_origin"
            if pd.isna(origin) or origin not in valid_origins:
                continue
            row = np.asarray(pdf[positions[origin]], dtype=float)
            if np.isnan(row).all():
                result.loc[ids, "education_secondary_od_status"] = "distance_proxy_unsupported_origin"
                continue
            if not np.isfinite(row).all() or (row < 0).any():
                raise ValueError("Invalid secondary OD probabilities")
            probabilities = row[[positions[z] for z in eligible_zones]]
            if probabilities.sum() <= 0:
                result.loc[ids, "education_secondary_od_status"] = "distance_proxy_no_compatible_destination"
                continue
            # Read the fitted SS target explicitly from location metadata. A
            # donor/reference MZ distance must not masquerade as SS evidence in
            # pure-function callers that do not supply the distance model.
            targets = (result.loc[ids, "education_secondary_target_distance_m"].to_numpy(dtype=float)
                       if "education_secondary_target_distance_m" in result else np.full(len(group), np.nan))
            destinations, conditioned, sigma = sample_secondary_zones(
                group, coordinates, eligible_zones, zone_sites, probabilities, targets, rng)
            result.loc[ids, "education_secondary_zone_distance_status"] = np.where(
                conditioned, "ss_age_programme_distance", "unconditioned_missing_ss_target")
            result.loc[ids, "education_secondary_zone_distance_sigma_m"] = sigma
            choices = group.assign(_destination_zone=destinations)
            for destination, members in choices.groupby("_destination_zone", sort=True):
                candidates = sites.loc[pools[destination]]
                selected, _ = choose_schools(members, candidates, rng)
                ids = members.person_id.to_numpy()
                result.loc[ids, ["destination_id", "x", "y"]] = candidates.loc[selected, ["enterprise_id", "x", "y"]].to_numpy()
                result.loc[ids, "education_secondary_destination_zone_id"] = destination
                result.loc[ids, "education_secondary_od_status"] = "pooled_secondary_od"
                result.loc[ids, "education_assignment_status"] = "pooled_secondary_od"
    return result.reset_index(drop=True)
