"""Choose compatible domestic tertiary zones from the pooled education OD matrix."""

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


LEVEL_CODES = {9: "higher_professional_examination", 10: "higher_vocational_college",
               11: "bachelor", 12: "master", 13: "doctorate_habilitation"}


def domestic_cohort(persons, active_person_ids=None):
    """Select the pooled domestic tertiary stock before diary matching.

    Enrolment alone does not imply daily travel. Unknown and foreign study
    destinations keep separate proxy handling and do not contribute to domestic OD.
    """
    required = {"person_id", "age", "is_student", "education_destination_status"}
    if not required.issubset(persons.columns):
        return persons.iloc[0:0].copy().assign(study_level=pd.Series(dtype=str))
    tertiary = persons.get("study_level", pd.Series("", index=persons.index)).eq("tertiary")
    if "current_education" in persons:
        tertiary |= persons.current_education.isin(LEVEL_CODES)
    selected = (persons["age"].ge(15)
                & persons["is_student"].eq(1) & tertiary
                & persons["education_destination_status"].eq("domestic"))
    if active_person_ids is not None:
        selected &= persons.person_id.isin(active_person_ids)
    result = persons.loc[selected.fillna(False)].copy()
    result["study_level"] = "tertiary"
    return result


def zone_candidates(statent, compatibility, level):
    ids = compatibility.loc[compatibility["study_level"].eq(level), "enterprise_id"]
    sites = statent[statent["enterprise_id"].isin(ids)].copy()
    # Carry campus identity into establishment choice. A large campus with many
    # nearby buildings must not exclude another campus from a nearest-site list.
    campus = compatibility[compatibility["study_level"].eq(level)][["enterprise_id", "campus_id"]]
    sites = sites.merge(campus, on="enterprise_id", validate="one_to_one")
    pools = {}
    for site in sites.itertuples():
        # A municipality includes its quarters; a quarter includes only its sites.
        for zone in set([site.zone_id, site.zone_municipality_id]):
            if pd.notna(zone):
                pools.setdefault(zone, []).append(site.enterprise_id)
    return sites.set_index("enterprise_id", drop=False), pools


def sample_balanced_alternatives(model, origin, count, rng):
    """Assign the rounded origin/campus counts, then sample zones within groups."""
    row = model["origins"][origin]
    counts = model["integer_flows"][row]
    if counts.sum() != count:
        raise ValueError("Education assignment cohort differs from its rounded origin margin")
    labels = np.repeat(np.arange(len(counts)), counts)
    rng.shuffle(labels)
    selected = np.empty(count, dtype=int)
    alternative_groups = np.array([group for zone, group in model["alternatives"]])
    for column, group in enumerate(model["groups"]):
        people = np.flatnonzero(labels == column)
        if not len(people):
            continue
        alternatives = np.flatnonzero(alternative_groups == group)
        probabilities = model["probabilities"][row, alternatives]
        if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
            raise ValueError("Rounded campus count has no supported destination-zone probability")
        selected[people] = rng.choice(alternatives, size=len(people), p=probabilities / probabilities.sum())
    return selected


def choose_campus_establishments(persons, candidates, rng):
    """Choose a campus by eligible FTE before ranking buildings by commute distance.

    Named calibration alternatives already contain a single campus. Residual
    groups may contain several: selecting among their aggregate FTE first keeps
    every compatible campus eligible, even when another has over 20 buildings.
    The existing commute-distance preference is retained inside the chosen campus.
    """
    from synthesis.population.spatial.primary.education.locations import get_distance_weight, MIN_PROB

    if candidates["campus_id"].isna().any():
        raise ValueError("Tertiary candidates require a campus identity")
    fte = candidates["number_employees_fte"].to_numpy(dtype=float)
    if not np.isfinite(fte).all() or (fte < 0).any():
        raise ValueError("Campus choice requires nonnegative finite eligible-site FTE")
    campus_weights = candidates.groupby("campus_id", sort=True)["number_employees_fte"].sum()
    weights = campus_weights.to_numpy(dtype=float)
    weights = weights if weights.sum() > 0 else np.ones(len(weights))
    campus_choices = rng.choice(campus_weights.index.to_numpy(), size=len(persons), p=weights / weights.sum())
    picked_ids = np.empty(len(persons), dtype=object)
    for campus in campus_weights.index:
        selected = np.flatnonzero(campus_choices == campus)
        if not len(selected):
            continue
        buildings = candidates[candidates["campus_id"].eq(campus)]
        distance, indices = KDTree(buildings[["x", "y"]].to_numpy()).query(
            persons.iloc[selected][["home_x", "home_y"]].to_numpy(), min(20, len(buildings)))
        weight = buildings["number_employees"].to_numpy(dtype=float)
        weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.)
        low, high = np.quantile(weight, [.25, .75])
        weight = np.clip(weight, low, high) if low < high else weight
        weight = np.maximum(weight / weight.sum(), MIN_PROB) if weight.sum() > 0 else np.ones(len(weight))
        probabilities = get_distance_weight(
            distance - persons.iloc[selected]["commute_home_distance"].to_numpy()[:, None]) * weight[indices]
        probabilities /= probabilities.sum(axis=1)[:, None]
        picks = np.array([rng.multinomial(1, row).argmax() for row in probabilities])
        picked_ids[selected] = buildings.iloc[indices[np.arange(len(selected)), picks]]["enterprise_id"].to_numpy()
    return picked_ids


def assign_od_locations(persons, locations, statent, compatibility, zones, pdf, active_person_ids, seed, balancing=None):
    if zones["zone_id"].duplicated().any() or np.shape(pdf) != (len(zones), len(zones)):
        raise ValueError("Tertiary OD matrix must follow the unique spatial-zone ordering")
    if locations["person_id"].duplicated().any() or not locations.index.is_unique:
        raise ValueError("Primary education locations must be unique per person")
    result = locations.copy()
    # Start from complete compatible proxy locations. A failed OD lookup records
    # a reason but never drops a person or leaves their location missing.
    result["education_od_status"] = "outside_domestic_tertiary_cohort"
    # Match data.spatial.zones: identifiers use object columns, including missing
    # values. Pandas 3 rejects object ID arrays written into NaN-initialised float
    # columns, even when the non-missing IDs happen to be integers.
    for column in ["education_od_origin_zone_id", "education_od_destination_zone_id"]:
        result[column] = pd.Series(np.nan, index=result.index, dtype=object)
    result["education_od_compatible_mass"] = np.nan
    result["education_od_calibration_group"] = ""
    cohort = domestic_cohort(persons, active_person_ids)
    if cohort.empty:
        return result
    if cohort["person_id"].duplicated().any() or not cohort["person_id"].isin(result["person_id"]).all():
        raise ValueError("Domestic tertiary education participants require one existing primary location")
    if "home_zone_id" not in cohort:
        raise ValueError("Domestic tertiary OD assignment requires home_zone_id")
    zone_position = dict(zip(zones["zone_id"], range(len(zones))))
    valid_origins = set(zones.loc[zones["zone_level"].isin(["municipality", "quarter"]), "zone_id"])
    rng = np.random.RandomState(seed)
    row_position = dict(zip(result["person_id"], result.index))
    for level, students in cohort.groupby("study_level", sort=True):
        model = balancing["models"].get(level) if balancing is not None else None
        sites, pools = zone_candidates(statent, compatibility, level)
        if sites.empty:
            raise ValueError("No eligible domestic establishments for %s" % level)
        eligible_zones = sorted(zone for zone in pools if zone in zone_position)
        positions = np.array([zone_position[zone] for zone in eligible_zones], dtype=int)
        choices = []
        grouping = "home_zone_id"
        if model is not None and "age_limit" in model:
            # Use the same row keys as balancing, so each origin/age cohort
            # receives exactly its reserved integer campus counts.
            students = students.assign(education_od_age_band=students.age.ge(model["age_limit"]).astype(int))
            grouping = ["home_zone_id", "education_od_age_band"]
        for origin_key, group in students.groupby(grouping, sort=True, dropna=False):
            origin = origin_key[0] if isinstance(grouping, list) else origin_key
            indices = [row_position[person] for person in group["person_id"]]
            result.loc[indices, "education_od_origin_zone_id"] = origin
            if pd.isna(origin) or origin not in valid_origins:
                result.loc[indices, "education_od_status"] = "compatible_location_proxy_missing_origin"
                continue
            row = np.asarray(pdf[zone_position[origin]], dtype=float)
            if np.isnan(row).all() or (np.isfinite(row).all() and row.sum() == 0 and not (row < 0).any()):
                result.loc[indices, "education_od_status"] = "compatible_location_proxy_unsupported_origin"
                continue
            if not np.isfinite(row).all() or (row < 0).any():
                raise ValueError("Invalid tertiary OD probabilities for origin %s" % origin)
            row = row / row.sum()
            mass = row[positions].sum()
            result.loc[indices, "education_od_compatible_mass"] = mass
            if mass <= 0:
                result.loc[indices, "education_od_status"] = "compatible_location_proxy_no_supported_destination"
                continue
            if model is not None and origin_key in model["origins"]:
                # Choose the campus/type group together with the zone. Choosing
                # an unrestricted establishment afterward would undo calibration.
                selected = sample_balanced_alternatives(model, origin_key, len(group), rng)
                for person, alternative in zip(group["person_id"], selected):
                    destination, calibration_group = model["alternatives"][alternative]
                    choices.append((person, destination, int(alternative), calibration_group))
            else:
                selected_zones = rng.choice(eligible_zones, size=len(group), p=row[positions] / mass)
                for person, destination in zip(group["person_id"], selected_zones):
                    choices.append((person, destination, -1, ""))
        if not choices:
            continue
        chosen = pd.DataFrame(choices, columns=["person_id", "destination_zone_id", "alternative", "calibration_group"]).merge(
            students, on="person_id", validate="one_to_one")
        for (destination, alternative), group in chosen.groupby(["destination_zone_id", "alternative"], sort=True):
            candidates = sites.loc[model["site_pools"][alternative] if alternative >= 0 else pools[destination]]
            picked_sites = candidates.loc[choose_campus_establishments(group, candidates, rng)]
            indices = [row_position[person] for person in group["person_id"]]
            result.loc[indices, ["destination_id", "x", "y"]] = picked_sites[["enterprise_id", "x", "y"]].to_numpy()
            result.loc[indices, "education_od_destination_zone_id"] = destination
            result.loc[indices, "education_od_status"] = "pooled_tertiary_od"
            if alternative >= 0:
                result.loc[indices, "education_od_status"] = "balanced_tertiary_od"
                result.loc[indices, "education_od_calibration_group"] = group["calibration_group"].to_numpy()
    return result
