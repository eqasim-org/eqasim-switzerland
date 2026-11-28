import hashlib
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data.od.matrix import DEFAULT_SEGMENT_KEY
from data.structural_survey.structural_survey import get_filtered_data
from .geography import destination_geography, round_workers
from .wla_tools import COMP_SATURATION_FACTOR, calculate_company_weights, correct_companies_number_of_employees
from .work_locations import get_segment_key

logger = logging.getLogger("synpp")
WORK = "synthesis.population.spatial.primary.work."
MATRIX = {"fixed": "data.od.matrix", "moving": "data.od.matrix_moving"}


def configure_assignment(context, kind):
    for stage in ("data.constants", "data.statent.statent", "data.spatial.zones", "data.spatial.quarters",
                  "data.spatial.municipalities", "data.spatial.zone_shapes", "data.structural_survey.structural_survey",
                  "synthesis.population.sampled", "synthesis.population.destinations", MATRIX[kind],
                  WORK + "work_at_different_locations"):
        context.stage(stage)
    context.stage(WORK + ("work_remotly" if kind == "fixed" else "fixed_work_locations"))
    if context.config("include_cross_border", False):
        context.stage("data.cross_border.destinations")
    if context.config("include_external_population", True):
        context.stage("data.external_population.commutes")
        context.stage("data.spatial.swiss_border")
        context.config("fr_sample_rate", default=1.0)
    if context.config("apprentice_geographic_trial", default=False):
        context.stage(WORK + "apprentice_municipalities")
    context.config("workplace_unbiased_rounding", default=False)
    context.config("random_seed")
    context.config("input_downsampling")


def assignment_version():
    # Synpp hashes stage entry points. Include the shared allocator and its helpers.
    digest = hashlib.sha256()
    for name in ("assignment.py", "geography.py", "wla_tools.py", "work_locations.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def assign(context, kind):
    persons = context.stage("synthesis.population.sampled")
    moving = context.stage(WORK + "work_at_different_locations").person_id
    mask = persons.employed.eq(context.stage("data.constants").EMPLOYED)
    if kind == "fixed":
        mask &= ~persons.person_id.isin(moving) & ~persons.person_id.isin(context.stage(WORK + "work_remotly").person_id)
    else:
        mask &= persons.person_id.isin(moving)
    workers = persons.loc[mask].sort_values("car_availability", kind="stable")
    fixed = context.stage(WORK + "fixed_work_locations") if kind == "moving" else None

    # Construct the destination support and remaining establishment capacity.

    zones = context.stage("data.spatial.zones")
    statent = context.stage("data.statent.statent")
    companies = statent[["enterprise_id", "x", "y", "zone_id", "number_employees"]]
    companies = companies.dropna().loc[lambda d: d.zone_id.isin(zones.zone_id) & d.number_employees.gt(0)]
    destinations = context.stage("synthesis.population.destinations")[["destination_id", "ovgk"]]
    companies = companies.merge(destinations, left_on="enterprise_id", right_on="destination_id",
                                how="left", validate="many_to_one")
    companies = correct_companies_number_of_employees(context, companies, fixed)
    survey = get_filtered_data(context, kind)
    mapping, _ = destination_geography(
        zones, context.stage("data.spatial.quarters"), context.stage("data.spatial.municipalities")[1],
        context.stage("data.spatial.zone_shapes"), companies, survey)
    ids = pd.Index(zones.zone_id)
    origins = ids.get_indexer(workers.home_zone_id)
    segments = [get_segment_key(sex, age) for sex, age in zip(workers.sex, workers.age)]
    keys = list(zip(segments, origins))
    matrix, _ = context.stage(MATRIX[kind])
    fallback = matrix[DEFAULT_SEGMENT_KEY]

    def probabilities(key):
        segment, origin = key
        if origin < 0:
            return None
        row = matrix.get(segment, fallback)[origin]
        if not np.isfinite(row).all() or row.sum() <= 0:
            row = fallback[origin]
        if not np.isfinite(row).all():
            return None
        result = mapping @ row
        return result / result.sum() if result.sum() > 0 else None

    # Round OD quotas independently of the establishment draws.

    training = pd.Series(np.nan, index=workers.index)
    by_municipality = {}
    if context.config("apprentice_geographic_trial"):
        training = workers.person_id.map(context.stage(WORK + "apprentice_municipalities"))
        municipality = companies.enterprise_id.map(statent.set_index("enterprise_id").municipality_id)
        by_municipality = companies.groupby(municipality, observed=True).indices
    training = training.to_numpy()
    ordinary = pd.isna(training)
    counts = Counter(key for key, use in zip(keys, ordinary) if use)
    seed = context.config("random_seed")
    rounding = np.random.default_rng(np.random.SeedSequence([seed, int(kind == "moving"), 1]))
    if not context.config("workplace_unbiased_rounding"):
        rounding = None
    quotas = {}
    for key, n in counts.items():
        row = probabilities(key)
        if row is not None:
            quotas[key] = round_workers(n, row, rounding)

    # Draw establishments using remaining capacity, accessibility and fixed-worker distance.

    company_zones = ids.get_indexer(companies.zone_id)
    by_zone = pd.Series(np.arange(len(companies))).groupby(company_zones).indices
    xy = companies[["x", "y"]].to_numpy()
    home = workers[["home_x", "home_y"]].to_numpy()
    jobs = companies.number_employees.to_numpy(dtype=float, copy=True)
    minimum = jobs * 1e-6
    decrement = COMP_SATURATION_FACTOR / context.config("input_downsampling")
    good_pt = companies.ovgk.isin(["A", "B"]).to_numpy()
    other_pt = companies.ovgk.isin(["C", "D"]).to_numpy()
    cars = workers.car_availability.to_numpy(dtype=bool)
    rng = np.random.RandomState(seed)
    selected = np.empty(len(keys), dtype=int)
    tree = None
    missing = 0
    with context.progress(total=len(keys), label=f"Assigning {kind} workplaces") as progress:
        for i, key in enumerate(keys):
            if not ordinary[i] or key in quotas:
                if ordinary[i]:
                    remaining = quotas[key]
                    candidates = np.concatenate([by_zone[z] for z in remaining])
                else:
                    candidates = by_municipality[training[i]]
                distances = None
                if kind == "fixed":
                    distances = np.hypot(xy[candidates, 0] - home[i, 0], xy[candidates, 1] - home[i, 1])
                weights = calculate_company_weights(candidates, cars[i], jobs, good_pt, other_pt, distances)
                chosen = rng.choice(candidates, p=weights)
                if ordinary[i]:
                    zone = company_zones[chosen]
                    remaining[zone] -= 1
                    if remaining[zone] == 0:
                        del remaining[zone]
            else:
                if tree is None:
                    tree = cKDTree(xy)
                _, chosen = tree.query(home[i])
                missing += 1
            selected[i] = chosen
            jobs[chosen] = max(jobs[chosen] - decrement, minimum[chosen])
            progress.update()
    logger.info("%s workers assigned using nearest workplace to home (missing domestic OD): %d", kind, missing)
    work_xy = xy[selected]
    return pd.DataFrame({"person_id": workers.person_id.to_numpy(),
                         "destination_id": companies.enterprise_id.to_numpy()[selected],
                         "commute_distance": np.linalg.norm(work_xy - home, axis=1),
                         "x": work_xy[:, 0], "y": work_xy[:, 1]})
