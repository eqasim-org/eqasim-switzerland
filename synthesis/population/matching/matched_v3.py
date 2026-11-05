"""Survey-weighted categorical matching without ordered attribute relaxation."""

import hashlib
import json
import logging
from pathlib import Path
import time

import numba
import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series
from synthesis.population.models.non_private import home_only_mask
from . import matched_v1
from .V3 import donors as donor_tools, inputs

logger = logging.getLogger("synpp")


def configure(context):
    matched_v1.configure(context)
    context.config("threads", volatile=True)


def execute(context):

    # Prepare V1 categories while keeping the explicit home-only group outside matching.

    started = time.perf_counter()
    const = context.stage("data.constants")
    source, population, roster = inputs.prepare_inputs(context, const)
    threshold = int(context.config("matching_minimum_observations"))
    if threshold < 1:
        raise ValueError("matching_minimum_observations must be positive")
    ids = ["person_id"] + (["household_id"] if "household_id" in roster else [])
    result = roster[ids].copy()
    result["mz_person_id"] = -1
    eligible = population.age.ge(const.MZ_AGE_THRESHOLD) if const.census == "statpop" else population.age_class.gt(0)
    collective = coerce_boolean_series(population.get("collective_housing_resident",
        pd.Series(False, index=population.index)), name="collective_housing_resident")

    # Match each age/residence segment using exact hard blocks and JIT profile searches.

    diagnostics, removed = [], set()
    previous_threads = numba.get_num_threads()
    requested_threads = int(context.config("threads"))
    if requested_threads < 1:
        raise ValueError("threads must be positive")
    numba.set_num_threads(min(requested_threads, numba.config.NUMBA_NUM_THREADS))
    try:
        for segment, group, columns, mandatory, target_mask, source_mask in _segments(population, source, const):
            for is_collective in (False, True):
                target = population.loc[eligible & target_mask & collective.eq(is_collective)].copy()
                if target.empty:
                    continue
                donors = source.loc[(source_mask if not is_collective else True) &
                    (source.home_only if is_collective else True)].copy()
                weight = "household_weight" if is_collective or const.census == "are_synpop" else "person_weight"
                weights = pd.to_numeric(donors[weight], errors="coerce")
                valid = np.isfinite(weights) & weights.gt(0)
                if not valid.all():
                    logger.warning("V3 excluded %d invalid-weight donors in %s", (~valid).sum(), segment)
                    donors = donors.loc[valid].copy()
                assigned, details = donor_tools.select_donors(donors, target, columns, mandatory, inputs.HARD[group],
                    seed=int(context.config("random_seed")) + segment * 2 + int(is_collective),
                    threshold=threshold, weight=weight)
                details["age_group"], details["collective"] = group, is_collective
                diagnostics.append(details)
                mapping = pd.Series(assigned, index=target.person_id)
                selector = result.person_id.isin(target.person_id)
                result.loc[selector, "mz_person_id"] = result.loc[selector, "person_id"].map(mapping).to_numpy()
                removed.update(target.loc[assigned < 0, "person_id"])
                if is_collective and "household_id" in target:
                    households = target.loc[assigned < 0, "household_id"]
                    removed.update(population.loc[population.household_id.isin(households), "person_id"])
    finally:
        numba.set_num_threads(previous_threads)

    # Export search support and preserve the original mapping contract and recipient order.

    columns = ["person_id", "mz_person_id", "mandatory_mismatches", "optional_radius", "donor_pool_size", "below_threshold", "age_group", "collective"]
    details = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame(columns=columns)
    details["removed"] = details.person_id.isin(removed)
    output = Path(context.path())
    output.mkdir(parents=True, exist_ok=True)
    details.to_csv(output / "matching_diagnostics.csv", index=False)
    report = {"algorithm": "V3", "threshold": threshold, "eligible_recipients": len(details),
        "matched": int((details.mz_person_id.ge(0) & ~details.removed).sum()), "removed": len(removed),
        "removed_eligible": int(details.removed.sum()), "retained_population": len(roster) - len(removed),
        "held_home_only": int(home_only_mask(roster).sum()),
        "below_threshold": int(details.below_threshold.sum()),
        "mandatory_relaxed": int(details.mandatory_mismatches.gt(0).sum()),
        "elapsed_seconds": time.perf_counter() - started}
    (output / "matching_report.json").write_text(json.dumps(report, indent=2))
    logger.info("V3 matching: %s", report)
    result = result.loc[~result.person_id.isin(removed)].reset_index(drop=True)
    if not set(details.person_id) == set(population.loc[eligible, "person_id"]):
        raise RuntimeError("V3 segments did not cover every eligible recipient exactly once")
    return result, removed


def _segments(population, source, const):
    if const.census == "statpop":
        for number, (group, low, high, columns, mandatory) in enumerate([
            ("child", 6, 15, inputs.CHILD, inputs.CHILD_MANDATORY),
            ("youth", 15, 24, inputs.YOUTH, inputs.YOUTH_MANDATORY),
            ("adult", 24, np.inf, inputs.ADULT, inputs.ADULT_MANDATORY)]):
            yield number, group, columns, mandatory, population.age.ge(low) & population.age.lt(high), source.age.ge(low) & source.age.lt(high)
    else:
        # ARE provides intervals: its 6–17 class cannot be split at age 15.
        # Use youth constraints throughout 6–24, and adult constraints thereafter.
        columns = ["ovgk", "age_class", "sex", "employment_status", "number_of_cars_class", "N_children_under_18", "car_availability", "employed"]
        for number, group, low, high in ((0, "youth", 1, 3), (1, "adult", 3, 100)):
            yield number, group, columns, columns[:4], population.age_class.ge(low) & population.age_class.lt(high), source.age_class.ge(low) & source.age_class.lt(high)


def validate(context):
    paths = [Path(donor_tools.__file__), Path(inputs.__file__), Path(matched_v1.__file__),
             Path(__file__).with_name("commute.py"), Path(__file__).parent.parent / "models" / "non_private.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
