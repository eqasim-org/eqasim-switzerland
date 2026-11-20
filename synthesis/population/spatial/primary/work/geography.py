import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

logger = logging.getLogger("synpp")


def destination_geography(zones, quarters, municipalities, shapes, facilities, survey):
    """Map each survey destination to feasible establishment zones before rounding."""
    ids = pd.Index(zones.zone_id)
    levels = zones.zone_level.to_numpy()
    codes = pd.to_numeric(zones.zone_level_id, errors="coerce")
    quarter_parent = quarters.set_index("quarter_id").source_municipality_id
    translation = municipalities.set_index("deprecated_municipality_id").municipality_id
    parents = codes.map(quarter_parent).map(translation).where(levels == "quarter", codes.where(levels == "municipality"))
    jobs = facilities.groupby("zone_id").number_employees.sum().reindex(ids, fill_value=0).to_numpy()
    reported = survey.groupby("work_zone_id").weight.sum().reindex(ids, fill_value=0).to_numpy()
    municipality_ids = parents.to_numpy()
    feasible = np.flatnonzero(jobs > 0)
    if not len(feasible):
        raise ValueError("No Swiss workplace capacity")
    tree = cKDTree(facilities[["x", "y"]])
    locations = shapes.set_index("zone_id").geometry.representative_point()
    rows, cols, weights = [], [], []
    no_local_jobs = []
    for i, (zone, level, municipality) in enumerate(zip(ids, levels, parents)):
        if level not in ("quarter", "municipality"):
            continue
        if level == "quarter" and jobs[i] > 0:
            choices, shares = [i], [1.]
        else:
            within = feasible[municipality_ids[feasible] == municipality]
            if len(within):
                shares = jobs[within] / jobs[within].sum()
                known = (levels[within] == "quarter") & (reported[within] > 0)
                if known.any():
                    quarter_share = shares[levels[within] == "quarter"].sum()
                    shares[levels[within] == "quarter"] = 0
                    shares[known] = quarter_share * reported[within[known]] / reported[within[known]].sum()
                choices = within
            else:
                no_local_jobs.append(i)
                point = locations.loc[zone]
                _, j = tree.query([point.x, point.y])
                choices, shares = [ids.get_loc(facilities.zone_id.iat[j])], [1.]
                logger.debug("Work zone %s uses nearest available establishment zone %s", zone, ids[choices[0]])
        rows.extend([i] * len(choices))
        cols.extend(choices)
        weights.extend(shares)
    logger.info("%d domestic destination zones have no municipal establishment, covering %.0f in pooled survey weight",
                len(no_local_jobs), reported[no_local_jobs].sum())
    return csr_matrix((weights, (cols, rows)), shape=(len(ids), len(ids))), parents.to_numpy()


def round_workers(n, probabilities, rng):
    """Round an origin total systematically, or by largest remainder when rng is None."""
    indices = np.flatnonzero(probabilities > 0)
    mass = probabilities[indices]
    expected = n * mass / mass.sum()
    counts = np.floor(expected).astype(int)
    remaining = n - counts.sum()
    if remaining and rng is None:
        picks = np.argsort(expected - counts, kind="stable")[-remaining:]
        counts[picks] += 1
    elif remaining:
        cumulative = np.cumsum(expected - counts)
        cumulative[-1] = remaining
        picks = np.searchsorted(cumulative, np.arange(remaining) + rng.random(), side="right")
        np.add.at(counts, picks, 1)
    return dict(zip(indices[counts > 0], counts[counts > 0]))


def municipal_factors(cities, parents, rows, targets):
    """Fit municipality margins in expectation, preserving each origin's OD support."""
    if len(set(cities)) != len(cities) or not set(cities).issubset(parents):
        raise ValueError("Specify distinct, existing BFS municipality IDs for workplace calibration")
    groups = np.zeros(len(parents), dtype=int)
    for index, city in enumerate(cities, start=1):
        groups[parents == city] = index
    totals, probs = [], []
    for count, row in rows:
        totals.append(count)
        probs.append(np.bincount(groups, weights=row, minlength=len(cities) + 1))
    if not totals or np.any(targets <= 0):
        raise ValueError(f"Insufficient survey or resident support for workplace calibration {cities}")
    probs = np.array(probs)

    def residual(beta):
        weighted = probs * np.r_[1., np.exp(beta)]
        fitted = totals @ (weighted / weighted.sum(axis=1)[:, None])
        return (fitted[1:] - targets) / targets

    fit = least_squares(residual, np.zeros(len(cities)), bounds=(-25, 25), gtol=1e-12)
    if np.max(np.abs(residual(fit.x))) > 1e-5:
        raise ValueError(f"Infeasible municipal workplace targets for {cities}: {targets}")
    logger.info("Municipal workplace odds multipliers: %s", dict(zip(cities, np.exp(fit.x))))
    return np.r_[1., np.exp(fit.x)][groups]
