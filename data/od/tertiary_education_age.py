"""A national age preference on the shared tertiary OD support.

Only students below 24 receive a fitted distance tilt. Held-out years support
this preference; an opposite tilt for older students worsened their log loss.
No degree-specific or canton-specific matrices are estimated here.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from data.od.education_matrix import prepare_observations

AGE_LIMIT = 24
MAX_DECAY = .05  # Numerical fitting bound per km, not a prescribed coefficient.
logger = logging.getLogger("synpp")


def configure(context):
    for stage in ["data.structural_survey.structural_survey", "data.spatial.zones",
                  "data.spatial.municipalities", "data.spatial.quarters",
                  "data.od.tertiary_education_matrix"]:
        context.stage(stage)


def validate(context):
    import hashlib
    from pathlib import Path
    from data.od import education_matrix
    from data.structural_survey import education_origins
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(education_matrix.__file__), Path(education_origins.__file__)]}


def zone_centres(zones, municipalities, quarters):
    """Use the same municipality/quarter resolution as the OD probabilities."""
    centres = {}
    for level, frame, field in [("municipality", municipalities, "municipality_id"),
                                ("quarter", quarters, "quarter_id")]:
        geometry = frame.to_crs(2056).geometry.centroid
        lookup = dict(zip(frame[field], zip(geometry.x, geometry.y)))
        for row in zones.loc[zones.zone_level.eq(level)].itertuples():
            centres[row.zone_id] = lookup[row.zone_level_id]
    return centres


def distance_matrix(origins, destinations, centres):
    a = np.asarray([centres[i] for i in origins], dtype=float)
    b = np.asarray([centres[i] for i in destinations], dtype=float)
    distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2) / 1000.
    if not np.isfinite(distance).all():
        raise ValueError("Age-distance adjustment requires finite zone coordinates")
    return distance


def adjust_probabilities(probabilities, distance_km, decay):
    """Soft reweighting conserves support; no long trip is explicitly forbidden."""
    values = np.asarray(probabilities, dtype=float)
    distance = np.asarray(distance_km, dtype=float)
    if (values.shape != distance.shape or not np.isfinite(values).all()
            or (values < 0).any() or not np.isfinite(distance).all()
            or (distance < 0).any() or not np.isfinite(decay)
            or decay < 0 or decay > MAX_DECAY or (values.sum(axis=-1) <= 0).any()):
        raise ValueError("Invalid age-distance probabilities, distances or decay")
    # Subtract the nearest supported distance to keep exponentials stable.
    nearest = np.where(values > 0, distance, np.inf).min(axis=-1, keepdims=True)
    adjusted = values * np.exp(np.maximum(-decay * (distance - nearest), -700.))
    return adjusted / adjusted.sum(axis=-1, keepdims=True)


def fit_decay(profiles, distances, observations):
    """Fit one coefficient by conditional destination likelihood, not mean matching.

    Reported departure locations are primary evidence. Residence fallback stays
    in the pooled base OD model but cannot teach the age preference an assumed
    long commute. Missing reported distance is acceptable: both fitting and
    prediction use observed OD-zone centroids, never mix distance definitions.
    """
    selected = observations.loc[observations.age.lt(AGE_LIMIT)
                                & observations.education_origin_source.eq("school_departure")]
    rows = profiles.index.get_indexer(selected.home_zone_id)
    columns = profiles.columns.get_indexer(selected.education_zone_id)
    known = (rows >= 0) & (columns >= 0)
    base = profiles.to_numpy(dtype=float)
    known[known] &= base[rows[known], columns[known]] > 0
    weight = pd.to_numeric(selected.weight, errors="coerce").to_numpy(dtype=float)
    known &= np.isfinite(weight) & (weight > 0)
    if not known.any():
        return {"decay_per_km": 0., "observations": 0, "status": "no_supported_departures"}
    rows, columns, weight = rows[known], columns[known], weight[known]
    row_weight = np.bincount(rows, weights=weight, minlength=len(base)) / weight.sum()
    observed_distance = np.average(distances[rows, columns], weights=weight)

    def loss(decay):
        normaliser = (base * np.exp(-decay * distances)).sum(axis=1)
        return decay * observed_distance + row_weight @ np.log(normaliser)

    fit = minimize_scalar(loss, bounds=(0., MAX_DECAY), method="bounded",
                          options={"xatol": 1e-9})
    if not fit.success:
        raise ValueError("Could not fit tertiary age-distance preference")
    decay = float(fit.x) if fit.fun < loss(0.) else 0.
    return {"decay_per_km": decay, "observations": int(known.sum()),
            "effective_n": float(weight.sum() ** 2 / (weight ** 2).sum()),
            "status": "bound_reached" if decay > MAX_DECAY - 1e-6 else "fitted"}


def execute(context):
    survey = context.stage("data.structural_survey.structural_survey")
    observations = prepare_observations(survey, tertiary_only=True)
    observations["age"] = survey.loc[observations.index, "age"]
    zones = context.stage("data.spatial.zones")
    centres = zone_centres(zones, context.stage("data.spatial.municipalities")[0],
                           context.stage("data.spatial.quarters"))
    origins = zones.loc[zones.zone_level.isin(["municipality", "quarter"]), "zone_id"]
    destinations = sorted(observations.education_zone_id.unique())
    positions = dict(zip(zones.zone_id, range(len(zones))))
    pdf = context.stage("data.od.tertiary_education_matrix")[0]
    profiles = pd.DataFrame(pdf[np.ix_([positions[i] for i in origins],
                                      [positions[i] for i in destinations])],
                            index=origins, columns=destinations)
    result = fit_decay(profiles, distance_matrix(origins, destinations, centres), observations)
    result.update(age_limit=AGE_LIMIT, zone_centres=centres)
    logger.info("Young tertiary OD preference: decay %.6f per km, %d reported-departure observations (%s)",
                result["decay_per_km"], result["observations"], result["status"])
    if result["status"] == "bound_reached":
        logger.warning("Tertiary age-distance fit reached its numerical bound; inspect held-out-year diagnostics")
    return result
