"""Swiss secondary-activity sites within 25 metres of STATENT schools or daycare."""
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from . import subtypes

OPTION = "secondary_activities.force_model.child_escort_school_multiplier"
STAGE = "synthesis.population.spatial.secondary.force_model.school_proximity"
MULTI_CHILD = "other_escort_child_multi"


def configure(context):
    context.stage(subtypes.STAGE)
    context.stage("data.statent.statent")
    context.stage("data.spatial.swiss_border")


def execute(context):

    # Match the same school and daycare sectors used when fitting the MZ preferences.

    jobs = context.stage("data.statent.statent").dropna(subset=["x", "y"])
    schools = jobs.loc[jobs.noga.astype(str).str.startswith(subtypes.SECTORS["school_daycare"]), ["x", "y"]]
    index = context.stage(subtypes.STAGE).index
    xy = index.to_frame(index=False).to_numpy()
    border = context.stage("data.spatial.swiss_border").to_crs(2056).geometry.union_all()
    shapely.prepare(border)

    # Apply the Swiss facility inventory only within its geographic coverage.

    inside = shapely.intersects_xy(border, *xy.T)
    near = np.zeros(len(index), dtype=bool)
    near[inside] = cKDTree(schools).query(xy[inside])[0] <= 25.
    return pd.Series(near, index=index, name="school_adjacent")


def multipliers(value):
    """Return weights for one and for two or more secondary stops between anchors."""
    values = value if isinstance(value, (tuple, list)) else (value, value)
    if len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                               not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"{OPTION} needs one positive finite weight or [single_stop, multi_stop]")
    return tuple(values)


def adjusted_scores(scores, near, multiplier):
    if not scores.index.equals(near.index):
        raise ValueError("School proximity must be aligned with the eligible location scores")
    single, multi = multipliers(multiplier)
    adjusted = scores.copy()
    if single != multi:
        adjusted[MULTI_CHILD] = adjusted["other_escort_child"]
        adjusted.loc[near, MULTI_CHILD] *= multi
    adjusted.loc[near, "other_escort_child"] *= single
    return adjusted
