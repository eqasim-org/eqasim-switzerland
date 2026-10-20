"""A stable person split shared by all experts and their selector."""

import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.microcensus.persons")
    context.config("random_seed")
    context.config("secondary_nn_selector_person_fraction", 0.3)


def execute(context):
    fraction = float(context.config("secondary_nn_selector_person_fraction"))
    if not 0 < fraction < 1:
        raise ValueError("secondary_nn_selector_person_fraction must lie between zero and one")
    persons = context.stage("data.microcensus.persons")
    ids = np.sort(persons.loc[~persons.weekend, "person_id"].unique())
    if len(ids) < 10:
        raise ValueError("Expert/selector training requires at least ten weekday persons")
    rng = np.random.default_rng(int(context.config("random_seed")))
    ids = rng.permutation(ids)
    count = max(2, min(len(ids) - 2, int(np.ceil(len(ids) * fraction))))
    return pd.DataFrame({"person_id": ids, "selector": np.arange(len(ids)) < count})


def expert_persons(context, frame):
    if context.config("secondary_nn_use_target_distance"):
        return frame.copy()
    split = context.stage("synthesis.population.spatial.secondary_nn.training_split")
    return frame.loc[frame.person_id.isin(split.loc[~split.selector, "person_id"])].copy()
