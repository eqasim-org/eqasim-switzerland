"""Descriptive residence masks and the agreed older-resident home-only bypass."""
import pandas as pd
import logging
import numpy as np

from data.utils import coerce_boolean_series


def dorm_candidate_mask(persons):
    # Returning False for older caches/private populations keeps legacy callers
    # usable. Parse logical values explicitly rather than bool('False').
    values = persons.get("non_private_dorm_candidate_households", pd.Series(False, index=persons.index))
    return coerce_boolean_series(values, name="non_private_dorm_candidate_households")


def home_only_mask(persons):
    values = persons.get("non_private_home_only_households", pd.Series(False, index=persons.index))
    return coerce_boolean_series(values, name="non_private_home_only_households")


def execute_without_home_only(context, input_stage, model_execute, defaults):
    """Keep home-only people outside predictions AND the model's diagnostics.

    A narrow context proxy changes only the population input; training and all
    other dependencies keep the stage's existing configure/execute structure.
    Restore the held-out rows afterward with explicit unmodelled placeholders.
    These placeholders describe the home-only simulation scenario, not measured
    income, licences or vehicle ownership, and never enter that model's validation.
    """
    population = context.stage(input_stage)
    held = home_only_mask(population)
    if not held.any():
        return model_execute(context)
    active, excluded = population.loc[~held].copy(), population.loc[held].copy()
    class PopulationContext:
        def stage(self, name, *args, **kwargs):
            return active if name == input_stage else context.stage(name, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(context, name)
    logging.getLogger("synpp").info("Excluded %d scenario-held-out residents from %s prediction/validation", len(excluded), input_stage)
    result = model_execute(PopulationContext()) if len(active) else active
    for column, value in defaults.items():
        excluded[column] = value
        if column not in result:
            result[column] = pd.Series(index=result.index, dtype=excluded[column].dtype)
    # Preserve residence flags even in models with an explicit output column list.
    flags = [c for c in population if c.startswith("non_private_") or c == "is_synthetic_household"]
    missing_flags = [c for c in flags if c not in result]
    if missing_flags:
        result = result.merge(population[["person_id"] + missing_flags], on="person_id", validate="one_to_one")
    excluded["non_private_model_status"] = "home_only_not_modelled"
    result["non_private_model_status"] = "ordinary_models"
    combined = pd.concat([result, excluded.reindex(columns=result.columns)], ignore_index=True)
    # Map IDs to their original row positions without building Python sets of
    # millions of residents for every model in a full-population run.
    order = pd.Series(np.arange(len(population)), index=population.person_id)
    positions = combined.person_id.map(order)
    if len(combined) != len(population) or combined.person_id.duplicated().any() or positions.isna().any():
        raise ValueError("Home-only model bypass changed the population identity set")
    return combined.iloc[positions.argsort()].reset_index(drop=True)
