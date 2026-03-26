import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("synpp")

PT1_PREFERENCE_FACTOR = 5.0
PT2_PREFERENCE_FACTOR = 3.0
OTHER_PREFERENCE_FACTOR = 0.5
COMP_SATURATION_FACTOR = 1.4

def multinomial_sample(n, probs):
    """
    Deterministic version of multinomial sampling.

    Parameters
    ----------
    n : int
        Total number of items to allocate.
    probs : array-like
        Probabilities for each category. Does not need to sum to 1.

    Returns
    -------
    counts : np.ndarray
        Integer counts per category, summing exactly to n.
    """
    probs = np.array(probs, dtype=float)
    probs /= probs.sum()  # normalize just in case
    
    # Step 1: initial deterministic allocation
    raw_counts = n * probs
    counts = np.floor(raw_counts).astype(int)
    
    # Step 2: distribute leftover counts based on largest fractional parts
    remainder = n - counts.sum()
    if remainder > 0:
        fractional = raw_counts - counts
        # pick indices with largest fractional remainder
        top_indices = np.argsort(fractional)[-remainder:]
        counts[top_indices] += 1

    return counts

def _apply_distance_reweighting(weights, distances):    
    mean_d = distances.mean()
    min_d = distances.min()
    max_d = distances.max()
    normalized_distance = (distances - min_d) / (max_d - min_d + 1e-6)
    
    if mean_d > 40000:        
        distance_factor = 0.9 * normalized_distance
        return weights * (1 - distance_factor)
    elif mean_d < 10000:
        distance_factor = 0.3 + 0.7 * normalized_distance
        return weights * distance_factor

    return weights


def calculate_company_weights(cand_idx, has_car, comp_emp, comp_pt1, comp_pt2, distances=None):
    weights = comp_emp[cand_idx].astype(float, copy=True)

    if not has_car:
        preference_factors = np.full(len(cand_idx), OTHER_PREFERENCE_FACTOR, dtype=float)
        preference_factors[comp_pt2[cand_idx]] = PT2_PREFERENCE_FACTOR
        preference_factors[comp_pt1[cand_idx]] = PT1_PREFERENCE_FACTOR
        weights *= preference_factors

    if distances is not None:
        weights = _apply_distance_reweighting(weights, distances)

    total = weights.sum()
    if total <= 0.0:
        return np.ones(len(cand_idx), dtype=float) / len(cand_idx)

    return weights / total




def correct_companies_number_of_employees(context, df_statent, df_fixed_locations=None):
    persons_per_agent = 1 / context.config("input_downsampling")
    capacity_decrement = persons_per_agent * COMP_SATURATION_FACTOR

    if context.config("include_cross_border"):
        logger.info("\t Adjusting company employee counts to account for cross-border commuters...")
        destinations_cb         = context.stage("data.cross_border.destinations")
        destinations_cb_commute = destinations_cb[destinations_cb["trip_purpose"]=="work"]

        nb_empl_cb = destinations_cb_commute.groupby("destination_id")["cross_border_person_id"].count().reset_index()
        nb_empl_cb.columns = ["enterprise_id", "nb_employees_crossborder"]
        nb_empl_cb["enterprise_id"] = nb_empl_cb["enterprise_id"].astype(int)

        df_statent = df_statent.merge(nb_empl_cb, on = "enterprise_id", how="left")
        df_statent["nb_employees_crossborder"] = df_statent["nb_employees_crossborder"].fillna(0).astype(int)

        df_statent["number_employees"] = np.maximum( (df_statent["number_employees"] - df_statent["nb_employees_crossborder"] * capacity_decrement),
                                                      df_statent["number_employees"] * 0.01) 

        del df_statent["nb_employees_crossborder"], destinations_cb, destinations_cb_commute, nb_empl_cb


    if df_fixed_locations is not None:
        logger.info("\t Adjusting company employee counts to account for agents with fixed work locations...")
        nb_fixed_agents = df_fixed_locations.groupby("destination_id")["person_id"].count().reset_index()
        nb_fixed_agents.columns = ["enterprise_id", "nb_fixed_employees"]

        df_statent = df_statent.merge(nb_fixed_agents, on = "enterprise_id", how="left")
        df_statent["nb_fixed_employees"] = df_statent["nb_fixed_employees"].fillna(0).astype(int)

        df_statent["number_employees"] = np.maximum( (df_statent["number_employees"] - df_statent["nb_fixed_employees"] * capacity_decrement),
                                                      df_statent["number_employees"] * 0.01)

        del df_statent["nb_fixed_employees"], df_fixed_locations, nb_fixed_agents
    
    return df_statent