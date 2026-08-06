import numpy as np
import pandas as pd
import logging
from scipy.spatial import cKDTree

logger = logging.getLogger("synpp")


PT1_PREFERENCE_FACTOR   = 5.0
PT2_PREFERENCE_FACTOR   = 3.0
OTHER_PREFERENCE_FACTOR = 0.5
COMP_SATURATION_FACTOR  = 1.4


# @numba.njit(cache=True)
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
    probs      = np.array(probs, dtype=np.float64)
    total_prob = probs.sum()

    if total_prob <= 0.0:
        return np.zeros(len(probs), dtype=np.int64)
    
    probs /= total_prob  # normalize just in case
    
    # Step 1: initial deterministic allocation
    raw_counts = n * probs
    counts     = np.floor(raw_counts).astype(np.int64)
    
    # Step 2: distribute leftover counts based on largest fractional parts
    remainder = n - counts.sum()
    if remainder > 0:
        fractional = raw_counts - counts
        # pick indices with largest fractional remainder
        top_indices = np.argsort(fractional)[-remainder:]
        counts[top_indices] += 1

    return counts

# @numba.njit(cache=True)
def _apply_distance_reweighting(weights, distances):    
    mean_d = distances.mean()
    min_d  = distances.min()
    max_d  = distances.max()

    normalized_distance = (distances - min_d) / (max_d - min_d + 1e-6)
    
    if mean_d > 40000:        
        distance_factor = 0.9 * normalized_distance
        return weights * (1 - distance_factor)
    
    elif mean_d < 10000:
        distance_factor = 0.3 + 0.7 * normalized_distance
        return weights * distance_factor

    return weights


# @numba.njit(cache=True)
def _calculate_company_weights_no_distance(cand_idx, has_car, comp_emp, comp_pt1, comp_pt2):
    weights = comp_emp[cand_idx].astype(float, copy=True)

    if not has_car:
        preference_factors = np.full(len(cand_idx), OTHER_PREFERENCE_FACTOR, dtype=float)
        preference_factors[comp_pt2[cand_idx]] = PT2_PREFERENCE_FACTOR
        preference_factors[comp_pt1[cand_idx]] = PT1_PREFERENCE_FACTOR
        weights *= preference_factors

    total = weights.sum()
    if total <= 0.0:
        return np.ones(len(cand_idx), dtype=float) / len(cand_idx)

    return weights / total


# @numba.njit(cache=True)
def _calculate_company_weights_with_distance(cand_idx, has_car, comp_emp, comp_pt1, comp_pt2, distances):
    weights = comp_emp[cand_idx].astype(float, copy=True)

    if not has_car:
        preference_factors = np.full(len(cand_idx), OTHER_PREFERENCE_FACTOR, dtype=float)
        preference_factors[comp_pt2[cand_idx]] = PT2_PREFERENCE_FACTOR
        preference_factors[comp_pt1[cand_idx]] = PT1_PREFERENCE_FACTOR
        weights *= preference_factors

    weights = _apply_distance_reweighting(weights, distances)

    total = weights.sum()
    
    if total <= 0.0:
        return np.ones(len(cand_idx), dtype=float) / len(cand_idx)

    return weights / total


def calculate_company_weights(cand_idx, has_car, comp_emp, comp_pt1, comp_pt2, distances=None):
    if distances is None:
        return _calculate_company_weights_no_distance(cand_idx, has_car, comp_emp, comp_pt1, comp_pt2)
    return _calculate_company_weights_with_distance(cand_idx, has_car, comp_emp, comp_pt1, comp_pt2, distances)


def _log_capacity_change(label, before, after):
    delta = before - after
    share = 100 * delta / before if before > 0 else 0.0
    logger.info("\t\t %s: %.0f -> %.0f remaining employees (-%.0f, -%.2f%%)" % (label, before, after, delta, share))


def correct_companies_number_of_employees(context, df_statent, df_fixed_locations=None):
    persons_per_agent = 1 / context.config("input_downsampling")
    capacity_decrement = persons_per_agent * COMP_SATURATION_FACTOR

    total_before_all = df_statent["number_employees"].sum()

    if context.config("include_cross_border"):
        logger.info("\t Adjusting company employee counts to account for cross-border commuters...")
        destinations_cb         = context.stage("data.cross_border.destinations")
        destinations_cb_commute = destinations_cb[destinations_cb["trip_purpose"]=="work"]

        nb_empl_cb = destinations_cb_commute.groupby("destination_id")["cross_border_person_id"].count().reset_index()
        nb_empl_cb.columns = ["enterprise_id", "nb_employees_crossborder"]

        df_statent = df_statent.merge(nb_empl_cb, on = "enterprise_id", how="left")
        df_statent["nb_employees_crossborder"] = df_statent["nb_employees_crossborder"].fillna(0).astype(int)

        total_before = df_statent["number_employees"].sum()
        df_statent["number_employees"] = np.maximum( (df_statent["number_employees"] - df_statent["nb_employees_crossborder"] * capacity_decrement),
                                                      df_statent["number_employees"] * 0.01)
        _log_capacity_change("cross-border commuters (%d real persons)" % nb_empl_cb["nb_employees_crossborder"].sum(),
                             total_before, df_statent["number_employees"].sum())

        del df_statent["nb_employees_crossborder"], destinations_cb, destinations_cb_commute, nb_empl_cb


    if context.config("include_external_population"):
        logger.info("\t Adjusting company employee counts to account for French commuters...")
        commutes = context.stage("data.external_population.commutes")

        # Commutes only carry a raw destination point (no enterprise_id), so
        # snap each one to its nearest STATENT enterprise.
        tree = cKDTree(df_statent[["x", "y"]].to_numpy(dtype=float))
        _, nearest_idx = tree.query(commutes[["destination_x", "destination_y"]].to_numpy(dtype=float))

        nb_empl_fr = pd.Series(df_statent["enterprise_id"].to_numpy()[nearest_idx]).value_counts()
        nb_empl_fr = nb_empl_fr.rename_axis("enterprise_id").reset_index(name="nb_employees_french")

        df_statent = df_statent.merge(nb_empl_fr, on = "enterprise_id", how="left")
        df_statent["nb_employees_french"] = df_statent["nb_employees_french"].fillna(0).astype(int)

        total_before = df_statent["number_employees"].sum()
        df_statent["number_employees"] = np.maximum( (df_statent["number_employees"] - df_statent["nb_employees_french"] * capacity_decrement),
                                                      df_statent["number_employees"] * 0.01)
        _log_capacity_change("French commuters (%d real persons)" % nb_empl_fr["nb_employees_french"].sum(),
                             total_before, df_statent["number_employees"].sum())

        del df_statent["nb_employees_french"], commutes, nb_empl_fr


    if df_fixed_locations is not None:
        logger.info("\t Adjusting company employee counts to account for agents with fixed work locations...")
        nb_fixed_agents = df_fixed_locations.groupby("destination_id")["person_id"].count().reset_index()
        nb_fixed_agents.columns = ["enterprise_id", "nb_fixed_employees"]

        df_statent = df_statent.merge(nb_fixed_agents, on = "enterprise_id", how="left")
        df_statent["nb_fixed_employees"] = df_statent["nb_fixed_employees"].fillna(0).astype(int)

        total_before = df_statent["number_employees"].sum()
        df_statent["number_employees"] = np.maximum( (df_statent["number_employees"] - df_statent["nb_fixed_employees"] * capacity_decrement),
                                                      df_statent["number_employees"] * 0.01)
        _log_capacity_change("agents with fixed work locations (%d agents)" % nb_fixed_agents["nb_fixed_employees"].sum(),
                             total_before, df_statent["number_employees"].sum())

        del df_statent["nb_fixed_employees"], df_fixed_locations, nb_fixed_agents

    logger.info("\t Total remaining employees after all corrections: %.0f -> %.0f" % (total_before_all, df_statent["number_employees"].sum()))

    return df_statent

