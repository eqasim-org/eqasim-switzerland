import numpy as np
import shapely
import logging
from scipy.spatial import cKDTree

logger = logging.getLogger("synpp")


PT1_PREFERENCE_FACTOR   = 5.0
PT2_PREFERENCE_FACTOR   = 3.0
OTHER_PREFERENCE_FACTOR = 0.5
COMP_SATURATION_FACTOR  = 1.4


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
        _log_capacity_change("cross-border commuters (%d sampled records)" % nb_empl_cb["nb_employees_crossborder"].sum(),
                             total_before, df_statent["number_employees"].sum())

        del df_statent["nb_employees_crossborder"], destinations_cb, destinations_cb_commute, nb_empl_cb


    if context.config("include_external_population"):
        logger.info("\t Adjusting company employee counts to account for French commuters...")
        commutes = context.stage("data.external_population.commutes")

        total_before = df_statent["number_employees"].sum()
        border = context.stage("data.spatial.swiss_border").to_crs(2056).union_all()
        xy = commutes[["destination_x", "destination_y"]].to_numpy(dtype=float)
        inside = shapely.intersects_xy(border, xy[:, 0], xy[:, 1])
        rate = context.config("fr_sample_rate")
        if not 0 < rate <= 1:
            raise ValueError("fr_sample_rate must be in (0, 1]")
        jobs = df_statent["number_employees"].to_numpy()
        sites, groups = np.unique(df_statent[["x", "y"]], axis=0, return_inverse=True)
        groups = groups.ravel()
        _, nearest = cKDTree(sites).query(xy[inside])
        reservations = np.bincount(nearest, minlength=len(sites)) * COMP_SATURATION_FACTOR / rate
        site_jobs = np.bincount(groups, weights=jobs)
        fraction = np.divide(reservations, site_jobs, out=np.zeros_like(site_jobs), where=site_jobs > 0)
        df_statent = df_statent.assign(number_employees=jobs * np.maximum(1 - fraction[groups], 0.01))
        _log_capacity_change(f"French commuters ({inside.sum()} Swiss endpoints at source rate {rate})",
                             total_before, df_statent["number_employees"].sum())

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

