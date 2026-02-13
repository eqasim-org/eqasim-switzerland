import numpy as np
import pandas as pd
from itertools import chain, zip_longest
import data.spatial.utils as spatial_utils

def configure(context):
    context.stage("data.statent.statent")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.zone_shapes")
    context.stage("synthesis.population.enriched")
    context.stage("data.microcensus.commute")
    context.stage("data.od.matrix")
    context.stage("data.od.distances")

    if context.config("include_cross_border"):
        context.stage("data.cross_border.destinations")

    context.config("random_seed")
    context.config("input_downsampling")

# Algorithm:
# 1. get the home location
# 2. from the od matrices, get the probability that the agent work in each zone, based on their residence zone and mode.
# 3. sample destintion zones
# 3. give to each company a weight, based on the number of employees.
# 4. estimate a probability based on the companies' distance to agent's home and the agent's commute distance
# 5. get the joint probability by multipling the weight and the estimated probability
# 6. sample one company based on that probability for each agent (sampling is done without replacement, each time a company
# is selected, its weight decreases)
# 7. special case (heuristics) for very short commute distances

STANDARD_DEVIATION_DISTANCE = 150  # meters (radius of 450 m for 99.7% of distribution)

def get_distance_weight(dx):
    # Gaussian around 0 with adaptive std for numerical stability
    std = max(STANDARD_DEVIATION_DISTANCE, float(np.min(dx)) / 2.0 if dx.size else STANDARD_DEVIATION_DISTANCE)
    coef = 1.0 / (std * np.sqrt(2.0 * np.pi))
    return coef * np.exp(-0.5 * (dx / std) ** 2)


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


def sort_group(group):
    # Sort distances ascending
    return group.sort_values('commute_home_distance').reset_index(drop=True)


def execute(context):
    # Number of real persons represented by one simulated agent
    persons_per_agent = 1 / context.config("input_downsampling")

    # Persons with home location and zone
    persons = context.stage("synthesis.population.enriched")[["person_id", "mz_person_id", "home_zone_id", "home_x", "home_y"]]

    # Microcensus commute (work)
    commute = context.stage("data.microcensus.commute")["work"][["person_id", "commute_mode", "commute_home_distance"]]
    commute = commute.rename(columns={"person_id": "mz_person_id"})
    
    # Merge commute info
    df = pd.merge(persons, commute, on="mz_person_id", how="inner")
    df = df.groupby('home_zone_id', group_keys=False).apply(sort_group).reset_index(drop=True)

    # Zones and index mapping
    df_zones = context.stage("data.spatial.zones").copy()
    zone_ids = df_zones["zone_id"].values
    zone_index = {z: i for i, z in enumerate(zone_ids)}

    # OD matrices and distances
    pdf_matrices, _ = context.stage("data.od.matrix")

    # STATENT enterprises
    df_statent = context.stage("data.statent.statent")[["enterprise_id", "x", "y", "zone_id", "zone_municipality_id", "number_employees"]].copy()
    df_statent = df_statent.dropna(subset=["x", "y", "number_employees", "zone_id"])

    if context.config("include_cross_border"):
        destinations_cb         = context.stage("data.cross_border.destinations")
        destinations_cb_commute = destinations_cb[destinations_cb["trip_purpose"]=="work"]

        nb_empl_cb = destinations_cb_commute.groupby("destination_id")["cross_border_person_id"].count().reset_index()
        nb_empl_cb.columns = ["enterprise_id", "nb_employees_crossborder"]
        nb_empl_cb["enterprise_id"] = nb_empl_cb["enterprise_id"].astype(int)

        df_statent = df_statent.merge(nb_empl_cb, on = "enterprise_id", how="left")
        df_statent["nb_employees_crossborder"] = df_statent["nb_employees_crossborder"].fillna(0).astype(int)

        df_statent["number_employees"] = (df_statent["number_employees"] - df_statent["nb_employees_crossborder"]).clip(lower = 0)

        del df_statent["nb_employees_crossborder"]

    # Convert to arrays for speed
    comp_x = df_statent["x"].to_numpy(dtype=float)
    comp_y = df_statent["y"].to_numpy(dtype=float)
    comp_emp = df_statent["number_employees"].to_numpy(dtype=float)
    comp_eid = df_statent["enterprise_id"].to_numpy()
    comp_zone_ids = df_statent["zone_id"].to_numpy()

    # Map company zone_id -> matrix index; -1 if unknown (filtered out)
    comp_zone_idx = np.array([zone_index.get(z, -1) for z in comp_zone_ids], dtype=int)
    valid_comp_mask = comp_zone_idx >= 0
    comp_x = comp_x[valid_comp_mask]
    comp_y = comp_y[valid_comp_mask]
    comp_emp = comp_emp[valid_comp_mask]
    comp_eid = comp_eid[valid_comp_mask]
    comp_zone_idx = comp_zone_idx[valid_comp_mask]

    # RNG
    rng = np.random.RandomState(context.config("random_seed"))

    # Helper to get source mode
    def normalize_mode(mode):
        return "car" if mode == "car_passenger" else mode
    
    # Build zone -> company index list
    zone_to_company_idx = [[] for _ in range(len(zone_ids))]
    for i, zi in enumerate(comp_zone_idx):
        zone_to_company_idx[zi].append(i)
    zone_to_company_idx = [np.array(lst, dtype=int) if len(lst) else np.array([], dtype=int) for lst in zone_to_company_idx]

    # Build zone -> destination zones list from OD matrices
    commuters_per_zone = df.groupby("home_zone_id").size().to_dict()
    modes_counts_per_zone = df.groupby(["home_zone_id", "commute_mode"]).size().unstack(fill_value=0).T.to_dict(orient="dict")
    num_destination_zones_per_zone = {}
    for origin_idx, origin_zone in enumerate(zone_ids):
        n_commuters = commuters_per_zone.get(origin_zone, 0)
        if n_commuters == 0:
            num_destination_zones_per_zone[origin_zone] = 0
            continue
        modes_counts = modes_counts_per_zone.get(origin_zone, {})
        # We use this to respect the mode split in determining possible destination zones
        zone_probs = np.sum([mode_count * pdf_matrices[normalize_mode(mode)][origin_idx, :]  for mode, mode_count in modes_counts.items()],
                        axis=0)
        destination_zones = multinomial_sample(n_commuters, zone_probs / zone_probs.sum())
        num_destination_zones_per_zone[origin_zone] = {zone_ids[i]: count for i, count in enumerate(destination_zones) if count > 0}
            
    # now we prepare array in order to use numPy for work assignment, it is faster than pandas dataframe manipulation
    # Prepare output arrays
    n = len(df)
    work_x = np.full(n, np.nan, dtype=float)
    work_y = np.full(n, np.nan, dtype=float)
    work_loc_id = np.full(n, np.nan, dtype=object)

    # Pre-fetch person arrays
    p_home_zone = df["home_zone_id"].to_numpy()
    p_home_x = df["home_x"].to_numpy(dtype=float)
    p_home_y = df["home_y"].to_numpy(dtype=float)
    # p_mode = df["commute_mode"].to_numpy()
    p_target_dist = df["commute_home_distance"].to_numpy(dtype=float)

    # starting assignement
    no_comp = []
    include_origin_zone= False
    DISTANCE_LIMIT = 1000 #meters
    with context.progress(total=n, label="Assigning work locations (OD+distance)") as prog:
        for idx in range(n):
            # get origin zone index
            origin_zone = p_home_zone[idx]
            origin_idx = zone_index.get(origin_zone, None)

            # get tarket distance
            target_distance = p_target_dist[idx]
            
            # Select top zones to reach cumulative threshold
            candidate_zones = [z for z, c in num_destination_zones_per_zone[origin_zone].items() if c > 0]
            candidate_zone_idx = [zone_index[z] for z in candidate_zones]

            if target_distance<DISTANCE_LIMIT:
                if origin_zone not in candidate_zones:
                    candidate_zone_idx.append(origin_idx)
                    include_origin_zone = True
                    
            # Gather candidate companies in those zones
            cand_lists = [zone_to_company_idx[zi] for zi in candidate_zone_idx]            
            cand_idx = np.concatenate(cand_lists)
            
            if len(cand_idx)==0:
                # If no company is found, consider all companies, then let the distance decides
                cand_idx = np.arange(len(comp_emp))
                no_comp.append(candidate_zone_idx)
            
            # Compute weights
            dx = comp_x[cand_idx] - p_home_x[idx]
            dy = comp_y[cand_idx] - p_home_y[idx]
            d = np.hypot(dx, dy)

            ## Compute distance weights
            dist_weight = get_distance_weight(np.abs(d - target_distance))

            ## Company weights
            emp = comp_emp[cand_idx]

            ## Final probabilities
            weights = dist_weight * emp

            if include_origin_zone:
                weights[comp_zone_idx[cand_idx]==origin_idx] /= 60 # because are not suppose to be here, just included them in case the agent doesn't find any other place

            sumw = weights.sum()
            weights = weights / sumw
            sel_local = rng.choice(len(cand_idx), p=weights)
            sel = cand_idx[sel_local]
            work_x[idx] = comp_x[sel]
            work_y[idx] = comp_y[sel]
            work_loc_id[idx] = comp_eid[sel]
            
            # Correct distributions after sampling to respect mode/zone splits
            dest_zone = comp_zone_ids[sel]
            if dest_zone in num_destination_zones_per_zone[origin_zone]:
                num_destination_zones_per_zone[origin_zone][dest_zone] -= 1
            comp_emp[sel] = max(comp_emp[sel] - persons_per_agent, 0.1)  # reduce available capacity, but keep non-zero to avoid issues

            include_origin_zone = False
            prog.update()                

    # Build result frame
    out = df[["person_id"]].copy()
    out["x"] = work_x
    out["y"] = work_y
    out["destination_id"] = work_loc_id

    # Ensure no missing coordinates
    assert np.isfinite(out["x"]).all() and np.isfinite(out["y"]).all()

    out = spatial_utils.to_gpd(context, out, coord_type="work")
    return out[["person_id", "destination_id", "geometry"]]


