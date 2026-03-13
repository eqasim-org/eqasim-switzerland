import numpy as np
import pandas as pd
from itertools import chain, zip_longest
import data.spatial.utils as spatial_utils
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.statent.statent")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.zone_shapes")
    context.stage("synthesis.population.models.employment")
    context.stage("data.od.matrix")
    context.stage("data.od.distances")
    context.stage("synthesis.population.spatial.primary.work.work_remotly")
    if context.config("include_cross_border"):
        context.stage("data.cross_border.destinations")

    context.config("random_seed")
    context.config("input_downsampling")


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

def execute(context):
    logger.info("\t Assigning work locations to agents based on OD matrices and company data...")
    # Number of real persons represented by one simulated agent
    persons_per_agent = 1 / context.config("input_downsampling")

    # Persons with home location and zone
    persons = context.stage("synthesis.population.models.employment")[["person_id", "household_id", "home_zone_id", "home_x", "home_y", "employed"]]

    # Removed unemployed agents
    persons = persons[persons["employed"]==1].reset_index(drop=True)

    # Remove those working remotely (they will be assigned a work location later, which is their household_id)
    remote_working = context.stage("synthesis.population.spatial.primary.work.work_remotly")
    remote_working = remote_working[remote_working["work_remotly"]==1]
    work_remotly = persons["person_id"].isin(remote_working.person_id.unique())    

    df = persons[~work_remotly].reset_index(drop=True)

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
        logger.info("\t Adjusting company employee counts to account for cross-border commuters...")
        destinations_cb         = context.stage("data.cross_border.destinations")
        destinations_cb_commute = destinations_cb[destinations_cb["trip_purpose"]=="work"]

        nb_empl_cb = destinations_cb_commute.groupby("destination_id")["cross_border_person_id"].count().reset_index()
        nb_empl_cb.columns = ["enterprise_id", "nb_employees_crossborder"]
        nb_empl_cb["enterprise_id"] = nb_empl_cb["enterprise_id"].astype(int)

        df_statent = df_statent.merge(nb_empl_cb, on = "enterprise_id", how="left")
        df_statent["nb_employees_crossborder"] = df_statent["nb_employees_crossborder"].fillna(0).astype(int)

        df_statent["number_employees"] = (df_statent["number_employees"] - df_statent["nb_employees_crossborder"]*persons_per_agent).clip(lower = 0)

        del df_statent["nb_employees_crossborder"], destinations_cb, destinations_cb_commute, nb_empl_cb

    # Convert to arrays for speed
    logger.info("\t Converting data to numpy arrays for efficient processing...")
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
    comp_zone_ids = comp_zone_ids[valid_comp_mask]    
    comp_min_capacity = comp_emp * 1e-3

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

    # starting assignement
    no_comp = set()
    with context.progress(total=n, label="Assigning work locations (OD+distance)") as prog:    
        for idx in range(n):
            # get origin zone index
            origin_zone = p_home_zone[idx]
            origin_idx = zone_index.get(origin_zone, None)
            
            # Select zones
            candidate_zones = [z for z, c in num_destination_zones_per_zone[origin_zone].items() if c > 0]
            candidate_zones_idx = [zone_index[z] for z in candidate_zones]
                    
            # Gather candidate companies in those zones
            cand_lists = [zone_to_company_idx[zi] for zi in candidate_zones_idx]            
            cand_idx = np.concatenate(cand_lists)
            
            if len(cand_idx)==0:
                # If no company is found, consider only the residence zone (if it has companies)                
                cand_lists = zone_to_company_idx[zone_index[origin_zone]]  
                if len(cand_idx)==0:
                    cand_idx = np.arange(len(comp_emp))
                    no_comp.update(candidate_zones_idx)
            
            ## Company weights
            weights = comp_emp[cand_idx]

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
            comp_emp[sel] = max(comp_emp[sel] - persons_per_agent, comp_min_capacity[sel])  # reduce available capacity, but keep non-zero to avoid issues    

            prog.update()                

    if len(no_comp):
        logger.warning(f"There are {len(no_comp)} zones without companies. These are tese zones:")
        logger.warning(no_comp)

    # Build result frame
    out = df[["person_id","home_x", "home_y"]].copy()
    out["x"] = work_x
    out["y"] = work_y
    out["destination_id"] = work_loc_id    
    out["work_remotly"] = False

    # concate agents working remotely (their work location is their household_id)
    remote_agents = persons[work_remotly][["person_id", "household_id", "home_x", "home_y"]].copy()
    remote_agents = remote_agents.rename(columns={"household_id": "destination_id"})
    remote_agents["x"] = remote_agents["home_x"]
    remote_agents["y"] = remote_agents["home_y"]
    remote_agents["work_remotly"] = True
    
    out = pd.concat([out, remote_agents], ignore_index=True)

    # compute commute distance
    out["commute_distance"] = np.sqrt((out["home_x"] - out["x"])**2 + (out["home_y"] - out["y"])**2)

    # Ensure no missing coordinates
    assert np.isfinite(out["x"]).all() and np.isfinite(out["y"]).all()

    out = spatial_utils.to_gpd(context, out, coord_type="work")
    return out[["person_id", "destination_id", "work_remotly", "commute_distance", "geometry"]] 


