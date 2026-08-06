import numpy as np
import logging
from .wla_tools import multinomial_sample, calculate_company_weights, correct_companies_number_of_employees, COMP_SATURATION_FACTOR
from data.od.matrix import (AGE_BIN_EDGES, DEFAULT_SEGMENT_KEY)
from .work_locations import get_segment_key

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.constants")
    context.stage("data.statent.statent")
    context.stage("data.spatial.zones")    
    context.stage("data.structural_survey.structural_survey")
    context.stage("synthesis.population.sampled")
    context.stage("data.od.matrix_moving")    
        
    context.stage("synthesis.population.spatial.primary.work.work_at_different_locations")
    context.stage("synthesis.population.spatial.primary.work.fixed_work_locations")
    context.stage("synthesis.population.destinations")    

    if context.config("include_cross_border"):
        context.stage("data.cross_border.destinations")

    context.config("include_external_population", default = False)
    if context.config("include_external_population"):
        context.stage("data.external_population.commutes")

    context.config("random_seed")
    context.config("input_downsampling")


def execute(context):
    logger.info("\t Assigning work locations to agents based on OD matrices and company data...")
    c = context.stage("data.constants")

    # Number of real persons represented by one simulated agent
    persons_per_agent = 1 / context.config("input_downsampling")
    capacity_decrement = persons_per_agent * COMP_SATURATION_FACTOR

    # Persons with home location and zone
    persons = context.stage("synthesis.population.sampled")[[
        "person_id", "household_id", "home_zone_id", "home_x", "home_y",
        "employed", "car_availability", "sex", "age"
    ]].copy()

    # Removed unemployed agents
    persons = persons[persons["employed"] == c.EMPLOYED].reset_index(drop=True)

    # Keep only those working from different places
    work_different_location = context.stage("synthesis.population.spatial.primary.work.work_at_different_locations")["person_id"].unique()    
    df = persons[persons["person_id"].isin(work_different_location)].reset_index(drop=True)

    # Zones and index mapping
    df_zones = context.stage("data.spatial.zones").copy()
    zone_ids = df_zones["zone_id"].values
    zone_index = {z: i for i, z in enumerate(zone_ids)}

    # OD matrices and distances
    pdf_matrices, _ = context.stage("data.od.matrix_moving")
    default_pdf_matrix = pdf_matrices.get(DEFAULT_SEGMENT_KEY, next(iter(pdf_matrices.values())))

    # STATENT enterprises (get OVGK from destinations)
    df_statent  = context.stage("data.statent.statent")[["enterprise_id", "x", "y", "zone_id", "zone_municipality_id", "number_employees"]].copy()
    destinations = context.stage("synthesis.population.destinations")[["destination_id","ovgk"]]
    df_statent  = df_statent.merge(destinations, left_on="enterprise_id", right_on="destination_id", how="left")

    df_statent = df_statent.dropna(subset=["x", "y", "number_employees", "zone_id"])
    
    df_fixed_locations = context.stage("synthesis.population.spatial.primary.work.fixed_work_locations")[["person_id", "destination_id"]]
    df_statent = correct_companies_number_of_employees(context, df_statent, df_fixed_locations)

    # Convert to arrays for speed
    logger.info("\t Converting data to numpy arrays for efficient processing...")
    comp_x = df_statent["x"].to_numpy(dtype=float)
    comp_y = df_statent["y"].to_numpy(dtype=float)
    comp_emp = df_statent["number_employees"].to_numpy(dtype=float)
    comp_eid = df_statent["enterprise_id"].to_numpy()
    comp_zone_ids = df_statent["zone_id"].to_numpy()    
    comp_pt1 = df_statent["ovgk"].isin(["A","B"]).to_numpy()
    comp_pt2 = df_statent["ovgk"].isin(["C","D"]).to_numpy()

    # Map company zone_id -> matrix index; -1 if unknown (filtered out)
    comp_zone_idx = np.array([zone_index.get(z, -1) for z in comp_zone_ids], dtype=int)
    valid_comp_mask = comp_zone_idx >= 0
    comp_x = comp_x[valid_comp_mask]
    comp_y = comp_y[valid_comp_mask]
    comp_emp = comp_emp[valid_comp_mask]
    comp_eid = comp_eid[valid_comp_mask]
    comp_zone_idx = comp_zone_idx[valid_comp_mask]
    comp_zone_ids = comp_zone_ids[valid_comp_mask]
    comp_pt1 = comp_pt1[valid_comp_mask]
    comp_pt2 = comp_pt2[valid_comp_mask]
    comp_min_capacity = comp_emp * 1e-6

    # RNG
    rng = np.random.RandomState(context.config("random_seed"))

    # Build zone -> company index list
    zone_to_company_idx = [[] for _ in range(len(zone_ids))]
    for i, zi in enumerate(comp_zone_idx):
        zone_to_company_idx[zi].append(i)
    zone_to_company_idx = [np.array(lst, dtype=int) if len(lst) else np.array([], dtype=int) for lst in zone_to_company_idx]

    # now we prepare array in order to use numPy for work assignment, it is faster than pandas dataframe manipulation
    # we sort df in order to assign work location to those who don't have a car, then to thos who have a car
    df = df.sort_values(by="car_availability").reset_index(drop=True)

    # Build destination-zone budgets per (sex, age_bin, origin_zone).
    p_home_zone = df["home_zone_id"].to_numpy()
    p_sex = df["sex"].to_numpy()
    p_age = df["age"].to_numpy(dtype=float)
    person_segment_keys = [get_segment_key(sex, age) for sex, age in zip(p_sex, p_age)]

    commuters_per_segment_zone = {}
    for segment_key, origin_zone in zip(person_segment_keys, p_home_zone):
        key = (segment_key, origin_zone)
        commuters_per_segment_zone[key] = commuters_per_segment_zone.get(key, 0) + 1

    num_destination_zones_per_segment = {}
    for segment_key in set(person_segment_keys):
        matrix = pdf_matrices.get(segment_key, default_pdf_matrix)
        num_destination_zones_per_segment[segment_key] = {}

        for origin_idx, origin_zone in enumerate(zone_ids):
            n_commuters = commuters_per_segment_zone.get((segment_key, origin_zone), 0)
            if n_commuters <= 0:
                num_destination_zones_per_segment[segment_key][origin_zone] = {}
                continue

            zone_probs = matrix[origin_idx, :]
            prob_sum = np.nansum(zone_probs)

            if (not np.isfinite(prob_sum)) or (prob_sum <= 0.0):
                zone_probs = default_pdf_matrix[origin_idx, :]
                prob_sum = np.nansum(zone_probs)

            if (not np.isfinite(prob_sum)) or (prob_sum <= 0.0):
                num_destination_zones_per_segment[segment_key][origin_zone] = {}
                continue

            destination_zones = multinomial_sample(n_commuters, zone_probs / prob_sum)
            num_destination_zones_per_segment[segment_key][origin_zone] = {
                zone_ids[i]: count for i, count in enumerate(destination_zones) if count > 0
            }
    
    # Prepare output arrays
    n = len(df)
    work_x = np.full(n, np.nan, dtype=float)
    work_y = np.full(n, np.nan, dtype=float)
    work_loc_id = np.full(n, np.nan, dtype=object)

    # Pre-fetch person arrays
    p_has_car = df["car_availability"].to_numpy(dtype=bool)

    # starting assignement
    no_comp = set()
    with context.progress(total=n, label="Assigning moving work locations (OD+distance)") as prog:    
        for idx in range(n):
            # get origin zone index
            origin_zone = p_home_zone[idx]
            origin_idx = zone_index.get(origin_zone, None)

            segment_key = person_segment_keys[idx]
            segment_counts = num_destination_zones_per_segment.get(segment_key)
            if segment_counts is None:
                segment_counts = num_destination_zones_per_segment.get(DEFAULT_SEGMENT_KEY, {})
            destination_counts = segment_counts.get(origin_zone, {})
            
            # Select zones
            candidate_zones = [z for z, c in destination_counts.items() if c > 0]
            candidate_zones_idx = [zone_index[z] for z in candidate_zones]
                    
            # Gather candidate companies in those zones
            cand_lists = [zone_to_company_idx[zi] for zi in candidate_zones_idx if len(zone_to_company_idx[zi])]
            cand_idx = np.concatenate(cand_lists) if len(cand_lists) else np.array([], dtype=int)
            
            if len(cand_idx)==0:
                no_comp.update(candidate_zones_idx)
                if origin_idx is not None:
                    candidate_zones_idx = [j for j in range(origin_idx-3, origin_idx+4) if 0 <= j < len(zone_ids)]
                    cand_lists = [zone_to_company_idx[zi] for zi in candidate_zones_idx if len(zone_to_company_idx[zi])]
                    cand_idx = np.concatenate(cand_lists) if len(cand_lists) else np.array([], dtype=int)
                if len(cand_idx)==0:
                    cand_idx = np.arange(len(comp_emp))                

            ## Company weights
            weights = calculate_company_weights(cand_idx, p_has_car[idx], comp_emp, comp_pt1, comp_pt2)

            sel_local = rng.choice(len(cand_idx), p=weights)
            sel = cand_idx[sel_local]
            work_x[idx] = comp_x[sel]
            work_y[idx] = comp_y[sel]
            work_loc_id[idx] = comp_eid[sel]
            
            # Correct distributions after sampling to respect mode/zone splits
            dest_zone = comp_zone_ids[sel]
            if dest_zone in destination_counts:
                destination_counts[dest_zone] -= 1
            comp_emp[sel] = max(comp_emp[sel] - capacity_decrement, comp_min_capacity[sel])  # reduce available capacity, but keep non-zero to avoid issues

            prog.update()                

    if len(no_comp):
        logger.warning(f"There are {len(no_comp)} zones without companies. These are these zones:")
        logger.warning(no_comp)

    # Build result frame
    out = df[["person_id","home_x", "home_y"]].copy()
    out["x"] = work_x
    out["y"] = work_y
    out["destination_id"] = work_loc_id        

    # compute commute distance
    out["commute_distance"] = np.sqrt((out["home_x"] - out["x"])**2 + (out["home_y"] - out["y"])**2)

    # Ensure no missing coordinates
    assert np.isfinite(out["x"]).all() and np.isfinite(out["y"]).all()

    return out[["person_id", "destination_id", "commute_distance", "x", "y"]] 





