import numpy as np
from numba import njit, prange

SECONDARY_ACTIVITIES = ["other", "shop", "leisure", "work_secondary", "home_secondary", "education_secondary"]


@njit(parallel=True, fastmath=True)
def build_coarse_numerical_batch_numba(
    hx,
    hy,
    wx,
    wy,
    has_work,
    ox,
    oy,
    ages,
    daily_longest_distance_from_home,
    daily_crowfly_total,
    crowfly_consumed_before_trip,
    trip_position_class,
    income_class,
    centroid_x,
    centroid_y,
    statent_per_h3,
    employees_per_h3,
    urban_core_per_h3,
    urban_per_h3,
    education_per_h3,
    shop_per_h3,
    leisure_per_h3,
    ovgk_share_a_per_h3,
    ovgk_share_b_per_h3,
    ovgk_share_c_per_h3,
    ovgk_share_d_per_h3,
    ovgk_share_none_per_h3,
    outside_fraction,
):
    n = hx.shape[0]
    num_h3 = centroid_x.shape[0]
    out = np.empty((n, num_h3, 22), dtype=np.float64)

    for i in prange(n):
        valid_work = has_work[i]
        age_i = ages[i]
        daily_longest_i = daily_longest_distance_from_home[i]
        daily_total_i = daily_crowfly_total[i]
        consumed_i = crowfly_consumed_before_trip[i]
        trip_position_i = trip_position_class[i]
        income_i = income_class[i]

        for j in range(num_h3):
            cx = centroid_x[j]
            cy = centroid_y[j]

            dx_home = cx - hx[i]
            dy_home = cy - hy[i]
            out[i, j, 0] = np.sqrt(dx_home * dx_home + dy_home * dy_home)

            if valid_work:
                dx_work = cx - wx[i]
                dy_work = cy - wy[i]
                out[i, j, 1] = np.sqrt(dx_work * dx_work + dy_work * dy_work)
            else:
                out[i, j, 1] = 0.0

            dx_last = cx - ox[i]
            dy_last = cy - oy[i]
            out[i, j, 2] = np.sqrt(dx_last * dx_last + dy_last * dy_last)
            out[i, j, 3] = age_i
            out[i, j, 4] = statent_per_h3[j]
            out[i, j, 5] = employees_per_h3[j]
            out[i, j, 6] = urban_core_per_h3[j]
            out[i, j, 7] = urban_per_h3[j]
            out[i, j, 8] = education_per_h3[j]
            out[i, j, 9] = shop_per_h3[j]
            out[i, j, 10] = leisure_per_h3[j]
            out[i, j, 11] = ovgk_share_a_per_h3[j]
            out[i, j, 12] = ovgk_share_b_per_h3[j]
            out[i, j, 13] = ovgk_share_c_per_h3[j]
            out[i, j, 14] = ovgk_share_d_per_h3[j]
            out[i, j, 15] = ovgk_share_none_per_h3[j]
            out[i, j, 16] = outside_fraction[j]
            out[i, j, 17] = daily_longest_i
            out[i, j, 18] = daily_total_i
            out[i, j, 19] = consumed_i
            out[i, j, 20] = trip_position_i
            out[i, j, 21] = income_i

    return out


def build_coarse_numerical_batch(
    start,
    end,
    home_x,
    home_y,
    work_x,
    work_y,
    has_work,
    origin_x,
    origin_y,
    age,
    daily_longest_distance_from_home,
    daily_crowfly_total,
    crowfly_consumed_before_trip,
    trip_position_class,
    income_class,
    centroid_x,
    centroid_y,
    statent_per_h3,
    employees_per_h3,
    urban_core_per_h3,
    urban_per_h3,
    education_per_h3,
    shop_per_h3,
    leisure_per_h3,
    ovgk_share_a_per_h3,
    ovgk_share_b_per_h3,
    ovgk_share_c_per_h3,
    ovgk_share_d_per_h3,
    ovgk_share_none_per_h3,
    outside_fraction,
):
    return build_coarse_numerical_batch_numba(
        home_x[start:end],
        home_y[start:end],
        work_x[start:end],
        work_y[start:end],
        has_work[start:end],
        origin_x[start:end],
        origin_y[start:end],
        age[start:end],
        daily_longest_distance_from_home[start:end],
        daily_crowfly_total[start:end],
        crowfly_consumed_before_trip[start:end],
        trip_position_class[start:end],
        income_class[start:end],
        centroid_x,
        centroid_y,
        statent_per_h3,
        employees_per_h3,
        urban_core_per_h3,
        urban_per_h3,
        education_per_h3,
        shop_per_h3,
        leisure_per_h3,
        ovgk_share_a_per_h3,
        ovgk_share_b_per_h3,
        ovgk_share_c_per_h3,
        ovgk_share_d_per_h3,
        ovgk_share_none_per_h3,
        outside_fraction,
    )


def build_coarse_scaled_feature_batch(
    start,
    end,
    home_x,
    home_y,
    work_x,
    work_y,
    has_work,
    origin_x,
    origin_y,
    age,
    daily_longest_distance_from_home,
    daily_crowfly_total,
    crowfly_consumed_before_trip,
    trip_position_class,
    income_class,
    centroid_x,
    centroid_y,
    statent_per_h3,
    employees_per_h3,
    urban_core_per_h3,
    urban_per_h3,
    education_per_h3,
    shop_per_h3,
    leisure_per_h3,
    ovgk_share_a_per_h3,
    ovgk_share_b_per_h3,
    ovgk_share_c_per_h3,
    ovgk_share_d_per_h3,
    ovgk_share_none_per_h3,
    outside_fraction,
    scaler,
    numerical_len,
    num_h3,
    features_len,
    sex,
    employed,
    car_availability,
    purpose_one_hot,
):
    numerical_batch = build_coarse_numerical_batch(
        start,
        end,
        home_x,
        home_y,
        work_x,
        work_y,
        has_work,
        origin_x,
        origin_y,
        age,
        daily_longest_distance_from_home,
        daily_crowfly_total,
        crowfly_consumed_before_trip,
        trip_position_class,
        income_class,
        centroid_x,
        centroid_y,
        statent_per_h3,
        employees_per_h3,
        urban_core_per_h3,
        urban_per_h3,
        education_per_h3,
        shop_per_h3,
        leisure_per_h3,
        ovgk_share_a_per_h3,
        ovgk_share_b_per_h3,
        ovgk_share_c_per_h3,
        ovgk_share_d_per_h3,
        ovgk_share_none_per_h3,
        outside_fraction,
    )

    scaled = scaler.transform(numerical_batch.reshape(-1, numerical_len)).reshape(end - start, num_h3, numerical_len)
    out = np.empty((end - start, num_h3, features_len), dtype=np.float32)
    out[:, :, :numerical_len] = scaled.astype(np.float32)
    out[:, :, numerical_len] = sex[start:end, None]
    out[:, :, numerical_len + 1] = employed[start:end, None]
    out[:, :, numerical_len + 2] = car_availability[start:end, None]
    out[:, :, numerical_len + 3:] = np.broadcast_to(
        purpose_one_hot[start:end, None, :],
        (end - start, num_h3, purpose_one_hot.shape[1]),
    )
    return start, end, out


@njit(parallel=True, fastmath=True)
def build_numerical_batch_numba(
    hx,
    hy,
    wx,
    wy,
    has_work,
    ox,
    oy,
    ages,
    crowfly,
    cand_x,
    cand_y,
    cand_statent,
    valid_mask,
):
    n, max_children = cand_x.shape
    out = np.zeros((n, max_children, 6), dtype=np.float64)

    for i in prange(n):
        valid_work = has_work[i]
        age_i = ages[i]
        crowfly_i = crowfly[i]

        for j in range(max_children):
            if not valid_mask[i, j]:
                continue

            cx = cand_x[i, j]
            cy = cand_y[i, j]

            dx_home = cx - hx[i]
            dy_home = cy - hy[i]
            out[i, j, 0] = np.sqrt(dx_home * dx_home + dy_home * dy_home)

            if valid_work:
                dx_work = cx - wx[i]
                dy_work = cy - wy[i]
                out[i, j, 1] = np.sqrt(dx_work * dx_work + dy_work * dy_work)
            else:
                out[i, j, 1] = 0.0

            dx_last = cx - ox[i]
            dy_last = cy - oy[i]
            out[i, j, 2] = np.sqrt(dx_last * dx_last + dy_last * dy_last)

            out[i, j, 3] = cand_statent[i, j]
            out[i, j, 4] = age_i
            out[i, j, 5] = crowfly_i

    return out


@njit(parallel=True, fastmath=True)
def build_hierarchical_numerical_batch_numba(
    hx,
    hy,
    wx,
    wy,
    has_work,
    ox,
    oy,
    ages,
    daily_longest_distance_from_home,
    daily_crowfly_total,
    crowfly_consumed_before_trip,
    trip_position_class,
    income_class,
    cand_x,
    cand_y,
    cand_statent,
    cand_employees,
    cand_urban_core,
    cand_urban,
    cand_education,
    cand_shop,
    cand_leisure,
    cand_ovgk_share_a,
    cand_ovgk_share_b,
    cand_ovgk_share_c,
    cand_ovgk_share_d,
    cand_ovgk_share_none,
    cand_outside_fraction,
    valid_mask,
):
    n, max_children = cand_x.shape
    out = np.zeros((n, max_children, 22), dtype=np.float64)

    for i in prange(n):
        valid_work = has_work[i]
        age_i = ages[i]
        daily_longest_i = daily_longest_distance_from_home[i]
        daily_total_i = daily_crowfly_total[i]
        consumed_i = crowfly_consumed_before_trip[i]
        trip_position_i = trip_position_class[i]
        income_i = income_class[i]

        for j in range(max_children):
            if not valid_mask[i, j]:
                continue

            cx = cand_x[i, j]
            cy = cand_y[i, j]

            dx_home = cx - hx[i]
            dy_home = cy - hy[i]
            out[i, j, 0] = np.sqrt(dx_home * dx_home + dy_home * dy_home)

            if valid_work:
                dx_work = cx - wx[i]
                dy_work = cy - wy[i]
                out[i, j, 1] = np.sqrt(dx_work * dx_work + dy_work * dy_work)
            else:
                out[i, j, 1] = 0.0

            dx_last = cx - ox[i]
            dy_last = cy - oy[i]
            out[i, j, 2] = np.sqrt(dx_last * dx_last + dy_last * dy_last)
            out[i, j, 3] = age_i
            out[i, j, 4] = cand_statent[i, j]
            out[i, j, 5] = cand_employees[i, j]
            out[i, j, 6] = cand_urban_core[i, j]
            out[i, j, 7] = cand_urban[i, j]
            out[i, j, 8] = cand_education[i, j]
            out[i, j, 9] = cand_shop[i, j]
            out[i, j, 10] = cand_leisure[i, j]
            out[i, j, 11] = cand_ovgk_share_a[i, j]
            out[i, j, 12] = cand_ovgk_share_b[i, j]
            out[i, j, 13] = cand_ovgk_share_c[i, j]
            out[i, j, 14] = cand_ovgk_share_d[i, j]
            out[i, j, 15] = cand_ovgk_share_none[i, j]
            out[i, j, 16] = cand_outside_fraction[i, j]
            out[i, j, 17] = daily_longest_i
            out[i, j, 18] = daily_total_i
            out[i, j, 19] = consumed_i
            out[i, j, 20] = trip_position_i
            out[i, j, 21] = income_i

    return out


def get_h3_stage_outputs(context):
    h3_stage = context.stage("synthesis.population.spatial.secondary.locations_v2.h3")
    if not isinstance(h3_stage, (tuple, list)) or len(h3_stage) < 2:
        raise RuntimeError("Unexpected output format from synthesis.population.spatial.secondary.locations_v2.h3")

    h3_data = h3_stage[0]
    h3_geo = h3_stage[1]
    h3_tree = h3_stage[2] if len(h3_stage) > 2 else None
    return h3_data, h3_geo, h3_tree


def build_tree_from_geo(h3_geo, level0_res=5, level1_res=7):
    import h3 as h3lib

    level0_df = h3_geo["level_0"]
    level1_df = h3_geo["level_1"]
    level2_df = h3_geo["level_2"]

    level0_cells = set(level0_df["h3_index"].tolist())
    tree = {l0: {} for l0 in level0_cells}

    for l1 in level1_df["h3_index"].tolist():
        try:
            l0 = h3lib.cell_to_parent(l1, level0_res)
        except Exception:
            continue
        tree.setdefault(l0, {})
        tree[l0].setdefault(l1, [])

    for l2 in level2_df["h3_index"].tolist():
        try:
            l1 = h3lib.cell_to_parent(l2, level1_res)
            l0 = h3lib.cell_to_parent(l2, level0_res)
        except Exception:
            continue
        tree.setdefault(l0, {})
        tree[l0].setdefault(l1, []).append(l2)

    for l0 in tree:
        for l1 in tree[l0]:
            tree[l0][l1] = sorted(set(tree[l0][l1]))

    return tree


def build_level1_children_by_level0(h3_tree, centroid_x_by_l1, centroid_y_by_l1):
    children = {
        l0: sorted([l1 for l1 in level1_dict.keys() if l1 in centroid_x_by_l1 and l1 in centroid_y_by_l1])
        for l0, level1_dict in h3_tree.items()
    }
    return {l0: vals for l0, vals in children.items() if len(vals) > 0}


def build_level1_candidate_attributes_by_level0(
    children_by_level0,
    centroid_x_by_l1,
    centroid_y_by_l1,
    statent_count,
    employees_count,
    urban_core_count,
    urban_count,
    education_count,
    shop_count,
    leisure_count,
    ovgk_share_a_by_l1,
    ovgk_share_b_by_l1,
    ovgk_share_c_by_l1,
    ovgk_share_d_by_l1,
    ovgk_share_none_by_l1,
    outside_fraction_by_l1,
):
    candidate_attributes = {}
    for l0, children in children_by_level0.items():
        candidate_attributes[l0] = {
            "children": list(children),
            "x": [float(centroid_x_by_l1[c]) for c in children],
            "y": [float(centroid_y_by_l1[c]) for c in children],
            "num_statent": [float(statent_count.get(c, 0.0)) for c in children],
            "employees": [float(employees_count.get(c, 0.0)) for c in children],
            "urban_core": [float(urban_core_count.get(c, 0.0)) for c in children],
            "urban": [float(urban_count.get(c, 0.0)) for c in children],
            "education": [float(education_count.get(c, 0.0)) for c in children],
            "shop": [float(shop_count.get(c, 0.0)) for c in children],
            "leisure": [float(leisure_count.get(c, 0.0)) for c in children],
            "ovgk_share_a": [float(ovgk_share_a_by_l1.get(c, 0.0)) for c in children],
            "ovgk_share_b": [float(ovgk_share_b_by_l1.get(c, 0.0)) for c in children],
            "ovgk_share_c": [float(ovgk_share_c_by_l1.get(c, 0.0)) for c in children],
            "ovgk_share_d": [float(ovgk_share_d_by_l1.get(c, 0.0)) for c in children],
            "ovgk_share_none": [float(ovgk_share_none_by_l1.get(c, 0.0)) for c in children],
            "outside_fraction": [float(outside_fraction_by_l1.get(c, 0.0)) for c in children],
        }
    return candidate_attributes


def build_level2_children_by_level1(h3_tree, centroid_x_by_l2, centroid_y_by_l2):
    children = {}
    for l0, level1_dict in h3_tree.items():
        for l1, level2_list in level1_dict.items():
            filtered = sorted([l2 for l2 in level2_list if l2 in centroid_x_by_l2 and l2 in centroid_y_by_l2])
            if len(filtered) > 0:
                children[(l0, l1)] = filtered
    return children


def build_level2_candidate_attributes_by_level1(
    children_by_level1,
    centroid_x_by_l2,
    centroid_y_by_l2,
    statent_count,
    employees_count,
    urban_core_count,
    urban_count,
    education_count,
    shop_count,
    leisure_count,
    ovgk_share_a_by_l2,
    ovgk_share_b_by_l2,
    ovgk_share_c_by_l2,
    ovgk_share_d_by_l2,
    ovgk_share_none_by_l2,
    outside_fraction_by_l2,
):
    candidate_attributes = {}
    for key, children in children_by_level1.items():
        candidate_attributes[key] = {
            "children": list(children),
            "x": [float(centroid_x_by_l2[c]) for c in children],
            "y": [float(centroid_y_by_l2[c]) for c in children],
            "num_statent": [float(statent_count.get(c, 0.0)) for c in children],
            "employees": [float(employees_count.get(c, 0.0)) for c in children],
            "urban_core": [float(urban_core_count.get(c, 0.0)) for c in children],
            "urban": [float(urban_count.get(c, 0.0)) for c in children],
            "education": [float(education_count.get(c, 0.0)) for c in children],
            "shop": [float(shop_count.get(c, 0.0)) for c in children],
            "leisure": [float(leisure_count.get(c, 0.0)) for c in children],
            "ovgk_share_a": [float(ovgk_share_a_by_l2.get(c, 0.0)) for c in children],
            "ovgk_share_b": [float(ovgk_share_b_by_l2.get(c, 0.0)) for c in children],
            "ovgk_share_c": [float(ovgk_share_c_by_l2.get(c, 0.0)) for c in children],
            "ovgk_share_d": [float(ovgk_share_d_by_l2.get(c, 0.0)) for c in children],
            "ovgk_share_none": [float(ovgk_share_none_by_l2.get(c, 0.0)) for c in children],
            "outside_fraction": [float(outside_fraction_by_l2.get(c, 0.0)) for c in children],
        }
    return candidate_attributes


def make_purpose_one_hot(purpose_series, categories):
    purpose_to_idx = {p: i for i, p in enumerate(categories)}
    purpose_idx = purpose_series.astype(str).map(purpose_to_idx).fillna(0).astype(np.int64).to_numpy()
    return np.eye(len(categories), dtype=np.float32)[purpose_idx]


def sanitize_work_coordinates(work_x, work_y):
    has_work = np.isfinite(work_x) & np.isfinite(work_y)
    safe_work_x = np.where(has_work, work_x, 0.0)
    safe_work_y = np.where(has_work, work_y, 0.0)
    return has_work, safe_work_x, safe_work_y


def sanitize_crowfly(crowfly):
    return np.where(np.isfinite(crowfly) & (crowfly >= 0.0), crowfly, 0.0)
