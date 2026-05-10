import numpy as np
from numba import njit, prange
from .feature_encoding import ORIGIN_PURPOSE_REMAP

SECONDARY_ACTIVITIES = ["other", "shop", "leisure", "work_secondary", "home_secondary", "education_secondary"]
PRIMARY_ACTIVITIES = ["home", "work", "education"]
ALL_ACTIVITIES = SECONDARY_ACTIVITIES + PRIMARY_ACTIVITIES

ORIGIN_PURPOSE_REMAP = {"home": "home_secondary", "work": "work_secondary", "education": "education_secondary"}
PURPOSES_INDEX = {purpose: idx for idx, purpose in enumerate(SECONDARY_ACTIVITIES)}
PURPOSES_IDENTITY = np.eye(len(ALL_ACTIVITIES), dtype=np.float32)

@njit(parallel=True, fastmath=True)
def build_hierarchical_candidate_batch_numba(hx, hy, wx, wy, has_work, ox, oy, cand_x, cand_y, cand_statent, cand_employees, cand_urban_core, cand_urban, cand_education, cand_shop, cand_leisure, cand_ovgk_share_a, cand_ovgk_share_b, cand_ovgk_share_c, cand_ovgk_share_d, cand_ovgk_share_none, cand_outside_fraction, valid_mask):
    n, max_children = cand_x.shape
    out = np.zeros((n, max_children, 16), dtype=np.float64)

    for i in prange(n):
        valid_work = has_work[i]
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
            out[i, j, 4] = cand_employees[i, j]
            out[i, j, 5] = cand_urban_core[i, j]
            out[i, j, 6] = cand_urban[i, j]
            out[i, j, 7] = cand_education[i, j]
            out[i, j, 8] = cand_shop[i, j]
            out[i, j, 9] = cand_leisure[i, j]
            out[i, j, 10] = cand_ovgk_share_a[i, j]
            out[i, j, 11] = cand_ovgk_share_b[i, j]
            out[i, j, 12] = cand_ovgk_share_c[i, j]
            out[i, j, 13] = cand_ovgk_share_d[i, j]
            out[i, j, 14] = cand_ovgk_share_none[i, j]
            out[i, j, 15] = cand_outside_fraction[i, j]

    return out


@njit(parallel=True, fastmath=True)
def build_coarse_candidate_batch_numba(hx, hy, wx, wy, has_work, ox, oy, centroid_x, centroid_y, statent_per_h3, employees_per_h3, urban_core_per_h3, urban_per_h3, education_per_h3, shop_per_h3, leisure_per_h3, ovgk_share_a_per_h3, ovgk_share_b_per_h3, ovgk_share_c_per_h3, ovgk_share_d_per_h3, ovgk_share_none_per_h3, outside_fraction):
    n = hx.shape[0]
    num_h3 = centroid_x.shape[0]
    out = np.zeros((n, num_h3, 16), dtype=np.float64)

    for i in prange(n):
        valid_work = has_work[i]
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
            out[i, j, 3] = statent_per_h3[j]
            out[i, j, 4] = employees_per_h3[j]
            out[i, j, 5] = urban_core_per_h3[j]
            out[i, j, 6] = urban_per_h3[j]
            out[i, j, 7] = education_per_h3[j]
            out[i, j, 8] = shop_per_h3[j]
            out[i, j, 9] = leisure_per_h3[j]
            out[i, j, 10] = ovgk_share_a_per_h3[j]
            out[i, j, 11] = ovgk_share_b_per_h3[j]
            out[i, j, 12] = ovgk_share_c_per_h3[j]
            out[i, j, 13] = ovgk_share_d_per_h3[j]
            out[i, j, 14] = ovgk_share_none_per_h3[j]
            out[i, j, 15] = outside_fraction[j]

    return out


def build_level1_children_by_level0(h3_tree, centroid_x_by_l1, centroid_y_by_l1):
    children = {l0: sorted([l1 for l1 in level1_dict.keys() if l1 in centroid_x_by_l1 and l1 in centroid_y_by_l1]) for l0, level1_dict in h3_tree.items()}
    return {l0: values for l0, values in children.items() if len(values) > 0}


def build_level1_candidate_attributes_by_level0(children_by_level0, centroid_x_by_l1, centroid_y_by_l1, statent_count, employees_count, urban_core_count, urban_count, education_count, shop_count, leisure_count, ovgk_share_a_by_l1, ovgk_share_b_by_l1, ovgk_share_c_by_l1, ovgk_share_d_by_l1, ovgk_share_none_by_l1, outside_fraction_by_l1):
    candidate_attributes = {}
    for l0, children in children_by_level0.items():
        x = np.array([float(centroid_x_by_l1[c]) for c in children], dtype=np.float64)
        y = np.array([float(centroid_y_by_l1[c]) for c in children], dtype=np.float64)
        num_statent = np.array([float(statent_count.get(c, 0.0)) for c in children], dtype=np.float64)
        employees = np.array([float(employees_count.get(c, 0.0)) for c in children], dtype=np.float64)
        urban_core = np.array([float(urban_core_count.get(c, 0.0)) for c in children], dtype=np.float64)
        urban = np.array([float(urban_count.get(c, 0.0)) for c in children], dtype=np.float64)
        education = np.array([float(education_count.get(c, 0.0)) for c in children], dtype=np.float64)
        shop = np.array([float(shop_count.get(c, 0.0)) for c in children], dtype=np.float64)
        leisure = np.array([float(leisure_count.get(c, 0.0)) for c in children], dtype=np.float64)
        ovgk_share_a = np.array([float(ovgk_share_a_by_l1.get(c, 0.0)) for c in children], dtype=np.float64)
        ovgk_share_b = np.array([float(ovgk_share_b_by_l1.get(c, 0.0)) for c in children], dtype=np.float64)
        ovgk_share_c = np.array([float(ovgk_share_c_by_l1.get(c, 0.0)) for c in children], dtype=np.float64)
        ovgk_share_d = np.array([float(ovgk_share_d_by_l1.get(c, 0.0)) for c in children], dtype=np.float64)
        ovgk_share_none = np.array([float(ovgk_share_none_by_l1.get(c, 0.0)) for c in children], dtype=np.float64)
        outside_fraction = np.array([float(outside_fraction_by_l1.get(c, 0.0)) for c in children], dtype=np.float64)

        candidate_attributes[l0] = {
            "children": list(children),
            "index_by_child": {child: idx for idx, child in enumerate(children)},
            "x": x,
            "y": y,
            "num_statent": num_statent,
            "employees": employees,
            "urban_core": urban_core,
            "urban": urban,
            "education": education,
            "shop": shop,
            "leisure": leisure,
            "ovgk_share_a": ovgk_share_a,
            "ovgk_share_b": ovgk_share_b,
            "ovgk_share_c": ovgk_share_c,
            "ovgk_share_d": ovgk_share_d,
            "ovgk_share_none": ovgk_share_none,
            "outside_fraction": outside_fraction,
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


def build_level2_candidate_attributes_by_level1(children_by_level1, centroid_x_by_l2, centroid_y_by_l2, statent_count, employees_count, urban_core_count, urban_count, education_count, shop_count, leisure_count, ovgk_share_a_by_l2, ovgk_share_b_by_l2, ovgk_share_c_by_l2, ovgk_share_d_by_l2, ovgk_share_none_by_l2, outside_fraction_by_l2):
    candidate_attributes = {}
    for key, children in children_by_level1.items():
        x = np.array([float(centroid_x_by_l2[c]) for c in children], dtype=np.float64)
        y = np.array([float(centroid_y_by_l2[c]) for c in children], dtype=np.float64)
        num_statent = np.array([float(statent_count.get(c, 0.0)) for c in children], dtype=np.float64)
        employees = np.array([float(employees_count.get(c, 0.0)) for c in children], dtype=np.float64)
        urban_core = np.array([float(urban_core_count.get(c, 0.0)) for c in children], dtype=np.float64)
        urban = np.array([float(urban_count.get(c, 0.0)) for c in children], dtype=np.float64)
        education = np.array([float(education_count.get(c, 0.0)) for c in children], dtype=np.float64)
        shop = np.array([float(shop_count.get(c, 0.0)) for c in children], dtype=np.float64)
        leisure = np.array([float(leisure_count.get(c, 0.0)) for c in children], dtype=np.float64)
        ovgk_share_a = np.array([float(ovgk_share_a_by_l2.get(c, 0.0)) for c in children], dtype=np.float64)
        ovgk_share_b = np.array([float(ovgk_share_b_by_l2.get(c, 0.0)) for c in children], dtype=np.float64)
        ovgk_share_c = np.array([float(ovgk_share_c_by_l2.get(c, 0.0)) for c in children], dtype=np.float64)
        ovgk_share_d = np.array([float(ovgk_share_d_by_l2.get(c, 0.0)) for c in children], dtype=np.float64)
        ovgk_share_none = np.array([float(ovgk_share_none_by_l2.get(c, 0.0)) for c in children], dtype=np.float64)
        outside_fraction = np.array([float(outside_fraction_by_l2.get(c, 0.0)) for c in children], dtype=np.float64)

        candidate_attributes[key] = {
            "children": list(children),
            "index_by_child": {child: idx for idx, child in enumerate(children)},
            "x": x,
            "y": y,
            "num_statent": num_statent,
            "employees": employees,
            "urban_core": urban_core,
            "urban": urban,
            "education": education,
            "shop": shop,
            "leisure": leisure,
            "ovgk_share_a": ovgk_share_a,
            "ovgk_share_b": ovgk_share_b,
            "ovgk_share_c": ovgk_share_c,
            "ovgk_share_d": ovgk_share_d,
            "ovgk_share_none": ovgk_share_none,
            "outside_fraction": outside_fraction,
        }
    return candidate_attributes


def sanitize_work_coordinates(work_x, work_y):
    has_work = np.isfinite(work_x) & np.isfinite(work_y)
    safe_work_x = np.where(has_work, work_x, 0.0)
    safe_work_y = np.where(has_work, work_y, 0.0)
    return has_work, safe_work_x, safe_work_y


@njit(fastmath=True)
def build_dynamic_vector(hx, hy, wx, wy, ox, oy, centroid_x, centroid_y, has_work):

    n = centroid_x.shape[0]
    out = np.empty((n, 3), dtype=np.float32)

    for i in range(n):
        cx = centroid_x[i]
        cy = centroid_y[i]

        dx = cx - hx
        dy = cy - hy
        out[i, 0] = (dx * dx + dy * dy) ** 0.5

        if has_work:
            dx = cx - wx
            dy = cy - wy
            out[i, 1] = (dx * dx + dy * dy) ** 0.5
        else:
            out[i, 1] = 0.0

        dx = cx - ox
        dy = cy - oy
        out[i, 2] = (dx * dx + dy * dy) ** 0.5

    return out

# Basic sanity check for the build_dynamic_vector function
_ = build_dynamic_vector(12.5, 45.0, 8.0, 40.0, 10.0, 42.0, np.array([11.0, 13.0], dtype=np.float64), np.array([44.0, 46.0], dtype=np.float64), True)


def encode_purpose(purpose):
    remapped_purpose = ORIGIN_PURPOSE_REMAP.get(purpose, purpose)
    purpose_index = PURPOSES_INDEX[remapped_purpose]
    return PURPOSES_IDENTITY[purpose_index]