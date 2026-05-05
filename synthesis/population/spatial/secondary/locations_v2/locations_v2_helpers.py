import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import h3

from .NNModel import MNLWrapper, MediumLevel1Wrapper, DetailedLevel2Wrapper
from .hierarchical_model_utils import SECONDARY_ACTIVITIES

logger = logging.getLogger("synpp: locations_v2")

SECONDARY_SET = set(SECONDARY_ACTIVITIES)



def _prepare_primary_locations(context):
    df_home = context.stage("synthesis.population.spatial.home.locations").rename(columns={"geometry": "home"})
    df_work = context.stage("synthesis.population.spatial.primary.locations")[0].rename(columns={"geometry": "work"})
    df_education = context.stage("synthesis.population.spatial.primary.locations")[1].rename(columns={"geometry": "education"})

    df_locations = context.stage("synthesis.population.enriched")[["person_id", "household_id"]].copy()
    df_locations = df_locations.merge(df_home[["household_id", "home"]], how="left", on="household_id")
    df_locations = df_locations.merge(df_work[["person_id", "work"]], how="left", on="person_id")
    df_locations = df_locations.merge(df_education[["person_id", "education"]], how="left", on="person_id")

    return df_locations[["person_id", "home", "work", "education"]].sort_values(by="person_id").reset_index(drop=True)


def _prepare_person_attributes(context):
    df = context.stage("synthesis.population.enriched").copy()
    required = ["person_id", "age", "sex", "employed", "income_class", "car_availability", "driving_license"]
    out = df[required].copy()
    
    ca = out["car_availability"].astype(int) == 1
    dl = out["driving_license"].astype(int) == 1
    out["car_availability"] = (ca & dl).astype(float)

    return out[["person_id", "age", "sex", "employed", "income_class", "car_availability"]]


def _load_wrapper(stage_output, wrapper_cls):
    if isinstance(stage_output, tuple) and len(stage_output) > 0:
        first = stage_output[0]
        if isinstance(first, wrapper_cls):
            return first
    if isinstance(stage_output, wrapper_cls):
        return stage_output
    if isinstance(stage_output, str):
        return wrapper_cls.load(stage_output)
    if isinstance(stage_output, tuple):
        for item in stage_output:
            if isinstance(item, wrapper_cls):
                return item
            if isinstance(item, str) and item.endswith(".pt"):
                return wrapper_cls.load(item)
    raise RuntimeError(f"Could not load wrapper for {wrapper_cls.__name__}")


def _safe_xy(point):
    if point is None or not hasattr(point, "x") or pd.isna(point.x) or pd.isna(point.y):
        return np.nan, np.nan
    return float(point.x), float(point.y)


def _euclidean(x1, y1, x2, y2):
    if not np.isfinite(x1) or not np.isfinite(y1) or not np.isfinite(x2) or not np.isfinite(y2):
        return 0.0
    dx = x1 - x2
    dy = y1 - y2
    return float(np.sqrt(dx * dx + dy * dy))


def _build_level_attributes(h3_data, h3_geo_level0, all_h3 = None):
    h3_geo_level0 = h3_geo_level0.set_index("h3_index")
    centroids = h3_geo_level0["centroid"]

    def _col_or_zero(name):
        return h3_geo_level0[name] if name in h3_geo_level0.columns else pd.Series(0.0, index=h3_geo_level0.index)

    outfrac = _col_or_zero("outside_fraction")
    num_statent = _col_or_zero("num_statent")
    employees = _col_or_zero("employees")
    urban_core = _col_or_zero("urban_core")
    urban = _col_or_zero("urban")
    education = _col_or_zero("education")
    shop = _col_or_zero("shop")
    leisure = _col_or_zero("leisure")
    ovgk_share_a = _col_or_zero("ovgk_share_a")
    ovgk_share_b = _col_or_zero("ovgk_share_b")
    ovgk_share_c = _col_or_zero("ovgk_share_c")
    ovgk_share_d = _col_or_zero("ovgk_share_d")
    ovgk_share_none = _col_or_zero("ovgk_share_none")

    attrs = {}
    if all_h3 is None:
        all_h3 = centroids.index.tolist()

    for h in all_h3:
        c = centroids.get(h, None)
        if c is None:
            continue
        attrs[h] = {
            "x": float(c.x),
            "y": float(c.y),
            "num_statent": float(num_statent.get(h, 0.0)),
            "employees": float(employees.get(h, 0.0)),
            "urban_core": float(urban_core.get(h, 0.0)),
            "urban": float(urban.get(h, 0.0)),
            "education": float(education.get(h, 0.0)),
            "shop": float(shop.get(h, 0.0)),
            "leisure": float(leisure.get(h, 0.0)),
            "ovgk_share_a": float(ovgk_share_a.get(h, 0.0)),
            "ovgk_share_b": float(ovgk_share_b.get(h, 0.0)),
            "ovgk_share_c": float(ovgk_share_c.get(h, 0.0)),
            "ovgk_share_d": float(ovgk_share_d.get(h, 0.0)),
            "ovgk_share_none": float(ovgk_share_none.get(h, 0.0)),
            "outside_fraction": float(outfrac.get(h, 0.0)),
        }
    return attrs


def _prepare_destination_level2_index(context):
    df_dest = context.stage("synthesis.population.destinations").copy()
    if not isinstance(df_dest, gpd.GeoDataFrame):
        df_dest = gpd.GeoDataFrame(df_dest, geometry="geometry", crs="EPSG:2056")
    if df_dest.crs is None:
        df_dest = df_dest.set_crs("EPSG:2056", allow_override=True)

    wgs = df_dest.to_crs("EPSG:4326")
    h3_l2 = [h3.latlng_to_cell(pt.y, pt.x, 9) if pt is not None else None for pt in wgs.geometry]
    df_dest = df_dest.assign(level_2=h3_l2)

    index = {purpose: {} for purpose in SECONDARY_ACTIVITIES}
    for purpose in SECONDARY_ACTIVITIES:
        col = f"offers_{purpose}"
        if col in df_dest.columns:
            sel = df_dest[col].astype(bool).fillna(False)
            sub = df_dest[sel]
        else:
            raise RuntimeError(f"Expected column {col} in destinations data for purpose {purpose}")

        for l2, grp in sub.groupby("level_2"):
            arr = list(zip(grp["destination_id"].tolist(), grp.geometry.tolist()))
            if len(arr) > 0:
                index[purpose][l2] = arr

    fallback = {purpose: [] for purpose in SECONDARY_ACTIVITIES}
    for purpose in SECONDARY_ACTIVITIES:
        all_vals = []
        for vals in index[purpose].values():
            all_vals.extend(vals)
        fallback[purpose] = all_vals

    return index, fallback


def _build_coarse_X(wrapper, coarse_attrs, feature_inputs, purpose):
    all_h3 = wrapper.all_h3
    n = len(all_h3)
    numerical = np.zeros((n, len(wrapper.numerical_cols)), dtype=np.float64)

    for j, h in enumerate(all_h3):
        attr = coarse_attrs.get(h)
        if attr is None:
            continue
        cx = attr["x"]
        cy = attr["y"]
        dist_home = _euclidean(cx, cy, feature_inputs["home_x"], feature_inputs["home_y"])
        dist_work = _euclidean(cx, cy, feature_inputs["work_x"], feature_inputs["work_y"]) if feature_inputs["has_work"] else 0.0
        dist_last = _euclidean(cx, cy, feature_inputs["origin_x"], feature_inputs["origin_y"])

        numerical[j, 0] = dist_home
        numerical[j, 1] = dist_work
        numerical[j, 2] = dist_last
        numerical[j, 3] = feature_inputs["age"]
        numerical[j, 4] = attr["num_statent"]
        numerical[j, 5] = attr["employees"]
        numerical[j, 6] = attr["urban_core"]
        numerical[j, 7] = attr["urban"]
        numerical[j, 8] = attr["education"]
        numerical[j, 9] = attr["shop"]
        numerical[j, 10] = attr["leisure"]
        numerical[j, 11] = attr["ovgk_share_a"]
        numerical[j, 12] = attr["ovgk_share_b"]
        numerical[j, 13] = attr["ovgk_share_c"]
        numerical[j, 14] = attr["ovgk_share_d"]
        numerical[j, 15] = attr["ovgk_share_none"]
        numerical[j, 16] = attr["outside_fraction"]
        numerical[j, 17] = feature_inputs["daily_longest"]
        numerical[j, 18] = feature_inputs["daily_total"]
        numerical[j, 19] = feature_inputs["consumed_before"]
        numerical[j, 20] = feature_inputs["trip_position"]
        numerical[j, 21] = feature_inputs["income_class"]

    scaled = wrapper.scaler.transform(numerical).astype(np.float32)
    X = np.zeros((1, wrapper.model.num_h3, len(wrapper.features)), dtype=np.float32)
    X[0, :, :len(wrapper.numerical_cols)] = scaled
    X[0, :, len(wrapper.numerical_cols)] = feature_inputs["sex"]
    X[0, :, len(wrapper.numerical_cols) + 1] = feature_inputs["employed"]
    X[0, :, len(wrapper.numerical_cols) + 2] = feature_inputs["car_availability"]

    for k, f in enumerate(wrapper.features[len(wrapper.numerical_cols) + 3:]):
        expected = f[len("purpose_"):]
        X[0, :, len(wrapper.numerical_cols) + 3 + k] = 1.0 if str(purpose) == expected else 0.0

    return X


def _build_detailed_X(wrapper, children, level2_attrs, feature_inputs, purpose):
    n_children = len(children)
    X = np.zeros((1, wrapper.model.num_h3, len(wrapper.features)), dtype=np.float32)
    valid_mask = np.zeros((1, wrapper.model.num_h3), dtype=bool)
    valid_mask[0, :n_children] = True

    numerical = np.zeros((n_children, len(wrapper.numerical_cols)), dtype=np.float64)
    for j, l2 in enumerate(children):
        attr = level2_attrs.get(l2)
        if attr is None:
            continue
        cx = attr["x"]
        cy = attr["y"]
        dist_home = _euclidean(cx, cy, feature_inputs["home_x"], feature_inputs["home_y"])
        dist_work = _euclidean(cx, cy, feature_inputs["work_x"], feature_inputs["work_y"]) if feature_inputs["has_work"] else 0.0
        dist_last = _euclidean(cx, cy, feature_inputs["origin_x"], feature_inputs["origin_y"])

        numerical[j, 0] = dist_home
        numerical[j, 1] = dist_work
        numerical[j, 2] = dist_last
        numerical[j, 3] = feature_inputs["age"]
        numerical[j, 4] = attr["num_statent"]
        numerical[j, 5] = attr["employees"]
        numerical[j, 6] = attr["urban_core"]
        numerical[j, 7] = attr["urban"]
        numerical[j, 8] = attr["education"]
        numerical[j, 9] = attr["shop"]
        numerical[j, 10] = attr["leisure"]
        numerical[j, 11] = attr["ovgk_share_a"]
        numerical[j, 12] = attr["ovgk_share_b"]
        numerical[j, 13] = attr["ovgk_share_c"]
        numerical[j, 14] = attr["ovgk_share_d"]
        numerical[j, 15] = attr["ovgk_share_none"]
        numerical[j, 16] = attr["outside_fraction"]
        numerical[j, 17] = feature_inputs["daily_longest"]
        numerical[j, 18] = feature_inputs["daily_total"]
        numerical[j, 19] = feature_inputs["consumed_before"]
        numerical[j, 20] = feature_inputs["trip_position"]
        numerical[j, 21] = feature_inputs["income_class"]

    scaled = wrapper.scaler.transform(numerical).astype(np.float32)
    X[0, :n_children, :len(wrapper.numerical_cols)] = scaled
    X[0, :n_children, len(wrapper.numerical_cols)] = feature_inputs["sex"]
    X[0, :n_children, len(wrapper.numerical_cols) + 1] = feature_inputs["employed"]
    X[0, :n_children, len(wrapper.numerical_cols) + 2] = feature_inputs["car_availability"]

    for k, f in enumerate(wrapper.features[len(wrapper.numerical_cols) + 3:]):
        expected = f[len("purpose_"):]
        X[0, :n_children, len(wrapper.numerical_cols) + 3 + k] = 1.0 if str(purpose) == expected else 0.0

    return X, valid_mask


def _sample_company_in_l2(purpose, l2_h3, destination_l2_index, destination_fallback, rng):
    pool = destination_l2_index.get(purpose, {}).get(l2_h3, [])
    if len(pool) == 0:
        pool = destination_fallback.get(purpose, [])
    if len(pool) == 0:
        return None, None
    idx = int(rng.randint(0, len(pool)))
    return pool[idx]


def _coarse_company_mask(wrapper, coarse_attrs):
    return np.array([coarse_attrs.get(h, {}).get("num_statent", 0.0) > 0.0 for h in wrapper.all_h3], dtype=bool)


