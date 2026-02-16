import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier


# ---------------------------------------------------------
# helper: stochastic draw from class probabilities
# ---------------------------------------------------------
def draw_multinomial_from_proba(proba_matrix, classes, seed=None):
    rng = np.random.default_rng(seed)
    cum_proba = np.cumsum(proba_matrix, axis=1)
    r = rng.random(proba_matrix.shape[0])[:, None]
    chosen_idx = (r < cum_proba).argmax(axis=1)
    return classes[chosen_idx]


def _weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if m.sum() == 0:
        return np.nan
    return np.average(x[m], weights=w[m])


def _as_df(obj):
    # some stages return [df]; handle both
    if isinstance(obj, (list, tuple)):
        return obj[0]
    return obj


# =========================================================
# NEW: COMMUTE helpers (Cartesian / projected coordinates)
# =========================================================
def weighted_quantile(values, quantiles, sample_weight=None):
    """
    values: 1D array
    quantiles: list/array in [0,1]
    sample_weight: 1D array of same length, non-negative
    """
    v = np.asarray(values, dtype=float)
    q = np.asarray(quantiles, dtype=float)

    if sample_weight is None:
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.full_like(q, np.nan, dtype=float)
        return np.quantile(v, q)

    w = np.asarray(sample_weight, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.full_like(q, np.nan, dtype=float)

    v = v[m]
    w = w[m]

    sorter = np.argsort(v)
    v = v[sorter]
    w = w[sorter]

    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(q, cw, v)


def point_xy_from_geometry(geom):
    """
    Supports shapely Points or objects with .x/.y.
    Returns (x,y) or (nan,nan).
    """
    if geom is None:
        return (np.nan, np.nan)
    try:
        return (float(geom.x), float(geom.y))
    except Exception:
        return (np.nan, np.nan)


def compute_commute_km_cartesian(home_x, home_y, dest_x, dest_y):
    """
    Euclidean (Cartesian) distance. Assumes coordinates are in meters.
    Returns km.
    """
    hx = np.asarray(home_x, dtype=float)
    hy = np.asarray(home_y, dtype=float)
    dx = np.asarray(dest_x, dtype=float)
    dy = np.asarray(dest_y, dtype=float)

    m = np.isfinite(hx) & np.isfinite(hy) & np.isfinite(dx) & np.isfinite(dy)
    out = np.full(hx.shape, np.nan, dtype=float)
    if m.sum() == 0:
        return out

    out[m] = np.sqrt((hx[m] - dx[m]) ** 2 + (hy[m] - dy[m]) ** 2) / 1000.0
    return out


def commute_from_location_df(pop_loc_df, pop_df, how="min"):
    """
    pop_loc_df columns: person_id, geometry (Point in projected meters)
    pop_df columns: person_id, home_x, home_y

    Returns pd.Series commute_km per person_id.
    If multiple destinations per person, aggregates by min/mean/max.
    """
    if pop_loc_df is None or len(pop_loc_df) == 0:
        return pd.Series(dtype=float)

    loc = pop_loc_df[["person_id", "geometry"]].copy()

    xy = loc["geometry"].apply(point_xy_from_geometry)
    loc["dest_x"] = [t[0] for t in xy]
    loc["dest_y"] = [t[1] for t in xy]

    home = pop_df[["person_id", "home_x", "home_y"]].copy()
    loc = loc.merge(home, on="person_id", how="left")

    loc["commute_km"] = compute_commute_km_cartesian(
        loc["home_x"].values, loc["home_y"].values,
        loc["dest_x"].values, loc["dest_y"].values
    )

    if how == "mean":
        s = loc.groupby("person_id")["commute_km"].mean()
    elif how == "max":
        s = loc.groupby("person_id")["commute_km"].max()
    else:
        s = loc.groupby("person_id")["commute_km"].min()

    return s


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.household_persons")
    context.stage("synthesis.population.spatial.primary.locations")
    context.stage("synthesis.population.enriched")
    context.stage("data.microcensus.commute")


def execute(context):
    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------
    PT_MODEL = "catboost"      # "rf" or "gbm" or "catboost"
    SEED_PT  = 2026
    DIAG_CANTON_ID = "1"
    MIN_AGE = 6
    USE_DRAW_DIAG = True

    # split ages into 6-15 and 16+
    AGE_SPLIT = 16
    SEED_PT_YOUTH = 20261   # separate seed so draws are independent (optional)

    # Survey columns (per your assumption)
    SURVEY_TARGET_COL = "subscription"        # values 0..4
    SURVEY_WEIGHT_COL = "person_weight"

    # Survey car ownership proxy (so survey and pop align on a "car ownership class" feature)
    SURVEY_CARCOUNT_COL = "number_of_cars_class"   # if present; used only to derive HH_CAR_OWN_class in survey

    # Population car ownership variable
    POP_CAR_OWN_COL = "number_of_cars_class"

    # If your survey commute_home_distance is in meters set True (else assume km)
    SURVEY_COMMUTE_IN_METERS = True

    COMMUTE_DIAG_THRESHOLD_KM = 0

    # -------------------------------------------------------------------
    # 0. LOAD DATA
    # -------------------------------------------------------------------
    survey_df = _as_df(context.stage("data.microcensus.persons")).copy()
    # only use those interviewed on a workday, because those on teh weekend do not have commmute information
    survey_df = survey_df[survey_df["workday"]]
    # booleans (make sure they're boolean and missing -> False)
    cols = [
        "subscriptions_ga",
        "subscriptions_halbtax",
        "subscriptions_verbund",
        "subscriptions_strecke",
        "subscriptions_gleis7",
        "subscriptions_junior",
        "subscriptions_other",
    ]
    for c in cols:
        survey_df[c] = survey_df[c].fillna(False).astype(bool)

    ga  = survey_df["subscriptions_ga"]
    ht  = survey_df["subscriptions_halbtax"]
    vb  = survey_df["subscriptions_verbund"]
    junior = survey_df["subscriptions_junior"]
    strecke = survey_df["subscriptions_strecke"]

    # ---------------------------------------------------------
    # combine junior and ga for those below 16
    # combine strecke and verbund abo into one
    # ---------------------------------------------------------

    survey_df["subscriptions_ga_combined"] = ga | (junior & (pd.to_numeric(survey_df["age"], errors="coerce") < 16))
    survey_df["subscriptions_verbund_combined"] = vb | strecke

    # use combined flags from here on
    ga = survey_df["subscriptions_ga_combined"]
    vb = survey_df["subscriptions_verbund_combined"]

    # any subscription at all (includes ga/ht/vb and the other types)
    any_sub = survey_df[cols].any(axis=1)

    # default 0 if none
    survey_df["subscription"] = 0

    # prioritize GA first
    survey_df.loc[ga, "subscription"] = 1

    # then the verbund/halbtax combos for non-GA people
    non_ga = ~ga
    survey_df.loc[non_ga & vb & ~ht, "subscription"] = 2
    survey_df.loc[non_ga & ht & ~vb, "subscription"] = 3
    survey_df.loc[non_ga & ht & vb,  "subscription"] = 4

    survey_hh_df = _as_df(context.stage("data.microcensus.household_persons"))
    survey_commute = context.stage("data.microcensus.commute")
    pop_df = _as_df(context.stage("synthesis.population.enriched"))
    pop_locations_work = context.stage("synthesis.population.spatial.primary.locations")[0]
    pop_locations_education = context.stage("synthesis.population.spatial.primary.locations")[1]

    # -------------------------------------------------------------------
    # 0b. NEW FEATURE: COMMUTE (survey + population) -> commute_class
    # -------------------------------------------------------------------
    # ---- SURVEY: merge commute_home_distance from commute dict (work + education)
    def _get_commute_df(commute_obj, key_candidates):
        if isinstance(commute_obj, dict):
            for k in key_candidates:
                if k in commute_obj:
                    return _as_df(commute_obj[k]).copy()
        if isinstance(commute_obj, (list, tuple)) and len(commute_obj) >= 2:
            # fallback assumption ordering [work, education]
            if "work" in key_candidates:
                return _as_df(commute_obj[0]).copy()
            else:
                return _as_df(commute_obj[1]).copy()
        return None

    survey_work = _get_commute_df(survey_commute, ["work", "employment", "W", "Work"])
    survey_edu  = _get_commute_df(survey_commute, ["education", "edu", "E", "Education"])

    for name, dfc in [("work", survey_work), ("education", survey_edu)]:
        if dfc is None:
            continue
        if "person_id" not in dfc.columns or "commute_home_distance" not in dfc.columns:
            raise KeyError(
                f"survey_commute[{name}] must contain columns ['person_id','commute_home_distance']"
            )
        dfc["commute_home_distance"] = pd.to_numeric(dfc["commute_home_distance"], errors="coerce")

    if survey_work is not None:
        s_work = survey_work[["person_id", "commute_home_distance"]].rename(
            columns={"commute_home_distance": "commute_work_raw"}
        )
    else:
        s_work = pd.DataFrame(columns=["person_id", "commute_work_raw"])

    if survey_edu is not None:
        s_edu = survey_edu[["person_id", "commute_home_distance"]].rename(
            columns={"commute_home_distance": "commute_edu_raw"}
        )
    else:
        s_edu = pd.DataFrame(columns=["person_id", "commute_edu_raw"])

    survey_df = survey_df.merge(s_work, on="person_id", how="left")
    survey_df = survey_df.merge(s_edu,  on="person_id", how="left")

    # convert to km
    if SURVEY_COMMUTE_IN_METERS:
        survey_df["commute_work_km"] = pd.to_numeric(survey_df["commute_work_raw"], errors="coerce") / 1000.0
        survey_df["commute_edu_km"]  = pd.to_numeric(survey_df["commute_edu_raw"], errors="coerce") / 1000.0
    else:
        survey_df["commute_work_km"] = pd.to_numeric(survey_df["commute_work_raw"], errors="coerce")
        survey_df["commute_edu_km"]  = pd.to_numeric(survey_df["commute_edu_raw"], errors="coerce")

    # combine work+edu: use max as "overall commute exposure"
    survey_df["commute_km"] = np.nanmax(
        np.c_[survey_df["commute_work_km"].values, survey_df["commute_edu_km"].values],
        axis=1
    )
    survey_df["commute_km"] = survey_df["commute_km"].fillna(0.0).clip(lower=0.0)

    # ---- POP: compute from home_x/home_y to POINT geometries (Cartesian, meters)
    pop_work_s = commute_from_location_df(pop_locations_work, pop_df, how="min")
    pop_edu_s  = commute_from_location_df(pop_locations_education, pop_df, how="min")

    pop_df = pop_df.merge(pop_work_s.rename("commute_work_km"), left_on="person_id", right_index=True, how="left")
    pop_df = pop_df.merge(pop_edu_s.rename("commute_edu_km"),  left_on="person_id", right_index=True, how="left")

    pop_df["commute_work_km"] = pd.to_numeric(pop_df["commute_work_km"], errors="coerce")
    pop_df["commute_edu_km"]  = pd.to_numeric(pop_df["commute_edu_km"], errors="coerce")

    pop_df["commute_km"] = np.nanmax(
        np.c_[pop_df["commute_work_km"].values, pop_df["commute_edu_km"].values],
        axis=1
    )
    pop_df["commute_km"] = pop_df["commute_km"].fillna(0.0).clip(lower=0.0)

    # ---- define bins from survey: weighted tertiles among commuters (>0)
    ww = pd.to_numeric(survey_df.get(SURVEY_WEIGHT_COL), errors="coerce").values if SURVEY_WEIGHT_COL in survey_df.columns else None
    commuter_mask = (survey_df["commute_km"].values > 0) & np.isfinite(survey_df["commute_km"].values)

    if commuter_mask.sum() >= 50:
        q30, q95 = weighted_quantile(
            survey_df.loc[commuter_mask, "commute_km"].values,
            [3/10, 95/100],
            sample_weight=(ww[commuter_mask] if ww is not None else None)
        )
    else:
        # fallback fixed km cutoffs if too few commuters in training sample
        q33, q66 = 5.0, 15.0

    def commute_class_from_km(km):
        if not np.isfinite(km) or km <= 0:
            return "none"
        elif km <= q30:
            return "short"
        elif km <= q95:
            return "medium"
        else:
            return "long"

    survey_df["commute_class"] = survey_df["commute_km"].apply(commute_class_from_km).astype(str)
    pop_df["commute_class"]    = pop_df["commute_km"].apply(commute_class_from_km).astype(str)

    # -------------------------------------------------------------------
    # 1. SURVEY HH AGGREGATES FROM household_persons: adult DL intensity
    #    (Same pattern as car ownership script)
    # -------------------------------------------------------------------
    # IDs:
    if "person_id" not in survey_df.columns:
        raise KeyError("survey_df must contain 'person_id' (used as household id in microcensus persons).")
    if "household_id" not in survey_hh_df.columns:
        raise KeyError("survey_hh_df must contain 'household_id'.")

    survey_hh_df["age"] = pd.to_numeric(survey_hh_df.get("age"), errors="coerce")
    survey_hh_df["driving_license"] = pd.to_numeric(survey_hh_df.get("driving_license"), errors="coerce")  # expects 1/0 or similar

    hh_lic = survey_hh_df.loc[survey_hh_df["age"].notna()].copy()
    hh_lic["is_adult"] = (hh_lic["age"] >= 18).astype(int)

    # treat missing DL as 0; only count among adults
    hh_lic["dl_adult"] = hh_lic["driving_license"].fillna(0).where(hh_lic["age"] >= 18, 0)

    hh_agg = (
        hh_lic.groupby("household_id", as_index=False)
              .agg(
                  N_adults_survey=("is_adult", "sum"),
                  N_drivers_license_adults=("dl_adult", "sum"),
              )
    )
    hh_agg["N_drivers_license_per_adult"] = (
        hh_agg["N_drivers_license_adults"] / hh_agg["N_adults_survey"].replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)

    # merge onto survey_df by: survey_df.person_id == survey_hh_df.household_id
    survey_df = survey_df.merge(
        hh_agg[["household_id", "N_adults_survey", "N_drivers_license_adults", "N_drivers_license_per_adult"]],
        left_on="person_id", right_on="household_id", how="left"
    ).drop(columns=["household_id"])

    # -------------------------------------------------------------------
    # 2. POP: basic household composition + DL intensity (same idea as car ownership)
    # -------------------------------------------------------------------
    for df in (survey_df, pop_df):
        df["age"] = pd.to_numeric(df.get("age"), errors="coerce")

    # N_adults / N_children in pop
    adult_mask_pop = pop_df["age"].notna() & (pop_df["age"] >= 18)
    pop_df["N_adults"] = (
        adult_mask_pop.astype(int)
        .groupby(pop_df["household_id"])
        .transform("sum")
    )

    child_mask_pop = pop_df["age"].notna() & (pop_df["age"] < 18)
    pop_df["N_children_under_18"] = (
        child_mask_pop.astype(int)
        .groupby(pop_df["household_id"])
        .transform("sum")
    )

    pop_df["presence_of_children_under_18"] = (pop_df["N_children_under_18"] > 0).astype(int)

    # DL intensity in pop
    if "DL_has_or_learning_draw" in pop_df.columns:
        dl_pop_col = "DL_has_or_learning_draw"
    elif "DL_has_or_learning_hat" in pop_df.columns:
        dl_pop_col = "DL_has_or_learning_hat"
    elif "driving_license" in pop_df.columns:
        dl_pop_col = "driving_license"
    else:
        dl_pop_col = None

    if dl_pop_col is not None:
        dl_vals = pd.to_numeric(pop_df[dl_pop_col], errors="coerce")
        dl_adult = dl_vals.where(pop_df["age"] >= 18, 0.0).fillna(0.0)
        pop_df["_dl_adult"] = dl_adult
        dl_sum = pop_df.groupby("household_id")["_dl_adult"].transform("sum")
        denom = pop_df["N_adults"].replace(0, np.nan)
        pop_df["N_drivers_license_per_adult"] = (dl_sum / denom).fillna(0.0).clip(0.0, 1.0)
    else:
        pop_df["N_drivers_license_per_adult"] = 0.0

    # -------------------------------------------------------------------
    # 3. CAR OWNERSHIP FEATURE (align survey + pop as HH_CAR_OWN_class)
    # -------------------------------------------------------------------
    # Survey: derive HH_CAR_OWN_class from number_of_cars_class if available
    if SURVEY_CARCOUNT_COL in survey_df.columns:
        survey_df[SURVEY_CARCOUNT_COL] = pd.to_numeric(survey_df[SURVEY_CARCOUNT_COL], errors="coerce")
        # clip to 0..3 for "3+"
        survey_df["HH_CAR_OWN_class"] = survey_df[SURVEY_CARCOUNT_COL].fillna(0).clip(lower=0).astype(int).clip(upper=3)
    else:
        survey_df["HH_CAR_OWN_class"] = np.nan

    # Pop: use HH_CAR_OWN_draw and clip to 0..3
    if POP_CAR_OWN_COL not in pop_df.columns:
        raise KeyError(
            f"pop_df must contain '{POP_CAR_OWN_COL}' to use car ownership as a feature. "
            f"Use the population stage that already has HH_CAR_OWN_draw."
        )
    pop_df[POP_CAR_OWN_COL] = pd.to_numeric(pop_df[POP_CAR_OWN_COL], errors="coerce").fillna(0).clip(lower=0)
    pop_df["HH_CAR_OWN_class"] = pop_df[POP_CAR_OWN_COL].astype(int).clip(upper=3)

    # amke sure car_availability is coded 0/1 also in survey dataframe
    var_raw = pd.to_numeric(survey_df["car_availability"], errors="coerce")
    survey_df["car_availability"] = np.where(var_raw == 2, 0, 1).astype("int64")

    # -------------------------------------------------------------------
    # 3b. NEW FEATURE: number of cars per adult
    # -------------------------------------------------------------------
    # Survey denominator: prefer N_adults_survey (from household_persons), else fall back to N_adults if it exists
    if "N_adults_survey" in survey_df.columns:
        denom_s = pd.to_numeric(survey_df["N_adults_survey"], errors="coerce")
    elif "N_adults" in survey_df.columns:
        denom_s = pd.to_numeric(survey_df["N_adults"], errors="coerce")
    else:
        denom_s = pd.Series(np.nan, index=survey_df.index)

    denom_s = denom_s.replace(0, np.nan)
    survey_df["N_cars_per_adult"] = (
        pd.to_numeric(survey_df["HH_CAR_OWN_class"], errors="coerce") / denom_s
    ).fillna(0.0).clip(lower=0.0)

    # Pop denominator: N_adults computed earlier in pop
    denom_p = pd.to_numeric(pop_df["N_adults"], errors="coerce").replace(0, np.nan)
    pop_df["N_cars_per_adult"] = (
        pd.to_numeric(pop_df["HH_CAR_OWN_class"], errors="coerce") / denom_p
    ).fillna(0.0).clip(lower=0.0)

    #
    # Adjust ovgk
    #
    survey_df["ovgk"] = (~survey_df["ovgk"].isin(["None", "D"])).astype("int64")
    pop_df["ovgk"] = (~pop_df["ovgk"].isin(["None", "D"])).astype("int64")
    # -------------------------------------------------------------------
    # 4. TARGET PREP (survey)
    # -------------------------------------------------------------------
    if SURVEY_TARGET_COL not in survey_df.columns:
        raise KeyError(f"Survey target '{SURVEY_TARGET_COL}' not found in survey_df.")

    survey_df[SURVEY_TARGET_COL] = pd.to_numeric(survey_df[SURVEY_TARGET_COL], errors="coerce")

    # Keep only valid classes 0..4 and age >= MIN_AGE
    survey_df = survey_df[survey_df["age"].notna()].copy()
    survey_train_mask = (
        (survey_df["age"] >= MIN_AGE) &
        survey_df[SURVEY_TARGET_COL].notna() &
        survey_df[SURVEY_TARGET_COL].between(0, 4, inclusive="both")
    )

    # weights
    if SURVEY_WEIGHT_COL not in survey_df.columns:
        raise KeyError(f"Survey weight column '{SURVEY_WEIGHT_COL}' not found in survey_df.")
    survey_df[SURVEY_WEIGHT_COL] = pd.to_numeric(survey_df[SURVEY_WEIGHT_COL], errors="coerce")
    survey_train_mask = survey_train_mask & survey_df[SURVEY_WEIGHT_COL].notna() & (survey_df[SURVEY_WEIGHT_COL] > 0)

    # -------------------------------------------------------------------
    # 5. FEATURE ENGINEERING (same general style as car ownership + person vars)
    # -------------------------------------------------------------------
    # age bins
    age_bins = [0, 6, 16, 18, 26, 45, 60, 71, 81, 200]
    age_labels = ["0-5", "6-15", "16-17", "18-25", "26-44", "45-59", "60-70", "71-80", "81+"]

    for df in (survey_df, pop_df):
        df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
        df["age_sq"] = df["age"] ** 2
        df["is_16_17"] = ((df["age"] >= 16) & (df["age"] < 18)).astype(int)

    # for the 16+ model, make lowest bin 16-25 (discounted youth tickets)
    age_bins_16p = [0, 6, 16, 26, 45, 60, 71, 81, 200]
    age_labels_16p = ["0-5", "6-15", "16-25", "26-44", "45-59", "60-70", "71-80", "81+"]

    for df in (survey_df, pop_df):
        df["age_bin_16plus"] = pd.cut(df["age"], bins=age_bins_16p, labels=age_labels_16p, right=False)
        df["age_bin_16plus"] = df["age_bin_16plus"].astype(str).fillna("Missing")

    # common household-ish features that may exist in survey already; if missing, fill later
    if "N_children_under_18" in survey_df.columns:
        survey_df["presence_of_children_under_18"] = (pd.to_numeric(survey_df["N_children_under_18"], errors="coerce").fillna(0) > 0).astype(int)
    else:
        survey_df["presence_of_children_under_18"] = 0

    # feature lists
    candidate_cat = [
        "age_bin",  # will be swapped per-model below
        "sex",
        "canton_id",
        "municipality_type",
        "ovgk",
        "employment_status",
        "HH_CAR_OWN_class",
        "car_availability",
        #"commute_class",   # <--- NEW
    ]

    candidate_num = [
        "age",
        "commute_km"
    ]

    # We will align names by only taking intersection of columns we actually have in each df
    cat_cols = [c for c in candidate_cat if c in survey_df.columns and c in pop_df.columns]
    num_cols = [c for c in candidate_num if c in survey_df.columns and c in pop_df.columns]

    # clean categoricals
    for df in (survey_df, pop_df):
        for c in cat_cols:
            df[c] = df[c].astype(str).fillna("Missing")

    # clean numerics
    for df in (survey_df, pop_df):
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # make sure driving license is coded 0/1 and not True/False
    s = survey_df["driving_license"].astype(str).str.strip().str.lower()
    survey_df["driving_license"] = s.map({"true": 1, "false": 0})

    # -------------------------------------------------------------------
    # 6. DESIGN MATRICES
    # -------------------------------------------------------------------
    # TWO design matrices: one for ages 6-15, one for ages 16+

    # Split masks
    train_mask_youth = survey_train_mask & (survey_df["age"] >= MIN_AGE) & (survey_df["age"] < AGE_SPLIT)
    train_mask_adult = survey_train_mask & (survey_df["age"] >= AGE_SPLIT)

    pop_mask_youth = (pop_df["age"] >= MIN_AGE) & (pop_df["age"] < AGE_SPLIT)
    pop_mask_adult = (pop_df["age"] >= AGE_SPLIT)

    # -------------------------
    # youth model features
    # -------------------------
    youth_cat_wanted = ["age_bin", "sex", "canton_id", "municipality_type", "ovgk", "commute_class"]
    youth_num_wanted = ["age"]

    cat_cols_youth = [c for c in youth_cat_wanted if c in survey_df.columns and c in pop_df.columns]
    num_cols_youth = [c for c in youth_num_wanted if c in survey_df.columns and c in pop_df.columns]

    # ensure proper types for youth-only cols
    for df in (survey_df, pop_df):
        for c in cat_cols_youth:
            df[c] = df[c].astype(str).fillna("Missing")
        for c in num_cols_youth:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    Xs_y = pd.get_dummies(survey_df.loc[train_mask_youth, cat_cols_youth], drop_first=False)
    Xp_y = pd.get_dummies(pop_df.loc[pop_mask_youth, cat_cols_youth], drop_first=False)
    if num_cols_youth:
        Xs_y[num_cols_youth] = survey_df.loc[train_mask_youth, num_cols_youth].astype(float).values
        Xp_y[num_cols_youth] = pop_df.loc[pop_mask_youth, num_cols_youth].astype(float).values
    Xp_y = Xp_y.reindex(columns=Xs_y.columns, fill_value=0.0)

    y_y = survey_df.loc[train_mask_youth, SURVEY_TARGET_COL].astype("int64").values
    w_y = survey_df.loc[train_mask_youth, SURVEY_WEIGHT_COL].astype(float).values

    # --- Adult/16+ model: swap to age_bin_16plus (so lowest bin is 16-25)
    cat_cols_adult = []
    for c in cat_cols:
        if c == "age_bin":
            cat_cols_adult.append("age_bin_16plus")
        else:
            cat_cols_adult.append(c)
    cat_cols_adult = [c for c in cat_cols_adult if c in survey_df.columns and c in pop_df.columns]

    Xs_a = pd.get_dummies(survey_df.loc[train_mask_adult, cat_cols_adult], drop_first=False)
    Xp_a = pd.get_dummies(pop_df.loc[pop_mask_adult, cat_cols_adult], drop_first=False)
    if num_cols:
        Xs_a[num_cols] = survey_df.loc[train_mask_adult, num_cols].astype(float).values
        Xp_a[num_cols] = pop_df.loc[pop_mask_adult, num_cols].astype(float).values
    Xp_a = Xp_a.reindex(columns=Xs_a.columns, fill_value=0.0)

    y_a = survey_df.loc[train_mask_adult, SURVEY_TARGET_COL].astype("int64").values
    w_a = survey_df.loc[train_mask_adult, SURVEY_WEIGHT_COL].astype(float).values

    # -------------------------------------------------------------------
    # 7. FIT MODEL (multiclass: 0..4)
    # -------------------------------------------------------------------
    def build_pt_model(model_type: str):
        mt = str(model_type).lower().strip()
        if mt == "gbm":
            return HistGradientBoostingClassifier(
                loss="log_loss",
                max_depth=8,
                learning_rate=0.05,
                max_iter=600,
                min_samples_leaf=80,
                random_state=42
            )
        elif mt == "rf":
            return RandomForestClassifier(
                n_estimators=900,
                max_depth=None,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        elif mt in ("catboost", "cat"):
            return CatBoostClassifier(
                loss_function="MultiClass",
                iterations=2200,
                learning_rate=0.05,
                depth=10,
                l2_leaf_reg=6.0,
                random_seed=42,
                verbose=False,
                bootstrap_type="Bernoulli",
                subsample=0.8
            )
        else:
            raise ValueError(f"Unknown model_type={model_type}, use 'gbm', 'rf', or 'catboost'.")

    # Fit youth model
    pt_model_youth = build_pt_model(PT_MODEL)
    Xs_y_np = Xs_y.to_numpy(dtype=float, copy=False)
    Xp_y_np = Xp_y.to_numpy(dtype=float, copy=False)

    print("Started to fit PT subscription model (6-15) using:", PT_MODEL)
    pt_model_youth.fit(Xs_y_np, y_y, sample_weight=w_y)
    print("Fitted PT subscription model (6-15) using:", PT_MODEL)

    # Fit adult model (16+)
    pt_model_adult = build_pt_model(PT_MODEL)
    Xs_a_np = Xs_a.to_numpy(dtype=float, copy=False)
    Xp_a_np = Xp_a.to_numpy(dtype=float, copy=False)

    print("Started to fit PT subscription model (16+) using:", PT_MODEL)
    pt_model_adult.fit(Xs_a_np, y_a, sample_weight=w_a)
    print("Fitted PT subscription model (16+) using:", PT_MODEL)

    # -------------------------------------------------------------------
    # 8. PREDICT + STOCHASTIC DRAW (person-level in pop)
    # -------------------------------------------------------------------
    classes = np.array([0, 1, 2, 3, 4], dtype="int64")

    # youth predictions
    proba_y = pt_model_youth.predict_proba(Xp_y_np)
    draw_y = draw_multinomial_from_proba(proba_y, classes, seed=SEED_PT_YOUTH).astype("int64")
    hat_y  = classes[proba_y.argmax(axis=1)].astype("int64")

    # adult predictions
    proba_a = pt_model_adult.predict_proba(Xp_a_np)
    draw_a = draw_multinomial_from_proba(proba_a, classes, seed=SEED_PT).astype("int64")
    hat_a  = classes[proba_a.argmax(axis=1)].astype("int64")

    pop_df["PT_SUB_hat"] = 0
    pop_df["PT_SUB_draw"] = 0

    # assign back by segment
    pop_df.loc[pop_mask_youth, "PT_SUB_hat"]  = hat_y
    pop_df.loc[pop_mask_youth, "PT_SUB_draw"] = draw_y

    pop_df.loc[pop_mask_adult, "PT_SUB_hat"]  = hat_a
    pop_df.loc[pop_mask_adult, "PT_SUB_draw"] = draw_a

    # For ages < MIN_AGE keep 0
    pop_df.loc[pop_df["age"] < MIN_AGE, ["PT_SUB_hat", "PT_SUB_draw"]] = 0

    # -------------------------------------------------------------------
    # 9. DIAGNOSTICS (survey weighted vs pop modeled, person-level)
    # -------------------------------------------------------------------
    print("\n================== DIAGNOSTICS (Survey vs Modeled Pop, PT subscription) ==================")

    pop_ycol = "PT_SUB_draw" if USE_DRAW_DIAG else "PT_SUB_hat"
    classes_sorted = [0, 1, 2, 3, 4]

    # add a diagnostics age_group (mirrors your earlier style)
    diag_age_bins = [0, 6, 16, 26, 45, 60, 71, 81, 200]
    diag_age_labels = ["0-5", "6-15", "16-25", "26-44", "45-59", "60-70", "71-80", "81+"]

    survey_df["age_group"] = pd.cut(survey_df["age"], bins=diag_age_bins, labels=diag_age_labels, right=False)
    pop_df["age_group"] = pd.cut(pop_df["age"], bins=diag_age_bins, labels=diag_age_labels, right=False)

    # masks (same universe: age>=MIN_AGE)
    survey_mask = (
        (survey_df["age"] >= MIN_AGE) &
        survey_df[SURVEY_TARGET_COL].notna() &
        survey_df[SURVEY_TARGET_COL].between(0, 4, inclusive="both") &
        survey_df[SURVEY_WEIGHT_COL].notna() & (survey_df[SURVEY_WEIGHT_COL] > 0)
    )
    pop_mask = (pop_df["age"] >= MIN_AGE) & pop_df[pop_ycol].notna()

    def overall_dist_weighted(df, ycol, wcol):
        out = {}
        wv = df.loc[:, wcol].to_numpy(dtype=float)
        yv = df.loc[:, ycol].to_numpy(dtype=int)
        tot = np.sum(wv)
        for c in classes_sorted:
            out[c] = (np.sum(wv[yv == c]) / tot * 100.0) if tot > 0 else 0.0
        return pd.Series(out)

    def overall_dist_unweighted(df, ycol):
        return (df[ycol].value_counts(normalize=True).reindex(classes_sorted, fill_value=0.0) * 100.0)

    print("\n[OVERALL | Survey weighted % by class 0..4]")
    print(overall_dist_weighted(survey_df.loc[survey_mask, :], SURVEY_TARGET_COL, SURVEY_WEIGHT_COL).round(2).to_string())

    print("\n[OVERALL | Pop modeled % by class 0..4]")
    print(overall_dist_unweighted(pop_df.loc[pop_mask, :], pop_ycol).round(2).to_string())

    def compare_multiclass(group_col, canton_id=None, order=None):
        classes_sorted = [0, 1, 2, 3, 4]

        # helper: unique list preserving order
        def _uniq(cols):
            return list(dict.fromkeys(cols))

        # -------------------------
        # SURVEY weighted shares
        # -------------------------
        s_cols = _uniq([group_col, SURVEY_TARGET_COL, SURVEY_WEIGHT_COL, "canton_id"])
        s = survey_df.loc[survey_mask, s_cols].copy()

        # filter canton if requested (works even if group_col == "canton_id")
        if canton_id is not None:
            s["canton_id"] = s["canton_id"].astype(str).fillna("Missing")
            s = s[s["canton_id"] == str(canton_id)]

        s[group_col] = s[group_col].astype(str).fillna("Missing")

        s_mass = (
            s.groupby([group_col, SURVEY_TARGET_COL])[SURVEY_WEIGHT_COL]
            .sum()
            .rename("w")
            .reset_index()
        )
        s_tot = s.groupby(group_col)[SURVEY_WEIGHT_COL].sum().rename("w_tot").reset_index()
        s_sh = s_mass.merge(s_tot, on=group_col, how="left")
        s_sh["share"] = s_sh["w"] / s_sh["w_tot"]

        s_piv = s_sh.pivot(index=group_col, columns=SURVEY_TARGET_COL, values="share").fillna(0.0)
        s_piv = s_piv.reindex(columns=classes_sorted, fill_value=0.0)
        s_piv = (s_piv * 100.0).add_prefix("survey_pct_").reset_index()

        # -------------------------
        # POP shares (unweighted persons)
        # -------------------------
        p_cols = _uniq([group_col, pop_ycol, "canton_id"])
        p = pop_df.loc[pop_mask, p_cols].copy()

        if canton_id is not None:
            p["canton_id"] = p["canton_id"].astype(str).fillna("Missing")
            p = p[p["canton_id"] == str(canton_id)]

        p[group_col] = p[group_col].astype(str).fillna("Missing")

        p_cnt = p.groupby([group_col, pop_ycol]).size().rename("n").reset_index()
        p_tot = p.groupby(group_col).size().rename("n_tot").reset_index()
        p_sh = p_cnt.merge(p_tot, on=group_col, how="left")
        p_sh["share"] = p_sh["n"] / p_sh["n_tot"]

        p_piv = p_sh.pivot(index=group_col, columns=pop_ycol, values="share").fillna(0.0)
        p_piv = p_piv.reindex(columns=classes_sorted, fill_value=0.0)
        p_piv = (p_piv * 100.0).add_prefix("pop_pct_").reset_index()

        out = s_piv.merge(p_piv, on=group_col, how="outer").fillna(0.0)

        for c in classes_sorted:
            out[f"diff_{c}"] = out.get(f"pop_pct_{c}", 0.0) - out.get(f"survey_pct_{c}", 0.0)

        if order is not None:
            out[group_col] = pd.Categorical(out[group_col], categories=order, ordered=True)
            out = out.sort_values(group_col)

        return out


    # diagnostics groups (only print those that exist in both)
    diag_groups = [
        ("age_group", diag_age_labels),
        ("sex", None),
        ("driving_license", None),
        ("canton_id", None),
        ("municipality_type", None),
        ("ovgk", None),
        ("income_class", None),
        ("employment_status", None),
        ("is_swiss", None),
        ("HH_CAR_OWN_class", ["0", "1", "2", "3"]),
        ("commute_class", ["none", "short", "medium", "long"]),
    ]

    for g, order in diag_groups:
        if g in survey_df.columns and g in pop_df.columns:
            print(f"\n[{g.upper()} | ALL] survey vs pop (% by class)")
            print(compare_multiclass(g, canton_id=None, order=order).to_string(index=False))

            print(f"\n[{g.upper()} | canton_id={DIAG_CANTON_ID}] survey vs pop (% by class)")
            print(compare_multiclass(g, canton_id=DIAG_CANTON_ID, order=order).to_string(index=False))

    print("\n==========================================================================================")

    # =========================================================
    # ADD THIS DIAGNOSTIC BLOCK near the bottom of execute()
    # Put it AFTER your existing diagnostics loop
    # (i.e., after the for g, order in diag_groups loop)
    # and BEFORE keep_columns / return pop_df
    # =========================================================
    print("\n================== COMMUTE DIAGNOSTIC (WORK ONLY | subscription==1) ==================")

    pop_sub_col = "PT_SUB_draw" if USE_DRAW_DIAG else "PT_SUB_hat"

    required_s = ["commute_work_km", SURVEY_TARGET_COL, SURVEY_WEIGHT_COL]
    required_p = ["commute_work_km", pop_sub_col]

    missing_s = [c for c in required_s if c not in survey_df.columns]
    missing_p = [c for c in required_p if c not in pop_df.columns]

    if missing_s:
        print("Skipping commute diagnostic: missing in survey_df:", missing_s)
    elif missing_p:
        print("Skipping commute diagnostic: missing in pop_df:", missing_p)
    else:
        # --- build bins (km)
        bins = [COMMUTE_DIAG_THRESHOLD_KM, 1, 2, 5, 10, 15, 20, 30, 50, 100, np.inf]
        # keep only increasing and >= threshold
        bins = [b for b in bins if (np.isfinite(b) and b >= COMMUTE_DIAG_THRESHOLD_KM)]
        bins = sorted(set(bins))
        bins = bins + [np.inf]

        # if threshold is large and we collapsed bins too much, make a simple ladder
        if len(bins) < 4:
            t = COMMUTE_DIAG_THRESHOLD_KM
            bins = [t, t + 5, t + 10, t + 20, t + 40, np.inf]

        labels = []
        for i in range(len(bins) - 1):
            a, b = bins[i], bins[i + 1]
            labels.append(f"{a:g}+" if np.isinf(b) else f"{a:g}-{b:g}")

        # --- SURVEY: all persons with work commute > threshold
        s_work = pd.to_numeric(survey_df["commute_work_km"], errors="coerce")
        s_sub1 = (pd.to_numeric(survey_df[SURVEY_TARGET_COL], errors="coerce") == 1)
        s_w = pd.to_numeric(survey_df[SURVEY_WEIGHT_COL], errors="coerce")

        s_mask = (s_work > COMMUTE_DIAG_THRESHOLD_KM) & s_work.notna() & s_w.notna() & (s_w > 0)
        s_bin = pd.cut(s_work[s_mask], bins=bins, labels=labels, right=False, include_lowest=False)

        s_tmp = pd.DataFrame({
            "bin": s_bin.astype(str),
            "w": s_w[s_mask].values,
            "is_sub1": s_sub1[s_mask].astype(int).values
        })

        # weighted rate in bin = sum(w * is_sub1) / sum(w)
        s_grp = s_tmp.groupby("bin", as_index=False).apply(
            lambda g: pd.Series({
                "survey_den_w": g["w"].sum(),
                "survey_num_w": (g["w"] * g["is_sub1"]).sum()
            })
        ).reset_index(drop=True)
        s_grp["survey_p_sub1_pct"] = np.where(
            s_grp["survey_den_w"] > 0,
            s_grp["survey_num_w"] / s_grp["survey_den_w"] * 100.0,
            0.0
        )

        # --- POP: all persons with work commute > threshold
        p_work = pd.to_numeric(pop_df["commute_work_km"], errors="coerce")
        p_sub1 = (pd.to_numeric(pop_df[pop_sub_col], errors="coerce") == 1)

        p_mask = (p_work > COMMUTE_DIAG_THRESHOLD_KM) & p_work.notna()
        p_bin = pd.cut(p_work[p_mask], bins=bins, labels=labels, right=False, include_lowest=False)

        p_tmp = pd.DataFrame({
            "bin": p_bin.astype(str),
            "is_sub1": p_sub1[p_mask].astype(int).values
        })

        p_grp = p_tmp.groupby("bin", as_index=False).agg(
            pop_den_n=("is_sub1", "size"),
            pop_num_n=("is_sub1", "sum")
        )
        p_grp["pop_p_sub1_pct"] = np.where(
            p_grp["pop_den_n"] > 0,
            p_grp["pop_num_n"] / p_grp["pop_den_n"] * 100.0,
            0.0
        )

        # --- combine in bin order
        out = (
            pd.DataFrame({"bin": [str(l) for l in labels]})
            .merge(s_grp[["bin", "survey_p_sub1_pct", "survey_den_w"]], on="bin", how="left")
            .merge(p_grp[["bin", "pop_p_sub1_pct", "pop_den_n"]], on="bin", how="left")
            .fillna(0.0)
        )
        out["diff_pop_minus_survey"] = out["pop_p_sub1_pct"] - out["survey_p_sub1_pct"]

        print(f"\nFilter universe: work commute > {COMMUTE_DIAG_THRESHOLD_KM:g} km (ALL people, not conditioned on subscription)")
        print(f"Pop sub column used: {pop_sub_col}")

        print("\nP(subscription==1 | work-commute bin) [%]:")
        print(out.to_string(index=False, formatters={
            "survey_p_sub1_pct": "{:.2f}".format,
            "pop_p_sub1_pct": "{:.2f}".format,
            "diff_pop_minus_survey": "{:.2f}".format,
            "survey_den_w": "{:.2f}".format,
            "pop_den_n": "{:.0f}".format,
        }))

    print("\n==========================================================================================")
    keep_columns = ['person_id', 'household_id', 'sex', 'age', 'home_x', 'home_y', 'home_municipality_id', 'home_zone_id',
       'marital_status', 'household_size', 'municipality_type','canton_id','collective_housing_resident','nationality',
       'N_children_under_3', 'N_children_under_6', 'N_children_under_12',
       'N_children_under_18', 'employed', 'job_position',
       'is_student', 'ovgk', "sp_region",
       'driving_license', 'number_of_cars_class', 'car_availability', 'bike_availability', 'PT_SUB_draw', 'mz_person_id', 'is_car_passenger',
       'income_class']
    pop_df = pop_df[keep_columns]
    pop_df = pop_df.rename(columns={"PT_SUB_draw": "pt_subscription"})

    return pop_df
