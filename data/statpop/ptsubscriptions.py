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


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.household_persons")
    # population should already include HH_CAR_OWN_draw and DL model outputs
    context.stage("data.statpop.caravailability")


def execute(context):
    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------
    PT_MODEL = "catboost"      # "rf" or "gbm" or "catboost"
    SEED_PT  = 2026
    DIAG_CANTON_ID = "1"
    MIN_AGE = 18              
    USE_DRAW_DIAG = True

    # Survey columns (per your assumption)
    SURVEY_TARGET_COL = "subscription"        # values 0..4
    SURVEY_WEIGHT_COL = "person_weight"       # change if your weight column differs

    # Survey car ownership proxy (so survey and pop align on a "car ownership class" feature)
    SURVEY_CARCOUNT_COL = "number_of_cars_class"   # if present; used only to derive HH_CAR_OWN_class in survey

    # Population car ownership variable (per your note)
    POP_CAR_OWN_COL = "HH_CAR_OWN_draw"

    # -------------------------------------------------------------------
    # 0. LOAD DATA
    # -------------------------------------------------------------------
    survey_df = _as_df(context.stage("data.microcensus.persons")).copy()
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

    survey_hh_df = _as_df(context.stage("data.microcensus.household_persons")).copy()
    pop_df = _as_df(context.stage("data.statpop.caravailability")).copy()

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

    # common household-ish features that may exist in survey already; if missing, fill later
    if "N_children_under_18" in survey_df.columns:
        survey_df["presence_of_children_under_18"] = (pd.to_numeric(survey_df["N_children_under_18"], errors="coerce").fillna(0) > 0).astype(int)
    else:
        survey_df["presence_of_children_under_18"] = 0

    # Suggested additional useful vars (if present):
    # - income_class (strong)
    # - employment_status / is_student
    # - is_swiss
    # - sp_region / urbanity (municipality_type, ovgk)
    # - car ownership (HH_CAR_OWN_class) + possibly car_avail_draw
    # survey_df = survey_df[survey_df["age"]>17]
    # pop_df = pop_df[pop_df["age"]>17]
    # print(survey_df)
    # print(survey_df[survey_df["driving_license"]==0])
    # print(pop_df)
    # print(pop_df[pop_df["driving_license"]==0])
    # print(survey_df)
    # print(survey_df[survey_df["car_availability"]==0])
    # print(pop_df)
    # print(pop_df[pop_df["car_availability"]==0])
    # exit()
    # feature lists: keep robust (only use columns that exist in BOTH after we create them)
    candidate_cat = [
        "age_bin",
        "sex",
        "canton_id",
        "municipality_type",
        "ovgk",
        # "sp_region",
        # "marital_status",
        "employment_status",
        # "income_class",
        # "is_swiss",
        # "presence_of_children_under_18",
        # "HH_CAR_OWN_class",
        # "driving_license",  #TODO: this variable is not coded properly it seems
        "car_availability",
    ]

    candidate_num = [
        "age",
        # "age_sq",
        # "is_16_17",
        # "N_adults",
        # "N_children_under_18",
        # "N_drivers_license_per_adult",
        # "N_cars_per_adult",
        
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
    Xs = pd.get_dummies(survey_df.loc[survey_train_mask, cat_cols], drop_first=False)
    Xp = pd.get_dummies(pop_df.loc[pop_df["age"] >= MIN_AGE, cat_cols], drop_first=False)

    # add numeric
    if num_cols:
        Xs[num_cols] = survey_df.loc[survey_train_mask, num_cols].astype(float).values
        Xp[num_cols] = pop_df.loc[pop_df["age"] >= MIN_AGE, num_cols].astype(float).values

    # align
    Xp = Xp.reindex(columns=Xs.columns, fill_value=0.0)

    y = survey_df.loc[survey_train_mask, SURVEY_TARGET_COL].astype("int64").values
    w = survey_df.loc[survey_train_mask, SURVEY_WEIGHT_COL].astype(float).values

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

    pt_model = build_pt_model(PT_MODEL)

    Xs_np = Xs.to_numpy(dtype=float, copy=False)
    Xp_np = Xp.to_numpy(dtype=float, copy=False)
   
    print("Started to fit public transport subscription model using:", PT_MODEL)
    pt_model.fit(Xs_np, y, sample_weight=w)
    print("Fitted public transport subscription model using:", PT_MODEL)

    # -------------------------------------------------------------------
    # 8. PREDICT + STOCHASTIC DRAW (person-level in pop)
    # -------------------------------------------------------------------
    proba = pt_model.predict_proba(Xp_np)
    classes = getattr(pt_model, "classes_", np.array([0, 1, 2, 3, 4])).astype("int64")

    draw = draw_multinomial_from_proba(proba, classes, seed=SEED_PT).astype("int64")
    hat  = classes[proba.argmax(axis=1)].astype("int64")

    pop_df["PT_SUB_hat"] = 0
    pop_df["PT_SUB_draw"] = 0
    pop_mask_pred = pop_df["age"] >= MIN_AGE
    pop_df.loc[pop_mask_pred, "PT_SUB_hat"]  = hat
    pop_df.loc[pop_mask_pred, "PT_SUB_draw"] = draw

    # For ages < MIN_AGE you can keep 0, or set Missing — here we keep 0
    pop_df.loc[pop_df["age"] < MIN_AGE, ["PT_SUB_hat", "PT_SUB_draw"]] = 0

    # -------------------------------------------------------------------
    # 9. DIAGNOSTICS (survey weighted vs pop modeled, person-level)
    # -------------------------------------------------------------------
    print("\n================== DIAGNOSTICS (Survey vs Modeled Pop, PT subscription) ==================")

    pop_ycol = "PT_SUB_draw" if USE_DRAW_DIAG else "PT_SUB_hat"
    classes_sorted = [0, 1, 2, 3, 4]

    # add a diagnostics age_group (mirrors your earlier style)
    diag_age_bins = [0, 16, 18, 26, 45, 60, 71, 81, 200]
    diag_age_labels = ["0-15", "16-17", "18-25", "26-44", "45-59", "60-70", "71-80", "81+"]

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
    ]

    for g, order in diag_groups:
        if g in survey_df.columns and g in pop_df.columns:
            print(f"\n[{g.upper()} | ALL] survey vs pop (% by class)")
            print(compare_multiclass(g, canton_id=None, order=order).to_string(index=False))

            print(f"\n[{g.upper()} | canton_id={DIAG_CANTON_ID}] survey vs pop (% by class)")
            print(compare_multiclass(g, canton_id=DIAG_CANTON_ID, order=order).to_string(index=False))

    print("\n==========================================================================================")

    return pop_df
