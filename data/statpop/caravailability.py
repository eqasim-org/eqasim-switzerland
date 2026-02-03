import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier


# ---------------------------------------------------------
# helper: stochastic draw from class probabilities (binary ok)
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


def _resolve_col(df, candidates, name_for_error):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find {name_for_error}. Tried: {candidates}. "
        f"Available columns sample: {list(df.columns)[:80]}"
    )


def _to_dl_has(series):
    """Robust conversion to {0,1} for driving license flags."""
    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce").fillna(0)
        return (s > 0).astype("int64")
    return series.astype("boolean").fillna(False).astype("int64")


def configure(context):
    # Survey:
    context.stage("data.microcensus.21.persons")             # 1 row per household, with features + car_availability
    context.stage("data.microcensus.21.household_persons")   # multiple rows per household, with driving_license per member

    # Population:
    context.stage("data.statpop.carownership")               # person-level population incl. HH_CAR_OWN_draw (cars)


def execute(context):
    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------
    CA_MODEL = "catboost"   # "rf" or "gbm" or "catboost"
    SEED_CA  = 2026
    DIAG_CANTON_ID = "25"
    USE_DRAW = True  # diagnostics compare on draw vs hat

    # -------------------------------------------------------------------
    # 0. LOAD DATA
    # -------------------------------------------------------------------
    persons_df = context.stage("data.microcensus.21.persons").copy()
    var_raw = pd.to_numeric(persons_df["car_availability"], errors="coerce")
    persons_df["car_availability"] = np.where(var_raw == 2, 0, 1).astype("int64")
    hh_persons_df = context.stage("data.microcensus.21.household_persons")[0].copy()
    pop_df = context.stage("data.statpop.carownership").copy()
    print(pop_df)
    print(pop_df[pop_df["age"]>15])
    exit()
    

    # -------------------------------------------------------------------
    # 1. KEY COLUMNS & REQUIRED FIELDS
    # -------------------------------------------------------------------
    # Survey persons has person_id and it is the household identifier for household_persons.household_id
    PERSON_ID_COL = _resolve_col(persons_df, ["person_id"], "person_id (persons_df)")
    HHP_HHID_COL  = _resolve_col(hh_persons_df, ["household_id"], "household_id (household_persons_df)")

    # Population household id (for aggregating DL counts + cars max)
    POP_HHID_COL  = _resolve_col(pop_df, ["household_id", "hh_id", "id_household"], "household_id (pop_df)")

    # Survey outcome name
    CAR_AV_COL = _resolve_col(persons_df, ["car_availability"], "car_availability outcome (persons_df)")

    # Cars variable names per your note:
    # - survey persons: number_of_cars_class
    # - pop: HH_CAR_OWN_draw
    CARS_COL_PERS = _resolve_col(persons_df, ["number_of_cars_class"], "number_of_cars_class (persons_df)")
    CARS_COL_POP  = _resolve_col(pop_df, ["HH_CAR_OWN_draw"], "HH_CAR_OWN_draw (pop_df)")

    # Survey weights
    if "person_weight" not in persons_df.columns:
        raise KeyError("persons_df must contain 'person_weight' for weighted training/diagnostics.")

    # Household-persons must contain driving_license
    DL_COL_HHP = _resolve_col(hh_persons_df, ["driving_license"], "driving_license (household_persons_df)")

    # Population DL indicator (prefer your modeled draw if present; else fall back)
    DL_POP_COL = (
        "DL_has_or_learning_draw" if "DL_has_or_learning_draw" in pop_df.columns else
        "DL_has_or_learning_hat"  if "DL_has_or_learning_hat"  in pop_df.columns else
        ("driving_license" if "driving_license" in pop_df.columns else
         _resolve_col(pop_df, ["dl_has", "DL_has_or_learning"], "DL indicator (pop_df)"))
    )

    # -------------------------------------------------------------------
    # 2. BASIC CLEANING
    # -------------------------------------------------------------------
    persons_df["age"] = pd.to_numeric(persons_df.get("age"), errors="coerce")
    pop_df["age"]     = pd.to_numeric(pop_df.get("age"), errors="coerce")

    persons_df = persons_df[persons_df["age"].notna()].copy()
    pop_df     = pop_df[pop_df["age"].notna()].copy()

    # transform to children presence
    if "N_children_under_18" in persons_df.columns:
        persons_df["N_children_under_18"] = persons_df["N_children_under_18"] > 0
        pop_df["N_children_under_18"] = pop_df["N_children_under_18"] > 0

    # -------------------------------------------------------------------
    # 3. BUILD HH-LEVEL DL COUNTS FROM household_persons_df
    #     household_persons.household_id == persons.person_id (your setup)
    # -------------------------------------------------------------------
    hh_persons_df = hh_persons_df.copy()
    hh_persons_df["dl_has"] = _to_dl_has(hh_persons_df[DL_COL_HHP])

    hh_n_dl = (
        hh_persons_df
        .groupby(HHP_HHID_COL)["dl_has"]
        .sum()
        .rename("hh_n_dl")
        .astype("int64")
        .reset_index()
    )

    # Merge hh_n_dl onto persons_df using persons.person_id == hh_persons.household_id
    persons_df = persons_df.merge(
        hh_n_dl,
        left_on=PERSON_ID_COL,
        right_on=HHP_HHID_COL,
        how="left"
    )
    if HHP_HHID_COL in persons_df.columns and HHP_HHID_COL != PERSON_ID_COL:
        persons_df = persons_df.drop(columns=[HHP_HHID_COL])

    persons_df["hh_n_dl"] = pd.to_numeric(persons_df["hh_n_dl"], errors="coerce").fillna(0).astype("int64")

    # -------------------------------------------------------------------
    # 4. CARS COUNTS (survey + pop) AND SCARCITY FEATURES
    # -------------------------------------------------------------------
    # Survey cars: number_of_cars_class (could be class-like; coerce to numeric if possible)
    persons_df[CARS_COL_PERS] = pd.to_numeric(persons_df[CARS_COL_PERS], errors="coerce")
    persons_df[CARS_COL_PERS] = persons_df[CARS_COL_PERS].fillna(0.0).clip(lower=0.0)
    persons_df["hh_n_cars"] = persons_df[CARS_COL_PERS].astype(float)

    persons_df["cars_per_dl"] = np.where(
        persons_df["hh_n_dl"].to_numpy(dtype=float) > 0,
        persons_df["hh_n_cars"].to_numpy(dtype=float) / persons_df["hh_n_dl"].to_numpy(dtype=float),
        0.0
    )
    persons_df["cars_shortage"] = (persons_df["hh_n_dl"] - persons_df["hh_n_cars"]).clip(lower=0).astype(float)

    # Population cars: HH_CAR_OWN_draw (already numeric draw)
    pop_df[CARS_COL_POP] = pd.to_numeric(pop_df[CARS_COL_POP], errors="coerce").fillna(0.0).clip(lower=0.0)

    pop_df[DL_POP_COL] = pd.to_numeric(pop_df[DL_POP_COL], errors="coerce").fillna(0).astype("int64")

    # HH aggregates in pop
    pop_df["hh_n_dl"] = (
        pop_df[DL_POP_COL]
        .groupby(pop_df[POP_HHID_COL])
        .transform("sum")
        .astype("int64")
    )
    pop_df["hh_n_cars"] = (
        pop_df[CARS_COL_POP]
        .groupby(pop_df[POP_HHID_COL])
        .transform("max")
        .astype(float)
    )

    pop_df["cars_per_dl"] = np.where(
        pop_df["hh_n_dl"].to_numpy(dtype=float) > 0,
        pop_df["hh_n_cars"].to_numpy(dtype=float) / pop_df["hh_n_dl"].to_numpy(dtype=float),
        0.0
    )
    pop_df["cars_shortage"] = (pop_df["hh_n_dl"] - pop_df["hh_n_cars"]).clip(lower=0).astype(float)

    # -------------------------------------------------------------------
    # 5. SURVEY: FOCAL PERSON DL FLAG (needed for rules + training mask)
    #     Since persons_df is 1 row per household, we infer focal DL:
    #     - if persons_df already has driving_license -> use it
    #     - else: assume focal has DL if hh_n_dl>0 (weak fallback; but avoids crash)
    # -------------------------------------------------------------------
    if "driving_license" in persons_df.columns:
        persons_df["dl_has_focal"] = _to_dl_has(persons_df["driving_license"])
    else:
        # fallback (NOT ideal, but consistent with your schema description where persons lacks member rows)
        persons_df["dl_has_focal"] = (persons_df["hh_n_dl"] > 0).astype("int64")

    # -------------------------------------------------------------------
    # 6. ENFORCE DETERMINISTIC RULES IN SURVEY OUTCOME
    # -------------------------------------------------------------------
    persons_df[CAR_AV_COL] = pd.to_numeric(persons_df[CAR_AV_COL], errors="coerce").astype("Int64")

    # rule: no DL => 0
    persons_df.loc[persons_df["dl_has_focal"] == 0, CAR_AV_COL] = 0

    # rule: if cars >= #DL in hh => focal (DL holder) has availability
    det_s = (persons_df["dl_has_focal"] == 1) & (persons_df["hh_n_cars"] >= persons_df["hh_n_dl"])
    persons_df.loc[det_s, CAR_AV_COL] = 1

    # scarce households (for training)
    scarce_s = (persons_df["dl_has_focal"] == 1) & (persons_df["hh_n_cars"] < persons_df["hh_n_dl"])

    # -------------------------------------------------------------------
    # 7. POPULATION: INITIALIZE OUTPUTS + DETERMINISTIC RULES
    # -------------------------------------------------------------------
    pop_df["car_avail_hat"] = 0
    pop_df["car_avail_draw"] = 0

    # deterministic: no DL => 0
    no_dl_p = pop_df[DL_POP_COL] == 0
    pop_df.loc[no_dl_p, ["car_avail_hat", "car_avail_draw"]] = 0

    # deterministic: cars enough => 1
    det_p = (pop_df[DL_POP_COL] == 1) & (pop_df["hh_n_cars"] >= pop_df["hh_n_dl"])
    pop_df.loc[det_p, ["car_avail_hat", "car_avail_draw"]] = 1

    scarce_p = (pop_df[DL_POP_COL] == 1) & (pop_df["hh_n_cars"] < pop_df["hh_n_dl"])

    # -------------------------------------------------------------------
    # 8. FEATURES (same structure as your DL code + shortage features)
    # -------------------------------------------------------------------
    age_bins = [0, 18, 21, 26, 45, 60, 71, 81, 200]
    age_labels = ["0-17", "18-20", "21-25", "25-44", "45-59", "60-70", "71-80", "81+"]

    for df in (persons_df, pop_df):
        df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
        df["age_sq"] = df["age"] ** 2

    # categoricals as strings
    cat_cols = ["age_bin", "sex", "canton_id", "municipality_type", "sp_region", "marital_status", "employment_status"]
    for df in (persons_df, pop_df):
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("Missing")

    # numeric safety casting
    for df in (persons_df, pop_df):
        for col in ["household_size", "N_adults", "N_children_under_18"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    feature_cols = [
        "age",
        "age_sq",
        "age_bin",
        "sex",
        "canton_id",
        "municipality_type",
        #"N_children_under_18",
        "marital_status",
        "N_adults",
        # shortage features
        "hh_n_cars",
        "hh_n_dl",
        "cars_per_dl",
        "cars_shortage",
    ]

    num_cols = [
        "age", "age_sq", "household_size", "N_adults", "N_children_under_18",
        "hh_n_cars", "hh_n_dl", "cars_per_dl", "cars_shortage",
    ]

    # Train on adults, scarce households, nonmissing outcome and weights
    train_s = (
        (persons_df["age"] >= 18) &
        scarce_s &
        persons_df[CAR_AV_COL].notna() &
        persons_df["person_weight"].notna()
    )

    X_survey = pd.get_dummies(persons_df.loc[train_s, feature_cols], drop_first=False)
    X_pop = pd.get_dummies(pop_df.loc[scarce_p, feature_cols], drop_first=False)

    # numeric block overwrite
    for col in num_cols:
        if col in persons_df.columns:
            X_survey[col] = persons_df.loc[train_s, col].astype(float).values
        if col in pop_df.columns and X_pop.shape[0] > 0:
            X_pop[col] = pop_df.loc[scarce_p, col].astype(float).values

    global_feature_cols = X_survey.columns
    X_pop = X_pop.reindex(columns=global_feature_cols, fill_value=0)

    y = persons_df.loc[train_s, CAR_AV_COL].astype("int64")
    w = persons_df.loc[train_s, "person_weight"].astype(float)

    # -------------------------------------------------------------------
    # 9. FIT MODEL
    # -------------------------------------------------------------------
    def build_ca_model(model_type: str):
        mt = str(model_type).lower().strip()
        if mt == "gbm":
            return HistGradientBoostingClassifier(
                loss="log_loss",
                max_depth=8,
                learning_rate=0.05,
                max_iter=400,
                min_samples_leaf=50,
                random_state=42
            )
        elif mt == "rf":
            return RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif mt in ("catboost", "cat"):
            return CatBoostClassifier(
                loss_function="Logloss",
                iterations=1500,
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

    ca_model = build_ca_model(CA_MODEL)

    Xs = X_survey.to_numpy(dtype=float, copy=False)
    Xp = X_pop.to_numpy(dtype=float, copy=False)

    if (len(y) > 0) and (y.nunique() > 1) and (Xp.shape[0] > 0):
        ca_model.fit(Xs, y, sample_weight=w)
        print("Fitted car availability model (scarce households) using:", CA_MODEL)

        proba = ca_model.predict_proba(Xp)
        classes = getattr(ca_model, "classes_", np.array([0, 1])).astype("int64")

        draw = draw_multinomial_from_proba(proba, classes, seed=SEED_CA).astype("int64")
        hat = classes[proba.argmax(axis=1)].astype("int64")

        pop_df.loc[scarce_p, "car_avail_hat"] = hat
        pop_df.loc[scarce_p, "car_avail_draw"] = draw
    else:
        print(
            "Skipped car availability model fit: not enough training signal "
            "(no scarce DL cases in persons_df, constant outcome, or no scarce cases in pop)."
        )
        pop_df.loc[scarce_p, ["car_avail_hat", "car_avail_draw"]] = 0

    # Under-18: force 0
    under18 = pop_df["age"].notna() & (pop_df["age"] < 18)
    pop_df.loc[under18, ["car_avail_hat", "car_avail_draw"]] = 0

    # -------------------------------------------------------------------
    # 10. DIAGNOSTICS (Survey weighted % vs Pop modeled %)
    # -------------------------------------------------------------------
    print("\n================== DIAGNOSTICS (Survey vs Modeled Pop) ==================")
    pop_ycol = "car_avail_draw" if USE_DRAW else "car_avail_hat"

    # ensure key cols are string typed
    for col in ["canton_id", "sex", "municipality_type", "income_class", "marital_status", "sp_region"]:
        if col in persons_df.columns:
            persons_df[col] = persons_df[col].astype(str).fillna("Missing")
        if col in pop_df.columns:
            pop_df[col] = pop_df[col].astype(str).fillna("Missing")

    # diagnostics age groups
    diag_age_bins = [0, 18, 22, 26, 45, 60, 71, 81, 200]
    diag_age_labels = ["0-17", "18-21", "22-25", "25-44", "45-59", "60-70", "71-80", "81+"]

    persons_df["age_group"] = pd.cut(
        pd.to_numeric(persons_df["age"], errors="coerce"),
        bins=diag_age_bins, labels=diag_age_labels, right=False
    )
    pop_df["age_group"] = pd.cut(
        pd.to_numeric(pop_df["age"], errors="coerce"),
        bins=diag_age_bins, labels=diag_age_labels, right=False
    )

    # exact children (string)
    def _children_exact_str(s):
        s = pd.to_numeric(s, errors="coerce").astype("Int64")
        return s.astype("string").fillna("Missing")

    persons_df["N_children_under_18_exact"] = (
        _children_exact_str(persons_df["N_children_under_18"]) if "N_children_under_18" in persons_df.columns else "Missing"
    )
    pop_df["N_children_under_18_exact"] = (
        _children_exact_str(pop_df["N_children_under_18"]) if "N_children_under_18" in pop_df.columns else "Missing"
    )

    # survey/pop masks (same universe: age>=18)
    survey_mask = (
        (persons_df["age"] >= 18) &
        persons_df[CAR_AV_COL].notna() &
        persons_df["person_weight"].notna()
    )
    pop_mask = (
        (pop_df["age"] >= 18) &
        pop_df[pop_ycol].notna()
    )

    overall_s = 100.0 * _weighted_mean(
        persons_df.loc[survey_mask, CAR_AV_COL],
        persons_df.loc[survey_mask, "person_weight"]
    ) if survey_mask.sum() > 0 else np.nan

    overall_p = 100.0 * pop_df.loc[pop_mask, pop_ycol].mean() if pop_mask.sum() > 0 else np.nan

    print(
        f"\n[OVERALL | age>=18] Survey weighted % car_avail=1: {overall_s:.2f} | "
        f"Pop modeled % car_avail=1: {overall_p:.2f} | Diff(pop-survey): {(overall_p - overall_s):.2f}"
    )

    def compare_pct(group_col, canton_id=None, order=None):
        """
        Returns:
        group_col | survey_weighted_pct_has | pop_pct_has | diff | abs_diff
        """
        # survey
        s_cols = [group_col, CAR_AV_COL, "person_weight", "canton_id"]
        s_cols = [c for c in s_cols if c in persons_df.columns]
        s = persons_df.loc[survey_mask, s_cols].copy()
        if canton_id is not None and "canton_id" in s.columns:
            s = s[s["canton_id"] == str(canton_id)]

        if len(s) == 0:
            survey_tab = pd.DataFrame(columns=[group_col, "survey_weighted_pct_has"])
        else:
            survey_tab = (
                s.groupby(group_col, dropna=False)
                 .apply(lambda g: 100.0 * _weighted_mean(g[CAR_AV_COL], g["person_weight"]))
                 .rename("survey_weighted_pct_has")
                 .reset_index()
            )

        # pop
        p_cols = [group_col, pop_ycol, "canton_id"]
        p_cols = [c for c in p_cols if c in pop_df.columns]
        p = pop_df.loc[pop_mask, p_cols].copy()
        if canton_id is not None and "canton_id" in p.columns:
            p = p[p["canton_id"] == str(canton_id)]

        if len(p) == 0:
            pop_tab = pd.DataFrame(columns=[group_col, "pop_pct_has"])
        else:
            pop_tab = (
                p.groupby(group_col, dropna=False)[pop_ycol]
                 .mean()
                 .mul(100.0)
                 .rename("pop_pct_has")
                 .reset_index()
            )

        out = pd.merge(survey_tab, pop_tab, on=group_col, how="outer")

        if order is not None:
            out[group_col] = pd.Categorical(out[group_col], categories=order, ordered=True)
            out = out.sort_values(group_col)

        out["survey_weighted_pct_has"] = pd.to_numeric(out["survey_weighted_pct_has"], errors="coerce")
        out["pop_pct_has"] = pd.to_numeric(out["pop_pct_has"], errors="coerce")
        out["diff_pop_minus_survey"] = out["pop_pct_has"] - out["survey_weighted_pct_has"]
        out["abs_diff"] = out["diff_pop_minus_survey"].abs()
        return out

    # AGE GROUP
    age_all = compare_pct("age_group", order=diag_age_labels)
    print("\n[AGE GROUP | ALL] age_group | survey_weighted_pct_has | pop_pct_has")
    print(age_all.to_string(index=False))

    age_c = compare_pct("age_group", canton_id=DIAG_CANTON_ID, order=diag_age_labels)
    print(f"\n[AGE GROUP | canton_id={DIAG_CANTON_ID}] age_group | survey_weighted_pct_has | pop_pct_has")
    print(age_c.to_string(index=False))

    # SEX
    if "sex" in persons_df.columns and "sex" in pop_df.columns:
        sex_all = compare_pct("sex")
        print("\n[SEX | ALL] sex | survey_weighted_pct_has | pop_pct_has")
        print(sex_all.to_string(index=False))

        sex_c = compare_pct("sex", canton_id=DIAG_CANTON_ID)
        print(f"\n[SEX | canton_id={DIAG_CANTON_ID}] sex | survey_weighted_pct_has | pop_pct_has")
        print(sex_c.to_string(index=False))

    # MUNICIPALITY TYPE
    if "municipality_type" in persons_df.columns and "municipality_type" in pop_df.columns:
        mun_all = compare_pct("municipality_type")
        print("\n[MUNICIPALITY TYPE | ALL] municipality_type | survey_weighted_pct_has | pop_pct_has")
        print(mun_all.to_string(index=False))

        mun_c = compare_pct("municipality_type", canton_id=DIAG_CANTON_ID)
        print(f"\n[MUNICIPALITY TYPE | canton_id={DIAG_CANTON_ID}] municipality_type | survey_weighted_pct_has | pop_pct_has")
        print(mun_c.to_string(index=False))

    # N_CHILDREN_UNDER_18 exact
    child_all = compare_pct("N_children_under_18_exact")
    if "N_children_under_18_exact" in child_all.columns:
        def _sort_key(v):
            try:
                return (0, int(v))
            except Exception:
                return (1, 10**9)
        child_all = child_all.sort_values(
            by="N_children_under_18_exact",
            key=lambda s: s.map(_sort_key)
        )
    print("\n[N_CHILDREN_UNDER_18 (exact) | ALL] N_children_under_18_exact | survey_weighted_pct_has | pop_pct_has")
    print(child_all.to_string(index=False))

    child_c = compare_pct("N_children_under_18_exact", canton_id=DIAG_CANTON_ID)
    if "N_children_under_18_exact" in child_c.columns:
        child_c = child_c.sort_values(
            by="N_children_under_18_exact",
            key=lambda s: s.map(_sort_key)
        )
    print(f"\n[N_CHILDREN_UNDER_18 (exact) | canton_id={DIAG_CANTON_ID}] N_children_under_18_exact | survey_weighted_pct_has | pop_pct_has")
    print(child_c.to_string(index=False))

    # Shortage bucket diagnostics
    pop_df["cars_shortage_bucket"] = pd.cut(
        pop_df["cars_shortage"].astype(float),
        bins=[-0.1, 0.0, 1.0, 2.0, 5.0, 9999],
        labels=["0", "1", "2", "3-5", "6+"],
        right=True
    ).astype(str).fillna("Missing")

    persons_df["cars_shortage_bucket"] = pd.cut(
        persons_df["cars_shortage"].astype(float),
        bins=[-0.1, 0.0, 1.0, 2.0, 5.0, 9999],
        labels=["0", "1", "2", "3-5", "6+"],
        right=True
    ).astype(str).fillna("Missing")

    shortage_all = compare_pct("cars_shortage_bucket", order=["0", "1", "2", "3-5", "6+", "Missing"])
    print("\n[CARS SHORTAGE BUCKET | ALL] cars_shortage_bucket | survey_weighted_pct_has | pop_pct_has")
    print(shortage_all.to_string(index=False))

    shortage_c = compare_pct("cars_shortage_bucket", canton_id=DIAG_CANTON_ID, order=["0", "1", "2", "3-5", "6+", "Missing"])
    print(f"\n[CARS SHORTAGE BUCKET | canton_id={DIAG_CANTON_ID}] cars_shortage_bucket | survey_weighted_pct_has | pop_pct_has")
    print(shortage_c.to_string(index=False))

    print("\n==========================================================================")


    return pop_df
