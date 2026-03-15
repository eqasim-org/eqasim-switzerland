import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
import logging

logger = logging.getLogger("synpp")
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


def configure(context):
    context.stage("data.microcensus.21.persons")
    context.stage("synthesis.population.models.students")

def execute(context):
    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------
    DL_MODEL = "catboost"         # "rf" or "gbm" or "catboost"
    SEED_DL  = 2026
    DIAG_CANTON_ID = "25"    # default canton selection for diagnostics (string after preprocessing)

    # -------------------------------------------------------------------
    # 0. LOAD DATA
    # -------------------------------------------------------------------
    survey_df = context.stage("data.microcensus.21.persons")
    pop_df    = context.stage("synthesis.population.models.students")

    #survey_df = survey_df[survey_df["income_imputed"]== False] #keep only those that do not have imputed income

    for df in (survey_df, pop_df):
        df["N_children_under_18"] = pd.to_numeric(df["N_children_under_18"], errors="coerce")
        df["N_children_under_18"] = (df["N_children_under_18"].fillna(0) > 0).astype(int)
    
    pop_df["is_swiss"] = pop_df["nationality"] == 0

    # N_adults in population (>=18)
    adult_mask = pop_df["age"] >= 18
    pop_df["N_adults"] = (
        adult_mask.astype(int)
        .groupby(pop_df["household_id"])
        .transform("sum")
    )

    # -------------------------------------------------------------------
    # 1. PREP SURVEY: outcome from driving_license (car) + learning_driving_license (learning)
    #    Rule: model only age >= 18
    # -------------------------------------------------------------------
    survey_df = survey_df.copy()
    pop_df    = pop_df.copy()

    # (keeping your employment_status construction in pop, in case used later)
    pop_df.loc[:, "employment_status"] = 0
    pop_df.loc[pop_df["employed"] == 1, "employment_status"] = 1
    pop_df.loc[(pop_df["employed"] == 3) & (pop_df["is_student"] == 1), "employment_status"] = 2
    pop_df.loc[(pop_df["employed"] == 2) & (pop_df["is_student"] == 1), "employment_status"] = 2
    pop_df.loc[(pop_df["employed"] == 1) & (pop_df["is_student"] == 1), "employment_status"] = 3

    needed_survey = [
        "age", "sex", "household_size", "employment_status", "N_adults",
        "canton_id", "income_class", "person_weight",
        "driving_license", "learning_driving_license", "is_swiss",
        "N_children_under_18", "municipality_type", "sp_region", "marital_status", "ovgk",
    ]
    survey_df = survey_df.dropna(subset=[c for c in needed_survey if c in survey_df.columns])

    survey_df["age"] = pd.to_numeric(survey_df["age"], errors="coerce")
    survey_df = survey_df[survey_df["age"].notna()]
    survey_df = survey_df[survey_df["age"] >= 18].copy()

    for col in ["driving_license", "learning_driving_license"]:
        survey_df[col] = pd.to_numeric(survey_df[col], errors="coerce").astype("Int64")

    a_yes = survey_df["driving_license"] == True
    a_no  = survey_df["driving_license"] == False

    survey_df["dl_has_or_learning"] = pd.NA
    survey_df.loc[a_yes, "dl_has_or_learning"] = 1
    survey_df.loc[a_no,  "dl_has_or_learning"] = 0
    survey_df["dl_has_or_learning"] = survey_df["dl_has_or_learning"].astype("Int64")
    survey_df = survey_df.dropna(subset=["dl_has_or_learning"]).copy()

    # income_class cleaning
    survey_df["income_class"] = pd.to_numeric(survey_df["income_class"], errors="coerce").astype("Int64")
    pop_df["income_class"] = pd.to_numeric(pop_df["income_class"], errors="coerce").astype("Int64")
    survey_df = survey_df.dropna(subset=["income_class"])
    survey_df = survey_df[survey_df["income_class"] >= 0].copy()

    # recode ovgk
    survey_df["ovgk_grouped"] = survey_df["ovgk"].isin(["A","B","C"]).map({True: "ABC", False: "D_or_None"})
    pop_df["ovgk_grouped"] = pop_df["ovgk"].isin(["A","B","C"]).map({True: "ABC", False: "D_or_None"})

    # -------------------------------------------------------------------
    # 2. PREP POP: ensure income_class exists
    # -------------------------------------------------------------------
    pop_df["age"] = pd.to_numeric(pop_df["age"], errors="coerce")
    if "income_class" not in pop_df.columns:
        pop_df["income_class"] = pd.NA

    # -------------------------------------------------------------------
    # 3. AGE BINS + CATEGORICAL CLEANING
    # -------------------------------------------------------------------
    age_bins = [0, 18, 21, 26, 45, 60, 71, 81, 200]
    age_labels = [
        "0-17", "18-20", "21-25",
        "25-44", "45-59", "60-70", "71-80", "81+"
    ]

    for df in (survey_df, pop_df):
        df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)

    cat_cols = [
        "age_bin", "sex", "canton_id",
        "municipality_type", "marital_status",  "ovgk_grouped", "employment_status"
    ]
    num_cols = ["age", "age_sq", "household_size", "N_adults", "N_children_under_18"]

    for df in (survey_df, pop_df):
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna("Missing")

    for df in (survey_df, pop_df):
        df["household_size"] = df["household_size"].astype(float)
        df["N_adults"] = df["N_adults"].astype(float)
        df["age_sq"] = df["age"] ** 2

    # -------------------------------------------------------------------
    # 4. DESIGN MATRICES
    # -------------------------------------------------------------------

    X_survey = pd.get_dummies(survey_df[cat_cols], drop_first=False)
    X_pop    = pd.get_dummies(pop_df[cat_cols], drop_first=False)

    # add numeric columns
    X_survey[num_cols] = survey_df[num_cols].astype(float).values
    X_pop[num_cols]    = pop_df[num_cols].astype(float).values

    global_feature_cols = X_survey.columns
    X_pop = X_pop.reindex(columns=global_feature_cols, fill_value=0)

    y = survey_df["dl_has_or_learning"].astype("int64")
    sample_weight = survey_df["person_weight"].astype(float)

    # -------------------------------------------------------------------
    # 5. FIT MODEL
    # -------------------------------------------------------------------
    def build_dl_model(model_type: str):
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
                iterations=2500,
                learning_rate=0.03,
                depth=10,
                l2_leaf_reg=6.0,
                random_seed=42,
                verbose=False,
                bootstrap_type="Bernoulli",
                subsample=0.8
            )
        else:
            raise ValueError(f"Unknown model_type={model_type}, use 'gbm', 'rf', or 'catboost'.")

    dl_model = build_dl_model(DL_MODEL)

    # CatBoost / sklearn models all handle numpy arrays; standardize dtype
    Xs = X_survey.to_numpy(dtype=float, copy=False)
    Xp = X_pop.to_numpy(dtype=float, copy=False)

    dl_model.fit(Xs, y, sample_weight=sample_weight)
    logger.info("Fitted driver's license (has OR learning) model using: %s", DL_MODEL)

    # -------------------------------------------------------------------
    # 6. PREDICT + STOCHASTIC DRAW (then enforce under-18 rule)
    # -------------------------------------------------------------------
    proba_pop = dl_model.predict_proba(Xp)

    # Most sklearn-style classifiers expose .classes_
    classes = getattr(dl_model, "classes_", np.array([0, 1])).astype("int64")

    draw = draw_multinomial_from_proba(proba_pop, classes, seed=SEED_DL).astype("int64")
    hat  = classes[proba_pop.argmax(axis=1)].astype("int64")

    pop_df["DL_has_or_learning_hat"]  = hat
    pop_df["DL_has_or_learning_draw"] = draw

    under18 = pop_df["age"].notna() & (pop_df["age"] < 18)
    pop_df.loc[under18, "DL_has_or_learning_hat"]  = 0
    pop_df.loc[under18, "DL_has_or_learning_draw"] = 0

    # -------------------------------------------------------------------
    # 7. DIAGNOSTICS: Survey (weighted %) vs Pop (modeled %) for:
    #    - age_group
    #    - sex
    #    - income_class
    #    - municipality_type  
    #    - N_children_under_18 (exact)
    #
    # IMPORTANT: Comparisons are done on the SAME universe: age >= 18
    # -------------------------------------------------------------------
    logger.info("\n================== DIAGNOSTICS (Survey vs Modeled Pop) ==================")

    USE_DRAW = True
    pop_ycol = "DL_has_or_learning_draw" if USE_DRAW else "DL_has_or_learning_hat"

    # Ensure string typing for merge/group keys
    survey_df["canton_id"] = survey_df["canton_id"].astype(str)#.fillna("Missing")
    pop_df["canton_id"]    = pop_df["canton_id"].astype(str)#.fillna("Missing")

    survey_df["sex"] = survey_df["sex"].astype(str)#.fillna("Missing")
    pop_df["sex"]    = pop_df["sex"].astype(str)#.fillna("Missing")

    survey_df["income_class"] = survey_df["income_class"].astype(str)#.fillna("Missing")
    pop_df["income_class"]    = pop_df["income_class"].astype(str)#.fillna("Missing")

    # municipality_type already cast to str in cat_cols above, but keep safe
    survey_df["municipality_type"] = survey_df["municipality_type"].astype(str)#.fillna("Missing")
    pop_df["municipality_type"]    = pop_df["municipality_type"].astype(str)#.fillna("Missing")

    #  N_children_under_18 as a string key for grouping (0,1,2,3,... + Missing)
    def _children_exact_str(s):
        s = pd.to_numeric(s, errors="coerce").astype("Int64")
        # to string but keep missing as "Missing"
        out = s.astype("string")
        #out = out.fillna("Missing")
        return out

    survey_df["N_children_under_18_exact"] = _children_exact_str(survey_df["N_children_under_18"])
    pop_df["N_children_under_18_exact"]    = _children_exact_str(pop_df["N_children_under_18"])

    # Age groups for diagnostics
    diag_age_bins = [0, 18, 22, 26, 45, 60, 71, 81, 200]
    diag_age_labels = [
        "0-17", "18-21", "22-25",
        "25-44", "45-59", "60-70", "71-80", "81+"
    ]

    survey_df["age_group"] = pd.cut(
        pd.to_numeric(survey_df["age"], errors="coerce"),
        bins=diag_age_bins, labels=diag_age_labels, right=False
    )
    pop_df["age_group"] = pd.cut(
        pd.to_numeric(pop_df["age"], errors="coerce"),
        bins=diag_age_bins, labels=diag_age_labels, right=False
    )

    # Masks: SAME universe for both survey and pop (age >= 18)
    survey_mask = (
        survey_df["age"].notna() &
        (survey_df["age"] >= 18) &
        survey_df["dl_has_or_learning"].notna() &
        survey_df["person_weight"].notna()
    )

    pop_mask = (
        pop_df["age"].notna() &
        (pop_df["age"] >= 18) &
        pop_df[pop_ycol].notna()
    )

    def compare_pct(group_col, canton_id=None, order=None):
        """
        Returns:
        group_col | survey_weighted_pct_has | pop_pct_has
        for the SAME universe (age >= 18).
        """
        # ---- Survey weighted percentage ----
        s_cols = [group_col, "dl_has_or_learning", "person_weight", "canton_id"]
        s = survey_df.loc[survey_mask, s_cols].copy()

        if canton_id is not None:
            s = s[s["canton_id"] == str(canton_id)]

        if len(s) == 0:
            survey_tab = pd.DataFrame(columns=[group_col, "survey_weighted_pct_has"])
        else:
            survey_tab = (
                s.groupby(group_col, dropna=False)
                 .apply(lambda g: 100.0 * _weighted_mean(g["dl_has_or_learning"], g["person_weight"]))
                 .rename("survey_weighted_pct_has")
                 .reset_index()
            )

        # ---- Pop modeled percentage (unweighted persons) ----
        p_cols = [group_col, pop_ycol, "canton_id"]
        p = pop_df.loc[pop_mask, p_cols].copy()

        if canton_id is not None:
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

        # enforce ordering where desired
        if order is not None:
            out[group_col] = pd.Categorical(out[group_col], categories=order, ordered=True)
            out = out.sort_values(group_col)

        out["survey_weighted_pct_has"] = pd.to_numeric(out["survey_weighted_pct_has"], errors="coerce")
        out["pop_pct_has"] = pd.to_numeric(out["pop_pct_has"], errors="coerce")

        out["diff_pop_minus_survey"] = out["pop_pct_has"] - out["survey_weighted_pct_has"]
        out["abs_diff"] = out["diff_pop_minus_survey"].abs()

        return out

    # -------------------------
    # AGE GROUP (overall + canton)
    # -------------------------
    age_comp_all = compare_pct("age_group", canton_id=None, order=diag_age_labels)
    logger.info("\n[AGE GROUP | ALL] age_group | survey_weighted_pct_has | pop_pct_has")
    logger.info(age_comp_all.to_string(index=False))

    age_comp_c = compare_pct("age_group", canton_id=DIAG_CANTON_ID, order=diag_age_labels)
    logger.info(f"\n[AGE GROUP | canton_id={DIAG_CANTON_ID}] age_group | survey_weighted_pct_has | pop_pct_has")
    logger.info(age_comp_c.to_string(index=False))

    # -------------------------
    # SEX (overall + canton)
    # -------------------------
    sex_comp_all = compare_pct("sex", canton_id=None)
    logger.info("\n[SEX | ALL] sex | survey_weighted_pct_has | pop_pct_has")
    logger.info(sex_comp_all.to_string(index=False))

    sex_comp_c = compare_pct("sex", canton_id=DIAG_CANTON_ID)
    logger.info(f"\n[SEX | canton_id={DIAG_CANTON_ID}] sex | survey_weighted_pct_has | pop_pct_has")
    logger.info(sex_comp_c.to_string(index=False))

    # -------------------------
    # INCOME CLASS (overall + canton)
    # -------------------------
    inc_comp_all = compare_pct("income_class", canton_id=None)
    logger.info("\n[INCOME CLASS | ALL] income_class | survey_weighted_pct_has | pop_pct_has")
    logger.info(inc_comp_all.to_string(index=False))

    inc_comp_c = compare_pct("income_class", canton_id=DIAG_CANTON_ID)
    logger.info(f"\n[INCOME CLASS | canton_id={DIAG_CANTON_ID}] income_class | survey_weighted_pct_has | pop_pct_has")
    logger.info(inc_comp_c.to_string(index=False))

    # -------------------------
    # MUNICIPALITY TYPE (overall + canton)
    # -------------------------
    mun_comp_all = compare_pct("municipality_type", canton_id=None)
    logger.info("\n[MUNICIPALITY TYPE | ALL] municipality_type | survey_weighted_pct_has | pop_pct_has")
    logger.info(mun_comp_all.to_string(index=False))

    mun_comp_c = compare_pct("municipality_type", canton_id=DIAG_CANTON_ID)
    logger.info(f"\n[MUNICIPALITY TYPE | canton_id={DIAG_CANTON_ID}] municipality_type | survey_weighted_pct_has | pop_pct_has")
    logger.info(mun_comp_c.to_string(index=False))

    # -------------------------
    # N_CHILDREN_UNDER_18 exact values (overall + canton)
    # (sorting is numeric where possible, with 'Missing' last)
    # -------------------------
    child_comp_all = compare_pct("N_children_under_18_exact", canton_id=None)
    # nicer ordering: numeric ascending then "Missing"
    if "N_children_under_18_exact" in child_comp_all.columns:
        def _sort_key(v):
            try:
                return (0, int(v))
            except Exception:
                return (1, 10**9)
        child_comp_all = child_comp_all.sort_values(
            by="N_children_under_18_exact",
            key=lambda s: s.map(_sort_key)
        )
    logger.info("\n[N_CHILDREN_UNDER_18 (exact) | ALL] N_children_under_18_exact | survey_weighted_pct_has | pop_pct_has")
    logger.info(child_comp_all.to_string(index=False))

    child_comp_c = compare_pct("N_children_under_18_exact", canton_id=DIAG_CANTON_ID)
    if "N_children_under_18_exact" in child_comp_c.columns:
        child_comp_c = child_comp_c.sort_values(
            by="N_children_under_18_exact",
            key=lambda s: s.map(_sort_key)
        )
    logger.info(f"\n[N_CHILDREN_UNDER_18 (exact) | canton_id={DIAG_CANTON_ID}] N_children_under_18_exact | survey_weighted_pct_has | pop_pct_has")
    logger.info(child_comp_c.to_string(index=False))

    # -------------------------
    # total #licenses (pop) + #licenses by canton
    # -------------------------
    pop_mask2 = (
        pop_df["age"].notna() &
        (pop_df["age"] >= 18) &
        pop_df[pop_ycol].notna()
    )

    total_licenses_pop = pop_df.loc[pop_mask2, pop_ycol].sum()
    logger.info(f"\n[POP] Total # driver's licenses (modeled) age>=18 using {pop_ycol}: {int(total_licenses_pop):,}")

    licenses_by_canton = (
        pop_df.loc[pop_mask2]
        .groupby("canton_id")[pop_ycol]
        .sum()
        .rename("n_licenses")
        .sort_values(ascending=False)
        .to_frame()
    )

    canton_n = (
        pop_df.loc[pop_mask2]
        .groupby("canton_id")[pop_ycol]
        .size()
        .rename("n_people")
        .to_frame()
    )

    licenses_by_canton = licenses_by_canton.join(canton_n, how="left")
    licenses_by_canton["pct_has"] = (licenses_by_canton["n_licenses"] / licenses_by_canton["n_people"] * 100).round(2)

    logger.info("\n[POP] # driver's licenses by canton (age>=18):")
    logger.info(licenses_by_canton.to_string())

    logger.info("\n==========================================================================")

    pop_df = pop_df.rename(columns={"DL_has_or_learning_draw": "driving_license"})
    return pop_df
