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

def freq_table(df, col, top=None):
    vc = df[col].value_counts(dropna=False)
    pct = df[col].value_counts(dropna=False, normalize=True).mul(100)
    out = pd.DataFrame({"count": vc, "pct": pct.round(2)})
    if top:
        out = out.head(top)
    return out

def _weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if m.sum() == 0:
        return np.nan
    return np.average(x[m], weights=w[m])

def configure(context):
    context.stage("data.microcensus.21.persons")
    context.stage("data.microcensus.21.household_persons")
    context.stage("synthesis.population.models.drlicense")

def execute(context):
    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------
    CAR_MODEL = "catboost"     # "rf" or "gbm" or "catboost"
    SEED_CAR  = 2026
    DIAG_CANTON_ID = "25"

    # Survey car count column (change if needed)
    SURVEY_CARCOUNT_COL = "number_of_cars_class"

    # -------------------------------------------------------------------
    # 0. LOAD DATA
    # -------------------------------------------------------------------
    #TODO: add household type to the variables
    # we need to create the types as reported in mz in hhtyp variable
    survey_df = context.stage("data.microcensus.21.persons").copy()
    survey_hh_df = context.stage("data.microcensus.21.household_persons")[0]

    # --- ensure types (do this BEFORE using age) ---
    survey_hh_df["age"] = pd.to_numeric(survey_hh_df["age"], errors="coerce")
    survey_hh_df["driving_license"] = pd.to_numeric(survey_hh_df["driving_license"], errors="coerce")

    # --- hh_oldest_age (1 row per household) ---
    hh_oldest = (
        survey_hh_df.groupby("household_id", as_index=False)["age"]
        .max()
        .rename(columns={"age": "hh_oldest_age"})
    )

    # --- compute household aggregates ---
    hh_lic = survey_hh_df.loc[survey_hh_df["age"].notna()].copy()
    hh_lic["is_adult"] = (hh_lic["age"] >= 18).astype(int)
    hh_lic["is_adult_30_64"] = hh_lic["age"].between(25, 60, inclusive="both").astype(int)


    hh_lic["dl_adult"] = hh_lic["driving_license"].fillna(0).where(hh_lic["age"] >= 18, 0)

    hh_agg = (
        hh_lic.groupby("household_id", as_index=False)
            .agg(
                N_adults_survey=("is_adult", "sum"),
                N_adults_30_64=("is_adult_30_64", "sum"),
                N_drivers_license_adults=("dl_adult", "sum"),
            )
    )

    hh_agg["N_drivers_license_per_adult"] = (
        hh_agg["N_drivers_license_adults"] / hh_agg["N_adults_survey"].replace(0, np.nan)
    ).fillna(0.0)

    # --- merge hh_agg + hh_oldest, then onto survey_df by household_id ---
    hh_out = hh_agg.merge(hh_oldest, on="household_id", how="left")

    survey_df = survey_df.merge(
        hh_out[[
            "household_id",
            "N_adults_survey",
            "N_adults_30_64", 
            "N_drivers_license_adults",
            "N_drivers_license_per_adult",
            "hh_oldest_age",
        ]],
        left_on="person_id", right_on="household_id",
        how="left"
    )

    pop_df    = context.stage("synthesis.population.models.drlicense").copy()

    # Make sure essentials exist
    for df in (survey_df, pop_df):
        df["age"] = pd.to_numeric(df.get("age"), errors="coerce")

    # Adults in pop (>=18)
    adult_mask_pop = pop_df["age"].notna() & (pop_df["age"] >= 18)
    pop_df["N_adults"] = (
        adult_mask_pop.astype(int)
        .groupby(pop_df["household_id"])
        .transform("sum")
    )

    # Children in pop (<18)
    child_mask_pop = pop_df["age"].notna() & (pop_df["age"] < 18)
    pop_df["N_children_under_18"] = (
        child_mask_pop.astype(int)
        .groupby(pop_df["household_id"])
        .transform("sum")
    )

    # adults age 30..64 (inclusive) in pop
    mask_30_64 = pop_df["age"].notna() & pop_df["age"].between(25, 60, inclusive="both")
    pop_df["N_adults_30_64"] = (
        mask_30_64.astype(int)
        .groupby(pop_df["household_id"])
        .transform("sum")
    )


    dl_pop_col = "driving_license"
  
    # -------------------------------------------------------------------
    # 1. PREP SURVEY: build household-level table with target 0/1/2/3+
    # -------------------------------------------------------------------
    # keep only cols that exist; we'll error later if target missing
    missing_target = SURVEY_CARCOUNT_COL not in survey_df.columns
    if missing_target:
        raise KeyError(f"Survey target column '{SURVEY_CARCOUNT_COL}' not found. "
                       f"Set SURVEY_CARCOUNT_COL to the correct car-count variable.")

    # Ensure numeric car count
    survey_df[SURVEY_CARCOUNT_COL] = pd.to_numeric(survey_df[SURVEY_CARCOUNT_COL], errors="coerce")

    # Optional: if your survey has DL at person level and you want N_drivers_license_per_adult from survey too
    # (this is important if you use that feature in training, so survey and pop align)

    if "driving_license" in survey_df.columns:
        # driving_license assumed boolean; convert to 0/1 on adults
        lic = survey_df["driving_license"].astype(float)
        lic = lic.where(survey_df["age"] >= 18, 0.0)
        survey_df["_dl_adult"] = lic.fillna(0.0)
    else:
        # If not available, set missing; feature will be dropped if you enforce notna
        survey_df["_dl_adult"] = np.nan

    # Household-level aggregation helpers
    def first_nonnull(x):
        x = x.dropna()
        return x.iloc[0] if len(x) else np.nan
    survey_df["presence_of_children_under_18"] = (survey_df["N_children_under_18"] > 0).astype(int)
    # Build HH table
    hh_s = survey_df.copy()
    # rename + weights
    hh_s = hh_s.rename(columns={
        SURVEY_CARCOUNT_COL: "car_count_raw",
        "household_weight": "hh_weight",   # since 1 row per household
    })
    # Clean weight
    hh_s["hh_weight"] = pd.to_numeric(hh_s["hh_weight"], errors="coerce")
    hh_s = hh_s.dropna(subset=["hh_weight"])
    hh_s = hh_s[hh_s["hh_weight"] > 0].copy()

    # Target: 0/1/2/3+ classes
    hh_s["car_count_raw"] = pd.to_numeric(hh_s["car_count_raw"], errors="coerce")
    # keep plausible non-negative
    hh_s = hh_s[hh_s["car_count_raw"].notna() & (hh_s["car_count_raw"] >= 0)].copy()

    hh_s["HH_CAR_OWN_class"] = hh_s["car_count_raw"].astype(int).clip(upper=3)  # 3 means 3+

    # -------------------------------------------------------------------
    # 2. PREP POP: household-level features, then predict HH car ownership
    # -------------------------------------------------------------------
    # Household size in pop: if you already have household_size column, use max; else compute
    if "household_size" not in pop_df.columns:
        pop_df["household_size"] = pop_df.groupby("household_id")["household_id"].transform("size")

    # Children already built above in pop_df["N_children_under_18"]
    pop_df["presence_of_children_under_18"] = (pop_df["N_children_under_18"] > 0).astype(int)

    adult = pop_df["age"].ge(18)
    # if driving_license is already 0/1; otherwise do: (pop_df["driving_license"] == 1).astype(int)
    dl = pd.to_numeric(pop_df["driving_license"], errors="coerce").fillna(0).astype("int8")

    # group-wise totals aligned to pop_df rows
    n_adults = adult.groupby(pop_df["household_id"]).transform("sum").astype("float32")
    n_dl = (dl.where(adult, 0)).groupby(pop_df["household_id"]).transform("sum").astype("float32")

    pop_df["N_drivers_license_per_adult"] = n_dl / n_adults
    pop_df["N_drivers_license_adults"] = n_dl
    pop_df.loc[n_adults == 0, "N_drivers_license_per_adult"] = np.nan  
    # Build HH table in pop
    def mode_or_first(x):
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        m = x.mode()
        return m.iloc[0] if len(m) else x.iloc[0]

    hh_p = (
    pop_df
    .drop_duplicates("household_id", keep="first")
    [["household_id",
      "household_size",
      "N_adults",
      "N_adults_30_64", 
      "presence_of_children_under_18",
      "N_drivers_license_per_adult",
      'N_drivers_license_adults',
      "canton_id",
      "municipality_type",
      "ovgk",           
      "income_class",     
      "hh_oldest_age"
     ]].copy()
)

    # --- hh_s: weighted frequency (and %), by presence_of_children_under_18 ---
    wcol = "hh_weight"  # change if your weight column name differs
    gcol = "presence_of_children_under_18"

    tmp = hh_s[[gcol, wcol]].copy()
    tmp[gcol] = tmp[gcol].astype("string").fillna("<NA>")
    tmp[wcol] = pd.to_numeric(tmp[wcol], errors="coerce")

    w_counts = tmp.groupby(gcol)[wcol].sum().sort_values(ascending=False)
    w_freq = w_counts.to_frame("weighted_count")
    w_freq["weighted_pct"] = (w_freq["weighted_count"] / w_freq["weighted_count"].sum() * 100).round(2)

    # --- hh_p: unweighted frequency (and %), by presence_of_children_under_18 ---
    p_counts = hh_p[gcol].astype("string").fillna("<NA>").value_counts(dropna=False)
    p_freq = p_counts.to_frame("count")
    p_freq["pct"] = (p_freq["count"] / p_freq["count"].sum() * 100).round(2)

    # -------------------------------------------------------------------
    # 3. FEATURE ENGINEERING / CLEANING
    # -------------------------------------------------------------------
    # Use income_class if available (high impact); if you truly don't want it, remove from feature list below
    use_income = ("income_class" in hh_s.columns) and ("income_class" in hh_p.columns)

    # categorical + numeric feature lists
    cat_cols = [ "ovgk","canton_id",  "municipality_type", "presence_of_children_under_18"]#, "ovgk" "municipality_type", "presence_of_children_under_18", "income_class"] #presence_of_children_under_18 reduces the  number of those not owning a car

    num_cols = ["N_adults", "N_adults_30_64", "N_drivers_license_adults", "hh_oldest_age", "income_class"]
    
    for df in (hh_s, hh_p):
        # numeric clean
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # categorical clean
        for c in cat_cols:
            df[c] = df[c].astype(str).fillna("Missing")

        # clip ratio
        df["N_drivers_license_per_adult"] = df["N_drivers_license_per_adult"].clip(0.0, 1.0)
   
    col = "N_drivers_license_per_adult"
    # -------------------------------------------------------------------
    # 4. DESIGN MATRICES
    # -------------------------------------------------------------------
    Xs = pd.get_dummies(hh_s[cat_cols], drop_first=False)
    Xp = pd.get_dummies(hh_p[cat_cols], drop_first=False)

    # add numeric
    Xs[num_cols] = hh_s[num_cols].astype(float).values
    Xp[num_cols] = hh_p[num_cols].astype(float).values

    # align
    Xp = Xp.reindex(columns=Xs.columns, fill_value=0.0)

    y = hh_s["HH_CAR_OWN_class"].astype("int64").values
    w = hh_s["hh_weight"].astype(float).values

    # -------------------------------------------------------------------
    # 5. FIT MODEL (multiclass: 0,1,2,3(=3+))
    # -------------------------------------------------------------------
    def build_car_model(model_type: str):
        mt = str(model_type).lower().strip()
        if mt == "gbm":
            return HistGradientBoostingClassifier(
                loss="log_loss",
                max_depth=8,
                learning_rate=0.05,
                max_iter=500,
                min_samples_leaf=80,
                random_state=42
            )
        elif mt == "rf":
            return RandomForestClassifier(
                n_estimators=800,
                max_depth=None,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        elif mt in ("catboost", "cat"):
            return CatBoostClassifier(
                loss_function="MultiClass",
                iterations=1000,
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

    car_model = build_car_model(CAR_MODEL)

    Xs_np = Xs.to_numpy(dtype=float, copy=False)
    Xp_np = Xp.to_numpy(dtype=float, copy=False)
    
    logger.info("Started to fit household car-ownership model using: %s", CAR_MODEL)
    car_model.fit(Xs_np, y, sample_weight=w)
    logger.info("Fitted household car-ownership model using: %s", CAR_MODEL)

    # -------------------------------------------------------------------
    # 6. PREDICT + STOCHASTIC DRAW
    # -------------------------------------------------------------------
    proba_hh = car_model.predict_proba(Xp_np)
    classes = getattr(car_model, "classes_", np.array([0, 1, 2, 3])).astype("int64")

    hh_draw = draw_multinomial_from_proba(proba_hh, classes, seed=SEED_CAR).astype("int64")
    hh_hat  = classes[proba_hh.argmax(axis=1)].astype("int64")

    hh_p["HH_CAR_OWN_hat"]  = hh_hat
    hh_p["HH_CAR_OWN_draw"] = hh_draw

    # Merge household result back to persons (pop_df)
    pop_df = pop_df.merge(
        hh_p[["household_id", "HH_CAR_OWN_hat", "HH_CAR_OWN_draw"]],
        on="household_id",
        how="left"
    )
    
    # -------------------------------------------------------------------
    # 7. DIAGNOSTICS (household level): survey weighted vs pop modeled
    #    Compare distributions of classes 0/1/2/3+ overall and by group (and by canton choice).
    # -------------------------------------------------------------------
    logger.info("\n================== DIAGNOSTICS (Survey vs Modeled Pop, HH car ownership) ==================")

    USE_DRAW = True
    pop_ycol = "HH_CAR_OWN_draw" if USE_DRAW else "HH_CAR_OWN_hat"
    classes_sorted = [0, 1, 2, 3]

    def compare_multiclass(group_col, canton_id=None, order=None):
        classes_sorted = [0, 1, 2, 3]

        # --- SURVEY weighted shares ---
        s_cols = [group_col, "HH_CAR_OWN_class", "hh_weight"]
        # only add canton_id for filtering if it's not already the grouping col
        if group_col != "canton_id":
            s_cols.append("canton_id")

        s = hh_s[s_cols].copy()

        if canton_id is not None:
            # ensure we can filter by canton even if group_col == canton_id
            if "canton_id" not in s.columns:
                s["canton_id"] = hh_s["canton_id"].astype(str).fillna("Missing")
            s = s[s["canton_id"].astype(str) == str(canton_id)]

        s[group_col] = s[group_col].astype(str).fillna("Missing")

        s_mass = (
            s.groupby([group_col, "HH_CAR_OWN_class"])["hh_weight"]
            .sum()
            .rename("w")
            .reset_index()
        )
        s_tot = s.groupby(group_col)["hh_weight"].sum().rename("w_tot").reset_index()
        s_sh = s_mass.merge(s_tot, on=group_col, how="left")
        s_sh["share"] = s_sh["w"] / s_sh["w_tot"]

        s_piv = s_sh.pivot(index=group_col, columns="HH_CAR_OWN_class", values="share").fillna(0.0)
        s_piv = s_piv.reindex(columns=classes_sorted, fill_value=0.0)
        s_piv = (s_piv * 100.0).add_prefix("survey_pct_").reset_index()

        # --- POP shares (unweighted households) ---
        p_cols = [group_col, pop_ycol]
        if group_col != "canton_id":
            p_cols.append("canton_id")

        p = hh_p[p_cols].copy()

        if canton_id is not None:
            if "canton_id" not in p.columns:
                p["canton_id"] = hh_p["canton_id"].astype(str).fillna("Missing")
            p = p[p["canton_id"].astype(str) == str(canton_id)]

        p[group_col] = p[group_col].astype(str).fillna("Missing")

        p_cnt = p.groupby([group_col, pop_ycol]).size().rename("n").reset_index()
        p_tot = p.groupby(group_col).size().rename("n_tot").reset_index()
        p_sh = p_cnt.merge(p_tot, on=group_col, how="left")
        p_sh["share"] = p_sh["n"] / p_sh["n_tot"]

        p_piv = p_sh.pivot(index=group_col, columns=pop_ycol, values="share").fillna(0.0)
        p_piv = p_piv.reindex(columns=classes_sorted, fill_value=0.0)
        p_piv = (p_piv * 100.0).add_prefix("pop_pct_").reset_index()

        out = s_piv.merge(p_piv, on=group_col, how="outer").fillna(0.0)

        # diffs per class
        for c in classes_sorted:
            out[f"diff_{c}"] = out.get(f"pop_pct_{c}", 0.0) - out.get(f"survey_pct_{c}", 0.0)

        if order is not None:
            out[group_col] = pd.Categorical(out[group_col], categories=order, ordered=True)
            out = out.sort_values(group_col)

        return out


    # overall distribution
    overall = compare_multiclass(group_col="canton_id", canton_id=None)

    def overall_dist(df, ycol, wcol=None):
        if wcol is None:
            vc = df[ycol].value_counts(normalize=True).reindex(classes_sorted, fill_value=0.0) * 100
            return vc
        out = {}
        w = df[wcol].values
        yv = df[ycol].values
        tot = np.sum(w)
        for c in classes_sorted:
            out[c] = (np.sum(w[yv == c]) / tot * 100.0) if tot > 0 else 0.0
        return pd.Series(out)

    logger.info("\n[OVERALL | Survey weighted % by class 0/1/2/3+]")
    logger.info(overall_dist(hh_s, "HH_CAR_OWN_class", wcol="hh_weight").to_string())

    logger.info("\n[OVERALL | Pop modeled % by class 0/1/2/3+]")
    logger.info(overall_dist(hh_p, pop_ycol, wcol=None).to_string())

    # Diagnostics by requested variables (overall + selected canton)
    diag_groups = [
        ("canton_id", None),
        ("municipality_type", None),
        ("ovgk", None),
        ("presence_of_children_under_18", ["0", "1"]),
    ]
    if use_income:
        diag_groups.append(("income_class", None))

    # Add household size bins (helpful)
    def hhsize_bin(x):
        try:
            x = float(x)
        except Exception:
            return "Missing"
        if x <= 1: return "1"
        if x == 2: return "2"
        if x == 3: return "3"
        if x == 4: return "4"
        return "5+"

    #hh_s["hhsize_bin"] = hh_s["household_size"].apply(hhsize_bin)
    #hh_p["hhsize_bin"] = hh_p["household_size"].apply(hhsize_bin)
    #diag_groups.append(("hhsize_bin", ["1", "2", "3", "4", "5+"]))

    for g, order in diag_groups:
        logger.info(f"\n[{g.upper()} | ALL] survey vs pop (% by class)")
        logger.info(compare_multiclass(g, canton_id=None, order=order).to_string(index=False))

        logger.info(f"\n[{g.upper()} | canton_id={DIAG_CANTON_ID}] survey vs pop (% by class)")
        logger.info(compare_multiclass(g, canton_id=DIAG_CANTON_ID, order=order).to_string(index=False))

    logger.info("\n==========================================================================================")
    pop_df = pop_df.rename(columns={"HH_CAR_OWN_draw": "number_of_cars_class"}) # 0, 1, 2, 3+ (coded as 3)
    return pop_df
