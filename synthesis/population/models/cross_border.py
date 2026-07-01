"""
Stage: synthesis.population.models.cross_border

Learns two border-crossing models from data.microcensus.persons — one for
workdays (Mon–Fri) and one for weekends (Sat–Sun) — then applies the model
that matches the current ``specific_day_scenario`` to the synthetic population
from synthesis.population.models.students.

For each day-type subset two classifiers are fitted:
  1. Weighted logistic regression (statsmodels GLM) — odds ratios, 95 % CI,
       and p-values are written to the pipeline log.
  2. Weighted CatBoost.

MZ has very few border-crossing observations per municipality, so its spatial
signal is weak. data.cross_border.swiss_residents_od (built from on-site
border-crossing interviews) has a much more reliable spatial distribution but
no individual covariates. To combine both sources:
  1. A canton-level theoretical crossing rate (from swiss_residents_od,
     scaled by canton resident population) is added as a model feature, both
     in training and at prediction time.
  2. After prediction, a post-hoc spatial calibration alternates rescaling
     individual probabilities to match canton totals, then municipality
     totals (raking), until both are within tolerance of swiss_residents_od
     or a max iteration count is reached. Municipalities with little OD
     evidence are shrunk back toward their canton's calibration ratio rather
     than over-corrected on a noisy small-sample target. A single canton-then-
     municipality pass isn't enough on its own, since the shrunk municipality
     step can drift the canton totals away from what the canton step just
     achieved — alternating re-fits each margin against the other.
This stage runs before synthesis.population.sampled (where input_downsampling
is applied), so the population here is the full StatPop population — no
downsampling correction is needed when using it as a rate denominator.

Pipeline config keys
--------------------
specific_day_scenario   (inherited, default "workday")
    Controls which day-type model is applied to the population.
    "workday" | individual weekday name → uses the workday model.
    "weekend" | "Saturday" | "Sunday"  → uses the weekend model.

cross_border_model_type (default "catboost")
    Which classifier to use for the stochastic population assignment.
    "catboost" — higher predictive accuracy.
    "logit"    — interpretable, calibrated probabilities.

cross_border_use_spatial_calibration (default True)
    Whether to add the canton crossing-rate feature and apply the post-hoc
    spatial calibration described above. When False, behavior matches the
    original individual-covariates-only model.

Features used (available in both MZ and the synthetic population):
  age, sex, is_swiss, income_class, municipality_type (ordinal),
  log(dist_home_to_border + 1), employment_status one-hot dummies,
  and (if spatial calibration is enabled) canton_crossing_rate.

Note: highest_education is present in the MZ but not in StatPop, so it is
excluded from the feature set.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from catboost import CatBoostClassifier
import logging

logger = logging.getLogger("synpp")

_MUNICIPALITY_ORD = {"rural": 0, "suburban": 1, "urban": 2, "urbancore": 3}

_BASE_FEATURE_LABELS = {
    "age":            "Age",
    "sex":            "Sex (female=1)",
    "is_swiss":       "Swiss nationality",
    "income_class":   "Income class (0–8)",
    "muni_ord":       "Municipality type (0=rural … 3=urban core)",
    "log_dist_border":"log(distance to border + 1 m)",
    "empl_employed":  "Employed (vs inactive)",
    "empl_student":   "Student (vs inactive)",
    "empl_stud_empl": "Student + Employed (vs inactive)",
}
_CANTON_RATE_LABELS = {
    "canton_crossing_rate": "Theoretical canton crossing rate (swiss_residents_od)",
}

# Spatial calibration constants
_CALIBRATION_SHRINKAGE_K = 10        # OD evidence (n_theoretical) needed before trusting the municipality ratio over the canton ratio
_CANTON_SCALE_CLIP        = (0.1, 10.0)  # bounds for the canton-level calibration multiplier
_MUNI_SCALE_CLIP          = (0.2, 5.0)   # bounds for the municipality-level calibration multiplier
_RAKING_ITERATIONS       = 10         # max alternating canton/municipality calibration rounds
_RAKING_TOLERANCE        = 0.05      # stop early once both max relative errors (areas with n_theoretical >= _RAKING_MIN_N) drop below this
_RAKING_MIN_N            = 5         # ignore areas with fewer theoretical observations when checking convergence (too noisy to be a useful signal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_bernoulli(proba, seed=None):
    rng = np.random.default_rng(seed)
    return (rng.random(len(proba)) < proba).astype("int64")


def _weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return np.average(x[m], weights=w[m]) if m.sum() > 0 else np.nan


def _add_survey_features(df):
    """Add derived model columns to MZ persons data (in-place)."""
    df["is_swiss"]        = df["is_swiss"].astype(int)
    df["log_dist_border"] = np.log1p(df["dist_home_to_border"])
    df["muni_ord"]        = df["municipality_type"].map(_MUNICIPALITY_ORD)
    df["empl_employed"]   = (df["employment_status"] == 1).astype(int)
    df["empl_student"]    = (df["employment_status"] == 2).astype(int)
    df["empl_stud_empl"]  = (df["employment_status"] == 3).astype(int)
    return df


# ---------------------------------------------------------------------------
# Theoretical (swiss_residents_od) spatial counts and rates
# ---------------------------------------------------------------------------

def _compute_od_tables(context, pop_df: pd.DataFrame):
    """
    Loads data.cross_border.swiss_residents_od and the spatial reference
    layers, and derives:
      - od_muni:   theoretical border-crosser count per municipality_id
      - od_canton: theoretical border-crosser count per canton_id
      - canton_crossing_rate: od_canton count / canton resident population
        (pop_df is the full StatPop population at this stage of the
        pipeline — no downsampling correction is needed)
      - df_municipalities, df_cantons: geometry layers, for the diff gpkgs

    od_df rows already represent population-scale expanded observations
    (the survey weight was consumed during sampling in swiss_residents_od.py),
    so a plain row count per area is the theoretical population-scale count.
    """
    od_df                = context.stage("data.cross_border.swiss_residents_od")
    df_municipalities, _ = context.stage("data.spatial.municipalities")
    df_cantons           = context.stage("data.spatial.cantons")

    od_muni = (
        od_df.groupby("origin_municipality")
        .size()
        .reset_index(name="n_theoretical")
        .rename(columns={"origin_municipality": "municipality_id"})
    )
    od_muni["municipality_id"] = pd.to_numeric(od_muni["municipality_id"], errors="coerce")

    muni_to_canton = (
        pop_df[["home_municipality_id", "canton_id"]]
        .drop_duplicates("home_municipality_id")
        .rename(columns={"home_municipality_id": "municipality_id"})
    )
    muni_to_canton["municipality_id"] = pd.to_numeric(muni_to_canton["municipality_id"], errors="coerce")

    od_canton = (
        od_muni.merge(muni_to_canton, on="municipality_id", how="left")
        .groupby("canton_id", dropna=True)["n_theoretical"]
        .sum()
        .reset_index()
    )
    od_canton["canton_id"] = pd.to_numeric(od_canton["canton_id"], errors="coerce")

    canton_population = (
        pop_df.groupby("canton_id")
        .size()
        .reset_index(name="population")
    )
    canton_population["canton_id"] = pd.to_numeric(canton_population["canton_id"], errors="coerce")

    canton_crossing_rate = od_canton.merge(canton_population, on="canton_id", how="right")
    canton_crossing_rate["n_theoretical"] = canton_crossing_rate["n_theoretical"].fillna(0.0)
    canton_crossing_rate["canton_crossing_rate"] = np.where(
        canton_crossing_rate["population"] > 0,
        canton_crossing_rate["n_theoretical"] / canton_crossing_rate["population"],
        0.0,
    )
    canton_crossing_rate = canton_crossing_rate[["canton_id", "canton_crossing_rate"]]

    return od_muni, od_canton, canton_crossing_rate, df_municipalities, df_cantons


# ---------------------------------------------------------------------------
# Logistic regression (statsmodels)
# ---------------------------------------------------------------------------

def _fit_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted GLM logit; returns the fitted result object."""
    w_norm = w / w.mean()
    X_c    = sm.add_constant(X)
    return sm.GLM(
        y, X_c,
        family=sm.families.Binomial(),
        var_weights=w_norm,
    ).fit(disp=False)


def _logit_or_table(result, feature_cols, feature_labels) -> pd.DataFrame:
    """Format odds-ratio table from a fitted statsmodels logit result."""
    coefs = result.params[1:]
    ci    = np.array(result.conf_int())[1:]
    pvals = result.pvalues[1:]
    table = pd.DataFrame({
        "feature":  feature_cols,
        "label":    [feature_labels[f] for f in feature_cols],
        "OR":       np.exp(coefs),
        "CI_lower": np.exp(ci[:, 0]),
        "CI_upper": np.exp(ci[:, 1]),
        "p_value":  pvals,
    })
    table["sig"] = table["p_value"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01
        else "*" if p < 0.05 else "ns"
    )
    return table


# ---------------------------------------------------------------------------
# CatBoost classifier
# ---------------------------------------------------------------------------

def _fit_catboost(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )
    model.fit(X, y, sample_weight=w)
    return model


# ---------------------------------------------------------------------------
# Per-subset training pipeline
# ---------------------------------------------------------------------------

def _train_subset(label: str, df: pd.DataFrame, model_type: str, feature_cols, feature_labels):
    """
    Fits the selected model on one day-type subset and returns it.
    model_type: "catboost" | "logit"
    """
    X = df[feature_cols].astype(float).values
    y = df["is_crossing_the_border"].values.astype("int64")
    w = df["person_weight"].values.astype(float)

    n_pos   = int(y.sum())
    n_tot   = len(y)
    w_share = _weighted_mean(y, w) * 100
    logger.info(
        "[%s] n=%d | crossers=%d (%.1f%% unweighted, %.2f%% weighted)",
        label, n_tot, n_pos, n_pos / n_tot * 100, w_share,
    )

    if model_type == "logit":
        logger.info("[%s] Fitting logistic regression …", label)
        result   = _fit_logit(X, y, w)
        or_table = _logit_or_table(result, feature_cols, feature_labels)
        logger.info(
            "[%s] Logit odds ratios:\n%s",
            label,
            or_table[["label", "OR", "CI_lower", "CI_upper", "p_value", "sig"]]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        )
        return result
    else:  # catboost
        logger.info("[%s] Fitting CatBoost …", label)
        model = _fit_catboost(X, y, w)
        logger.info("[%s] CatBoost fitted.", label)
        return model


# ---------------------------------------------------------------------------
# Spatial calibration
# ---------------------------------------------------------------------------

def _calibrate_canton(pop_df: pd.DataFrame, proba: np.ndarray, od_canton: pd.DataFrame) -> np.ndarray:
    """Rescale proba so the sum per canton matches od_canton's n_theoretical."""
    pred_canton = (
        pd.DataFrame({"canton_id": pd.to_numeric(pop_df["canton_id"], errors="coerce"), "proba": proba})
        .groupby("canton_id")["proba"]
        .sum()
        .reset_index(name="pred_canton")
    )

    ratios = od_canton.merge(pred_canton, on="canton_id", how="right")
    ratios["n_theoretical"] = ratios["n_theoretical"].fillna(0.0)
    ratios["s_canton"] = np.where(
        ratios["pred_canton"] > 0,
        ratios["n_theoretical"] / ratios["pred_canton"],
        1.0,
    )
    ratios["s_canton"] = ratios["s_canton"].clip(*_CANTON_SCALE_CLIP)

    s_map = ratios.set_index("canton_id")["s_canton"]
    s = pd.to_numeric(pop_df["canton_id"], errors="coerce").map(s_map).fillna(1.0).values

    return np.clip(proba * s, 0.0, 1.0)


def _calibrate_municipality(pop_df: pd.DataFrame, proba: np.ndarray, od_muni: pd.DataFrame) -> np.ndarray:
    """
    Rescale proba so the sum per municipality matches od_muni's n_theoretical,
    shrinking municipalities with little OD evidence back toward 1 (i.e. no
    further adjustment beyond the canton-level calibration already applied).
    """
    pred_muni = (
        pd.DataFrame({"municipality_id": pd.to_numeric(pop_df["home_municipality_id"], errors="coerce"), "proba": proba})
        .groupby("municipality_id")["proba"]
        .sum()
        .reset_index(name="pred_muni")
    )

    ratios = od_muni.merge(pred_muni, on="municipality_id", how="right")
    ratios["n_theoretical"] = ratios["n_theoretical"].fillna(0.0)
    ratios["r_muni"] = np.where(
        ratios["pred_muni"] > 0,
        ratios["n_theoretical"] / ratios["pred_muni"],
        1.0,
    )
    # Shrink toward 1 (no further adjustment) when little OD evidence exists
    # for that municipality; trust the local ratio more as evidence grows.
    shrink = ratios["n_theoretical"] / (ratios["n_theoretical"] + _CALIBRATION_SHRINKAGE_K)
    ratios["s_muni"] = 1.0 + shrink * (ratios["r_muni"] - 1.0)
    ratios["s_muni"] = ratios["s_muni"].clip(*_MUNI_SCALE_CLIP)

    s_map = ratios.set_index("municipality_id")["s_muni"]
    s = pd.to_numeric(pop_df["home_municipality_id"], errors="coerce").map(s_map).fillna(1.0).values

    return np.clip(proba * s, 0.0, 1.0)


def _max_relative_error(pop_df, proba, od_table, pop_col, id_col, min_n=_RAKING_MIN_N) -> float:
    """
    Max |predicted - theoretical| / theoretical across areas with at least
    min_n theoretical observations (small-n areas are excluded since their
    ratio is dominated by sampling noise rather than calibration quality).
    """
    pred = (
        pd.DataFrame({id_col: pd.to_numeric(pop_df[pop_col], errors="coerce"), "proba": proba})
        .groupby(id_col)["proba"]
        .sum()
        .reset_index(name="pred")
    )
    merged = od_table.merge(pred, on=id_col, how="inner")
    merged = merged[merged["n_theoretical"] >= min_n]
    if len(merged) == 0:
        return 0.0
    rel_err = (merged["pred"] - merged["n_theoretical"]).abs() / merged["n_theoretical"]
    return float(rel_err.max())


def _iterative_spatial_calibration(
    pop_df: pd.DataFrame, proba: np.ndarray, od_canton: pd.DataFrame, od_muni: pd.DataFrame
) -> np.ndarray:
    """
    Alternates canton- and municipality-level calibration (raking) until the
    aggregated counts at both levels are within _RAKING_TOLERANCE of the
    theoretical targets, or _RAKING_ITERATIONS rounds are exhausted.

    A single canton-then-municipality pass isn't enough: shrinkage in
    _calibrate_municipality intentionally leaves low-evidence municipalities
    under-corrected, which means the municipality step alone can drift the
    canton totals away from the target the canton step just achieved.
    Alternating the two re-fits each margin against the other, similar to
    iterative proportional fitting. Convergence can plateau short of the
    target in capacity-constrained areas (where proba is already clipped at
    1.0 for most residents and the canton/municipality target is simply
    larger than achievable) — that is an inherent limit, not a bug.
    """
    for i in range(1, _RAKING_ITERATIONS + 1):
        proba = _calibrate_canton(pop_df, proba, od_canton)
        proba = _calibrate_municipality(pop_df, proba, od_muni)

        canton_err = _max_relative_error(pop_df, proba, od_canton, "canton_id", "canton_id")
        muni_err   = _max_relative_error(pop_df, proba, od_muni, "home_municipality_id", "municipality_id")
        logger.info(
            "  Raking iteration %d/%d: max canton rel. error=%.1f%%, max municipality rel. error=%.1f%% (areas with n_theoretical>=%d)",
            i, _RAKING_ITERATIONS, canton_err * 100, muni_err * 100, _RAKING_MIN_N,
        )

        if max(canton_err, muni_err) < _RAKING_TOLERANCE:
            logger.info("  Spatial calibration converged after %d iteration(s).", i)
            break

    return proba


# ---------------------------------------------------------------------------
# Stage interface
# ---------------------------------------------------------------------------

def configure(context):
    context.config("specific_day_scenario", default="workday")
    context.config("cross_border_model_type", default="logit")
    context.config("cross_border_use_spatial_calibration", default=True)
    context.stage("data.microcensus.persons")
    context.stage("synthesis.population.models.students")
    context.stage("data.spatial.swiss_border")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.cantons")
    context.stage("data.cross_border.swiss_residents_od")
    context.stage("data.constants")


def execute(context):
    c               = context.stage("data.constants")
    day_scenario    = context.config("specific_day_scenario")
    model_type      = context.config("cross_border_model_type")
    use_calibration = context.config("cross_border_use_spatial_calibration")

    assert model_type in ("catboost", "logit"), (
        f"cross_border_model_type must be 'catboost' or 'logit', got '{model_type}'"
    )
    SEED_CB = 2027

    logger.info(
        "Cross-border model: specific_day_scenario='%s', model='%s', spatial_calibration=%s",
        day_scenario, model_type, use_calibration,
    )

    feature_labels = dict(_BASE_FEATURE_LABELS)
    if use_calibration:
        feature_labels.update(_CANTON_RATE_LABELS)
    feature_cols = list(feature_labels.keys())

    # -------------------------------------------------------------------
    # 1. SYNTHETIC POPULATION (loaded early — needed as the OD-rate denominator)
    # -------------------------------------------------------------------
    pop_df = context.stage("synthesis.population.models.students").copy()

    # -------------------------------------------------------------------
    # 2. THEORETICAL SPATIAL DISTRIBUTION (data.cross_border.swiss_residents_od)
    # -------------------------------------------------------------------
    od_muni, od_canton, canton_crossing_rate, df_municipalities, df_cantons = (
        _compute_od_tables(context, pop_df)
    )

    # -------------------------------------------------------------------
    # 3. TRAINING DATA
    # -------------------------------------------------------------------
    survey_df = context.stage("data.microcensus.persons").copy()

    # Exclude persons fully outside Switzerland
    survey_df = survey_df[~survey_df["is_outside_of_switzerland"]]

    survey_df["income_class"] = pd.to_numeric(survey_df["income_class"], errors="coerce")
    survey_df = survey_df[survey_df["income_class"] >= 0].copy()
    survey_df["income_class"] = survey_df["income_class"].astype(int)

    survey_df["is_crossing_the_border"] = survey_df["is_crossing_the_border"].astype(int)
    survey_df = _add_survey_features(survey_df)

    if use_calibration:
        survey_df["canton_id"] = pd.to_numeric(survey_df["canton_id"], errors="coerce")
        survey_df = survey_df.merge(canton_crossing_rate, on="canton_id", how="left")
        survey_df["canton_crossing_rate"] = survey_df["canton_crossing_rate"].fillna(0.0)

    survey_df = survey_df.dropna(
        subset=feature_cols + ["is_crossing_the_border", "person_weight", "workday", "weekend"]
    ).copy()

    # Filter to the relevant day subset — mirrors matched_v1.py logic exactly
    if day_scenario == "workday":
        selected_survey = survey_df[survey_df["workday"]].copy()
    elif day_scenario == "weekend":
        selected_survey = survey_df[survey_df["weekend"]].copy()
    elif day_scenario in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        selected_survey = survey_df[survey_df["day"] == day_scenario].copy()
    else:
        raise ValueError(f"Unimplemented day for specific_day_scenario: '{day_scenario}'")

    logger.info(
        "Training border-crossing model on subset: specific_day_scenario='%s' (n=%d)",
        day_scenario, len(selected_survey),
    )

    # -------------------------------------------------------------------
    # 4. FIT MODEL (selected day subset only)
    # -------------------------------------------------------------------
    selected_model = _train_subset(day_scenario, selected_survey, model_type, feature_cols, feature_labels)

    logger.info("Applying '%s' %s model to synthetic population.", day_scenario, model_type)

    # -------------------------------------------------------------------
    # 5. POPULATION: distance from home to Swiss border
    # -------------------------------------------------------------------
    swiss_border = context.stage("data.spatial.swiss_border").copy().unary_union
    swiss_border = swiss_border.simplify(tolerance=100)
    border_line  = swiss_border.boundary

    # Deduplicate by household to avoid repeating the distance computation
    hh_home = pop_df[["household_id", "home_x", "home_y"]].drop_duplicates("household_id")
    hh_gdf  = gpd.GeoDataFrame(
        hh_home.reset_index(drop=True),
        geometry=gpd.points_from_xy(hh_home["home_x"], hh_home["home_y"]),
        crs="epsg:2056",
    )
    hh_gdf["dist_home_to_border"] = hh_gdf.geometry.apply(
        lambda p: p.distance(border_line)
    )
    pop_df = pop_df.merge(
        hh_gdf[["household_id", "dist_home_to_border"]], on="household_id", how="left"
    )
    logger.info("Computed home-to-border distances for %d households.", len(hh_home))

    # -------------------------------------------------------------------
    # 6. POPULATION FEATURE ENGINEERING
    # -------------------------------------------------------------------
    pop_df["is_swiss"]        = (pop_df["nationality"] == 0).astype(int)
    pop_df["log_dist_border"] = np.log1p(pop_df["dist_home_to_border"].fillna(0.0))
    pop_df["muni_ord"]        = pop_df["municipality_type"].map(_MUNICIPALITY_ORD).fillna(1.0)

    pop_df["income_class"] = pd.to_numeric(pop_df["income_class"], errors="coerce")
    valid_income = pop_df["income_class"].notna() & (pop_df["income_class"] >= 0)
    pop_df.loc[~valid_income, "income_class"] = 0  # placeholder; probability zeroed below

    # Construct employment_status from employed + is_student (same logic as drlicense.py)
    emp = pop_df["employed"].values
    stu = pop_df["is_student"].values
    es  = np.full(len(pop_df), c.EMPLOYEMENT_STATUS.INACTIVE, dtype=int)
    es[emp == c.EMPLOYED]                          = c.EMPLOYEMENT_STATUS.EMPLOYED
    es[(emp == c.INACTIVE)   & (stu == 1)]         = c.EMPLOYEMENT_STATUS.STUDENT
    es[(emp == c.UNEMPLOYED) & (stu == 1)]         = c.EMPLOYEMENT_STATUS.STUDENT
    es[(emp == c.EMPLOYED)   & (stu == 1)]         = c.EMPLOYEMENT_STATUS.EMPLOYED_STUDENT

    pop_df["empl_employed"]  = (es == c.EMPLOYEMENT_STATUS.EMPLOYED).astype(int)
    pop_df["empl_student"]   = (es == c.EMPLOYEMENT_STATUS.STUDENT).astype(int)
    pop_df["empl_stud_empl"] = (es == c.EMPLOYEMENT_STATUS.EMPLOYED_STUDENT).astype(int)

    if use_calibration:
        pop_df["canton_id"] = pd.to_numeric(pop_df["canton_id"], errors="coerce")
        pop_df = pop_df.merge(canton_crossing_rate, on="canton_id", how="left")
        pop_df["canton_crossing_rate"] = pop_df["canton_crossing_rate"].fillna(0.0)

    X_pop = pop_df[feature_cols].astype(float).values

    # -------------------------------------------------------------------
    # 7. PREDICT
    # -------------------------------------------------------------------
    if model_type == "catboost":
        proba = selected_model.predict_proba(X_pop)[:, 1]
    else:  # logit
        proba = selected_model.predict(sm.add_constant(X_pop))

    proba = np.asarray(proba, dtype=float)
    proba[~valid_income.values] = 0.0

    # -------------------------------------------------------------------
    # 8. SPATIAL CALIBRATION (alternating canton / municipality raking)
    # -------------------------------------------------------------------
    if use_calibration:
        mean_before = proba.mean() * 100
        logger.info(
            "Spatial calibration: raking up to %d iteration(s), shrinkage K=%d, tolerance=%.1f%%",
            _RAKING_ITERATIONS, _CALIBRATION_SHRINKAGE_K, _RAKING_TOLERANCE * 100,
        )
        proba = _iterative_spatial_calibration(pop_df, proba, od_canton, od_muni)
        proba[~valid_income.values] = 0.0
        logger.info(
            "Spatial calibration done. Mean predicted probability: %.3f%% -> %.3f%%",
            mean_before, proba.mean() * 100,
        )

    # -------------------------------------------------------------------
    # 9. STOCHASTIC DRAW
    # -------------------------------------------------------------------
    pop_df["is_crossing_the_border"] = _draw_bernoulli(proba, seed=SEED_CB)

    # -------------------------------------------------------------------
    # 10. DIAGNOSTICS
    # -------------------------------------------------------------------
    n_pop   = len(pop_df)
    n_cross = int(pop_df["is_crossing_the_border"].sum())
    logger.info(
        "[CROSS BORDER | %s | %s] Population: %d | Crossers: %d (%.2f%%)",
        day_scenario, model_type, n_pop, n_cross, n_cross / n_pop * 100,
    )

    logger.info("\n[CROSS BORDER] By municipality_type  (pop vs. survey %s):", day_scenario)
    for muni in sorted(_MUNICIPALITY_ORD, key=_MUNICIPALITY_ORD.get):
        pop_mask  = pop_df["municipality_type"].astype(str) == muni
        surv_mask = selected_survey["municipality_type"].astype(str) == muni
        if pop_mask.sum() == 0:
            continue
        pop_rate  = pop_df.loc[pop_mask, "is_crossing_the_border"].mean() * 100
        surv_rate = _weighted_mean(
            selected_survey.loc[surv_mask, "is_crossing_the_border"].values,
            selected_survey.loc[surv_mask, "person_weight"].values,
        ) * 100
        logger.info("  %-12s  pop: %.2f%%  survey: %.2f%%", muni, pop_rate, surv_rate)

    logger.info("\n[CROSS BORDER] By nationality  (pop vs. survey %s):", day_scenario)
    for swiss_val, label in [(1, "Swiss"), (0, "Non-Swiss")]:
        pop_mask  = pop_df["is_swiss"] == swiss_val
        surv_mask = selected_survey["is_swiss"] == swiss_val
        if pop_mask.sum() == 0:
            continue
        pop_rate  = pop_df.loc[pop_mask, "is_crossing_the_border"].mean() * 100
        surv_rate = _weighted_mean(
            selected_survey.loc[surv_mask, "is_crossing_the_border"].values,
            selected_survey.loc[surv_mask, "person_weight"].values,
        ) * 100
        logger.info("  %-10s  pop: %.2f%%  survey: %.2f%%", label, pop_rate, surv_rate)

    # -------------------------------------------------------------------
    # 11. SPATIAL COMPARISON vs. data.cross_border.swiss_residents_od
    #     (reuses od_muni / od_canton / df_municipalities / df_cantons from step 2)
    # -------------------------------------------------------------------
    # --- Municipality level ---
    syn_muni = (
        pop_df[pop_df["is_crossing_the_border"] == 1]
        .groupby("home_municipality_id")
        .size()
        .reset_index(name="n_synthesized")
        .rename(columns={"home_municipality_id": "municipality_id"})
    )
    syn_muni["municipality_id"] = pd.to_numeric(syn_muni["municipality_id"], errors="coerce")

    gdf_muni = df_municipalities[["municipality_id", "municipality_name", "geometry"]].copy()
    gdf_muni["municipality_id"] = pd.to_numeric(gdf_muni["municipality_id"], errors="coerce")
    gdf_muni = gdf_muni.merge(od_muni,  on="municipality_id", how="left")
    gdf_muni = gdf_muni.merge(syn_muni, on="municipality_id", how="left")
    gdf_muni["n_theoretical"] = gdf_muni["n_theoretical"].fillna(0).astype(int)
    gdf_muni["n_synthesized"] = gdf_muni["n_synthesized"].fillna(0).astype(int)
    gdf_muni["abs_diff"] = gdf_muni["n_synthesized"] - gdf_muni["n_theoretical"]
    gdf_muni["rel_diff"] = np.where(
        gdf_muni["n_theoretical"] > 0,
        gdf_muni["abs_diff"] / gdf_muni["n_theoretical"],
        np.nan,
    )
    gdf_muni.to_file(f"{context.path()}/diff_by_municipality.gpkg", driver="GPKG")
    logger.info("Saved diff_by_municipality.gpkg")

    # --- Canton level ---
    syn_canton = (
        pop_df[pop_df["is_crossing_the_border"] == 1]
        .groupby("canton_id")
        .size()
        .reset_index(name="n_synthesized")
    )
    syn_canton["canton_id"] = pd.to_numeric(syn_canton["canton_id"], errors="coerce")

    gdf_canton = df_cantons[["canton_id", "canton_name", "geometry"]].copy()
    gdf_canton["canton_id"] = pd.to_numeric(gdf_canton["canton_id"], errors="coerce")
    gdf_canton = gdf_canton.merge(od_canton,  on="canton_id", how="left")
    gdf_canton = gdf_canton.merge(syn_canton, on="canton_id", how="left")
    gdf_canton["n_theoretical"] = gdf_canton["n_theoretical"].fillna(0).astype(int)
    gdf_canton["n_synthesized"] = gdf_canton["n_synthesized"].fillna(0).astype(int)
    gdf_canton["abs_diff"] = gdf_canton["n_synthesized"] - gdf_canton["n_theoretical"]
    gdf_canton["rel_diff"] = np.where(
        gdf_canton["n_theoretical"] > 0,
        gdf_canton["abs_diff"] / gdf_canton["n_theoretical"],
        np.nan,
    )
    gdf_canton.to_file(f"{context.path()}/diff_by_canton.gpkg", driver="GPKG")
    logger.info("Saved diff_by_canton.gpkg")

    # -------------------------------------------------------------------
    # 12. CLEANUP
    # -------------------------------------------------------------------
    drop_cols = [
        "is_swiss",
        "empl_employed", "empl_student", "empl_stud_empl",
        "muni_ord", "log_dist_border",
    ]
    if use_calibration:
        drop_cols.append("canton_crossing_rate")

    pop_df = pop_df.drop(columns=drop_cols, errors="ignore")

    return pop_df
