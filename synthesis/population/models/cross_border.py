"""
Stage: synthesis.population.models.cross_border

Fits a weighted border-crossing classifier on the data.microcensus.persons
subset selected by ``specific_day_scenario`` and applies it to
synthesis.population.models.students. Each predicted crosser is then matched
to a data.cross_border.swiss_residents_od record, the single source of truth
for their destination country, crossing point and trip mode.

MZ has too few crossers per municipality for a usable spatial signal, so
swiss_residents_od (reliable spatially, but without individual covariates) is
used twice: as a canton-level crossing-rate feature, and as the target of the
post-hoc raking in _iterative_spatial_calibration. This stage runs before
synthesis.population.sampled, so the population is the full StatPop one and
needs no downsampling correction as a rate denominator.

Pipeline config keys
--------------------
specific_day_scenario   (inherited, default "workday")
    "workday" | "weekend" | a weekday name; selects the survey subset.
cross_border_model_type (default "logit")
    "logit" — interpretable, odds ratios logged | "catboost" — more accurate.
cross_border_use_spatial_calibration (default True)
    Adds the canton crossing-rate feature and the raking step. When False,
    behavior matches the original individual-covariates-only model.
cross_border_exclude_shapefiles (default None)
    Region(s) covered by the external population, as in
    data.cross_border.generate_od. Survey observations and agents living
    inside are left out of training and prediction, and the swiss_residents_od
    records starting there are dropped from the calibration targets.

Features (available in both MZ and the synthetic population): age, sex,
is_swiss, income_class, municipality_type (ordinal), log(dist_home_to_border
+ 1), employment_status dummies, and canton_crossing_rate when calibration is
enabled. highest_education is in the MZ but not in StatPop, so it is excluded.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from catboost import CatBoostClassifier
from data.osm.clean import read_outside_region
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


def _read_exclude_region(context):
    """
    Region(s) given by cross_border_exclude_shapefiles, or None when the config
    key is unset. Everyone living there is covered by the external population,
    which brings its own border-crossing behaviour, so this stage must neither
    learn from nor predict for them.
    """

    exclude_file = context.config("cross_border_exclude_shapefiles")

    if exclude_file is None:
        return None

    return read_outside_region(exclude_file)


def _is_within_region(df, exclude_region, x_col="home_x", y_col="home_y"):
    """
    Boolean array flagging the rows of df whose x_col/y_col point (EPSG:2056)
    falls inside exclude_region. df must have a unique index.
    """

    points = gpd.GeoDataFrame(
        df[[x_col, y_col]].copy(),
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs="EPSG:2056",
    )

    joined = gpd.sjoin(points, exclude_region[["geometry"]], how="left", predicate="within")

    # A home inside several overlapping polygons of the region comes back once
    # per match, so keep one row per person before realigning on df.
    within = joined["index_right"].notna()
    within = within[~within.index.duplicated(keep="first")]

    return within.reindex(df.index, fill_value=False).to_numpy()


def _age_compatible_od_candidates(od_group: pd.DataFrame, age_value, c) -> pd.DataFrame:
    """Return OD candidates compatible with the person's age constraints."""
    if pd.isna(age_value):
        return pd.DataFrame(columns=od_group.columns)

    age_val = float(age_value)
    if age_val <= c.MZ_AGE_THRESHOLD:
        return pd.DataFrame(columns=od_group.columns)

    if age_val < 18:
        return od_group[~od_group["trip_mode"].astype(str).str.lower().eq("car")]

    return od_group


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

def _compute_od_tables(context, pop_df: pd.DataFrame, od_df: pd.DataFrame):
    """
    From the (possibly region-filtered) swiss_residents_od records and the
    spatial reference layers, derives:
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


def _match_border_crossings(pop_df: pd.DataFrame, od_df: pd.DataFrame, crossers_mask: pd.Series, seed: int, c) -> pd.DataFrame:
    """
    For every person with is_crossing_the_border == 1, samples one specific
    data.cross_border.swiss_residents_od record from the same canton (falling
    back to the national table when the canton has no record of its own, or
    when canton_id itself is missing). This is the single source of truth for
    that person's destination country, border-crossing point, and trip mode --
    synthesis.population.trips reads the result directly instead of matching
    independently, so the label used in matsim/scenario/population.py and the
    spatial location used in synthesis.population.spatial.locations always
    refer to the same underlying record.

    Returns a DataFrame aligned with pop_df.index with columns
    cross_border_person_id, destination_country_raw, interview_geometry_point,
    border_crossing_trip_mode, interview_point_id -- all pd.NA outside of crossers_mask.

    interview_point_id and interview_geometry_point are the id and the coordinates
    of one and the same surveyed crossing, so the border activity ends up exactly
    on the facility it refers to (MATSim's ScenarioValidator rejects anything else).

    The sampled record is constrained to be compatible with the person's age:
    no one at or below the microcensus age threshold can be assigned a crossing,
    and car trips are only assigned to adults.
    """
    od = od_df.sort_values("trip_mode").copy()
    od["origin_canton_id"] = pd.to_numeric(od["origin_canton_id"], errors="coerce")
    od_by_canton = {canton: group for canton, group in od.groupby("origin_canton_id")}

    match_cols = ["cross_border_person_id", "destination_country_raw", "interview_geometry_point", "trip_mode", "interview_point_id"]
    result = pd.DataFrame({
        col: pd.Series(pd.NA, index=pop_df.index, dtype="object") for col in match_cols
    })

    rng = np.random.default_rng(seed)

    crosser_canton = pd.to_numeric(pop_df.loc[crossers_mask, "canton_id"], errors="coerce")
    crosser_ages = pd.to_numeric(pop_df.loc[crossers_mask, "age"], errors="coerce")

    for canton_id, sub in crosser_canton.groupby(crosser_canton, dropna=False):
        candidates = None if pd.isna(canton_id) else od_by_canton.get(canton_id)
        if candidates is None or len(candidates) == 0:
            candidates = od

        sub_indices = sub.index
        sub_ages = crosser_ages.loc[sub_indices]

        for person_idx, person_age in sub_ages.items():
            if pd.notna(person_age) and person_age <= c.MZ_AGE_THRESHOLD:
                continue

            person_candidates = _age_compatible_od_candidates(candidates, person_age, c)
            if person_candidates.empty:
                person_candidates = _age_compatible_od_candidates(od, person_age, c)
            if person_candidates.empty:
                continue

            sampled = person_candidates.sample(
                n=1, replace=True, random_state=int(rng.integers(0, 2**31 - 1))
            ).reset_index(drop=True).iloc[0]

            for col in match_cols:
                result.loc[person_idx, col] = sampled[col]

    return result.rename(columns={"trip_mode": "border_crossing_trip_mode"})


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


def _iterative_spatial_calibration(pop_df: pd.DataFrame, proba: np.ndarray, od_canton: pd.DataFrame, od_muni: pd.DataFrame, age_eligible: pd.Series, valid_income: pd.Series) -> np.ndarray:
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

        proba[~age_eligible] = 0.0
        proba[~valid_income.values] = 0.0

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
    context.config("cross_border_exclude_shapefiles", default=None)
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

    exclude_region = _read_exclude_region(context)

    # -------------------------------------------------------------------
    # 1. SYNTHETIC POPULATION (loaded early — needed as the OD-rate denominator)
    # -------------------------------------------------------------------
    pop_df = context.stage("synthesis.population.models.students").copy().reset_index(drop=True)
    pop_df["_row_order"] = np.arange(len(pop_df))

    # Agents living in the excluded region are set aside before anything else,
    # so they count neither in the canton rate denominators nor in the
    # calibration margins. They are added back as non-crossers in step 12.
    if exclude_region is None:
        df_excluded = pop_df.iloc[:0].copy()
    else:
        is_excluded = _is_within_region(pop_df, exclude_region)
        df_excluded = pop_df[is_excluded].copy()
        pop_df      = pop_df[~is_excluded].copy().reset_index(drop=True)

        logger.info(
            "Excluded region: %d of %d agents live inside it and are kept out of the model.",
            len(df_excluded), len(df_excluded) + len(pop_df),
        )

        assert len(pop_df) > 0, "cross_border_exclude_shapefiles covers the entire population."

    # -------------------------------------------------------------------
    # 2. THEORETICAL SPATIAL DISTRIBUTION (data.cross_border.swiss_residents_od)
    # -------------------------------------------------------------------
    od_df = context.stage("data.cross_border.swiss_residents_od")

    # The crossings starting in the excluded region belong to the agents we
    # just set aside, so they have to go as well: otherwise the calibration
    # would keep their counts in the target while the agents that produced
    # them are no longer there to be scaled up, and the residents remaining in
    # a partly excluded municipality would be pushed up to cover them.
    # origin_municipality is too coarse for this — the region cuts through
    # municipalities — hence the point test on the OD origin coordinates.
    if exclude_region is not None:
        assert "origin_x" in od_df.columns, (
            "data.cross_border.swiss_residents_od does not provide origin_x/origin_y; "
            "re-run that stage so the excluded region can be applied to its records."
        )

        od_df       = od_df.reset_index(drop=True)
        is_excluded = _is_within_region(od_df, exclude_region, "origin_x", "origin_y")

        logger.info(
            "Excluded region: %d of %d swiss_residents_od records start inside it and are dropped from the calibration targets.",
            int(is_excluded.sum()), len(od_df),
        )

        od_df = od_df[~is_excluded].copy()

    od_muni, od_canton, canton_crossing_rate, df_municipalities, df_cantons = (
        _compute_od_tables(context, pop_df, od_df)
    )

    # -------------------------------------------------------------------
    # 3. TRAINING DATA
    # -------------------------------------------------------------------
    survey_df = context.stage("data.microcensus.persons").copy()

    # Exclude persons fully outside Switzerland
    survey_df = survey_df[~survey_df["is_outside_of_switzerland"]]

    # ... and those living in the region covered by the external population,
    # whose crossings are not the ones this model should learn from.
    if exclude_region is not None:
        survey_df   = survey_df.reset_index(drop=True)
        is_excluded = _is_within_region(survey_df, exclude_region)

        logger.info(
            "Excluded region: %d of %d microcensus observations have their home inside it and are dropped from training.",
            int(is_excluded.sum()), len(survey_df),
        )

        survey_df = survey_df[~is_excluded]

    survey_df["income_class"] = pd.to_numeric(survey_df["income_class"], errors="coerce")
    survey_df = survey_df[survey_df["income_class"] >= 0].copy()
    survey_df["income_class"] = survey_df["income_class"].astype(int)

    survey_df = survey_df[pd.to_numeric(survey_df["age"], errors="coerce") > c.MZ_AGE_THRESHOLD].copy()
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
    swiss_border = context.stage("data.spatial.swiss_border").copy().geometry.union_all()
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
    age_eligible = pd.to_numeric(pop_df["age"], errors="coerce") > c.MZ_AGE_THRESHOLD
    age_eligible = age_eligible.fillna(False).to_numpy()
    proba[~age_eligible] = 0.0
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
        proba = _iterative_spatial_calibration(pop_df, proba, od_canton, od_muni, age_eligible, valid_income)
        logger.info(
            "Spatial calibration done. Mean predicted probability: %.3f%% -> %.3f%%",
            mean_before, proba.mean() * 100,
        )

    # -------------------------------------------------------------------
    # 9. STOCHASTIC DRAW
    # -------------------------------------------------------------------
    pop_df["is_crossing_the_border"] = _draw_bernoulli(proba, seed=SEED_CB)

    # -------------------------------------------------------------------
    # 9b. MATCH EACH CROSSER TO A SPECIFIC data.cross_border.swiss_residents_od
    #     RECORD (destination country, crossing point, trip mode)
    # -------------------------------------------------------------------
    # od_df is the region-filtered set from step 2, so a modeled crosser can
    # only be matched to a crossing that starts outside the excluded region.
    crossers_mask = pop_df["is_crossing_the_border"] == 1
    border_match  = _match_border_crossings(pop_df, od_df, crossers_mask, seed=SEED_CB + 1, c=c)
    for col in border_match.columns:
        pop_df[col] = border_match[col].values

    # -------------------------------------------------------------------
    # 10. DIAGNOSTICS
    # -------------------------------------------------------------------
    n_pop   = len(pop_df)
    n_cross = int(pop_df["is_crossing_the_border"].sum())
    logger.info(
        "[CROSS BORDER | %s | %s] Modeled population: %d (+%d excluded) | Crossers: %d (%.2f%%)",
        day_scenario, model_type, n_pop, len(df_excluded), n_cross, n_cross / n_pop * 100,
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

    # Put the excluded agents back as non-crossers, in the order they arrived
    # in. They never went through step 5, so dist_home_to_border stays empty
    # for them, and they get the same empty match columns as any non-crosser.
    if len(df_excluded) > 0:
        df_excluded["is_crossing_the_border"] = 0
        for col in border_match.columns:
            df_excluded[col] = pd.NA

        pop_df = pd.concat([pop_df, df_excluded], ignore_index=True)

    pop_df = pop_df.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)

    return pop_df
